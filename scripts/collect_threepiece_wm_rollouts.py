"""Collect ThreePieceAssembly world-model rollouts with 20 Hz obs and 200 Hz tactile.

The policy is still evaluated at the trained 20 Hz control rate. During each
20 Hz action, this script mirrors robosuite's internal 500 Hz stepping loop and
records ten high-rate samples at 5 ms target intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch


DEFAULT_DEXMG_ROOT = Path(__file__).resolve().parents[1]
DEXMG_ROOT = Path(os.environ.get("DEXMG_ROOT", str(DEFAULT_DEXMG_ROOT))).resolve()
POLICY_ROOT = Path(
    os.environ.get(
        "POLICY_ROOT", "/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising"
    )
).resolve()
ROBOSUITE_ROOT = DEXMG_ROOT / "robosuite"

for path in (POLICY_ROOT, ROBOSUITE_ROOT, DEXMG_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TACTILE_RENDERER", "virtual")
os.environ.setdefault("TACTILE_VIRTUAL_SIGMA", "8.0")
os.environ.setdefault("TACTILE_VIRTUAL_FORCE_SCALE", "25.0")
os.environ.setdefault("TACTILE_VIRTUAL_MAX", "1.0")
os.environ.setdefault("TACTILE_VIRTUAL_CANONICAL", "1")
os.environ.setdefault("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015")

import dexmimicgen  # noqa: E402,F401
import tactile_recollect.env as tactile_env  # noqa: E402
from examples.eval_dexmg_rollout import (  # noqa: E402
    LOWDIM_KEYS,
    RGB_KEYS,
    TrainingAgent,
    action20_to_action14,
    make_config,
    make_dataset,
    make_env,
    normalize_obs,
    obs_from_raw,
    render_cameras,
    stack_last,
)
from mip.dataset_utils import RotationTransformer  # noqa: E402


CAMERA_NAMES = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default=str(DEXMG_ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"),
    )
    parser.add_argument(
        "--env-dataset-path",
        default=None,
        help="HDF5 used for robosuite env metadata. Defaults to --dataset-path.",
    )
    parser.add_argument(
        "--task-config", default="dexmg_three_piece_image_tactile_cnn_virtual_s8fs25"
    )
    parser.add_argument("--success-model-path", required=True)
    parser.add_argument("--failure-model-path", required=True)
    parser.add_argument("--success-nfe", type=int, default=4)
    parser.add_argument("--failure-nfe", type=int, default=2)
    parser.add_argument("--target-success", type=int, default=50)
    parser.add_argument("--target-failure", type=int, default=50)
    parser.add_argument("--max-attempts-per-split", type=int, default=250)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-every", type=int, default=25)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--image-compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--tactile-compression", default="gzip", choices=["lzf", "gzip", "none"])
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument("--tactile-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument(
        "--high-action-mode",
        choices=["hold", "interp"],
        default="interp",
        help="How to execute and save 200 Hz actions inside each 20 Hz interval.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def compression_kwargs(kind: str, gzip_level: int):
    if kind == "none":
        return {}
    if kind == "gzip":
        return {"compression": "gzip", "compression_opts": gzip_level, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def robot_joint_arrays(env) -> tuple[np.ndarray, np.ndarray]:
    pos = []
    vel = []
    for robot in getattr(env, "robots", []):
        if hasattr(robot, "_joint_positions"):
            pos.append(np.asarray(robot._joint_positions, dtype=np.float32).reshape(-1))
        if hasattr(robot, "_joint_velocities"):
            vel.append(np.asarray(robot._joint_velocities, dtype=np.float32).reshape(-1))
    if not pos:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(pos).astype(np.float32), np.concatenate(vel).astype(np.float32)


def robot_gripper_arrays(env) -> tuple[np.ndarray, np.ndarray]:
    pos = []
    vel = []
    for robot in getattr(env, "robots", []):
        for arm in getattr(robot, "arms", []):
            if not getattr(robot, "has_gripper", {}).get(arm, False):
                continue
            pos_idx = robot._ref_gripper_joint_pos_indexes[arm]
            vel_idx = robot._ref_gripper_joint_vel_indexes[arm]
            pos.append(np.asarray(robot.sim.data.qpos[pos_idx], dtype=np.float32).reshape(-1))
            vel.append(np.asarray(robot.sim.data.qvel[vel_idx], dtype=np.float32).reshape(-1))
    if not pos:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(pos).astype(np.float32), np.concatenate(vel).astype(np.float32)


def high_sample(env, action: np.ndarray, target_time_s: float, actual_time_s: float, sim_step_index: int):
    robot_pos, robot_vel = robot_joint_arrays(env)
    gripper_pos, gripper_vel = robot_gripper_arrays(env)
    return {
        "target_time_ms": np.float32(target_time_s * 1000.0),
        "time_ms": np.float32(actual_time_s * 1000.0),
        "sim_step_index": np.int64(sim_step_index),
        "action": np.asarray(action, dtype=np.float32).copy(),
        "sim_qpos": env.sim.data.qpos.copy().astype(np.float32),
        "sim_qvel": env.sim.data.qvel.copy().astype(np.float32),
        "robot_joint_pos": robot_pos,
        "robot_joint_vel": robot_vel,
        "robot_gripper_qpos": gripper_pos,
        "robot_gripper_qvel": gripper_vel,
        "tactile": tactile_env.read_tactile_image(env).astype(np.float32),
    }


def step_with_high200(
    env,
    action: np.ndarray,
    low_step_index: int,
    next_action: np.ndarray | None = None,
    high_action_mode: str = "hold",
):
    if env.done:
        raise ValueError("executing action in terminated episode")

    env.timestep += 1
    start_time = float(env.cur_time)
    model_dt = float(env.model_timestep)
    control_dt = float(env.control_timestep)
    n_substeps = int(round(control_dt / model_dt))
    sample_dt = 0.005
    next_sample = 0
    samples = []
    action = np.asarray(action, dtype=np.float32)
    if high_action_mode == "interp":
        end_action = np.asarray(next_action if next_action is not None else action, dtype=np.float32)
        high_actions = np.linspace(action, end_action, 10, endpoint=True, dtype=np.float32)
    elif high_action_mode == "hold":
        high_actions = np.repeat(action[None], 10, axis=0).astype(np.float32)
    else:
        raise ValueError(f"unknown high_action_mode: {high_action_mode}")

    last_bin = -1
    last_action = high_actions[0]

    for i in range(n_substeps):
        elapsed = i * model_dt
        high_bin = min(int((elapsed + 1e-12) / sample_dt), 9)
        last_action = high_actions[high_bin]
        if env.lite_physics:
            env.sim.step1()
        else:
            env.sim.forward()
        env._pre_action(last_action, policy_step=(high_bin != last_bin))
        if env.lite_physics:
            env.sim.step2()
        else:
            env.sim.step()
        env._update_observables()
        last_bin = high_bin

        actual_time = start_time + (i + 1) * model_dt
        while next_sample < 10:
            target_time = start_time + (next_sample + 1) * sample_dt
            if actual_time + 1e-12 < target_time:
                break
            global_sim_step = low_step_index * n_substeps + i + 1
            samples.append(high_sample(env, high_actions[next_sample], target_time, actual_time, global_sim_step))
            next_sample += 1

    while next_sample < 10:
        target_time = start_time + (next_sample + 1) * sample_dt
        samples.append(
            high_sample(
                env,
                high_actions[next_sample],
                target_time,
                start_time + control_dt,
                (low_step_index + 1) * n_substeps,
            )
        )
        next_sample += 1

    env.cur_time += env.control_timestep
    reward, done, info = env._post_action(last_action)
    observations = env.viewer._get_observations() if env.viewer_get_obs else env._get_observations()
    return observations, reward, done, info, samples


def make_agent(config, model_path: str):
    agent = TrainingAgent(config)
    agent.load(model_path, load_optimizer=False)
    agent.eval()
    return agent


def maybe_video_writer(video_dir: str | None, split: str, accepted_idx: int, video_every: int):
    if video_dir is None or accepted_idx % video_every != 0:
        return None, None
    path = Path(video_dir) / f"{split}_{accepted_idx:03d}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=20), str(path)


def collect_episode(
    env,
    config,
    dataset,
    agent,
    nfe: int,
    seed: int,
    high_action_mode: str,
    video_writer=None,
):
    device = config.optimization.device
    rotation_transformer = RotationTransformer(
        from_rep="axis_angle", to_rep="rotation_6d"
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    raw_obs = env.reset()
    obs_hist = deque(maxlen=config.task.obs_steps)
    first_obs = obs_from_raw(raw_obs, config, dataset)
    for _ in range(config.task.obs_steps):
        obs_hist.append(first_obs)

    low = {
        "time_ms": [],
        "control_step_index": [],
        "action": [],
        "reward": [],
        "success": [],
    }
    for key in RGB_KEYS + LOWDIM_KEYS:
        low[key] = []

    high = {
        "target_time_ms": [],
        "time_ms": [],
        "sim_step_index": [],
        "action": [],
        "sim_qpos": [],
        "sim_qvel": [],
        "robot_joint_pos": [],
        "robot_joint_vel": [],
        "robot_gripper_qpos": [],
        "robot_gripper_qvel": [],
        "tactile": [],
    }

    total_reward = 0.0
    success = False
    low_steps = 0
    action_min = np.inf
    action_max = -np.inf
    action_absmax = 0.0

    if video_writer is not None:
        video_writer.append_data(render_cameras(env, CAMERA_NAMES, 384, 384))

    while low_steps < config.task.max_episode_steps:
        obs_seq = stack_last(obs_hist, config.task.obs_steps)
        obs = normalize_obs(obs_seq, dataset, device)
        act_0 = torch.randn(
            (1, config.task.horizon, config.task.act_dim),
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            act_normed = agent.sample(act_0=act_0, obs=obs, num_steps=nfe, use_ema=True)
        act20 = dataset.normalizer["action"].unnormalize(
            act_normed.detach().cpu().numpy()
        )[0]
        start = config.task.obs_steps - 1
        end = start + config.task.act_steps
        act14_seq = action20_to_action14(act20[start:end], rotation_transformer)

        for act_i, act14 in enumerate(act14_seq):
            act14 = act14.astype(np.float32)
            next_act14 = (
                act14_seq[act_i + 1].astype(np.float32)
                if act_i + 1 < len(act14_seq)
                else act14
            )
            raw_obs, reward, done, _, samples = step_with_high200(
                env,
                act14,
                low_steps,
                next_action=next_act14,
                high_action_mode=high_action_mode,
            )
            high_action_block = np.stack([sample["action"] for sample in samples], axis=0)
            action_min = min(action_min, float(np.min(high_action_block)))
            action_max = max(action_max, float(np.max(high_action_block)))
            action_absmax = max(action_absmax, float(np.max(np.abs(high_action_block))))

            low["time_ms"].append(np.float32(env.cur_time * 1000.0))
            low["control_step_index"].append(np.int64(low_steps))
            low["action"].append(act14)
            low["reward"].append(np.float32(reward))
            total_reward += float(reward)
            success = success or bool(env._check_success()) or float(reward) > 0.0
            low["success"].append(np.float32(success))
            for key in RGB_KEYS + LOWDIM_KEYS:
                arr = np.asarray(raw_obs[key])
                if key in RGB_KEYS:
                    # Match DexMimicGen / robomimic HDF5 image orientation.
                    # Online robosuite raw_obs images are vertically flipped
                    # relative to the stored training observations.
                    arr = arr[::-1]
                if key in LOWDIM_KEYS:
                    arr = arr.astype(np.float32)
                low[key].append(arr)
            for key in high:
                high[key].append(np.stack([sample[key] for sample in samples], axis=0))

            obs_hist.append(obs_from_raw(raw_obs, config, dataset))
            low_steps += 1
            if video_writer is not None and low_steps % 2 == 0:
                video_writer.append_data(render_cameras(env, CAMERA_NAMES, 384, 384))
            if done or success or low_steps >= config.task.max_episode_steps:
                break

    for key, values in low.items():
        low[key] = np.asarray(values)
    for key, values in high.items():
        high[key] = np.asarray(values)

    meta = {
        "success": bool(success),
        "reward_sum": float(total_reward),
        "steps": int(low_steps),
        "action_min": float(action_min),
        "action_max": float(action_max),
        "action_absmax": float(action_absmax),
    }
    return meta, low, high


def write_episode(
    root,
    name: str,
    meta: dict,
    low: dict,
    high: dict,
    policy_label: str,
    policy_ckpt: str,
    nfe: int,
    image_kwargs: dict,
    tactile_kwargs: dict,
    tactile_dtype: str,
):
    group = root.create_group(name)
    group.attrs.update(meta)
    group.attrs["policy_label"] = policy_label
    group.attrs["policy_ckpt"] = policy_ckpt
    group.attrs["nfe"] = int(nfe)
    group.attrs["control_hz"] = 20
    group.attrs["tactile_hz"] = 200
    group.attrs["sim_hz"] = 500
    group.attrs["time_unit"] = "ms"

    low_group = group.create_group("low20")
    for key, arr in low.items():
        kwargs = image_kwargs if key in RGB_KEYS else {}
        low_group.create_dataset(key, data=arr, **kwargs)

    high_group = group.create_group("high200")
    for key, arr in high.items():
        if key == "tactile":
            arr = arr.astype(np.float16 if tactile_dtype == "float16" else np.float32)
            high_group.create_dataset(key, data=arr, **tactile_kwargs)
        else:
            high_group.create_dataset(key, data=arr)


def run_split(
    h5,
    env,
    config,
    dataset,
    agent,
    policy_label: str,
    model_path: str,
    nfe: int,
    desired_success: bool,
    target_count: int,
    max_attempts: int,
    seed_base: int,
    args,
):
    accepted = 0
    attempts = 0
    image_kwargs = compression_kwargs(args.image_compression, args.gzip_level)
    tactile_kwargs = compression_kwargs(args.tactile_compression, args.gzip_level)
    root = h5["data"]

    while accepted < target_count and attempts < max_attempts:
        writer, video_path = maybe_video_writer(args.video_dir, policy_label, accepted, args.video_every)
        try:
            meta, low, high = collect_episode(
                env,
                config,
                dataset,
                agent,
                nfe=nfe,
                seed=seed_base + attempts,
                high_action_mode=args.high_action_mode,
                video_writer=writer,
            )
        finally:
            if writer is not None:
                writer.close()

        got_success = bool(meta["success"])
        keep = got_success == desired_success
        print(
            f"[{policy_label}] attempt={attempts:03d} keep={int(keep)} "
            f"success={int(got_success)} steps={meta['steps']} reward={meta['reward_sum']:.3f} "
            f"accepted={accepted}/{target_count}"
            + (f" video={video_path}" if video_path else ""),
            flush=True,
        )
        if keep:
            name = f"{policy_label}_{accepted:03d}"
            write_episode(
                root,
                name,
                meta,
                low,
                high,
                policy_label=policy_label,
                policy_ckpt=model_path,
                nfe=nfe,
                image_kwargs=image_kwargs,
                tactile_kwargs=tactile_kwargs,
                tactile_dtype=args.tactile_dtype,
            )
            accepted += 1
            h5.flush()
        attempts += 1

    if accepted < target_count:
        raise RuntimeError(
            f"{policy_label} only collected {accepted}/{target_count} after {attempts} attempts"
        )


def main():
    args = parse_args()
    if args.env_dataset_path is None:
        args.env_dataset_path = args.dataset_path

    class ConfigArgs:
        pass

    cfg_args = ConfigArgs()
    cfg_args.task_config = args.task_config
    cfg_args.model_path = args.success_model_path
    cfg_args.dataset_path = args.dataset_path
    cfg_args.max_episode_steps = args.max_episode_steps
    cfg_args.episodes = 1
    config = make_config(cfg_args)
    if args.device is not None:
        config.optimization.device = args.device
    config.task.max_episode_steps = args.max_episode_steps

    print(f"[setup] dexmg_root={DEXMG_ROOT}")
    print(f"[setup] policy_root={POLICY_ROOT}")
    print(f"[setup] dataset={args.dataset_path}")
    print(f"[setup] env_dataset={args.env_dataset_path}")
    print(f"[setup] output={args.output_path}")
    print(
        "[setup] tactile renderer "
        f"sigma={os.environ.get('TACTILE_VIRTUAL_SIGMA')} "
        f"force_scale={os.environ.get('TACTILE_VIRTUAL_FORCE_SCALE')} "
        f"max={os.environ.get('TACTILE_VIRTUAL_MAX')}"
    )

    print("[setup] making tactile env")
    env = make_env(args.env_dataset_path, enable_tactile=True)
    print(
        f"[setup] env={env.__class__.__name__} action_dim={env.action_dim} "
        f"control_dt={env.control_timestep} model_dt={env.model_timestep}"
    )
    if abs(float(env.control_timestep) - 0.05) > 1e-9:
        raise RuntimeError(f"Expected 20 Hz control_dt=0.05, got {env.control_timestep}")
    if abs(float(env.model_timestep) - 0.002) > 1e-9:
        raise RuntimeError(f"Expected 500 Hz model_dt=0.002, got {env.model_timestep}")

    print("[setup] building dataset normalizers")
    dataset = make_dataset(config.task)
    print(f"[setup] dataset={dataset}")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(out_path)

    with h5py.File(out_path, "w") as h5:
        h5.attrs["task"] = "two_arm_three_piece_assembly"
        h5.attrs["format"] = "wm_rollout_low20_high200"
        h5.attrs["time_unit"] = "ms"
        h5.attrs["control_hz"] = 20
        h5.attrs["tactile_hz"] = 200
        h5.attrs["action_hz"] = 200
        h5.attrs["high_action_mode"] = args.high_action_mode
        h5.attrs["action_semantics"] = (
            "high200/action stores the controller goal used for each 200 Hz tactile/joint sample; "
            "interp uses linspace(current low20 action, next low20 action, 10)"
        )
        h5.attrs["sim_hz"] = 500
        h5.attrs["tactile_renderer"] = os.environ.get("TACTILE_RENDERER", "virtual")
        h5.attrs["tactile_sigma"] = float(os.environ.get("TACTILE_VIRTUAL_SIGMA", "8.0"))
        h5.attrs["tactile_force_scale"] = float(
            os.environ.get("TACTILE_VIRTUAL_FORCE_SCALE", "25.0")
        )
        h5.attrs["tactile_max"] = float(os.environ.get("TACTILE_VIRTUAL_MAX", "1.0"))
        h5.create_group("data")

        if args.target_success > 0:
            print("[setup] loading success policy")
            success_agent = make_agent(config, args.success_model_path)
            run_split(
                h5,
                env,
                config,
                dataset,
                success_agent,
                policy_label="success",
                model_path=args.success_model_path,
                nfe=args.success_nfe,
                desired_success=True,
                target_count=args.target_success,
                max_attempts=args.max_attempts_per_split,
                seed_base=args.seed,
                args=args,
            )
            del success_agent
            torch.cuda.empty_cache()

        if args.target_failure > 0:
            print("[setup] loading failure policy")
            failure_agent = make_agent(config, args.failure_model_path)
            run_split(
                h5,
                env,
                config,
                dataset,
                failure_agent,
                policy_label="failure",
                model_path=args.failure_model_path,
                nfe=args.failure_nfe,
                desired_success=False,
                target_count=args.target_failure,
                max_attempts=args.max_attempts_per_split,
                seed_base=args.seed + 100000,
                args=args,
            )

    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
