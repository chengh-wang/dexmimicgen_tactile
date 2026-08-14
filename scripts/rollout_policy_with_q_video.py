#!/usr/bin/env python3
"""Roll out a tactile policy and overlay WM/Q-head scores in one MP4.

The policy runs at the trained 20 Hz control rate. The Q head is scored on the
same 200 Hz stream used by the WM training: tactile + robot joint position are
sampled ten times per 20 Hz action, and the 20 Hz action is repeated ten times.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
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
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

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
from train_threepiece_high200_wm import ThreePieceWM  # noqa: E402
from train_threepiece_q_head import QHead  # noqa: E402


CAMERA_NAMES = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]


def robot_joint_pos(env) -> np.ndarray:
    pos = []
    for robot in getattr(env, "robots", []):
        if hasattr(robot, "_joint_positions"):
            pos.append(np.asarray(robot._joint_positions, dtype=np.float32).reshape(-1))
    if not pos:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(pos).astype(np.float32)


def high_sample(env) -> dict[str, np.ndarray]:
    return {
        "robot_joint_pos": robot_joint_pos(env),
        "tactile": tactile_env.read_tactile_image(env).astype(np.float32),
    }


def step_with_high200_minimal(env, action: np.ndarray, low_step_index: int):
    if env.done:
        raise ValueError("executing action in terminated episode")

    env.timestep += 1
    policy_step = True
    start_time = float(env.cur_time)
    model_dt = float(env.model_timestep)
    control_dt = float(env.control_timestep)
    n_substeps = int(round(control_dt / model_dt))
    sample_dt = 0.005
    next_sample = 0
    samples = []

    for i in range(n_substeps):
        if env.lite_physics:
            env.sim.step1()
        else:
            env.sim.forward()
        env._pre_action(action, policy_step)
        if env.lite_physics:
            env.sim.step2()
        else:
            env.sim.step()
        env._update_observables()
        policy_step = False

        actual_time = start_time + (i + 1) * model_dt
        while next_sample < 10:
            target_time = start_time + (next_sample + 1) * sample_dt
            if actual_time + 1e-12 < target_time:
                break
            samples.append(high_sample(env))
            next_sample += 1

    while next_sample < 10:
        samples.append(high_sample(env))
        next_sample += 1

    env.cur_time += env.control_timestep
    reward, done, info = env._post_action(action)
    observations = env.viewer._get_observations() if env.viewer_get_obs else env._get_observations()
    return observations, reward, done, info, samples


def pad_slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    n = arr.shape[0]
    idx = np.arange(start, start + length)
    idx = np.clip(idx, 0, max(n - 1, 0))
    return arr[idx]


@torch.no_grad()
def score_q_values(
    wm: ThreePieceWM,
    q_head: QHead,
    high_tactile: np.ndarray,
    high_joint: np.ndarray,
    low_actions: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    n_low = low_actions.shape[0]
    if n_low == 0:
        return np.zeros((0,), dtype=np.float32)

    tactile_flat = high_tactile.reshape(-1, 4, 32, 32).astype(np.float32)
    tactile_flat = np.clip(tactile_flat, 0.0, 1.0)
    joint_flat = high_joint.reshape(-1, high_joint.shape[-1]).astype(np.float32)
    action_flat = np.repeat(low_actions.astype(np.float32), 10, axis=0)

    if joint_flat.shape[-1] != 14:
        raise RuntimeError(f"expected 14-D robot_joint_pos for WM, got {joint_flat.shape}")

    q_values = []
    window = history + chunk
    for low_start in range(0, n_low, batch_size):
        lows = list(range(low_start, min(low_start + batch_size, n_low)))
        tactile_batch = []
        joint_batch = []
        action_batch = []
        for low_idx in lows:
            high_idx = low_idx * 10 + 9
            start = high_idx - history + 1
            tactile_batch.append(pad_slice(tactile_flat, start, history))
            joint_batch.append(pad_slice(joint_flat, start, history))
            action_batch.append(pad_slice(action_flat, start, window))

        tactile = np.stack(tactile_batch, axis=0)
        joint = np.stack(joint_batch, axis=0)
        action = np.stack(action_batch, axis=0)
        joint = (joint - normalizers["joint_mean"]) / normalizers["joint_std"]
        action = (action - normalizers["action_mean"]) / normalizers["action_std"]

        sample = {
            "tactile": torch.from_numpy(tactile).to(device, non_blocking=True),
            "joint": torch.from_numpy(joint).to(device, non_blocking=True),
            "action": torch.from_numpy(action).to(device, non_blocking=True),
        }
        z_hist = wm.encode({"tactile": sample["tactile"], "joint": sample["joint"]})
        z_window = z_hist
        for k in range(chunk):
            a_win = wm.action(sample["action"][:, k : k + history])
            pred_seq = wm.predictor(z_window, a_win)
            next_z = pred_seq[:, -1]
            z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
        action_chunk = sample["action"][:, history : history + chunk]
        logits = q_head(z_hist, action_chunk, z_window[:, -1])
        q_values.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).tolist())
    return np.asarray(q_values, dtype=np.float32)


def resize_nearest(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def put_label(img: np.ndarray, text: str, xy=(12, 28), scale=0.75) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def tactile_panel(tactile_block: np.ndarray, mode: str = "max") -> np.ndarray:
    block = tactile_block.astype(np.float32)
    if mode == "last":
        tactile4 = block[-1]
    elif mode == "mean":
        tactile4 = block.mean(axis=0)
    else:
        tactile4 = block.max(axis=0)

    tiles = []
    for c in range(4):
        gray = (np.clip(tactile4[c], 0.0, 1.0) * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        heat = resize_nearest(heat, (192, 192))
        put_label(heat, f"taxel {c}", xy=(8, 24), scale=0.55)
        tiles.append(heat)
    return np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)


def q_plot_panel(q_values: np.ndarray, current: int, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 56, 18, 18, 34
    x0, x1 = margin_l, width - margin_r
    y0, y1 = margin_t, height - margin_b
    cv2.rectangle(panel, (x0, y0), (x1, y1), (40, 40, 40), 1)
    for frac in (0.25, 0.5, 0.75):
        y = int(y1 - frac * (y1 - y0))
        cv2.line(panel, (x0, y), (x1, y), (210, 210, 210), 1)
    put_label(panel, "Q value", xy=(10, 28), scale=0.55)
    cv2.putText(panel, "0", (18, y1 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(panel, "1", (18, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

    if q_values.size > 1:
        xs = np.linspace(x0, x1, q_values.size)
        ys = y1 - np.clip(q_values, 0.0, 1.0) * (y1 - y0)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        upto = min(current + 1, len(pts))
        if upto >= 2:
            cv2.polylines(panel, [pts[:upto]], False, (20, 90, 220), 2, cv2.LINE_AA)
        cv2.line(panel, (pts[current, 0], y0), (pts[current, 0], y1), (220, 80, 40), 1, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts[current]), 5, (220, 80, 40), -1, cv2.LINE_AA)
    put_label(panel, f"step={current:03d}  q={float(q_values[current]):.3f}", xy=(width - 270, 28), scale=0.55)
    return panel


def make_video_frame(
    rgb_row: np.ndarray,
    tactile_block: np.ndarray,
    q_values: np.ndarray,
    step: int,
    episode_idx: int,
    success: bool,
    reward_sum: float,
    mode: str,
) -> np.ndarray:
    rgb = rgb_row.copy()
    put_label(
        rgb,
        f"rollout {episode_idx}  step={step:03d}  success={int(success)}  reward={reward_sum:.2f}",
        xy=(12, 32),
        scale=0.75,
    )
    tac = tactile_panel(tactile_block, mode=mode)
    info = np.full((tac.shape[0], rgb.shape[1] - tac.shape[1], 3), 25, dtype=np.uint8)
    put_label(info, "20Hz policy rollout", xy=(18, 44), scale=0.8)
    put_label(info, "Q scored on high200 tactile+joint", xy=(18, 88), scale=0.65)
    put_label(info, "history=4 high samples, chunk=20 high actions", xy=(18, 126), scale=0.65)
    mid = np.concatenate([tac, info], axis=1)
    plot = q_plot_panel(q_values, step, rgb.shape[1], 220)
    return np.concatenate([rgb, mid, plot], axis=0)


def collect_one_rollout(env, config, dataset, agent, nfe: int, episode_idx: int, args):
    device = config.optimization.device
    rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    np.random.seed(args.seed + episode_idx)
    torch.manual_seed(args.seed + episode_idx)

    raw_obs = env.reset()
    obs_hist = deque(maxlen=config.task.obs_steps)
    first_obs = obs_from_raw(raw_obs, config, dataset)
    for _ in range(config.task.obs_steps):
        obs_hist.append(first_obs)

    rgb_frames = []
    low_actions = []
    rewards = []
    successes = []
    high_tactile = []
    high_joint = []
    total_reward = 0.0
    success = False
    low_steps = 0

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
        act20 = dataset.normalizer["action"].unnormalize(act_normed.detach().cpu().numpy())[0]
        start = config.task.obs_steps - 1
        end = start + config.task.act_steps
        act14_seq = action20_to_action14(act20[start:end], rotation_transformer)

        for act14 in act14_seq:
            act14 = act14.astype(np.float32)
            raw_obs, reward, done, _, samples = step_with_high200_minimal(env, act14, low_steps)
            total_reward += float(reward)
            success = success or bool(env._check_success()) or float(reward) > 0.0

            rgb_frames.append(render_cameras(env, CAMERA_NAMES, args.video_height, args.video_width))
            low_actions.append(act14)
            rewards.append(np.float32(reward))
            successes.append(np.float32(success))
            high_tactile.append(np.stack([s["tactile"] for s in samples], axis=0))
            high_joint.append(np.stack([s["robot_joint_pos"] for s in samples], axis=0))

            obs_hist.append(obs_from_raw(raw_obs, config, dataset))
            low_steps += 1
            if done or success or low_steps >= config.task.max_episode_steps:
                break
        if success or low_steps >= config.task.max_episode_steps:
            break

    return {
        "episode": episode_idx,
        "success": bool(success),
        "reward_sum": float(total_reward),
        "steps": int(low_steps),
        "rgb": np.asarray(rgb_frames, dtype=np.uint8),
        "actions": np.asarray(low_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "high_tactile": np.asarray(high_tactile, dtype=np.float32),
        "high_joint": np.asarray(high_joint, dtype=np.float32),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default=str(
            DEXMG_ROOT
            / "datasets/generated_tactile_actionrollout_virtual_s8fs25_shards/20260726_122743_s8fs25/two_arm_three_piece_assembly"
        ),
    )
    parser.add_argument(
        "--env-dataset-path",
        default=str(DEXMG_ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"),
    )
    parser.add_argument("--task-config", default="dexmg_three_piece_image_tactile_cnn_virtual_s8fs25")
    parser.add_argument(
        "--model-path",
        default=str(
            POLICY_ROOT
            / "logs_dexmg_three_piece_tactile_cnn_virtual_s8fs25_flow_100k_20260726/models/model_step50000.pt"
        ),
    )
    parser.add_argument(
        "--q-ckpt",
        default=str(DEXMG_ROOT / "outputs/threepiece_q_head_vae8k_wm007020_chunk20_stride10_10k_20260727/q_best.pt"),
    )
    parser.add_argument(
        "--wm-ckpt",
        default=str(DEXMG_ROOT / "outputs/threepiece_high200_wm_vae8k_h4_attn_jointpos_bs512_520epoch_evalfix_20260727/wm_step007020.pt"),
    )
    parser.add_argument(
        "--vae-ckpt",
        default=str(
            POLICY_ROOT
            / "runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727/vae_step008000.pt"
        ),
    )
    parser.add_argument(
        "--normalizers",
        default=str(
            DEXMG_ROOT
            / "outputs/threepiece_high200_wm_vae8k_h4_attn_jointpos_bs512_520epoch_evalfix_20260727/normalizers.npz"
        ),
    )
    parser.add_argument("--output", default=str(DEXMG_ROOT / "outputs/qhead_policy_rollouts/flow50k_tactile_q_overlay_5rollouts.mp4"))
    parser.add_argument("--summary", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-height", type=int, default=384)
    parser.add_argument("--video-width", type=int, default=384)
    parser.add_argument("--tactile-mode", choices=["max", "mean", "last"], default="max")
    parser.add_argument("--seed", type=int, default=2700)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary is None:
        args.summary = str(Path(args.output).with_suffix(".json"))

    class ConfigArgs:
        pass

    cfg_args = ConfigArgs()
    cfg_args.task_config = args.task_config
    cfg_args.model_path = args.model_path
    cfg_args.dataset_path = args.dataset_path
    cfg_args.max_episode_steps = args.max_episode_steps
    cfg_args.episodes = args.episodes
    config = make_config(cfg_args)
    if args.device is not None:
        config.optimization.device = args.device
    config.task.max_episode_steps = args.max_episode_steps
    device = torch.device(config.optimization.device if torch.cuda.is_available() else "cpu")

    print(f"[setup] policy_model={args.model_path}")
    print(f"[setup] q_ckpt={args.q_ckpt}")
    print(f"[setup] wm_ckpt={args.wm_ckpt}")
    print(f"[setup] q scoring: history={args.history} chunk={args.chunk} at 200Hz", flush=True)

    print("[setup] making tactile env", flush=True)
    env = make_env(args.env_dataset_path, enable_tactile=True)
    if abs(float(env.control_timestep) - 0.05) > 1e-9:
        raise RuntimeError(f"Expected 20 Hz control_dt=0.05, got {env.control_timestep}")
    if abs(float(env.model_timestep) - 0.002) > 1e-9:
        raise RuntimeError(f"Expected 500 Hz model_dt=0.002, got {env.model_timestep}")

    print("[setup] building policy dataset normalizers", flush=True)
    dataset = make_dataset(config.task)
    print("[setup] loading policy", flush=True)
    agent = TrainingAgent(config)
    agent.load(args.model_path, load_optimizer=False)
    agent.eval()

    print("[setup] loading WM + Q", flush=True)
    wm = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    wm_ckpt = torch.load(args.wm_ckpt, map_location=device)
    wm.load_state_dict(wm_ckpt["model"], strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    q_head = QHead(history=args.history, chunk=args.chunk).to(device)
    q_ckpt = torch.load(args.q_ckpt, map_location=device)
    q_head.load_state_dict(q_ckpt["q_head"], strict=True)
    q_head.eval()
    for p in q_head.parameters():
        p.requires_grad = False

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_meta = []
    with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for ep in range(args.episodes):
            print(f"[rollout] episode={ep} nfe={args.nfe}", flush=True)
            data = collect_one_rollout(env, config, dataset, agent, args.nfe, ep, args)
            q_values = score_q_values(
                wm,
                q_head,
                data["high_tactile"],
                data["high_joint"],
                data["actions"],
                normalizers,
                args.history,
                args.chunk,
                device,
            )
            meta = {
                "episode": ep,
                "success": data["success"],
                "reward_sum": data["reward_sum"],
                "steps": data["steps"],
                "q_min": float(q_values.min()) if q_values.size else float("nan"),
                "q_max": float(q_values.max()) if q_values.size else float("nan"),
                "q_first": float(q_values[0]) if q_values.size else float("nan"),
                "q_last": float(q_values[-1]) if q_values.size else float("nan"),
            }
            all_meta.append(meta)
            print(f"[rollout] {json.dumps(meta)}", flush=True)
            for step in range(data["steps"]):
                frame = make_video_frame(
                    data["rgb"][step],
                    data["high_tactile"][step],
                    q_values,
                    step,
                    ep,
                    data["success"],
                    data["reward_sum"],
                    args.tactile_mode,
                )
                writer.append_data(frame)

    summary = {
        "output": str(out),
        "policy_model": args.model_path,
        "q_ckpt": args.q_ckpt,
        "wm_ckpt": args.wm_ckpt,
        "nfe": args.nfe,
        "episodes": all_meta,
        "q_timebase": "scored per 20Hz video frame using high200 history/action windows",
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"[done] video={out}", flush=True)
    print(f"[done] summary={args.summary}", flush=True)


if __name__ == "__main__":
    main()
