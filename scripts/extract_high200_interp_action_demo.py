#!/usr/bin/env python3
"""Replay one DexMimicGen demo with interpolated 200 Hz actions.

The source demo observations remain at their original 20 Hz rate under
``low20/obs``. Tactile, robot joint position, and action are generated at 200 Hz
under ``high200`` by resetting to the demo initial state and updating the
controller goal ten times inside each 20 Hz interval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import cv2


DEXMG_ROOT = Path(__file__).resolve().parents[1]
ROBOSUITE_ROOT = DEXMG_ROOT / "robosuite"
for p in (ROBOSUITE_ROOT, DEXMG_ROOT):
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TACTILE_RENDERER", "virtual")
os.environ.setdefault("TACTILE_VIRTUAL_SIGMA", "8.0")
os.environ.setdefault("TACTILE_VIRTUAL_FORCE_SCALE", "25.0")
os.environ.setdefault("TACTILE_VIRTUAL_MAX", "1.0")
os.environ.setdefault("TACTILE_VIRTUAL_CANONICAL", "1")
os.environ.setdefault("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015")

import dexmimicgen  # noqa: E402,F401
from tactile_recollect.env import make_env_from_env_args, read_tactile_image  # noqa: E402


COMP = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def reset_to(env, state: dict) -> None:
    if "model" in state:
        ep_meta = json.loads(state["ep_meta"]) if state.get("ep_meta") else {}
        if hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        elif hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    elif hasattr(env, "update_sites"):
        env.update_sites()


def robot_joint_pos(env) -> np.ndarray:
    vals = []
    for robot in getattr(env, "robots", []):
        if hasattr(robot, "_joint_positions"):
            vals.append(np.asarray(robot._joint_positions, dtype=np.float32).reshape(-1))
    if not vals:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vals).astype(np.float32)


def step_model_tick(env, action: np.ndarray, policy_step: bool) -> None:
    if env.lite_physics:
        env.sim.step1()
    else:
        env.sim.forward()
    env._pre_action(action, policy_step=policy_step)
    if env.lite_physics:
        env.sim.step2()
    else:
        env.sim.step()
    env._update_observables()


def high_sample(env, action: np.ndarray, target_time_s: float, actual_time_s: float, sim_step_index: int) -> dict:
    return {
        "target_time_ms": np.float32(target_time_s * 1000.0),
        "time_ms": np.float32(actual_time_s * 1000.0),
        "sim_step_index": np.int64(sim_step_index),
        "robot_joint_pos": robot_joint_pos(env),
        "action": np.asarray(action, dtype=np.float32).copy(),
        "tactile": read_tactile_image(env).astype(np.float32),
    }


def step_with_interp_high200(env, action0: np.ndarray, action1: np.ndarray, low_step_index: int) -> list[dict]:
    if env.done:
        raise ValueError("executing action in terminated episode")

    env.timestep += 1
    start_time = float(env.cur_time)
    model_dt = float(env.model_timestep)
    control_dt = float(env.control_timestep)
    n_substeps = int(round(control_dt / model_dt))
    sample_dt = 0.005

    high_actions = np.linspace(
        np.asarray(action0, dtype=np.float32),
        np.asarray(action1, dtype=np.float32),
        10,
        endpoint=True,
        dtype=np.float32,
    )

    samples: list[dict] = []
    next_sample = 0
    last_bin = -1
    last_action = high_actions[0]
    for i in range(n_substeps):
        elapsed = i * model_dt
        high_bin = min(int((elapsed + 1e-12) / sample_dt), 9)
        last_action = high_actions[high_bin]
        step_model_tick(env, last_action, policy_step=(high_bin != last_bin))
        last_bin = high_bin

        actual_time = start_time + (i + 1) * model_dt
        while next_sample < 10:
            target_time = start_time + (next_sample + 1) * sample_dt
            if actual_time + 1e-12 < target_time:
                break
            global_sim_step = low_step_index * n_substeps + i + 1
            samples.append(
                high_sample(env, high_actions[next_sample], target_time, actual_time, global_sim_step)
            )
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
    env._post_action(last_action)
    return samples


def copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_group(src, dst) -> None:
    copy_attrs(src, dst)
    for k, obj in src.items():
        if isinstance(obj, h5py.Group):
            copy_group(obj, dst.create_group(k))
        else:
            kwargs = COMP if obj.ndim > 0 and obj.size > 1024 else {}
            dst.create_dataset(k, data=obj[()], **kwargs)


def render_tactile_grid(tactile: np.ndarray, cell: int = 128) -> np.ndarray:
    chans = np.asarray(tactile, dtype=np.float32)
    if chans.ndim != 3:
        raise ValueError(f"expected tactile (C,H,W), got {chans.shape}")
    c, h, w = chans.shape
    cols = int(np.ceil(np.sqrt(c)))
    rows = int(np.ceil(c / cols))
    panel = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for idx in range(c):
        img = np.clip(chans[idx], 0.0, 1.0)
        gray = (img * 255).astype(np.uint8)
        sy = max(int(np.ceil(cell / h)), 1)
        sx = max(int(np.ceil(cell / w)), 1)
        up = np.repeat(np.repeat(gray, sy, axis=0), sx, axis=1)
        up = up[:cell, :cell]
        heat = cv2.applyColorMap(up, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        r = idx // cols
        col = idx % cols
        panel[r * cell : (r + 1) * cell, col * cell : (col + 1) * cell] = heat
    return panel


def resize_nn(img: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def write_mp4(out_h5: Path, mp4_path: Path, fps: int) -> None:
    with h5py.File(out_h5, "r") as f:
        demo = next(iter(f["data"].keys()))
        g = f["data"][demo]
        rgb = g["low20/obs/agentview_image"]
        tactile = g["high200/tactile"]
        n_low = tactile.shape[0]
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(mp4_path), fps=fps, codec="libx264", quality=7)
        try:
            for low_i in range(n_low):
                rgb_panel = resize_nn(rgb[low_i], 4)
                for hi_i in range(10):
                    tac_panel = render_tactile_grid(tactile[low_i, hi_i], cell=168)
                    h = max(rgb_panel.shape[0], tac_panel.shape[0])
                    canvas = np.zeros((h, rgb_panel.shape[1] + tac_panel.shape[1], 3), dtype=np.uint8)
                    canvas[: rgb_panel.shape[0], : rgb_panel.shape[1]] = rgb_panel
                    canvas[: tac_panel.shape[0], rgb_panel.shape[1] :] = tac_panel
                    writer.append_data(canvas)
        finally:
            writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DEXMG_ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--out", required=True)
    p.add_argument("--mp4", default=None)
    p.add_argument("--mp4-fps", type=int, default=60)
    p.add_argument("--max-low-steps", type=int, default=0, help="0 means full demo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with h5py.File(args.dataset, "r") as fin:
        data = fin["data"]
        src = data[args.demo]
        states = src["states"][:]
        actions = src["actions"][:].astype(np.float32)
        n_low = min(states.shape[0] - 1, actions.shape[0] - 1)
        if args.max_low_steps > 0:
            n_low = min(n_low, args.max_low_steps)

        env = make_env_from_env_args(data.attrs["env_args"])
        reset_to(
            env,
            {
                "model": src.attrs["model_file"],
                "states": states[0],
                "ep_meta": src.attrs.get("ep_meta", None),
            },
        )
        tactile_shape = read_tactile_image(env).shape
        joint_dim = robot_joint_pos(env).shape[0]
        action_dim = actions.shape[-1]

        tactile = np.zeros((n_low, 10) + tactile_shape, dtype=np.float32)
        joint = np.zeros((n_low, 10, joint_dim), dtype=np.float32)
        high_action = np.zeros((n_low, 10, action_dim), dtype=np.float32)
        target_time_ms = np.zeros((n_low, 10), dtype=np.float32)
        actual_time_ms = np.zeros((n_low, 10), dtype=np.float32)
        sim_step_index = np.zeros((n_low, 10), dtype=np.int64)
        state_err = np.zeros((n_low,), dtype=np.float32)

        for i in range(n_low):
            samples = step_with_interp_high200(env, actions[i], actions[i + 1], i)
            for j, sample in enumerate(samples):
                tactile[i, j] = sample["tactile"]
                joint[i, j] = sample["robot_joint_pos"]
                high_action[i, j] = sample["action"]
                target_time_ms[i, j] = sample["target_time_ms"]
                actual_time_ms[i, j] = sample["time_ms"]
                sim_step_index[i, j] = sample["sim_step_index"]
            state_err[i] = float(np.max(np.abs(env.sim.get_state().flatten() - states[i + 1])))
            if (i + 1) % 25 == 0 or i + 1 == n_low:
                print(
                    f"[{i + 1}/{n_low}] state_err={state_err[i]:.4g} "
                    f"tactile_max={float(tactile[: i + 1].max()):.4g}",
                    flush=True,
                )

        with h5py.File(tmp_path, "w") as fout:
            fout.attrs["format"] = "dexmg_demo_low20_obs_high200_interp_action"
            fout.attrs["source_dataset"] = str(Path(args.dataset).resolve())
            fout.attrs["source_demo"] = args.demo
            fout.attrs["control_hz_original"] = 20
            fout.attrs["tactile_hz"] = 200
            fout.attrs["joint_hz"] = 200
            fout.attrs["action_hz"] = 200
            fout.attrs["action_semantics"] = (
                "true dynamics replay: controller goal updated with linspace(action_t, action_t+1, 10)"
            )
            fout.attrs["tactile_renderer"] = os.environ.get("TACTILE_RENDERER", "virtual")
            fout.attrs["tactile_sigma"] = float(os.environ.get("TACTILE_VIRTUAL_SIGMA", "8.0"))
            fout.attrs["tactile_force_scale"] = float(os.environ.get("TACTILE_VIRTUAL_FORCE_SCALE", "25.0"))
            fout.attrs["tactile_max"] = float(os.environ.get("TACTILE_VIRTUAL_MAX", "1.0"))
            fout.attrs["tactile_canonical"] = int(os.environ.get("TACTILE_VIRTUAL_CANONICAL", "1"))
            fout.attrs["tactile_max_surface_dist"] = float(
                os.environ.get("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015")
            )
            fout.attrs["tactile_video_colormap"] = "turbo"
            od = fout.create_group("data")
            copy_attrs(data, od)
            og = od.create_group(args.demo)
            copy_attrs(src, og)
            og.attrs["num_low_intervals"] = n_low
            og.attrs["state_err_p50"] = float(np.percentile(state_err, 50)) if len(state_err) else 0.0
            og.attrs["state_err_p95"] = float(np.percentile(state_err, 95)) if len(state_err) else 0.0
            og.attrs["state_err_max"] = float(np.max(state_err)) if len(state_err) else 0.0

            low = og.create_group("low20")
            copy_group(src["obs"], low.create_group("obs"))
            low.create_dataset("actions", data=actions[:n_low], **COMP)
            low.create_dataset("states", data=states[: n_low + 1], **COMP)
            high = og.create_group("high200")
            high.create_dataset("tactile", data=tactile, **COMP)
            high.create_dataset("robot_joint_pos", data=joint, **COMP)
            high.create_dataset("action", data=high_action, **COMP)
            high.create_dataset("target_time_ms", data=target_time_ms, **COMP)
            high.create_dataset("time_ms", data=actual_time_ms, **COMP)
            high.create_dataset("sim_step_index", data=sim_step_index, **COMP)
            high.create_dataset("state_err_after_low_step", data=state_err, **COMP)

    tmp_path.rename(out_path)
    print(f"[done] wrote {out_path} in {time.time() - t0:.1f}s", flush=True)
    if args.mp4:
        write_mp4(out_path, Path(args.mp4), fps=args.mp4_fps)
        print(f"[done] wrote {args.mp4}", flush=True)


if __name__ == "__main__":
    main()
