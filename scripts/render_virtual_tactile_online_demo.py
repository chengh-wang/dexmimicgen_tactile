#!/usr/bin/env python3
"""Render online action replay with TacSL-style virtual tactile.

Unlike render_virtual_tactile_demo.py, this does not force every recorded state.
It resets to the demo initial state, steps recorded actions in MuJoCo, and reads
the tactile observable from the live simulator state after each step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robosuite"), str(ROOT)]
os.environ.setdefault("MUJOCO_GL", "egl")

from tactile_recollect.env import make_env_from_env_args, read_tactile_image  # noqa: E402


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


def render_cameras(env, camera_names: list[str], height: int, width: int) -> dict[str, np.ndarray]:
    return {
        name: env.sim.render(height=height, width=width, camera_name=name)[::-1]
        for name in camera_names
    }


def infer_cameras(env, obs_group, requested: list[str] | None) -> list[str]:
    if requested:
        candidates = requested
    else:
        preferred = (
            "agentview_image",
            "frontview_image",
            "robot0_eye_in_hand_image",
            "robot1_eye_in_hand_image",
            "robot0_eye_in_left_hand_image",
            "robot0_eye_in_right_hand_image",
            "shouldercamera0_image",
            "shouldercamera1_image",
        )
        obs_keys = [k for k in preferred if k in obs_group]
        obs_keys += sorted(k for k in obs_group.keys() if k.endswith("_image") and k not in obs_keys)
        candidates = [k.removesuffix("_image") for k in obs_keys]

    valid = []
    for name in candidates:
        try:
            env.sim.model.camera_name2id(name)
        except ValueError:
            continue
        if name not in valid:
            valid.append(name)
    if not valid:
        raise RuntimeError(f"No requested or dataset image cameras exist in this env: {candidates}")
    return valid


def draw_frame(fig, rgb: dict[str, np.ndarray], tactile: np.ndarray, title: str, vmax: float) -> np.ndarray:
    fig.clear()
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.1],
        hspace=0.16,
        left=0.025,
        right=0.99,
        top=0.92,
        bottom=0.04,
    )
    keys = list(rgb.keys())
    gcam = outer[0].subgridspec(1, len(keys), wspace=0.04)
    for ci, key in enumerate(keys):
        ax = fig.add_subplot(gcam[0, ci])
        ax.imshow(rgb[key])
        ax.set_title(key, fontsize=10)
        ax.axis("off")

    gt = outer[1].subgridspec(1, tactile.shape[0], wspace=0.08)
    for ch in range(tactile.shape[0]):
        ax = fig.add_subplot(gt[0, ch])
        im = np.clip(tactile[ch] / vmax, 0, 1) ** 0.6
        ax.imshow(im, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"ch{ch} peak={float(tactile[ch].max()):.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]


def metrics(tactile: np.ndarray, state_err: np.ndarray) -> dict[str, float]:
    x = np.asarray(tactile, dtype=np.float32)
    total = float(x.sum()) + 1e-9
    edge = np.zeros((32, 32), bool)
    edge[:4, :] = edge[-4:, :] = edge[:, :4] = edge[:, -4:] = True
    center8 = np.zeros((32, 32), bool)
    center8[12:20, 12:20] = True
    frame_sum = x.sum(axis=(1, 2, 3))
    active = frame_sum > 1e-6
    return {
        "frames": int(len(x)),
        "contact_frame_frac": float(active.mean()),
        "mean_active_pixels_contact_frames": float((x > 1e-4).sum(axis=(1, 2, 3))[active].mean())
        if active.any()
        else 0.0,
        "edge4_force_mass": float(x[:, :, edge].sum() / total),
        "center8_force_mass": float(x[:, :, center8].sum() / total),
        "max_value": float(x.max()) if x.size else 0.0,
        "state_err_p50": float(np.percentile(state_err, 50)) if len(state_err) else 0.0,
        "state_err_p95": float(np.percentile(state_err, 95)) if len(state_err) else 0.0,
        "state_err_max": float(np.max(state_err)) if len(state_err) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    parser.add_argument("--demo", default="demo_258")
    parser.add_argument("--out", default=str(ROOT / "outputs/tactile_h5_videos/demo258_online_virtual_tacsl_s8_fs25.mp4"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
    )
    args = parser.parse_args()

    os.environ["TACTILE_RENDERER"] = "virtual"

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        env = make_env_from_env_args(
            data.attrs["env_args"],
            has_offscreen_renderer=True,
            use_camera_obs=False,
        )
        g = data[args.demo]
        camera_names = infer_cameras(env, g["obs"], args.cameras)
        print(f"[setup] cameras={camera_names}", flush=True)
        states = g["states"][()]
        actions = g["actions"][()].astype(np.float32)
        reset_to(env, {"model": g.attrs["model_file"], "states": states[0], "ep_meta": g.attrs.get("ep_meta", None)})

        state_err = np.zeros(max(len(states) - 1, 0), np.float32)
        frames = []
        positive_values = []
        rgb_cache = []
        first_tactile = read_tactile_image(env)
        tactile = np.zeros((len(states),) + first_tactile.shape, np.float32)
        tactile[0] = first_tactile
        rgb_cache.append(render_cameras(env, camera_names, args.height, args.width))
        if np.any(tactile[0] > 0):
            positive_values.append(tactile[0][tactile[0] > 0])

        for t in range(1, len(states)):
            env.step(actions[t - 1])
            tactile[t] = read_tactile_image(env)
            rgb_cache.append(render_cameras(env, camera_names, args.height, args.width))
            state_err[t - 1] = np.max(np.abs(env.sim.get_state().flatten() - states[t]))
            if np.any(tactile[t] > 0):
                positive_values.append(tactile[t][tactile[t] > 0])

    positive = np.concatenate(positive_values) if positive_values else np.array([1.0], dtype=np.float32)
    vmax = max(float(np.percentile(positive, 99)), 1e-3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, macro_block_size=None)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    try:
        for t, rgb in enumerate(rgb_cache):
            frame = draw_frame(
                fig,
                rgb,
                tactile[t],
                f"{args.demo} online action replay virtual TacSL-style s8/fs25 frame {t + 1}/{len(tactile)}",
                vmax,
            )
            frames.append(frame)
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)

    metric_path = out.with_suffix(".metrics.json")
    metric_path.write_text(json.dumps(metrics(tactile, state_err), indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"wrote {metric_path}")


if __name__ == "__main__":
    main()
