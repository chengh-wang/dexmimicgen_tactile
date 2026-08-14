#!/usr/bin/env python3
"""Render one DexMimicGen demo with TacSL-style virtual tactile.

The video aligns stored RGB observations with tactile generated from the same
stored MuJoCo states. Set TACTILE_RENDERER=virtual so tactile_recollect.env uses
the contact-projector soft renderer instead of injected taxel collision boxes.
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


def image_keys(obs):
    preferred = ("agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image")
    keys = [k for k in preferred if k in obs]
    return keys or [k for k in obs.keys() if k.endswith("_image")]


def draw_video(rgb: dict[str, np.ndarray], tactile: np.ndarray, out: Path, title: str, fps: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, macro_block_size=None)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    positive = tactile[tactile > 0]
    vmax = max(float(np.percentile(positive, 99)) if positive.size else 1.0, 1e-3)
    try:
        for t in range(len(tactile)):
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
            cam_keys = list(rgb.keys())
            gcam = outer[0].subgridspec(1, len(cam_keys), wspace=0.04)
            for ci, key in enumerate(cam_keys):
                ax = fig.add_subplot(gcam[0, ci])
                ax.imshow(rgb[key][t])
                ax.set_title(key.replace("_image", ""), fontsize=10)
                ax.axis("off")

            gt = outer[1].subgridspec(1, tactile.shape[1], wspace=0.08)
            for ch in range(tactile.shape[1]):
                ax = fig.add_subplot(gt[0, ch])
                im = np.clip(tactile[t, ch] / vmax, 0, 1) ** 0.6
                ax.imshow(im, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"ch{ch} peak={float(tactile[t, ch].max()):.2f}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{title} frame {t + 1}/{len(tactile)}", fontsize=13)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            writer.append_data(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3])
    finally:
        writer.close()
        plt.close(fig)


def metrics(tactile: np.ndarray) -> dict[str, float]:
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    parser.add_argument("--demo", default="demo_258")
    parser.add_argument("--out", default=str(ROOT / "outputs/tactile_h5_videos/demo258_virtual_tacsl_style.mp4"))
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    os.environ["TACTILE_RENDERER"] = "virtual"

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        env = make_env_from_env_args(data.attrs["env_args"])
        g = data[args.demo]
        obs = g["obs"]
        keys = image_keys(obs)
        rgb = {k: obs[k][()] for k in keys}
        states = g["states"][()]
        tactile = np.zeros((len(states), 4, 32, 32), np.float32)
        reset_to(env, {"model": g.attrs["model_file"], "states": states[0], "ep_meta": g.attrs.get("ep_meta", None)})
        for t in range(len(states)):
            reset_to(env, {"states": states[t]})
            tactile[t] = read_tactile_image(env)

    out = Path(args.out)
    draw_video(rgb, tactile, out, f"{args.demo} virtual TacSL-style tactile", args.fps)
    metric_path = out.with_suffix(".metrics.json")
    metric_path.write_text(json.dumps(metrics(tactile), indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"wrote {metric_path}")


if __name__ == "__main__":
    main()

