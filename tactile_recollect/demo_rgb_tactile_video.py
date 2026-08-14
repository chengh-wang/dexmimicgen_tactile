"""
Render one replayed DexMimicGen demo as RGB observations plus extracted tactile.

Input should be an HDF5 produced by tactile_recollect.extract, i.e. it contains
the original obs/* copied through and obs/robot0_tactile added by replaying the
stored states with tactile geoms injected.
"""
import argparse
import json

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import ENV_EMBODIMENT
from .inject import grid_index
from .layout import load_layout


CAM_PREF = [
    "frontview_image",
    "agentview_image",
    "sideview_image",
    "robot0_eye_in_right_hand_image",
    "robot0_eye_in_left_hand_image",
    "robot0_robotview_image",
    "robot1_robotview_image",
]


def _image_keys(obs):
    keys = [
        k for k in obs.keys()
        if k.endswith("_image") and getattr(obs[k], "ndim", 0) == 4
    ]
    ordered = [k for k in CAM_PREF if k in keys]
    ordered += sorted(k for k in keys if k not in ordered)
    return ordered[:3]


def _label_body(body):
    s = body
    for prefix in ("gripper0_right_", "gripper0_left_", "gripper1_right_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for prefix in ("R_", "L_", "r_", "l_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("_intermediate_link", "").replace("_distal_link", "").replace("_distal", "").replace("_", " ")


def _disp(img, vmax, gamma):
    n = np.clip(img / (vmax + 1e-9), 0.0, 1.0)
    if gamma != 1.0:
        n = n ** gamma
    return n


def _embodiment_from_dataset(f):
    env_args = json.loads(f["data"].attrs["env_args"])
    env_name = env_args["env_name"]
    if env_name in ENV_EMBODIMENT:
        return ENV_EMBODIMENT[env_name]
    if "CanSort" in env_name or "Coffee" in env_name or "Pouring" in env_name:
        return "gr1"
    if any(x in env_name for x in ("BoxCleanup", "DrawerCleanup", "LiftTray")):
        return "inspire"
    if any(x in env_name for x in ("Threading", "ThreePieceAssembly", "Transport")):
        return "panda"
    raise KeyError(f"cannot infer embodiment for env_name={env_name!r}")


def render(dataset, demo, out, fps=20, max_frames=0, gamma=0.5, pct=97.0,
           flip_images=False):
    f = h5py.File(dataset, "r")
    g = f[f"data/{demo}"]
    obs = g["obs"]
    tac = obs["robot0_tactile"][()]
    T = tac.shape[0] if max_frames <= 0 else min(tac.shape[0], max_frames)

    embodiment = _embodiment_from_dataset(f)
    layout, n = load_layout(embodiment=embodiment)
    bodies, _, _ = grid_index(layout, n)
    cams = _image_keys(obs)
    cam_arrs = {k: obs[k] for k in cams}

    vmax = []
    for bi in range(tac.shape[1]):
        vals = tac[:T, bi][tac[:T, bi] > 0]
        vmax.append(max(float(np.percentile(vals, pct)) if vals.size else 1.0, 1e-3))

    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    fig = plt.figure(figsize=(18, 11), dpi=100)
    try:
        for t in range(T):
            fig.clear()
            outer = fig.add_gridspec(
                2, 1,
                height_ratios=[1.05, 1.35],
                hspace=0.12,
                left=0.025,
                right=0.99,
                top=0.93,
                bottom=0.035,
            )

            gcam = outer[0].subgridspec(1, max(len(cams), 1), wspace=0.04)
            if cams:
                for ci, cam in enumerate(cams):
                    ax = fig.add_subplot(gcam[0, ci])
                    img = cam_arrs[cam][t]
                    ax.imshow(img[::-1] if flip_images else img)
                    ax.set_title(cam.replace("_image", ""), fontsize=12)
                    ax.axis("off")
            else:
                ax = fig.add_subplot(gcam[0, 0])
                ax.text(0.5, 0.5, "no stored RGB obs", ha="center", va="center")
                ax.axis("off")

            cols = 6 if embodiment in ("gr1", "inspire") else 4
            rows = int(np.ceil(len(bodies) / cols))
            gtac = outer[1].subgridspec(rows, cols, wspace=0.08, hspace=0.32)
            for bi, body in enumerate(bodies):
                ax = fig.add_subplot(gtac[bi // cols, bi % cols])
                ax.imshow(
                    _disp(tac[t, bi], vmax[bi], gamma),
                    cmap="inferno",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                peak = float(tac[t, bi].max())
                title = _label_body(body)
                ax.set_title(f"{title}\n{peak:.1f} N", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            for bi in range(len(bodies), rows * cols):
                ax = fig.add_subplot(gtac[bi // cols, bi % cols])
                ax.axis("off")

            fig.suptitle(
                f"{embodiment}  {demo}  frame {t + 1}/{T}  "
                f"RGB from original demo + tactile from replay",
                fontsize=15,
            )
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            writer.append_data(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3])
    finally:
        writer.close()
        plt.close(fig)
        f.close()
    print(f"wrote {out} ({T} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--pct", type=float, default=97.0)
    parser.add_argument("--flip-images", action="store_true")
    args = parser.parse_args()
    render(
        args.dataset,
        args.demo,
        args.out,
        fps=args.fps,
        max_frames=args.max_frames,
        gamma=args.gamma,
        pct=args.pct,
        flip_images=args.flip_images,
    )


if __name__ == "__main__":
    main()
