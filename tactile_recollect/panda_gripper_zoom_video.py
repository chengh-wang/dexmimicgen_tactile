"""
Render a far-to-close dolly-in video for the Panda parallel gripper pads.

This is only a visual inspection helper. It can render either the default
layout or a temporary candidate layout via --layout-path.
"""
import argparse
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .closeup_tactile_coverage_video import _body_points, _render_free
from .env import make_tactile_env
from .layout import load_layout


def _smoothstep(x):
    x = np.clip(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _panda_groups(layout):
    bodies = sorted(layout)
    return [
        ("robot0 gripper", [b for b in bodies if b.startswith("gripper0_right_")]),
        ("robot1 gripper", [b for b in bodies if b.startswith("gripper1_right_")]),
    ]


def render_zoom(out, layout_path=None, seconds=10.0, fps=10, visual_scale=5.0,
                title="panda parallel gripper tactile pads"):
    os.environ["TACTILE_VISIBLE"] = "1"
    os.environ["TACTILE_PREVIEW_PROBE"] = "0"
    os.environ["TACTILE_VISUAL_SCALE"] = str(visual_scale)
    if layout_path is not None:
        os.environ["TACTILE_LAYOUT_PATH"] = os.path.abspath(layout_path)

    layout, _ = load_layout(path=layout_path, embodiment="panda")
    env = make_tactile_env(
        "TwoArmThreading",
        has_offscreen_renderer=True,
        embodiment="panda",
        layout_path=layout_path,
    )
    env.reset()

    frames = int(round(seconds * fps))
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    fig = plt.figure(figsize=(13, 6), dpi=100)
    try:
        for t in range(frames):
            u = _smoothstep(t / max(frames - 1, 1))
            distance_scale = 7.0 * (1.0 - u) + 1.45 * u
            az_swing = 8.0 * np.sin(2.0 * np.pi * t / max(frames - 1, 1))

            fig.clear()
            gs = fig.add_gridspec(
                1, 2,
                left=0.015,
                right=0.985,
                top=0.88,
                bottom=0.04,
                wspace=0.035,
            )
            for i, (name, bodies) in enumerate(_panda_groups(layout)):
                ax = fig.add_subplot(gs[0, i])
                pts = _body_points(env, layout, bodies)
                img = _render_free(
                    env,
                    pts,
                    azimuth=0.0 + az_swing,
                    elevation=8.0,
                    distance_scale=distance_scale,
                    width=720,
                    height=520,
                )
                ax.imshow(img)
                ax.set_title(name, fontsize=12)
                ax.axis("off")

            fig.suptitle(
                f"{title}  |  blue taxel boxes, visual scale {visual_scale:g}x",
                fontsize=14,
            )
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            writer.append_data(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3])
    finally:
        writer.close()
        plt.close(fig)
        env.close()
    print(f"wrote {out} ({frames} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--layout-path", default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--visual-scale", type=float, default=5.0)
    parser.add_argument("--title", default="panda parallel gripper tactile pads")
    args = parser.parse_args()
    render_zoom(
        args.out,
        layout_path=args.layout_path,
        seconds=args.seconds,
        fps=args.fps,
        visual_scale=args.visual_scale,
        title=args.title,
    )


if __name__ == "__main__":
    main()
