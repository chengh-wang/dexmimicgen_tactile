"""
Render close-up videos of tactile coverage on the actual hand/gripper meshes.

This is a visual inspection helper: it sets TACTILE_VISIBLE=1 and enlarges the
taxel boxes via TACTILE_VISUAL_SCALE so the covered regions are obvious. The
layout files and normal tactile extraction path are not modified.
"""
import argparse
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .env import make_tactile_env
from .layout import load_layout


TASK_BY_EMBODIMENT = {
    "gr1": "TwoArmCoffee",
    "inspire": "TwoArmBoxCleanup",
    "panda": "TwoArmThreading",
}


def _body_points(env, layout, bodies):
    points = []
    for body in bodies:
        d = layout[body]
        bid = env.sim.model.body_name2id(body)
        R = env.sim.data.body_xmat[bid].reshape(3, 3)
        points.append(env.sim.data.body_xpos[bid] + d["pos"] @ R.T)
    return np.concatenate(points, axis=0)


def _render_free(env, points, azimuth, elevation, distance_scale, width, height):
    ctx = env.sim._render_context_offscreen
    if ctx.scn.maxgeom < 50000:
        ctx.scn = mujoco.MjvScene(env.sim.model._model, maxgeom=50000)
    ctx.vopt.geomgroup[4] = 1
    ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    center = points.mean(axis=0)
    span = max(float(np.ptp(points, axis=0).max()), 0.025)
    ctx.cam.lookat[:] = center
    ctx.cam.distance = span * distance_scale
    ctx.cam.azimuth = azimuth
    ctx.cam.elevation = elevation
    return env.sim.render(width=width, height=height, camera_name=None)[::-1]


def _region_groups(embodiment, layout):
    bodies = sorted(layout)
    if embodiment == "inspire":
        right = [b for b in bodies if b.startswith("gripper0_right_r_")]
        left = [b for b in bodies if b.startswith("gripper1_right_l_")]
        return [
            ("right palm face", right, 60, 20, 1.25),
            ("right fingertips/palm", right, -100, 5, 1.45),
            ("left palm face", left, -60, 20, 1.25),
            ("left fingertips/palm", left, 100, 5, 1.45),
        ]
    if embodiment == "gr1":
        right = [b for b in bodies if b.startswith("gripper0_right_")]
        left = [b for b in bodies if b.startswith("gripper0_left_")]
        return [
            ("right palm face", right, 60, 20, 1.25),
            ("right fingertips/palm", right, -100, 5, 1.45),
            ("left palm face", left, -60, 20, 1.25),
            ("left fingertips/palm", left, 100, 5, 1.45),
        ]
    return [
        ("robot0 left pad", [b for b in bodies if b == "gripper0_right_finger_joint1_tip"], 0, 20, 1.8),
        ("robot0 right pad", [b for b in bodies if b == "gripper0_right_finger_joint2_tip"], 180, 20, 1.8),
        ("robot1 left pad", [b for b in bodies if b == "gripper1_right_finger_joint1_tip"], 0, 20, 1.8),
        ("robot1 right pad", [b for b in bodies if b == "gripper1_right_finger_joint2_tip"], 180, 20, 1.8),
    ]


def _draw_point_cloud(ax, env, layout, active_bodies):
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]
    all_pts = []
    for idx, body in enumerate(sorted(layout)):
        pts = _body_points(env, layout, [body])
        all_pts.append(pts)
        active = body in active_bodies
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=3.0 if active else 0.8,
            c=colors[idx % len(colors)] if active else "#748895",
            alpha=0.95 if active else 0.18,
            depthshade=False,
        )
    pts = np.concatenate(all_pts, axis=0)
    center = pts.mean(axis=0)
    radius = max(float(np.ptp(pts, axis=0).max()) * 0.6, 0.025)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=25, azim=-60)
    ax.set_title("taxel center coverage map", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))


def render_coverage(embodiment, out, seconds=10.0, fps=10, visual_scale=4.0):
    os.environ["TACTILE_VISIBLE"] = "1"
    os.environ["TACTILE_PREVIEW_PROBE"] = "0"
    os.environ["TACTILE_VISUAL_SCALE"] = str(visual_scale)

    layout, _ = load_layout(embodiment=embodiment)
    env = make_tactile_env(
        TASK_BY_EMBODIMENT[embodiment],
        has_offscreen_renderer=True,
        embodiment=embodiment,
    )
    env.reset()
    groups = _region_groups(embodiment, layout)

    frames = int(round(seconds * fps))
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    fig = plt.figure(figsize=(16, 10), dpi=100)
    try:
        for t in range(frames):
            phase = np.sin(2.0 * np.pi * t / max(frames - 1, 1))
            fig.clear()
            gs = fig.add_gridspec(
                3, 2,
                height_ratios=[1.0, 1.0, 1.05],
                hspace=0.16,
                wspace=0.06,
                left=0.025,
                right=0.99,
                top=0.93,
                bottom=0.035,
            )
            active = []
            for i, (title, bodies, az, el, dist) in enumerate(groups):
                active.extend(bodies)
                ax = fig.add_subplot(gs[i // 2, i % 2])
                pts = _body_points(env, layout, bodies)
                img = _render_free(
                    env,
                    pts,
                    azimuth=az + 12.0 * phase,
                    elevation=el + 4.0 * phase,
                    distance_scale=dist,
                    width=680,
                    height=430,
                )
                ax.imshow(img)
                ax.set_title(title, fontsize=11)
                ax.axis("off")

            ax3 = fig.add_subplot(gs[2, :], projection="3d")
            _draw_point_cloud(ax3, env, layout, set(active))
            fig.suptitle(
                f"{embodiment} close-up tactile coverage "
                f"(blue enlarged taxel boxes, visual scale {visual_scale:g}x)",
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
    parser.add_argument("--embodiment", choices=sorted(TASK_BY_EMBODIMENT), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--visual-scale", type=float, default=4.0)
    args = parser.parse_args()
    render_coverage(
        args.embodiment,
        args.out,
        seconds=args.seconds,
        fps=args.fps,
        visual_scale=args.visual_scale,
    )


if __name__ == "__main__":
    main()
