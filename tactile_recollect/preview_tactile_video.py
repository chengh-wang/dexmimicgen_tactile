"""
Generate a short tactile-layout preview video.

The video shows:
  * offscreen robosuite cameras with taxel geoms visible;
  * a small red probe moved across taxels;
  * the live read_tactile_image() force image caused by the probe.

This is a geometry / wiring preview. It does not require a demonstration
dataset, and it leaves normal env construction unchanged unless this script sets
TACTILE_VISIBLE and TACTILE_PREVIEW_PROBE.
"""
import argparse
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import make_tactile_env, read_tactile_image
from .layout import load_layout


TASK_BY_EMBODIMENT = {
    "inspire": "TwoArmBoxCleanup",
    "panda": "TwoArmThreading",
}


def _short_label(body):
    parts = body.split("_")
    arm = "R0" if body.startswith("gripper0_") else ("R1" if body.startswith("gripper1_") else "")
    if "palm" in parts:
        hand = "right" if "_r_" in body else ("left" if "_l_" in body else arm)
        return f"{hand} palm"
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in parts:
            hand = "right" if "_r_" in body else ("left" if "_l_" in body else arm)
            return f"{hand} {finger}"
    if "finger" in body:
        pad = "left pad" if "finger_joint1_tip" in body else "right pad"
        return f"{arm} {pad}"
    return body.replace("gripper", "g")


def _tile_shape(nbodies):
    if nbodies <= 4:
        return 1, nbodies
    return 2, int(np.ceil(nbodies / 2))


def _choose_taxel(layout, frame, total_frames):
    bodies = sorted(layout)
    block = max(total_frames / len(bodies), 1.0)
    bidx = min(int(frame / block), len(bodies) - 1)
    phase = (frame - bidx * block) / block
    body = bodies[bidx]
    count = len(layout[body]["pos"])
    tidx = min(int(phase * count), count - 1)
    return body, tidx


def _set_probe(env, body, taxel):
    model = env.sim.model
    data = env.sim.data
    bid = model.body_name2id(body)
    R = data.body_xmat[bid].reshape(3, 3)
    p_local = taxel["pos"] + taxel["rad"] * 0.001
    p_world = data.body_xpos[bid] + R @ p_local
    jid = model.joint_name2id("tactile_probe_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = p_world
    data.qpos[adr + 3:adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[model.jnt_dofadr[jid]:model.jnt_dofadr[jid] + 6] = 0.0
    env.sim.forward()
    return p_world


def _world_taxel_points(env, layout):
    out = {}
    for body, d in layout.items():
        bid = env.sim.model.body_name2id(body)
        R = env.sim.data.body_xmat[bid].reshape(3, 3)
        out[body] = env.sim.data.body_xpos[bid] + d["pos"] @ R.T
    return out


def _bounds(points_by_body):
    pts = np.concatenate(list(points_by_body.values()), axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo)) * 0.58, 0.02)
    return center, radius


def _render_camera(env, camera, width, height):
    img = env.sim.render(width=width, height=height, camera_name=camera)
    return img[::-1]


def _draw_position_panel(ax, points_by_body, center, radius, active_body, active_point,
                         title, elev, azim):
    ax.view_init(elev=elev, azim=azim)
    for body, pts in points_by_body.items():
        if body == active_body:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c="#6e7f8d", alpha=0.28, depthshade=False)
    pts = points_by_body[active_body]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3.0, c="crimson", alpha=0.9, depthshade=False)
    ax.scatter([active_point[0]], [active_point[1]], [active_point[2]],
               s=26, c="yellow", edgecolors="black", linewidths=0.5, depthshade=False)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))


def _draw_video_frame(fig, env, cameras, tactile, bodies, active_body, active_point,
                      points_by_body, point_center, point_radius, frame_idx,
                      total_frames, embodiment, width, height):
    fig.clear()
    rows, cols = _tile_shape(len(bodies))
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[0.9, 1.0, 1.45 if rows == 2 else 0.85],
        hspace=0.18,
        left=0.025,
        right=0.99,
        top=0.94,
        bottom=0.04,
    )

    gcam = outer[0].subgridspec(1, len(cameras), wspace=0.04)
    for i, cam in enumerate(cameras):
        ax = fig.add_subplot(gcam[0, i])
        ax.imshow(_render_camera(env, cam, width, height))
        ax.set_title(cam, fontsize=10)
        ax.axis("off")

    gpos = outer[1].subgridspec(1, 2, wspace=0.02)
    ax_iso = fig.add_subplot(gpos[0, 0], projection="3d")
    _draw_position_panel(
        ax_iso, points_by_body, point_center, point_radius, active_body, active_point,
        "taxel centers: oblique view", elev=22, azim=-58,
    )
    ax_top = fig.add_subplot(gpos[0, 1], projection="3d")
    _draw_position_panel(
        ax_top, points_by_body, point_center, point_radius, active_body, active_point,
        "taxel centers: top view", elev=84, azim=-90,
    )

    gheat = outer[2].subgridspec(rows, cols, wspace=0.08, hspace=0.28)
    vmax = max(float(tactile.max()), 1e-3)
    for bi, body in enumerate(bodies):
        ax = fig.add_subplot(gheat[bi // cols, bi % cols])
        im = np.clip(tactile[bi] / vmax, 0.0, 1.0) ** 0.5
        ax.imshow(im, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        peak = float(tactile[bi].max())
        color = "crimson" if body == active_body else "black"
        ax.set_title(f"{_short_label(body)}  {peak:.2f}N", fontsize=9, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        if body == active_body:
            for spine in ax.spines.values():
                spine.set_edgecolor("crimson")
                spine.set_linewidth(2.0)

    fig.suptitle(
        f"{embodiment} tactile preview  frame {frame_idx + 1}/{total_frames}  "
        f"peak={float(tactile.max()):.2f}N",
        fontsize=13,
    )
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]


def render_preview(embodiment, out, seconds=10.0, fps=10, task=None):
    os.environ["TACTILE_VISIBLE"] = "1"
    os.environ["TACTILE_PREVIEW_PROBE"] = "1"
    task = task or TASK_BY_EMBODIMENT[embodiment]
    frames = int(round(seconds * fps))

    layout, _ = load_layout(embodiment=embodiment)
    bodies = sorted(layout)
    env = make_tactile_env(task, has_offscreen_renderer=True, embodiment=embodiment)
    env.reset()
    points_by_body = _world_taxel_points(env, layout)
    point_center, point_radius = _bounds(points_by_body)

    cameras = ["agentview", "robot0_robotview", "robot1_robotview"]
    available = {env.sim.model.camera(i).name for i in range(env.sim.model.ncam)}
    cameras = [c for c in cameras if c in available]
    if not cameras:
        cameras = ["frontview"]

    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    fig = plt.figure(figsize=(14, 10), dpi=110)
    try:
        for frame_idx in range(frames):
            body, tidx = _choose_taxel(layout, frame_idx, frames)
            taxel = {k: layout[body][k][tidx] for k in ("pos", "rad")}
            active_point = _set_probe(env, body, taxel)
            tactile = read_tactile_image(env)
            frame = _draw_video_frame(
                fig, env, cameras, tactile, bodies, body, active_point,
                points_by_body, point_center, point_radius, frame_idx, frames,
                embodiment, width=360, height=260,
            )
            writer.append_data(frame)
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
    parser.add_argument("--task", default=None)
    args = parser.parse_args()
    render_preview(args.embodiment, args.out, seconds=args.seconds, fps=args.fps, task=args.task)


if __name__ == "__main__":
    main()
