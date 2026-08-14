#!/usr/bin/env python3
"""Generate 32x32 tactile-readout experiment videos for ThreePiece.

This script is intentionally an experiment harness, not a training-data writer.
It compares three cheap routes that all output ``(4, 32, 32)`` tactile:

1. ``gaussian``: original Panda pad contacts -> fixed Gaussian splat.
2. ``ellipse``: original Panda pad contacts -> fixed elliptical contact patch.
3. ``diffuse_raw``: existing raw taxel H5 -> sensor-space diffusion.

The first two run in a no-taxel MuJoCo env: the 1024 small collision geoms are
disabled, while the existing DexMG XML path fix is kept. This avoids the taxel
boxes changing physics and lets the 32x32 grid be a readout, not collision
geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robosuite"), str(ROOT)]

import tactile_recollect.env as tactile_env  # noqa: E402
from tactile_recollect.layout import layout_path_for, load_layout  # noqa: E402


PAD_GEOMS = [
    "gripper0_right_finger1_pad_collision",
    "gripper0_right_finger2_pad_collision",
    "gripper1_right_finger1_pad_collision",
    "gripper1_right_finger2_pad_collision",
]

PAD_BODIES = [
    "gripper0_right_finger_joint1_tip",
    "gripper0_right_finger_joint2_tip",
    "gripper1_right_finger_joint1_tip",
    "gripper1_right_finger_joint2_tip",
]


@dataclass(frozen=True)
class Variant:
    route: str
    name: str
    sigma: float = 2.5
    rx: float = 3.0
    ry: float = 6.0
    force_scale: float = 20.0
    blur_sigma: float = 2.0
    threshold: float = 1e-6
    canonical: bool = True


VARIANTS = [
    Variant("gaussian", "g_sigma1p5_s20", sigma=1.5, force_scale=20.0),
    Variant("gaussian", "g_sigma2p5_s20", sigma=2.5, force_scale=20.0),
    Variant("gaussian", "g_sigma3p5_s30", sigma=3.5, force_scale=30.0),
    Variant("ellipse", "e_rx2_ry4_s20", rx=2.0, ry=4.0, force_scale=20.0),
    Variant("ellipse", "e_rx3_ry6_s20", rx=3.0, ry=6.0, force_scale=20.0),
    Variant("ellipse", "e_rx4_ry8_s30", rx=4.0, ry=8.0, force_scale=30.0),
    Variant("diffuse_raw", "d_sigma1p5", blur_sigma=1.5, force_scale=20.0),
    Variant("diffuse_raw", "d_sigma2p5", blur_sigma=2.5, force_scale=20.0),
    Variant("diffuse_raw", "d_sigma3p5", blur_sigma=3.5, force_scale=30.0),
]


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


def make_no_taxel_env(env_args):
    """Keep tactile_recollect's DexMG path fix but disable taxel geom injection."""
    tactile_env._inject.inject_tactile = lambda *args, **kwargs: []
    return tactile_env.make_env_from_env_args(env_args)


def build_pad_projectors(env):
    layout, _ = load_layout(layout_path_for("panda"))
    out = []
    for ch, (body, geom) in enumerate(zip(PAD_BODIES, PAD_GEOMS)):
        d = layout[body]
        pos = d["pos"]
        ij = d["ij"]
        p00 = pos[np.where((ij[:, 0] == 0) & (ij[:, 1] == 0))[0][0]]
        p31 = pos[np.where((ij[:, 0] == 31) & (ij[:, 1] == 0))[0][0]]
        p03 = pos[np.where((ij[:, 0] == 0) & (ij[:, 1] == 31))[0][0]]
        vi = p31 - p00
        vj = p03 - p00
        basis = np.stack([vi, vj], axis=1)
        pinv = np.linalg.pinv(basis)
        gid = env.sim.model.geom_name2id(geom)
        bid = env.sim.model.body_name2id(body)
        out.append(dict(ch=ch, geom=geom, gid=gid, body=body, bid=bid, p00=p00, pinv=pinv))
    return out


def canonicalize(img: np.ndarray) -> np.ndarray:
    """Put opposing Panda fingers into a consistent visual convention.

    Channels 1 and 3 are the opposing finger pads. Flip both axes so a symmetric
    grasp has roughly symmetric maps instead of purely local-frame diagonals.
    """
    y = img.copy()
    y[1] = y[1, ::-1, ::-1]
    y[3] = y[3, ::-1, ::-1]
    return y


def contact_events(env, projectors, keep_base: bool = False):
    geom_to_projector = {p["gid"]: p for p in projectors}
    model = env.sim.model
    m = model._model
    d = env.sim.data._data
    f6 = np.zeros(6)
    events = []
    allowed = ("piece_1_root", "piece_2_root", "base_root") if keep_base else ("piece_1_root", "piece_2_root")
    for ci in range(d.ncon):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in geom_to_projector:
            proj, other = geom_to_projector[g1], g2
        elif g2 in geom_to_projector:
            proj, other = geom_to_projector[g2], g1
        else:
            continue
        other_body = model.body_id2name(int(model.geom_bodyid[other]))
        if other_body not in allowed:
            continue
        mujoco.mj_contactForce(m, d, ci, f6)
        force = abs(float(f6[0]))
        if force <= 0:
            continue
        bid = proj["bid"]
        R = env.sim.data.body_xmat[bid].reshape(3, 3)
        p_local = R.T @ (np.array(c.pos) - env.sim.data.body_xpos[bid])
        uv = proj["pinv"] @ (p_local - proj["p00"])
        i = float(np.clip(31.0 * uv[0], 0.0, 31.0))
        j = float(np.clip(31.0 * uv[1], 0.0, 31.0))
        events.append((proj["ch"], i, j, force))
    return events


def add_gaussian(img, ch, i, j, force, sigma, force_scale):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    i0, i1 = max(0, int(math.floor(i)) - radius), min(31, int(math.floor(i)) + radius)
    j0, j1 = max(0, int(math.floor(j)) - radius), min(31, int(math.floor(j)) + radius)
    ii = np.arange(i0, i1 + 1, dtype=np.float32)[:, None]
    jj = np.arange(j0, j1 + 1, dtype=np.float32)[None, :]
    w = np.exp(-0.5 * (((ii - i) / sigma) ** 2 + ((jj - j) / sigma) ** 2))
    img[ch, i0 : i1 + 1, j0 : j1 + 1] += (force / force_scale) * w


def add_ellipse(img, ch, i, j, force, rx, ry, force_scale):
    radius_i = max(1, int(math.ceil(rx)))
    radius_j = max(1, int(math.ceil(ry)))
    i0, i1 = max(0, int(math.floor(i)) - radius_i), min(31, int(math.floor(i)) + radius_i)
    j0, j1 = max(0, int(math.floor(j)) - radius_j), min(31, int(math.floor(j)) + radius_j)
    ii = np.arange(i0, i1 + 1, dtype=np.float32)[:, None]
    jj = np.arange(j0, j1 + 1, dtype=np.float32)[None, :]
    r2 = ((ii - i) / rx) ** 2 + ((jj - j) / ry) ** 2
    w = np.clip(1.0 - r2, 0.0, 1.0)
    s = float(w.sum())
    if s > 1e-9:
        w = w / s * max(rx * ry * 0.75, 1.0)
    img[ch, i0 : i1 + 1, j0 : j1 + 1] += (force / force_scale) * w


def render_virtual_frame(events, variant: Variant):
    img = np.zeros((4, 32, 32), np.float32)
    for ch, i, j, force in events:
        if variant.route == "gaussian":
            add_gaussian(img, ch, i, j, force, variant.sigma, variant.force_scale)
        elif variant.route == "ellipse":
            add_ellipse(img, ch, i, j, force, variant.rx, variant.ry, variant.force_scale)
        else:
            raise ValueError(variant.route)
    img = np.clip(img, 0.0, 1.0)
    return canonicalize(img) if variant.canonical else img


def gaussian_kernel1d(sigma: float):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur2d(x: np.ndarray, sigma: float):
    k = gaussian_kernel1d(sigma)
    r = len(k) // 2
    y = np.pad(x, ((0, 0), (r, r), (0, 0)), mode="edge")
    tmp = np.zeros_like(x, dtype=np.float32)
    for a, w in enumerate(k):
        tmp += float(w) * y[:, a : a + x.shape[1], :]
    y = np.pad(tmp, ((0, 0), (0, 0), (r, r)), mode="edge")
    out = np.zeros_like(x, dtype=np.float32)
    for a, w in enumerate(k):
        out += float(w) * y[:, :, a : a + x.shape[2]]
    return out


def render_diffuse_raw(raw: np.ndarray, variant: Variant):
    img = np.clip(raw.astype(np.float32), 0.0, None) / max(variant.force_scale, 1e-6)
    img = blur2d(img, variant.blur_sigma)
    img = np.clip(img, 0.0, 1.0)
    return canonicalize(img) if variant.canonical else img


def compute_metrics(tactile: np.ndarray):
    x = np.asarray(tactile, dtype=np.float32)
    total = float(x.sum()) + 1e-9
    edge = np.zeros((32, 32), bool)
    edge[:4, :] = edge[-4:, :] = edge[:, :4] = edge[:, -4:] = True
    center16 = np.zeros((32, 32), bool)
    center16[8:24, 8:24] = True
    center8 = np.zeros((32, 32), bool)
    center8[12:20, 12:20] = True
    frame_sum = x.sum(axis=(1, 2, 3))
    active = frame_sum > 1e-6
    pixels = (x > 1e-4).sum(axis=(1, 2, 3))
    diff = np.abs(np.diff(x, axis=0)).sum(axis=(1, 2, 3))
    return {
        "frames": int(len(x)),
        "contact_frame_frac": float(active.mean()),
        "mean_active_pixels_contact_frames": float(pixels[active].mean()) if active.any() else 0.0,
        "edge4_force_mass": float(x[:, :, edge].sum() / total),
        "center16_force_mass": float(x[:, :, center16].sum() / total),
        "center8_force_mass": float(x[:, :, center8].sum() / total),
        "mean_frame_peak": float(x.reshape(len(x), -1).max(axis=1).mean()),
        "p95_frame_peak": float(np.percentile(x.reshape(len(x), -1).max(axis=1), 95)),
        "temporal_l1_mean": float(diff.mean()) if len(diff) else 0.0,
    }


def image_keys(obs):
    pref = ["agentview_image", "robot0_eye_in_hand_image", "robot1_eye_in_hand_image"]
    return [k for k in pref if k in obs]


def draw_video(rgb, tactile, out, title, fps=20):
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    vmax = max(float(np.percentile(tactile[tactile > 0], 99)) if np.any(tactile > 0) else 1.0, 1e-3)
    try:
        for t in range(len(tactile)):
            fig.clear()
            outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.16, left=0.025, right=0.99, top=0.92, bottom=0.04)
            cams = list(rgb)
            gcam = outer[0].subgridspec(1, len(cams), wspace=0.04)
            for ci, key in enumerate(cams):
                ax = fig.add_subplot(gcam[0, ci])
                ax.imshow(rgb[key][t])
                ax.set_title(key.replace("_image", ""), fontsize=10)
                ax.axis("off")
            gt = outer[1].subgridspec(1, 4, wspace=0.08)
            for ch in range(4):
                ax = fig.add_subplot(gt[0, ch])
                im = np.clip(tactile[t, ch] / vmax, 0, 1) ** 0.6
                ax.imshow(im, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"ch{ch} peak={float(tactile[t,ch].max()):.2f}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{title} frame {t + 1}/{len(tactile)}", fontsize=13)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
            writer.append_data(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3])
    finally:
        writer.close()
        plt.close(fig)


def generate_contact_tactile(raw_h5, demo, variants):
    with h5py.File(raw_h5, "r") as f:
        env = make_no_taxel_env(f["data"].attrs["env_args"])
        g = f[f"data/{demo}"]
        states = g["states"][()]
        reset_to(env, {"model": g.attrs["model_file"], "states": states[0], "ep_meta": g.attrs.get("ep_meta", None)})
        projectors = build_pad_projectors(env)
        out = {v.name: np.zeros((len(states), 4, 32, 32), np.float32) for v in variants if v.route in ("gaussian", "ellipse")}
        for t in range(len(states)):
            reset_to(env, {"states": states[t]})
            events = contact_events(env, projectors)
            for v in variants:
                if v.route in ("gaussian", "ellipse"):
                    out[v.name][t] = render_virtual_frame(events, v)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-h5", default=str(ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    parser.add_argument("--raw-tactile-h5", default=str(ROOT / "datasets/generated_tactile_actionrollout_proud2mm_shards/20260725_173000/two_arm_three_piece_assembly/shard00.hdf5"))
    parser.add_argument("--demo", default="demo_258")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/virtual_tactile_experiments"))
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = VARIANTS

    with h5py.File(args.raw_h5, "r") as f:
        obs = f[f"data/{args.demo}/obs"]
        rgb = {k: obs[k][()] for k in image_keys(obs)}
        T = next(iter(rgb.values())).shape[0]

    contact_maps = generate_contact_tactile(args.raw_h5, args.demo, variants)

    with h5py.File(args.raw_tactile_h5, "r") as f:
        raw_tactile = f[f"data/{args.demo}/obs/robot0_tactile"][:T]

    metrics = {}
    for v in variants:
        if v.route == "diffuse_raw":
            tactile = np.stack([render_diffuse_raw(raw_tactile[t], v) for t in range(T)], axis=0)
        else:
            tactile = contact_maps[v.name]
        metrics[v.name] = {"route": v.route, "params": v.__dict__, **compute_metrics(tactile)}
        out = out_dir / f"{args.demo}_{v.name}.mp4"
        draw_video(rgb, tactile, str(out), f"{args.demo} {v.route} {v.name}", fps=args.fps)
        print(f"wrote {out}")

    metrics_path = out_dir / f"{args.demo}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
