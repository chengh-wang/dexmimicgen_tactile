#!/usr/bin/env python3
"""Render an online action replay with a TacSL-like spring-damper tactile field.

This is a fast MuJoCo approximation of the TacSL force-field idea:

  * keep the existing 32x32 tactile layout and online action replay path
  * do not inject taxel collision geoms
  * for every taxel point, ray-cast along the local outward normal
  * convert distance inside a soft skin thickness to penetration depth
  * apply a spring-damper law per taxel

It is intentionally separate from the training/eval renderer so it cannot change
the current virtual_s8fs25 policy pipeline.
"""

from __future__ import annotations

import argparse
import json
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
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robosuite"), str(ROOT)]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["TACTILE_RENDERER"] = "virtual"
os.environ.setdefault("TACTILE_VIRTUAL_SIGMA", "8.0")
os.environ.setdefault("TACTILE_VIRTUAL_FORCE_SCALE", "25.0")
os.environ.setdefault("TACTILE_VIRTUAL_MAX", "1.0")
os.environ.setdefault("TACTILE_VIRTUAL_CANONICAL", "1")
os.environ.setdefault("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015")

from tactile_recollect.env import make_env_from_env_args  # noqa: E402
from tactile_recollect.layout import PROUD, load_layout  # noqa: E402


@dataclass
class Patch:
    ch: int
    body: str
    bid: int
    pos: np.ndarray
    rad: np.ndarray
    ij: np.ndarray


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
        raise RuntimeError(f"No valid cameras in env from candidates={candidates}")
    return valid


def render_cameras(env, camera_names: list[str], height: int, width: int) -> dict[str, np.ndarray]:
    return {
        name: env.sim.render(height=height, width=width, camera_name=name)[::-1]
        for name in camera_names
    }


def build_patches(env) -> list[Patch]:
    layout, n = load_layout(getattr(env, "_tactile_layout_path", None))
    if n != 32:
        raise RuntimeError(f"expected 32x32 tactile layout, got {n}x{n}")
    patches: list[Patch] = []
    for ch, body in enumerate(sorted(layout.keys())):
        try:
            bid = int(env.sim.model.body_name2id(body))
        except Exception:
            continue
        d = layout[body]
        pos = np.asarray(d["pos"], dtype=np.float64)
        rad = np.asarray(d["rad"], dtype=np.float64)
        ij = np.asarray(d["ij"], dtype=np.int32)
        if len(pos) == 0:
            continue
        patches.append(Patch(ch=ch, body=body, bid=bid, pos=pos, rad=rad, ij=ij))
    if not patches:
        raise RuntimeError("no tactile patches found for this env/layout")
    return patches


def canonicalize(img: np.ndarray) -> np.ndarray:
    y = img.copy()
    if y.shape[0] == 4:
        y[1] = y[1, ::-1, ::-1]
        y[3] = y[3, ::-1, ::-1]
    return y


def gaussian_blur_channels(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = k / max(float(k.sum()), 1e-9)
    y = np.pad(img, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    tmp = np.zeros_like(img, dtype=np.float32)
    for a, w in enumerate(k):
        tmp += float(w) * y[:, a : a + img.shape[1], :]
    y = np.pad(tmp, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for a, w in enumerate(k):
        out += float(w) * y[:, :, a : a + img.shape[2]]
    return out


def read_tacsl_spring_image(
    env,
    patches: list[Patch],
    prev_depths: list[np.ndarray] | None,
    dt: float,
    skin: float,
    kn: float,
    kd: float,
    force_scale: float,
    max_value: float,
    surface_offset: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    model = env.sim.model
    data = env.sim.data
    m = model._model
    d = data._data
    img = np.zeros((max(p.ch for p in patches) + 1, 32, 32), np.float32)
    next_depths: list[np.ndarray] = []
    geomid = np.zeros(1, dtype=np.int32)

    for pi, patch in enumerate(patches):
        body_rot = data.body_xmat[patch.bid].reshape(3, 3)
        body_pos = data.body_xpos[patch.bid]
        pos_w = body_pos[None, :] + patch.pos @ body_rot.T
        rad_w = patch.rad @ body_rot.T
        rad_w /= np.linalg.norm(rad_w, axis=1, keepdims=True) + 1e-12
        pos_w = pos_w + surface_offset * rad_w
        prev = np.zeros(len(patch.pos), dtype=np.float32) if prev_depths is None else prev_depths[pi]
        cur = np.zeros(len(patch.pos), dtype=np.float32)

        for k in range(len(patch.pos)):
            geomid[0] = -1
            dist = mujoco.mj_ray(
                m,
                d,
                pos_w[k],
                rad_w[k],
                None,
                1,
                int(patch.bid),
                geomid,
            )
            if dist < 0 or dist > skin:
                continue
            depth = skin - float(dist)
            d_dot = (depth - float(prev[k])) / max(dt, 1e-6)
            force = max(kn * depth + kd * d_dot, 0.0)
            i, j = patch.ij[k]
            img[patch.ch, int(i), int(j)] = min(max_value, force / force_scale)
            cur[k] = depth
        next_depths.append(cur)

    if max_value > 0:
        img = np.clip(img, 0.0, max_value)
    return canonicalize(img), next_depths


def read_tacsl_multiray_image(
    env,
    patches: list[Patch],
    prev_depths: list[np.ndarray] | None,
    dt: float,
    skin: float,
    kn: float,
    kd: float,
    force_scale: float,
    max_value: float,
    ray_grid: int,
    ray_radius: float,
    surface_offset: float,
    surface_base_proud: float,
    blur_sigma: float,
    front_depth: float,
    back_depth: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    model = env.sim.model
    data = env.sim.data
    m = model._model
    d = data._data
    img = np.zeros((max(p.ch for p in patches) + 1, 32, 32), np.float32)
    next_depths: list[np.ndarray] = []
    geomid = np.zeros(1, dtype=np.int32)
    offsets_1d = np.linspace(-ray_radius, ray_radius, max(ray_grid, 1), dtype=np.float64)

    for pi, patch in enumerate(patches):
        body_rot = data.body_xmat[patch.bid].reshape(3, 3)
        body_pos = data.body_xpos[patch.bid]
        pos_w = body_pos[None, :] + patch.pos @ body_rot.T
        rad_w = patch.rad @ body_rot.T
        rad_w /= np.linalg.norm(rad_w, axis=1, keepdims=True) + 1e-12
        pos_w = pos_w - surface_base_proud * rad_w
        pos_w = pos_w + surface_offset * rad_w
        ax_w = np.asarray(load_layout(getattr(env, "_tactile_layout_path", None))[0][patch.body]["ax"]) @ body_rot.T
        ctan_w = np.asarray(load_layout(getattr(env, "_tactile_layout_path", None))[0][patch.body]["ctan"]) @ body_rot.T
        ax_w /= np.linalg.norm(ax_w, axis=1, keepdims=True) + 1e-12
        ctan_w /= np.linalg.norm(ctan_w, axis=1, keepdims=True) + 1e-12
        prev = np.zeros(len(patch.pos), dtype=np.float32) if prev_depths is None else prev_depths[pi]
        cur = np.zeros(len(patch.pos), dtype=np.float32)

        for k in range(len(patch.pos)):
            best_depth = 0.0
            for da in offsets_1d:
                for dc in offsets_1d:
                    origin = pos_w[k] + da * ax_w[k] + dc * ctan_w[k]
                    if front_depth > 0 or back_depth > 0:
                        if front_depth > 0:
                            geomid[0] = -1
                            dist = mujoco.mj_ray(
                                m, d, origin, rad_w[k], None, 1, int(patch.bid), geomid
                            )
                            if 0 <= dist <= front_depth:
                                best_depth = max(best_depth, front_depth - float(dist))
                        if back_depth > 0:
                            geomid[0] = -1
                            dist = mujoco.mj_ray(
                                m, d, origin, -rad_w[k], None, 1, int(patch.bid), geomid
                            )
                            if 0 <= dist <= back_depth:
                                best_depth = max(best_depth, back_depth - float(dist))
                    else:
                        geomid[0] = -1
                        dist = mujoco.mj_ray(m, d, origin, rad_w[k], None, 1, int(patch.bid), geomid)
                        if 0 <= dist <= skin:
                            best_depth = max(best_depth, skin - float(dist))
            if best_depth <= 0:
                continue
            d_dot = (best_depth - float(prev[k])) / max(dt, 1e-6)
            force = max(kn * best_depth + kd * d_dot, 0.0)
            i, j = patch.ij[k]
            img[patch.ch, int(i), int(j)] = min(max_value, force / force_scale)
            cur[k] = best_depth
        next_depths.append(cur)

    if max_value > 0:
        img = np.clip(img, 0.0, max_value)
    if blur_sigma > 0:
        img = gaussian_blur_channels(img, blur_sigma)
        if max_value > 0:
            img = np.clip(img, 0.0, max_value)
    return canonicalize(img), next_depths


def object_surface_points(env, body_patterns: tuple[str, ...], max_points: int) -> np.ndarray:
    model = env.sim.model
    data = env.sim.data
    pts = []
    for gid in range(model.ngeom):
        body_name = model.body_id2name(int(model.geom_bodyid[gid])) or ""
        if body_patterns and not any(p in body_name for p in body_patterns):
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        gtype = int(model.geom_type[gid])
        dataid = int(model.geom_dataid[gid])
        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH) and dataid >= 0:
            adr = int(model.mesh_vertadr[dataid])
            num = int(model.mesh_vertnum[dataid])
            v = np.asarray(model.mesh_vert[adr : adr + num], dtype=np.float64)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            sx, sy, sz = np.asarray(model.geom_size[gid], dtype=np.float64)[:3]
            n = max(5, int(round(np.sqrt(max_points if max_points > 0 else 400))))
            xs = np.linspace(-sx, sx, n)
            ys = np.linspace(-sy, sy, n)
            zs = np.linspace(-sz, sz, n)
            faces = []
            yy, zz = np.meshgrid(ys, zs, indexing="ij")
            faces.append(np.c_[np.full(yy.size, -sx), yy.ravel(), zz.ravel()])
            faces.append(np.c_[np.full(yy.size, sx), yy.ravel(), zz.ravel()])
            xx, zz = np.meshgrid(xs, zs, indexing="ij")
            faces.append(np.c_[xx.ravel(), np.full(xx.size, -sy), zz.ravel()])
            faces.append(np.c_[xx.ravel(), np.full(xx.size, sy), zz.ravel()])
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            faces.append(np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, -sz)])
            faces.append(np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, sz)])
            v = np.concatenate(faces, axis=0)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            r = float(model.geom_size[gid, 0])
            n_theta = max(8, int(round(np.sqrt(max_points if max_points > 0 else 400))))
            n_phi = max(16, 2 * n_theta)
            theta = np.linspace(0.0, np.pi, n_theta)
            phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
            tt, pp = np.meshgrid(theta, phi, indexing="ij")
            v = np.c_[
                r * np.sin(tt).ravel() * np.cos(pp).ravel(),
                r * np.sin(tt).ravel() * np.sin(pp).ravel(),
                r * np.cos(tt).ravel(),
            ]
        else:
            continue
        if len(v) == 0:
            continue
        if max_points > 0 and len(v) > max_points:
            idx = np.linspace(0, len(v) - 1, max_points).astype(np.int64)
            v = v[idx]
        R = data.geom_xmat[gid].reshape(3, 3)
        o = data.geom_xpos[gid]
        pts.append(o[None, :] + v @ R.T)
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(pts, axis=0)


def read_tacsl_kdtree_image(
    env,
    patches: list[Patch],
    prev_depths: list[np.ndarray] | None,
    dt: float,
    skin: float,
    kn: float,
    kd: float,
    force_scale: float,
    max_value: float,
    body_patterns: tuple[str, ...],
    max_points_per_geom: int,
    normal_gate: float,
    blur_sigma: float,
    surface_offset: float,
    surface_base_proud: float,
    symmetric_band: bool,
    front_depth: float,
    back_depth: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    pts = object_surface_points(env, body_patterns, max_points_per_geom)
    img = np.zeros((max(p.ch for p in patches) + 1, 32, 32), np.float32)
    next_depths: list[np.ndarray] = []
    if len(pts) == 0:
        return img, [np.zeros(len(p.pos), dtype=np.float32) for p in patches]
    tree = cKDTree(pts)
    data = env.sim.data

    for pi, patch in enumerate(patches):
        body_rot = data.body_xmat[patch.bid].reshape(3, 3)
        body_pos = data.body_xpos[patch.bid]
        pos_w = body_pos[None, :] + patch.pos @ body_rot.T
        rad_w = patch.rad @ body_rot.T
        rad_w /= np.linalg.norm(rad_w, axis=1, keepdims=True) + 1e-12
        pos_w = pos_w - surface_base_proud * rad_w
        pos_w = pos_w + surface_offset * rad_w
        prev = np.zeros(len(patch.pos), dtype=np.float32) if prev_depths is None else prev_depths[pi]
        cur = np.zeros(len(patch.pos), dtype=np.float32)
        dist, idx = tree.query(pos_w, k=1, workers=-1)
        vec = pts[idx] - pos_w
        along = np.sum(vec * rad_w, axis=1)
        if front_depth > 0 or back_depth > 0:
            max_depth = max(front_depth, back_depth)
            side_margin = np.where(along >= 0.0, front_depth - along, back_depth + along)
            valid = (dist <= max_depth) & (side_margin > 0.0)
            depths = np.where(valid, np.minimum(side_margin, max_depth - dist), 0.0)
        elif symmetric_band:
            valid = dist <= skin
            depths = np.where(valid, skin - dist, 0.0)
        else:
            valid = (dist <= skin) & (along > normal_gate)
            depths = np.where(valid, skin - dist, 0.0)
        d_dot = (depths - prev.astype(np.float64)) / max(dt, 1e-6)
        force = np.maximum(kn * depths + kd * d_dot, 0.0)
        vals = np.minimum(max_value, force / force_scale)
        for k in np.nonzero(vals > 0)[0]:
            i, j = patch.ij[k]
            img[patch.ch, int(i), int(j)] = float(vals[k])
        cur[:] = depths.astype(np.float32)
        next_depths.append(cur)

    if max_value > 0:
        img = np.clip(img, 0.0, max_value)
    if blur_sigma > 0:
        img = gaussian_blur_channels(img, blur_sigma)
        if max_value > 0:
            img = np.clip(img, 0.0, max_value)
    return canonicalize(img), next_depths


def draw_frame(
    fig,
    rgb: dict[str, np.ndarray],
    tactile: np.ndarray,
    title: str,
    vmax: float,
) -> np.ndarray:
    fig.clear()
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.15],
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
        ax.set_title(f"ch{ch} peak={float(tactile[ch].max()):.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    return buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]


def metrics(tactile: np.ndarray, state_err: np.ndarray) -> dict[str, float]:
    x = np.asarray(tactile, dtype=np.float32)
    frame_sum = x.sum(axis=(1, 2, 3))
    active = frame_sum > 1e-6
    return {
        "frames": int(len(x)),
        "contact_frame_frac": float(active.mean()),
        "mean_active_pixels_contact_frames": float((x > 1e-4).sum(axis=(1, 2, 3))[active].mean())
        if active.any()
        else 0.0,
        "mean_value": float(x.mean()) if x.size else 0.0,
        "max_value": float(x.max()) if x.size else 0.0,
        "state_err_p50": float(np.percentile(state_err, 50)) if len(state_err) else 0.0,
        "state_err_p95": float(np.percentile(state_err, 95)) if len(state_err) else 0.0,
        "state_err_max": float(np.max(state_err)) if len(state_err) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    parser.add_argument("--demo", default="demo_258")
    parser.add_argument("--out", default=str(ROOT / "outputs/tactile_h5_videos/demo258_online_tacsl_spring_s8_fs25.mp4"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skin", type=float, default=0.003)
    parser.add_argument("--kn", type=float, default=2500.0)
    parser.add_argument("--kd", type=float, default=20.0)
    parser.add_argument("--force-scale", type=float, default=25.0)
    parser.add_argument("--max-value", type=float, default=1.0)
    parser.add_argument("--mode", choices=["single_ray", "multiray", "kdtree"], default="single_ray")
    parser.add_argument("--ray-grid", type=int, default=3)
    parser.add_argument("--ray-radius", type=float, default=0.0010)
    parser.add_argument("--body-patterns", nargs="*", default=["piece", "base"])
    parser.add_argument("--max-points-per-geom", type=int, default=2500)
    parser.add_argument("--normal-gate", type=float, default=0.0)
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    parser.add_argument("--surface-offset", type=float, default=0.0)
    parser.add_argument(
        "--replay-mode",
        choices=["actions", "states"],
        default="actions",
        help=(
            "actions: start from state[0] and step stored actions. "
            "states: force every stored simulator state before rendering."
        ),
    )
    parser.add_argument(
        "--surface-base-proud",
        type=float,
        default=0.0,
        help=(
            "Subtract this much along the tactile normal before applying "
            "--surface-offset. Use tactile_recollect.layout.PROUD to move from "
            "the cached proud taxel points back to the underlying mesh surface."
        ),
    )
    parser.add_argument(
        "--use-layout-proud-base",
        action="store_true",
        help=f"Equivalent to --surface-base-proud {PROUD:g}.",
    )
    parser.add_argument(
        "--symmetric-band",
        action="store_true",
        help="Use distance <= skin only, without one-sided normal gating.",
    )
    parser.add_argument(
        "--front-depth",
        type=float,
        default=0.0,
        help="Asymmetric signed band depth on the positive tactile-normal side.",
    )
    parser.add_argument(
        "--back-depth",
        type=float,
        default=0.0,
        help="Asymmetric signed band depth on the negative tactile-normal side.",
    )
    parser.add_argument("--cameras", nargs="+", default=None)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        env = make_env_from_env_args(
            data.attrs["env_args"],
            has_offscreen_renderer=True,
            use_camera_obs=False,
        )
        g = data[args.demo]
        states = g["states"][()]
        actions = g["actions"][()].astype(np.float32)
        if args.max_frames > 0:
            states = states[: args.max_frames]
            actions = actions[: max(args.max_frames - 1, 0)]
        camera_names = infer_cameras(env, g["obs"], args.cameras)
        reset_to(env, {"model": g.attrs["model_file"], "states": states[0], "ep_meta": g.attrs.get("ep_meta", None)})
        patches = build_patches(env)
        surface_base_proud = PROUD if args.use_layout_proud_base else args.surface_base_proud
        print(
            f"[setup] demo={args.demo} frames={len(states)} cameras={camera_names} "
            f"patches={[(p.ch, p.body, len(p.pos)) for p in patches]}",
            flush=True,
        )

        dt = float(getattr(env, "control_timestep", 1.0 / args.fps))
        state_err = np.zeros(max(len(states) - 1, 0), np.float32)
        tactile = []
        rgb_cache = []
        prev_depths = None

        def read_image():
            if args.mode == "multiray":
                return read_tacsl_multiray_image(
                    env,
                    patches,
                    prev_depths,
                    dt,
                    args.skin,
                    args.kn,
                    args.kd,
                    args.force_scale,
                    args.max_value,
                    args.ray_grid,
                    args.ray_radius,
                    args.surface_offset,
                    surface_base_proud,
                    args.blur_sigma,
                    args.front_depth,
                    args.back_depth,
                )
            if args.mode == "kdtree":
                return read_tacsl_kdtree_image(
                    env,
                    patches,
                    prev_depths,
                    dt,
                    args.skin,
                    args.kn,
                    args.kd,
                    args.force_scale,
                    args.max_value,
                    tuple(args.body_patterns),
                    args.max_points_per_geom,
                    args.normal_gate,
                    args.blur_sigma,
                    args.surface_offset,
                    surface_base_proud,
                    args.symmetric_band,
                    args.front_depth,
                    args.back_depth,
                )
            return read_tacsl_spring_image(
                env,
                patches,
                prev_depths,
                dt,
                args.skin,
                args.kn,
                args.kd,
                args.force_scale,
                args.max_value,
                args.surface_offset,
            )

        if args.replay_mode == "states":
            state_err = np.zeros(max(len(states) - 1, 0), np.float32)
            for t in range(len(states)):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                if hasattr(env, "update_state"):
                    env.update_state()
                elif hasattr(env, "update_sites"):
                    env.update_sites()
                img, prev_depths = read_image()
                tactile.append(img)
                rgb_cache.append(render_cameras(env, camera_names, args.height, args.width))
                if t % 25 == 0 and t > 0:
                    print(f"[frame] {t}/{len(states)} max={float(img.max()):.4f}", flush=True)
        else:
            img, prev_depths = read_image()
            tactile.append(img)
            rgb_cache.append(render_cameras(env, camera_names, args.height, args.width))

            for t in range(1, len(states)):
                env.step(actions[t - 1])
                img, prev_depths = read_image()
                tactile.append(img)
                rgb_cache.append(render_cameras(env, camera_names, args.height, args.width))
                state_err[t - 1] = np.max(np.abs(env.sim.get_state().flatten() - states[t]))
                if t % 25 == 0:
                    print(f"[frame] {t}/{len(states)} max={float(img.max()):.4f}", flush=True)

    tactile_arr = np.stack(tactile, axis=0)
    positive = tactile_arr[tactile_arr > 0]
    vmax = max(float(np.percentile(positive, 99)) if positive.size else 1.0, 1e-3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, macro_block_size=None)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    try:
        for t, rgb in enumerate(rgb_cache):
            frame = draw_frame(
                fig,
                rgb,
                tactile_arr[t],
                (
                    f"{args.demo} TacSL-like {args.mode} normal field "
                    f"skin={args.skin*1000:.1f}mm kn={args.kn:g} kd={args.kd:g} "
                    f"frame {t + 1}/{len(tactile_arr)}"
                ),
                vmax,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)

    metric_path = out.with_suffix(".metrics.json")
    metric_path.write_text(
        json.dumps(
            {
                **metrics(tactile_arr, state_err),
                "skin": args.skin,
                "kn": args.kn,
                "kd": args.kd,
                "force_scale": args.force_scale,
                "mode": args.mode,
                "ray_grid": args.ray_grid,
                "ray_radius": args.ray_radius,
                "body_patterns": args.body_patterns,
                "max_points_per_geom": args.max_points_per_geom,
                "normal_gate": args.normal_gate,
                "blur_sigma": args.blur_sigma,
                "surface_offset": args.surface_offset,
                "surface_base_proud": surface_base_proud,
                "symmetric_band": args.symmetric_band,
                "front_depth": args.front_depth,
                "back_depth": args.back_depth,
                "replay_mode": args.replay_mode,
                "renderer": "tacsl_like_per_taxel_normal_spring_damper",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {out}", flush=True)
    print(f"wrote {metric_path}", flush=True)


if __name__ == "__main__":
    main()
