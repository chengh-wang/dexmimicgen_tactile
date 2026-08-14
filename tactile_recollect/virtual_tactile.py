"""TacSL-style virtual tactile readout for MuJoCo grippers / hands.

This module treats the 32x32 tactile image as a sensor readout, not as injected
collision geoms. It projects MuJoCo contacts on real hand collision geoms into
the tactile layout's local surface coordinates, then splats each contact force
into a soft pressure patch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from .layout import load_layout


@dataclass(frozen=True)
class VirtualTactileConfig:
    sigma: float = 5.0
    force_scale: float = 20.0
    max_value: float = 1.0
    canonicalize: bool = True
    allowed_bodies: tuple[str, ...] = ()
    max_surface_dist: float = 0.015


def config_from_env() -> VirtualTactileConfig:
    allowed = os.environ.get("TACTILE_VIRTUAL_ALLOWED_BODIES", "")
    return VirtualTactileConfig(
        sigma=float(os.environ.get("TACTILE_VIRTUAL_SIGMA", "5.0")),
        force_scale=float(os.environ.get("TACTILE_VIRTUAL_FORCE_SCALE", "20.0")),
        max_value=float(os.environ.get("TACTILE_VIRTUAL_MAX", "1.0")),
        canonicalize=os.environ.get("TACTILE_VIRTUAL_CANONICAL", "1").lower()
        not in ("0", "false", "no", "off"),
        allowed_bodies=tuple(x.strip() for x in allowed.split(",") if x.strip()),
        max_surface_dist=float(os.environ.get("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015")),
    )


def build_pad_projectors(env):
    layout, n = load_layout(getattr(env, "_tactile_layout_path", None))
    if n != 32:
        raise RuntimeError(f"virtual tactile expects 32x32 layout, got {n}x{n}")

    out = []
    for ch, body in enumerate(sorted(layout.keys())):
        try:
            bid = int(env.sim.model.body_name2id(body))
        except Exception:
            continue

        geom_ids = [
            int(gid)
            for gid in range(env.sim.model.ngeom)
            if int(env.sim.model.geom_bodyid[gid]) == bid
            and not (env.sim.model.geom_id2name(gid) or "").startswith("tac_")
        ]
        if not geom_ids:
            continue

        d = layout[body]
        pos = np.asarray(d["pos"], dtype=np.float64)
        ij = np.asarray(d["ij"], dtype=np.int32)
        if len(pos) == 0:
            continue
        out.append(
            {
                "ch": ch,
                "bid": bid,
                "body": body,
                "gids": tuple(geom_ids),
                "pos": pos,
                "ij": ij,
            }
        )
    return out


def _projectors(env):
    cached = getattr(env, "_virtual_tactile_projectors", None)
    model_id = id(env.sim.model._model)
    if (
        cached is None
        or not isinstance(cached, dict)
        or cached.get("model_id") != model_id
    ):
        cached = {
            "model_id": model_id,
            "projectors": build_pad_projectors(env),
        }
        env._virtual_tactile_projectors = cached
    return cached["projectors"]


def _canonicalize(img: np.ndarray) -> np.ndarray:
    y = img.copy()
    # Opposing Panda finger pads have opposite local frame conventions. Flipping
    # them makes left/right gripper contacts visually comparable channel-wise.
    if y.shape[0] == 4:
        y[1] = y[1, ::-1, ::-1]
        y[3] = y[3, ::-1, ::-1]
    return y


def _add_gaussian(img: np.ndarray, ch: int, i: float, j: float, value: float, sigma: float):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    ic = int(math.floor(i))
    jc = int(math.floor(j))
    i0, i1 = max(0, ic - radius), min(31, ic + radius)
    j0, j1 = max(0, jc - radius), min(31, jc + radius)
    ii = np.arange(i0, i1 + 1, dtype=np.float32)[:, None]
    jj = np.arange(j0, j1 + 1, dtype=np.float32)[None, :]
    w = np.exp(-0.5 * (((ii - i) / sigma) ** 2 + ((jj - j) / sigma) ** 2))
    img[ch, i0 : i1 + 1, j0 : j1 + 1] += float(value) * w


def read_virtual_tactile_image(env, config: VirtualTactileConfig | None = None) -> np.ndarray:
    config = config or config_from_env()
    projectors = _projectors(env)
    channels = max((int(p["ch"]) for p in projectors), default=-1) + 1
    img = np.zeros((channels, 32, 32), np.float32)
    if not projectors:
        return img

    geom_to_projector = {gid: p for p in projectors for gid in p["gids"]}
    tactile_bodies = {int(p["bid"]) for p in projectors}
    model = env.sim.model
    m = model._model
    d = env.sim.data._data
    f6 = np.zeros(6, dtype=np.float64)

    for ci in range(d.ncon):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in geom_to_projector:
            proj, other = geom_to_projector[g1], g2
        elif g2 in geom_to_projector:
            proj, other = geom_to_projector[g2], g1
        else:
            continue

        other_bid = int(model.geom_bodyid[other])
        if other_bid in tactile_bodies:
            continue
        other_body = model.body_id2name(other_bid)
        if config.allowed_bodies and other_body not in config.allowed_bodies:
            continue

        mujoco.mj_contactForce(m, d, ci, f6)
        force = abs(float(f6[0]))
        if force <= 0:
            continue

        bid = proj["bid"]
        body_rot = env.sim.data.body_xmat[bid].reshape(3, 3)
        body_pos = env.sim.data.body_xpos[bid]
        p_local = body_rot.T @ (np.asarray(c.pos, dtype=np.float64) - body_pos)
        dist2 = np.sum((proj["pos"] - p_local) ** 2, axis=1)
        nearest = int(np.argmin(dist2))
        if config.max_surface_dist > 0 and math.sqrt(float(dist2[nearest])) > config.max_surface_dist:
            continue
        i = float(proj["ij"][nearest, 0])
        j = float(proj["ij"][nearest, 1])
        _add_gaussian(img, int(proj["ch"]), i, j, force / config.force_scale, config.sigma)

    if config.max_value > 0:
        img = np.clip(img, 0.0, config.max_value)
    if config.canonicalize:
        img = _canonicalize(img)
    return img.astype(np.float32, copy=False)
