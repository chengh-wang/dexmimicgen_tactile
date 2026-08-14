"""
Build / patch DexMimicGen Fourier-hand environments so they carry 32x32 piezo
tactile on every fingertip, for both offline obs re-extraction (Path B replay)
and online policy inference.

What this installs (idempotent, monkeypatched onto the shared base
TwoArmDexMGEnv so all 3 Fourier tasks inherit it):

  * edit_model_xml  -> exists-guarded asset-path fix (kills the repo-name path
    bug) + inject per-taxel collision geoms. This runs inside reset_to during
    replay, so every reloaded demo model gets tactile.

  * _setup_observables -> adds robot0_tactile (nbodies, 32, 32) computed from the
    live contact set via mj_contactForce, so an online policy sees tactile in its
    obs dict. The observable name is historical; for Panda two-arm tasks its
    channels cover both grippers, e.g. gripper0 finger pads and gripper1 finger
    pads.

read_tactile_image(env) computes the same (nbodies, 32, 32) image from the
current contacts for the offline extractor / verification (no Observable needed).
"""
import os
import numpy as np
import xml.etree.ElementTree as ET

import dexmimicgen  # noqa: F401  (registers TwoArm* envs with robosuite.make)
import robosuite
from dexmimicgen.environments.two_arm_dexmg_env import TwoArmDexMGEnv
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.observables import Observable, sensor

from .layout import load_layout, layout_path_for
from . import inject as _inject

ENV_ROBOTS = {
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
    "TwoArmCanSortBlue": ["GR1ArmsOnly"],
}

ENV_EMBODIMENT = {
    "TwoArmThreading": "panda",
    "TwoArmThreePieceAssembly": "panda",
    "TwoArmTransport": "panda",
    "TwoArmLiftTray": "inspire",
    "TwoArmBoxCleanup": "inspire",
    "TwoArmDrawerCleanup": "inspire",
    "TwoArmCoffee": "gr1",
    "TwoArmPouring": "gr1",
    "TwoArmCanSortRandom": "gr1",
    "TwoArmCanSortBlue": "gr1",
}

_LAYOUT = None
_N = None
_GRID = None  # (bodies, name->(b,i,j), N)
_LAYOUT_PATH = None


def _env_truthy(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _renderer_kind():
    kind = os.environ.get("TACTILE_RENDERER", "taxel").strip().lower()
    if kind in ("virtual", "tacsl", "tacsl_style", "soft"):
        return "virtual"
    return "taxel"


def _resolve_layout_path(path=None, embodiment=None):
    if path is not None:
        return os.path.abspath(path)
    env_path = os.environ.get("TACTILE_LAYOUT_PATH")
    if env_path:
        return os.path.abspath(env_path)
    if embodiment is None and _LAYOUT_PATH is not None:
        return _LAYOUT_PATH
    embodiment = embodiment or os.environ.get("TACTILE_EMBODIMENT", "gr1")
    return os.path.abspath(layout_path_for(embodiment))


def _ensure_layout(path=None, embodiment=None):
    global _LAYOUT, _N, _GRID, _LAYOUT_PATH
    resolved = _resolve_layout_path(path, embodiment)
    if _LAYOUT is None or _LAYOUT_PATH != resolved:
        _LAYOUT, _N = load_layout(resolved)
        _GRID = _inject.grid_index(_LAYOUT, _N)
        _LAYOUT_PATH = resolved
    return _LAYOUT, _N, _GRID


def tactile_shape(path=None, embodiment=None):
    _, n, (bodies, _, _) = _ensure_layout(path, embodiment)
    return (len(bodies), n, n)


# ---------------- patched edit_model_xml (path fix + tactile inject) -----------
def _fixed_edit_model_xml(self, xml_str):
    # robosuite base handling (skip the buggy DexMG override entirely)
    xml_str = super(TwoArmDexMGEnv, self).edit_model_xml(xml_str)
    layout, n, _ = _ensure_layout()
    root = ET.fromstring(xml_str)
    _inject.fix_asset_paths(root, os.path.split(dexmimicgen.__file__)[0])
    if _renderer_kind() == "taxel":
        visual_scale = float(os.environ.get("TACTILE_VISUAL_SCALE", "1.0"))
        _inject.inject_tactile(
            root,
            layout=layout,
            N=n,
            visible=_env_truthy("TACTILE_VISIBLE"),
            visual_scale=visual_scale,
        )
    if _env_truthy("TACTILE_PREVIEW_PROBE"):
        worldbody = root.find("worldbody")
        if worldbody is not None and worldbody.find(".//body[@name='tactile_probe']") is None:
            probe = ET.SubElement(worldbody, "body", {"name": "tactile_probe", "pos": "0 0 1"})
            ET.SubElement(probe, "freejoint", {"name": "tactile_probe_free"})
            ET.SubElement(probe, "geom", {
                "name": "tactile_probe_geom",
                "type": "sphere",
                "size": "0.003",
                "mass": "0.001",
                "contype": "1",
                "conaffinity": "1",
                "condim": "3",
                "rgba": "1 0.05 0.02 0.85",
            })
    return ET.tostring(root, encoding="utf8").decode("utf8")


# ---------------- contacts -> (nbodies, 32, 32) normal-force image -------------
def _tactile_addr(env):
    """Cache per-env: geom_id -> (body_index, i, j) for every injected taxel geom,
    plus the raw MjModel/MjData handles and image shape."""
    cached = getattr(env, "_tactile_addr_cache", None)
    if cached is not None:
        return cached
    _, n, (bodies, mapping, _) = _ensure_layout()
    model = env.sim.model
    geom2tax = {}
    for name, (b, i, j) in mapping.items():
        try:
            gid = int(model.geom_name2id(name))
        except Exception:
            continue
        geom2tax[gid] = (b, i, j)
    cache = (geom2tax, (len(bodies), n, n))
    env._tactile_addr_cache = cache
    return cache


def read_tactile_image(env):
    """(nbodies, 32, 32) float32 normal-force image from the current contact set.

    For every MuJoCo contact that involves a taxel geom, mj_contactForce gives the
    6-vector (normal, 2 friction, 3 torque) in the contact frame; we scatter the
    normal component into that taxel's (body, i, j) cell. Dense contact -> dense
    image (unlike a touch sensor, which only sees the few mesh-mesh contact points).
    """
    if _renderer_kind() == "virtual":
        from .virtual_tactile import read_virtual_tactile_image

        return read_virtual_tactile_image(env)

    import mujoco
    geom2tax, shape = _tactile_addr(env)
    img = np.zeros(shape, np.float32)
    if not geom2tax:
        return img
    m = env.sim.model._model
    d = env.sim.data._data
    f6 = np.zeros(6)
    ncon = d.ncon
    for ci in range(ncon):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        tg = g1 if g1 in geom2tax else (g2 if g2 in geom2tax else -1)
        if tg < 0:
            continue
        mujoco.mj_contactForce(m, d, ci, f6)
        b, i, j = geom2tax[tg]
        img[b, i, j] += abs(float(f6[0]))
    return img


# ---------------- observable for online inference -----------------------------
def _patched_setup_observables(self):
    observables = super(TwoArmDexMGEnv, self)._setup_observables()

    @sensor(modality="tactile")
    def robot0_tactile(obs_cache):
        return read_tactile_image(self)

    observables["robot0_tactile"] = Observable(
        name="robot0_tactile",
        sensor=robot0_tactile,
        sampling_rate=self.control_freq,
        enabled=True,
        active=True,
    )
    return observables


_PATCHED = False


def install_patches():
    global _PATCHED
    if _PATCHED:
        return
    TwoArmDexMGEnv.edit_model_xml = _fixed_edit_model_xml
    TwoArmDexMGEnv._setup_observables = _patched_setup_observables
    _PATCHED = True


def make_tactile_env(task, control_freq=20, layout_path=None, embodiment=None, **overrides):
    """Construct a DexMG task with the matching tactile layout wired in."""
    assert task in ENV_ROBOTS, f"{task} is not a DexMimicGen tactile task {list(ENV_ROBOTS)}"
    install_patches()
    embodiment = embodiment or ENV_EMBODIMENT[task]
    _ensure_layout(layout_path, embodiment)
    kwargs = dict(
        env_name=task,
        robots=ENV_ROBOTS[task],
        controller_configs=load_composite_controller_config(robot=ENV_ROBOTS[task][0]),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=control_freq,
    )
    kwargs.update(overrides)
    env = robosuite.make(**kwargs)
    env._tactile_layout_path = _LAYOUT_PATH
    env._tactile_embodiment = embodiment
    return env


def _make_drop_unsupported(kwargs):
    """robosuite.make, dropping recorder-only kwargs (e.g. env_lang,
    translucent_robot) that older robosuite constructors reject."""
    kwargs = dict(kwargs)
    while True:
        try:
            return robosuite.make(**kwargs)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" not in msg:
                raise
            bad = msg.split("unexpected keyword argument")[1].strip().strip("'\"")
            if bad not in kwargs:
                raise
            kwargs.pop(bad)


def make_env_from_env_args(env_args, **overrides):
    """Build the exact env a dataset was recorded with (env_args dict or JSON
    str), with tactile patches installed. Cameras are off by default since the
    re-extraction only adds tactile and keeps the stored image obs untouched."""
    import json as _json
    if isinstance(env_args, str):
        env_args = _json.loads(env_args)
    install_patches()
    layout_path = overrides.pop("layout_path", None)
    embodiment = overrides.pop("embodiment", None) or ENV_EMBODIMENT.get(env_args["env_name"], "gr1")
    _ensure_layout(layout_path, embodiment)
    kwargs = dict(env_args.get("env_kwargs", {}))
    kwargs.update(
        env_name=env_args["env_name"],
        has_renderer=False,
        has_offscreen_renderer=overrides.pop("has_offscreen_renderer", False),
        use_camera_obs=overrides.pop("use_camera_obs", False),
        ignore_done=True,
    )
    kwargs.update(overrides)
    env = _make_drop_unsupported(kwargs)
    env._tactile_layout_path = _LAYOUT_PATH
    env._tactile_embodiment = embodiment
    return env


if __name__ == "__main__":
    # smoke test: build TwoArmCoffee, reset, confirm tactile sensors exist
    env = make_tactile_env("TwoArmCoffee")
    env.reset()
    img = read_tactile_image(env)
    adrs, _, shape = _tactile_addr(env)
    print("tactile sensors wired:", len(adrs))
    print("tactile image shape:", img.shape, "sum:", float(img.sum()))
