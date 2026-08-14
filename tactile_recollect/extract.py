"""
Path B: replay an existing DexMimicGen demo dataset and re-extract observations
with 32x32 piezo tactile injected, WITHOUT rerunning the (unreleased) mimicgen
generation pipeline.

For every demo we reload its stored model (now carrying tactile sites/sensors via
the patched edit_model_xml), force each recorded sim state with
set_state_from_flattened + sim.forward(), and read the tactile image straight from
sim.data.sensordata. Tactile is a pure function of the forced contact state, so it
is reproduced exactly. All other groups (states, actions, action_dict, the
existing obs/* incl. images and proprio) are copied through unchanged; we only add
obs/robot0_tactile (and next_obs/robot0_tactile when next_obs exists).

Usage:
    python -m tactile_recollect.extract \
        --dataset /path/to/two_arm_can_sort_random.hdf5 \
        --out     /path/to/two_arm_can_sort_random_tactile.hdf5 \
        --n 0            # 0 = all demos
"""
import argparse
import json
import os
import time

import h5py
import numpy as np

from .env import make_env_from_env_args, read_tactile_image, tactile_shape

# tactile stored compressed: contact is sparse so gzip shrinks it ~20-100x.
_COMP = dict(compression="gzip", compression_opts=4)


def _copy_through(src, dst, skip=()):
    """Recursively copy groups/datasets/attrs from src to dst, skipping names in
    `skip` at the top level."""
    for k in src.keys():
        if k in skip:
            continue
        if isinstance(src[k], h5py.Group):
            _copy_through(src[k], dst.create_group(k))
        else:
            dst.create_dataset(k, data=src[k][()], **(_COMP if src[k].ndim else {}))
    for ak, av in src.attrs.items():
        dst.attrs[ak] = av


def _reset_to(env, state):
    """Minimal reset_to (mirrors scripts/playback_datasets.reset_to)."""
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


def extract_demo(env, g):
    """Return (T, 10, 32, 32) float32 tactile for one demo group `g`."""
    states = g["states"][()]
    model = g.attrs["model_file"]
    ep_meta = g.attrs.get("ep_meta", None)
    T = states.shape[0]
    shape = tactile_shape()
    out = np.zeros((T,) + shape, np.float32)
    _reset_to(env, {"model": model, "states": states[0], "ep_meta": ep_meta})
    for i in range(T):
        _reset_to(env, {"states": states[i]})
        out[i] = read_tactile_image(env)
    return out


def run(dataset, out, n=0, demos=None, resume=False):
    fin = h5py.File(dataset, "r")
    data = fin["data"]
    env_args = data.attrs["env_args"]
    env = make_env_from_env_args(env_args)

    all_demos = sorted(data.keys(), key=lambda d: int(d.split("_")[1]))
    if demos:
        all_demos = [d for d in all_demos if d in set(demos)]
    elif n:
        all_demos = all_demos[:n]

    # resume: skip demos already present (with tactile) in an existing output.
    existing = set()
    mode = "w"
    if resume and os.path.exists(out):
        mode = "a"
        with h5py.File(out, "r") as chk:
            if "data" in chk:
                existing = {d for d in chk["data"].keys()
                            if "obs/robot0_tactile" in chk["data"][d]}
        print(f"resume: {len(existing)} demos already done, skipping them")

    fout = h5py.File(out, mode)
    od = fout.require_group("data")
    for ak, av in data.attrs.items():
        od.attrs[ak] = av

    t0 = time.time()
    nz_frac_acc = []
    done = 0
    for di, d in enumerate(all_demos):
        if d in existing:
            continue
        g = data[d]
        tac = extract_demo(env, g)
        og = od.create_group(d)
        _copy_through(g, og)                      # states, actions, action_dict, obs/*
        obs = og.require_group("obs")
        if "robot0_tactile" in obs:
            del obs["robot0_tactile"]
        obs.create_dataset("robot0_tactile", data=tac, **_COMP)
        if "next_obs" in og:
            nxt = tac[list(range(1, len(tac))) + [len(tac) - 1]]
            no = og["next_obs"]
            if "robot0_tactile" in no:
                del no["robot0_tactile"]
            no.create_dataset("robot0_tactile", data=nxt, **_COMP)
        fout.flush()                              # persist each demo (crash-safe)
        nz = float((tac > 1e-6).any(axis=(1, 2, 3)).mean())
        nz_frac_acc.append(nz)
        peak = float(tac.max())
        done += 1
        print(f"[{di+1}/{len(all_demos)}] {d}: T={len(tac)} "
              f"frames_with_contact={nz*100:4.1f}% peakN={peak:7.3f} "
              f"({(time.time()-t0)/done:.1f}s/demo)", flush=True)
    od.attrs["total"] = int(sum(int(data[d].attrs["num_samples"]) for d in all_demos))
    fout.close()
    fin.close()
    mc = np.mean(nz_frac_acc) * 100 if nz_frac_acc else 0.0
    print(f"\nwrote {out}  ({done} new demos, "
          f"mean contact frames={mc:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all demos")
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="append to existing --out, skipping already-done demos")
    a = ap.parse_args()
    run(a.dataset, a.out, n=a.n, demos=a.demos, resume=a.resume)
