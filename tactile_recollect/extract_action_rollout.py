"""
Extract tactile by online action replay instead of forced-state replay.

For each demo, this starts from the recorded initial MuJoCo state, then steps the
recorded actions in the tactile-injected environment and reads robot0_tactile
from live contacts. This makes tactile values match the online sensor dynamics
seen during policy rollout, unlike extract.py which overwrites the simulator
state every frame and then calls sim.forward().
"""

from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np

from .env import make_env_from_env_args, read_tactile_image, tactile_shape


_COMP = dict(compression="gzip", compression_opts=4)


def _copy_through(src, dst, skip=()):
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


def extract_demo_action_rollout(env, g):
    states = g["states"][()]
    actions = g["actions"][()].astype(np.float32)
    model = g.attrs["model_file"]
    ep_meta = g.attrs.get("ep_meta", None)
    t = states.shape[0]
    out = np.zeros((t,) + tactile_shape(), np.float32)
    state_err = np.zeros(max(t - 1, 0), np.float32)

    _reset_to(env, {"model": model, "states": states[0], "ep_meta": ep_meta})
    out[0] = read_tactile_image(env)
    for i in range(1, t):
        env.step(actions[i - 1])
        out[i] = read_tactile_image(env)
        state_err[i - 1] = np.max(
            np.abs(env.sim.get_state().flatten() - states[i])
        )
    return out, state_err


def _select_demos(all_demos, n=0, demos=None, shard_index=None, num_shards=None):
    if demos:
        selected = [d for d in all_demos if d in set(demos)]
    elif n:
        selected = all_demos[:n]
    else:
        selected = list(all_demos)
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("shard_index and num_shards must be set together")
        selected = [d for i, d in enumerate(selected) if i % num_shards == shard_index]
    return selected


def run(
    dataset,
    out,
    n=0,
    demos=None,
    resume=False,
    shard_index=None,
    num_shards=None,
):
    fin = h5py.File(dataset, "r")
    data = fin["data"]
    env = make_env_from_env_args(data.attrs["env_args"])

    all_demos = sorted(data.keys(), key=lambda d: int(d.split("_")[1]))
    selected = _select_demos(
        all_demos,
        n=n,
        demos=demos,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    existing = set()
    mode = "w"
    if resume and os.path.exists(out):
        mode = "a"
        with h5py.File(out, "r") as chk:
            if "data" in chk:
                existing = {
                    d for d in chk["data"].keys()
                    if "obs/robot0_tactile" in chk["data"][d]
                }
        print(f"resume: {len(existing)} demos already done, skipping them")

    fout = h5py.File(out, mode)
    od = fout.require_group("data")
    for ak, av in data.attrs.items():
        od.attrs[ak] = av
    od.attrs["tactile_extraction_mode"] = "action_rollout"
    od.attrs["tactile_extraction_note"] = (
        "reset to recorded state[0], then env.step(recorded actions) and read live contacts"
    )
    if shard_index is not None:
        od.attrs["shard_index"] = int(shard_index)
        od.attrs["num_shards"] = int(num_shards)

    t0 = time.time()
    done = 0
    contact_fracs = []
    peaks = []
    drift_p95 = []
    for di, demo_key in enumerate(selected):
        if demo_key in existing:
            continue
        g = data[demo_key]
        tac, state_err = extract_demo_action_rollout(env, g)
        if demo_key in od:
            del od[demo_key]
        og = od.create_group(demo_key)
        _copy_through(g, og)
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

        og.attrs["tactile_extraction_mode"] = "action_rollout"
        if len(state_err):
            og.attrs["action_rollout_state_err_p50"] = float(np.percentile(state_err, 50))
            og.attrs["action_rollout_state_err_p95"] = float(np.percentile(state_err, 95))
            og.attrs["action_rollout_state_err_max"] = float(np.max(state_err))
            drift_p95.append(float(np.percentile(state_err, 95)))
        fout.flush()

        nz = float((tac > 1e-6).any(axis=(1, 2, 3)).mean())
        peak = float(tac.max())
        contact_fracs.append(nz)
        peaks.append(peak)
        done += 1
        print(
            f"[{di + 1}/{len(selected)}] {demo_key}: T={len(tac)} "
            f"contact={nz * 100:4.1f}% peak={peak:7.3f} "
            f"state_err_p95={(drift_p95[-1] if drift_p95 else 0.0):.4g} "
            f"({(time.time() - t0) / done:.1f}s/demo)",
            flush=True,
        )

    od.attrs["total"] = int(sum(int(data[d].attrs["num_samples"]) for d in selected))
    if contact_fracs:
        od.attrs["action_rollout_contact_frac_mean"] = float(np.mean(contact_fracs))
        od.attrs["action_rollout_peak_max"] = float(np.max(peaks))
        od.attrs["action_rollout_state_err_p95_mean"] = float(np.mean(drift_p95))
    fout.close()
    fin.close()
    print(
        f"\nwrote {out} ({done} new demos, "
        f"mean contact={np.mean(contact_fracs) * 100 if contact_fracs else 0.0:.1f}%, "
        f"max peak={np.max(peaks) if peaks else 0.0:.3f})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all demos")
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    args = ap.parse_args()
    run(
        args.dataset,
        args.out,
        n=args.n,
        demos=args.demos,
        resume=args.resume,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
