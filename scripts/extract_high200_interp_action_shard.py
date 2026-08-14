#!/usr/bin/env python3
"""Shard/batch extractor for interpolated-action 200 Hz DexMimicGen demos."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

from extract_high200_interp_action_demo import (
    COMP,
    DEXMG_ROOT,
    copy_attrs,
    copy_group,
    make_env_from_env_args,
    read_tactile_image,
    reset_to,
    robot_joint_pos,
    step_with_interp_high200,
)


def demo_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def select_demos(all_demos: list[str], args: argparse.Namespace) -> list[str]:
    selected = list(all_demos)
    if args.demos:
        wanted = set(args.demos)
        selected = [d for d in selected if d in wanted]
    if args.n:
        selected = selected[: args.n]
    if args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise ValueError("--shard-index and --num-shards must be set together")
        selected = [d for i, d in enumerate(selected) if i % args.num_shards == args.shard_index]
    return selected


def write_file_attrs(out: h5py.File, dataset: str) -> None:
    out.attrs["format"] = "dexmg_demo_low20_obs_high200_interp_action_shard"
    out.attrs["source_dataset"] = str(Path(dataset).resolve())
    out.attrs["control_hz_original"] = 20
    out.attrs["tactile_hz"] = 200
    out.attrs["joint_hz"] = 200
    out.attrs["action_hz"] = 200
    out.attrs["action_semantics"] = (
        "true dynamics replay: controller goal updated with linspace(action_t, action_t+1, 10)"
    )
    out.attrs["tactile_renderer"] = os.environ.get("TACTILE_RENDERER", "virtual")
    out.attrs["tactile_sigma"] = float(os.environ.get("TACTILE_VIRTUAL_SIGMA", "8.0"))
    out.attrs["tactile_force_scale"] = float(os.environ.get("TACTILE_VIRTUAL_FORCE_SCALE", "25.0"))
    out.attrs["tactile_max"] = float(os.environ.get("TACTILE_VIRTUAL_MAX", "1.0"))
    out.attrs["tactile_canonical"] = int(os.environ.get("TACTILE_VIRTUAL_CANONICAL", "1"))
    out.attrs["tactile_max_surface_dist"] = float(os.environ.get("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015"))


def write_demo(out_data: h5py.Group, env, src: h5py.Group, demo_key: str, max_low_steps: int) -> dict[str, float]:
    states = src["states"][:]
    actions = src["actions"][:].astype(np.float32)
    n_low = min(states.shape[0] - 1, actions.shape[0] - 1)
    if max_low_steps > 0:
        n_low = min(n_low, max_low_steps)

    reset_to(
        env,
        {
            "model": src.attrs["model_file"],
            "states": states[0],
            "ep_meta": src.attrs.get("ep_meta", None),
        },
    )
    tactile_shape = read_tactile_image(env).shape
    joint_dim = robot_joint_pos(env).shape[0]
    action_dim = actions.shape[-1]

    tactile = np.zeros((n_low, 10) + tactile_shape, dtype=np.float32)
    joint = np.zeros((n_low, 10, joint_dim), dtype=np.float32)
    high_action = np.zeros((n_low, 10, action_dim), dtype=np.float32)
    target_time_ms = np.zeros((n_low, 10), dtype=np.float32)
    actual_time_ms = np.zeros((n_low, 10), dtype=np.float32)
    sim_step_index = np.zeros((n_low, 10), dtype=np.int64)
    state_err = np.zeros((n_low,), dtype=np.float32)

    for i in range(n_low):
        samples = step_with_interp_high200(env, actions[i], actions[i + 1], i)
        for j, sample in enumerate(samples):
            tactile[i, j] = sample["tactile"]
            joint[i, j] = sample["robot_joint_pos"]
            high_action[i, j] = sample["action"]
            target_time_ms[i, j] = sample["target_time_ms"]
            actual_time_ms[i, j] = sample["time_ms"]
            sim_step_index[i, j] = sample["sim_step_index"]
        state_err[i] = float(np.max(np.abs(env.sim.get_state().flatten() - states[i + 1])))

    if demo_key in out_data:
        del out_data[demo_key]
    og = out_data.create_group(demo_key)
    copy_attrs(src, og)
    og.attrs["num_low_intervals"] = n_low
    og.attrs["state_err_p50"] = float(np.percentile(state_err, 50)) if len(state_err) else 0.0
    og.attrs["state_err_p95"] = float(np.percentile(state_err, 95)) if len(state_err) else 0.0
    og.attrs["state_err_max"] = float(np.max(state_err)) if len(state_err) else 0.0

    low = og.create_group("low20")
    copy_group(src["obs"], low.create_group("obs"))
    low.create_dataset("actions", data=actions[:n_low], **COMP)
    low.create_dataset("states", data=states[: n_low + 1], **COMP)
    high = og.create_group("high200")
    high.create_dataset("tactile", data=tactile, **COMP)
    high.create_dataset("robot_joint_pos", data=joint, **COMP)
    high.create_dataset("action", data=high_action, **COMP)
    high.create_dataset("target_time_ms", data=target_time_ms, **COMP)
    high.create_dataset("time_ms", data=actual_time_ms, **COMP)
    high.create_dataset("sim_step_index", data=sim_step_index, **COMP)
    high.create_dataset("state_err_after_low_step", data=state_err, **COMP)

    return {
        "n_low": float(n_low),
        "tactile_max": float(tactile.max()) if tactile.size else 0.0,
        "state_err_p95": float(og.attrs["state_err_p95"]),
        "state_err_max": float(og.attrs["state_err_max"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(DEXMG_ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    p.add_argument("--out", required=True)
    p.add_argument("--demos", nargs="*", default=None)
    p.add_argument("--n", type=int, default=0, help="0 means all selected demos")
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--max-low-steps", type=int, default=0, help="0 means full demo")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.resume:
        raise FileExistsError(out_path)

    t0 = time.time()
    mode = "a" if args.resume else "w"
    with h5py.File(args.dataset, "r") as fin, h5py.File(out_path, mode) as fout:
        in_data = fin["data"]
        all_demos = sorted([d for d in in_data.keys() if d.startswith("demo_")], key=demo_sort_key)
        selected = select_demos(all_demos, args)
        write_file_attrs(fout, args.dataset)
        out_data = fout.require_group("data")
        copy_attrs(in_data, out_data)
        if args.shard_index is not None:
            out_data.attrs["shard_index"] = int(args.shard_index)
            out_data.attrs["num_shards"] = int(args.num_shards)
        out_data.attrs["selected_demo_count"] = int(len(selected))

        env = make_env_from_env_args(in_data.attrs["env_args"])
        done = 0
        skipped = 0
        for local_i, demo_key in enumerate(selected):
            if args.resume and demo_key in out_data and "high200/tactile" in out_data[demo_key]:
                skipped += 1
                print(f"[skip] {demo_key}", flush=True)
                continue
            dt0 = time.time()
            stats = write_demo(out_data, env, in_data[demo_key], demo_key, args.max_low_steps)
            fout.flush()
            done += 1
            print(
                f"[{local_i + 1}/{len(selected)}] {demo_key} "
                f"low={int(stats['n_low'])} tactile_max={stats['tactile_max']:.4g} "
                f"state_err_p95={stats['state_err_p95']:.4g} "
                f"state_err_max={stats['state_err_max']:.4g} "
                f"dt={time.time() - dt0:.1f}s",
                flush=True,
            )

    print(
        f"[done] wrote {out_path} new={done} skipped={skipped} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
