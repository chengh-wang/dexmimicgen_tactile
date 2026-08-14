#!/usr/bin/env python3
"""Wait for virtual s8/fs25 tactile shards, validate coverage, then train VAE."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np


TASKS = [
    "two_arm_box_cleanup",
    "two_arm_can_sort_random",
    "two_arm_coffee",
    "two_arm_drawer_cleanup",
    "two_arm_lift_tray",
    "two_arm_pouring",
    "two_arm_threading",
    "two_arm_three_piece_assembly",
    "two_arm_transport",
]


def demo_key(name: str) -> int:
    return int(name.split("_")[1])


def count_tactile_demos(path: Path) -> set[str]:
    if not path.exists():
        return set()
    demos: set[str] = set()
    with h5py.File(path, "r") as f:
        root = f.get("data")
        if root is None:
            return demos
        for demo in root.keys():
            obs = root[demo].get("obs")
            if obs is None or "robot0_tactile" not in obs:
                continue
            shape = obs["robot0_tactile"].shape
            if len(shape) == 4 and shape[-2:] == (32, 32):
                demos.add(demo)
    return demos


def extraction_running() -> bool:
    proc = subprocess.run(
        ["ps", "-eo", "cmd"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    for line in proc.stdout.splitlines():
        if "python" in line and "-m tactile_recollect.extract_action_rollout" in line:
            return True
    return False


def validate_and_collect(raw_dir: Path, shard_root: Path) -> list[str]:
    h5_paths: list[str] = []
    total_expected = 0
    total_covered = 0
    for task in TASKS:
        raw = raw_dir / f"{task}.hdf5"
        with h5py.File(raw, "r") as f:
            expected = set(f["data"].keys())
        covered: set[str] = set()
        shard_paths = sorted((shard_root / task).glob("shard*.hdf5"))
        for shard in shard_paths:
            covered |= count_tactile_demos(shard)
            h5_paths.append(str(shard))
        missing = sorted(expected - covered, key=demo_key)
        extra = sorted(covered - expected, key=demo_key)
        if missing:
            raise RuntimeError(f"{task}: missing {len(missing)} demos, first={missing[:10]}")
        if extra:
            raise RuntimeError(f"{task}: extra {len(extra)} demos, first={extra[:10]}")
        total_expected += len(expected)
        total_covered += len(covered)
        print(f"{task}: coverage ok {len(covered)}/{len(expected)} shards={len(shard_paths)}", flush=True)
    print(f"coverage ok total {total_covered}/{total_expected}", flush=True)
    return h5_paths


def sample_tactile_stats(h5_paths: list[str], max_demos_per_file: int = 3) -> dict[str, float]:
    mins: list[float] = []
    maxs: list[float] = []
    means: list[float] = []
    active_fracs: list[float] = []
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=demo_key)[:max_demos_per_file]
            for demo in demos:
                x = f["data"][demo]["obs"]["robot0_tactile"][:].astype(np.float32)
                mins.append(float(x.min()))
                maxs.append(float(x.max()))
                means.append(float(x.mean()))
                active_fracs.append(float((x > 1e-4).mean()))
    return {
        "sample_min": float(np.min(mins)),
        "sample_max": float(np.max(maxs)),
        "sample_mean_mean": float(np.mean(means)),
        "sample_active_frac_mean": float(np.mean(active_fracs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-root",
        default="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/"
        "generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/"
        "20260727_0828_s8fs25_alltasks",
    )
    parser.add_argument(
        "--raw-dir",
        default="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated",
    )
    parser.add_argument(
        "--out",
        default="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/runs/"
        "dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727",
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    root = Path("/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile")
    wm = Path("/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising")
    uv = Path("/home/labeng/.local/bin/uv")
    shard_root = Path(args.shard_root)
    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    log_dir = root / "tactile_recollect/out/vae_virtual_s8fs25_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    while extraction_running():
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] extraction still running", flush=True)
        time.sleep(args.poll_seconds)

    print("extraction finished; validating tactile shard coverage", flush=True)
    h5_paths = validate_and_collect(raw_dir, shard_root)
    stats = sample_tactile_stats(h5_paths)
    print("sample tactile stats:", json.dumps(stats, sort_keys=True), flush=True)
    if stats["sample_max"] > 1.001 or stats["sample_min"] < -1e-6:
        raise RuntimeError(f"virtual tactile should be in [0,1], got {stats}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root / 'robosuite'}"
    cmd = [
        str(uv),
        "run",
        "--no-sync",
        "--project",
        str(wm),
        "python",
        str(root / "tactile_recollect/train_shared_tactile_vae.py"),
        "--h5",
        *h5_paths,
        "--out",
        str(out),
        "--steps",
        str(args.steps),
        "--save-every",
        str(args.save_every),
        "--batch-size",
        str(args.batch_size),
        "--latent-dim",
        str(args.latent_dim),
        "--num-workers",
        str(args.num_workers),
        "--eval-batches",
        "64",
        "--tactile-scale",
        "1.0",
        "--clip-input",
    ]
    (out.parent / f"{out.name}.cmd.txt").write_text(" ".join(cmd) + "\n")
    print("starting VAE:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    print(f"VAE complete: {out}", flush=True)


if __name__ == "__main__":
    main()
