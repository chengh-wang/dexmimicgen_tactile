#!/usr/bin/env python3
"""Launch TacSL-style virtual tactile action-rollout extraction for all tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S_s8fs25_alltasks"))
    ap.add_argument("--shards-per-task", type=int, default=2)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--start", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    uv = Path("/home/labeng/.local/bin/uv")
    raw_dir = root / "datasets/generated"
    out_root = root / "datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards" / args.run_id
    log_root = root / "tactile_recollect/out/actionrollout_virtual_s8fs25_alltasks_logs" / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "root": str(root),
        "run_id": args.run_id,
        "out_root": str(out_root),
        "log_root": str(log_root),
        "shards_per_task": args.shards_per_task,
        "renderer": {
            "TACTILE_RENDERER": "virtual",
            "TACTILE_VIRTUAL_SIGMA": "8.0",
            "TACTILE_VIRTUAL_FORCE_SCALE": "25.0",
            "TACTILE_VIRTUAL_MAX": "1.0",
            "TACTILE_VIRTUAL_CANONICAL": "1",
            "TACTILE_VIRTUAL_MAX_SURFACE_DIST": "0.015",
        },
        "units": [],
    }

    for task in args.tasks:
        dataset = raw_dir / f"{task}.hdf5"
        if not dataset.exists():
            raise FileNotFoundError(dataset)
        task_out = out_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(args.shards_per_task):
            out = task_out / f"shard{shard_idx:02d}.hdf5"
            log = log_root / f"{task}_shard{shard_idx:02d}.log"
            unit = f"dexmg-vtact-{args.run_id}-{task.replace('_', '-')}-s{shard_idx:02d}"
            cmd = (
                f"cd '{root}' && "
                "export PYTHONPATH='{root}:{root}/robosuite' && "
                "export MUJOCO_GL=egl HDF5_USE_FILE_LOCKING=FALSE && "
                "export TACTILE_RENDERER=virtual "
                "TACTILE_VIRTUAL_SIGMA=8.0 "
                "TACTILE_VIRTUAL_FORCE_SCALE=25.0 "
                "TACTILE_VIRTUAL_MAX=1.0 "
                "TACTILE_VIRTUAL_CANONICAL=1 "
                "TACTILE_VIRTUAL_MAX_SURFACE_DIST=0.015 && "
                f"'{uv}' run --no-sync python -m tactile_recollect.extract_action_rollout "
                f"--dataset '{dataset}' "
                f"--out '{out}' "
                "--resume "
                f"--shard-index {shard_idx} "
                f"--num-shards {args.shards_per_task} "
                f"> '{log}' 2>&1"
            ).format(root=root)
            manifest["units"].append(
                {
                    "task": task,
                    "shard_index": shard_idx,
                    "num_shards": args.shards_per_task,
                    "unit": unit,
                    "dataset": str(dataset),
                    "out": str(out),
                    "log": str(log),
                    "cmd": cmd,
                }
            )
            if args.start:
                subprocess.run(
                    [
                        "systemd-run",
                        "--user",
                        "--unit",
                        unit,
                        "--same-dir",
                        "--collect",
                        "/bin/bash",
                        "-lc",
                        cmd,
                    ],
                    check=True,
                )

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest={manifest_path}")
    print(f"out_root={out_root}")
    print(f"log_root={log_root}")
    print(f"units={len(manifest['units'])}")
    if not args.start:
        print("dry run only; pass --start to launch services")


if __name__ == "__main__":
    main()
