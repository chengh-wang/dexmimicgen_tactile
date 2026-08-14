#!/usr/bin/env python3
"""Wait for sharded tactile extraction, validate coverage, then train VAE."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import h5py


def demo_key(name: str) -> int:
    return int(name.split("_")[1])


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tactile_demos(path: str) -> set[str]:
    demos: set[str] = set()
    if not os.path.exists(path):
        return demos
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return demos
        for demo in f["data"].keys():
            obs = f["data"][demo].get("obs")
            if obs is not None and "robot0_tactile" in obs:
                shape = obs["robot0_tactile"].shape
                if len(shape) == 4 and shape[-2:] == (32, 32):
                    demos.add(demo)
    return demos


def validate_coverage(manifest: dict) -> list[str]:
    h5_paths: list[str] = []
    for task, info in manifest["tasks"].items():
        raw = info["raw"]
        with h5py.File(raw, "r") as f:
            expected = set(f["data"].keys())
        covered = tactile_demos(info["snapshot"])
        h5_paths.append(info["snapshot"])
        for shard in info["shards"]:
            out = shard["out"]
            covered |= tactile_demos(out)
            h5_paths.append(out)
        missing = sorted(expected - covered, key=demo_key)
        if missing:
            raise RuntimeError(f"{task}: missing {len(missing)} demos, first={missing[:10]}")
        extra = sorted(covered - expected, key=demo_key)
        if extra:
            raise RuntimeError(f"{task}: extra demos {extra[:10]}")
        print(f"{task}: coverage ok {len(covered)}/{len(expected)}", flush=True)
    return [p for p in h5_paths if os.path.exists(p)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    pids = [int(p) for p in manifest["pids"]]
    print(f"watching {len(pids)} shard workers", flush=True)
    while True:
        live = [p for p in pids if pid_alive(p)]
        if not live:
            break
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] shard workers still running: {len(live)}", flush=True)
        time.sleep(60)

    print("all shard workers finished; validating coverage", flush=True)
    h5_paths = validate_coverage(manifest)

    root = Path("/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile")
    wm = Path("/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising")
    uv = Path("/home/labeng/.local/bin/uv")
    out = wm / "runs" / f"dexmg_shared_tactile_vae_sharded_{manifest['run_id']}"
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
        "50000",
        "--save-every",
        "2000",
        "--batch-size",
        "1024",
        "--num-workers",
        "8",
        "--eval-batches",
        "64",
        "--tactile-scale",
        "300.0",
        "--clip-input",
    ]
    print("starting VAE:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    print(f"VAE complete: {out}", flush=True)


if __name__ == "__main__":
    main()
