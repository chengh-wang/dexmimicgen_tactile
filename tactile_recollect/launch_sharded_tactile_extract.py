#!/usr/bin/env python3
"""Launch sharded tactile extraction for remaining DexMimicGen demos.

This is intentionally separate from extract.py because HDF5 does not support
multiple writers to the same output file. Each shard writes its own HDF5; a
watcher validates coverage and starts VAE training over snapshot + shard files.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import h5py


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


def copy_tactile_snapshot(raw_path: Path, src_path: Path, dst_path: Path) -> set[str]:
    """Copy completed tactile observations into a tactile-only snapshot HDF5."""
    done: set[str] = set()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(raw_path, "r") as fr, h5py.File(src_path, "r") as fs, h5py.File(tmp_path, "w") as fd:
        rd = fr["data"]
        sd = fs["data"] if "data" in fs else None
        dd = fd.require_group("data")
        for ak, av in rd.attrs.items():
            dd.attrs[ak] = av
        if sd is None:
            tmp_path.replace(dst_path)
            return done
        for demo in sorted(sd.keys(), key=demo_key):
            if demo not in rd:
                continue
            try:
                sg = sd[demo]
                if "obs" not in sg or "robot0_tactile" not in sg["obs"]:
                    continue
                ds = sg["obs"]["robot0_tactile"]
                if len(ds.shape) != 4 or ds.shape[-2:] != (32, 32):
                    continue
                arr = ds[()]
            except (KeyError, OSError):
                # The old single-writer extractor may be creating this demo
                # concurrently. Skip it; the shard pass will cover it.
                continue
            dg = dd.create_group(demo)
            for ak, av in rd[demo].attrs.items():
                dg.attrs[ak] = av
            obs = dg.create_group("obs")
            obs.create_dataset("robot0_tactile", data=arr, compression="gzip", compression_opts=4)
            done.add(demo)
    tmp_path.replace(dst_path)
    return done


def terminate_old_extractors(root: Path) -> None:
    """Stop old monolithic extractors after snapshotting their completed demos."""
    try:
        out = subprocess.check_output(["pgrep", "-af", "from tactile_recollect.extract import run"], text=True)
    except subprocess.CalledProcessError:
        out = ""
    my_pid = os.getpid()
    old_pids: list[int] = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        pid = int(parts[0])
        cmd = parts[1] if len(parts) > 1 else ""
        if pid == my_pid:
            continue
        if "generated_tactile_proud2mm_shards" in cmd:
            continue
        if str(root / "datasets/generated_tactile_proud2mm") in cmd:
            old_pids.append(pid)
    for pid in old_pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in old_pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        out = subprocess.check_output(["pgrep", "-af", "watch_extract_then_train_vae"], text=True)
    except subprocess.CalledProcessError:
        out = ""
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if parts:
            try:
                os.kill(int(parts[0]), signal.SIGTERM)
            except ProcessLookupError:
                pass


def chunks(items: list[str], n: int) -> list[list[str]]:
    ret = [[] for _ in range(n)]
    for i, item in enumerate(items):
        ret[i % n].append(item)
    return [x for x in ret if x]


def main() -> None:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile")
    ap.add_argument("--shards-per-task", type=int, default=3)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Do not reuse partially extracted tactile files; shard every raw demo.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    uv = Path("/home/labeng/.local/bin/uv")
    dex_py = root / ".venv/bin/python"
    raw_dir = root / "datasets/generated"
    partial_dir = root / "datasets/generated_tactile_proud2mm"
    snapshot_dir = root / "datasets/generated_tactile_proud2mm_snapshot"
    shard_dir = root / "datasets/generated_tactile_proud2mm_shards" / args.run_id
    log_dir = root / "tactile_recollect/out/extract_shard_logs" / args.run_id
    list_dir = root / "tactile_recollect/out/extract_shard_lists" / args.run_id
    for d in (snapshot_dir, shard_dir, log_dir, list_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root / 'robosuite'}"
    env["MUJOCO_GL"] = "egl"
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"

    manifest: dict[str, object] = {
        "run_id": args.run_id,
        "snapshot_dir": str(snapshot_dir),
        "shard_dir": str(shard_dir),
        "log_dir": str(log_dir),
        "tasks": {},
        "pids": [],
    }

    if args.skip_snapshot:
        print("stopping old monolithic extractors before full sharded run", flush=True)
        terminate_old_extractors(root)
        print("skipping snapshot; all raw demos will be sharded", flush=True)
    else:
        print(f"snapshotting completed tactile into {snapshot_dir}", flush=True)
    for task in TASKS:
        raw = raw_dir / f"{task}.hdf5"
        partial = partial_dir / f"{task}.hdf5"
        snap = snapshot_dir / f"{task}.hdf5"
        with h5py.File(raw, "r") as fr:
            all_demos = sorted(fr["data"].keys(), key=demo_key)
        done = set() if args.skip_snapshot else copy_tactile_snapshot(raw, partial, snap) if partial.exists() else set()
        missing = [d for d in all_demos if d not in done]
        task_info = {
            "raw": str(raw),
            "snapshot": str(snap),
            "done_at_snapshot": len(done),
            "total": len(all_demos),
            "missing": len(missing),
            "shards": [],
        }
        for si, demos in enumerate(chunks(missing, args.shards_per_task)):
            demo_list = list_dir / f"{task}_shard{si:02d}.txt"
            demo_list.write_text("\n".join(demos) + "\n")
            shard_out = shard_dir / task / f"shard{si:02d}.hdf5"
            shard_out.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{task}_shard{si:02d}.log"
            code = (
                "import sys; import dexmimicgen; "
                "from tactile_recollect.extract import run; "
                "demos=[x.strip() for x in open(sys.argv[3]) if x.strip()]; "
                "run(sys.argv[1], sys.argv[2], demos=demos, resume=True)"
            )
            log_f = open(log_path, "ab", buffering=0)
            cmd = [
                str(uv),
                "run",
                "--no-sync",
                "--python",
                str(dex_py),
                "python",
                "-c",
                code,
                str(raw),
                str(shard_out),
                str(demo_list),
            ]
            p = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            manifest["pids"].append(p.pid)  # type: ignore[index]
            task_info["shards"].append(  # type: ignore[index]
                {
                    "index": si,
                    "demos": len(demos),
                    "demo_list": str(demo_list),
                    "out": str(shard_out),
                    "log": str(log_path),
                    "pid": p.pid,
                }
            )
            print(f"launched {task} shard{si:02d}: {len(demos)} demos pid={p.pid}", flush=True)
        manifest["tasks"][task] = task_info  # type: ignore[index]

    manifest_path = root / "tactile_recollect/out" / f"sharded_extract_manifest_{args.run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    terminate_old_extractors(root)
    print(f"manifest: {manifest_path}", flush=True)

    watcher_log = log_dir / "watch_shards_then_train_vae.log"
    watcher_cmd = [
        str(uv),
        "run",
        "--no-sync",
        "--python",
        str(dex_py),
        "python",
        str(root / "tactile_recollect/watch_shards_then_train_vae.py"),
        "--manifest",
        str(manifest_path),
    ]
    with open(watcher_log, "ab", buffering=0) as log_f:
        watcher = subprocess.Popen(
            watcher_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    print(f"watcher pid={watcher.pid} log={watcher_log}", flush=True)


if __name__ == "__main__":
    main()
