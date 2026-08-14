#!/usr/bin/env python3
"""Launch DexMG ABC setpoint residual evaluations with bounded parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_done(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    if int(obj.get("episodes_n", 0)) <= 0:
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=8200)
    parser.add_argument("--uv", default="/home/labeng/.local/bin/uv")
    parser.add_argument("--project", default="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--env-dataset-path", required=True)
    parser.add_argument("--task-config", default="dexmg_drawer_cleanup_image_tactile_vae_virtual_s8fs25")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--trust-delta", type=float, default=0.01)
    parser.add_argument("--qp-lambda-u", type=float, default=20.0)
    parser.add_argument("--tactile-channels", type=int, default=12)
    parser.add_argument("--joint-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    params_root = Path(args.params_root)
    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    params = sorted(params_root.glob("*/contrastive_params.npz"))
    if not params:
        raise RuntimeError(f"no params under {params_root}")

    pending = []
    rows = []
    for idx, p in enumerate(params):
        name = p.parent.name
        out_dir = eval_root / name
        out_json = out_dir / "summary.json"
        done = parse_done(out_json)
        if done and int(done.get("episodes_n", 0)) >= args.episodes:
            rows.append({"name": name, "status": "done", **{k: done.get(k) for k in ("success_count", "episodes_n", "success_rate")}})
            continue
        pending.append((idx, name, p, out_dir, out_json))

    running: list[tuple[int, str, subprocess.Popen, object, Path]] = []
    completed = []
    status_path = eval_root / "launcher_status.json"
    table_path = eval_root / "summary_table.json"

    def write_status() -> None:
        all_rows = list(rows)
        for name, out_json in [(name, out_json) for _, name, _, _, out_json in pending]:
            all_rows.append({"name": name, "status": "pending"})
        for _, name, proc, _, out_json in running:
            done = parse_done(out_json)
            rec = {"name": name, "status": "running", "pid": proc.pid}
            if done:
                rec.update({k: done.get(k) for k in ("success_count", "episodes_n", "success_rate")})
            all_rows.append(rec)
        for item in completed:
            all_rows.append(item)
        status_path.write_text(
            json.dumps(
                {
                    "pending": len(pending),
                    "running": [{"name": name, "pid": proc.pid} for _, name, proc, _, _ in running],
                    "completed": completed,
                    "time": time.time(),
                },
                indent=2,
            )
        )
        table_path.write_text(json.dumps(sorted(all_rows, key=lambda x: x["name"]), indent=2))

    while pending or running:
        while pending and len(running) < args.max_parallel:
            idx, name, param_path, out_dir, out_json = pending.pop(0)
            out_dir.mkdir(parents=True, exist_ok=True)
            log_f = (out_dir / "run.log").open("w")
            cmd = [
                args.uv,
                "run",
                "--no-sync",
                "--project",
                args.project,
                "python",
                "scripts/rollout_dexmg_with_contrastive_setpoint_residual.py",
                "--dataset-path",
                args.dataset_path,
                "--env-dataset-path",
                args.env_dataset_path,
                "--task-config",
                args.task_config,
                "--model-path",
                args.model_path,
                "--params",
                str(param_path),
                "--wm-ckpt",
                args.wm_ckpt,
                "--vae-ckpt",
                args.vae_ckpt,
                "--normalizers",
                args.normalizers,
                "--out",
                str(out_json),
                "--episodes",
                str(args.episodes),
                "--nfe",
                str(args.nfe),
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--history",
                str(args.history),
                "--chunk",
                str(args.chunk),
                "--trust-delta",
                str(args.trust_delta),
                "--qp-lambda-u",
                str(args.qp_lambda_u),
                "--tactile-channels",
                str(args.tactile_channels),
                "--joint-dim",
                str(args.joint_dim),
                "--action-dim",
                str(args.action_dim),
                "--seed",
                str(args.seed + idx * args.episodes),
                "--progress-every",
                str(args.progress_every),
            ]
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((idx, name, proc, log_f, out_json))
            print(json.dumps({"event": "start", "name": name, "pid": proc.pid}), flush=True)
        write_status()
        time.sleep(10)
        still = []
        for idx, name, proc, log_f, out_json in running:
            rc = proc.poll()
            if rc is None:
                still.append((idx, name, proc, log_f, out_json))
                continue
            log_f.close()
            done = parse_done(out_json)
            rec = {
                "name": name,
                "status": "done" if rc == 0 and done else "failed",
                "returncode": rc,
                "summary": str(out_json),
            }
            if done:
                rec.update({k: done.get(k) for k in ("success_count", "episodes_n", "success_rate")})
            completed.append(rec)
            print(json.dumps({"event": "done", **rec}), flush=True)
        running = still
    write_status()
    print(json.dumps({"event": "all_done", "completed": completed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
