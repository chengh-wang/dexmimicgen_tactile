#!/usr/bin/env python3
"""Launch progress-aware setpoint residual eval sweep with bounded parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7200)
    parser.add_argument("--uv", default="/home/labeng/.local/bin/uv")
    parser.add_argument("--project", default="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising")
    parser.add_argument(
        "--model-path",
        default="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_three_piece_tactile_cnn_virtual_s8fs25_flow_100k_20260726/models/model_step70000.pt",
    )
    parser.add_argument("--wm-ckpt", default="outputs/threepiece_20hz_wm_vae8k_h4_attn_jointpos_bs512_520epoch_20260727/wm_best.pt")
    parser.add_argument("--normalizers", default="outputs/threepiece_20hz_wm_vae8k_h4_attn_jointpos_bs512_520epoch_20260727/normalizers.npz")
    args = parser.parse_args()

    params_root = Path(args.params_root)
    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    params = sorted(params_root.glob("*/contrastive_params.npz"))
    if not params:
        raise RuntimeError(f"no params under {params_root}")

    pending = []
    for p in params:
        name = p.parent.name
        out = eval_root / name
        if (out / "summary.json").exists():
            continue
        pending.append((name, p, out))

    running: list[tuple[str, subprocess.Popen, object]] = []
    completed = []
    status_path = eval_root / "launcher_status.json"

    def write_status() -> None:
        status_path.write_text(
            json.dumps(
                {
                    "pending": len(pending),
                    "running": [name for name, _, _ in running],
                    "completed": completed,
                    "time": time.time(),
                },
                indent=2,
            )
        )

    while pending or running:
        while pending and len(running) < args.max_parallel:
            name, param_path, out = pending.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log_f = (out / "run.log").open("w")
            cmd = [
                args.uv,
                "--project",
                args.project,
                "run",
                "python",
                "scripts/rollout_policy_with_contrastive_setpoint_residual.py",
                "--params",
                str(param_path),
                "--out-dir",
                str(out),
                "--model-path",
                args.model_path,
                "--wm-ckpt",
                args.wm_ckpt,
                "--normalizers",
                args.normalizers,
                "--nfe",
                "4",
                "--episodes",
                str(args.episodes),
                "--max-episode-steps",
                "400",
                "--seed",
                str(args.seed),
                "--trust-delta",
                "0.010",
                "--qp-lambda-u",
                "20",
                "--progress-every",
                "100",
            ]
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((name, proc, log_f))
            print(json.dumps({"event": "start", "name": name, "pid": proc.pid}), flush=True)
        write_status()
        time.sleep(5)
        still = []
        for name, proc, log_f in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, log_f))
            else:
                log_f.close()
                completed.append({"name": name, "returncode": rc})
                print(json.dumps({"event": "done", "name": name, "returncode": rc}), flush=True)
        running = still
    write_status()
    print(json.dumps({"event": "all_done", "completed": completed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
