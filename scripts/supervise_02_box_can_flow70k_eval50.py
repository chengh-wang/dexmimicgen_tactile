#!/usr/bin/env python3
"""Wait for 02 box/can flow training and run 50-episode rollout evals."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path


DEX = Path("/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile")
POL = Path("/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising")
UV = Path("/home/labeng/.local/bin/uv")

RUNS = {
    "box_cleanup": {
        "unit": "dexmg-box-cleanup-tactile-vae-flow-70k-20260728.service",
        "task_config": "dexmg_box_cleanup_image_tactile_vae_virtual_s8fs25",
        "train_log_dir": POL / "logs_dexmg_box_cleanup_tactile_vae_virtual_s8fs25_flow_70k_20260728",
        "dataset_path": DEX
        / "datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards"
        / "20260727_0828_s8fs25_alltasks/two_arm_box_cleanup",
        "env_dataset_path": DEX
        / "datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards"
        / "20260727_0828_s8fs25_alltasks/two_arm_box_cleanup/shard00.hdf5",
    },
    "can_sort_random": {
        "unit": "dexmg-can-sort-random-tactile-vae-flow-70k-20260728.service",
        "task_config": "dexmg_can_sort_random_image_tactile_vae_virtual_s8fs25",
        "train_log_dir": POL / "logs_dexmg_can_sort_random_tactile_vae_virtual_s8fs25_flow_70k_20260728",
        "dataset_path": DEX
        / "datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards"
        / "20260727_0828_s8fs25_alltasks/two_arm_can_sort_random",
        "env_dataset_path": DEX
        / "datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards"
        / "20260727_0828_s8fs25_alltasks/two_arm_can_sort_random/shard00.hdf5",
    },
}

OUT_ROOT = DEX / "outputs/flow70k_02_box_can_eval50_20260728"
SUMMARY_RE = re.compile(
    r"\[summary\] nfe=(?P<nfe>\d+) success_rate=(?P<success>[0-9.]+) "
    r"mean_reward=(?P<reward>[0-9.]+) "
    r"mean_steps=(?P<steps>[0-9.]+) "
    r"mean_action_absmax=(?P<action_absmax>[0-9.]+)"
)


def is_active(unit: str) -> bool:
    proc = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.strip() == "active"


def wait_for_training() -> None:
    while True:
        active = [name for name, run in RUNS.items() if is_active(str(run["unit"]))]
        print(f"[wait] active_train={active}", flush=True)
        if not active:
            return
        time.sleep(300)


def parse_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        match = SUMMARY_RE.search(line)
        if match and int(match.group("nfe")) == 4:
            return {
                "success": float(match.group("success")),
                "reward": float(match.group("reward")),
                "steps": float(match.group("steps")),
                "action_absmax": float(match.group("action_absmax")),
            }
    return None


def run_eval(name: str, run: dict) -> dict:
    ckpt = run["train_log_dir"] / "models/model_step70000.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing 70k checkpoint for {name}: {ckpt}")

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_step70000_nfe4_50eps.out"
    summary = parse_summary(out_path)
    if summary is None:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{DEX}:{DEX / 'robosuite'}:{POL}"
        env["MUJOCO_GL"] = "egl"
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        env["TACTILE_RENDERER"] = "virtual"
        env["TACTILE_VIRTUAL_SIGMA"] = "8.0"
        env["TACTILE_VIRTUAL_FORCE_SCALE"] = "25.0"
        cmd = [
            str(UV),
            "run",
            "--no-sync",
            "python",
            "examples/eval_dexmg_rollout.py",
            "--dataset-path",
            str(run["dataset_path"]),
            "--env-dataset-path",
            str(run["env_dataset_path"]),
            "--task-config",
            str(run["task_config"]),
            "--model-path",
            str(ckpt),
            "--episodes",
            "50",
            "--nfe",
            "4",
            "--max-episode-steps",
            "400",
            "--skip-sanity",
        ]
        print(f"[eval] {name} -> {out_path}", flush=True)
        with out_path.open("w") as f:
            subprocess.run(
                cmd,
                cwd=POL,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = parse_summary(out_path)
        if summary is None:
            raise RuntimeError(f"eval finished without nfe=4 summary: {out_path}")

    result = {
        "task": name,
        "checkpoint": str(ckpt),
        "eval_out": str(out_path),
        "nfe": 4,
        "episodes": 50,
        **summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_summary(results: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "| task | step | nfe | episodes | success | success/50 | reward | mean_steps | eval_out |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        successes = int(round(r["success"] * r["episodes"]))
        lines.append(
            f"| {r['task']} | 70000 | {r['nfe']} | {r['episodes']} | "
            f"{r['success']:.3f} | {successes}/50 | {r['reward']:.2f} | "
            f"{r['steps']:.1f} | {r['eval_out']} |"
        )
    (OUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[summary] {OUT_ROOT / 'summary.md'}", flush=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    wait_for_training()
    results = [run_eval(name, run) for name, run in RUNS.items()]
    write_summary(results)


if __name__ == "__main__":
    main()
