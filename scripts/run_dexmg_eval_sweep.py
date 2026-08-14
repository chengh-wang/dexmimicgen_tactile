#!/usr/bin/env python3
"""Sequential checkpoint eval sweep for ThreePiece tactile vs no-tactile."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path


DEX = Path("/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile")
POL = Path("/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising")
RAW_H5 = DEX / "datasets/generated/two_arm_three_piece_assembly.hdf5"
TACTILE_H5 = (
    DEX
    / "datasets/generated_tactile_actionrollout_virtual_s8fs25_shards"
    / "20260726_122743_s8fs25/two_arm_three_piece_assembly"
)
TACTILE_LOG = POL / "logs_dexmg_three_piece_tactile_cnn_virtual_s8fs25_flow_100k_20260726"
NOTACTILE_LOG = POL / "logs_dexmg_three_piece_notactile_flow_100k_20260723"
STEPS = list(range(10_000, 100_001, 10_000))
NFES = (2, 4, 8)

SUMMARY_RE = re.compile(
    r"\[summary\] nfe=(?P<nfe>\d+) success_rate=(?P<success>[0-9.]+) "
    r"mean_reward=(?P<reward>[0-9.]+) "
    r"mean_steps=(?P<steps>[0-9.]+) "
    r"mean_action_absmax=(?P<action_absmax>[0-9.]+)"
)


EXISTING = {
    ("tactile", 10_000): TACTILE_LOG / "eval_rollout_step10000_20eps_20260726/eval.out",
    ("tactile", 30_000): TACTILE_LOG / "eval_rollout_step30000_20eps_20260726/eval.out",
    ("tactile", 100_000): TACTILE_LOG / "eval_rollout_step100000_20eps_20260727/eval.out",
    ("notactile", 30_000): NOTACTILE_LOG / "eval_rollout_step30000_20eps_20260726_compare_virtual/eval.out",
}


def parse_summary(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = SUMMARY_RE.search(line)
        if m:
            out[int(m.group("nfe"))] = {
                "success": float(m.group("success")),
                "reward": float(m.group("reward")),
                "steps": float(m.group("steps")),
                "action_absmax": float(m.group("action_absmax")),
            }
    return out


def complete(path: Path) -> bool:
    return all(nfe in parse_summary(path) for nfe in NFES)


def wait_for_unit_inactive(unit: str) -> None:
    while True:
        p = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if p.stdout.strip() != "active":
            return
        print(f"[wait] {unit} still active", flush=True)
        time.sleep(60)


def eval_out(kind: str, step: int, out_root: Path) -> Path:
    existing = EXISTING.get((kind, step))
    if existing is not None and complete(existing):
        return existing
    return out_root / kind / f"step{step:06d}" / "eval.out"


def run_eval(kind: str, step: int, out_path: Path) -> None:
    if complete(out_path):
        print(f"[skip] {kind} step={step}: complete at {out_path}", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = TACTILE_LOG if kind == "tactile" else NOTACTILE_LOG
    model = model_dir / "models" / f"model_step{step}.pt"
    if not model.exists():
        raise FileNotFoundError(model)

    if kind == "tactile":
        dataset = TACTILE_H5
        task = "dexmg_three_piece_image_tactile_cnn_virtual_s8fs25"
    else:
        dataset = RAW_H5
        task = "dexmg_three_piece_image_notactile"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DEX}:{DEX / 'robosuite'}:{POL}"
    env["MUJOCO_GL"] = "egl"
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if kind == "tactile":
        env["TACTILE_RENDERER"] = "virtual"
        env["TACTILE_VIRTUAL_SIGMA"] = "8.0"
        env["TACTILE_VIRTUAL_FORCE_SCALE"] = "25.0"
    else:
        env.pop("TACTILE_RENDERER", None)
        env.pop("TACTILE_VIRTUAL_SIGMA", None)
        env.pop("TACTILE_VIRTUAL_FORCE_SCALE", None)

    cmd = [
        "/home/labeng/.local/bin/uv",
        "run",
        "--no-sync",
        "python",
        "examples/eval_dexmg_rollout.py",
        "--dataset-path",
        str(dataset),
        "--env-dataset-path",
        str(RAW_H5),
        "--task-config",
        task,
        "--model-path",
        str(model),
        "--episodes",
        "20",
        "--nfe",
        "2",
        "4",
        "8",
        "--max-episode-steps",
        "400",
        "--skip-sanity",
    ]
    print(f"[run] {kind} step={step} -> {out_path}", flush=True)
    with out_path.open("w") as f:
        subprocess.run(cmd, cwd=POL, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    if not complete(out_path):
        raise RuntimeError(f"eval finished without complete summaries: {out_path}")


def write_tables(out_root: Path) -> None:
    rows = []
    for kind in ("tactile", "notactile"):
        for step in STEPS:
            path = eval_out(kind, step, out_root)
            summary = parse_summary(path)
            row = {"kind": kind, "step": step, "eval_out": str(path)}
            for nfe in NFES:
                values = summary.get(nfe, {})
                row[f"nfe{nfe}_success"] = values.get("success", float("nan"))
                row[f"nfe{nfe}_reward"] = values.get("reward", float("nan"))
                row[f"nfe{nfe}_steps"] = values.get("steps", float("nan"))
                row[f"nfe{nfe}_action_absmax"] = values.get("action_absmax", float("nan"))
            rows.append(row)

    csv_path = out_root / "summary.csv"
    md_path = out_root / "summary.md"
    fields = [
        "kind",
        "step",
        "nfe2_success",
        "nfe4_success",
        "nfe8_success",
        "nfe2_reward",
        "nfe4_reward",
        "nfe8_reward",
        "nfe2_steps",
        "nfe4_steps",
        "nfe8_steps",
        "nfe2_action_absmax",
        "nfe4_action_absmax",
        "nfe8_action_absmax",
        "eval_out",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| kind | step | nfe2 succ | nfe4 succ | nfe8 succ | nfe2 rew | nfe4 rew | nfe8 rew | nfe2 steps | nfe4 steps | nfe8 steps | nfe2 absmax | nfe4 absmax | nfe8 absmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['kind']} | {r['step']} | {r['nfe2_success']:.3f} | "
            f"{r['nfe4_success']:.3f} | {r['nfe8_success']:.3f} | "
            f"{r['nfe2_reward']:.2f} | {r['nfe4_reward']:.2f} | {r['nfe8_reward']:.2f} | "
            f"{r['nfe2_steps']:.1f} | {r['nfe4_steps']:.1f} | {r['nfe8_steps']:.1f} | "
            f"{r['nfe2_action_absmax']:.3f} | {r['nfe4_action_absmax']:.3f} | {r['nfe8_action_absmax']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[table] {csv_path}", flush=True)
    print(f"[table] {md_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default=str(TACTILE_LOG / "eval_sweep_20eps_20260727"),
    )
    parser.add_argument("--wait-unit", default="dexmg-eval-virtual-s8fs25-step100000-20eps.service")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.wait_unit:
        wait_for_unit_inactive(args.wait_unit)

    for kind in ("tactile", "notactile"):
        for step in STEPS:
            path = eval_out(kind, step, out_root)
            run_eval(kind, step, path)
            write_tables(out_root)

    write_tables(out_root)


if __name__ == "__main__":
    main()
