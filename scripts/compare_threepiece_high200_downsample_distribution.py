#!/usr/bin/env python3
"""Compare ThreePiece high200 replay downsampled to 20 Hz against 20 Hz tactile demos.

The high200 files contain replayed high-rate tactile / joint trajectories plus
copied low20 observations. This script checks whether the high200 samples align
more closely to the current official 20 Hz observation (`k`) or the next one
(`k+1`), and summarizes replay drift / distribution differences.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def demo_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


@dataclass(frozen=True)
class DemoLoc:
    path: Path
    demo: str


class ScalarStats:
    def __init__(self) -> None:
        self.values: list[np.ndarray] = []

    def update(self, value: np.ndarray | float) -> None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            self.values.append(arr)

    def finish(self) -> dict[str, float | int | None]:
        if not self.values:
            return {"n": 0}
        x = np.concatenate(self.values)
        return {
            "n": int(x.size),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "p50": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "max": float(x.max()),
        }


def find_h5s(root_or_paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in root_or_paths:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("shard*.hdf5")))
        else:
            out.append(p)
    return out


def index_demos(paths: list[Path]) -> dict[str, DemoLoc]:
    index: dict[str, DemoLoc] = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            root = f["data"] if "data" in f else f
            for demo in root.keys():
                if demo in index:
                    raise KeyError(f"duplicate demo {demo}: {index[demo].path} and {path}")
                index[demo] = DemoLoc(path=path, demo=demo)
    return index


def open_files(paths: Iterable[Path]) -> dict[Path, h5py.File]:
    return {p: h5py.File(p, "r") for p in sorted(set(paths))}


def root_group(f: h5py.File) -> h5py.Group:
    return f["data"] if "data" in f else f


def joint20_from_obs(obs: h5py.Group, n: int | None = None) -> np.ndarray:
    a = obs["robot0_joint_pos"][:n].astype(np.float32)
    b = obs["robot1_joint_pos"][:n].astype(np.float32)
    return np.concatenate([a, b], axis=-1)


def frame_errors(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (a.astype(np.float32) - b.astype(np.float32)).reshape(a.shape[0], -1)
    abs_x = np.abs(x)
    mae = abs_x.mean(axis=1)
    rmse = np.sqrt(np.square(x).mean(axis=1))
    maxabs = abs_x.max(axis=1)
    return mae, rmse, maxabs


def update_frame_error(stats: dict[str, ScalarStats], name: str, a: np.ndarray, b: np.ndarray) -> None:
    mae, rmse, maxabs = frame_errors(a, b)
    s(stats, f"{name}_mae").update(mae)
    s(stats, f"{name}_rmse").update(rmse)
    s(stats, f"{name}_maxabs").update(maxabs)


def update_tactile_dist(stats: dict[str, ScalarStats], name: str, x: np.ndarray) -> None:
    flat = x.reshape(x.shape[0], -1).astype(np.float32)
    s(stats, f"{name}_pixel_mean").update(flat.mean(axis=1))
    s(stats, f"{name}_pixel_max").update(flat.max(axis=1))
    s(stats, f"{name}_active_frac_gt_0p01").update((flat > 0.01).mean(axis=1))
    s(stats, f"{name}_active_frac_gt_0p05").update((flat > 0.05).mean(axis=1))


def update_vector_dist(stats: dict[str, ScalarStats], name: str, x: np.ndarray) -> None:
    x = x.reshape(-1, x.shape[-1]).astype(np.float32)
    s(stats, f"{name}_l2").update(np.linalg.norm(x, axis=1))
    s(stats, f"{name}_abs_mean").update(np.abs(x).mean(axis=1))
    s(stats, f"{name}_abs_max").update(np.abs(x).max(axis=1))


def ensure_stats(stats: dict[str, ScalarStats], key: str) -> ScalarStats:
    if key not in stats:
        stats[key] = ScalarStats()
    return stats[key]


def s(stats: dict[str, ScalarStats], key: str) -> ScalarStats:
    return ensure_stats(stats, key)


def compare(args: argparse.Namespace) -> dict:
    high_paths = find_h5s(args.high200)
    official_paths = find_h5s(args.official20)
    high_index = index_demos(high_paths)
    official_index = index_demos(official_paths)
    demos = sorted(set(high_index) & set(official_index), key=demo_sort_key)
    if args.max_demos > 0:
        demos = demos[: args.max_demos]
    if not demos:
        raise RuntimeError("no common demos between high200 and official20 inputs")

    high_files = open_files(loc.path for loc in high_index.values())
    official_files = open_files(loc.path for loc in official_index.values())
    stats: dict[str, ScalarStats] = {}
    per_demo: list[dict] = []

    try:
        for i, demo_name in enumerate(demos, 1):
            hloc = high_index[demo_name]
            oloc = official_index[demo_name]
            hdemo = root_group(high_files[hloc.path])[demo_name]
            odemo = root_group(official_files[oloc.path])[demo_name]

            high_action = hdemo["low20/actions"][:].astype(np.float32)
            off_action = odemo["actions"][:].astype(np.float32)
            off_obs = odemo["obs"]
            off_joint = joint20_from_obs(off_obs)
            off_tactile = off_obs["robot0_tactile"][:].astype(np.float32)

            n = min(
                high_action.shape[0],
                off_action.shape[0] - 1,
                off_joint.shape[0] - 1,
                off_tactile.shape[0] - 1,
                hdemo["high200/robot_joint_pos"].shape[0],
                hdemo["high200/tactile"].shape[0],
            )
            if n <= 0:
                continue

            high_action = high_action[:n]
            high_action_first = hdemo["high200/action"][:n, 0].astype(np.float32)
            high_action_last = hdemo["high200/action"][:n, -1].astype(np.float32)
            high_joint_first = hdemo["high200/robot_joint_pos"][:n, 0].astype(np.float32)
            high_joint_last = hdemo["high200/robot_joint_pos"][:n, -1].astype(np.float32)
            high_tactile_first = hdemo["high200/tactile"][:n, 0].astype(np.float32)
            high_tactile_last = hdemo["high200/tactile"][:n, -1].astype(np.float32)

            off_action_cur = off_action[:n]
            off_action_next = off_action[1 : n + 1]
            off_joint_cur = off_joint[:n]
            off_joint_next = off_joint[1 : n + 1]
            off_tactile_cur = off_tactile[:n]
            off_tactile_next = off_tactile[1 : n + 1]

            low_obs_joint = joint20_from_obs(hdemo["low20/obs"], n + 1)
            low_states = hdemo["low20/states"][: n + 1].astype(np.float32)
            off_states = odemo["states"][: n + 1].astype(np.float32)

            update_frame_error(stats, "low20_obs_joint_vs_official_cur", low_obs_joint[:n], off_joint_cur)
            update_frame_error(stats, "low20_obs_joint_next_vs_official_next", low_obs_joint[1:], off_joint_next)
            update_frame_error(stats, "low20_states_vs_official_cur", low_states[:n], off_states[:n])
            update_frame_error(stats, "low20_states_next_vs_official_next", low_states[1:], off_states[1:])

            update_frame_error(stats, "low20_action_vs_official_cur", high_action, off_action_cur)
            update_frame_error(stats, "low20_action_vs_official_next", high_action, off_action_next)
            update_frame_error(stats, "high_action_first_vs_official_cur", high_action_first, off_action_cur)
            update_frame_error(stats, "high_action_last_vs_official_next", high_action_last, off_action_next)
            update_frame_error(stats, "high_action_first_vs_low20_action", high_action_first, high_action)
            update_frame_error(stats, "high_action_last_vs_low20_action", high_action_last, high_action)

            update_frame_error(stats, "high_joint_first_vs_official_cur", high_joint_first, off_joint_cur)
            update_frame_error(stats, "high_joint_last_vs_official_cur", high_joint_last, off_joint_cur)
            update_frame_error(stats, "high_joint_last_vs_official_next", high_joint_last, off_joint_next)
            update_frame_error(stats, "high_joint_last_vs_low20_obs_next", high_joint_last, low_obs_joint[1:])

            update_frame_error(stats, "high_tactile_first_vs_official_cur", high_tactile_first, off_tactile_cur)
            update_frame_error(stats, "high_tactile_last_vs_official_cur", high_tactile_last, off_tactile_cur)
            update_frame_error(stats, "high_tactile_last_vs_official_next", high_tactile_last, off_tactile_next)

            update_tactile_dist(stats, "official_tactile_cur", off_tactile_cur)
            update_tactile_dist(stats, "official_tactile_next", off_tactile_next)
            update_tactile_dist(stats, "high_tactile_first", high_tactile_first)
            update_tactile_dist(stats, "high_tactile_last", high_tactile_last)
            update_vector_dist(stats, "official_action_cur", off_action_cur)
            update_vector_dist(stats, "high_low20_action", high_action)
            update_vector_dist(stats, "official_joint_next", off_joint_next)
            update_vector_dist(stats, "high_joint_last", high_joint_last)

            state_err = hdemo["high200/state_err_after_low_step"][:n].astype(np.float32)
            s(stats, "state_err_after_low_step").update(state_err)
            s(stats, "demo_length20_high200").update(float(n))

            j_next_rmse = frame_errors(high_joint_last, off_joint_next)[1]
            j_cur_rmse = frame_errors(high_joint_last, off_joint_cur)[1]
            t_next_mae = frame_errors(high_tactile_last, off_tactile_next)[0]
            t_cur_mae = frame_errors(high_tactile_last, off_tactile_cur)[0]
            per_demo.append(
                {
                    "demo": demo_name,
                    "n": int(n),
                    "high_path": str(hloc.path),
                    "official_path": str(oloc.path),
                    "state_err_mean": float(state_err.mean()),
                    "state_err_max": float(state_err.max()),
                    "joint_last_vs_official_next_rmse_mean": float(j_next_rmse.mean()),
                    "joint_last_vs_official_cur_rmse_mean": float(j_cur_rmse.mean()),
                    "tactile_last_vs_official_next_mae_mean": float(t_next_mae.mean()),
                    "tactile_last_vs_official_cur_mae_mean": float(t_cur_mae.mean()),
                }
            )
            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"[progress] {i}/{len(demos)} demos", flush=True)
    finally:
        for f in high_files.values():
            f.close()
        for f in official_files.values():
            f.close()

    summary = {k: v.finish() for k, v in sorted(stats.items())}
    return {
        "high200_paths": [str(p) for p in high_paths],
        "official20_paths": [str(p) for p in official_paths],
        "paired_demos": len(per_demo),
        "summary": summary,
        "worst_by_state_err_mean": sorted(per_demo, key=lambda r: r["state_err_mean"], reverse=True)[:20],
        "worst_by_joint_next_rmse": sorted(
            per_demo,
            key=lambda r: r["joint_last_vs_official_next_rmse_mean"],
            reverse=True,
        )[:20],
        "worst_by_tactile_next_mae": sorted(
            per_demo,
            key=lambda r: r["tactile_last_vs_official_next_mae_mean"],
            reverse=True,
        )[:20],
        "per_demo": per_demo if args.include_per_demo else [],
    }


def fmt_metric(row: dict[str, float | int | None], key: str) -> str:
    v = row.get(key)
    if v is None:
        return ""
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(float(v)):
        return str(v)
    return f"{float(v):.6g}"


def write_markdown(result: dict, path: Path) -> None:
    summary = result["summary"]
    key_metrics = [
        "state_err_after_low_step",
        "low20_obs_joint_vs_official_cur_rmse",
        "low20_obs_joint_next_vs_official_next_rmse",
        "low20_action_vs_official_cur_rmse",
        "low20_action_vs_official_next_rmse",
        "high_action_first_vs_official_cur_rmse",
        "high_action_last_vs_official_next_rmse",
        "high_joint_first_vs_official_cur_rmse",
        "high_joint_last_vs_official_cur_rmse",
        "high_joint_last_vs_official_next_rmse",
        "high_joint_last_vs_low20_obs_next_rmse",
        "high_tactile_first_vs_official_cur_mae",
        "high_tactile_last_vs_official_cur_mae",
        "high_tactile_last_vs_official_next_mae",
        "official_tactile_next_pixel_mean",
        "high_tactile_last_pixel_mean",
        "official_tactile_next_active_frac_gt_0p05",
        "high_tactile_last_active_frac_gt_0p05",
    ]
    lines = [
        "# ThreePiece High200 Downsample Distribution Check",
        "",
        f"Paired demos: `{result['paired_demos']}`",
        "",
        "| Metric | N | Mean | P50 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in key_metrics:
        if key not in summary:
            continue
        row = summary[key]
        lines.append(
            "| "
            + key
            + " | "
            + " | ".join(fmt_metric(row, k) for k in ["n", "mean", "p50", "p95", "p99", "max"])
            + " |"
        )
    lines.extend(["", "## Worst State Drift", "", "| Demo | N | State Err Mean | State Err Max | Joint Next RMSE | Tactile Next MAE |", "|---|---:|---:|---:|---:|---:|"])
    for row in result["worst_by_state_err_mean"][:10]:
        lines.append(
            f"| `{row['demo']}` | {row['n']} | {row['state_err_mean']:.6g} | "
            f"{row['state_err_max']:.6g} | {row['joint_last_vs_official_next_rmse_mean']:.6g} | "
            f"{row['tactile_last_vs_official_next_mae_mean']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high200", nargs="+", required=True, help="High200 root dir or shard paths")
    parser.add_argument("--official20", nargs="+", required=True, help="Official 20Hz tactile root dir or shard paths")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--include-per-demo", action="store_true")
    args = parser.parse_args()

    result = compare(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    write_markdown(result, out_md)
    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()
