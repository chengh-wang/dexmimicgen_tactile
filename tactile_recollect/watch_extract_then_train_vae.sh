#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile"
WM="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising"
UV="/home/labeng/.local/bin/uv"
PYTHONPATH_VALUE="$ROOT:$ROOT/robosuite"
TACTILE_DIR="$ROOT/datasets/generated_tactile_proud2mm"
RAW_DIR="$ROOT/datasets/generated"
LOG_DIR="$ROOT/tactile_recollect/out/extract_logs"
VAE_OUT="$WM/runs/dexmg_shared_tactile_vae_proud2mm_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

echo "[$(date --iso-8601=seconds)] watcher started"
while pgrep -f "from tactile_recollect.extract import run" >/dev/null; do
  echo "[$(date --iso-8601=seconds)] extraction still running"
  sleep 60
done

echo "[$(date --iso-8601=seconds)] extraction finished; validating"
env PYTHONPATH="$PYTHONPATH_VALUE" "$UV" run --no-sync --project "$WM" python - <<'PY'
import h5py
import os

raw_dir = "/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated"
tactile_dir = "/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_proud2mm"
tasks = [
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
for task in tasks:
    raw = os.path.join(raw_dir, f"{task}.hdf5")
    tac = os.path.join(tactile_dir, f"{task}.hdf5")
    with h5py.File(raw, "r") as fr, h5py.File(tac, "r") as ft:
        raw_demos = set(fr["data"].keys())
        tac_demos = set(ft["data"].keys())
        if raw_demos != tac_demos:
            missing = sorted(raw_demos - tac_demos, key=lambda d: int(d.split("_")[1]))[:10]
            extra = sorted(tac_demos - raw_demos, key=lambda d: int(d.split("_")[1]))[:10]
            raise RuntimeError(f"{task}: demo mismatch missing={missing} extra={extra}")
        first = sorted(raw_demos, key=lambda d: int(d.split("_")[1]))[0]
        shape = ft["data"][first]["obs"]["robot0_tactile"].shape
        if len(shape) != 4 or shape[-2:] != (32, 32):
            raise RuntimeError(f"{task}: bad tactile shape {shape}")
        print(f"{task}: demos={len(raw_demos)} first_tactile_shape={shape}")
print("validation ok")
PY

echo "[$(date --iso-8601=seconds)] starting shared tactile VAE: $VAE_OUT"
env PYTHONPATH="$PYTHONPATH_VALUE" "$UV" run --no-sync --project "$WM" python \
  "$ROOT/tactile_recollect/train_shared_tactile_vae.py" \
  --h5-glob "$TACTILE_DIR/*.hdf5" \
  --out "$VAE_OUT" \
  --steps 50000 \
  --save-every 2000 \
  --batch-size 1024 \
  --num-workers 8 \
  --eval-batches 64 \
  --tactile-scale 300.0 \
  --clip-input

echo "[$(date --iso-8601=seconds)] VAE complete: $VAE_OUT"
