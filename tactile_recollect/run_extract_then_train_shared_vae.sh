#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile"
WM="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising"
UV="/home/labeng/.local/bin/uv"
DEX_PY="$ROOT/.venv/bin/python"
PYTHONPATH_VALUE="$ROOT:$ROOT/robosuite"
RAW_DIR="$ROOT/datasets/generated"
TACTILE_DIR="$ROOT/datasets/generated_tactile_proud2mm"
LOG_DIR="$ROOT/tactile_recollect/out/extract_logs"
VAE_OUT="$WM/runs/dexmg_shared_tactile_vae_proud2mm_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$TACTILE_DIR" "$LOG_DIR"

TASKS=(
  two_arm_box_cleanup
  two_arm_can_sort_random
  two_arm_coffee
  two_arm_drawer_cleanup
  two_arm_lift_tray
  two_arm_pouring
  two_arm_threading
  two_arm_three_piece_assembly
  two_arm_transport
)

run_extract() {
  local task="$1"
  local raw="$RAW_DIR/${task}.hdf5"
  local out="$TACTILE_DIR/${task}.hdf5"
  local log="$LOG_DIR/${task}.log"
  {
    echo "[$(date --iso-8601=seconds)] start extract $task"
    env PYTHONPATH="$PYTHONPATH_VALUE" MUJOCO_GL=egl \
      "$DEX_PY" -c \
      'import sys; import dexmimicgen; from tactile_recollect.extract import run; run(sys.argv[1], sys.argv[2], resume=True)' \
      "$raw" "$out"
    echo "[$(date --iso-8601=seconds)] done extract $task"
  } >>"$log" 2>&1
}

delay=0
for task in "${TASKS[@]}"; do
  (
    sleep "$delay"
    run_extract "$task"
  ) &
  delay=$((delay + 3))
done
wait

echo "[$(date --iso-8601=seconds)] validating tactile outputs"
env PYTHONPATH="$PYTHONPATH_VALUE" "$UV" run --no-sync --project "$WM" python - <<'PY'
import glob
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
        ds = ft["data"][first]["obs"]["robot0_tactile"]
        if len(ds.shape) != 4 or ds.shape[-2:] != (32, 32):
            raise RuntimeError(f"{task}: bad tactile shape {ds.shape}")
        print(f"{task}: demos={len(raw_demos)} tactile_shape_first={ds.shape}")
print("validation ok")
PY

echo "[$(date --iso-8601=seconds)] starting shared tactile VAE -> $VAE_OUT"
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

echo "[$(date --iso-8601=seconds)] pipeline complete: $VAE_OUT"
