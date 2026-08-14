#!/usr/bin/env bash
set -euo pipefail

cd /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SHARDS="${SHARDS:-6}"
MAX_LOW_STEPS="${MAX_LOW_STEPS:-0}"
DATASET="${DATASET:-datasets/generated/two_arm_three_piece_assembly.hdf5}"
OUT_ROOT="${OUT_ROOT:-outputs/high200_interp/threepiece_interp_high200_all_${STAMP}}"
UV="${UV:-/home/labeng/.local/bin/uv}"
PROJECT="${PROJECT:-/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising}"

mkdir -p "$OUT_ROOT/logs"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TACTILE_RENDERER=virtual
export TACTILE_VIRTUAL_SIGMA=8.0
export TACTILE_VIRTUAL_FORCE_SCALE=25.0
export TACTILE_VIRTUAL_MAX=1.0
export TACTILE_VIRTUAL_CANONICAL=1
export TACTILE_VIRTUAL_MAX_SURFACE_DIST=0.015

echo "[launch] dataset=$DATASET"
echo "[launch] out_root=$OUT_ROOT"
echo "[launch] shards=$SHARDS max_low_steps=$MAX_LOW_STEPS"

for shard in $(seq 0 $((SHARDS - 1))); do
  shard_name="$(printf 'shard%02d' "$shard")"
  out="$OUT_ROOT/${shard_name}.hdf5"
  log="$OUT_ROOT/logs/${shard_name}.log"
  echo "[launch] $shard_name -> $out"
  (
    "$UV" run --no-sync --project "$PROJECT" \
      python scripts/extract_high200_interp_action_shard.py \
        --dataset "$DATASET" \
        --out "$out" \
        --shard-index "$shard" \
        --num-shards "$SHARDS" \
        --max-low-steps "$MAX_LOW_STEPS" \
        --resume
  ) >"$log" 2>&1 &
done

wait
echo "[done] all shards finished under $OUT_ROOT"
