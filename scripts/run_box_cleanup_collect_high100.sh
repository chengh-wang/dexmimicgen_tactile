#!/usr/bin/env bash
set -euo pipefail

cd /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile

STAMP="${1:?stamp required}"
TARGET_SUCCESS="${2:?target success required}"
TARGET_FAILURE="${3:?target failure required}"
SEED="${4:?seed required}"
MAX_ATTEMPTS="${5:-350}"
DEVICE="${6:-cuda}"

CKPT="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_box_cleanup_tactile_vae_virtual_s8fs25_flow_70k_20260728/models/model_step60000.pt"
DATA_ROOT="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/two_arm_box_cleanup"
ENV_H5="${DATA_ROOT}/shard00.hdf5"
OUT="outputs/world_model_rollouts/box_cleanup_wm_${TARGET_SUCCESS}s${TARGET_FAILURE}f_low20_high100_${STAMP}.hdf5"
LOG="outputs/world_model_rollouts/box_cleanup_wm_${TARGET_SUCCESS}s${TARGET_FAILURE}f_low20_high100_${STAMP}.log"
PIDF="outputs/world_model_rollouts/box_cleanup_wm_${TARGET_SUCCESS}s${TARGET_FAILURE}f_low20_high100_${STAMP}.pid"
VIDEO_DIR="outputs/world_model_rollouts/box_cleanup_wm_${TARGET_SUCCESS}s${TARGET_FAILURE}f_low20_high100_videos_${STAMP}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TACTILE_RENDERER=virtual
export TACTILE_VIRTUAL_SIGMA=8.0
export TACTILE_VIRTUAL_FORCE_SCALE=25.0
export TACTILE_VIRTUAL_MAX=1.0
export TACTILE_VIRTUAL_CANONICAL=1
export TACTILE_VIRTUAL_MAX_SURFACE_DIST=0.015

mkdir -p outputs/world_model_rollouts

nohup /home/labeng/.local/bin/uv run --no-sync \
  --project /home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising \
  python -u scripts/collect_dexmg_wm_rollouts.py \
  --dataset-path "$DATA_ROOT" \
  --env-dataset-path "$ENV_H5" \
  --task-config dexmg_box_cleanup_image_tactile_vae_virtual_s8fs25 \
  --task-name two_arm_box_cleanup \
  --success-model-path "$CKPT" \
  --failure-model-path "$CKPT" \
  --success-nfe 4 \
  --failure-nfe 4 \
  --target-success "$TARGET_SUCCESS" \
  --target-failure "$TARGET_FAILURE" \
  --max-attempts-per-split "$MAX_ATTEMPTS" \
  --max-episode-steps 400 \
  --seed "$SEED" \
  --output-path "$OUT" \
  --video-dir "$VIDEO_DIR" \
  --video-every 50 \
  --high-rate 100 \
  --image-compression lzf \
  --tactile-compression gzip \
  --gzip-level 4 \
  --tactile-dtype float16 \
  --device "$DEVICE" \
  > "$LOG" 2>&1 &

echo "$!" > "$PIDF"
echo "PID=$(cat "$PIDF")"
echo "OUT=$OUT"
echo "LOG=$LOG"
echo "PIDF=$PIDF"
