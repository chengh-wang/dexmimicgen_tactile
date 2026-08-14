#!/usr/bin/env bash
set -euo pipefail

cd /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT="outputs/world_model_rollouts/drawer_cleanup_wm_250success_250failure_low20_high200_${STAMP}.hdf5"
LOG="outputs/world_model_rollouts/drawer_cleanup_wm_250success_250failure_low20_high200_${STAMP}.log"
PIDF="outputs/world_model_rollouts/drawer_cleanup_wm_250success_250failure_low20_high200_${STAMP}.pid"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TACTILE_RENDERER=virtual
export TACTILE_VIRTUAL_SIGMA=8.0
export TACTILE_VIRTUAL_FORCE_SCALE=25.0
export TACTILE_VIRTUAL_MAX=1.0
export TACTILE_VIRTUAL_CANONICAL=1
export TACTILE_VIRTUAL_MAX_SURFACE_DIST=0.015

nohup /home/labeng/.local/bin/uv run --no-sync \
  --project /home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising \
  python scripts/collect_dexmg_wm_rollouts.py \
  --dataset-path /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/two_arm_drawer_cleanup/shard00.hdf5 \
  --env-dataset-path /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated/two_arm_drawer_cleanup.hdf5 \
  --task-config dexmg_drawer_cleanup_image_tactile_vae_virtual_s8fs25 \
  --task-name two_arm_drawer_cleanup \
  --success-model-path outputs/policy_ckpts/drawer_cleanup_flow100k_20260730_04_model_step100000.pt \
  --failure-model-path outputs/policy_ckpts/drawer_cleanup_flow100k_20260730_04_model_step100000.pt \
  --success-nfe 8 \
  --failure-nfe 2 \
  --target-success 250 \
  --target-failure 250 \
  --max-attempts-per-split 2500 \
  --max-episode-steps 400 \
  --seed 8101 \
  --output-path "$OUT" \
  --video-dir "outputs/world_model_rollouts/drawer_cleanup_wm_250success_250failure_videos_${STAMP}" \
  --video-every 50 \
  --image-compression lzf \
  --tactile-compression gzip \
  --gzip-level 4 \
  --tactile-dtype float16 \
  --device cuda \
  > "$LOG" 2>&1 &

echo "$!" > "$PIDF"
echo "PID=$(cat "$PIDF")"
echo "OUT=$OUT"
echo "LOG=$LOG"
echo "PIDF=$PIDF"
