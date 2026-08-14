#!/usr/bin/env bash
set -euo pipefail

cd /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT="outputs/world_model_rollouts/drawer_cleanup_wm_250failure_low20_high200_${STAMP}.hdf5"
CKPT="${2:-/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_drawer_cleanup_tactile_vae_virtual_s8fs25_flow_100k_20260730_04/models/model_step100000.pt}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TACTILE_RENDERER=virtual
export TACTILE_VIRTUAL_SIGMA=8.0
export TACTILE_VIRTUAL_FORCE_SCALE=25.0
export TACTILE_VIRTUAL_MAX=1.0
export TACTILE_VIRTUAL_CANONICAL=1
export TACTILE_VIRTUAL_MAX_SURFACE_DIST=0.015

/home/labeng/.local/bin/uv run --no-sync \
  --project /home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising \
  python scripts/collect_dexmg_wm_rollouts.py \
  --dataset-path /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/two_arm_drawer_cleanup/shard00.hdf5 \
  --env-dataset-path /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated/two_arm_drawer_cleanup.hdf5 \
  --task-config dexmg_drawer_cleanup_image_tactile_vae_virtual_s8fs25 \
  --task-name two_arm_drawer_cleanup \
  --success-model-path "$CKPT" \
  --failure-model-path "$CKPT" \
  --success-nfe 8 \
  --failure-nfe 2 \
  --target-success 0 \
  --target-failure 250 \
  --max-attempts-per-split 1000 \
  --max-episode-steps 400 \
  --seed 9101 \
  --output-path "$OUT" \
  --video-dir "outputs/world_model_rollouts/drawer_cleanup_wm_250failure_videos_${STAMP}" \
  --video-every 50 \
  --image-compression lzf \
  --tactile-compression gzip \
  --gzip-level 4 \
  --tactile-dtype float16 \
  --device cuda
