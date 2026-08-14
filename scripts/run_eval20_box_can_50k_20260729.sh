#!/usr/bin/env bash
set -euo pipefail

cd /home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export HDF5_USE_FILE_LOCKING=FALSE

eval_one() {
  local name="$1"
  local task_config="$2"
  local data_path="$3"
  local env_data_path="$4"
  local model_path="$5"
  local out_dir="$6"
  local nfe="$7"

  mkdir -p "$out_dir"
  echo "[eval-start] ${name} step50000 nfe=${nfe} episodes=20 $(date)"
  /home/labeng/.local/bin/uv run --no-sync python -u examples/eval_dexmg_rollout.py \
    --dataset-path "$data_path" \
    --env-dataset-path "$env_data_path" \
    --task-config "$task_config" \
    --model-path "$model_path" \
    --episodes 20 \
    --nfe "$nfe" \
    --skip-sanity > "$out_dir/eval.out" 2>&1
  echo "[eval-done] ${name} step50000 nfe=${nfe} $(date)"
  grep -E "\\[summary\\]|\\[summary_json\\]" "$out_dir/eval.out" | tail -20 || true
}

ROOT=/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks
BOX_LOG=/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_box_cleanup_tactile_vae_virtual_s8fs25_flow_70k_20260728
CAN_LOG=/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_can_sort_random_tactile_vae_virtual_s8fs25_flow_70k_20260728

eval_one box_cleanup dexmg_box_cleanup_image_tactile_vae_virtual_s8fs25 \
  "$ROOT/two_arm_box_cleanup" "$ROOT/two_arm_box_cleanup/shard00.hdf5" \
  "$BOX_LOG/models/model_step50000.pt" \
  "$BOX_LOG/eval_rollout_step50000_20eps_nfe2_20260729_rerun" 2
eval_one box_cleanup dexmg_box_cleanup_image_tactile_vae_virtual_s8fs25 \
  "$ROOT/two_arm_box_cleanup" "$ROOT/two_arm_box_cleanup/shard00.hdf5" \
  "$BOX_LOG/models/model_step50000.pt" \
  "$BOX_LOG/eval_rollout_step50000_20eps_nfe4_20260729_rerun" 4
eval_one box_cleanup dexmg_box_cleanup_image_tactile_vae_virtual_s8fs25 \
  "$ROOT/two_arm_box_cleanup" "$ROOT/two_arm_box_cleanup/shard00.hdf5" \
  "$BOX_LOG/models/model_step50000.pt" \
  "$BOX_LOG/eval_rollout_step50000_20eps_nfe8_20260729_rerun" 8

eval_one can_sort_random dexmg_can_sort_random_image_tactile_vae_virtual_s8fs25 \
  "$ROOT/two_arm_can_sort_random" "$ROOT/two_arm_can_sort_random/shard00.hdf5" \
  "$CAN_LOG/models/model_step50000.pt" \
  "$CAN_LOG/eval_rollout_step50000_20eps_nfe2_20260729_rerun" 2
eval_one can_sort_random dexmg_can_sort_random_image_tactile_vae_virtual_s8fs25 \
  "$ROOT/two_arm_can_sort_random" "$ROOT/two_arm_can_sort_random/shard00.hdf5" \
  "$CAN_LOG/models/model_step50000.pt" \
  "$CAN_LOG/eval_rollout_step50000_20eps_nfe4_20260729_rerun" 4
eval_one can_sort_random dexmg_can_sort_random_image_tactile_vae_virtual_s8fs25 \
  "$ROOT/two_arm_can_sort_random" "$ROOT/two_arm_can_sort_random/shard00.hdf5" \
  "$CAN_LOG/models/model_step50000.pt" \
  "$CAN_LOG/eval_rollout_step50000_20eps_nfe8_20260729_rerun" 8
