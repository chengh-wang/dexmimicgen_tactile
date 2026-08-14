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
  local max_steps="$8"

  mkdir -p "$out_dir"
  echo "[eval-start] ${name} step50000 nfe=${nfe} episodes=20 max_steps=${max_steps} $(date)"
  /home/labeng/.local/bin/uv run --no-sync python -u examples/eval_dexmg_rollout.py \
    --dataset-path "$data_path" \
    --env-dataset-path "$env_data_path" \
    --task-config "$task_config" \
    --model-path "$model_path" \
    --episodes 20 \
    --nfe "$nfe" \
    --max-episode-steps "$max_steps" \
    --skip-sanity > "$out_dir/eval.out" 2>&1
  echo "[eval-done] ${name} step50000 nfe=${nfe} $(date)"
  grep -E "\\[summary\\]|\\[summary_json\\]" "$out_dir/eval.out" | tail -20 || true
}

ROOT=/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks
LIFT_LOG=/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/imported_06_flow70k_50k_ckpts/lift_tray
POUR_LOG=/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/imported_06_flow70k_50k_ckpts/pouring
TAG=20260729_fixed

for nfe in 2 4 8; do
  eval_one lift_tray dexmg_lift_tray_image_tactile_vae_virtual_s8fs25 \
    "$ROOT/two_arm_lift_tray" "$ROOT/two_arm_lift_tray/shard00.hdf5" \
    "$LIFT_LOG/models/model_step50000.pt" \
    "$LIFT_LOG/eval_rollout_step50000_20eps_nfe${nfe}_${TAG}_max700" "$nfe" 700
done

for nfe in 2 4 8; do
  eval_one pouring dexmg_pouring_image_tactile_vae_virtual_s8fs25 \
    "$ROOT/two_arm_pouring" "$ROOT/two_arm_pouring/shard00.hdf5" \
    "$POUR_LOG/models/model_step50000.pt" \
    "$POUR_LOG/eval_rollout_step50000_20eps_nfe${nfe}_${TAG}" "$nfe" 400
done
