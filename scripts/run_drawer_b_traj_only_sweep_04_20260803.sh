#!/usr/bin/env bash
set -euo pipefail

cd /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile

ROOT="outputs/drawer_cleanup_B_traj_only_sweep20_20260803"
mkdir -p "$ROOT"

DATASET="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/two_arm_drawer_cleanup/shard00.hdf5"
ENV_DATASET="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated/two_arm_drawer_cleanup.hdf5"
MODEL="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_drawer_cleanup_tactile_vae_virtual_s8fs25_flow_100k_20260730_04/models/model_step100000.pt"
PARAMS="outputs/drawer_cleanup_ABC_500demo_100k_20260802/B_basis8_l21e-03/contrastive_params.npz"
WM="outputs/drawer_cleanup_20hz_wm_vae50k_h4_attn_jointpos_bs512_official_plus_250s250f_100k_20260801_1355/wm_step100000.pt"
VAE="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_alltasks_50k_20260728/vae_best.pt"
NORMALIZERS="outputs/drawer_cleanup_20hz_wm_vae50k_h4_attn_jointpos_bs512_official_plus_250s250f_100k_20260801_1355/normalizers.npz"
UV="/home/labeng/.local/bin/uv"
PROJECT="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising"

export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TACTILE_RENDERER=virtual
export TACTILE_VIRTUAL_SIGMA=8.0
export TACTILE_VIRTUAL_FORCE_SCALE=25.0
export TACTILE_VIRTUAL_MAX=1.0
export TACTILE_VIRTUAL_CANONICAL=1
export TACTILE_VIRTUAL_MAX_SURFACE_DIST=0.015
export HDF5_USE_FILE_LOCKING=FALSE

run_one() {
  local name="$1"
  local residual_optimizer="$2"
  local traj_gamma="$3"
  local iter_steps="$4"
  local iter_lr="$5"
  local out_dir="$ROOT/$name"
  mkdir -p "$out_dir"
  echo "[start] $name $(date --iso-8601=seconds)" | tee "$out_dir/status.txt"
  "$UV" run --no-sync --project "$PROJECT" \
    python scripts/rollout_dexmg_with_contrastive_setpoint_residual.py \
    --dataset-path "$DATASET" \
    --env-dataset-path "$ENV_DATASET" \
    --task-config dexmg_drawer_cleanup_image_tactile_vae_virtual_s8fs25 \
    --model-path "$MODEL" \
    --params "$PARAMS" \
    --wm-ckpt "$WM" \
    --vae-ckpt "$VAE" \
    --normalizers "$NORMALIZERS" \
    --out "$out_dir/summary.json" \
    --episodes 20 \
    --nfe 8 \
    --max-episode-steps 400 \
    --history 4 \
    --chunk 20 \
    --trust-delta 0.01 \
    --qp-lambda-u 20 \
    --residual-optimizer "$residual_optimizer" \
    --iter-steps "$iter_steps" \
    --iter-lr "$iter_lr" \
    --score-mode trajectory \
    --traj-gamma "$traj_gamma" \
    --tactile-channels 12 \
    --joint-dim 14 \
    --action-dim 24 \
    --seed 10400 \
    --device cuda \
    --progress-every 0 \
    > "$out_dir/run.log" 2>&1
  echo "[done] $name $(date --iso-8601=seconds)" | tee -a "$out_dir/status.txt"
}

run_one traj_closed_g090 closed_form 0.90 1 0.01 &
run_one traj_closed_g095 closed_form 0.95 1 0.01 &
wait
run_one traj_closed_g098 closed_form 0.98 1 0.01 &
run_one combo_traj_g095_k2_lr0005 iter_value 0.95 2 0.0005 &
wait

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/drawer_cleanup_B_traj_only_sweep20_20260803")
print("baseline,B_basis8_l21e-03,43/50,86.0%")
for p in sorted(root.glob("*/summary.json")):
    s = json.loads(p.read_text())
    print(f"{p.parent.name},{s['success_count']}/{s['episodes_n']},{100*s['success_rate']:.1f}%")
PY
