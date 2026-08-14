#!/usr/bin/env bash
set -euo pipefail

cd /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile

STAMP="${1:-20260803_1125}"
SUCCESS="outputs/world_model_rollouts/pouring_wm_250success_low20_high200_20260803_pouring_s250_02_v2.hdf5"
FAILURE="outputs/world_model_rollouts/pouring_wm_250failure_low20_high200_20260803_pouring_f250_07_v2.hdf5"
CACHE="outputs/world_model_rollouts/pouring_official_plus_250s250f_20hz_last_vae50k_mu_cache_${STAMP}.hdf5"
WM_OUT="outputs/pouring_20hz_wm_vae50k_h4_attn_jointpos_bs512_official_plus_250s250f_100k_${STAMP}"
VAE="/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_alltasks_50k_20260728/vae_best.pt"

OFFICIAL0="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/two_arm_pouring/shard00.hdf5"
OFFICIAL1="/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/two_arm_pouring/shard01.hdf5"

echo "[validate] rollout h5"
/home/labeng/.local/bin/uv run --no-sync \
  --project /home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising \
  python - <<PY
import h5py
for p, want_s, want_f in [
    ("$SUCCESS", 250, 0),
    ("$FAILURE", 0, 250),
]:
    with h5py.File(p, "r") as f:
        groups = list(f["data"].keys())
        ns = sum(g.startswith("success_") for g in groups)
        nf = sum(g.startswith("failure_") for g in groups)
        print({"path": p, "groups": len(groups), "success": ns, "failure": nf})
        assert ns == want_s and nf == want_f, (p, ns, nf)
PY

echo "[cache] building $CACHE"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/labeng/.local/bin/uv run --no-sync \
  --project /home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising \
  python scripts/build_dexmg_wm_cache.py \
  --rollout-h5 "$SUCCESS" "$FAILURE" \
  --official-h5 "$OFFICIAL0" "$OFFICIAL1" \
  --vae-ckpt "$VAE" \
  --out "$CACHE" \
  --batch-size 4096 \
  --tactile-channels 12

echo "[train] training WM to $WM_OUT"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/labeng/.local/bin/uv run --no-sync \
  --project /home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising \
  python scripts/train_dexmg_20hz_wm_from_cache.py \
  --cache "$CACHE" \
  --vae-ckpt "$VAE" \
  --out "$WM_OUT" \
  --history 4 \
  --tactile-channels 12 \
  --joint-dim 14 \
  --action-dim 24 \
  --steps 100000 \
  --batch-size 512 \
  --lr 5e-5 \
  --weight-decay 1e-3 \
  --sigreg-weight 0.09 \
  --val-ratio 0.1 \
  --num-workers 8 \
  --save-every 5000 \
  --eval-every 1000 \
  --eval-batches 64 \
  --seed 42

echo "[done] pouring WM pipeline complete"
