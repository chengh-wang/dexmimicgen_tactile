# DexMimicGen Tactile

This repository is a tactile fork of DexMimicGen for bimanual dexterous
manipulation. It adds virtual tactile observations to the official DexMimicGen
tasks, stores the tactile base-policy handoff artifacts, and supports tactile
world models plus value-gradient residual-action controllers.

Code lives in this GitHub repository:

- https://github.com/chengh-wang/dexmimicgen_tactile

Large datasets, checkpoints, fitted value heads, and handoff manifests live in
the Hugging Face dataset repository:

- https://huggingface.co/datasets/chengh-wang/dexmimicgen_tactile

Do not expect `datasets/`, `outputs/`, or checkpoint files to be tracked by git.
They are intentionally ignored and should be restored from Hugging Face with
selective downloads.

## Scope

The project currently covers the 9 official DexMimicGen tasks:

- `two_arm_box_cleanup`
- `two_arm_can_sort_random`
- `two_arm_coffee`
- `two_arm_drawer_cleanup`
- `two_arm_lift_tray`
- `two_arm_pouring`
- `two_arm_threading`
- `two_arm_three_piece_assembly`
- `two_arm_transport`

The tactile stream is a virtual tactile image stack attached to each policy
rollout. In the current handoff artifacts, high-rate tactile tensors are stored
as 12 sensor maps of size `32 x 32`, usually with shape like
`[low_step, high_substep, 12, 32, 32]`. For example, 20 Hz control with 100 Hz
tactile stores 5 high-rate substeps per control step, and 20 Hz control with
200 Hz tactile stores 10 high-rate substeps per control step.

The internal name `virtual_s8fs25` is the main virtual tactile extraction setup
used for these runs. Treat it as the current default virtual tactile dataset
variant, not as a public-facing task name.

## Repository Layout

- `dexmimicgen/environments/`: DexMimicGen task environments.
- `robosuite/`: synchronized robosuite dependency used by the environments.
- `tactile_recollect/`: virtual tactile layout, extraction, injection,
  visualization, sharded extraction launchers, and shared tactile VAE training.
- `scripts/`: policy evaluation, high-rate rollout collection, world-model
  training, value fitting, residual-action rollout, and experiment utilities.
- `environments.md`: upstream environment documentation.
- `TACTILE_PLAN.md`, `TACTILE_WM_RESIDUAL_ROADMAP.md`: working notes from the
  tactile and world-model experiments.

Large files are excluded by `.gitignore`, including `datasets/`, `outputs/`,
`train_logs/`, `*.hdf5`, `*.pt`, `*.npz`, videos, logs, and archives.

## Installation

The repo assumes Python `>=3.9`. The experimental machines have mostly used
Python 3.11 with MuJoCo/robosuite rendering configured for the local GPU or
headless EGL setup.

Using `uv`:

```bash
git clone https://github.com/chengh-wang/dexmimicgen_tactile.git
cd dexmimicgen_tactile
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e robosuite
uv pip install -e .
```

Using plain `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e robosuite
python3 -m pip install -e .
```

For rendering or simulation playback on headless machines, configure MuJoCo and
OpenGL/EGL according to the local cluster setup before running rollouts.

## Hugging Face Artifacts

HF dataset repo:

```text
chengh-wang/dexmimicgen_tactile
```

At this 2026-08-14 handoff, the dataset repository contains 233 files. The
selected handoff manifest is about 100.55 GiB, and the Drawer Cleanup add-on
manifest is about 10.27 GiB, so use selective downloads instead of cloning
everything by default.

### Top-Level Handoff Files

| HF path | Purpose |
| --- | --- |
| `README.selected_handoff.md` | Human-readable summary for the selected main tactile handoff bundle. |
| `MANIFEST.selected_handoff.json` | Machine-readable manifest for the selected main bundle, 102 files, about 100.55 GiB. |
| `README.drawer_handoff.md` | Drawer Cleanup specific handoff summary. |
| `MANIFEST.drawer_handoff.json` | Drawer Cleanup specific manifest, 80 files, about 10.27 GiB. |

### Main Tactile Dataset

| HF path | Contents |
| --- | --- |
| `datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/` | Main all-task virtual tactile shards for all 9 official DexMimicGen tasks. Each task has sharded HDF5 files and a manifest. |
| `wm/much-ado-about-noising/runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_alltasks_50k_20260728/` | Shared tactile patch VAE trained on the all-task tactile shards. Includes `config.json`, `metrics.jsonl`, `vae_best.pt`, and `vae_latest.pt`. |
| `policies/base_tactile_best/` | Best available tactile base flow policy checkpoint per task, with `metadata.json`, `train_metrics.jsonl`, `eval_best.out`, and a `manifest.json`. |

The shared VAE uses `latent_dim=16`, was trained for 50k steps, and its config
records 9,178 demo entries across the all-task tactile shard set.

### ThreePiece Assembly

| HF path | Contents |
| --- | --- |
| `outputs/high200_interp/threepiece_interp_high200_all_20260806_113436/` | 1006 official ThreePiece demos converted to 200 Hz tactile, joint position, and action using interpolation between 20 Hz ticks. |
| `outputs/world_model_rollouts/threepiece_wm_200success_200failure_low20_high200_20260728.hdf5` | ThreePiece success/failure rollout dataset with both low20 and high200 fields. |
| `outputs/threepiece_high200_methodB_pipeline_20260806_121858/rollout_shards/` | ThreePiece success/failure rollout shards used for high-rate Method B experiments. |
| `outputs/threepiece_oldcache_aligned_compare_20260808_1229/wm_250s250f_100k_aligned/` | Old/aligned 20 Hz ThreePiece WM artifacts: `config.json`, `normalizers.npz`, `wm_best.pt`, `wm_latest.pt`. |
| `outputs/threepiece_oldcache_aligned_compare_20260808_1229/fit_identity_50s50f_late20_earlyneg/` | 20 Hz identity contrastive value head parameters and metrics. |
| `outputs/threepiece_oldcache_aligned_compare_20260808_1229/fit_tail25_basis8_l2e-3_250s250f/` | 20 Hz tail/progress value head parameters and metrics. |
| `outputs/threepiece_fair_newvae_rate20_vs_existing_rate200_20260810_200fair/rate20/` | Fair 20 Hz downsample pipeline from the 200 Hz source data, with WM and identity/tail value heads. |
| `outputs/threepiece_multirate_aligned_identity_tail_20260809_overnight/rate40/` | 40 Hz aligned ThreePiece WM and identity/tail value heads. |
| `outputs/threepiece_multirate_aligned_identity_tail_20260809_overnight/rate100/` | 100 Hz aligned ThreePiece WM and identity/tail value heads. |
| `outputs/threepiece_multirate_aligned_identity_tail_20260809_overnight/rate200/` | 200 Hz aligned ThreePiece WM and identity/tail value heads. |
| `outputs/tactile_vae_smoke_high200_current/` | Current high200 tactile VAE smoke/current checkpoint bundle: `config.json`, `vae_best.pt`, `vae_latest.pt`. |

The ThreePiece value-head directories mainly contain `contrastive_params.npz`,
`config.json`, and metrics files. Those parameter files are the important
handoff artifacts for residual-action inference.

### Drawer Cleanup

| HF path | Contents |
| --- | --- |
| `outputs/world_model_rollouts/drawer_cleanup_wm_250success_250failure_low20_high200_20260801_1355.hdf5` | Core Drawer Cleanup rollout dataset, 250 success plus 250 failure demos, with low20 and high200 fields. |
| `outputs/world_model_rollouts/drawer_cleanup_official_plus_250s250f_20hz_last_vae50k_mu_cache_20260801_1355.hdf5` | Drawer 20 Hz tactile-VAE latent cache. |
| `outputs/drawer_cleanup_20hz_wm_vae50k_h4_attn_jointpos_bs512_official_plus_250s250f_100k_20260801_1355/` | Drawer 20 Hz WM artifacts, including `config.json`, `metrics.jsonl`, `normalizers.npz`, `wm_best.pt`, `wm_latest.pt`, and selected step checkpoints. |
| `outputs/drawer_cleanup_ABC_500demo_100k_20260802/` | Drawer A/B/C contrastive value-head sweep parameters and metrics. |
| `outputs/drawer_cleanup_B_progress_logistic_500demo_20260801_1355/B_basis8_l2e-4/` | Drawer B progress value head for the 100k WM. |
| `outputs/drawer_cleanup_B_progress_logistic_500demo_70k_20260801_1355/B_basis8_l2e-4/` | Drawer B progress value head for the 70k WM. |
| `outputs/policy_ckpts/drawer_cleanup_flow100k_20260730_04_model_step100000.pt` | Drawer Cleanup base tactile flow policy checkpoint. |

### Box Cleanup

| HF path | Contents |
| --- | --- |
| `datasets/residual_rollouts/two_arm_box_cleanup/high100_low20_250s250f_20260811/` | Box Cleanup residual-rollout dataset collected at 20 Hz control and 100 Hz tactile/joint/action. |

The Box Cleanup high100 rollout bundle contains 500 demos total: 250 success,
250 failure, split into 3 shards from `wilm-rob-02`, `wilm-rob-04`, and
`wilm-rob-07`. Each episode is 20 seconds, with 400 low20 steps and 5 high100
substeps per low20 step. The main high-rate tensors include:

- `high100/tactile`: `[400, 5, 12, 32, 32]`, `float16`
- `high100/action`: `[400, 5, 24]`, `float32`
- `high100/robot_joint_pos`: `[400, 5, 14]`, `float32`

Box Cleanup WM/value-head artifacts are not yet part of the HF handoff bundle.
The available Box artifacts are the all-task tactile shards, the selected base
policy under `policies/base_tactile_best/`, and the 100 Hz residual-rollout
dataset above.

## Selected Base Policy Checkpoints

The HF directory `policies/base_tactile_best/` contains the best selected tactile
base flow policy checkpoint per task. The current manifest records:

| Task | Checkpoint | Eval |
| --- | --- | --- |
| `two_arm_box_cleanup` | `model_step90000.pt` | 13/20 = 65.0%, NFE 8 |
| `two_arm_can_sort_random` | `model_step20000.pt` | 13/20 = 65.0%, NFE 8 |
| `two_arm_coffee` | `model_step100000.pt` | 0/50 = 0.0%, NFE 4 |
| `two_arm_drawer_cleanup` | `model_step100000.pt` | 43/50 = 86.0%, NFE 8 |
| `two_arm_lift_tray` | `model_step60000.pt` | 5/20 = 25.0%, NFE 16 |
| `two_arm_pouring` | `model_step50000.pt` | 14/20 = 70.0%, NFE 16 |
| `two_arm_threading` | `model_step90000.pt` | 8/20 = 40.0%, NFE 4 |
| `two_arm_three_piece_assembly` | `model_step50000.pt` | 19/20 = 95.0%, NFE 4 |
| `two_arm_transport` | `model_step90000.pt` | 1/20 = 5.0%, NFE 4 |

These numbers are manifest snapshots from the policy-selection runs. Compare
new results only against the same task, checkpoint, NFE, simulator setup, and
evaluation protocol.

## Download Examples

Install the HF client:

```bash
uv pip install huggingface_hub
```

Download only the base policy bundle:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="chengh-wang/dexmimicgen_tactile",
    repo_type="dataset",
    local_dir="hf_data",
    allow_patterns=["policies/base_tactile_best/**"],
)
PY
```

Download the all-task tactile shards:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="chengh-wang/dexmimicgen_tactile",
    repo_type="dataset",
    local_dir="hf_data",
    allow_patterns=[
        "datasets/generated_tactile_actionrollout_virtual_s8fs25_alltasks_shards/20260727_0828_s8fs25_alltasks/**",
    ],
)
PY
```

Download ThreePiece 20/40/100/200 Hz WM and value artifacts:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="chengh-wang/dexmimicgen_tactile",
    repo_type="dataset",
    local_dir="hf_data",
    allow_patterns=[
        "outputs/threepiece_oldcache_aligned_compare_20260808_1229/**",
        "outputs/threepiece_fair_newvae_rate20_vs_existing_rate200_20260810_200fair/rate20/**",
        "outputs/threepiece_multirate_aligned_identity_tail_20260809_overnight/rate40/**",
        "outputs/threepiece_multirate_aligned_identity_tail_20260809_overnight/rate100/**",
        "outputs/threepiece_multirate_aligned_identity_tail_20260809_overnight/rate200/**",
    ],
)
PY
```

Download Box Cleanup 100 Hz residual rollouts:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="chengh-wang/dexmimicgen_tactile",
    repo_type="dataset",
    local_dir="hf_data",
    allow_patterns=[
        "datasets/residual_rollouts/two_arm_box_cleanup/high100_low20_250s250f_20260811/**",
    ],
)
PY
```

If the dataset becomes private later, run `hf auth login` first or set
`HF_TOKEN` in the environment.

## Main Code Entry Points

Base policy config and evaluation:

- `scripts/generate_training_config.py`
- `scripts/rollout_policy_base_eval.py`
- `scripts/run_dexmg_eval_sweep.py`
- `scripts/supervise_01_coffee_drawer_flow70k_eval50.py`
- `scripts/supervise_02_box_can_flow70k_eval50.py`

Virtual tactile generation:

- `tactile_recollect/extract_action_rollout.py`
- `tactile_recollect/launch_sharded_tactile_extract.py`
- `scripts/launch_virtual_tactile_alltasks.py`
- `scripts/render_virtual_tactile_demo.py`
- `scripts/render_virtual_tactile_online_demo.py`

Tactile VAE:

- `tactile_recollect/train_shared_tactile_vae.py`
- `tactile_recollect/watch_virtual_s8fs25_shards_then_train_vae.py`

High-rate rollout/cache generation:

- `scripts/extract_high200_interp_action_demo.py`
- `scripts/extract_high200_interp_action_shard.py`
- `scripts/launch_threepiece_high200_interp_all.sh`
- `scripts/collect_dexmg_wm_rollouts.py`
- `scripts/collect_threepiece_wm_rollouts.py`
- `scripts/run_box_cleanup_collect_high100.sh`
- `scripts/run_drawer_collect_250_250.sh`

World-model training:

- `scripts/train_dexmg_20hz_wm.py`
- `scripts/train_dexmg_20hz_wm_from_cache.py`
- `scripts/train_threepiece_20hz_wm.py`
- `scripts/train_threepiece_20hz_wm_from_cache.py`
- `scripts/train_threepiece_high200_wm.py`

Value fitting and residual-action inference:

- `scripts/fit_dexmg_abc_setpoints.py`
- `scripts/fit_dexmg_progress_timevariant_setpoints.py`
- `scripts/fit_threepiece_tail_progress_linear_value.py`
- `scripts/rollout_dexmg_with_contrastive_setpoint_residual.py`
- `scripts/rollout_policy_with_contrastive_setpoint_residual.py`
- `scripts/rollout_policy_with_high200_contrastive_setpoint_residual.py`
- `scripts/probe_threepiece_mppi_vs_grad.py`
- `scripts/probe_threepiece_mppi_vs_grad_demo_cache.py`

Some older scripts contain absolute paths from the lab machines
(`/home/labeng/workspaces/cwang17_ws/...`). When moving to a new machine, update
the dataset/checkpoint arguments or hard-coded defaults before launching.

The exact flow-policy training stack that produced the `policies/base_tactile_best`
checkpoints is represented here by configs, evaluation scripts, logs/metrics, and
HF checkpoints; verify the trainer dependency before attempting a from-scratch
policy retrain on a fresh machine.

## Experimental Status

Current handoff status:

- All-task virtual tactile shards: available on HF.
- Shared all-task tactile VAE: available on HF.
- Best tactile base policies for all 9 tasks: available on HF.
- ThreePiece Assembly 200 Hz data, multi-rate WMs, and identity/tail value
  heads: available on HF.
- Drawer Cleanup rollout data, 20 Hz WM, value heads, and base policy: available
  on HF.
- Box Cleanup 100 Hz success/failure rollout data: available on HF.
- Box Cleanup WM/value heads: not yet included in the HF handoff bundle.
- Original non-tactile official DexMimicGen `datasets/generated/` mirror: not
  included here; use the original DexMimicGen release if needed.

For ThreePiece residual-control experiments, the strongest 20 Hz gradient-based
value residual runs were the identity contrastive residual and the tail/progress
value residual. The value-head artifacts are stored in the ThreePiece directories
listed above; use the matching WM, normalizers, value parameters, base policy,
and evaluation harness when reproducing those numbers.

## Notes for Reproduction

- Keep code and artifacts separate: git for source, HF for HDF5/checkpoints.
- Use `allow_patterns` when downloading from HF. Pulling the whole dataset repo
  can exceed 100 GiB.
- Preserve the original HF directory structure under a local artifact root when
  possible. Many scripts assume the historical `datasets/...` and `outputs/...`
  layout.
- Check each `config.json`, `metadata.json`, and manifest before mixing WMs,
  normalizers, value heads, and VAE checkpoints across rates or tasks.
- For high-rate experiments, the rate is encoded in the stored group name
  (`high100`, `high200`) and in the manifest. Do not infer it only from the
  parent directory name.

## Upstream Attribution

This fork builds on DexMimicGen:

- Website: https://dexmimicgen.github.io
- Paper: https://arxiv.org/abs/2410.24185

The original DexMimicGen code is released under the NVIDIA Source Code License,
and the original datasets are released under CC-BY 4.0. See `LICENSE` and the
upstream project for the base terms. The tactile datasets and checkpoints in the
HF repo are handoff artifacts for this tactile fork.

If you use the DexMimicGen environments or generated demonstrations, cite the
DexMimicGen paper:

```bibtex
@inproceedings{jiang2024dexmimicgen,
  title     = {DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning},
  author    = {Jiang, Zhenyu and Xie, Yuqi and Lin, Kevin and Xu, Zhenjia and Wan, Weikang and Mandlekar, Ajay and Fan, Linxi and Zhu, Yuke},
  booktitle = {2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```
