# Action-Rollout Tactile For ThreePiece

This note records the fix for the tactile train/eval mismatch found in
TwoArmThreePieceAssembly.

## Problem

The original tactile shards were generated with `tactile_recollect/extract.py`.
That path is forced-state replay:

1. Load recorded `states[t]` into MuJoCo with `set_state_from_flattened`.
2. Call `sim.forward()`.
3. Read tactile from live contacts with `read_tactile_image(env)`.

This perfectly reproduces the stored tactile HDF5 values, but it does not match
online policy rollout. The injected tactile taxels are real collision boxes. In
forced-state replay, their dynamics are overwritten every frame, so they only act
as readout geometry. In online rollout, the same taxel geoms participate in
MuJoCo contact dynamics and perturb the trajectory.

Measured on ThreePiece demos:

- `forced[t] vs h5[t]`: exact match, max error `0`.
- `action rollout[t] vs h5[t]`: large mismatch.
- `action rollout[t+1] vs h5[t+1]`: still large, so this is not an off-by-one.
- Example over the first 5 demos:
  - forced H5 peak: `45` to `157`
  - online action-rollout peak: usually `3` to `8`
  - tactile-injected action replay state drift p50 around `0.03` to `0.06`
  - plain no-tactile action replay state drift p50 around `2e-4` to `5e-4`

Conclusion: the old tactile labels are self-consistent, but not online-rollout
consistent.

## Fix

Use action replay to regenerate tactile:

1. Reset to the recorded initial state `states[0]`.
2. Read `robot0_tactile[0]`.
3. For frame `t > 0`, call `env.step(actions[t - 1])`.
4. Read tactile from live contacts.
5. Keep the original demo groups and replace/add only `obs/robot0_tactile`.

Script:

```bash
tactile_recollect/extract_action_rollout.py
```

The script stores metadata:

- `data.attrs["tactile_extraction_mode"] = "action_rollout"`
- `data.attrs["tactile_extraction_note"]`
- per-demo `action_rollout_state_err_p50/p95/max`

## Generated Dataset

Dataset generated on 2026-07-25:

```bash
/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_proud2mm_shards/20260725_173000/two_arm_three_piece_assembly
```

Shards:

```bash
shard00.hdf5
shard01.hdf5
shard02.hdf5
```

Stats:

- demos: `1006`
- steps: `239827`
- tactile shape: `(T, 4, 32, 32)`
- mean contact-frame fraction: `0.7507`
- demo peak:
  - mean `13.55`
  - p50 `11.51`
  - p95 `27.55`
  - p99 `47.52`
  - max `84.59`
- frame peak:
  - mean `1.44`
  - p50 `1.11`
  - p95 `4.30`
  - p99 `8.55`
  - max `84.59`
- action-rollout state error p95:
  - mean `0.873`
  - p95 `1.382`
  - max `3.254`

## Generation Command

The full dataset was generated in three shards:

```bash
PYTHONPATH=/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile:/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/robosuite \
MUJOCO_GL=egl \
/home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/.venv/bin/python \
  -m tactile_recollect.extract_action_rollout \
  --dataset /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --out /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile/datasets/generated_tactile_actionrollout_proud2mm_shards/20260725_173000/two_arm_three_piece_assembly/shard00.hdf5 \
  --shard-index 0 \
  --num-shards 3 \
  --resume
```

Repeat with `--shard-index 1` and `--shard-index 2`, writing to `shard01.hdf5`
and `shard02.hdf5`.

## Policy Config

New policy config:

```bash
/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/examples/configs/task/dexmg_three_piece_image_tactile_cnn_actionrollout.yaml
```

Important differences from the old forced-state tactile config:

- `dataset_path` points to `generated_tactile_actionrollout_proud2mm_shards`.
- `tactile_cnn_scale: 100.0`

Reason for scale `100.0`: action-rollout tactile is much lower than forced-state
tactile. Across the new ThreePiece dataset, frame peak p99 is about `8.55` and
max is about `84.59`, so clipping/dividing by `100` preserves signal scale with
some headroom.

## Training Run

Training run:

```bash
/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_three_piece_tactile_cnn_actionrollout_flow_100k_20260725
```

Command is saved at:

```bash
/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_three_piece_tactile_cnn_actionrollout_flow_100k_20260725/run_cmd.sh
```

The training is managed by user systemd:

```bash
systemctl --user status dexmg-threepiece-actionrollout-train.service
```

## 10k Eval

Checkpoint:

```bash
/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_three_piece_tactile_cnn_actionrollout_flow_100k_20260725/models/model_step10000.pt
```

Eval output:

```bash
/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/logs_dexmg_three_piece_tactile_cnn_actionrollout_flow_100k_20260725/eval_rollout_step10000_20eps_20260725/eval.out
```

20 episodes:

| nfe | success | mean reward |
| --- | ---: | ---: |
| 2 | 0.20 | 29.35 |
| 4 | 0.15 | 27.20 |
| 8 | 0.15 | 17.25 |

This is an early checkpoint. The meaningful comparison is expected at 40k and
100k.

## Limitations

This fix makes tactile match online rollout better, but it is not a perfect
multimodal demonstration dataset.

The dataset still keeps recorded RGB, low-dim obs, states, and actions from the
original demos, while tactile is regenerated from action rollout in the
tactile-injected environment. Because tactile taxels perturb dynamics, the
action-rollout state can drift from recorded states. The stored per-demo
`action_rollout_state_err_*` attrs quantify this drift.

So this is best understood as a pragmatic distribution-matching fix for tactile,
not a fully regenerated demo dataset.

Longer-term cleaner options:

- Make online tactile non-perturbing, then forced-state tactile can match online
  dynamics.
- Regenerate the entire demo dataset in the tactile-injected environment.
- Use lower-dimensional robust tactile features such as contact mask, contact
  area, centroid, total normal force, or temporal tactile deltas.
