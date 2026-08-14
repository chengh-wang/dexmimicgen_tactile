# Tactile Layout Validation Record

Date: 2026-07-23

This file records the user-validated final tactile placement for the three
DexMimicGen end-effector embodiments in this workspace. Treat this file as the
authoritative layout note for the current `.npz` files.

## Runtime

Use the project-local uv environment:

```bash
cd /home/labeng/workspaces/cwang17_ws/dexmimicgen_tactile
PYTHONPATH=$PWD/robosuite:$PWD VIRTUAL_ENV=$PWD/.venv \
  .venv/bin/uv run --python .venv/bin/python --no-sync <command>
```

For offscreen MuJoCo rendering, also set:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

## Final Layout Files

| Embodiment | Layout file | Bodies | Taxels | Design area |
|---|---:|---:|---:|---:|
| GR1 / Fourier hand | `tactile_recollect/taxel_layout.npz` | 12 | 11638 | 7818.6 mm^2 |
| PandaDex / Inspire hand | `tactile_recollect/taxel_layout_inspire.npz` | 12 | 11331 | 7841.6 mm^2 |
| Panda parallel gripper | `tactile_recollect/taxel_layout_panda.npz` | 4 | 4096 | 900.0 mm^2 |

The design area is computed from the stored taxel pitch
`pitch = box / BOX_FACTOR`, not from the enlarged preview geometry.

## GR1 / Fourier Hand

Validated placement:

- Fingertips cover the finger pad side.
- Palm uses a large square palmar patch, not full irregular palm coverage.
- Proximal finger pads remain disabled via `INCLUDE_PROX = False`.

Current file:

- `tactile_recollect/taxel_layout.npz`

Validation video:

- `tactile_recollect/out/gr1_tactile_coverage_closeup_squarepalm_10s.mp4`

## PandaDex / Inspire Hand

Validated placement:

- Fingertips cover the finger pad side.
- Palm uses a large square palmar patch, not full irregular palm coverage.
- Thumb placement uses right-hand candidate I and the mirrored left-hand
  candidate J:

```python
thumb_local_axis={"r": (1, 1), "l": (1, -1)}
```

Current file:

- `tactile_recollect/taxel_layout_inspire.npz`

Validation videos:

- `tactile_recollect/out/inspire_thumb_closeup_rightI_leftJ_final_10s.mp4`
- `tactile_recollect/out/inspire_tactile_coverage_closeup_rightI_leftJ_final_10s.mp4`

## Panda Parallel Gripper

Validated placement:

- One 32x32 pad on each inner grasping face.
- The earlier outside-face Panda layout was rejected.
- Final pad normal signs face the gripper gap:

```python
PANDA_PADS = (
    ("finger_joint1_tip", "finger1_pad_collision", 1,  1.0, (0, 2)),
    ("finger_joint2_tip", "finger2_pad_collision", 1, -1.0, (0, 2)),
)
```

- Final pad position is shifted downward by 3.0 mm along the pad vertical axis:

```python
PANDA_PAD_DOWN_MM = 3.0
```

Current file:

- `tactile_recollect/taxel_layout_panda.npz`

Validation videos:

- `tactile_recollect/out/panda_gripper_dolly_inside_candidate_10s.mp4`
- `tactile_recollect/out/panda_gripper_dolly_inside_candidate_down1p5mm_10s.mp4`
- `tactile_recollect/out/panda_gripper_dolly_inside_candidate_down3p0mm_10s.mp4`
- `tactile_recollect/out/panda_gripper_dolly_inside_down3p0mm_final_10s.mp4`

The final accepted Panda version is `down3p0mm`.

## Preview Scale Caveat

The blue boxes in inspection videos are enlarged for visibility, usually with:

```bash
--visual-scale 5
```

This does not change the layout `.npz` taxel positions or true box sizes used by
the tactile pipeline.

## Regeneration Commands

```bash
PYTHONPATH=$PWD/robosuite:$PWD VIRTUAL_ENV=$PWD/.venv \
  .venv/bin/uv run --python .venv/bin/python --no-sync \
  python -m tactile_recollect.layout --embodiment gr1

PYTHONPATH=$PWD/robosuite:$PWD VIRTUAL_ENV=$PWD/.venv \
  .venv/bin/uv run --python .venv/bin/python --no-sync \
  python -m tactile_recollect.layout --embodiment inspire

PYTHONPATH=$PWD/robosuite:$PWD VIRTUAL_ENV=$PWD/.venv \
  .venv/bin/uv run --python .venv/bin/python --no-sync \
  python -m tactile_recollect.layout --embodiment panda
```
