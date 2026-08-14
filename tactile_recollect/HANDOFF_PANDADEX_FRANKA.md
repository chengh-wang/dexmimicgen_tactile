# HANDOFF — extend 32×32 tactile to PandaDex (Inspire hand) + Franka gripper

**Historical / superseded:** this handoff was written before the final visual
validation pass. It contains outdated setup notes and open TODOs. The current
authoritative tactile placement record is
`tactile_recollect/TACTILE_LAYOUT_VALIDATION.md`.

**Date:** 2026-07-23
**Audience:** the next AI/engineer picking up this task.
**Author of handoff:** previous session (Copilot CLI).

---

## 0. One-paragraph goal

We already have a working 32×32 piezo-style tactile pipeline for the **Fourier
GR1 hand** (3 DexMimicGen tasks: Coffee / Pouring / CanSortRandom). We now want
to add the **same kind of tactile sensing to the other two end-effectors used in
DexMimicGen** so we can eventually train/inference tactile-augmented policies
(cf. ResFiT paper arXiv:2509.19301, which finetunes BC on Coffee/CanSort[GR1] +
BoxCleanup[PandaDex]).

Two new embodiments to instrument:
1. **PandaDex hand = Inspire dexterous hand** — attach taxels *like GR1*: all 5
   fingertips + palm per hand (no proximal pads, matching current
   `INCLUDE_PROX=False`). Used by `TwoArmLiftTray`, `TwoArmBoxCleanup`,
   `TwoArmDrawerCleanup`.
2. **Franka parallel-jaw gripper** — simple: one 32×32 sensor per fingerpad
   (2 pads). Used by `TwoArmThreading`, `TwoArmThreePieceAssembly`,
   `TwoArmTransport`.

This handoff = "attach the sensors" (layout authoring) + get it running on
rob-02. It does NOT include collecting datasets or training (no full datasets
exist locally; see §6).

---

## 1. Where the work lives + machines

- **Mac (dev, env already works):**
  `/Users/chenghao.wang/Documents/git/dexmimicgen/`
  - venv: `tactile_debug/.venv/bin/python` (Python 3.11.15, all deps).
  - Run pipeline modules with:
    `PYTHONPATH=/Users/chenghao.wang/Documents/git/dexmimicgen tactile_debug/.venv/bin/python -m tactile_recollect.<mod>`
- **rob-02 (GPU box, target for heavy runs):** `ssh labeng@10.66.98.143`
  (RTX 6000 Ada, 48 GB, idle). **All code must live under
  `~/workspaces/cwang17_ws/`.**
  - Task folder created: `~/workspaces/cwang17_ws/dexmimicgen_tactile/`
  - Code already rsynced there (662 MB: dexmimicgen + full robosuite
    models/meshes + tactile_recollect). See §3 for exact state.

**Recommended workflow:** author + smoke-test layout on Mac (fast, env works,
meshes present), then `rsync` to rob-02 and run the GPU-heavy extraction/render
there. My edit/view tools only touch the Mac filesystem, so edit on Mac → rsync.

rsync command that was used (openrsync on macOS — do NOT use `--info=progress2`,
it's unsupported):
```bash
cd /Users/chenghao.wang/Documents/git/dexmimicgen
rsync -a \
  --exclude='.git/' --exclude='.venv/' --exclude='**/.venv/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  --exclude='tactile_recollect/data/' --exclude='tactile_debug/tactile_envs/' \
  --exclude='*.hdf5' --exclude='*.h5' --exclude='*.mp4' \
  ./ labeng@10.66.98.143:~/workspaces/cwang17_ws/dexmimicgen_tactile/
```
ssh compound commands (`a && b`) are occasionally flaky on these rob boxes
(random exit 255) — prefer separate ssh calls or a single-quoted script with `;`.

---

## 2. How the existing tactile pipeline works (so you can extend it correctly)

Chain: **layout.py → inject.py → env.py → extract.py → viz.py**

- **`layout.py`** — computes the equal-area 32×32 taxel grid per instrumented
  body by **ray-casting a parametric window onto the real collision mesh**, and
  caches everything to `taxel_layout.npz`. THIS FILE IS FOURIER-SPECIFIC and is
  the main thing you must generalize. Key constants (top of file):
  - `N=32`, `AXIAL_MM=14.7`, `ARC_MM=14.7`, `MARGIN_MM=2.0`
  - `PROUD=0.0015` — taxel box sits 1.5 mm proud of the mesh (protrusion; drives
    penetration→force magnitude in forced replay). **Currently 1.5 mm.**
  - `BOX_FACTOR=0.8` — taxel collision-box half-size = 0.8 × pitch (pitch =
    14.7/32 mm ≈ 0.459 mm → half-size ≈ 0.367 mm; boxes overlap → bigger lit area).
  - `INCLUDE_PROX=False` — proximal pads dropped (rarely contact). Keep False.
  - `REGION_SUFFIX` — list of `(finger, body_suffix, kind)`; `HANDS` — list of
    `(prefix, palm_prefix, standalone_xml, mounted_prefix)`.
  - Generic geometry helpers (REUSE THESE, they're embodiment-agnostic):
    `_mesh_verts_body`, `_fingertip_mesh_tip`, `cyl_taxels` (digit tips/segments),
    `palm_taxels` (planar PCA face), `_pad_normal_world` (flex a finger to find
    the outward pad normal — **actuator-name dependent, see gotcha §4**).
- **`inject.py`** — `inject_tactile(root, layout, N)` adds, for every body in the
  npz, a physical box geom per taxel:
  `<geom type=box mass=0 contype=0 conaffinity=1 condim=3 solref="0.01 1"
  solimp="0.9 0.95 0.001">` glued to the link, sitting `PROUD` above the mesh.
  **This file is DRIVEN BY THE NPZ body keys — mostly embodiment-agnostic.**
  Region/hand tags come from the body NAME via string matching:
  - `_hand_of(body)`: returns "right" if `"right"` in name, "left" if `"left"`.
  - `_region_tag(body)`: `"palm"` if `"palm"` in name; else `"prox"` if
    `"proximal"` in name; else `"tip"`.
  - taxel geom/site name: `tac_{hand}_{region}_{i:02d}_{j:02d}`.
- **`env.py`** — overrides robosuite's `edit_model_xml` to inject taxels; adds
  `robot0_tactile` observable of shape `(nbodies, N, N)`. `read_tactile_image`
  walks `data.contact[]`, calls `mj_contactForce` → 6D wrench, scatters
  `abs(f6[0])` (NORMAL force only) into the per-body 32×32 image. Also
  embodiment-agnostic (uses `load_layout()` body list).
- **`extract.py`** — Path-B replay of a demo (`_reset_to` uses
  `set_state_from_flattened` + `sim.forward()`, NO integration, state overwritten
  each frame → taxel reaction never perturbs the object in OFFLINE replay). Has
  `--resume`. Writes `obs/robot0_tactile` into an augmented HDF5.
- **`viz.py`** — QA "bubble" video. **THIS IS GR1-SPECIFIC** (assumes exactly
  2 hands × 6 tiles = 5 tips + palm, `FINGER_ORDER`/`COL_ORDER`, vertical
  "RIGHT/LEFT HAND" labels, 3 big cameras on top). Layout: `figsize=(22,13)`,
  `add_gridspec(3,1, height_ratios=[1.55,1.1,1.1])`. Default `flip_images=False`
  (DexMimicGen images are stored UPRIGHT — do NOT `[::-1]`). You'll need to
  generalize or fork this for Inspire (still 2 hands × 6 tiles — mostly reusable!)
  and for the Franka gripper (2 pads, one arm — needs a different tile layout).

Bottom line: **to add an embodiment you mostly only touch `layout.py`** (grid
authoring) and **`viz.py`** (display). `inject.py`/`env.py`/`extract.py` should
work unchanged as long as body names carry sane `right/left` + `palm/proximal`
substrings (see caveats §4).

---

## 3. rob-02 setup — EXACT current state (partially done)

Done:
- [x] Folder `~/workspaces/cwang17_ws/dexmimicgen_tactile/` created.
- [x] Code rsynced (662 MB; dexmimicgen 24M + robosuite 629M meshes +
      tactile_recollect 1.3M; inspire_{left,right}_hand.xml, panda_gripper.xml,
      GR1 meshes, and taxel_layout.npz all present).
- [x] venv created: `~/workspaces/cwang17_ws/dexmimicgen_tactile/.venv`
      (Python 3.11.15 via `uv venv --python 3.11`). `uv` is at
      `~/.local/bin/uv`; export `PATH=$HOME/.local/bin:$PATH`.

NOT done / BLOCKED (pick up here):
- [ ] **Core deps install FAILED on a version pin.** `opencv-python==4.13.0.92`
      requires `numpy>=2`, conflicting with the pinned `numpy==1.26.4`. On the
      Mac they coexist only because pip didn't enforce it. **cv2 is NOT used by
      the tactile_recollect pipeline** (verified: no `import cv2` in
      `tactile_recollect/*.py`; only robosuite's optional `opencv_renderer.py`
      imports it, lazily). **Fix:** either drop opencv from the install list, or
      pin a numpy<2-compatible build, e.g. `opencv-python==4.10.0.84`.
      Command to finish the core install (drop/repin opencv):
      ```bash
      ssh labeng@10.66.98.143
      cd ~/workspaces/cwang17_ws/dexmimicgen_tactile
      export PATH=$HOME/.local/bin:$PATH
      VIRTUAL_ENV=$PWD/.venv uv pip install --python .venv/bin/python \
        "mujoco==3.3.0" "numpy==1.26.4" "h5py==3.16.0" "imageio==2.37.3" \
        "imageio-ffmpeg==0.6.0" "matplotlib==3.11.0" "scipy==1.17.1" \
        "opencv-python==4.10.0.84" "Pillow==12.2.0" "termcolor==3.3.0" \
        "glfw==2.10.0" "fsspec==2026.6.0" "huggingface-hub==1.21.0"
      ```
- [ ] **Editable installs** of the local checkouts (do AFTER core deps):
      ```bash
      VIRTUAL_ENV=$PWD/.venv uv pip install --python .venv/bin/python -e robosuite
      VIRTUAL_ENV=$PWD/.venv uv pip install --python .venv/bin/python -e .
      ```
      (`-e .` installs dexmimicgen 0.1 from setup.py at repo root. Watch for extra
      deps their setup.py may pull, e.g. `robomimic` — the tactile pipeline does
      NOT need robomimic, so if it fails on robomimic you can install
      `robosuite`/`dexmimicgen` with `--no-deps` and hand-pick.)
- [ ] **Headless rendering:** rob-02 is a GPU box → use EGL for offscreen mujoco.
      Set `export MUJOCO_GL=egl` (and `PYOPENGL_PLATFORM=egl`) before running
      extract/viz. Verify with a 1-frame offscreen render.
- [ ] **Verify env import + GR1 baseline still works on rob-02** before touching
      new embodiments:
      ```bash
      cd ~/workspaces/cwang17_ws/dexmimicgen_tactile
      export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=egl
      PYTHONPATH=$PWD .venv/bin/python -c "import mujoco, robosuite, dexmimicgen; print('ok')"
      PYTHONPATH=$PWD .venv/bin/python -m tactile_recollect.layout   # rebuild GR1 npz, prints taxel counts
      ```

Local package versions to replicate (from Mac venv):
mujoco 3.3.0 · robosuite 1.5.2 (editable) · dexmimicgen 0.1 (editable) ·
numpy 1.26.4 · h5py 3.16.0 · imageio 2.37.3 · imageio-ffmpeg 0.6.0 ·
matplotlib 3.11.0 · scipy 1.17.1 · Pillow 12.2.0 · termcolor 3.3.0 ·
glfw 2.10.0 · fsspec 2026.6.0 · huggingface-hub 1.21.0 · (opencv optional).

---

## 4. The actual port — model facts already discovered + concrete plan

### 4a. PandaDex = Inspire hand
- `PandaDexRH.default_gripper = {"right": "InspireRightHand"}`,
  `PandaDexLH.default_gripper = {"right": "InspireLeftHand"}`
  (`robosuite/robosuite/models/robots/compositional.py:79-105`). PandaDexRH and
  PandaDexLH are TWO separate single-arm robots (robot0 / robot1) in the bimanual
  envs.
- Standalone XMLs:
  `robosuite/robosuite/models/assets/grippers/inspire_{right,left}_hand.xml`.
- **Inspire RIGHT hand body tree** (left mirrors with `l_`):
  - palm: **`r_palm`** (col mesh `rh_base_link.STL`)
  - thumb: r_thumb_proximal_1 → _2 → r_thumb_middle → **`r_thumb_distal`** (tip)
  - index:  r_index_proximal  → **`r_index_distal`**  (tip)
  - middle: r_middle_proximal → **`r_middle_distal`** (tip)
  - ring:   r_ring_proximal   → **`r_ring_distal`**   (tip)
  - pinky:  r_pinky_proximal  → **`r_pinky_distal`**  (tip)
  - Every link has a `*_col` mesh geom (needed by `_mesh_verts_body`, which looks
    for a geom whose name ends `_col`). ✓
- So with `INCLUDE_PROX=False`, instrument **6 bodies/hand = 5 `*_distal` tips +
  `r_palm`**, exactly the GR1 pattern. `_region_tag` already maps `*_distal`→"tip"
  and `*_palm`→"palm". ✓
- **Proposed layout.py spec for Inspire** (add as an embodiment branch — see 4c):
  ```python
  REGION_SUFFIX_INSPIRE = [   # tips only (INCLUDE_PROX=False)
      ("thumb",  "thumb_distal",  "tip"),
      ("index",  "index_distal",  "tip"),
      ("middle", "middle_distal", "tip"),
      ("ring",   "ring_distal",   "tip"),
      ("pinky",  "pinky_distal",  "tip"),
  ]
  HANDS_INSPIRE = [
      ("r", "r", "inspire_right_hand.xml", <mounted_prefix_RH>),
      ("l", "l", "inspire_left_hand.xml",  <mounted_prefix_LH>),
  ]
  # body   = f"{prefix}_{suffix}"  -> "r_thumb_distal"  ✓
  # palm   = f"{palm_prefix}_palm" -> "r_palm"          ✓
  # finger_bodies for palm normal: r_index_distal, r_middle_distal, r_ring_distal
  ```

### 4b. Franka parallel-jaw gripper
- `robosuite/robosuite/models/assets/grippers/panda_gripper.xml`:
  - fingers: bodies **`leftfinger`**, **`rightfinger`** (col mesh geoms
    `finger1_collision`, `finger2_collision`, mesh `finger.stl`).
  - pad geoms: `finger1_pad_collision`, `finger2_pad_collision`
    (`type=box size="0.008 0.004 0.008"`, i.e. half-sizes 8×4×8 mm) on child
    bodies `finger_joint1_tip`, `finger_joint2_tip`.
- Two pads only → 2 instrumented bodies, one 32×32 grid each on the INNER
  (grasping) face. Simplest robust approach: a **flat planar grid** on the pad's
  inner face (reuse the `palm_taxels` PCA-plane idea, or tile directly on the pad
  box's inner face — no finger-flex normal needed since the inner-face normal is
  a fixed local axis pointing at the opposing finger, local ±y here).
- **IMPORTANT window-size caveat:** the Panda pad is small (~8 mm × 16 mm),
  MUCH smaller than the GR1 `AXIAL_MM=ARC_MM=14.7 mm` window. If you reuse the
  14.7 mm window the grid won't fit the pad. **Parametrize the window per
  embodiment** (e.g. pad ≈ 8 mm wide × 16 mm long) OR just span the pad extent
  directly like `palm_taxels` does (it sizes to the real mesh bbox). Recommended:
  a dedicated `flat_pad_taxels(body, inner_normal_axis)` that spans the pad face.
- `_region_tag("leftfinger")` → "tip" (fine). `_hand_of("leftfinger")`→"left",
  `_hand_of("rightfinger")`→"right" — **but these are the TWO fingers of ONE
  gripper on ONE arm, not two hands.** That's cosmetically wrong for viz's
  hand-split, but functionally OK. Consider renaming region tags for the gripper
  (e.g. `pad_left`/`pad_right`) if viz clarity matters.

### 4c. Recommended refactor shape
`layout.py` is currently single-embodiment. Cleanest: add an `EMBODIMENT`
switch (e.g. `"gr1" | "inspire" | "panda"`) selecting the right
`REGION_SUFFIX` + `HANDS` + tip/palm/pad algorithm + window size, and write a
SEPARATE npz per embodiment (e.g. `taxel_layout_inspire.npz`,
`taxel_layout_panda.npz`). Then `env.py`/`load_layout()` should take a layout
path arg so each task loads the right npz. (Right now `load_layout()` hardcodes
`taxel_layout.npz`; add a param / env-var so GR1 vs Inspire vs Panda envs pick
the matching layout.)

### 4d. GOTCHAS to verify before trusting a new layout
1. **`_pad_normal_world` depends on actuator names** — it flexes actuators whose
   name contains a finger token (`_finger_of`) to measure the outward pad normal.
   VERIFY the Inspire hand's actuators are named per-finger; if not,
   `act_by_finger` is empty → normals default to +z and taxels may land on the
   wrong face. Fallback: derive the pad normal geometrically (mesh PCA / the
   local axis facing the palm) instead of by flex. For the Panda pad, skip flex
   entirely — the inner-face normal is a known local axis.
2. **`_build_standalone` softens `<joint damping>` and `<position kp>`** — assumes
   position actuators exist. Check Inspire/Panda actuator element types; adjust
   if they use `<motor>`/tendons.
3. **Mounted body-name prefix** — GR1 mounts as `gripper0_right_<body>`. For
   PandaDex (robot0=RH, robot1=LH) and Panda gripper the mounted prefix WILL be
   different and may NOT contain "right"/"left". BUILD THE ACTUAL ENV and print
   `[model.body(i).name for i in range(model.nbody)]` to read the true mounted
   names, then set `mounted_prefix` and confirm `_hand_of` still classifies
   correctly (you may need to special-case `_hand_of` for these). The two
   DexMimicGen bimanual PandaDex envs put the hands on robot0/robot1 → likely
   `gripper0_*` / `gripper1_*` prefixes.
4. **Window > mesh** (Panda pad) — see 4b. Size the window to the real pad.

---

## 5. Task/robot map (authoritative, from `dexmimicgen/scripts/demo_random_action.py::ENV_ROBOTS`)

| Task | Robot(s) | End-effector | Instrument with |
|---|---|---|---|
| TwoArmThreading | Panda, Panda | parallel-jaw | Franka gripper layout |
| TwoArmThreePieceAssembly | Panda, Panda | parallel-jaw | Franka gripper layout |
| TwoArmTransport | Panda, Panda | parallel-jaw | Franka gripper layout |
| TwoArmLiftTray | PandaDexRH, PandaDexLH | Inspire hand | Inspire layout |
| TwoArmBoxCleanup | PandaDexRH, PandaDexLH | Inspire hand | Inspire layout |
| TwoArmDrawerCleanup | PandaDexRH, PandaDexLH | Inspire hand | Inspire layout |
| TwoArmCoffee | GR1FixedLowerBody | Fourier hand | **DONE (GR1)** |
| TwoArmPouring | GR1FixedLowerBody | Fourier hand | **DONE (GR1)** |
| TwoArmCanSortRandom | GR1ArmsOnly | Fourier hand | **DONE (GR1)** |

9 tasks, 3 embodiments. GR1 (3 tasks) done. This handoff = Inspire (3 tasks) +
Franka gripper (3 tasks).

---

## 6. Data situation (read before planning extraction/QA)

- **No full DexMimicGen datasets exist on Mac or rob-02.** Only 3
  `/tmp/mini_*.hdf5` GR1 demo_0 files (Coffee/Pouring/CanSort) from earlier HF
  pulls. A prior full download stalled at 575 MB and was abandoned.
- To QA the NEW layouts you have two options:
  1. **No dataset needed** — build the env and run a **random-action rollout**
     (or scripted grasp), confirm the new taxels light up on contact via
     `read_tactile_image`. Sufficient to validate the attachment geometry.
  2. **Real QA video** — fetch one demo_0 for a PandaDex task (e.g.
     `TwoArmBoxCleanup`) and a Panda task (e.g. `TwoArmThreading`) from the HF
     `amandlek/dexmimicgen_datasets` (or the official DexMimicGen dataset repo),
     then run extract + viz. HF fetch on these boxes needed
     `SSL_CERT_FILE=/tmp/ca_combined.pem`; byte-range streaming was very slow —
     prefer full `hf_hub_download` of a single small demo file.
- rob-02 GPU is the place to eventually (a) re-extract tactile across ALL demos
  and (b) train the tactile policy (e.g. ResFiT). CPU-only work (layout authoring)
  can stay on Mac.

---

## 7. Suggested order of operations for the next session

1. Finish rob-02 env (fix opencv pin → core deps → `-e robosuite` → `-e .` →
   `MUJOCO_GL=egl` → verify GR1 `layout.py` rebuild + a 1-frame render). §3.
2. Generalize `layout.py` to Inspire (§4a, §4c). Rebuild `taxel_layout_inspire.npz`,
   eyeball printed taxel counts per body (expect ~hundreds/tip like GR1).
3. Make `load_layout()`/`env.py` accept a layout-path/embodiment arg so the
   Inspire npz is used for PandaDex tasks.
4. Smoke-test: build `TwoArmBoxCleanup` env, print mounted body names (§4d.3),
   fix `mounted_prefix`/`_hand_of` if needed, run a random-action rollout, assert
   `robot0_tactile` fires on contact.
5. Generalize `layout.py` for the Franka pad (§4b) → `taxel_layout_panda.npz`;
   smoke-test on `TwoArmThreading`.
6. Fork/extend `viz.py` for each (Inspire ≈ reuse GR1's 2-hand×6-tile; Panda =
   2-pad single-arm layout).
7. (Later, GPU) fetch demos → extract tactile across datasets → train.

---

## 8. Open decisions inherited from the GR1 phase (ask the user)

- `PROUD` is currently **1.5 mm** in `layout.py` (was 0.6 mm). The GR1
  `taxel_layout.npz` on disk is at 1.5 mm, but the `/tmp/mini_{coffee,can_sort}
  _tac12.hdf5` were extracted at 0.6 mm (stale). Decide whether 1.5 mm is the
  default and re-extract to match. Use the same `PROUD` for the new embodiments
  for consistency.
- Tactile currently uses **NORMAL force only** (`abs(f6[0])`). `mj_contactForce`
  gives 6D (normal + 2 shear + 3 torque). The debug rig
  `tactile_debug/fourier_tactile_full_hand.py` has a validated 3-channel
  (normal + 2 shear) readout if you later want shear.
- **Online dynamics caveat** (matters for policy inference, not offline replay):
  proud taxel geoms form real bilateral contacts. In offline replay
  (`sim.forward()`, state overwritten) they don't perturb anything. But under
  `sim.step()` (online rollout/inference) the protruding taxels WILL perturb
  contact dynamics. Mitigations to evaluate: set `PROUD=0` (flush skin) for the
  online model, soften `solref`, or fall back to MuJoCo native `<touch>` sensors
  for the online policy while keeping the box-geom version for offline QA.

---

## 9. Key file cheat-sheet

| File | Role | Embodiment-specific? |
|---|---|---|
| `tactile_recollect/layout.py` | build taxel grid → npz | **YES — main port target** |
| `tactile_recollect/inject.py` | add box geoms per taxel | no (npz-driven; check `_hand_of`) |
| `tactile_recollect/env.py` | inject + `robot0_tactile` obs + contact-force readout | no (npz-driven; add layout-path arg) |
| `tactile_recollect/extract.py` | Path-B replay → tactile HDF5 (`--resume`) | no |
| `tactile_recollect/viz.py` | QA bubble video | **YES — fork per embodiment** |
| `tactile_debug/fourier_tactile_full_hand.py` | reference rig (3-ch readout source) | GR1 |
| `dexmimicgen/scripts/demo_random_action.py` | `ENV_ROBOTS` map | — |
| `robosuite/.../grippers/inspire_{right,left}_hand.xml` | Inspire model | — |
| `robosuite/.../grippers/panda_gripper.xml` | Franka gripper model | — |
| `robosuite/.../models/robots/compositional.py` | PandaDexRH/LH defs | — |
