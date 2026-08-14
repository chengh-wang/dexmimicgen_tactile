"""
Compute equal-area 32x32 tactile taxel layouts and cache them keyed by the
*mounted* body names used inside DexMimicGen environments.

The taxel positions are expressed in each fingertip body's LOCAL frame, which is
identical whether the hand/gripper is standalone or mounted, so we compute the
layout on the cheap standalone XML and inject it into the heavy mounted env
later.

Fourier / Inspire dexterous-hand design:
  * fixed physical window AXIAL_MM x ARC_MM tessellated into N x N taxels, so
    every taxel has the same physical area on every finger;
  * axial: a fixed length ending MARGIN_MM short of the tip;
  * circumferential: a fixed arc length -> per-finger wrap angle
    theta_max = (ARC/2)/r_surf (thinner finger wraps further);
  * each parametric (axial, circumferential) sample is ray-cast onto the real
    collision mesh so the taxel sits on the actual curved pad surface.

Panda gripper design:
  * one flat 32x32 grid on each fingerpad collision box.

Run:
    .venv/bin/python -m tactile_recollect.layout --embodiment gr1
    .venv/bin/python -m tactile_recollect.layout --embodiment inspire
    .venv/bin/python -m tactile_recollect.layout --embodiment panda
"""
import argparse
import os
import numpy as np
import mujoco
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROBO = os.path.abspath(os.path.join(HERE, "..", "robosuite"))
GD = os.path.join(ROBO, "robosuite/models/assets/grippers")

# ---- equal-area design constants (identical to the approved full-hand rig) ----
N = 32
AXIAL_MM = 14.7
ARC_MM = 14.7
MARGIN_MM = 2.0
PROX_HALF_ARC_DEG = 55.0   # proximal patches: cap wrap to the clean palmar belly
PALM_MARGIN_MM = 3.0       # inset from the palm mesh edge (full-face coverage)
PALM_FLAT_BAND_MM = 4.0    # keep taxels within this depth of the flat palm plateau
PALM_SQUARE_MM = 55.0      # Inspire palm patch: large square on the palmar face
PROUD = 0.0020             # taxel sits this far proud of the mesh (local units, m)
BOX_FACTOR = 0.8           # taxel collision-box half-size = BOX_FACTOR * pitch
                           #   (0.52 tiles edge-to-edge; >0.5 overlaps -> larger
                           #    contact area / more taxels light per press)
INCLUDE_PROX = False       # proximal (finger-middle) pads almost never contact in
                           #   the grasp/pinch tasks -> dropped. Set True to restore
                           #   the full 11-region-per-hand layout.
PAD_MARGIN_MM = 0.5        # inset for flat Panda fingerpad grids
PANDA_PAD_DOWN_MM = 3.0    # user-validated downward shift on the inner pad face

# 11 tactile regions per hand = 5 fingertips + 5 proximal (finger-middle) pads +
# the palm. `finger` groups the actuators used to flex that link when deriving its
# palmar normal (None for the palm); `kind` selects the placement algorithm.
#   tip  -> window hugs the distal end, normal = raw kinematic flex normal.
#   prox -> window centred on the link middle (belly), normal = flex direction
#           SNAPPED to the nearest local face axis (orientation-independent; works
#           for both hands and the opposed thumb), arc-capped to the palmar belly.
REGION_SUFFIX_FOURIER = [
    ("thumb",  "thumb_proximal_link",       "prox"),
    ("thumb",  "thumb_distal_link",         "tip"),
    ("index",  "index_proximal_link",       "prox"),
    ("index",  "index_intermediate_link",   "tip"),
    ("middle", "middle_proximal_link",      "prox"),
    ("middle", "middle_intermediate_link",  "tip"),
    ("ring",   "ring_proximal_link",        "prox"),
    ("ring",   "ring_intermediate_link",    "tip"),
    ("pinky",  "pinky_proximal_link",       "prox"),
    ("pinky",  "pinky_intermediate_link",   "tip"),
]

HANDS_FOURIER = [
    # (upper prefix, palm prefix, standalone xml, mounted body prefix)
    ("R", "r", "fourier_right_hand.xml", "gripper0_right_"),
    ("L", "l", "fourier_left_hand.xml", "gripper0_left_"),
]

REGION_SUFFIX_INSPIRE = [
    ("thumb",  "thumb_distal",  "tip"),
    ("index",  "index_distal",  "tip"),
    ("middle", "middle_distal", "tip"),
    ("ring",   "ring_distal",   "tip"),
    ("pinky",  "pinky_distal",  "tip"),
]

HANDS_INSPIRE = [
    ("r", "r", "inspire_right_hand.xml", "gripper0_right_"),
    ("l", "l", "inspire_left_hand.xml",  "gripper1_right_"),
]

PALM_FINGER_SUFFIXES = {
    "gr1": ("index_intermediate_link", "middle_intermediate_link", "ring_intermediate_link"),
    "inspire": ("index_distal", "middle_distal", "ring_distal"),
}

PANDA_MOUNTED_PREFIXES = ("gripper0_right_", "gripper1_right_")
PANDA_PADS = (
    # body, pad geom, local geom-axis sign for the inner grasping face
    ("finger_joint1_tip", "finger1_pad_collision", 1,  1.0, (0, 2)),
    ("finger_joint2_tip", "finger2_pad_collision", 1, -1.0, (0, 2)),
)

DEFAULT_LAYOUT_FILES = {
    "gr1": "taxel_layout.npz",
    "inspire": "taxel_layout_inspire.npz",
    "panda": "taxel_layout_panda.npz",
}


# ---------- geometry helpers (validated in tactile_debug) ----------
def _quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _mesh_verts_body(model, body):
    """Fingertip collision-mesh vertices expressed in the body's local frame."""
    bid = model.body(body).id
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == bid and (model.geom(gid).name or "").endswith("_col"):
            mid = model.geom_dataid[gid]
            if mid < 0:
                break
            adr = model.mesh_vertadr[mid]
            n = model.mesh_vertnum[mid]
            V = model.mesh_vert[adr:adr + n].reshape(-1, 3)
            return (V @ _quat2mat(model.geom_quat[gid]).T) + model.geom_pos[gid]
    return None


def _fingertip_mesh_tip(model, body):
    """Body-frame position of the collision-mesh vertex farthest from the joint
    = the actual rounded fingertip end."""
    Vb = _mesh_verts_body(model, body)
    return Vb[np.argmax(np.linalg.norm(Vb, axis=1))]


def _finger_of(name):
    parts = name.lower().split("_")
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        if f in parts:
            return f
    return None


def _build_standalone(hand_xml):
    """Load a standalone hand/gripper, softened so position servos can flex."""
    tree = ET.parse(os.path.join(GD, hand_xml))
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"autolimits": "true"})
        root.insert(0, compiler)
    compiler.set("meshdir", GD)
    for jnt in root.iter("joint"):
        jnt.set("damping", "0.2")
    for act in root.iter("position"):
        act.set("kp", "20")
    f = f"/tmp/_layout_{os.path.basename(hand_xml)}"
    ET.ElementTree(root).write(f)
    m = mujoco.MjModel.from_xml_path(f)
    d = mujoco.MjData(m)
    return m, d


def _pad_normal_world(model, data, finger, body, tip_vec, act_by_finger):
    """True outward pad normal from a small flex: the fingertip leads along the
    grasp/closing direction (per-finger; the thumb does not face the palm-up +z)."""
    mujoco.mj_resetData(model, data)
    for _ in range(300):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    bid = model.body(body).id
    rest = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ tip_vec
    for _ in range(500):
        data.ctrl[:] = 0.0
        for a in act_by_finger[finger]:
            data.ctrl[a] = 0.4 * model.actuator_ctrlrange[a, 1]
        mujoco.mj_step(model, data)
    flex = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ tip_vec
    v = flex - rest
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0, 0, 1.0])


def cyl_taxels(model, data, body, normalL, axial="tip", half_arc_cap_deg=None):
    """Equal-area conforming cylindrical taxels on a digit segment (ported verbatim
    from the approved tactile_debug/fourier_tactile_full_hand rig).

    axial="tip"    -> window hugs the distal end (fingertips).
    axial="middle" -> window centred on the link's axial middle (the belly between
                      the joints), for the proximal segments.
    Each parametric (axial, arc) sample is ray-cast onto the real collision mesh so
    the taxel sits on the actual curved surface. Rays that graze a groove/edge and
    punch through to the far (dorsal) wall (dist > 30 mm) are rejected, so no taxel
    lands on the back of the link."""
    bid = model.body(body).id
    R = data.xmat[bid].reshape(3, 3)
    o = data.xpos[bid]
    tipv = _fingertip_mesh_tip(model, body)
    axisL = tipv / np.linalg.norm(tipv)
    nL = normalL - np.dot(normalL, axisL) * axisL
    nL /= np.linalg.norm(nL)
    circL = np.cross(axisL, nL)
    gid = np.zeros(1, np.int32)
    tip_a = float(tipv @ axisL)
    if axial == "middle":
        z_c = 0.5 * tip_a
        z_lo = z_c - AXIAL_MM / 2.0 / 1000.0
        z_hi = z_c + AXIAL_MM / 2.0 / 1000.0
    else:
        z_hi = tip_a - MARGIN_MM / 1000.0
        z_lo = z_hi - AXIAL_MM / 1000.0
        z_c = 0.5 * (z_lo + z_hi)
    Vb = _mesh_verts_body(model, body)
    win = np.abs((Vb @ axisL) - z_c) < (AXIAL_MM / 2.0 / 1000.0)
    r_surf = max(float(np.percentile((Vb[win] @ nL), 90)), 1e-3)
    theta_max = (ARC_MM / 2.0 / 1000.0) / r_surf
    if half_arc_cap_deg is not None:
        theta_max = min(theta_max, np.deg2rad(half_arc_cap_deg))
    pos, ij, rad, ax, ctan = [], [], [], [], []
    for i in range(N):
        za = z_lo + (z_hi - z_lo) * i / (N - 1)
        axis_pt = za * axisL
        for j in range(N):
            th = -theta_max + 2 * theta_max * j / (N - 1)
            radL = np.cos(th) * nL + np.sin(th) * circL
            ctanL = -np.sin(th) * nL + np.cos(th) * circL
            startW = o + R @ (axis_pt + radL * 0.03)
            dist = mujoco.mj_ray(model, data, startW, R @ (-radL), None, 1, -1, gid)
            if 0 < dist < 0.03 and gid[0] >= 0 and \
                    model.body(model.geom_bodyid[int(gid[0])]).name == body:
                hitL = (axis_pt + radL * 0.03) - radL * dist + radL * PROUD
                pos.append(hitL)
                ij.append((i, j))
                rad.append(radL)
                ax.append(axisL)
                ctan.append(ctanL)
    pitch = AXIAL_MM / N / 1000.0
    return dict(pos=np.asarray(pos), ij=np.asarray(ij, np.int32),
                rad=np.asarray(rad), ax=np.asarray(ax),
                ctan=np.asarray(ctan)), pitch


def cyl_taxels_between(model, data, body, startL, endL, axial="tip"):
    """Conforming cylindrical taxels spanning the shorter angular sector from
    startL to endL around the digit axis. Used for the Inspire thumb so the patch
    covers the pulp-to-palm side instead of a symmetric pulp-to-back window."""
    bid = model.body(body).id
    R = data.xmat[bid].reshape(3, 3)
    o = data.xpos[bid]
    tipv = _fingertip_mesh_tip(model, body)
    axisL = tipv / np.linalg.norm(tipv)

    aL = startL - np.dot(startL, axisL) * axisL
    aL /= np.linalg.norm(aL) + 1e-9
    bL = endL - np.dot(endL, axisL) * axisL
    bL /= np.linalg.norm(bL) + 1e-9
    theta_end = np.arctan2(np.dot(axisL, np.cross(aL, bL)), np.dot(aL, bL))
    if theta_end > np.pi:
        theta_end -= 2.0 * np.pi
    elif theta_end < -np.pi:
        theta_end += 2.0 * np.pi

    gid = np.zeros(1, np.int32)
    tip_a = float(tipv @ axisL)
    if axial == "middle":
        z_c = 0.5 * tip_a
        z_lo = z_c - AXIAL_MM / 2.0 / 1000.0
        z_hi = z_c + AXIAL_MM / 2.0 / 1000.0
    else:
        z_hi = tip_a - MARGIN_MM / 1000.0
        z_lo = z_hi - AXIAL_MM / 1000.0

    pos, ij, rad, ax, ctan = [], [], [], [], []
    for i in range(N):
        za = z_lo + (z_hi - z_lo) * i / (N - 1)
        axis_pt = za * axisL
        for j in range(N):
            th = theta_end * j / (N - 1)
            radL = np.cos(th) * aL + np.sin(th) * np.cross(axisL, aL)
            radL /= np.linalg.norm(radL) + 1e-9
            ctanL = np.sign(theta_end or 1.0) * np.cross(axisL, radL)
            ctanL /= np.linalg.norm(ctanL) + 1e-9
            startW = o + R @ (axis_pt + radL * 0.03)
            dist = mujoco.mj_ray(model, data, startW, R @ (-radL), None, 1, -1, gid)
            if 0 < dist < 0.03 and gid[0] >= 0 and \
                    model.body(model.geom_bodyid[int(gid[0])]).name == body:
                hitL = (axis_pt + radL * 0.03) - radL * dist + radL * PROUD
                pos.append(hitL)
                ij.append((i, j))
                rad.append(radL)
                ax.append(axisL)
                ctan.append(ctanL)
    pitch = AXIAL_MM / N / 1000.0
    return dict(pos=np.asarray(pos), ij=np.asarray(ij, np.int32),
                rad=np.asarray(rad), ax=np.asarray(ax),
                ctan=np.asarray(ctan)), pitch


def palm_taxels(model, data, body, finger_bodies):
    """Equal-area planar taxels covering the palm's palmar face (PCA plane), ported
    from the approved full-hand rig. Grid sized to the real mesh extent; hits that
    recede below the main plateau (rounded wrist / side edges) are rejected."""
    bid = model.body(body).id
    Vb = _mesh_verts_body(model, body)
    c = Vb.mean(0)
    _, _, Vt = np.linalg.svd(Vb - c, full_matrices=False)
    u, v, nL = Vt[0], Vt[1], Vt[2]
    R = data.xmat[bid].reshape(3, 3)
    o = data.xpos[bid]
    tips_w = np.mean([data.xpos[model.body(b).id] for b in finger_bodies], axis=0)
    to_fingers_L = R.T @ (tips_w - o)
    if np.dot(nL, to_fingers_L) < 0:
        nL = -nL
    pu = (Vb - c) @ u
    pv = (Vb - c) @ v
    mgn = PALM_MARGIN_MM / 1000.0
    umin, umax = pu.min() + mgn, pu.max() - mgn
    vmin, vmax = pv.min() + mgn, pv.max() - mgn
    gid = np.zeros(1, np.int32)
    raw = []
    for i in range(N):
        su = umin + (umax - umin) * i / (N - 1)
        for j in range(N):
            sv = vmin + (vmax - vmin) * j / (N - 1)
            base = c + u * su + v * sv
            startW = o + R @ (base + nL * 0.06)
            dist = mujoco.mj_ray(model, data, startW, R @ (-nL), None, 1, -1, gid)
            if dist > 0 and gid[0] >= 0 and \
                    model.body(model.geom_bodyid[int(gid[0])]).name == body:
                raw.append(dict(i=i, j=j, base=base, depth=0.06 - dist))
    depths = np.array([r["depth"] for r in raw])
    keep_lo = np.median(depths) - PALM_FLAT_BAND_MM / 1000.0
    pos, ij, rad, ax, ctan = [], [], [], [], []
    for r in raw:
        if r["depth"] < keep_lo:
            continue
        pos.append(r["base"] + nL * (r["depth"] + PROUD))
        ij.append((r["i"], r["j"]))
        rad.append(nL); ax.append(u); ctan.append(v)
    pitch = min(umax - umin, vmax - vmin) / N
    return dict(pos=np.asarray(pos), ij=np.asarray(ij, np.int32),
                rad=np.asarray(rad), ax=np.asarray(ax),
                ctan=np.asarray(ctan)), pitch


def square_palm_taxels(model, data, body, finger_bodies, side_mm=PALM_SQUARE_MM):
    """Square patch on the palm face, centered on the flat palm plateau."""
    bid = model.body(body).id
    Vb = _mesh_verts_body(model, body)
    c = Vb.mean(0)
    _, _, Vt = np.linalg.svd(Vb - c, full_matrices=False)
    u, v, nL = Vt[0], Vt[1], Vt[2]
    R = data.xmat[bid].reshape(3, 3)
    o = data.xpos[bid]
    tips_w = np.mean([data.xpos[model.body(b).id] for b in finger_bodies], axis=0)
    to_fingers_L = R.T @ (tips_w - o)
    if np.dot(nL, to_fingers_L) < 0:
        nL = -nL

    pu = (Vb - c) @ u
    pv = (Vb - c) @ v
    mgn = PALM_MARGIN_MM / 1000.0
    umin, umax = pu.min() + mgn, pu.max() - mgn
    vmin, vmax = pv.min() + mgn, pv.max() - mgn

    gid = np.zeros(1, np.int32)
    raw = []
    for i in range(N):
        su = umin + (umax - umin) * i / (N - 1)
        for j in range(N):
            sv = vmin + (vmax - vmin) * j / (N - 1)
            base = c + u * su + v * sv
            startW = o + R @ (base + nL * 0.06)
            dist = mujoco.mj_ray(model, data, startW, R @ (-nL), None, 1, -1, gid)
            if dist > 0 and gid[0] >= 0 and \
                    model.body(model.geom_bodyid[int(gid[0])]).name == body:
                raw.append(dict(su=su, sv=sv, depth=0.06 - dist))
    depths = np.array([r["depth"] for r in raw])
    keep_lo = np.median(depths) - PALM_FLAT_BAND_MM / 1000.0
    plateau = [r for r in raw if r["depth"] >= keep_lo]
    uc = float(np.median([r["su"] for r in plateau]))
    vc = float(np.median([r["sv"] for r in plateau]))

    half = min(side_mm / 2000.0, 0.5 * (umax - umin), 0.5 * (vmax - vmin))
    pos, ij, rad, ax, ctan = [], [], [], [], []
    for i in range(N):
        su = uc - half + 2.0 * half * i / (N - 1)
        for j in range(N):
            sv = vc - half + 2.0 * half * j / (N - 1)
            base = c + u * su + v * sv
            startW = o + R @ (base + nL * 0.06)
            dist = mujoco.mj_ray(model, data, startW, R @ (-nL), None, 1, -1, gid)
            if dist > 0 and gid[0] >= 0 and \
                    model.body(model.geom_bodyid[int(gid[0])]).name == body:
                depth = 0.06 - dist
                if depth < keep_lo:
                    continue
                pos.append(base + nL * (depth + PROUD))
                ij.append((i, j))
                rad.append(nL); ax.append(u); ctan.append(v)
    pitch = 2.0 * half / N
    return dict(pos=np.asarray(pos), ij=np.asarray(ij, np.int32),
                rad=np.asarray(rad), ax=np.asarray(ax),
                ctan=np.asarray(ctan)), pitch


def flat_pad_taxels(model, body, geom_name, normal_axis=1, normal_sign=1.0,
                    tangent_axes=(0, 2)):
    """Planar 32x32 grid on one face of a box geom, expressed in body frame."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"geom not found: {geom_name}")
    if model.body(model.geom_bodyid[gid]).name != body:
        raise ValueError(f"{geom_name} is not attached to {body}")

    Rg = _quat2mat(model.geom_quat[gid])
    center = model.geom_pos[gid].copy()
    size = model.geom_size[gid].copy()
    normal = normal_sign * Rg[:, normal_axis]
    u = Rg[:, tangent_axes[0]]
    v = Rg[:, tangent_axes[1]]
    mgn = PAD_MARGIN_MM / 1000.0
    hu = max(float(size[tangent_axes[0]] - mgn), float(size[tangent_axes[0]]) * 0.5)
    hv = max(float(size[tangent_axes[1]] - mgn), float(size[tangent_axes[1]]) * 0.5)
    face_center = center + normal * (float(size[normal_axis]) + PROUD)

    pos, ij, rad, ax, ctan = [], [], [], [], []
    for i in range(N):
        su = -hu + 2.0 * hu * i / (N - 1)
        for j in range(N):
            sv = -hv + 2.0 * hv * j / (N - 1)
            pos.append(face_center + u * su + v * sv)
            ij.append((i, j))
            rad.append(normal)
            ax.append(v)
            ctan.append(u)
    pitch = min(2.0 * hu, 2.0 * hv) / N
    return dict(pos=np.asarray(pos), ij=np.asarray(ij, np.int32),
                rad=np.asarray(rad), ax=np.asarray(ax),
                ctan=np.asarray(ctan)), pitch


def compute_hand_layout(prefix, palm_prefix, hand_xml, mounted_prefix,
                        region_suffix, palm_finger_suffixes, normal_sign=1.0,
                        square_palm=False, thumb_to_palm=False,
                        thumb_inward_blend=0.0, thumb_sector_to_palm=False,
                        thumb_to_fingers=False, thumb_local_axis=None):
    """Return {mounted_body_name: dict(pos, ij, rad, ax, ctan)} for one hand's 11
    tactile regions (5 fingertips + 5 proximal pads + palm)."""
    m0, d0 = _build_standalone(hand_xml)

    # actuators grouped per finger (to flex one finger at a time)
    act_by_finger = {f: [] for f in ("thumb", "index", "middle", "ring", "pinky")}
    for a in range(m0.nu):
        f = _finger_of(m0.actuator(a).name)
        if f is not None:
            act_by_finger[f].append(a)

    def settle(n=200):
        mujoco.mj_resetData(m0, d0)
        for _ in range(n):
            d0.ctrl[:] = 0.0
            mujoco.mj_step(m0, d0)

    # per-region palmar normal (body-local), derived kinematically so it is
    # independent of how the standalone hand happens to be oriented.
    region_nL = {}
    region_sector = {}
    for finger, suffix, kind in region_suffix:
        if kind == "prox" and not INCLUDE_PROX:
            continue
        body = f"{prefix}_{suffix}"
        tipv = _fingertip_mesh_tip(m0, body)
        nw = _pad_normal_world(m0, d0, finger, body, tipv, act_by_finger)
        settle()
        R = d0.xmat[m0.body(body).id].reshape(3, 3)
        nl = R.T @ nw
        nl /= np.linalg.norm(nl) + 1e-9
        nl *= normal_sign
        if thumb_local_axis is not None and finger == "thumb" and kind == "tip":
            if isinstance(thumb_local_axis, dict):
                axis_idx, axis_sign = thumb_local_axis[prefix]
            else:
                axis_idx, axis_sign = thumb_local_axis
            axisL = tipv / np.linalg.norm(tipv)
            local_dir = np.eye(3)[axis_idx] * axis_sign
            local_dir = local_dir - np.dot(local_dir, axisL) * axisL
            nl = local_dir / (np.linalg.norm(local_dir) + 1e-9)
            region_nL[body] = nl
            continue
        if (thumb_to_palm or thumb_to_fingers) and finger == "thumb" and kind == "tip":
            palm_body = f"{palm_prefix}_palm"
            to_palm = R.T @ (d0.xpos[m0.body(palm_body).id] - d0.xpos[m0.body(body).id])
            axisL = tipv / np.linalg.norm(tipv)
            to_palm = to_palm - np.dot(to_palm, axisL) * axisL
            to_palm = to_palm / (np.linalg.norm(to_palm) + 1e-9)
            oppose = np.mean([
                d0.xpos[m0.body(f"{prefix}_index_distal").id],
                d0.xpos[m0.body(f"{prefix}_middle_distal").id],
            ], axis=0)
            to_oppose = R.T @ (oppose - d0.xpos[m0.body(body).id])
            to_oppose = to_oppose - np.dot(to_oppose, axisL) * axisL
            to_oppose = to_oppose / (np.linalg.norm(to_oppose) + 1e-9)
            if thumb_to_fingers:
                nl = to_oppose
                region_nL[body] = nl
                continue
            if thumb_inward_blend > 0.0:
                nl = (1.0 - thumb_inward_blend) * to_palm + thumb_inward_blend * to_oppose
                nl = nl / (np.linalg.norm(nl) + 1e-9)
            else:
                nl = to_palm
            if thumb_sector_to_palm:
                oppose = np.mean([
                    d0.xpos[m0.body(f"{prefix}_index_distal").id],
                    d0.xpos[m0.body(f"{prefix}_middle_distal").id],
                ], axis=0)
                to_oppose = R.T @ (oppose - d0.xpos[m0.body(body).id])
                to_oppose = to_oppose - np.dot(to_oppose, axisL) * axisL
                to_oppose = to_oppose / (np.linalg.norm(to_oppose) + 1e-9)
                region_sector[body] = (to_oppose, to_palm)
        if kind == "prox":
            # SNAP the flex direction to the nearest local face axis (the flattened
            # proximal link's palmar face is a clean local axis; this removes the
            # base-joint arc tilt and lands the patch squarely on the belly). The
            # long axis is ~local X so it is never selected.
            axisL = tipv / np.linalg.norm(tipv)
            basis = np.eye(3)
            dots = np.abs(basis @ nl)
            dots[int(np.argmax(np.abs(basis @ axisL)))] = -1.0   # exclude long axis
            e = basis[int(np.argmax(dots))]
            nl = np.sign(nl @ e) * e
        region_nL[body] = nl

    # settle at the open pose for the ray-cast layout
    settle()
    finger_bodies = [f"{prefix}_{s}" for s in palm_finger_suffixes]

    out = {}
    for finger, suffix, kind in region_suffix:
        if kind == "prox" and not INCLUDE_PROX:
            continue
        body = f"{prefix}_{suffix}"
        if body in region_sector:
            startL, endL = region_sector[body]
            d, pitch = cyl_taxels_between(m0, d0, body, startL, endL, axial="tip")
        elif kind == "tip":
            d, pitch = cyl_taxels(m0, d0, body, region_nL[body], axial="tip")
        else:  # prox
            d, pitch = cyl_taxels(m0, d0, body, region_nL[body], axial="middle",
                                  half_arc_cap_deg=PROX_HALF_ARC_DEG)
        d["box"] = np.float64(BOX_FACTOR * pitch)  # taxel collision-box half-size
        mounted = mounted_prefix + body
        out[mounted] = d
        print(f"  {mounted:44s} {len(d['pos']):4d}/{N*N} taxels  ({kind})  "
              f"box={BOX_FACTOR*pitch*1000:.2f}mm")

    # palm
    palm_body = f"{palm_prefix}_palm"
    if square_palm:
        d, pitch = square_palm_taxels(m0, d0, palm_body, finger_bodies)
    else:
        d, pitch = palm_taxels(m0, d0, palm_body, finger_bodies)
    d["box"] = np.float64(BOX_FACTOR * pitch)
    mounted = mounted_prefix + palm_body
    out[mounted] = d
    print(f"  {mounted:44s} {len(d['pos']):4d}/{N*N} taxels  (palm)  "
          f"box={BOX_FACTOR*pitch*1000:.2f}mm")
    return out


def compute_panda_layout():
    """Return {mounted fingerpad body: flat 32x32 pad grid} for two Panda arms."""
    m0, _ = _build_standalone("panda_gripper.xml")
    out = {}
    for mounted_prefix in PANDA_MOUNTED_PREFIXES:
        print(f"[{mounted_prefix}] panda_gripper.xml")
        for body, geom, normal_axis, normal_sign, tangent_axes in PANDA_PADS:
            d, pitch = flat_pad_taxels(
                m0, body, geom,
                normal_axis=normal_axis,
                normal_sign=normal_sign,
                tangent_axes=tangent_axes,
            )
            d["pos"] = d["pos"] + d["ax"] * (PANDA_PAD_DOWN_MM / 1000.0)
            d["box"] = np.float64(BOX_FACTOR * pitch)
            mounted = mounted_prefix + body
            out[mounted] = d
            print(f"  {mounted:44s} {len(d['pos']):4d}/{N*N} taxels  (pad)   "
                  f"box={BOX_FACTOR*pitch*1000:.2f}mm")
    return out


def layout_path_for(embodiment):
    return os.path.join(HERE, DEFAULT_LAYOUT_FILES[embodiment])


def build_layout(embodiment="gr1"):
    if embodiment == "gr1":
        layout = {}
        for prefix, palm_prefix, hand_xml, mounted_prefix in HANDS_FOURIER:
            print(f"[{prefix}] {hand_xml}")
            layout.update(compute_hand_layout(
                prefix, palm_prefix, hand_xml, mounted_prefix,
                REGION_SUFFIX_FOURIER, PALM_FINGER_SUFFIXES["gr1"],
                square_palm=True,
            ))
        return layout
    if embodiment == "inspire":
        layout = {}
        for prefix, palm_prefix, hand_xml, mounted_prefix in HANDS_INSPIRE:
            print(f"[{prefix}] {hand_xml}")
            layout.update(compute_hand_layout(
                prefix, palm_prefix, hand_xml, mounted_prefix,
                REGION_SUFFIX_INSPIRE, PALM_FINGER_SUFFIXES["inspire"],
                normal_sign=-1.0,
                square_palm=True,
                thumb_local_axis={"r": (1, 1), "l": (1, -1)},
            ))
        return layout
    if embodiment == "panda":
        return compute_panda_layout()
    raise ValueError(f"unknown embodiment {embodiment!r}")


def build_and_save(path=None, embodiment="gr1"):
    path = path or layout_path_for(embodiment)
    layout = {}
    layout.update(build_layout(embodiment))
    # flatten into a single npz: <body>::<key> arrays + an index of bodies
    save = {"__N__": np.int32(N)}
    bodies = sorted(layout.keys())
    save["__bodies__"] = np.array(bodies)
    for b in bodies:
        for k, v in layout[b].items():
            save[f"{b}::{k}"] = v
    np.savez(path, **save)
    print("wrote", path, "bodies:", len(bodies))
    return path


def load_layout(path=None, embodiment=None):
    """Return {body: dict(pos, ij, rad, ax, ctan, box)} and N."""
    if path is None:
        path = os.environ.get("TACTILE_LAYOUT_PATH")
    if path is None:
        embodiment = embodiment or os.environ.get("TACTILE_EMBODIMENT", "gr1")
        path = layout_path_for(embodiment)
    z = np.load(path, allow_pickle=True)
    N_ = int(z["__N__"])
    bodies = list(z["__bodies__"])
    out = {}
    for b in bodies:
        d = {k: z[f"{b}::{k}"] for k in ("pos", "ij", "rad", "ax", "ctan")}
        bkey = f"{b}::box"
        d["box"] = float(z[bkey]) if bkey in z.files else 0.0
        out[b] = d
    return out, N_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", choices=sorted(DEFAULT_LAYOUT_FILES), default="gr1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    build_and_save(args.output, embodiment=args.embodiment)
