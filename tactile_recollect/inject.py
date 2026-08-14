"""
Inject 32x32 piezo tactile taxels into a stored DexMimicGen model XML.

Each taxel is a small oriented **collision box geom** (mass 0) glued to a
fingertip / proximal / palm body, sitting slightly proud of the link surface.
During Path-B replay we force the recorded qpos and call mj_forward (no
integration), so the taxel geoms never push the manipulated objects -- they only
register per-taxel contact forces at the held pose. read_tactile_image() then
reads mj_contactForce for every contact that involves a taxel geom and scatters
the normal force into a (nbodies, 32, 32) image. This reproduces the dense,
spatially-resolved contact of the standalone debug rig; the older <touch>-sensor
approach only lit 1-2 taxels because MuJoCo emits just a couple of contact points
between two convex meshes.

The taxel geoms are contype=0 / conaffinity=1: they collide with the manipulated
objects & environment (contype bit 1) but never with each other, and MuJoCo's
same-body exclusion keeps them off their own link mesh. Adding geoms does not
change nq/nv/na, so the original demo's flattened states load and replay
unchanged.

Also provides the path-bug fix used by the env override: the upstream
edit_model_xml remaps every asset path containing a "dexmimicgen" segment, which
corrupts *valid local* paths when the repo itself is named dexmimicgen. We guard
the remap with os.path.exists so only genuinely-foreign demo paths get rewritten.
"""
import os
import re
import numpy as np
import xml.etree.ElementTree as ET

from .layout import load_layout, N as LAYOUT_N

# fallback taxel box half-size (m) if a body has no per-region `box` in the
# layout (older npz). Real size comes from layout[body]["box"] (0.52 * pitch).
PITCH_MM = 14.7 / 32.0
TAN_HALF = (PITCH_MM / 2.0) / 1000.0
RAD_HALF = 0.0010


def _finger_of(name):
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        if f"_{f}_" in name.lower():
            return f
    return None


def _hand_of(body):
    if "right" in body:
        return "right"
    if "left" in body:
        return "left"
    return "x"


def _region_tag(body):
    """Unique per-region tag so the 11 regions of a hand never collide in a
    sensor/site name: '<finger>_prox' / '<finger>_tip' / 'palm'."""
    if "palm" in body:
        return "palm"
    finger = _finger_of(body) or "x"
    seg = "prox" if "proximal" in body else "tip"   # intermediate/distal -> tip
    return f"{finger}_{seg}"


def _body_tag(body):
    return re.sub(r"[^0-9A-Za-z_]+", "_", body).strip("_")


def _mat2quat(R):
    """3x3 rotation -> MuJoCo (w, x, y, z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def taxel_name(body, i, j):
    return f"tac_{_body_tag(body)}_{i:02d}_{j:02d}"


def _find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None


def fix_asset_paths(root, dexmimicgen_dir):
    """Upstream path remap, but only for paths that don't already resolve."""
    asset = root.find("asset")
    if asset is None:
        return
    path_split = dexmimicgen_dir.rstrip("/").split("/")
    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old_path = elem.get("file")
        if old_path is None or os.path.exists(old_path):
            continue
        sp = old_path.split("/")
        check = [i for i, v in enumerate(sp)
                 if v in ("dexmimicgen", "dexmimicgen_environments")]
        if check:
            elem.set("file", "/".join(path_split + sp[check[-1] + 1:]))


def _ensure_size_memory(root, memory="1G"):
    """Enlarge the contact/constraint arena: thousands of taxel geoms can spike
    the simultaneous-contact count when a link lies flat on a surface. MuJoCo
    forbids `memory` alongside the legacy njmax/nconmax/nstack, so drop those."""
    sz = root.find("size")
    if sz is None:
        sz = ET.SubElement(root, "size")
    for legacy in ("njmax", "nconmax", "nstack"):
        if legacy in sz.attrib:
            del sz.attrib[legacy]
    sz.set("memory", memory)


def inject_tactile(root, layout=None, N=None, group=4, visible=False, visual_scale=1.0):
    """Add per-taxel collision-box geoms to `root` (an ElementTree Element).

    Returns the ordered list of taxel geom names. Bodies in the layout that are
    not present in this XML are silently skipped.
    """
    if layout is None:
        layout, N = load_layout()
    if N is None:
        N = LAYOUT_N

    _ensure_size_memory(root)

    # idempotent: edit_model_xml is also run by the env's xml processor chain, so
    # this can be invoked twice on the same xml. If taxels are already present,
    # just return their names instead of injecting duplicates.
    existing = [g.get("name") for b in root.iter("body")
                for g in b.findall("geom")
                if (g.get("name") or "").startswith("tac_")]
    if existing:
        return existing

    rgba = "0.2 0.5 0.9 1" if visible else "0.2 0.5 0.9 0"
    names = []
    for body in sorted(layout.keys()):
        bel = _find_body(root, body)
        if bel is None:
            continue
        d = layout[body]
        half = float(d.get("box", 0.0)) or TAN_HALF     # cube half-size (m)
        if visible:
            half *= float(visual_scale)
        size = f"{half:.6f} {half:.6f} {half:.6f}"
        for k in range(len(d["pos"])):
            i, j = int(d["ij"][k][0]), int(d["ij"][k][1])
            R = np.stack([d["ctan"][k], d["ax"][k], d["rad"][k]], axis=1)
            q = _mat2quat(R)
            nm = taxel_name(body, i, j)
            ET.SubElement(bel, "geom", {
                "name": nm,
                "type": "box",
                "pos": "{:.6f} {:.6f} {:.6f}".format(*d["pos"][k]),
                "quat": "{:.6f} {:.6f} {:.6f} {:.6f}".format(*q),
                "size": size,
                "mass": "0",
                "contype": "0",           # never initiates; receive-only...
                "conaffinity": "1",       # ...but collides with contype-1 objects
                "condim": "3",
                "friction": "0.9 0.005 0.0001",
                "solref": "0.01 1",
                "solimp": "0.9 0.95 0.001",
                "group": str(group),
                "rgba": rgba,
            })
            names.append(nm)
    return names


def grid_index(layout=None, N=None):
    """Return (bodies, name->(b,i,j) map, N) so sensordata can be scattered into
    a (len(bodies), N, N) image. Body order is sorted(layout)."""
    if layout is None:
        layout, N = load_layout()
    if N is None:
        N = LAYOUT_N
    bodies = sorted(layout.keys())
    bidx = {b: bi for bi, b in enumerate(bodies)}
    mapping = {}
    for body in bodies:
        d = layout[body]
        for k in range(len(d["pos"])):
            i, j = int(d["ij"][k][0]), int(d["ij"][k][1])
            mapping[taxel_name(body, i, j)] = (bidx[body], i, j)
    return bodies, mapping, N
