"""
Render an mp4 for a tactile-augmented demo with a clean two-band layout:

  * TOP: the 3 stored cameras shown BIG and side-by-side, filling the width -
    third-person view (frontview/agentview) + right-wrist + left-wrist.
  * BOTTOM: the 32x32 piezo tactile grids, one row per hand. Each hand shows its
    5 fingertips (thumb->pinky) + palm = 6 tiles. Proximal pads are NOT shown
    (dropped from the layout).

No timeline / gripper / joint-position clutter. Images are shown upright (the
DexMimicGen datasets already store them right-side-up).

Usage:
    python -m tactile_recollect.viz \
        --dataset /tmp/mini_pouring_tac12.hdf5 --demo demo_0 \
        --out /tmp/pouring.mp4
"""
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from .inject import grid_index, _region_tag, _hand_of

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
# tactile tile column order within a hand row: 5 tips then the palm.
COL_ORDER = FINGER_ORDER + ["palm"]

# nice display names for the 3 cameras
CAM_LABEL = {
    "frontview_image": "third-view",
    "agentview_image": "third-view",
    "sideview_image": "third-view",
    "robot0_eye_in_right_hand_image": "right wrist",
    "robot0_eye_in_left_hand_image": "left wrist",
}


def _dilate(img, k):
    """Grey max-filter (radius k) so isolated lit taxels become visible blobs.
    Edge-aware shift (no wrap-around)."""
    if k <= 0:
        return img
    H, W = img.shape
    out = img.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            sy0, sy1 = max(0, dy), H + min(0, dy)
            ty0, ty1 = max(0, -dy), H + min(0, -dy)
            sx0, sx1 = max(0, dx), W + min(0, dx)
            tx0, tx1 = max(0, -dx), W + min(0, -dx)
            out[ty0:ty1, tx0:tx1] = np.maximum(out[ty0:ty1, tx0:tx1],
                                               img[sy0:sy1, sx0:sx1])
    return out


def _disp(img, vmax, gamma, dilate):
    """Normalize a 32x32 force image to [0,1]: clip by vmax, dilate for
    visibility, gamma-brighten low values."""
    d = _dilate(img, dilate)
    n = np.clip(d / (vmax + 1e-9), 0.0, 1.0)
    if gamma != 1.0:
        n = n ** gamma
    return n


def _is_prox(body):
    return _region_tag(body).endswith("_prox")


def _tile_cell(body):
    """(hand, col) inside its hand row. col follows COL_ORDER (tips then palm)."""
    hand = _hand_of(body)
    tag = _region_tag(body)
    if tag == "palm":
        return hand, COL_ORDER.index("palm")
    finger, seg = tag.rsplit("_", 1)
    return hand, COL_ORDER.index(finger)


def render(dataset, demo, out, fps=20, flip_images=False,
           gamma=0.5, pct=97.0, dilate=1, interp="bilinear"):
    f = h5py.File(dataset, "r")
    obs = f[f"data/{demo}/obs"]
    tac = obs["robot0_tactile"][()]                       # (T, 12, 32, 32)
    T = tac.shape[0]
    bodies, _, _ = grid_index()
    bi_of = {b: i for i, b in enumerate(bodies)}
    # tiles = tip + palm bodies only (defensive: skip any stray proximal)
    tile = {b: _tile_cell(b) for b in bodies if not _is_prox(b)}

    front = next((c for c in ("frontview_image", "agentview_image",
                              "sideview_image") if c in obs), None)
    cams = ([front] if front else []) + \
           [c for c in ("robot0_eye_in_right_hand_image",
                        "robot0_eye_in_left_hand_image") if c in obs]
    cam_arrs = {c: obs[c][()] for c in cams}

    # per-region normalization (regions differ by orders of magnitude)
    per_vmax = {}
    for b in tile:
        v = tac[:, bi_of[b]][tac[:, bi_of[b]] > 0]
        per_vmax[b] = max(float(np.percentile(v, pct)) if v.size else 1.0, 1e-3)

    def draw_hand(panel, hand, title, title_color):
        """Fill a 1x6 sub-gridspec (5 tips + palm) for one hand."""
        panel_ax = fig.add_subplot(panel)
        panel_ax.axis("off")
        panel_ax.text(-0.015, 0.5, title, transform=panel_ax.transAxes,
                      fontsize=15, weight="bold", color=title_color,
                      rotation=90, va="center", ha="right")
        gs = panel.subgridspec(1, 6, wspace=0.08)
        for b, (h, col) in tile.items():
            if h != hand:
                continue
            ax = fig.add_subplot(gs[0, col])
            tag = _region_tag(b)
            label = f"{hand[0].upper()} palm" if tag == "palm" \
                else tag.replace("_tip", "")
            ax.imshow(_disp(tac[t, bi_of[b]], per_vmax[b], gamma, dilate),
                      cmap="inferno", vmin=0, vmax=1, interpolation=interp,
                      aspect="equal")
            peak = tac[t, bi_of[b]].max()
            ax.set_title(f"{label}  {peak:.0f}N" if peak > 0 else label,
                         fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])

    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    fig = plt.figure(figsize=(22, 13), dpi=100)

    for t in range(T):
        fig.clear()
        # top band = cameras (tall), bottom = 2 hand rows of tactile
        outer = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.1, 1.1],
                                 hspace=0.18, left=0.04, right=0.99,
                                 top=0.95, bottom=0.02)

        gcam = outer[0].subgridspec(1, max(len(cams), 1), wspace=0.06)
        for ci, c in enumerate(cams):
            ax = fig.add_subplot(gcam[0, ci])
            a = cam_arrs[c][t]
            ax.imshow(a[::-1] if flip_images else a)
            ax.set_title(CAM_LABEL.get(c, c.replace("robot0_", "")), fontsize=16)
            ax.axis("off")

        draw_hand(outer[1], "right", "RIGHT HAND", "tab:blue")
        draw_hand(outer[2], "left", "LEFT HAND", "tab:green")

        fig.suptitle(f"{demo}   frame {t+1}/{T}    peak={tac[t].max():.0f} N",
                     fontsize=16, y=0.99)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
        frame = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame)
    writer.close()
    plt.close(fig)
    f.close()
    print("wrote", out, f"({T} frames @ {fps}fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--flip", action="store_true",
                    help="vertically flip camera images (default: off, upright)")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="display gamma (<1 brightens weak contacts)")
    ap.add_argument("--pct", type=float, default=97.0,
                    help="per-region percentile mapped to full brightness")
    ap.add_argument("--dilate", type=int, default=1,
                    help="max-filter radius so isolated taxels are visible")
    ap.add_argument("--interp", default="bilinear",
                    help="imshow interpolation (nearest|bilinear|gaussian)")
    a = ap.parse_args()
    render(a.dataset, a.demo, a.out, fps=a.fps, flip_images=a.flip,
           gamma=a.gamma, pct=a.pct, dilate=a.dilate, interp=a.interp)
