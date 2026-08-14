"""Create an RGB+tactile check video from WM rollout HDF5 files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np


RGB_KEYS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot1_eye_in_hand_image",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-h5", required=True)
    parser.add_argument("--failure-h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-success", type=int, default=5)
    parser.add_argument("--num-failure", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2700)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--tactile-reduction", choices=["max", "mean", "last"], default="max")
    return parser.parse_args()


def pick_keys(path: str, prefix: str, count: int, rng: random.Random):
    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f["data"].keys() if k.startswith(prefix))
    if len(keys) < count:
        raise RuntimeError(f"{path} has only {len(keys)} {prefix} trajectories")
    return rng.sample(keys, count)


def resize(img: np.ndarray, scale: int):
    return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def put_label(img: np.ndarray, text: str, xy=(12, 28), scale=0.8):
    cv2.putText(
        img,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def tactile_for_step(tactile: np.ndarray, step: int, reduction: str):
    block = tactile[step].astype(np.float32)
    if reduction == "max":
        return block.max(axis=0)
    if reduction == "mean":
        return block.mean(axis=0)
    return block[-1]


def tactile_panel(tactile4: np.ndarray):
    tiles = []
    for c in range(tactile4.shape[0]):
        arr = np.clip(tactile4[c], 0.0, 1.0)
        gray = (arr * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        heat = resize(heat, 6)
        put_label(heat, f"tactile {c}", xy=(8, 24), scale=0.55)
        tiles.append(heat)
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    panel = np.concatenate([top, bot], axis=0)
    return panel


def make_frame(group, step: int, label: str, key: str, reduction: str):
    rgb = [group["low20"][k][step] for k in RGB_KEYS]
    rgb_row = np.concatenate(rgb, axis=1)
    rgb_row = resize(rgb_row, 3)
    put_label(rgb_row, f"{label}  {key}  step={step:03d}", xy=(12, 30), scale=0.75)

    tactile4 = tactile_for_step(group["high200/tactile"], step, reduction)
    panel = tactile_panel(tactile4)
    put_label(panel, f"high200 tactile {reduction} over 10 samples", xy=(12, 28), scale=0.65)

    width = max(rgb_row.shape[1], panel.shape[1])
    canvas = np.zeros((rgb_row.shape[0] + panel.shape[0] + 10, width, 3), dtype=np.uint8)
    canvas[: rgb_row.shape[0], : rgb_row.shape[1]] = rgb_row
    y = rgb_row.shape[0] + 10
    canvas[y : y + panel.shape[0], : panel.shape[1]] = panel
    return canvas


def append_clip(writer, h5_path: str, key: str, label: str, stride: int, reduction: str):
    with h5py.File(h5_path, "r") as f:
        group = f[f"data/{key}"]
        steps = int(group.attrs["steps"])
        for step in range(0, steps, stride):
            writer.append_data(make_frame(group, step, label, key, reduction))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    success_keys = pick_keys(args.success_h5, "success_", args.num_success, rng)
    failure_keys = pick_keys(args.failure_h5, "failure_", args.num_failure, rng)

    clips = [("SUCCESS", args.success_h5, k) for k in success_keys]
    clips += [("FAILURE", args.failure_h5, k) for k in failure_keys]
    rng.shuffle(clips)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("[selection]")
    for label, _, key in clips:
        print(label, key)
    print(f"[write] {out}")
    with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for label, path, key in clips:
            append_clip(writer, path, key, label, args.stride, args.tactile_reduction)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
