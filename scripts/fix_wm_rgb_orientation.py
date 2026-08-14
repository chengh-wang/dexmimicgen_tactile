"""Flip WM rollout RGB images vertically to match DexMimicGen HDF5 orientation."""

from __future__ import annotations

import argparse

import h5py


RGB_KEYS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot1_eye_in_hand_image",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path")
    parser.add_argument("--mark-attr", default="rgb_orientation_fixed_vflip")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.h5_path, "r+") as f:
        if f.attrs.get(args.mark_attr, False) and not args.force:
            print(f"[skip] already marked {args.mark_attr}=True: {args.h5_path}")
            return
        data = f["data"]
        for demo_key in sorted(data.keys()):
            low = data[demo_key]["low20"]
            for key in RGB_KEYS:
                ds = low[key]
                ds[...] = ds[:][:, ::-1, :, :]
            data[demo_key].attrs[args.mark_attr] = True
            print(f"[fixed] {demo_key}")
        f.attrs[args.mark_attr] = True
    print(f"[done] {args.h5_path}")


if __name__ == "__main__":
    main()
