#!/usr/bin/env python3
"""Build a 20 Hz DexMimicGen WM latent cache from rollout and official tactile HDF5s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from train_dexmg_20hz_wm import TactileVAEProjector


def encode_tactile(
    encoder,
    tactile: np.ndarray,
    device: torch.device,
    batch_size: int,
    tactile_channels: int,
) -> np.ndarray:
    tactile = np.clip(tactile.astype(np.float32), 0.0, 1.0)
    if tactile.ndim != 4 or tactile.shape[1:] != (tactile_channels, 32, 32):
        raise ValueError(f"expected tactile (T,{tactile_channels},32,32), got {tactile.shape}")
    flat = tactile.reshape(-1, 1, 32, 32)
    mu_all = np.empty((flat.shape[0], 16), dtype=np.float32)
    for start in range(0, flat.shape[0], batch_size):
        x = torch.from_numpy(flat[start : start + batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            mu = encoder(x).detach().cpu().numpy()
        mu_all[start : start + mu.shape[0]] = mu
    return mu_all.reshape(tactile.shape[0], tactile_channels, 16)


def write_group(
    out,
    name: str,
    tactile: np.ndarray,
    joint: np.ndarray,
    action: np.ndarray,
    encoder,
    device: torch.device,
    batch_size: int,
    tactile_channels: int,
    attrs: dict,
) -> None:
    n = min(tactile.shape[0], joint.shape[0], action.shape[0])
    tactile = tactile[:n]
    joint = joint[:n].astype(np.float32)
    action = action[:n].astype(np.float32)
    mu = encode_tactile(encoder, tactile, device, batch_size, tactile_channels)
    g = out.create_group(name)
    g.create_dataset("tactile_mu", data=mu, compression="gzip", compression_opts=1)
    g.create_dataset("joint", data=joint, compression="gzip", compression_opts=1)
    g.create_dataset("action", data=action, compression="gzip", compression_opts=1)
    for k, v in attrs.items():
        g.attrs[k] = v


def add_rollout_h5(
    out,
    path: str,
    encoder,
    device: torch.device,
    batch_size: int,
    tactile_channels: int,
) -> int:
    count = 0
    with h5py.File(path, "r") as f:
        root = f["data"] if "data" in f else f
        for name in sorted(root.keys()):
            demo = root[name]
            tactile = demo["high200/tactile"][:, -1].astype(np.float32)
            joint = demo["high200/robot_joint_pos"][:, -1].astype(np.float32)
            action = demo["low20/action"][:].astype(np.float32)
            write_group(
                out,
                name,
                tactile,
                joint,
                action,
                encoder,
                device,
                batch_size,
                tactile_channels,
                {"source": "rollout", "source_path": path, "source_demo": name},
            )
            count += 1
    return count


def add_official_h5s(
    out,
    paths: list[str],
    encoder,
    device: torch.device,
    batch_size: int,
    tactile_channels: int,
    start_index: int,
) -> int:
    count = 0
    next_idx = start_index
    for path in paths:
        with h5py.File(path, "r") as f:
            root = f["data"] if "data" in f else f
            names = sorted(root.keys(), key=lambda x: int(x.split("_")[-1]))
            for demo_name in names:
                demo = root[demo_name]
                obs = demo["obs"]
                tactile = obs["robot0_tactile"][:].astype(np.float32)
                joint = np.concatenate(
                    [obs["robot0_joint_pos"][:], obs["robot1_joint_pos"][:]],
                    axis=-1,
                ).astype(np.float32)
                action = demo["actions"][:].astype(np.float32)
                out_name = f"success_{next_idx:06d}"
                write_group(
                    out,
                    out_name,
                    tactile,
                    joint,
                    action,
                    encoder,
                    device,
                    batch_size,
                    tactile_channels,
                    {
                        "source": "official_tactile",
                        "source_path": path,
                        "source_demo": demo_name,
                    },
                )
                next_idx += 1
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-h5", nargs="+", required=True)
    parser.add_argument("--official-h5", nargs="+", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--tactile-channels", type=int, default=12)
    parser.add_argument("--official-success-start-index", type=int, default=10000)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TactileVAEProjector(
        args.vae_ckpt,
        channels=args.tactile_channels,
    ).encoder.to(device).eval()

    with h5py.File(tmp, "w") as out:
        rollout_count = 0
        for rollout_h5 in args.rollout_h5:
            rollout_count += add_rollout_h5(
                out,
                rollout_h5,
                encoder,
                device,
                args.batch_size,
                args.tactile_channels,
            )
        official_count = add_official_h5s(
            out,
            args.official_h5,
            encoder,
            device,
            args.batch_size,
            args.tactile_channels,
            args.official_success_start_index,
        )
        out.attrs["rollout_count"] = rollout_count
        out.attrs["official_count"] = official_count
        out.attrs["tactile_channels"] = args.tactile_channels
        out.attrs["rate_hz"] = 20
        out.attrs["vae_ckpt"] = args.vae_ckpt

    tmp.rename(out_path)
    meta = vars(args).copy()
    meta.update({"rollout_count": rollout_count, "official_count": official_count})
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
