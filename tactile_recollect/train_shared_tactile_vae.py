#!/usr/bin/env python3
"""Train a shared VAE for DexMimicGen 32x32 tactile patches.

DexMimicGen tactile observations are usually stored as

    obs/robot0_tactile: (T, C, 32, 32)

or, for 200 Hz recollected rollouts, as

    high200/tactile: (T_low, high_steps, C, 32, 32)

where C is embodiment-dependent. This trainer treats every channel as one
sample and learns a shared single-patch encoder/decoder.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PatchEntry:
    path: str
    demo: str
    dataset_key: str
    shape: tuple[int, ...]
    frames: int
    channels: int

    @property
    def patches(self) -> int:
        return self.frames * self.channels


def demo_sort_key(name: str) -> tuple[str, int]:
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return ("_".join(parts[:-1]), int(part))
    return (name, -1)


def nested_dataset(group: h5py.Group, key: str) -> h5py.Dataset | None:
    obj: h5py.Group | h5py.Dataset = group
    for part in key.split("/"):
        if not isinstance(obj, h5py.Group) or part not in obj:
            return None
        obj = obj[part]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"{group.name}/{key} exists but is not a dataset")
    return obj


def shape_to_frames_channels(shape: tuple[int, ...], dataset_name: str) -> tuple[int, int]:
    if len(shape) == 4 and shape[-2:] == (32, 32):
        return int(shape[0]), int(shape[1])
    if len(shape) == 5 and shape[-2:] == (32, 32):
        return int(shape[0] * shape[1]), int(shape[2])
    raise ValueError(f"{dataset_name} has bad shape {shape}; expected (T,C,32,32) or (T,S,C,32,32)")


def discover_entries(paths: list[str], dataset_key: str) -> list[PatchEntry]:
    entries: list[PatchEntry] = []
    for path in paths:
        with h5py.File(path, "r") as f:
            root = f["data"] if "data" in f else f
            demos = sorted(root.keys(), key=demo_sort_key)
            for demo in demos:
                ds = nested_dataset(root[demo], dataset_key)
                if ds is None:
                    continue
                shape = tuple(int(x) for x in ds.shape)
                frames, channels = shape_to_frames_channels(shape, f"{path}:{demo}/{dataset_key}")
                entries.append(PatchEntry(path, demo, dataset_key, shape, frames, channels))
    if not entries:
        raise RuntimeError(f"no {dataset_key} datasets found")
    return entries


def split_entries(entries: list[PatchEntry], val_ratio: float, seed: int) -> tuple[list[PatchEntry], list[PatchEntry]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(entries))
    rng.shuffle(order)
    val_len = max(1, int(round(len(entries) * val_ratio)))
    val_ids = set(int(i) for i in order[:val_len])
    train = [e for i, e in enumerate(entries) if i not in val_ids]
    val = [e for i, e in enumerate(entries) if i in val_ids]
    return train, val


class TactilePatchDataset(Dataset):
    def __init__(
        self,
        entries: list[PatchEntry],
        tactile_scale: float,
        clip_input: bool,
        active_prob: float = 0.0,
        active_threshold: float = 1e-4,
        active_tries: int = 32,
    ) -> None:
        self.entries = entries
        self.tactile_scale = float(tactile_scale)
        self.clip_input = bool(clip_input)
        self.active_prob = float(active_prob)
        self.active_threshold = float(active_threshold)
        self.active_tries = int(active_tries)
        self.cum: list[int] = []
        total = 0
        for e in entries:
            total += e.patches
            self.cum.append(total)
        self._files: dict[str, h5py.File] = {}
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return self.cum[-1]

    def _file(self, path: str) -> h5py.File:
        f = self._files.get(path)
        if f is None:
            f = h5py.File(path, "r")
            self._files[path] = f
        return f

    def _random_entry_index(self) -> int:
        if self._rng is None:
            seed = (os.getpid() * 1000003 + int(time.time_ns() & 0xFFFFFFFF)) & 0xFFFFFFFF
            self._rng = np.random.default_rng(seed)
        return int(self._rng.integers(0, len(self.entries)))

    def _decode_index(self, idx: int) -> tuple[PatchEntry, int, int]:
        entry_i = bisect.bisect_right(self.cum, idx)
        prev = 0 if entry_i == 0 else self.cum[entry_i - 1]
        local = idx - prev
        e = self.entries[entry_i]
        t = local // e.channels
        c = local % e.channels
        return e, int(t), int(c)

    def _read_patch(self, e: PatchEntry, t: int, c: int) -> np.ndarray:
        root = self._file(e.path)
        demo = (root["data"] if "data" in root else root)[e.demo]
        ds = nested_dataset(demo, e.dataset_key)
        if ds is None:
            raise KeyError(f"{e.path}:{e.demo}/{e.dataset_key} disappeared")
        if len(e.shape) == 4:
            x = ds[t, c].astype(np.float32) / self.tactile_scale
        else:
            high_steps = e.shape[1]
            x = ds[t // high_steps, t % high_steps, c].astype(np.float32) / self.tactile_scale
        if self.clip_input:
            x = np.clip(x, 0.0, 1.0)
        return x[None, :, :]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.active_prob > 0.0:
            if self._rng is None:
                seed = (os.getpid() * 1000003 + int(time.time_ns() & 0xFFFFFFFF)) & 0xFFFFFFFF
                self._rng = np.random.default_rng(seed)
            if float(self._rng.random()) < self.active_prob:
                for _ in range(self.active_tries):
                    e = self.entries[self._random_entry_index()]
                    t = int(self._rng.integers(0, e.frames))
                    c = int(self._rng.integers(0, e.channels))
                    x = self._read_patch(e, t, c)
                    if float(x.max()) > self.active_threshold:
                        return torch.from_numpy(x)
        e, t, c = self._decode_index(idx % len(self))
        return torch.from_numpy(self._read_patch(e, t, c))


class ResBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 8))
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PatchEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
            ResBlock2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResBlock2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            ResBlock2d(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu(h), self.logvar(h)


class PatchDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            ResBlock2d(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResBlock2d(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
            ResBlock2d(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.shape[0], 128, 4, 4)
        return self.net(h)


class SharedPatchTactileVAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = PatchEncoder(latent_dim)
        self.decoder = PatchDecoder(latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z), mu, logvar


def loss_fn(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
    active_threshold: float,
    active_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = x > active_threshold
    weight = torch.ones_like(x)
    if active_weight != 1.0:
        weight = torch.where(active, torch.full_like(weight, active_weight), weight)
    recon_mse = torch.mean((recon - x).pow(2) * weight)
    kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    loss = recon_mse + kl_weight * kl
    active_mae = torch.mean(torch.abs(recon[active] - x[active])) if active.any() else torch.zeros((), device=x.device)
    return loss, {
        "weighted_recon_mse": recon_mse.detach(),
        "kl": kl.detach(),
        "mae": torch.mean(torch.abs(recon - x)).detach(),
        "active_mae": active_mae.detach(),
        "active_frac": active.float().mean().detach(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    active_threshold: float,
    active_weight: float,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "weighted_recon_mse": 0.0, "kl": 0.0, "mae": 0.0, "active_mae": 0.0, "active_frac": 0.0}
    n = 0
    for bi, x in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        recon, mu, logvar = model(x)
        loss, parts = loss_fn(recon, x, mu, logvar, kl_weight, active_threshold, active_weight)
        bs = x.shape[0]
        sums["loss"] += float(loss) * bs
        for k, v in parts.items():
            sums[k] += float(v) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def save_recon_grid(model: nn.Module, loader: DataLoader, device: torch.device, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    model.eval()
    x = next(iter(loader))[:8].to(device)
    with torch.no_grad():
        recon, _, _ = model(x)
    x_np = x.cpu().numpy()
    r_np = recon.cpu().numpy()
    fig, axes = plt.subplots(2, 8, figsize=(16, 4), constrained_layout=True)
    for i in range(x_np.shape[0]):
        axes[0, i].imshow(x_np[i, 0], cmap="turbo")
        axes[0, i].set_title(f"gt {i}")
        axes[1, i].imshow(r_np[i, 0], cmap="turbo")
        axes[1, i].set_title("recon")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", nargs="*", default=[])
    parser.add_argument("--h5-glob", default="")
    parser.add_argument("--dataset-key", default="obs/robot0_tactile")
    parser.add_argument("--out", default="runs/dexmg_shared_tactile_vae")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--tactile-scale", type=float, default=300.0)
    parser.add_argument("--clip-input", action="store_true")
    parser.add_argument("--kl-weight", type=float, default=1e-5)
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--active-weight", type=float, default=10.0)
    parser.add_argument("--active-sample-prob", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = [str(Path(p).expanduser()) for p in args.h5]
    if args.h5_glob:
        paths.extend(sorted(glob.glob(args.h5_glob)))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("provide --h5 or --h5-glob")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = discover_entries(paths, args.dataset_key)
    train_entries, val_entries = split_entries(entries, args.val_ratio, args.seed)
    train_ds = TactilePatchDataset(
        train_entries,
        tactile_scale=args.tactile_scale,
        clip_input=args.clip_input,
        active_prob=args.active_sample_prob,
        active_threshold=args.active_threshold,
    )
    val_ds = TactilePatchDataset(
        val_entries,
        tactile_scale=args.tactile_scale,
        clip_input=args.clip_input,
        active_prob=min(1.0, args.active_sample_prob),
        active_threshold=args.active_threshold,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, min(args.num_workers, 4)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = SharedPatchTactileVAE(args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    meta = vars(args).copy()
    meta.update(
        {
            "paths": paths,
            "num_demo_entries": len(entries),
            "train_demo_entries": len(train_entries),
            "val_demo_entries": len(val_entries),
            "train_patches": len(train_ds),
            "val_patches": len(val_ds),
            "device": str(device),
        }
    )
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out_dir / "metrics.jsonl"

    print(json.dumps(meta, indent=2), flush=True)
    it = iter(train_loader)
    start = time.time()
    best_val = math.inf
    for step in range(1, args.steps + 1):
        model.train()
        try:
            x = next(it)
        except StopIteration:
            it = iter(train_loader)
            x = next(it)
        x = x.to(device, non_blocking=True)
        recon, mu, logvar = model(x)
        loss, parts = loss_fn(
            recon,
            x,
            mu,
            logvar,
            args.kl_weight,
            args.active_threshold,
            args.active_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "time_s": time.time() - start,
                "lr": sched.get_last_lr()[0],
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
            }
            row.update({k: float(v) for k, v in parts.items()})
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            val = evaluate(
                model,
                val_loader,
                device,
                args.kl_weight,
                args.active_threshold,
                args.active_weight,
                args.eval_batches,
            )
            ckpt = {
                "model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "config": meta,
                "val": val,
            }
            torch.save(ckpt, out_dir / f"vae_step{step:06d}.pt")
            torch.save(ckpt, out_dir / "vae_latest.pt")
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save(ckpt, out_dir / "vae_best.pt")
                save_recon_grid(model, val_loader, device, out_dir / "recon_best.png")
            print(f"[eval step {step}] {json.dumps(val)}", flush=True)


if __name__ == "__main__":
    main()
