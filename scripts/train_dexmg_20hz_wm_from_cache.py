#!/usr/bin/env python3
"""Train a DexMimicGen 20 Hz WM directly from a latent-cache HDF5.

The input cache is flat groups with:

- tactile_mu: (T, C, 16)
- joint: (T, joint_dim), raw robot joint position
- action: (T, action_dim), raw low20 env action

This is for merged caches where some original raw rollout HDF5 files may no
longer be available.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_dexmg_20hz_wm import DexMGWM, SIGReg, batch_to_device


def group_sort_key(name: str) -> tuple[int, int, str]:
    demo = name.split("__")[-1]
    if "_" not in demo:
        return (2, 0, demo)
    prefix, idx = demo.rsplit("_", 1)
    try:
        idx_i = int(idx)
    except ValueError:
        idx_i = 0
    if prefix == "success":
        cls = 0
    elif prefix == "failure":
        cls = 1
    else:
        cls = 2
    return (cls, idx_i, demo)


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.n += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.sum / max(self.n, 1)
        var = self.sumsq / max(self.n, 1) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-8))
        return mean.astype(np.float32), std.astype(np.float32)


def split_groups(cache_path: str, val_ratio: float) -> tuple[list[str], list[str], dict[str, int]]:
    with h5py.File(cache_path, "r") as f:
        groups = sorted(f.keys(), key=group_sort_key)
        lengths = {g: int(f[g]["action"].shape[0]) for g in groups}
    if not groups:
        raise RuntimeError(f"no groups in cache: {cache_path}")
    n_val = int(round(len(groups) * val_ratio))
    n_val = max(1, n_val) if val_ratio > 0 else 0
    train = groups[:-n_val] if n_val else groups
    val = groups[-n_val:] if n_val else []
    return train, val, lengths


def compute_normalizers(
    cache_path: str,
    groups: list[str],
    joint_dim: int,
    action_dim: int,
) -> dict[str, np.ndarray]:
    joint_stats = RunningStats(joint_dim)
    action_stats = RunningStats(action_dim)
    with h5py.File(cache_path, "r") as f:
        for group in groups:
            joint_stats.update(f[group]["joint"][:])
            action_stats.update(f[group]["action"][:])
    joint_mean, joint_std = joint_stats.finish()
    action_mean, action_std = action_stats.finish()
    return {
        "joint_mean": joint_mean,
        "joint_std": joint_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


class CacheWindowDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        groups: list[str],
        lengths: dict[str, int],
        history: int,
        normalizers: dict[str, np.ndarray],
    ) -> None:
        self.cache_path = cache_path
        self.groups = groups
        self.lengths = lengths
        self.history = history
        self.window = history + 1
        self.normalizers = normalizers
        self.refs: list[tuple[str, int]] = []
        for group in groups:
            for start in range(lengths[group] - self.window + 1):
                self.refs.append((group, start))
        if not self.refs:
            raise RuntimeError("no training windows")
        self._cache: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def _file(self) -> h5py.File:
        if self._cache is None:
            self._cache = h5py.File(self.cache_path, "r")
        return self._cache

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group, start = self.refs[idx]
        g = self._file()[group]
        end = start + self.window
        tactile_mu = g["tactile_mu"][start:end].astype(np.float32)
        joint = g["joint"][start:end].astype(np.float32)
        action = g["action"][start:end].astype(np.float32)
        joint = (joint - self.normalizers["joint_mean"]) / self.normalizers["joint_std"]
        action = (action - self.normalizers["action_mean"]) / self.normalizers["action_std"]
        return {
            "tactile_mu": torch.from_numpy(tactile_mu),
            "joint": torch.from_numpy(joint.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
        }


@torch.no_grad()
def evaluate(model: DexMGWM, loader: DataLoader, sigreg: SIGReg, device: torch.device, args) -> dict[str, float]:
    model.eval()
    pred_losses: list[float] = []
    sig_losses: list[float] = []
    for i, batch in enumerate(loader):
        if i >= args.eval_batches:
            break
        batch = batch_to_device(batch, device)
        pred, target = model(batch, args.history)
        pred_losses.append(float(F.mse_loss(pred, target).detach().cpu()))
        sig_losses.append(float(sigreg(target.transpose(0, 1)).detach().cpu()))
    pred_loss = float(np.mean(pred_losses)) if pred_losses else float("nan")
    sig_loss = float(np.mean(sig_losses)) if sig_losses else float("nan")
    return {
        "val_pred_loss": pred_loss,
        "val_sigreg_loss": sig_loss,
        "val_loss": pred_loss + args.sigreg_weight * sig_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--tactile-channels", type=int, default=4)
    parser.add_argument("--joint-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_groups, val_groups, lengths = split_groups(args.cache, args.val_ratio)
    norms = compute_normalizers(
        args.cache,
        train_groups,
        joint_dim=args.joint_dim,
        action_dim=args.action_dim,
    )
    np.savez(out / "normalizers.npz", **norms)

    train_ds = CacheWindowDataset(args.cache, train_groups, lengths, args.history, norms)
    val_ds = CacheWindowDataset(args.cache, val_groups, lengths, args.history, norms) if val_groups else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, min(args.num_workers, 4)),
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    model = DexMGWM(
        args.vae_ckpt,
        history=args.history,
        tactile_channels=args.tactile_channels,
        joint_dim=args.joint_dim,
        action_dim=args.action_dim,
    ).to(device)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    meta = vars(args).copy()
    meta.update(
        {
            "device": str(device),
            "train_demos": len(train_groups),
            "val_demos": len(val_groups),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds) if val_ds is not None else 0,
            "rate_hz": 20,
            "obs": "cached tactile_mu + robot_joint_pos only",
            "action_semantics": f"raw {args.action_dim}D low20 env delta action, normalized from cache",
            "joint_dim": args.joint_dim,
            "action_dim": args.action_dim,
            "tactile_channels": args.tactile_channels,
            "cache_rate_hz": 20,
        }
    )
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out / "metrics.jsonl"
    print(json.dumps(meta, indent=2), flush=True)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda" and amp_dtype is torch.float16)
    it = iter(train_loader)
    start_time = time.time()
    best_val = math.inf
    for step in range(1, args.steps + 1):
        model.train()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
            pred, target = model(batch, args.history)
            pred_loss = F.mse_loss(pred, target)
            sigreg_loss = sigreg(target.transpose(0, 1))
            loss = pred_loss + args.sigreg_weight * sigreg_loss
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        row = None
        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "time_s": time.time() - start_time,
                "lr": sched.get_last_lr()[0],
                "loss": float(loss.detach().cpu()),
                "pred_loss": float(pred_loss.detach().cpu()),
                "sigreg_loss": float(sigreg_loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            }
            print(json.dumps(row), flush=True)

        if val_loader is not None and (step == 1 or step % args.eval_every == 0):
            eval_row = evaluate(model, val_loader, sigreg, device, args)
            if row is None:
                row = {"step": step, "time_s": time.time() - start_time, "lr": sched.get_last_lr()[0]}
            row.update(eval_row)
            print(f"[eval step {step}] {json.dumps(eval_row)}", flush=True)
            if eval_row["val_loss"] < best_val:
                best_val = eval_row["val_loss"]
                torch.save({"model": model.state_dict(), "step": step, "config": meta, "normalizers": norms}, out / "wm_best.pt")

        if row is not None:
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "config": meta, "normalizers": norms}, out / f"wm_step{step:06d}.pt")

    torch.save({"model": model.state_dict(), "step": args.steps, "config": meta, "normalizers": norms}, out / "wm_latest.pt")
    print(f"[done] out={out} best_val={best_val}", flush=True)


if __name__ == "__main__":
    main()
