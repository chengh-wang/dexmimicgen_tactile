#!/usr/bin/env python3
"""Fit a time-conditioned success-likelihood MLP on frozen 20 Hz WM latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from train_threepiece_contrastive_setpoint_value import (  # noqa: E402
    binary_auc,
    encode_group_latents,
    is_failure_group,
    is_success_group,
    list_cache_groups,
    load_normalizers,
    load_wm,
    split_groups,
)


def progress_array(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    return (np.arange(n, dtype=np.float32) / float(n - 1)).astype(np.float32)


def rbf_features(tau: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    phi = torch.exp(-0.5 * torch.square((tau[:, None] - centers[None]) / max(width, 1e-4)))
    return phi / torch.clamp(phi.sum(dim=-1, keepdim=True), min=1e-6)


class ProgressSuccessMLP(nn.Module):
    def __init__(self, z_dim: int, basis: int, hidden: int) -> None:
        super().__init__()
        self.fc0 = nn.Linear(z_dim + basis, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, z_norm: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_norm, phi], dim=-1)
        h = F.gelu(self.fc0(x))
        h = F.gelu(self.fc1(h))
        return self.out(h).squeeze(-1)


def collect_latents(args, wm, normalizers, groups, device):
    chunks, taus, labels = [], [], []
    with h5py.File(args.cache, "r") as f:
        for key in groups:
            z = encode_group_latents(wm, f[key], normalizers, device, args.encode_batch_steps)
            tau = progress_array(z.shape[0])
            if is_success_group(key):
                y = np.ones((z.shape[0],), dtype=np.float32)
            elif is_failure_group(key):
                y = np.zeros((z.shape[0],), dtype=np.float32)
            else:
                continue
            chunks.append(z.astype(np.float32))
            taus.append(tau.astype(np.float32))
            labels.append(y)
    return (
        np.concatenate(chunks, axis=0).astype(np.float32),
        np.concatenate(taus, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.float32),
    )


def metrics_from_scores(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pos = scores[y > 0.5]
    neg = scores[y < 0.5]
    return {
        "auc": binary_auc(scores.tolist(), y.astype(int).tolist()),
        "score_mean_pos": float(pos.mean()) if pos.size else float("nan"),
        "score_mean_neg": float(neg.mean()) if neg.size else float("nan"),
        "prob_mean_pos": float((1.0 / (1.0 + np.exp(-pos))).mean()) if pos.size else float("nan"),
        "prob_mean_neg": float((1.0 / (1.0 + np.exp(-neg))).mean()) if neg.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--basis", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=1.0,
        help="Scale logits during BCE training and bake the same scale into saved final-layer weights.",
    )
    parser.add_argument("--fit-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--val-demos-per-class", type=int, default=20)
    parser.add_argument("--encode-batch-steps", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    normalizers = load_normalizers(args.normalizers)
    wm = load_wm(args, device)
    success, failure, _ = list_cache_groups(args.cache)
    train_success, val_success = split_groups(success, args.val_demos_per_class)
    train_failure, val_failure = split_groups(failure, args.val_demos_per_class)

    z_train, tau_train, y_train = collect_latents(args, wm, normalizers, train_success + train_failure, device)
    z_val, tau_val, y_val = collect_latents(args, wm, normalizers, val_success + val_failure, device)

    z_mean = z_train.mean(axis=0).astype(np.float32)
    z_std = np.maximum(z_train.std(axis=0).astype(np.float32), 1e-6)
    zt = torch.from_numpy((z_train - z_mean) / z_std).to(device)
    taut = torch.from_numpy(tau_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    zv = torch.from_numpy((z_val - z_mean) / z_std).to(device)
    tauv = torch.from_numpy(tau_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    centers = torch.linspace(0.0, 1.0, args.basis, device=device)
    width = 1.0 / max(args.basis - 1, 1) * 1.5
    z_dim = int(z_train.shape[1])
    model = ProgressSuccessMLP(z_dim, args.basis, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    n = int(zt.shape[0])
    for step in range(1, args.fit_steps + 1):
        bi_np = rng.integers(0, n, size=min(args.batch_size, n), endpoint=False)
        bi = torch.as_tensor(bi_np, dtype=torch.long, device=device)
        logits = model(zt[bi], rbf_features(taut[bi], centers, width)) * float(args.logit_scale)
        loss = F.binary_cross_entropy_with_logits(logits, yt[bi])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu())}), flush=True)

    with torch.no_grad():
        train_logits = []
        for start in range(0, n, 65536):
            logits = model(zt[start : start + 65536], rbf_features(taut[start : start + 65536], centers, width))
            logits = logits * float(args.logit_scale)
            train_logits.append(logits.detach().cpu().numpy())
        val_logits = []
        for start in range(0, int(zv.shape[0]), 65536):
            logits = model(zv[start : start + 65536], rbf_features(tauv[start : start + 65536], centers, width))
            logits = logits * float(args.logit_scale)
            val_logits.append(logits.detach().cpu().numpy())
    train_scores = np.concatenate(train_logits, axis=0).astype(np.float32)
    val_scores = np.concatenate(val_logits, axis=0).astype(np.float32)

    np.savez(
        out / "contrastive_params.npz",
        param_type_code=np.array(13, dtype=np.float32),
        basis_centers=centers.detach().cpu().numpy().astype(np.float32),
        basis_width=np.array(width, dtype=np.float32),
        mlp_z_mean=z_mean,
        mlp_z_std=z_std,
        mlp_w0=model.fc0.weight.detach().cpu().numpy().astype(np.float32),
        mlp_b0=model.fc0.bias.detach().cpu().numpy().astype(np.float32),
        mlp_w1=model.fc1.weight.detach().cpu().numpy().astype(np.float32),
        mlp_b1=model.fc1.bias.detach().cpu().numpy().astype(np.float32),
        mlp_wout=(model.out.weight.detach().cpu().numpy().reshape(-1) * float(args.logit_scale)).astype(np.float32),
        mlp_bout=np.asarray(model.out.bias.detach().cpu().numpy().reshape(()) * float(args.logit_scale), dtype=np.float32),
        value_mode_code=np.array(2, dtype=np.float32),
    )
    meta = {
        "method": "progress_success_mlp",
        "cache": args.cache,
        "wm_ckpt": args.wm_ckpt,
        "vae_ckpt": args.vae_ckpt,
        "normalizers": args.normalizers,
        "out": str(out),
        "history": args.history,
        "embed_dim": z_dim,
        "basis": args.basis,
        "hidden": args.hidden,
        "l2": args.l2,
        "lr": args.lr,
        "logit_scale": args.logit_scale,
        "fit_steps": args.fit_steps,
        "batch_size": args.batch_size,
        "train_success": len(train_success),
        "train_failure": len(train_failure),
        "val_success": len(val_success),
        "val_failure": len(val_failure),
        "train_samples": int(y_train.shape[0]),
        "val_samples": int(y_val.shape[0]),
        "train": metrics_from_scores(train_scores, y_train),
        "val": metrics_from_scores(val_scores, y_val),
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
