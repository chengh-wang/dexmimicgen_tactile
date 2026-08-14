#!/usr/bin/env python3
"""Fit the episode-tail progress-conditioned linear success value for ThreePiece."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from train_threepiece_contrastive_setpoint_value import (  # noqa: E402
    binary_auc,
    encode_group_latents,
    is_failure_group,
    is_success_group,
    list_cache_groups,
    load_normalizers,
    split_groups,
)
from train_threepiece_high200_wm import ThreePieceWM  # noqa: E402


def load_wm(args: argparse.Namespace, device: torch.device) -> ThreePieceWM:
    wm = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    wm.load_state_dict(state, strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    return wm


def progress_array(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    return (np.arange(n, dtype=np.float32) / float(n - 1)).astype(np.float32)


def tail_indices(n: int, fraction: float) -> np.ndarray:
    k = int(math.ceil(float(fraction) * n))
    k = max(1, min(k, n))
    return np.arange(n - k, n, dtype=np.int64)


def rbf_features(tau: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    phi = torch.exp(-0.5 * torch.square((tau[:, None] - centers[None]) / max(float(width), 1e-6)))
    return phi / torch.clamp(phi.sum(dim=-1, keepdim=True), min=1e-6)


def collect_tail_latents(args, wm, normalizers, groups: list[str], device: torch.device):
    zs: list[np.ndarray] = []
    taus: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    selected_per_group: list[int] = []
    lengths: list[int] = []
    with h5py.File(args.cache, "r") as f:
        for key in groups:
            z = encode_group_latents(wm, f[key], normalizers, device, args.encode_batch_steps)
            idx = tail_indices(z.shape[0], args.tail_fraction)
            tau = progress_array(z.shape[0])[idx]
            if is_success_group(key):
                label = 1.0
            elif is_failure_group(key):
                label = 0.0
            else:
                continue
            zs.append(z[idx].astype(np.float32))
            taus.append(tau.astype(np.float32))
            labels.append(np.full((idx.size,), label, dtype=np.float32))
            selected_per_group.append(int(idx.size))
            lengths.append(int(z.shape[0]))
    return (
        np.concatenate(zs, axis=0).astype(np.float32),
        np.concatenate(taus, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.float32),
        {"selected_min": int(min(selected_per_group)), "selected_max": int(max(selected_per_group)), "length_min": int(min(lengths)), "length_max": int(max(lengths))},
    )


def score_linear(z_norm: torch.Tensor, tau: torch.Tensor, centers: torch.Tensor, width: float, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    phi = rbf_features(tau, centers, width)
    return torch.einsum("bm,md,bd->b", phi, w, z_norm) + phi @ b


@torch.no_grad()
def eval_metrics(z_norm: torch.Tensor, tau: torch.Tensor, y: torch.Tensor, centers: torch.Tensor, width: float, w: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    logits = score_linear(z_norm, tau, centers, width, w, b)
    scores = logits.detach().cpu().numpy().astype(np.float32)
    labels = y.detach().cpu().numpy().astype(np.float32)
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    return {
        "auc": binary_auc(scores.tolist(), labels.astype(int).tolist()),
        "score_mean_pos": float(pos.mean()) if pos.size else float("nan"),
        "score_mean_neg": float(neg.mean()) if neg.size else float("nan"),
        "prob_mean_pos": float((1.0 / (1.0 + np.exp(-np.clip(pos, -80, 80)))).mean()) if pos.size else float("nan"),
        "prob_mean_neg": float((1.0 / (1.0 + np.exp(-np.clip(neg, -80, 80)))).mean()) if neg.size else float("nan"),
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
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--basis", type=int, default=8)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--val-demos-per-class", type=int, default=20)
    parser.add_argument("--encode-batch-steps", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    normalizers = load_normalizers(args.normalizers)
    wm = load_wm(args, device)
    success, failure, _ = list_cache_groups(args.cache)
    train_success, val_success = split_groups(success, args.val_demos_per_class)
    train_failure, val_failure = split_groups(failure, args.val_demos_per_class)

    z_train, tau_train, y_train, train_sel = collect_tail_latents(args, wm, normalizers, train_success + train_failure, device)
    z_val, tau_val, y_val, val_sel = collect_tail_latents(args, wm, normalizers, val_success + val_failure, device)

    z_mean = z_train.mean(axis=0).astype(np.float32)
    z_std = np.maximum(z_train.std(axis=0).astype(np.float32), 1e-6)
    zt = torch.from_numpy((z_train - z_mean) / z_std).to(device)
    taut = torch.from_numpy(tau_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    zv = torch.from_numpy((z_val - z_mean) / z_std).to(device)
    tauv = torch.from_numpy(tau_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    centers = torch.linspace(0.0, 1.0, args.basis, device=device)
    width = float(args.width_scale) / max(args.basis - 1, 1)
    w = torch.zeros((args.basis, z_train.shape[1]), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros((args.basis,), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=args.lr, weight_decay=0.0)

    n_pos = float((y_train > 0.5).sum())
    n_neg = float((y_train < 0.5).sum())
    n_total = float(y_train.shape[0])
    alpha_pos = 0.5 * n_total / max(n_pos, 1.0)
    alpha_neg = 0.5 * n_total / max(n_neg, 1.0)
    weights_all = torch.where(yt > 0.5, torch.full_like(yt, alpha_pos), torch.full_like(yt, alpha_neg))

    best_auc = -math.inf
    best_state: dict[str, np.ndarray] | None = None
    metrics_log = out / "metrics.jsonl"
    with metrics_log.open("w") as log_f:
        for epoch in range(1, args.epochs + 1):
            perm_np = rng.permutation(zt.shape[0])
            loss_sum = 0.0
            batches = 0
            for start in range(0, zt.shape[0], args.batch_size):
                idx = torch.as_tensor(perm_np[start : start + args.batch_size], dtype=torch.long, device=device)
                logits = score_linear(zt[idx], taut[idx], centers, width, w, b)
                bce = F.binary_cross_entropy_with_logits(logits, yt[idx], reduction="none")
                loss = (weights_all[idx] * bce).mean() + float(args.l2) * torch.square(w).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach().cpu())
                batches += 1
            train_metrics = eval_metrics(zt, taut, yt, centers, width, w, b)
            val_metrics = eval_metrics(zv, tauv, yv, centers, width, w, b)
            row = {
                "epoch": epoch,
                "loss": loss_sum / max(batches, 1),
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "train_score_mean_pos": train_metrics["score_mean_pos"],
                "train_score_mean_neg": train_metrics["score_mean_neg"],
                "val_score_mean_pos": val_metrics["score_mean_pos"],
                "val_score_mean_neg": val_metrics["score_mean_neg"],
            }
            log_f.write(json.dumps(row) + "\n")
            if epoch == 1 or epoch % 25 == 0:
                print(json.dumps(row), flush=True)
            if val_metrics["auc"] > best_auc:
                best_auc = float(val_metrics["auc"])
                best_state = {
                    "W_table": w.detach().cpu().numpy().astype(np.float32),
                    "b_table": b.detach().cpu().numpy().astype(np.float32),
                }

    if best_state is None:
        raise RuntimeError("no best state was selected")

    W_best = torch.from_numpy(best_state["W_table"]).to(device)
    b_best = torch.from_numpy(best_state["b_table"]).to(device)
    train_best = eval_metrics(zt, taut, yt, centers, width, W_best, b_best)
    val_best = eval_metrics(zv, tauv, yv, centers, width, W_best, b_best)

    np.savez(
        out / "contrastive_params.npz",
        param_type_code=np.array(14, dtype=np.float32),
        basis_centers=centers.detach().cpu().numpy().astype(np.float32),
        basis_width=np.array(width, dtype=np.float32),
        W_table=best_state["W_table"],
        b_table=best_state["b_table"],
        z_mean=z_mean,
        z_std=z_std,
        value_mode_code=np.array(2, dtype=np.float32),
    )
    meta = {
        "method": "tail_progress_linear_balanced",
        "cache": args.cache,
        "wm_ckpt": args.wm_ckpt,
        "vae_ckpt": args.vae_ckpt,
        "normalizers": args.normalizers,
        "out": str(out),
        "history": args.history,
        "embed_dim": int(z_train.shape[1]),
        "tail_fraction": args.tail_fraction,
        "tau_train_min": float(tau_train.min()),
        "tau_train_max": float(tau_train.max()),
        "basis": args.basis,
        "width": width,
        "width_scale": args.width_scale,
        "l2": args.l2,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "balanced": True,
        "alpha_pos": alpha_pos,
        "alpha_neg": alpha_neg,
        "train_success": len(train_success),
        "train_failure": len(train_failure),
        "val_success": len(val_success),
        "val_failure": len(val_failure),
        "train_samples": int(y_train.shape[0]),
        "val_samples": int(y_val.shape[0]),
        "train_selection": train_sel,
        "val_selection": val_sel,
        "best_val_auc": best_auc,
        "train": train_best,
        "val": val_best,
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
