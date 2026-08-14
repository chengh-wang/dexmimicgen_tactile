#!/usr/bin/env python3
"""Fit A/B/C progress-aware contrastive setpoints for DexMimicGen WM caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from fit_dexmg_progress_timevariant_setpoints import (  # noqa: E402
    binary_auc,
    encode_group_latents,
    is_failure_group,
    is_success_group,
    list_groups,
    load_normalizers,
    load_wm,
    split_groups,
)


def progress_array(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    return (np.arange(n, dtype=np.float32) / float(n - 1)).astype(np.float32)


def collect_all_latents(args, wm, normalizers, groups, device):
    chunks = {"success": [], "failure": []}
    taus = {"success": [], "failure": []}
    with h5py.File(args.cache, "r") as f:
        for key in groups:
            z = encode_group_latents(wm, f[key], normalizers, device, args.encode_batch_steps)
            tau = progress_array(z.shape[0])
            if is_success_group(key):
                chunks["success"].append(z)
                taus["success"].append(tau)
            elif is_failure_group(key):
                chunks["failure"].append(z)
                taus["failure"].append(tau)
    return {
        "success_z": np.concatenate(chunks["success"], axis=0).astype(np.float32),
        "success_tau": np.concatenate(taus["success"], axis=0).astype(np.float32),
        "failure_z": np.concatenate(chunks["failure"], axis=0).astype(np.float32),
        "failure_tau": np.concatenate(taus["failure"], axis=0).astype(np.float32),
    }


def bin_mask(tau: np.ndarray, k: int, b: int) -> np.ndarray:
    idx = np.minimum((tau * k).astype(np.int64), k - 1)
    return idx == b


def fit_binned(data, out: Path, k: int, sigma_scale: float, sigma_gap_fraction: float) -> dict:
    sz, st = data["success_z"], data["success_tau"]
    fz, ft = data["failure_z"], data["failure_tau"]
    dim = sz.shape[1]
    v_table = np.zeros((k, dim), dtype=np.float32)
    beta_star = np.zeros((k,), dtype=np.float32)
    sigma = np.zeros((k,), dtype=np.float32)
    aucs = []
    for b in range(k):
        sp = sz[bin_mask(st, k, b)]
        fn = fz[bin_mask(ft, k, b)]
        if sp.size == 0 or fn.size == 0:
            raise RuntimeError(f"empty progress bin {b}/{k}: success={sp.shape} failure={fn.shape}")
        e = sp.mean(axis=0) - fn.mean(axis=0)
        v = e / max(float(np.linalg.norm(e)), 1e-8)
        bp = sp @ v
        bn = fn @ v
        gap = abs(float(bp.mean() - bn.mean()))
        v_table[b] = v.astype(np.float32)
        beta_star[b] = np.quantile(bp, 0.8).astype(np.float32)
        sigma[b] = max(float(bp.std()) * sigma_scale, gap * sigma_gap_fraction, 1e-3)
        aucs.append(binary_auc(np.concatenate([bp, bn]), np.concatenate([np.ones(bp.size), np.zeros(bn.size)])))
    np.savez(
        out / "contrastive_params.npz",
        param_type_code=np.array(10, dtype=np.float32),
        v_table=v_table,
        beta_star_table=beta_star,
        sigma_table=sigma,
        value_mode_code=np.array(0, dtype=np.float32),
    )
    return {
        "method": "A_binned_v",
        "k": k,
        "sigma_scale": sigma_scale,
        "sigma_gap_fraction": sigma_gap_fraction,
        "mean_bin_auc": float(np.mean(aucs)),
        "min_bin_auc": float(np.min(aucs)),
        "max_bin_auc": float(np.max(aucs)),
    }


def rbf_features(tau: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    phi = torch.exp(-0.5 * torch.square((tau[:, None] - centers[None]) / max(width, 1e-4)))
    return phi / torch.clamp(phi.sum(dim=-1, keepdim=True), min=1e-6)


def fit_logistic(data, out: Path, basis: int, width: float, l2: float, steps: int, batch_size: int, device, seed: int):
    rng = np.random.default_rng(seed)
    z = np.concatenate([data["success_z"], data["failure_z"]], axis=0).astype(np.float32)
    tau = np.concatenate([data["success_tau"], data["failure_tau"]], axis=0).astype(np.float32)
    y = np.concatenate(
        [np.ones((data["success_z"].shape[0],), dtype=np.float32), np.zeros((data["failure_z"].shape[0],), dtype=np.float32)]
    )
    max_fit = min(z.shape[0], 160000)
    idx = rng.choice(z.shape[0], size=max_fit, replace=False)
    zt = torch.from_numpy(z[idx]).to(device)
    taut = torch.from_numpy(tau[idx]).to(device)
    yt = torch.from_numpy(y[idx]).to(device)
    centers = torch.linspace(0.0, 1.0, basis, device=device)
    w = torch.zeros((basis, z.shape[1]), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros((basis,), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=3e-3, weight_decay=0.0)
    for _ in range(steps):
        bi = torch.randint(0, max_fit, (min(batch_size, max_fit),), device=device)
        phi = rbf_features(taut[bi], centers, width)
        logits = torch.einsum("bm,md,bd->b", phi, w, zt[bi]) + phi @ b
        loss = F.binary_cross_entropy_with_logits(logits, yt[bi]) + l2 * torch.square(w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        phi = rbf_features(taut, centers, width)
        scores = (torch.einsum("bm,md,bd->b", phi, w, zt) + phi @ b).detach().cpu().numpy().astype(np.float32)
    np.savez(
        out / "contrastive_params.npz",
        param_type_code=np.array(11, dtype=np.float32),
        basis_centers=centers.detach().cpu().numpy().astype(np.float32),
        basis_width=np.array(width, dtype=np.float32),
        W_table=w.detach().cpu().numpy().astype(np.float32),
        b_table=b.detach().cpu().numpy().astype(np.float32),
        value_mode_code=np.array(2, dtype=np.float32),
    )
    return {
        "method": "B_progress_logistic",
        "basis": basis,
        "width": width,
        "l2": l2,
        "fit_steps": steps,
        "fit_auc_sample": binary_auc(scores, y[idx]),
        "score_mean_pos": float(scores[y[idx] > 0.5].mean()),
        "score_mean_neg": float(scores[y[idx] < 0.5].mean()),
    }


def fit_reftraj(data, out: Path, k: int, sigma_scale: float) -> dict:
    sz, st = data["success_z"], data["success_tau"]
    fz, ft = data["failure_z"], data["failure_tau"]
    sp_late = sz[st >= 0.8]
    fn_late = fz[ft >= 0.8]
    e = sp_late.mean(axis=0) - fn_late.mean(axis=0)
    v = (e / max(float(np.linalg.norm(e)), 1e-8)).astype(np.float32)
    beta = sz @ v
    beta_ref = np.zeros((k,), dtype=np.float32)
    sigma = np.zeros((k,), dtype=np.float32)
    for b in range(k):
        vals = beta[bin_mask(st, k, b)]
        beta_ref[b] = vals.mean().astype(np.float32)
        sigma[b] = max(float(vals.std()) * sigma_scale, 1e-3)
    np.savez(
        out / "contrastive_params.npz",
        param_type_code=np.array(12, dtype=np.float32),
        v=v,
        beta_star_table=beta_ref,
        sigma_table=sigma,
        value_mode_code=np.array(0, dtype=np.float32),
    )
    beta_pos = sp_late @ v
    beta_neg = fn_late @ v
    return {
        "method": "C_ref_beta_traj",
        "k": k,
        "sigma_scale": sigma_scale,
        "late_auc": binary_auc(np.concatenate([beta_pos, beta_neg]), np.concatenate([np.ones(beta_pos.size), np.zeros(beta_neg.size)])),
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
    parser.add_argument("--tactile-channels", type=int, default=12)
    parser.add_argument("--joint-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=24)
    parser.add_argument("--val-demos-per-class", type=int, default=20)
    parser.add_argument("--source-filter", choices=["rollout", "official_tactile", "any"], default="rollout")
    parser.add_argument("--encode-batch-steps", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizers = load_normalizers(args.normalizers)
    wm = load_wm(args, device)
    success, failure = list_groups(args.cache, args.source_filter)
    train_success, val_success = split_groups(success, args.val_demos_per_class)
    train_failure, val_failure = split_groups(failure, args.val_demos_per_class)
    data = collect_all_latents(args, wm, normalizers, train_success + train_failure, device)

    configs = []
    for k in [4, 5, 6, 8, 10]:
        for sigma_scale in [1.5, 2.5]:
            configs.append(("A", f"A_bin{k}_sig{str(sigma_scale).replace('.', 'p')}", {"k": k, "sigma_scale": sigma_scale}))
    for basis in [4, 6, 8, 10, 12]:
        for l2 in [1e-4, 1e-3]:
            width = 1.0 / max(basis - 1, 1) * 1.5
            configs.append(("B", f"B_basis{basis}_l2{l2:.0e}", {"basis": basis, "width": width, "l2": l2}))
    for k in [4, 5, 6, 8, 10]:
        for sigma_scale in [1.0, 2.0]:
            configs.append(("C", f"C_ref{k}_sig{str(sigma_scale).replace('.', 'p')}", {"k": k, "sigma_scale": sigma_scale}))

    rows = []
    for idx, (kind, name, cfg) in enumerate(configs):
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        if kind == "A":
            metrics = fit_binned(data, out, cfg["k"], cfg["sigma_scale"], 0.25)
        elif kind == "B":
            metrics = fit_logistic(data, out, cfg["basis"], cfg["width"], cfg["l2"], 2000, 8192, device, args.seed + idx)
        else:
            metrics = fit_reftraj(data, out, cfg["k"], cfg["sigma_scale"])
        meta = {
            "name": name,
            "kind": kind,
            "cache": args.cache,
            "wm_ckpt": args.wm_ckpt,
            "train_success": len(train_success),
            "train_failure": len(train_failure),
            "val_success": len(val_success),
            "val_failure": len(val_failure),
            **metrics,
        }
        (out / "metrics.json").write_text(json.dumps(meta, indent=2))
        rows.append(meta)
        print(json.dumps(meta), flush=True)
    (out_root / "sweep_metrics.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
