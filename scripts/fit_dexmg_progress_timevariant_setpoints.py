#!/usr/bin/env python3
"""Fit progress-conditioned contrastive values for DexMimicGen WM caches.

This is the drawer/general DexMG version of the three-piece B experiment:

    p(success | z, tau) = sigmoid(w(tau)^T z + b(tau))
    w(tau) = sum_i phi_i(tau) W_i

The output ``contrastive_params.npz`` uses param_type_code=11, which is the
time-variant logistic format consumed by the contrastive residual rollout code.
"""

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
from train_dexmg_20hz_wm import DexMGWM  # noqa: E402


def group_sort_key(name: str) -> tuple[int, int, str]:
    demo = name.split("__")[-1]
    if "_" not in demo:
        return (2, 0, demo)
    prefix, idx = demo.rsplit("_", 1)
    try:
        idx_i = int(idx)
    except ValueError:
        idx_i = 0
    cls = 0 if prefix == "success" else 1 if prefix == "failure" else 2
    return (cls, idx_i, demo)


def is_success_group(name: str) -> bool:
    return name.split("__")[-1].startswith("success_")


def is_failure_group(name: str) -> bool:
    return name.split("__")[-1].startswith("failure_")


def progress_array(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    return (np.arange(n, dtype=np.float32) / float(n - 1)).astype(np.float32)


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pairs = sorted(zip(scores.reshape(-1).tolist(), labels.reshape(-1).astype(int).tolist()), key=lambda x: x[0])
    labels_list = [label for _, label in pairs]
    n_pos = sum(labels_list)
    n_neg = len(labels_list) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_normalizers(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    required = ["joint_mean", "joint_std", "action_mean", "action_std"]
    missing = [k for k in required if k not in npz.files]
    if missing:
        raise KeyError(f"normalizers missing keys: {missing}")
    return {k: npz[k].astype(np.float32) for k in npz.files}


def load_wm(args: argparse.Namespace, device: torch.device) -> DexMGWM:
    wm = DexMGWM(
        args.vae_ckpt,
        history=args.history,
        embed_dim=args.embed_dim,
        tactile_channels=args.tactile_channels,
        joint_dim=args.joint_dim,
        action_dim=args.action_dim,
    ).to(device)
    ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    wm.load_state_dict(state, strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    return wm


@torch.no_grad()
def encode_group_latents(
    wm: DexMGWM,
    group: h5py.Group,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    batch_steps: int,
) -> np.ndarray:
    tactile_mu = group["tactile_mu"]
    joint = group["joint"]
    n = int(joint.shape[0])
    chunks: list[np.ndarray] = []
    for start in range(0, n, batch_steps):
        end = min(start + batch_steps, n)
        tac = tactile_mu[start:end].astype(np.float32)
        jnt = (joint[start:end].astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
        batch = {
            "tactile_mu": torch.from_numpy(tac[None]).to(device, non_blocking=True),
            "joint": torch.from_numpy(jnt[None]).to(device, non_blocking=True),
        }
        z = wm.encode(batch).squeeze(0).detach().cpu().numpy().astype(np.float32)
        chunks.append(z)
    return np.concatenate(chunks, axis=0)


def list_groups(cache: str, source_filter: str) -> tuple[list[str], list[str]]:
    success: list[str] = []
    failure: list[str] = []
    with h5py.File(cache, "r") as f:
        for key in sorted(f.keys(), key=group_sort_key):
            if source_filter != "any" and f[key].attrs.get("source", "") != source_filter:
                continue
            if is_success_group(key):
                success.append(key)
            elif is_failure_group(key):
                failure.append(key)
    if not success or not failure:
        raise RuntimeError(
            f"need both success and failure groups in {cache}; got success={len(success)} failure={len(failure)} source={source_filter}"
        )
    return success, failure


def split_groups(groups: list[str], val_count: int) -> tuple[list[str], list[str]]:
    if val_count <= 0:
        return groups, []
    if len(groups) <= val_count:
        return groups, []
    return groups[:-val_count], groups[-val_count:]


def collect_latents(args, wm: DexMGWM, normalizers: dict[str, np.ndarray], groups: list[str], device: torch.device):
    zs: list[np.ndarray] = []
    taus: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with h5py.File(args.cache, "r") as f:
        for key in groups:
            z = encode_group_latents(wm, f[key], normalizers, device, args.encode_batch_steps)
            zs.append(z)
            taus.append(progress_array(z.shape[0]))
            label = 1.0 if is_success_group(key) else 0.0
            labels.append(np.full((z.shape[0],), label, dtype=np.float32))
    return (
        np.concatenate(zs, axis=0).astype(np.float32),
        np.concatenate(taus, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.float32),
    )


def rbf_features(tau: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    phi = torch.exp(-0.5 * torch.square((tau[:, None] - centers[None]) / max(width, 1e-4)))
    return phi / torch.clamp(phi.sum(dim=-1, keepdim=True), min=1e-6)


def fit_progress_logistic(
    z: np.ndarray,
    tau: np.ndarray,
    y: np.ndarray,
    basis: int,
    width: float,
    l2: float,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
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
        logits = torch.einsum("bm,md,bd->b", phi, w, zt) + phi @ b
        scores = logits.detach().cpu().numpy().astype(np.float32)
    params = {
        "param_type_code": np.array(11, dtype=np.float32),
        "basis_centers": centers.detach().cpu().numpy().astype(np.float32),
        "basis_width": np.array(width, dtype=np.float32),
        "W_table": w.detach().cpu().numpy().astype(np.float32),
        "b_table": b.detach().cpu().numpy().astype(np.float32),
        "value_mode_code": np.array(2, dtype=np.float32),
    }
    metrics = {
        "fit_auc_sample": binary_auc(scores, y[idx]),
        "score_mean_pos": float(scores[y[idx] > 0.5].mean()),
        "score_mean_neg": float(scores[y[idx] < 0.5].mean()),
        "fit_samples": int(max_fit),
    }
    return params, metrics


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
    parser.add_argument("--basis", type=int, default=8)
    parser.add_argument("--width", type=float, default=0.21428571428571427)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--fit-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--val-demos-per-class", type=int, default=20)
    parser.add_argument("--source-filter", choices=["rollout", "official_tactile", "any"], default="rollout")
    parser.add_argument("--encode-batch-steps", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizers = load_normalizers(args.normalizers)
    wm = load_wm(args, device)
    success, failure = list_groups(args.cache, args.source_filter)
    train_success, val_success = split_groups(success, args.val_demos_per_class)
    train_failure, val_failure = split_groups(failure, args.val_demos_per_class)
    train_groups = train_success + train_failure
    val_groups = val_success + val_failure

    z_train, tau_train, y_train = collect_latents(args, wm, normalizers, train_groups, device)
    params, metrics = fit_progress_logistic(
        z_train, tau_train, y_train, args.basis, args.width, args.l2, args.fit_steps, args.batch_size, device, args.seed
    )
    z_val, tau_val, y_val = collect_latents(args, wm, normalizers, val_groups, device) if val_groups else (None, None, None)
    if z_val is not None:
        with torch.no_grad():
            zt = torch.from_numpy(z_val).to(device)
            taut = torch.from_numpy(tau_val).to(device)
            centers = torch.from_numpy(params["basis_centers"]).to(device)
            width = float(params["basis_width"])
            W = torch.from_numpy(params["W_table"]).to(device)
            b = torch.from_numpy(params["b_table"]).to(device)
            phi = rbf_features(taut, centers, width)
            logits = torch.einsum("bm,md,bd->b", phi, W, zt) + phi @ b
            val_scores = logits.detach().cpu().numpy().astype(np.float32)
        metrics["val_auc"] = binary_auc(val_scores, y_val)
        metrics["val_score_mean_pos"] = float(val_scores[y_val > 0.5].mean())
        metrics["val_score_mean_neg"] = float(val_scores[y_val < 0.5].mean())

    np.savez(out / "contrastive_params.npz", **params)
    meta = vars(args).copy()
    meta.update(
        metrics
        | {
            "method": "B_progress_logistic",
            "train_success": len(train_success),
            "train_failure": len(train_failure),
            "val_success": len(val_success),
            "val_failure": len(val_failure),
        }
    )
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
