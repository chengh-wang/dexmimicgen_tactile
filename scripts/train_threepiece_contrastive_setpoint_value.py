#!/usr/bin/env python3
"""Contrastive setpoint value for ThreePiece 20 Hz WM latents.

This implements a minimal version of:

  D+ late success latents, D- late failure / early latents
  -> projection P
  -> contrastive direction v
  -> beta(z) = v^T P z
  -> V_alpha(z) = exp(-((beta* - beta(z)) / sigma)^2)

The script is intentionally bound to the 20 Hz tactile WM/cache semantics:
the cache stores one tactile_mu, joint, and raw low20 action per control step.
Joint and action are normalized with the 20 Hz WM normalizers before encoding
or rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402


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


def is_success_group(name: str) -> bool:
    return name.split("__")[-1].startswith("success_")


def is_failure_group(name: str) -> bool:
    return name.split("__")[-1].startswith("failure_")


def load_normalizers(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    required = ["joint_mean", "joint_std", "action_mean", "action_std"]
    missing = [k for k in required if k not in npz.files]
    if missing:
        raise KeyError(f"normalizers missing keys: {missing}")
    return {k: npz[k].astype(np.float32) for k in npz.files}


def args_to_jsonable(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def normalize_joint(joint: np.ndarray, normalizers: dict[str, np.ndarray]) -> np.ndarray:
    return ((joint.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]).astype(np.float32)


def normalize_action(action: np.ndarray, normalizers: dict[str, np.ndarray]) -> np.ndarray:
    return ((action.astype(np.float32) - normalizers["action_mean"]) / normalizers["action_std"]).astype(np.float32)


def fraction_indices(length: int, fraction: float, region: str) -> np.ndarray:
    n = max(1, int(round(length * fraction)))
    n = min(n, length)
    if region == "late":
        return np.arange(length - n, length, dtype=np.int64)
    if region == "early":
        return np.arange(0, n, dtype=np.int64)
    raise ValueError(f"unknown region: {region}")


def choose_indices(length: int, fraction: float, region: str, max_per_group: int, rng: np.random.Generator) -> np.ndarray:
    idx = fraction_indices(length, fraction, region)
    if max_per_group > 0 and idx.size > max_per_group:
        idx = np.sort(rng.choice(idx, size=max_per_group, replace=False))
    return idx


def summarize_values(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def binary_auc(scores: Iterable[float], labels: Iterable[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
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


@torch.no_grad()
def encode_group_latents(
    wm: ThreePieceWM,
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
        jnt = normalize_joint(joint[start:end], normalizers)
        batch = {
            "tactile_mu": torch.from_numpy(tac[None]).to(device, non_blocking=True),
            "joint": torch.from_numpy(jnt[None]).to(device, non_blocking=True),
        }
        z = wm.encode(batch).squeeze(0).detach().cpu().numpy().astype(np.float32)
        chunks.append(z)
    return np.concatenate(chunks, axis=0)


def load_wm(args: argparse.Namespace, device: torch.device) -> ThreePieceWM:
    wm = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    wm.load_state_dict(state, strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    return wm


def list_cache_groups(cache_path: str) -> tuple[list[str], list[str], dict[str, int]]:
    success: list[str] = []
    failure: list[str] = []
    lengths: dict[str, int] = {}
    with h5py.File(cache_path, "r") as f:
        for key in sorted(f.keys(), key=group_sort_key):
            n = int(f[key]["action"].shape[0])
            lengths[key] = n
            if is_success_group(key):
                success.append(key)
            elif is_failure_group(key):
                failure.append(key)
    if not success:
        raise RuntimeError(f"no success groups found in {cache_path}")
    if not failure:
        raise RuntimeError(f"no failure groups found in {cache_path}")
    return success, failure, lengths


def split_groups(groups: list[str], val_count: int) -> tuple[list[str], list[str]]:
    if val_count <= 0:
        return groups, []
    if len(groups) <= val_count:
        raise ValueError(f"not enough groups={len(groups)} for val_count={val_count}")
    return groups[:-val_count], groups[-val_count:]


def maybe_limit_groups(groups: list[str], max_groups: int) -> list[str]:
    if max_groups <= 0:
        return groups
    return groups[:max_groups]


def collect_latent_sets(
    args: argparse.Namespace,
    wm: ThreePieceWM,
    normalizers: dict[str, np.ndarray],
    groups: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    sets: dict[str, list[np.ndarray]] = {
        "success_late": [],
        "success_early": [],
        "failure_late": [],
        "failure_early": [],
    }
    with h5py.File(args.cache, "r") as f:
        for key in groups:
            z = encode_group_latents(wm, f[key], normalizers, device, args.encode_batch_steps)
            late = choose_indices(z.shape[0], args.late_fraction, "late", args.max_points_per_group, rng)
            early = choose_indices(z.shape[0], args.early_fraction, "early", args.max_points_per_group, rng)
            if is_success_group(key):
                sets["success_late"].append(z[late])
                sets["success_early"].append(z[early])
            elif is_failure_group(key):
                sets["failure_late"].append(z[late])
                sets["failure_early"].append(z[early])
    out: dict[str, np.ndarray] = {}
    for name, chunks in sets.items():
        if chunks:
            out[name] = np.concatenate(chunks, axis=0).astype(np.float32)
        else:
            out[name] = np.empty((0, args.embed_dim), dtype=np.float32)
    return out


def build_projection(x: np.ndarray, proj_dim: int, center: bool) -> tuple[np.ndarray, np.ndarray, str]:
    dim = int(x.shape[1])
    if proj_dim <= 0 or proj_dim >= dim:
        return np.eye(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32), "identity"
    mean = x.mean(axis=0).astype(np.float32) if center else np.zeros(dim, dtype=np.float32)
    xc = x.astype(np.float32) - mean
    # Full matrices are unnecessary. For the current 192-D WM latent this is cheap
    # and more stable than forming a covariance matrix with small sample counts.
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    p = vt[:proj_dim].astype(np.float32)
    return p, mean, "pca"


def project(z: np.ndarray, p: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (z.astype(np.float32) - mean.astype(np.float32)) @ p.astype(np.float32).T


def value_from_beta(beta: np.ndarray, beta_star: float, sigma: float, value_mode: str = "symmetric") -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    alpha = beta_star - beta.astype(np.float32)
    if value_mode == "one_sided":
        alpha = np.maximum(alpha, 0.0)
    return np.exp(-np.square(alpha / sigma)).astype(np.float32)


def compute_contrastive_params(args: argparse.Namespace, sets: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    positive = sets["success_late"]
    negative_mode = getattr(args, "negative_mode", "auto")
    if negative_mode == "auto":
        negative_mode = "oldstyle" if args.include_early_negative else "lateonly"
    if negative_mode == "oldstyle":
        negative_parts = [sets["failure_late"], sets["success_early"], sets["failure_early"]]
    elif negative_mode == "lateonly":
        negative_parts = [sets["failure_late"]]
    elif negative_mode == "failure_late_early":
        negative_parts = [sets["failure_late"], sets["failure_early"]]
    else:
        raise ValueError(f"unknown negative_mode: {negative_mode}")
    negative = np.concatenate([x for x in negative_parts if x.size], axis=0).astype(np.float32)
    if positive.size == 0 or negative.size == 0:
        raise RuntimeError("positive and negative latent sets must be non-empty")

    fit_x = np.concatenate([positive, negative], axis=0).astype(np.float32)
    p, p_mean, projection = build_projection(fit_x, args.proj_dim, args.pca_center)
    p_pos = project(positive, p, p_mean)
    p_neg = project(negative, p, p_mean)
    e = p_pos.mean(axis=0) - p_neg.mean(axis=0)
    e_norm = float(np.linalg.norm(e))
    if e_norm < 1e-8:
        raise RuntimeError("contrastive direction is degenerate; D+ and D- means are identical")
    direction_norm_before_normalize = e_norm
    bias = 0.0
    if args.direction == "mean":
        v = (e / e_norm).astype(np.float32)
    elif args.direction == "lda":
        cp = p_pos - p_pos.mean(axis=0, keepdims=True)
        cn = p_neg - p_neg.mean(axis=0, keepdims=True)
        cov = (cp.T @ cp + cn.T @ cn) / max(cp.shape[0] + cn.shape[0] - 2, 1)
        dim = cov.shape[0]
        scale = float(np.trace(cov) / max(dim, 1))
        cov = cov + np.eye(dim, dtype=np.float32) * max(scale * args.lda_shrinkage, 1e-5)
        raw = np.linalg.solve(cov.astype(np.float64), e.astype(np.float64)).astype(np.float32)
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm < 1e-8:
            raise RuntimeError("LDA direction is degenerate")
        direction_norm_before_normalize = raw_norm
        v = (raw / raw_norm).astype(np.float32)
    elif args.direction == "logistic":
        x = np.concatenate([p_pos, p_neg], axis=0).astype(np.float32)
        y = np.concatenate(
            [np.ones((p_pos.shape[0],), dtype=np.float32), np.zeros((p_neg.shape[0],), dtype=np.float32)],
            axis=0,
        )
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        w = torch.zeros((x.shape[1],), dtype=torch.float32, requires_grad=True)
        b = torch.zeros((), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=args.logistic_max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = xt @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, yt) + args.logistic_l2 * torch.square(w).mean()
            loss.backward()
            return loss

        opt.step(closure)
        raw = w.detach().numpy().astype(np.float32)
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm < 1e-8:
            raise RuntimeError("logistic direction is degenerate")
        direction_norm_before_normalize = raw_norm
        v = (raw / raw_norm).astype(np.float32)
        bias = float(b.detach().cpu()) / raw_norm
    else:
        raise ValueError(f"unknown direction: {args.direction}")
    beta_pos = p_pos @ v + bias
    beta_neg = p_neg @ v + bias

    if args.value_mode == "logistic":
        beta_star = 0.0
        sigma = 1.0
        sigma_source = "logistic"
    elif args.beta_star_mode == "mean":
        beta_star = float(beta_pos.mean())
    else:
        beta_star = float(np.quantile(beta_pos, args.beta_star_quantile))

    if args.value_mode == "logistic":
        pass
    elif args.sigma is not None and args.sigma > 0:
        sigma = float(args.sigma)
        sigma_source = "arg"
    else:
        spread = float(np.std(beta_pos))
        gap = abs(float(beta_pos.mean() - beta_neg.mean()))
        sigma = max(spread * args.sigma_scale, gap * args.sigma_gap_fraction, 1e-3)
        sigma_source = "positive_spread_and_gap"

    params = {
        "P": p.astype(np.float32),
        "P_mean": p_mean.astype(np.float32),
        "v": v.astype(np.float32),
        "bias": np.array(bias, dtype=np.float32),
        "beta_star": np.array(beta_star, dtype=np.float32),
        "sigma": np.array(sigma, dtype=np.float32),
        "value_mode_code": np.array({"symmetric": 0, "one_sided": 1, "logistic": 2}[args.value_mode], dtype=np.float32),
        "positive_mean": positive.mean(axis=0).astype(np.float32),
        "negative_mean": negative.mean(axis=0).astype(np.float32),
    }
    metrics: dict[str, float | int | str] = {
        "projection": projection,
        "direction": args.direction,
        "value_mode": args.value_mode,
        "negative_mode": negative_mode,
        "latent_dim": int(positive.shape[1]),
        "proj_dim": int(p.shape[0]),
        "positive_n": int(positive.shape[0]),
        "negative_n": int(negative.shape[0]),
        "direction_norm_before_normalize": direction_norm_before_normalize,
        "bias": bias,
        "beta_star": beta_star,
        "beta_star_mode": args.beta_star_mode,
        "beta_star_quantile": float(args.beta_star_quantile),
        "sigma": sigma,
        "sigma_source": sigma_source,
        "beta_pos_mean": float(beta_pos.mean()),
        "beta_pos_std": float(beta_pos.std()),
        "beta_neg_mean": float(beta_neg.mean()),
        "beta_neg_std": float(beta_neg.std()),
        "beta_gap": float(beta_pos.mean() - beta_neg.mean()),
        "fit_auc_beta_pos_vs_neg": binary_auc(
            np.concatenate([beta_pos, beta_neg]).tolist(),
            [1] * beta_pos.size + [0] * beta_neg.size,
        ),
    }
    return params, metrics


def load_contrastive_params(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    required = ["P", "P_mean", "v", "beta_star", "sigma"]
    missing = [k for k in required if k not in npz.files]
    if missing:
        raise KeyError(f"contrastive params missing keys: {missing}")
    return {k: npz[k].astype(np.float32) for k in npz.files}


def beta_value_for_z(z: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pz = project(z, params["P"], params["P_mean"])
    beta = pz @ params["v"] + float(params.get("bias", np.array(0.0, dtype=np.float32)))
    mode_code = int(float(params.get("value_mode_code", np.array(0.0, dtype=np.float32))))
    if mode_code == 2:
        value = (1.0 / (1.0 + np.exp(-beta))).astype(np.float32)
    else:
        value_mode = "one_sided" if mode_code == 1 else "symmetric"
        value = value_from_beta(beta, float(params["beta_star"]), float(params["sigma"]), value_mode)
    return beta.astype(np.float32), value.astype(np.float32)


def evaluate_sets(sets: dict[str, np.ndarray], params: dict[str, np.ndarray]) -> dict[str, object]:
    out: dict[str, object] = {}
    scores: list[float] = []
    labels: list[int] = []
    for name, z in sets.items():
        if z.size == 0:
            continue
        beta, value = beta_value_for_z(z, params)
        out[name] = {
            "beta": summarize_values(beta),
            "value": summarize_values(value),
        }
        if name == "success_late":
            scores.extend(beta.tolist())
            labels.extend([1] * beta.size)
        elif name == "failure_late":
            scores.extend(beta.tolist())
            labels.extend([0] * beta.size)
    out["success_late_vs_failure_late_auc_beta"] = binary_auc(scores, labels) if scores else float("nan")
    return out


@dataclass(frozen=True)
class WindowRef:
    group: str
    start: int


class ContrastiveWindowDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        groups: list[str],
        lengths: dict[str, int],
        normalizers: dict[str, np.ndarray],
        history: int,
        chunk: int,
        stride: int,
    ) -> None:
        self.cache_path = cache_path
        self.groups = groups
        self.lengths = lengths
        self.normalizers = normalizers
        self.history = history
        self.chunk = chunk
        self.window = history + chunk
        self.stride = stride
        self.refs: list[WindowRef] = []
        for group in groups:
            n = lengths[group]
            for start in range(0, n - self.window + 1, stride):
                self.refs.append(WindowRef(group, start))
        if not self.refs:
            raise RuntimeError("no train windows")
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
        ref = self.refs[idx]
        g = self._file()[ref.group]
        end = ref.start + self.window
        tactile_mu = g["tactile_mu"][ref.start : ref.start + self.history].astype(np.float32)
        joint = g["joint"][ref.start : ref.start + self.history].astype(np.float32)
        action = g["action"][ref.start:end].astype(np.float32)
        joint = normalize_joint(joint, self.normalizers)
        action = normalize_action(action, self.normalizers)
        return {
            "tactile_mu": torch.from_numpy(tactile_mu),
            "joint": torch.from_numpy(joint),
            "action": torch.from_numpy(action),
            "is_success": torch.tensor(1.0 if is_success_group(ref.group) else 0.0, dtype=torch.float32),
        }


def to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


class SetpointValueHead(nn.Module):
    def __init__(self, embed_dim: int = 192, action_dim: int = 14, history: int = 4, chunk: int = 20) -> None:
        super().__init__()
        self.history = history
        self.chunk = chunk
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=768,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.net = nn.Sequential(
            nn.Linear(history * embed_dim + embed_dim + embed_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z_hist: torch.Tensor, action_chunk: torch.Tensor, z_h: torch.Tensor) -> torch.Tensor:
        b = z_hist.shape[0]
        a = self.action_net(action_chunk.float())
        a = self.action_encoder(a).mean(dim=1)
        x = torch.cat([z_hist.reshape(b, -1), a, z_h], dim=-1)
        return self.net(x).squeeze(-1)


def wm_rollout(wm: ThreePieceWM, sample: dict[str, torch.Tensor], history: int, chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_hist = wm.encode({"tactile_mu": sample["tactile_mu"], "joint": sample["joint"]})
    z_window = z_hist
    actions = sample["action"]
    for k in range(chunk):
        a_win = wm.action(actions[:, k + 1 : k + 1 + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    action_chunk = actions[:, history : history + chunk]
    return z_hist, action_chunk, z_window[:, -1]


def torch_value_from_params(z: torch.Tensor, params_t: dict[str, torch.Tensor]) -> torch.Tensor:
    pz = (z.float() - params_t["P_mean"]) @ params_t["P"].T
    beta = pz @ params_t["v"]
    alpha = params_t["beta_star"] - beta
    return torch.exp(-torch.square(alpha / torch.clamp(params_t["sigma"], min=1e-6)))


def params_to_torch(params: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in params.items() if k in {"P", "P_mean", "v", "beta_star", "sigma"}}


@torch.no_grad()
def evaluate_head(
    head: SetpointValueHead,
    wm: ThreePieceWM,
    loader: DataLoader,
    params_t: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    head.eval()
    total = 0
    total_mse = 0.0
    total_l1 = 0.0
    pred_all: list[float] = []
    target_all: list[float] = []
    labels: list[int] = []
    for bi, batch in enumerate(loader):
        if bi >= args.eval_batches:
            break
        batch = to_device(batch, device)
        z_hist, action_chunk, z_h = wm_rollout(wm, batch, args.history, args.chunk)
        target = torch_value_from_params(z_h, params_t)
        pred = torch.sigmoid(head(z_hist, action_chunk, z_h))
        mse = F.mse_loss(pred, target, reduction="sum")
        l1 = F.l1_loss(pred, target, reduction="sum")
        total += pred.numel()
        total_mse += float(mse)
        total_l1 += float(l1)
        pred_all.extend(pred.detach().cpu().tolist())
        target_all.extend(target.detach().cpu().tolist())
        labels.extend(batch["is_success"].detach().cpu().int().tolist())
    return {
        "val_mse": total_mse / max(total, 1),
        "val_l1": total_l1 / max(total, 1),
        "val_pred_mean": float(np.mean(pred_all)) if pred_all else float("nan"),
        "val_target_mean": float(np.mean(target_all)) if target_all else float("nan"),
        "val_pred_success_auc": binary_auc(pred_all, labels),
        "val_target_success_auc": binary_auc(target_all, labels),
    }


def run_fit(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizers = load_normalizers(args.normalizers)
    wm = load_wm(args, device)
    success, failure, lengths = list_cache_groups(args.cache)
    train_success, val_success = split_groups(success, args.val_demos_per_class)
    train_failure, val_failure = split_groups(failure, args.val_demos_per_class)
    train_success = maybe_limit_groups(train_success, args.max_groups_per_class)
    train_failure = maybe_limit_groups(train_failure, args.max_groups_per_class)
    train_groups = train_success + train_failure
    sets = collect_latent_sets(args, wm, normalizers, train_groups, device)
    params, metrics = compute_contrastive_params(args, sets)
    eval_train = evaluate_sets(sets, params)

    np.savez(out / "contrastive_params.npz", **params)
    meta = args_to_jsonable(args)
    meta.update(
        {
            "mode": "fit",
            "device": str(device),
            "train_success_demos": len(train_success),
            "train_failure_demos": len(train_failure),
            "val_success_demos": len(val_success),
            "val_failure_demos": len(val_failure),
            "cache_rate_hz": 20,
            "wm_rate_hz": 20,
            "action_semantics": "raw 14D low20 env delta action, normalized by 20Hz WM normalizers",
            "obs": "tactile_mu + robot_joint_pos; joint normalized before WM encode",
            "wm_import": "train_threepiece_20hz_wm.ThreePieceWM",
            "length_min": int(min(lengths.values())),
            "length_max": int(max(lengths.values())),
        }
    )
    result = {"config": meta, "fit_metrics": metrics, "train_eval": eval_train}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def run_eval(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizers = load_normalizers(args.normalizers)
    wm = load_wm(args, device)
    success, failure, _ = list_cache_groups(args.cache)
    if args.eval_split == "val":
        _, success_groups = split_groups(success, args.val_demos_per_class)
        _, failure_groups = split_groups(failure, args.val_demos_per_class)
    elif args.eval_split == "train":
        success_groups, _ = split_groups(success, args.val_demos_per_class)
        failure_groups, _ = split_groups(failure, args.val_demos_per_class)
    else:
        success_groups, failure_groups = success, failure
    success_groups = maybe_limit_groups(success_groups, args.max_groups_per_class)
    failure_groups = maybe_limit_groups(failure_groups, args.max_groups_per_class)
    groups = success_groups + failure_groups
    params = load_contrastive_params(args.params)
    sets = collect_latent_sets(args, wm, normalizers, groups, device)
    result = {
        "mode": "eval",
        "eval_split": args.eval_split,
        "success_demos": len(success_groups),
        "failure_demos": len(failure_groups),
        "params": args.params,
        "summary": evaluate_sets(sets, params),
    }
    print(json.dumps(result, indent=2), flush=True)


def run_train_head(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    normalizers = load_normalizers(args.normalizers)
    params = load_contrastive_params(args.params)
    params_t = params_to_torch(params, device)
    success, failure, lengths = list_cache_groups(args.cache)
    train_success, val_success = split_groups(success, args.val_demos_per_class)
    train_failure, val_failure = split_groups(failure, args.val_demos_per_class)
    train_success = maybe_limit_groups(train_success, args.max_groups_per_class)
    train_failure = maybe_limit_groups(train_failure, args.max_groups_per_class)
    val_success = maybe_limit_groups(val_success, args.max_groups_per_class)
    val_failure = maybe_limit_groups(val_failure, args.max_groups_per_class)
    train_groups = train_success + train_failure
    val_groups = val_success + val_failure

    train_ds = ContrastiveWindowDataset(args.cache, train_groups, lengths, normalizers, args.history, args.chunk, args.stride)
    val_ds = ContrastiveWindowDataset(args.cache, val_groups, lengths, normalizers, args.history, args.chunk, args.stride)
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
        shuffle=False,
        num_workers=max(1, min(args.num_workers, 4)),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    wm = load_wm(args, device)
    head = SetpointValueHead(history=args.history, chunk=args.chunk).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    meta = args_to_jsonable(args)
    meta.update(
        {
            "mode": "train-head",
            "device": str(device),
            "train_demos": len(train_groups),
            "val_demos": len(val_groups),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "target": "V_alpha(z_H_pred) from frozen 20Hz WM rollout and contrastive_params.npz",
            "head_output": "logit trained with sigmoid + MSE to value target",
            "cache_rate_hz": 20,
            "wm_rate_hz": 20,
            "action_semantics": "raw 14D low20 env delta action, normalized by 20Hz WM normalizers",
        }
    )
    (out / "head_config.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out / "head_metrics.jsonl"
    print(json.dumps(meta, indent=2), flush=True)

    it = iter(train_loader)
    best = math.inf
    start_time = time.time()
    for step in range(1, args.steps + 1):
        head.train()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = to_device(batch, device)
        with torch.no_grad():
            z_hist, action_chunk, z_h = wm_rollout(wm, batch, args.history, args.chunk)
            target = torch_value_from_params(z_h, params_t)
        pred = torch.sigmoid(head(z_hist.detach(), action_chunk.detach(), z_h.detach()))
        mse = F.mse_loss(pred, target)
        loss = mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "time_s": time.time() - start_time,
                "lr": sched.get_last_lr()[0],
                "loss": float(loss.detach()),
                "mse": float(mse.detach()),
                "target_mean": float(target.mean().detach()),
                "pred_mean": float(pred.mean().detach()),
                "target_min": float(target.min().detach()),
                "target_max": float(target.max().detach()),
                "grad_norm": float(grad_norm),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_head(head, wm, val_loader, params_t, device, args)
            row = {"step": step, **metrics}
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[eval step {step}] {json.dumps(metrics)}", flush=True)
            if metrics["val_mse"] < best:
                best = metrics["val_mse"]
                torch.save({"value_head": head.state_dict(), "step": step, "config": meta, "metrics": metrics}, out / "value_head_best.pt")

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {"value_head": head.state_dict(), "step": step, "config": meta}
            torch.save(ckpt, out / f"value_head_step{step:06d}.pt")
            torch.save(ckpt, out / "value_head_latest.pt")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache", default="outputs/world_model_rollouts/threepiece_wm_50success_50failure_20hz_last_vae8k_mu_cache.hdf5")
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--val-demos-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encode-batch-steps", type=int, default=512)
    parser.add_argument("--late-fraction", type=float, default=0.20)
    parser.add_argument("--early-fraction", type=float, default=0.20)
    parser.add_argument("--max-points-per-group", type=int, default=0)
    parser.add_argument("--max-groups-per-class", type=int, default=0, help="debug/smoke limit; 0 uses all groups")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    fit = sub.add_parser("fit", help="fit P/v/beta*/sigma from cached 20Hz WM latents")
    add_common_args(fit)
    fit.add_argument("--out", required=True)
    fit.add_argument("--proj-dim", type=int, default=0, help="<=0 or >=latent_dim uses identity projection")
    fit.add_argument("--pca-center", action="store_true")
    fit.add_argument("--include-early-negative", action="store_true", default=True)
    fit.add_argument("--no-include-early-negative", dest="include_early_negative", action="store_false")
    fit.add_argument("--beta-star-mode", choices=["quantile", "mean"], default="quantile")
    fit.add_argument("--beta-star-quantile", type=float, default=0.80)
    fit.add_argument("--sigma", type=float, default=None)
    fit.add_argument("--sigma-scale", type=float, default=2.0)
    fit.add_argument("--sigma-gap-fraction", type=float, default=0.25)
    fit.add_argument("--direction", choices=["mean", "lda", "logistic"], default="mean")
    fit.add_argument("--value-mode", choices=["symmetric", "one_sided", "logistic"], default="symmetric")
    fit.add_argument(
        "--negative-mode",
        choices=["auto", "oldstyle", "lateonly", "failure_late_early"],
        default="auto",
        help="auto preserves --include-early-negative behavior",
    )
    fit.add_argument("--lda-shrinkage", type=float, default=0.05)
    fit.add_argument("--logistic-max-iter", type=int, default=200)
    fit.add_argument("--logistic-l2", type=float, default=1e-3)
    fit.set_defaults(func=run_fit)

    ev = sub.add_parser("eval", help="evaluate beta/V_alpha distributions for a fitted setpoint")
    add_common_args(ev)
    ev.add_argument("--params", required=True)
    ev.add_argument("--eval-split", choices=["train", "val", "all"], default="val")
    ev.set_defaults(func=run_eval)

    tr = sub.add_parser("train-head", help="train Q(z_hist, A_chunk, z_H)->V_alpha(z_H)")
    add_common_args(tr)
    tr.add_argument("--params", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--chunk", type=int, default=20)
    tr.add_argument("--stride", type=int, default=10)
    tr.add_argument("--steps", type=int, default=10000)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--grad-clip", type=float, default=10.0)
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--eval-batches", type=int, default=64)
    tr.add_argument("--save-every", type=int, default=2000)
    tr.set_defaults(func=run_train_head)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
