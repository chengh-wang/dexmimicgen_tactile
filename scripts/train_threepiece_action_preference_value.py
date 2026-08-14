#!/usr/bin/env python3
"""Train a counterfactual action-preference value head for ThreePiece 20 Hz WM.

For the same encoded history z_hist, the demonstrated future action chunk from
successful rollouts is trained to score above corrupted counterfactual chunks:

    Q(z_hist, A_demo, WM(z_hist, A_demo)) >
    Q(z_hist, A_bad,  WM(z_hist, A_bad))

The WM is frozen. Actions are raw low20 14D env delta commands in the cache and
are normalized with the 20 Hz WM normalizers before being passed to the WM and
Q action encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402


DEFAULT_CACHE = "outputs/world_model_rollouts/threepiece_wm_50success_50failure_20hz_last_vae8k_mu_cache.hdf5"
DEFAULT_WM_CKPT = "outputs/threepiece_20hz_wm_vae8k_h4_attn_jointpos_bs512_520epoch_20260727/wm_best.pt"
DEFAULT_NORMALIZERS = "outputs/threepiece_20hz_wm_vae8k_h4_attn_jointpos_bs512_520epoch_20260727/normalizers.npz"
DEFAULT_VAE_CKPT = (
    "/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/"
    "runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727/vae_step008000.pt"
)
DEFAULT_OUT = "outputs/threepiece_action_preference_value_20hz_10k"
VALID_NEG_MODES = {"time_shift", "reverse", "noise", "arm_desync"}


def group_sort_key(name: str) -> tuple[int, int, str]:
    demo = name.split("__")[-1]
    try:
        prefix, idx = demo.rsplit("_", 1)
        order = 0 if prefix == "success" else 1 if prefix == "failure" else 2
        return order, int(idx), name
    except Exception:
        return 2, 0, name


def list_groups(cache_path: str) -> tuple[list[str], list[str], dict[str, int]]:
    lengths: dict[str, int] = {}
    success: list[str] = []
    failure: list[str] = []
    with h5py.File(cache_path, "r") as f:
        for key in sorted(f.keys(), key=group_sort_key):
            n = int(f[key]["action"].shape[0])
            lengths[key] = n
            demo = key.split("__")[-1]
            if demo.startswith("success_"):
                success.append(key)
            elif demo.startswith("failure_"):
                failure.append(key)
    if not success:
        raise RuntimeError(f"expected at least one success_* group in {cache_path}")
    return success, failure, lengths


def split_groups(groups: list[str], val_count: int) -> tuple[list[str], list[str]]:
    if val_count <= 0:
        return groups, []
    if len(groups) <= val_count:
        raise ValueError(f"not enough groups={len(groups)} for val_count={val_count}")
    return groups[:-val_count], groups[-val_count:]


def parse_modes(value: str) -> list[str]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    bad = sorted(set(modes) - VALID_NEG_MODES)
    if bad:
        raise ValueError(f"unknown negative modes {bad}; valid={sorted(VALID_NEG_MODES)}")
    if not modes:
        raise ValueError("at least one negative mode is required")
    return modes


def binary_auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
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
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass(frozen=True)
class PrefRef:
    group: str
    start: int
    length: int
    mode: str


class ActionPreferenceDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        success_groups: list[str],
        lengths: dict[str, int],
        normalizers: dict[str, np.ndarray],
        history: int,
        chunk: int,
        stride: int,
        neg_modes: list[str],
        min_time_shift: int,
        noise_std_raw: float,
        arm_desync_delay: int,
        seed: int,
    ) -> None:
        self.cache_path = cache_path
        self.success_groups = success_groups
        self.lengths = lengths
        self.normalizers = normalizers
        self.history = history
        self.chunk = chunk
        self.obs_window = history
        self.action_window = history - 1 + chunk
        self.stride = stride
        self.neg_modes = neg_modes
        self.min_time_shift = min_time_shift
        self.noise_std_raw = noise_std_raw
        self.arm_desync_delay = arm_desync_delay
        self.seed = seed
        self.refs: list[PrefRef] = []
        for group in success_groups:
            n = lengths[group]
            for start in range(0, n - max(self.obs_window, self.action_window) + 1, stride):
                for mode in neg_modes:
                    self.refs.append(PrefRef(group, start, n, mode))
        if not self.refs:
            raise RuntimeError("no action-preference windows")
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

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + idx * 1009)

    def _time_shift_start(self, ref: PrefRef, idx: int) -> int:
        max_start = ref.length - max(self.obs_window, self.action_window)
        if max_start <= 0:
            return ref.start
        far: list[int] = [
            s
            for s in range(0, max_start + 1, self.stride)
            if abs(s - ref.start) >= self.min_time_shift
        ]
        if not far:
            fallback = 0 if ref.start > max_start // 2 else max_start
            return fallback if fallback != ref.start else max(0, max_start - ref.start)
        rng = self._rng(idx)
        return int(far[rng.integers(0, len(far))])

    @staticmethod
    def _shift_with_edge(x: np.ndarray, delay: int) -> np.ndarray:
        if delay == 0:
            return x.copy()
        n = x.shape[0]
        d = min(abs(delay), n - 1)
        y = np.empty_like(x)
        if delay > 0:
            y[:d] = x[:1]
            y[d:] = x[:-d]
        else:
            y[-d:] = x[-1:]
            y[:-d] = x[d:]
        return y

    def _make_negative(self, actions: h5py.Dataset, ref: PrefRef, idx: int, pos_raw: np.ndarray) -> np.ndarray:
        if ref.mode == "time_shift":
            start = self._time_shift_start(ref, idx)
            return actions[start : start + self.action_window].astype(np.float32)
        if ref.mode == "reverse":
            return pos_raw[::-1].copy()
        if ref.mode == "noise":
            rng = self._rng(idx)
            noise = rng.normal(0.0, self.noise_std_raw, size=pos_raw.shape).astype(np.float32)
            return (pos_raw + noise).astype(np.float32)
        if ref.mode == "arm_desync":
            if pos_raw.shape[-1] < 14:
                return pos_raw[::-1].copy()
            neg = pos_raw.copy()
            d = self.arm_desync_delay
            neg[:, :7] = self._shift_with_edge(pos_raw[:, :7], d)
            neg[:, 7:14] = self._shift_with_edge(pos_raw[:, 7:14], -d)
            return neg.astype(np.float32)
        raise ValueError(f"unsupported negative mode: {ref.mode}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ref = self.refs[idx]
        g = self._file()[ref.group]
        hist_end = ref.start + self.history
        action_end = ref.start + self.action_window
        tactile_mu = g["tactile_mu"][ref.start:hist_end].astype(np.float32)
        joint = g["joint"][ref.start:hist_end].astype(np.float32)
        pos_raw = g["action"][ref.start:action_end].astype(np.float32)
        neg_raw = self._make_negative(g["action"], ref, idx, pos_raw)

        joint = (joint - self.normalizers["joint_mean"]) / self.normalizers["joint_std"]
        pos_action = (pos_raw - self.normalizers["action_mean"]) / self.normalizers["action_std"]
        neg_action = (neg_raw - self.normalizers["action_mean"]) / self.normalizers["action_std"]
        return {
            "tactile_mu": torch.from_numpy(tactile_mu),
            "joint": torch.from_numpy(joint.astype(np.float32)),
            "pos_action": torch.from_numpy(pos_action.astype(np.float32)),
            "neg_action": torch.from_numpy(neg_action.astype(np.float32)),
            "mode": ref.mode,
            "group": ref.group,
            "start": torch.tensor(ref.start, dtype=torch.long),
        }


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch


class QHead(nn.Module):
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


@torch.no_grad()
def encode_history(wm: ThreePieceWM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return wm.encode({"tactile_mu": batch["tactile_mu"], "joint": batch["joint"]})


@torch.no_grad()
def wm_rollout_from_z(
    wm: ThreePieceWM,
    z_hist: torch.Tensor,
    actions: torch.Tensor,
    history: int,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = history - 1 + chunk
    if actions.shape[1] != expected:
        raise RuntimeError(f"expected action sequence length {expected}, got {actions.shape[1]}")
    z_window = z_hist
    for k in range(chunk):
        a_win = wm.action(actions[:, k : k + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    action_chunk = actions[:, history - 1 : history - 1 + chunk]
    return action_chunk, z_window[:, -1]


def compute_logits(
    q: QHead,
    wm: ThreePieceWM,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_hist = encode_history(wm, batch)
    pos_chunk, z_pos = wm_rollout_from_z(wm, z_hist, batch["pos_action"], args.history, args.chunk)
    neg_chunk, z_neg = wm_rollout_from_z(wm, z_hist, batch["neg_action"], args.history, args.chunk)
    q_pos = q(z_hist, pos_chunk, z_pos)
    q_neg = q(z_hist, neg_chunk, z_neg)
    return q_pos, q_neg


def mode_metrics(q_pos: torch.Tensor, q_neg: torch.Tensor, modes: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    pos = q_pos.detach().cpu()
    neg = q_neg.detach().cpu()
    for mode in sorted(set(modes)):
        mask = torch.tensor([m == mode for m in modes], dtype=torch.bool)
        if int(mask.sum()) == 0:
            continue
        out[f"pair_acc_{mode}"] = float((pos[mask] > neg[mask]).float().mean())
        out[f"margin_{mode}"] = float((pos[mask] - neg[mask]).mean())
    return out


def evaluate(q: QHead, wm: ThreePieceWM, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    q.eval()
    total_rank = 0.0
    total_bce = 0.0
    total_pair = 0
    total_pair_ok = 0
    scores: list[float] = []
    labels: list[int] = []
    per_mode_ok: dict[str, int] = {}
    per_mode_total: dict[str, int] = {}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.eval_batches:
                break
            batch = to_device(batch, device)
            q_pos, q_neg = compute_logits(q, wm, batch, args)
            rank = F.softplus(-(q_pos - q_neg - args.margin)).mean()
            logits = torch.cat([q_pos, q_neg], dim=0)
            targets = torch.cat([torch.ones_like(q_pos), torch.zeros_like(q_neg)], dim=0)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            total_rank += float(rank) * q_pos.numel()
            total_bce += float(bce) * logits.numel()
            total_pair += q_pos.numel()
            total_pair_ok += int((q_pos > q_neg).sum().item())
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
            labels.extend([1] * q_pos.numel() + [0] * q_neg.numel())
            for ok, mode in zip((q_pos > q_neg).detach().cpu().tolist(), batch["mode"]):
                per_mode_total[mode] = per_mode_total.get(mode, 0) + 1
                per_mode_ok[mode] = per_mode_ok.get(mode, 0) + int(bool(ok))

    preds = [1 if s >= 0.5 else 0 for s in scores]
    acc = sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
    metrics = {
        "val_rank_loss": total_rank / max(total_pair, 1),
        "val_bce": total_bce / max(2 * total_pair, 1),
        "val_pair_acc": total_pair_ok / max(total_pair, 1),
        "val_auc": binary_auc(scores, labels),
        "val_acc": acc,
    }
    for mode in sorted(per_mode_total):
        metrics[f"val_pair_acc_{mode}"] = per_mode_ok[mode] / max(per_mode_total[mode], 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--wm-ckpt", default=DEFAULT_WM_CKPT)
    parser.add_argument("--vae-ckpt", default=DEFAULT_VAE_CKPT)
    parser.add_argument("--normalizers", default=DEFAULT_NORMALIZERS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--val-demos-per-class", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--bce-weight", type=float, default=0.1)
    parser.add_argument("--neg-modes", default="time_shift,reverse,noise,arm_desync")
    parser.add_argument("--min-time-shift", type=int, default=60)
    parser.add_argument("--noise-std-raw", type=float, default=0.02)
    parser.add_argument("--arm-desync-delay", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.neg_modes = parse_modes(args.neg_modes)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}
    success_groups, failure_groups, lengths = list_groups(args.cache)
    train_success, val_success = split_groups(success_groups, args.val_demos_per_class)
    train_ds = ActionPreferenceDataset(
        args.cache,
        train_success,
        lengths,
        normalizers,
        args.history,
        args.chunk,
        args.stride,
        args.neg_modes,
        args.min_time_shift,
        args.noise_std_raw,
        args.arm_desync_delay,
        args.seed,
    )
    val_ds = ActionPreferenceDataset(
        args.cache,
        val_success,
        lengths,
        normalizers,
        args.history,
        args.chunk,
        args.stride,
        args.neg_modes,
        args.min_time_shift,
        args.noise_std_raw,
        args.arm_desync_delay,
        args.seed + 100000,
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
        shuffle=False,
        num_workers=max(1, min(args.num_workers, 4)),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    wm = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    ckpt = torch.load(args.wm_ckpt, map_location=device)
    wm.load_state_dict(ckpt["model"], strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    q = QHead(history=args.history, chunk=args.chunk).to(device)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    meta = vars(args).copy()
    meta.update(
        {
            "device": str(device),
            "train_success_demos": len(train_success),
            "val_success_demos": len(val_success),
            "available_failure_demos_unused": len(failure_groups),
            "train_pairs": len(train_ds),
            "val_pairs": len(val_ds),
            "wm_rate_hz": 20,
            "cache_rate_hz": 20,
            "objective": "same-history counterfactual action preference",
            "label": "positive=successful demo future action chunk; negative=corrupted action chunk",
            "action_semantics": "raw 14D low20 env delta action, normalized by 20Hz WM normalizers",
            "wm_import": "train_threepiece_20hz_wm.ThreePieceWM",
        }
    )
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out / "metrics.jsonl"
    print(json.dumps(meta, indent=2), flush=True)

    it = iter(train_loader)
    best_score = -math.inf
    start_time = time.time()
    for step in range(1, args.steps + 1):
        q.train()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = to_device(batch, device)
        q_pos, q_neg = compute_logits(q, wm, batch, args)
        rank = F.softplus(-(q_pos - q_neg - args.margin)).mean()
        logits = torch.cat([q_pos, q_neg], dim=0)
        targets = torch.cat([torch.ones_like(q_pos), torch.zeros_like(q_neg)], dim=0)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss = rank + args.bce_weight * bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()
        sched.step()

        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "time_s": time.time() - start_time,
                "lr": sched.get_last_lr()[0],
                "loss": float(loss.detach()),
                "rank_loss": float(rank.detach()),
                "bce": float(bce.detach()),
                "pair_acc": float((q_pos > q_neg).float().mean().detach()),
                "margin_mean": float((q_pos - q_neg).mean().detach()),
                "pos_score": float(torch.sigmoid(q_pos).mean().detach()),
                "neg_score": float(torch.sigmoid(q_neg).mean().detach()),
                "grad_norm": float(grad_norm),
                **mode_metrics(q_pos, q_neg, batch["mode"]),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(q, wm, val_loader, device, args)
            row = {"step": step, **metrics}
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[eval step {step}] {json.dumps(metrics)}", flush=True)
            score = metrics["val_pair_acc"] + 1e-3 * (metrics["val_auc"] if not math.isnan(metrics["val_auc"]) else 0.0)
            if score > best_score:
                best_score = score
                torch.save({"q_head": q.state_dict(), "step": step, "config": meta, "metrics": metrics}, out / "q_best.pt")

        if step % args.save_every == 0 or step == args.steps:
            ck = {"q_head": q.state_dict(), "step": step, "config": meta}
            torch.save(ck, out / f"q_step{step:06d}.pt")
            torch.save(ck, out / "q_latest.pt")


if __name__ == "__main__":
    main()
