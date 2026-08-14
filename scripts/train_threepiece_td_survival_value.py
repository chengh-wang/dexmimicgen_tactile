#!/usr/bin/env python3
"""Train a 20 Hz outcome-calibrated TD/survival value head for ThreePiece.

The head is action-conditioned:

  z_hist = frozen 20 Hz WM encoder(history obs)
  z_H    = frozen 20 Hz WM autoregressive rollout(z_hist, action window)
  Q(z_hist, A_chunk, z_H) -> probability-like survival/success target

The default target is discounted-terminal survival:

  success demo: y_t = gamma ** remaining_steps_after_rollout
  failure demo: y_t = 0

This keeps the 20 Hz action semantics explicit. The cache is expected to contain
one tactile_mu, joint, and raw low20 action per control step; joint/action are
normalized with the 20 Hz WM normalizers before encoding/rollout.

At decision time the latent history ends at index t. The action context is
history-1 previous actions plus chunk current/future actions. This makes
action_chunk[0] the action that maps z_t -> z_{t+1}, instead of shifting the
chunk one step into the future.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402


DEFAULT_CACHE = "outputs/world_model_rollouts/threepiece_wm_50success_50failure_20hz_last_vae8k_mu_cache.hdf5"
DEFAULT_WM_DIR = "outputs/threepiece_20hz_wm_vae8k_h4_attn_jointpos_bs512_520epoch_20260727"
DEFAULT_WM_CKPT = f"{DEFAULT_WM_DIR}/wm_best.pt"
DEFAULT_NORMALIZERS = f"{DEFAULT_WM_DIR}/normalizers.npz"
DEFAULT_VAE_CKPT = (
    "/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising/"
    "runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727/vae_step008000.pt"
)


def group_sort_key(name: str) -> tuple[int, int, str]:
    demo = name.split("__")[-1]
    if "_" in demo:
        prefix, idx = demo.rsplit("_", 1)
        try:
            return (0 if prefix == "success" else 1, int(idx), name)
        except ValueError:
            pass
    return (0 if "success" in demo else 1, 0, name)


def is_success_group(name: str) -> bool:
    return name.split("__")[-1].startswith("success")


@dataclass(frozen=True)
class WindowRef:
    group: str
    start: int
    length: int
    success: bool


def list_groups(cache_path: str) -> tuple[list[str], list[str], dict[str, int]]:
    success: list[str] = []
    failure: list[str] = []
    lengths: dict[str, int] = {}
    with h5py.File(cache_path, "r") as f:
        for key in sorted(f.keys(), key=group_sort_key):
            if "action" not in f[key] or "joint" not in f[key] or "tactile_mu" not in f[key]:
                continue
            n = int(f[key]["action"].shape[0])
            lengths[key] = n
            if is_success_group(key):
                success.append(key)
            else:
                failure.append(key)
    if not success or not failure:
        raise RuntimeError(f"expected success and failure groups in {cache_path}")
    return success, failure, lengths


def split_groups(groups: list[str], val_count: int) -> tuple[list[str], list[str]]:
    if val_count <= 0:
        return groups, []
    if len(groups) <= val_count:
        raise ValueError(f"not enough groups={len(groups)} for val_count={val_count}")
    return groups[:-val_count], groups[-val_count:]


class SurvivalQDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        groups: list[str],
        lengths: dict[str, int],
        normalizers: dict[str, np.ndarray],
        history: int,
        chunk: int,
        stride: int,
        target_mode: str,
        gamma: float,
        survival_horizon: int,
    ) -> None:
        self.cache_path = cache_path
        self.groups = groups
        self.lengths = lengths
        self.normalizers = normalizers
        self.history = history
        self.chunk = chunk
        self.state_window = history + chunk
        self.action_window = history - 1 + chunk
        self.stride = stride
        self.target_mode = target_mode
        self.gamma = gamma
        self.survival_horizon = survival_horizon
        self.refs: list[WindowRef] = []
        for group in groups:
            n = lengths[group]
            if n < self.state_window:
                continue
            success = is_success_group(group)
            for start in range(0, n - self.state_window + 1, stride):
                self.refs.append(WindowRef(group, start, n, success))
        if not self.refs:
            raise RuntimeError("no training windows built from cache")
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

    def _target(self, ref: WindowRef) -> tuple[float, int, int]:
        # z_H is the last latent after chunk autoregressive WM steps.
        # With a history ending at start+history-1, k=0 predicts start+history,
        # so after chunk steps the final prediction is start+history+chunk-1.
        z_h_index = min(ref.start + self.history + self.chunk - 1, ref.length - 1)
        remaining = max((ref.length - 1) - z_h_index, 0)
        if not ref.success:
            return 0.0, z_h_index, remaining
        if self.target_mode == "discounted_terminal":
            return float(self.gamma**remaining), z_h_index, remaining
        if self.target_mode == "within_horizon":
            return float(remaining <= self.survival_horizon), z_h_index, remaining
        raise ValueError(f"unknown target mode: {self.target_mode}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.refs[idx]
        g = self._file()[ref.group]
        action_end = ref.start + self.action_window
        tactile_mu = g["tactile_mu"][ref.start : ref.start + self.history].astype(np.float32)
        joint = g["joint"][ref.start : ref.start + self.history].astype(np.float32)
        action = g["action"][ref.start:action_end].astype(np.float32)
        if action.shape[0] != self.action_window:
            raise RuntimeError(f"bad action window {action.shape[0]} != {self.action_window} for {ref.group}:{ref.start}")

        joint = (joint - self.normalizers["joint_mean"]) / self.normalizers["joint_std"]
        action = (action - self.normalizers["action_mean"]) / self.normalizers["action_std"]
        target, z_h_index, remaining = self._target(ref)
        return {
            "tactile_mu": torch.from_numpy(tactile_mu),
            "joint": torch.from_numpy(joint),
            "action": torch.from_numpy(action),
            "target": torch.tensor(target, dtype=torch.float32),
            "outcome": torch.tensor(float(ref.success), dtype=torch.float32),
            "remaining": torch.tensor(float(remaining), dtype=torch.float32),
            "progress": torch.tensor(float(z_h_index / max(ref.length - 1, 1)), dtype=torch.float32),
        }


def to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


class SurvivalQHead(nn.Module):
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
def wm_rollout(
    wm: ThreePieceWM,
    sample: dict[str, torch.Tensor],
    history: int,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_hist = wm.encode({"tactile_mu": sample["tactile_mu"], "joint": sample["joint"]})
    z_window = z_hist
    actions = sample["action"]
    if actions.shape[1] != history - 1 + chunk:
        raise RuntimeError(f"expected action context length {history - 1 + chunk}, got {actions.shape[1]}")
    for k in range(chunk):
        a_win = wm.action(actions[:, k : k + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    action_chunk = actions[:, history - 1 : history - 1 + chunk]
    return z_hist, action_chunk, z_window[:, -1]


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


def reliability_bins(scores: list[float], targets: list[float], bins: int = 10) -> dict[str, float]:
    out: dict[str, float] = {}
    s = np.asarray(scores, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    for bi in range(bins):
        lo = bi / bins
        hi = (bi + 1) / bins
        mask = (s >= lo) & (s < hi if bi < bins - 1 else s <= hi)
        if not np.any(mask):
            continue
        out[f"cal_bin{bi}_count"] = int(mask.sum())
        out[f"cal_bin{bi}_score"] = float(s[mask].mean())
        out[f"cal_bin{bi}_target"] = float(y[mask].mean())
    return out


def evaluate(
    q: SurvivalQHead,
    wm: ThreePieceWM,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    q.eval()
    scores: list[float] = []
    targets_all: list[float] = []
    outcomes: list[int] = []
    total_loss = 0.0
    total_mse = 0.0
    total = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.eval_batches:
                break
            batch = to_device(batch, device)
            z_hist, action_chunk, z_h = wm_rollout(wm, batch, args.history, args.chunk)
            logits = q(z_hist, action_chunk, z_h)
            targets = batch["target"]
            probs = torch.sigmoid(logits)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            mse = F.mse_loss(probs, targets)
            total_loss += float(bce) * logits.numel()
            total_mse += float(mse) * logits.numel()
            total += logits.numel()
            scores.extend(probs.detach().cpu().tolist())
            targets_all.extend(targets.detach().cpu().tolist())
            outcomes.extend(batch["outcome"].detach().cpu().int().tolist())

    if not scores:
        return {}
    scores_np = np.asarray(scores, dtype=np.float32)
    targets_np = np.asarray(targets_all, dtype=np.float32)
    outcomes_np = np.asarray(outcomes, dtype=np.int32)
    pos = scores_np[outcomes_np == 1]
    neg = scores_np[outcomes_np == 0]
    metrics: dict[str, float] = {
        "val_bce": total_loss / max(total, 1),
        "val_brier_mse": total_mse / max(total, 1),
        "val_auc_outcome": binary_auc(scores, outcomes),
        "val_score_mean": float(scores_np.mean()),
        "val_target_mean": float(targets_np.mean()),
        "val_success_score_mean": float(pos.mean()) if pos.size else float("nan"),
        "val_failure_score_mean": float(neg.mean()) if neg.size else float("nan"),
        "val_success_target_mean": float(targets_np[outcomes_np == 1].mean()) if pos.size else float("nan"),
        "val_failure_target_mean": float(targets_np[outcomes_np == 0].mean()) if neg.size else float("nan"),
    }
    metrics.update(reliability_bins(scores, targets_all, bins=args.reliability_bins))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--wm-ckpt", default=DEFAULT_WM_CKPT)
    parser.add_argument("--vae-ckpt", default=DEFAULT_VAE_CKPT)
    parser.add_argument("--normalizers", default=DEFAULT_NORMALIZERS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--val-demos-per-class", type=int, default=10)
    parser.add_argument("--target-mode", choices=["discounted_terminal", "within_horizon"], default="discounted_terminal")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--survival-horizon", type=int, default=80)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}
    success_groups, failure_groups, lengths = list_groups(args.cache)
    train_success, val_success = split_groups(success_groups, args.val_demos_per_class)
    train_failure, val_failure = split_groups(failure_groups, args.val_demos_per_class)
    train_groups = train_success + train_failure
    val_groups = val_success + val_failure

    train_ds = SurvivalQDataset(
        args.cache,
        train_groups,
        lengths,
        normalizers,
        args.history,
        args.chunk,
        args.stride,
        args.target_mode,
        args.gamma,
        args.survival_horizon,
    )
    val_ds = SurvivalQDataset(
        args.cache,
        val_groups,
        lengths,
        normalizers,
        args.history,
        args.chunk,
        args.stride,
        args.target_mode,
        args.gamma,
        args.survival_horizon,
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

    q = SurvivalQHead(history=args.history, chunk=args.chunk).to(device)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    meta = vars(args).copy()
    meta.update(
        {
            "device": str(device),
            "train_success_demos": len(train_success),
            "train_failure_demos": len(train_failure),
            "val_success_demos": len(val_success),
            "val_failure_demos": len(val_failure),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "head_type": "action_conditioned_survival_q",
            "target_definition": {
                "discounted_terminal": "success y = gamma ** remaining_20hz_steps_after_zH; failure y = 0",
                "within_horizon": "success y = 1 if terminal within survival_horizon after zH else 0; failure y = 0",
            }[args.target_mode],
            "wm_rate_hz": 20,
            "cache_rate_hz": 20,
            "action_semantics": (
                "raw 14D low20 env delta action, normalized by 20Hz WM normalizers; "
                "action context is history-1 past actions plus chunk current/future actions"
            ),
            "wm_import": "train_threepiece_20hz_wm.ThreePieceWM",
        }
    )
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out / "metrics.jsonl"
    print(json.dumps(meta, indent=2), flush=True)

    it = iter(train_loader)
    best_score = math.inf
    start_time = time.time()
    for step in range(1, args.steps + 1):
        q.train()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = to_device(batch, device)
        z_hist, action_chunk, z_h = wm_rollout(wm, batch, args.history, args.chunk)
        logits = q(z_hist, action_chunk, z_h)
        targets = batch["target"]
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        mse = F.mse_loss(probs, targets)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()
        sched.step()

        if step == 1 or step % 100 == 0:
            outcome = batch["outcome"]
            success_mask = outcome > 0.5
            failure_mask = ~success_mask
            row = {
                "step": step,
                "time_s": time.time() - start_time,
                "lr": sched.get_last_lr()[0],
                "loss": float(loss.detach()),
                "bce": float(loss.detach()),
                "brier_mse": float(mse.detach()),
                "score_mean": float(probs.mean().detach()),
                "target_mean": float(targets.mean().detach()),
                "success_score": float(probs[success_mask].mean().detach()) if success_mask.any() else float("nan"),
                "failure_score": float(probs[failure_mask].mean().detach()) if failure_mask.any() else float("nan"),
                "remaining_mean": float(batch["remaining"].mean().detach()),
                "progress_mean": float(batch["progress"].mean().detach()),
                "grad_norm": float(grad_norm),
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
            # Brier/MSE is the calibration criterion; AUC is diagnostic only.
            if metrics and metrics["val_brier_mse"] < best_score:
                best_score = metrics["val_brier_mse"]
                torch.save({"q_head": q.state_dict(), "step": step, "config": meta, "metrics": metrics}, out / "q_best.pt")

        if step % args.save_every == 0 or step == args.steps:
            ck = {"q_head": q.state_dict(), "step": step, "config": meta}
            torch.save(ck, out / f"q_step{step:06d}.pt")
            torch.save(ck, out / "q_latest.pt")


if __name__ == "__main__":
    main()
