#!/usr/bin/env python3
"""Train a 20 Hz value/reward head on existing ThreePiece 50/50 rollouts.

This uses frozen WM dynamics:
  z_hist = WM.encode(history obs)
  zH_pred = WM rollout(z_hist, A_chunk)
  Q(z_hist, A_chunk, zH_pred) -> success value/logit

No new negatives are generated here. Negatives are the existing failed rollouts,
paired with successful rollouts at the same start index.

This script is intentionally bound to train_threepiece_20hz_wm. The cache is
expected to contain one tactile_mu, joint, and raw low20 action per control
step; the action is normalized with the 20 Hz WM normalizers before rollout.
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


def group_sort_key(name: str) -> tuple[int, int]:
    # cache keys look like: file_stem__success_000 or file_stem__failure_000
    demo = name.split("__")[-1]
    prefix, idx = demo.rsplit("_", 1)
    return (0 if prefix == "success" else 1, int(idx))


@dataclass(frozen=True)
class SampleRef:
    group: str
    start: int
    length: int


def list_groups(cache_path: str) -> tuple[list[str], list[str], dict[str, int]]:
    lengths: dict[str, int] = {}
    pos: list[str] = []
    neg: list[str] = []
    with h5py.File(cache_path, "r") as f:
        for key in sorted(f.keys(), key=group_sort_key):
            demo = key.split("__")[-1]
            n = int(f[key]["action"].shape[0])
            lengths[key] = n
            if demo.startswith("success_"):
                pos.append(key)
            elif demo.startswith("failure_"):
                neg.append(key)
    if not pos or not neg:
        raise RuntimeError(f"expected success and failure groups in {cache_path}")
    return pos, neg, lengths


def split_groups(groups: list[str], val_count: int) -> tuple[list[str], list[str]]:
    if val_count <= 0:
        return groups, []
    if len(groups) <= val_count:
        raise ValueError(f"not enough groups={len(groups)} for val_count={val_count}")
    return groups[:-val_count], groups[-val_count:]


class PairedQDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        pos_groups: list[str],
        neg_groups: list[str],
        lengths: dict[str, int],
        normalizers: dict[str, np.ndarray],
        history: int,
        chunk: int,
        stride: int,
    ) -> None:
        self.cache_path = cache_path
        self.pos_groups = pos_groups
        self.neg_groups = neg_groups
        self.lengths = lengths
        self.normalizers = normalizers
        self.history = history
        self.chunk = chunk
        self.window = history + chunk
        self.stride = stride
        self.pos_refs: list[SampleRef] = []
        for group in pos_groups:
            n = lengths[group]
            for start in range(0, n - self.window + 1, stride):
                self.pos_refs.append(SampleRef(group, start, n))
        if not self.pos_refs:
            raise RuntimeError("no positive windows")
        self._cache: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.pos_refs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def _file(self) -> h5py.File:
        if self._cache is None:
            self._cache = h5py.File(self.cache_path, "r")
        return self._cache

    def _read_one(self, group: str, start: int, target: float) -> dict[str, torch.Tensor]:
        f = self._file()
        g = f[group]
        end = start + self.window
        tactile_mu = g["tactile_mu"][start : start + self.history].astype(np.float32)
        joint = g["joint"][start : start + self.history].astype(np.float32)
        action = g["action"][start:end].astype(np.float32)
        joint = (joint - self.normalizers["joint_mean"]) / self.normalizers["joint_std"]
        action = (action - self.normalizers["action_mean"]) / self.normalizers["action_std"]
        return {
            "tactile_mu": torch.from_numpy(tactile_mu),
            "joint": torch.from_numpy(joint),
            "action": torch.from_numpy(action),
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        pos_ref = self.pos_refs[idx]
        neg_group = self.neg_groups[idx % len(self.neg_groups)]
        neg_n = self.lengths[neg_group]
        neg_start = min(pos_ref.start, neg_n - self.window)
        # Ramp target discourages a pure success-trajectory classifier.
        pos_target = float((pos_ref.start + self.chunk) / max(pos_ref.length - 1, 1))
        pos_target = min(max(pos_target, 0.05), 1.0)
        return {
            "pos": self._read_one(pos_ref.group, pos_ref.start, pos_target),
            "neg": self._read_one(neg_group, neg_start, 0.0),
        }


def to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


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
def wm_rollout(wm: ThreePieceWM, sample: dict[str, torch.Tensor], history: int, chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_hist = wm.encode({"tactile_mu": sample["tactile_mu"], "joint": sample["joint"]})
    z_window = z_hist
    actions = sample["action"]
    for k in range(chunk):
        a_win = wm.action(actions[:, k : k + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    action_chunk = actions[:, history : history + chunk]
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


def evaluate(q: QHead, wm: ThreePieceWM, loader: DataLoader, device: torch.device, args) -> dict[str, float]:
    q.eval()
    scores: list[float] = []
    labels: list[int] = []
    total_bce = 0.0
    total_rank = 0.0
    total_pair = 0
    total_pair_ok = 0
    total = 0
    for bi, batch in enumerate(loader):
        if bi >= args.eval_batches:
            break
        batch = to_device(batch, device)
        pos = batch["pos"]
        neg = batch["neg"]
        with torch.no_grad():
            zhp, ap, zfp = wm_rollout(wm, pos, args.history, args.chunk)
            zhn, an, zfn = wm_rollout(wm, neg, args.history, args.chunk)
            lp = q(zhp, ap, zfp)
            ln = q(zhn, an, zfn)
            logits = torch.cat([lp, ln], dim=0)
            targets = torch.cat([pos["target"], neg["target"]], dim=0)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            rank = F.softplus(-(lp - ln - args.margin)).mean()
        total_bce += float(bce) * logits.numel()
        total_rank += float(rank) * lp.numel()
        total += logits.numel()
        total_pair += lp.numel()
        total_pair_ok += int((lp > ln).sum().item())
        scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend([1] * lp.numel() + [0] * ln.numel())
    preds = [1 if s >= 0.5 else 0 for s in scores]
    acc = sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
    return {
        "val_bce": total_bce / max(total, 1),
        "val_rank_loss": total_rank / max(total_pair, 1),
        "val_auc": binary_auc(scores, labels),
        "val_acc": acc,
        "val_pair_acc": total_pair_ok / max(total_pair, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--val-demos-per-class", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}
    pos_groups, neg_groups, lengths = list_groups(args.cache)
    train_pos, val_pos = split_groups(pos_groups, args.val_demos_per_class)
    train_neg, val_neg = split_groups(neg_groups, args.val_demos_per_class)
    train_ds = PairedQDataset(args.cache, train_pos, train_neg, lengths, normalizers, args.history, args.chunk, args.stride)
    val_ds = PairedQDataset(args.cache, val_pos, val_neg, lengths, normalizers, args.history, args.chunk, args.stride)
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
            "train_pos_demos": len(train_pos),
            "train_neg_demos": len(train_neg),
            "val_pos_demos": len(val_pos),
            "val_neg_demos": len(val_neg),
            "train_pairs": len(train_ds),
            "val_pairs": len(val_ds),
            "label": "success ramp target for positive, 0 for failed rollout",
            "wm_rate_hz": 20,
            "cache_rate_hz": 20,
            "action_semantics": "raw 14D low20 env delta action, normalized by 20Hz WM normalizers",
            "wm_import": "train_threepiece_20hz_wm.ThreePieceWM",
        }
    )
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out / "metrics.jsonl"
    print(json.dumps(meta, indent=2), flush=True)

    it = iter(train_loader)
    best_auc = -math.inf
    start_time = time.time()
    for step in range(1, args.steps + 1):
        q.train()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = to_device(batch, device)
        pos = batch["pos"]
        neg = batch["neg"]
        with torch.no_grad():
            zhp, ap, zfp = wm_rollout(wm, pos, args.history, args.chunk)
            zhn, an, zfn = wm_rollout(wm, neg, args.history, args.chunk)
        lp = q(zhp, ap, zfp)
        ln = q(zhn, an, zfn)
        logits = torch.cat([lp, ln], dim=0)
        targets = torch.cat([pos["target"], neg["target"]], dim=0)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        rank = F.softplus(-(lp - ln - args.margin)).mean()
        loss = bce + args.rank_weight * rank
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
                "bce": float(bce.detach()),
                "rank_loss": float(rank.detach()),
                "pair_acc": float((lp > ln).float().mean().detach()),
                "pos_score": float(torch.sigmoid(lp).mean().detach()),
                "neg_score": float(torch.sigmoid(ln).mean().detach()),
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
            if metrics["val_auc"] > best_auc:
                best_auc = metrics["val_auc"]
                torch.save({"q_head": q.state_dict(), "step": step, "config": meta, "metrics": metrics}, out / "q_best.pt")

        if step % args.save_every == 0 or step == args.steps:
            ck = {"q_head": q.state_dict(), "step": step, "config": meta}
            torch.save(ck, out / f"q_step{step:06d}.pt")
            torch.save(ck, out / "q_latest.pt")


if __name__ == "__main__":
    main()
