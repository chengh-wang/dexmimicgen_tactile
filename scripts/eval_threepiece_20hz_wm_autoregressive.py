#!/usr/bin/env python3
"""Evaluate 20 Hz ThreePiece WM with open-loop latent autoregressive rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402


def group_sort_key(name: str) -> tuple[int, int, str]:
    demo = name.split("__")[-1]
    prefix, idx = demo.rsplit("_", 1)
    return (0 if prefix == "success" else 1, int(idx), name)


@dataclass(frozen=True)
class WindowRef:
    group: str
    start: int


def split_eval_groups(cache_path: str, mode: str, val_demos_per_class: int) -> list[str]:
    success: list[str] = []
    failure: list[str] = []
    with h5py.File(cache_path, "r") as f:
        for key in sorted(f.keys(), key=group_sort_key):
            demo = key.split("__")[-1]
            if demo.startswith("success_"):
                success.append(key)
            elif demo.startswith("failure_"):
                failure.append(key)
    if mode == "all":
        return success + failure
    if mode == "val":
        if len(success) <= val_demos_per_class or len(failure) <= val_demos_per_class:
            raise ValueError("not enough demos for requested val split")
        return success[-val_demos_per_class:] + failure[-val_demos_per_class:]
    if mode == "train":
        return success[:-val_demos_per_class] + failure[:-val_demos_per_class]
    if mode == "success":
        return success
    if mode == "failure":
        return failure
    raise ValueError(f"unknown eval group mode: {mode}")


class CachedRolloutDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        groups: list[str],
        normalizers: dict[str, np.ndarray],
        history: int,
        horizon: int,
        stride: int,
        max_windows: int | None,
    ) -> None:
        self.cache_path = cache_path
        self.groups = groups
        self.normalizers = normalizers
        self.history = history
        self.horizon = horizon
        self.window = history + horizon
        self.refs: list[WindowRef] = []
        with h5py.File(cache_path, "r") as f:
            for group in groups:
                n = int(f[group]["action"].shape[0])
                for start in range(0, n - self.window + 1, stride):
                    self.refs.append(WindowRef(group, start))
        if max_windows is not None:
            self.refs = self.refs[:max_windows]
        if not self.refs:
            raise RuntimeError("no eval windows")
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        ref = self.refs[idx]
        g = self._file()[ref.group]
        end = ref.start + self.window
        tactile_mu = g["tactile_mu"][ref.start:end].astype(np.float32)
        joint = g["joint"][ref.start:end].astype(np.float32)
        action = g["action"][ref.start:end].astype(np.float32)
        joint = (joint - self.normalizers["joint_mean"]) / self.normalizers["joint_std"]
        action = (action - self.normalizers["action_mean"]) / self.normalizers["action_std"]
        return {
            "tactile_mu": torch.from_numpy(tactile_mu),
            "joint": torch.from_numpy(joint),
            "action": torch.from_numpy(action),
        }


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def eval_batch(wm: ThreePieceWM, batch: dict[str, torch.Tensor], history: int, horizon: int):
    z_true = wm.encode({"tactile_mu": batch["tactile_mu"], "joint": batch["joint"]})
    actions = batch["action"]

    z_window = z_true[:, :history]
    ar_preds: list[torch.Tensor] = []
    one_step_preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for k in range(horizon):
        a_win = wm.action(actions[:, k + 1 : k + 1 + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        ar_preds.append(next_z)

        teacher_pred = wm.predictor(z_true[:, k : k + history], a_win)[:, -1]
        one_step_preds.append(teacher_pred)

        target = z_true[:, history + k]
        targets.append(target)
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)

    ar = torch.stack(ar_preds, dim=1)
    one = torch.stack(one_step_preds, dim=1)
    target = torch.stack(targets, dim=1)
    ar_mse = (ar - target).square().mean(dim=-1)
    one_mse = (one - target).square().mean(dim=-1)
    ar_cos = F.cosine_similarity(ar, target, dim=-1)
    one_cos = F.cosine_similarity(one, target, dim=-1)
    return ar_mse, one_mse, ar_cos, one_cos


def summarize(values: torch.Tensor) -> dict[str, list[float] | float]:
    # values: (N, H)
    mean_h = values.mean(dim=0).detach().cpu().numpy()
    return {
        "mean_by_horizon": [float(x) for x in mean_h],
        "mean_all": float(values.mean().detach().cpu()),
        "final": float(mean_h[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-groups", choices=["val", "train", "all", "success", "failure"], default="val")
    parser.add_argument("--val-demos-per-class", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}
    groups = split_eval_groups(args.cache, args.eval_groups, args.val_demos_per_class)
    ds = CachedRolloutDataset(
        args.cache,
        groups,
        normalizers,
        args.history,
        args.horizon,
        args.stride,
        args.max_windows,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    wm = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    ckpt = torch.load(args.wm_ckpt, map_location=device)
    wm.load_state_dict(ckpt["model"], strict=True)
    wm.eval()

    ar_mse_all: list[torch.Tensor] = []
    one_mse_all: list[torch.Tensor] = []
    ar_cos_all: list[torch.Tensor] = []
    one_cos_all: list[torch.Tensor] = []
    for batch in loader:
        batch = to_device(batch, device)
        ar_mse, one_mse, ar_cos, one_cos = eval_batch(wm, batch, args.history, args.horizon)
        ar_mse_all.append(ar_mse.cpu())
        one_mse_all.append(one_mse.cpu())
        ar_cos_all.append(ar_cos.cpu())
        one_cos_all.append(one_cos.cpu())

    ar_mse_t = torch.cat(ar_mse_all, dim=0)
    one_mse_t = torch.cat(one_mse_all, dim=0)
    ar_cos_t = torch.cat(ar_cos_all, dim=0)
    one_cos_t = torch.cat(one_cos_all, dim=0)
    result = {
        "config": {
            **vars(args),
            "device": str(device),
            "num_groups": len(groups),
            "num_windows": len(ds),
            "rate_hz": 20,
            "action_semantics": "raw 14D low20 env delta action, normalized by 20Hz WM normalizers",
            "wm_import": "train_threepiece_20hz_wm.ThreePieceWM",
        },
        "autoregressive_mse": summarize(ar_mse_t),
        "teacher_one_step_mse": summarize(one_mse_t),
        "autoregressive_rmse": summarize(torch.sqrt(ar_mse_t)),
        "teacher_one_step_rmse": summarize(torch.sqrt(one_mse_t)),
        "autoregressive_cosine": summarize(ar_cos_t),
        "teacher_one_step_cosine": summarize(one_cos_t),
    }
    (out / "autoregressive_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
