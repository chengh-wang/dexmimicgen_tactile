#!/usr/bin/env python3
"""Train a 20 Hz latent WM for ThreePieceAssembly tactile rollouts.

This intentionally matches the exp038-style structure:
frozen tactile VAE -> trainable tactile head/projector -> attention fusion with
joint_pos token -> action-conditioned history predictor.

Unlike train_threepiece_high200_wm.py, this script keeps the model at the
policy/control rate. Each sample uses one low20 action per step and one tactile
/ joint observation per step, pooled from the ten high200 samples in that
control interval.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset


def demo_sort_key(name: str) -> int:
    return int(name.split("_")[1])


@dataclass(frozen=True)
class DemoRef:
    path: str
    demo: str
    length20: int


def cache_key(ref: DemoRef) -> str:
    return f"{Path(ref.path).stem}__{ref.demo}"


def data_root(f: h5py.File) -> h5py.Group:
    return f["data"] if "data" in f else f


def low20_action_dataset(demo: h5py.Group) -> h5py.Dataset:
    if "low20/action" in demo:
        return demo["low20/action"]
    if "low20/actions" in demo:
        return demo["low20/actions"]
    raise KeyError("expected low20/action or low20/actions")


def collect_demo_refs(paths: list[str]) -> list[DemoRef]:
    refs: list[DemoRef] = []
    for path in paths:
        with h5py.File(path, "r") as f:
            root = data_root(f)
            for demo in sorted(root.keys(), key=demo_sort_key):
                length20 = int(low20_action_dataset(root[demo]).shape[0])
                refs.append(DemoRef(path, demo, length20))
    if not refs:
        raise RuntimeError("no demos found")
    return refs


def split_demo_refs(paths: list[str], val_ratio: float) -> tuple[list[DemoRef], list[DemoRef]]:
    refs = collect_demo_refs(paths)
    n_val = int(round(len(refs) * val_ratio))
    n_val = max(1, n_val) if val_ratio > 0 else 0
    return refs[:-n_val] if n_val else refs, refs[-n_val:] if n_val else []


def pool_high200(block: np.ndarray, mode: str) -> np.ndarray:
    """Pool (Tlow, 10, ...) high-rate samples into (Tlow, ...) 20 Hz samples."""
    if mode == "last":
        return block[:, -1]
    if mode == "mean":
        return block.mean(axis=1)
    if mode == "max":
        return block.max(axis=1)
    raise ValueError(f"unknown high200 pooling mode: {mode}")


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


def compute_normalizers(refs: list[DemoRef], joint_pool: str = "last") -> dict[str, np.ndarray]:
    joint_stats = RunningStats(14)
    action_stats = RunningStats(14)
    for ref in refs:
        with h5py.File(ref.path, "r") as f:
            demo = data_root(f)[ref.demo]
            joint = pool_high200(demo["high200/robot_joint_pos"][:], joint_pool).reshape(-1, 14)
            action = low20_action_dataset(demo)[:]
            joint_stats.update(joint)
            action_stats.update(action)
    joint_mean, joint_std = joint_stats.finish()
    action_mean, action_std = action_stats.finish()
    return {
        "joint_mean": joint_mean,
        "joint_std": joint_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


class ThreePiece20HzDataset(Dataset):
    def __init__(
        self,
        refs: list[DemoRef],
        history: int,
        normalizers: dict[str, np.ndarray],
        latent_cache: str | None = None,
        tactile_pool: str = "last",
        joint_pool: str = "last",
    ) -> None:
        self.refs = refs
        self.history = history
        self.window = history + 1
        self.normalizers = normalizers
        self.latent_cache = latent_cache
        self.tactile_pool = tactile_pool
        self.joint_pool = joint_pool
        self.starts: list[tuple[int, int]] = []
        for ref_i, ref in enumerate(refs):
            for start in range(ref.length20 - self.window + 1):
                self.starts.append((ref_i, start))
        self._files: dict[str, h5py.File] = {}
        self._cache: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.starts)

    def _file(self, path: str) -> h5py.File:
        f = self._files.get(path)
        if f is None:
            f = h5py.File(path, "r")
            self._files[path] = f
        return f

    def _cache_file(self) -> h5py.File:
        if self.latent_cache is None:
            raise RuntimeError("latent cache is not enabled")
        if self._cache is None:
            self._cache = h5py.File(self.latent_cache, "r")
        return self._cache

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        state["_cache"] = None
        return state

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref_i, start = self.starts[idx]
        ref = self.refs[ref_i]
        out: dict[str, torch.Tensor]
        if self.latent_cache is not None:
            group = self._cache_file()[cache_key(ref)]
            tactile_mu = group["tactile_mu"][start : start + self.window].astype(np.float32)
            joint = group["joint"][start : start + self.window].astype(np.float32)
            action = group["action"][start : start + self.window].astype(np.float32)
            out = {"tactile_mu": torch.from_numpy(tactile_mu)}
        else:
            demo = data_root(self._file(ref.path))[ref.demo]
            tactile_block = demo["high200/tactile"][start : start + self.window].astype(np.float32)
            joint_block = demo["high200/robot_joint_pos"][start : start + self.window].astype(np.float32)
            tactile = pool_high200(tactile_block, self.tactile_pool).astype(np.float32)
            joint = pool_high200(joint_block, self.joint_pool).astype(np.float32)
            action = low20_action_dataset(demo)[start : start + self.window].astype(np.float32)
            tactile = np.clip(tactile, 0.0, 1.0)
            out = {"tactile": torch.from_numpy(tactile)}

        joint = (joint - self.normalizers["joint_mean"]) / self.normalizers["joint_std"]
        action = (action - self.normalizers["action_mean"]) / self.normalizers["action_std"]
        out["joint"] = torch.from_numpy(joint)
        out["action"] = torch.from_numpy(action)
        return out


class ResBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 8))
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PatchEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
            ResBlock2d(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResBlock2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            ResBlock2d(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(self.backbone(x))


class TactileVAEProjector(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        latent_dim: int = 16,
        channels: int = 4,
        per_channel_dim: int = 48,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.encoder = PatchEncoder(latent_dim)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["encoder"] if "encoder" in ckpt else ckpt["model"]
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"missing VAE encoder weights: {missing}")
        bad_unexpected = [k for k in unexpected if not k.startswith("decoder.")]
        if bad_unexpected:
            raise RuntimeError(f"unexpected VAE encoder weights: {bad_unexpected}")
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.channel_head = nn.Sequential(
            nn.Linear(latent_dim, per_channel_dim),
            nn.LayerNorm(per_channel_dim),
            nn.SiLU(inplace=True),
        )
        self.projector = nn.Sequential(
            nn.Linear(channels * per_channel_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward_latents(self, tactile_mu: torch.Tensor) -> torch.Tensor:
        # tactile_mu: (B, T, 4, latent_dim), produced by the frozen VAE encoder.
        b, t, c, d = tactile_mu.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} tactile channels, got {c}")
        feat = self.channel_head(tactile_mu.reshape(b * t * c, d).float())
        feat = feat.reshape(b * t, c * self.channel_head[0].out_features)
        return self.projector(feat).reshape(b, t, -1)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        # tactile: (B, T, 4, 32, 32)
        b, t, c, h, w = tactile.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} tactile channels, got {c}")
        patches = tactile.reshape(b * t * c, 1, h, w).float()
        with torch.no_grad():
            mu = self.encoder(patches)
        return self.forward_latents(mu.reshape(b, t, c, -1))


class MLPToken(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int = 192, n_tokens: int = 2, heads: int = 4, mlp_dim: int = 768) -> None:
        super().__init__()
        self.mod_embed = nn.Embedding(n_tokens, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, *tokens: torch.Tensor) -> torch.Tensor:
        x = torch.stack(tokens, dim=2)  # (B, T, M, D)
        b, t, m, d = x.shape
        ids = torch.arange(m, device=x.device)
        x = x + self.mod_embed(ids).view(1, 1, m, d)
        y = x.reshape(b * t, m, d)
        attn, _ = self.attn(self.norm1(y), self.norm1(y), self.norm1(y), need_weights=False)
        y = y + attn
        y = y + self.ff(self.norm2(y))
        return y.reshape(b, t, m, d).mean(dim=2)


class ActionEmbedder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_dim: int = 768, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada(c).chunk(6, dim=-1)
        xa = self.norm1(x) * (1 + scale_a) + shift_a
        attn, _ = self.attn(xa, xa, xa, need_weights=False)
        x = x + gate_a * attn
        xf = self.norm2(x) * (1 + scale_f) + shift_f
        x = x + gate_f * self.ff(xf)
        return x


class HistoryPredictor(nn.Module):
    def __init__(self, history: int, embed_dim: int, depth: int = 4) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, history, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([ConditionalBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = z + self.pos[:, : z.shape[1]]
        for block in self.blocks:
            x = block(x, a)
        x = self.norm(x)
        b, t, d = x.shape
        return self.proj(x.reshape(b * t, d)).reshape(b, t, d)


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_proj = num_proj
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (T, B, D)
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


class ThreePieceWM(nn.Module):
    def __init__(self, vae_ckpt: str, history: int = 4, embed_dim: int = 192) -> None:
        super().__init__()
        self.tactile = TactileVAEProjector(vae_ckpt, embed_dim=embed_dim)
        self.joint = MLPToken(14, embed_dim)
        self.fusion = CrossModalFusion(embed_dim=embed_dim, n_tokens=2)
        self.action = ActionEmbedder(14, embed_dim)
        self.predictor = HistoryPredictor(history=history, embed_dim=embed_dim)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "tactile_mu" in batch:
            z_tac = self.tactile.forward_latents(batch["tactile_mu"])
        else:
            z_tac = self.tactile(batch["tactile"])
        z_joint = self.joint(batch["joint"])
        return self.fusion(z_tac, z_joint)

    def forward(self, batch: dict[str, torch.Tensor], history: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(batch)
        a = self.action(batch["action"])
        pred = self.predictor(z[:, :history], a[:, 1 : history + 1])
        target = z[:, 1 : history + 1]
        return pred, target


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_latent_cache(
    refs: list[DemoRef],
    ckpt_path: str,
    cache_path: str,
    device: torch.device,
    batch_size: int = 4096,
    tactile_pool: str = "last",
    joint_pool: str = "last",
) -> None:
    path = Path(cache_path)
    meta_path = path.with_suffix(path.suffix + ".json")
    wanted = {
        "vae_ckpt": str(Path(ckpt_path).expanduser()),
        "refs": [asdict(r) for r in refs],
        "latent_dim": 16,
        "channels": 4,
        "rate_hz": 20,
        "tactile_pool": tactile_pool,
        "joint_pool": joint_pool,
    }
    if path.exists() and meta_path.exists():
        try:
            got = json.loads(meta_path.read_text())
            if got == wanted:
                return
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    encoder = TactileVAEProjector(ckpt_path).encoder.to(device).eval()
    with h5py.File(tmp, "w") as out:
        for ref in refs:
            key = cache_key(ref)
            with h5py.File(ref.path, "r") as f:
                demo = data_root(f)[ref.demo]
                tactile_block = demo["high200/tactile"][:].astype(np.float32)
                joint_block = demo["high200/robot_joint_pos"][:].astype(np.float32)
                tactile = pool_high200(tactile_block, tactile_pool).astype(np.float32)
                joint = pool_high200(joint_block, joint_pool).reshape(-1, 14).astype(np.float32)
                action = low20_action_dataset(demo)[:].astype(np.float32)
                n = joint.shape[0]
                mu_all = np.empty((n, 4, 16), dtype=np.float32)
                flat = np.clip(tactile, 0.0, 1.0).reshape(-1, 1, 32, 32)
                for start in range(0, flat.shape[0], batch_size):
                    x = torch.from_numpy(flat[start : start + batch_size]).to(device, non_blocking=True)
                    with torch.no_grad():
                        mu = encoder(x).detach().cpu().numpy()
                    mu_all.reshape(-1, 16)[start : start + mu.shape[0]] = mu
                g = out.create_group(key)
                g.create_dataset("tactile_mu", data=mu_all, compression="gzip", compression_opts=1)
                g.create_dataset("joint", data=joint, compression="gzip", compression_opts=1)
                g.create_dataset("action", data=action, compression="gzip", compression_opts=1)
                g.attrs["source_path"] = ref.path
                g.attrs["demo"] = ref.demo
    tmp.rename(path)
    meta_path.write_text(json.dumps(wanted, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", type=int, default=4)
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
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--build-latent-cache", action="store_true")
    parser.add_argument("--cache-batch-size", type=int, default=4096)
    parser.add_argument("--tactile-pool", choices=["last", "max", "mean"], default="last")
    parser.add_argument("--joint-pool", choices=["last", "mean"], default="last")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = [str(Path(p).expanduser()) for p in args.data]
    train_refs, val_refs = split_demo_refs(paths, args.val_ratio)
    all_refs = train_refs + val_refs
    if args.build_latent_cache:
        if not args.latent_cache:
            raise ValueError("--build-latent-cache requires --latent-cache")
        build_latent_cache(
            all_refs,
            args.vae_ckpt,
            args.latent_cache,
            device,
            args.cache_batch_size,
            args.tactile_pool,
            args.joint_pool,
        )
    norms = compute_normalizers(train_refs, args.joint_pool)
    np.savez(out / "normalizers.npz", **norms)
    train_ds = ThreePiece20HzDataset(
        train_refs,
        args.history,
        norms,
        args.latent_cache,
        args.tactile_pool,
        args.joint_pool,
    )
    val_ds = (
        ThreePiece20HzDataset(
            val_refs,
            args.history,
            norms,
            args.latent_cache,
            args.tactile_pool,
            args.joint_pool,
        )
        if val_refs
        else None
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

    model = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    meta = vars(args).copy()
    meta.update(
        {
            "device": str(device),
            "train_demos": len(train_refs),
            "val_demos": len(val_refs),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds) if val_ds is not None else 0,
            "rate_hz": 20,
            "obs": f"tactile_vae_{Path(args.vae_ckpt).name} + robot_joint_pos only",
            "fusion": "attention over tactile token and joint_pos token",
            "action_semantics": "raw 14D low20 env delta action; no repeat-to-high200",
            "action_to_transition_alignment": "transition-aligned: action[t+1] conditions prediction z[t] -> z[t+1]",
            "tactile_pool": args.tactile_pool,
            "joint_pool": args.joint_pool,
            "latent_cache": args.latent_cache,
            "amp": args.amp,
            "amp_dtype": args.amp_dtype,
        }
    )
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    metrics_path = out / "metrics.jsonl"

    print(json.dumps(meta, indent=2), flush=True)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda" and amp_dtype is torch.float16)
    it = iter(train_loader)
    start = time.time()
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
            sreg_loss = sigreg(target.transpose(0, 1))
            loss = pred_loss + args.sigreg_weight * sreg_loss
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()

        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "time_s": time.time() - start,
                "lr": sched.get_last_lr()[0],
                "loss": float(loss.detach()),
                "pred_loss": float(pred_loss.detach()),
                "sigreg_loss": float(sreg_loss.detach()),
                "grad_norm": float(grad_norm),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        should_eval = step % args.eval_every == 0 or step == args.steps
        if should_eval and val_loader is not None:
            model.eval()
            sums = {"pred_loss": 0.0, "sigreg_loss": 0.0, "loss": 0.0}
            n = 0
            with torch.no_grad():
                for bi, batch in enumerate(val_loader):
                    if bi >= args.eval_batches:
                        break
                    batch = batch_to_device(batch, device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                        pred, target = model(batch, args.history)
                        pred_loss = F.mse_loss(pred, target)
                        sreg_loss = sigreg(target.transpose(0, 1))
                        loss = pred_loss + args.sigreg_weight * sreg_loss
                    bs = next(iter(batch.values())).shape[0]
                    sums["pred_loss"] += float(pred_loss) * bs
                    sums["sigreg_loss"] += float(sreg_loss) * bs
                    sums["loss"] += float(loss) * bs
                    n += bs
            val = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
            row = {"step": step, **val}
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[eval step {step}] {json.dumps(val)}", flush=True)
            if val["val_loss"] < best_val:
                best_val = val["val_loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "config": meta,
                        "normalizers": {k: v.tolist() for k, v in norms.items()},
                        "val": val,
                    },
                    out / "wm_best.pt",
                )

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "model": model.state_dict(),
                "step": step,
                "config": meta,
                "normalizers": {k: v.tolist() for k, v in norms.items()},
            }
            torch.save(ckpt, out / f"wm_step{step:06d}.pt")
            torch.save(ckpt, out / "wm_latest.pt")


if __name__ == "__main__":
    main()
