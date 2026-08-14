#!/usr/bin/env python3
"""Probe MPPI residual directions on many states sampled from demo caches.

This is an offline diagnostic: sample (demo, time) states from a cached tactile
latent dataset, use the demo action chunk as the base action sequence, then
compare full-chunk gradient/QP residuals with MPPI weighted and best-sample
residual chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

DEFAULT_DEXMG_ROOT = Path(__file__).resolve().parents[1]
DEXMG_ROOT = Path(os.environ.get("DEXMG_ROOT", str(DEFAULT_DEXMG_ROOT))).resolve()
POLICY_ROOT = Path(
    os.environ.get("POLICY_ROOT", "/home/labeng/workspaces/cwang17_ws/wm/much-ado-about-noising")
).resolve()
ROBOSUITE_ROOT = DEXMG_ROOT / "robosuite"

for path in (POLICY_ROOT, ROBOSUITE_ROOT, DEXMG_ROOT, Path(__file__).resolve().parent):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TACTILE_RENDERER", "virtual")
os.environ.setdefault("TACTILE_VIRTUAL_SIGMA", "8.0")
os.environ.setdefault("TACTILE_VIRTUAL_FORCE_SCALE", "25.0")
os.environ.setdefault("TACTILE_VIRTUAL_MAX", "1.0")
os.environ.setdefault("TACTILE_VIRTUAL_CANONICAL", "1")
os.environ.setdefault("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from rollout_policy_with_contrastive_setpoint_residual import (  # noqa: E402
    load_normalizers,
    load_setpoint_params,
    normalize_action,
    pad_future,
    setpoint_value_from_z,
)
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402


def parse_setting(text: str) -> dict[str, float | int | str]:
    parts = text.split(",")
    if len(parts) != 6:
        raise ValueError(f"setting must be name,chunk,samples,trust,sigma,temp, got {text!r}")
    return {
        "name": parts[0],
        "chunk": int(parts[1]),
        "samples": int(parts[2]),
        "trust": float(parts[3]),
        "sigma": float(parts[4]),
        "temp": float(parts[5]),
    }


def default_settings() -> list[dict[str, float | int | str]]:
    raw = [
        "c10_s64_tr010_sig010_t003,10,64,0.01,0.01,0.03",
        "c10_s256_tr010_sig010_t003,10,256,0.01,0.01,0.03",
        "c10_s1024_tr010_sig010_t003,10,1024,0.01,0.01,0.03",
        "c10_s2048_tr010_sig010_t003,10,2048,0.01,0.01,0.03",
        "c10_s256_tr030_sig030_t003,10,256,0.03,0.03,0.03",
        "c20_s64_tr010_sig010_t003,20,64,0.01,0.01,0.03",
        "c20_s256_tr010_sig010_t003,20,256,0.01,0.01,0.03",
        "c20_s1024_tr010_sig010_t003,20,1024,0.01,0.01,0.03",
        "c20_s2048_tr010_sig010_t003,20,2048,0.01,0.01,0.03",
        "c20_s256_tr030_sig030_t003,20,256,0.03,0.03,0.03",
    ]
    return [parse_setting(x) for x in raw]


def finite_mean(vals: list[float]) -> float:
    xs = [float(x) for x in vals if np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def finite_median(vals: list[float]) -> float:
    xs = [float(x) for x in vals if np.isfinite(float(x))]
    return float(np.median(xs)) if xs else float("nan")


def finite_frac(vals: list[float], threshold: float) -> float:
    xs = [float(x) for x in vals if np.isfinite(float(x))]
    return float(np.mean(np.asarray(xs) > float(threshold))) if xs else float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def per_step_cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(a.shape[0], -1)
    b = np.asarray(b, dtype=np.float32).reshape(b.shape[0], -1)
    vals = [cosine(a[i], b[i]) for i in range(min(a.shape[0], b.shape[0]))]
    return finite_mean(vals)


def source_class(key: str, attrs: h5py.AttributeManager) -> str:
    source_path = str(attrs.get("source_path", ""))
    demo = str(attrs.get("demo", ""))
    text = f"{key} {source_path} {demo}".lower()
    if "success" in text:
        return "success"
    if "failure" in text:
        return "failure"
    return "official"


def list_cache_demos(cache_path: str, history: int, max_chunk: int) -> list[dict[str, Any]]:
    demos = []
    with h5py.File(cache_path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            if not all(name in g for name in ("tactile_mu", "joint", "action")):
                continue
            length = int(g["action"].shape[0])
            if length < history + max_chunk + 2:
                continue
            demos.append(
                {
                    "key": key,
                    "length": length,
                    "source_class": source_class(key, g.attrs),
                    "source_path": str(g.attrs.get("source_path", "")),
                    "demo": str(g.attrs.get("demo", "")),
                }
            )
    if not demos:
        raise RuntimeError(f"no usable demos found in {cache_path}")
    return demos


def sample_states(
    demos: list[dict[str, Any]],
    states: int,
    history: int,
    max_chunk: int,
    time_bins: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    time_bins = max(int(time_bins), 1)
    for state_id in range(int(states)):
        demo = demos[int(rng.integers(0, len(demos)))]
        length = int(demo["length"])
        min_t = history - 1
        max_t = length - max_chunk - 2
        if max_t < min_t:
            continue
        bin_id = state_id % time_bins
        span = max_t - min_t + 1
        lo = min_t + int(math.floor(span * bin_id / time_bins))
        hi = min_t + int(math.floor(span * (bin_id + 1) / time_bins)) - 1
        hi = max(lo, min(hi, max_t))
        t = int(rng.integers(lo, hi + 1))
        out.append(
            {
                "state_id": state_id,
                "key": demo["key"],
                "hist_end": t,
                "length": length,
                "progress": min(1.0, max(0.0, float(t + max_chunk) / max(float(length), 1.0))),
                "time_bin": bin_id,
                "source_class": demo["source_class"],
                "source_demo": demo["demo"],
                "source_path": demo["source_path"],
            }
        )
    rng.shuffle(out)
    for i, state in enumerate(out):
        state["order"] = i
    return out


def read_state(
    cache_file: h5py.File,
    state: dict[str, Any],
    history: int,
    max_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = cache_file[str(state["key"])]
    t = int(state["hist_end"])
    start = t - history + 1
    if start < 0:
        raise ValueError(f"history underflow for state {state}")
    tactile_mu = g["tactile_mu"][start : t + 1].astype(np.float32)
    joint = g["joint"][start : t + 1].astype(np.float32)
    action_hist = g["action"][start : t + 1].astype(np.float32)
    future = g["action"][t + 1 : t + 1 + max_chunk].astype(np.float32)
    future = pad_future(future, max_chunk).astype(np.float32)
    return tactile_mu, joint, action_hist, future


def setpoint_value_mu_batch(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_mu_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_raw: torch.Tensor,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    progress: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    future_raw = future_raw.reshape(future_raw.shape[0], -1, 14)
    batch = int(future_raw.shape[0])
    history = int(tactile_mu_hist_np.shape[0])

    tactile_one = torch.from_numpy(tactile_mu_hist_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    tactile = tactile_one[None].expand(batch, -1, -1, -1).contiguous()
    joint_np = (joint_hist_np.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
    joint_one = torch.from_numpy(joint_np).to(device=device, dtype=torch.float32)
    joint = joint_one[None].expand(batch, -1, -1).contiguous()
    action_hist_one = torch.from_numpy(action_hist_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    action_hist = action_hist_one[None].expand(batch, -1, -1).contiguous()

    action_full = torch.cat([action_hist, future_raw], dim=1)
    action_norm = normalize_action(action_full, normalizers, device)

    z_window = wm.encode({"tactile_mu": tactile, "joint": joint})
    for k in range(future_raw.shape[1]):
        a_win = wm.action(action_norm[:, k + 1 : k + 1 + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    return setpoint_value_from_z(z_window[:, -1], params, progress)


def gradient_residual(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_mu_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_np: np.ndarray,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    progress: float,
    chunk: int,
    trust_delta: float,
    qp_lambda_u: float,
) -> dict[str, Any]:
    base = torch.from_numpy(pad_future(future_np, chunk)).to(device=device, dtype=torch.float32)
    residual = torch.zeros_like(base, requires_grad=True)
    with torch.enable_grad():
        base_value, base_beta = setpoint_value_mu_batch(
            wm,
            params,
            tactile_mu_hist_np,
            joint_hist_np,
            action_hist_np,
            base[None] + residual[None],
            normalizers,
            device,
            progress,
        )
        grad = torch.autograd.grad(base_value.sum(), residual, retain_graph=False, create_graph=False)[0]
    delta = torch.clamp(grad / max(float(qp_lambda_u), 1e-12), -float(trust_delta), float(trust_delta)).detach()
    with torch.no_grad():
        residual_value, residual_beta = setpoint_value_mu_batch(
            wm,
            params,
            tactile_mu_hist_np,
            joint_hist_np,
            action_hist_np,
            base[None] + delta[None],
            normalizers,
            device,
            progress,
        )
    delta_np = delta.detach().cpu().numpy().astype(np.float32)
    grad_np = grad.detach().cpu().numpy().astype(np.float32)
    return {
        "base": base.detach().cpu().numpy().astype(np.float32),
        "delta": delta_np,
        "grad": grad_np,
        "value_base": float(base_value.detach().cpu()[0]),
        "value_residual": float(residual_value.detach().cpu()[0]),
        "beta_base": float(base_beta.detach().cpu()[0]),
        "beta_residual": float(residual_beta.detach().cpu()[0]),
        "clip_frac": float(np.mean(np.abs(delta_np) >= float(trust_delta) - 1e-9)),
        "abs_max": float(np.max(np.abs(delta_np))),
        "norm": float(np.linalg.norm(delta_np)),
    }


@torch.no_grad()
def mppi_residual(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_mu_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    base_np: np.ndarray,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    progress: float,
    chunk: int,
    samples: int,
    trust_delta: float,
    sigma: float,
    temperature: float,
    action_l2: float,
    smooth_l2: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    base = torch.from_numpy(pad_future(base_np, chunk)).to(device=device, dtype=torch.float32)
    samples = max(int(samples), 2)
    batch_size = max(int(batch_size), 1)
    eps = torch.randn((samples, chunk, 14), device=device, dtype=torch.float32, generator=generator) * float(sigma)
    eps = torch.clamp(eps, -float(trust_delta), float(trust_delta))
    eps[0].zero_()
    candidates = base[None] + eps

    values = []
    betas = []
    for start in range(0, samples, batch_size):
        value_b, beta_b = setpoint_value_mu_batch(
            wm,
            params,
            tactile_mu_hist_np,
            joint_hist_np,
            action_hist_np,
            candidates[start : start + batch_size],
            normalizers,
            device,
            progress,
        )
        values.append(value_b.detach())
        betas.append(beta_b.detach())
    value = torch.cat(values, dim=0).float()
    beta = torch.cat(betas, dim=0).float()
    score = value.clone()
    if float(action_l2) != 0.0:
        score = score - float(action_l2) * torch.sum(torch.square(eps), dim=(1, 2))
    if float(smooth_l2) != 0.0 and chunk > 1:
        score = score - float(smooth_l2) * torch.sum(torch.square(eps[:, 1:] - eps[:, :-1]), dim=(1, 2))

    weights = torch.softmax((score - torch.max(score)) / max(float(temperature), 1e-6), dim=0)
    weighted_delta = torch.sum(weights[:, None, None] * eps, dim=0)
    weighted_delta = torch.clamp(weighted_delta, -float(trust_delta), float(trust_delta)).detach()
    best_idx = int(torch.argmax(score).detach().cpu())
    best_delta = eps[best_idx].detach()

    weighted_value, weighted_beta = setpoint_value_mu_batch(
        wm,
        params,
        tactile_mu_hist_np,
        joint_hist_np,
        action_hist_np,
        base[None] + weighted_delta[None],
        normalizers,
        device,
        progress,
    )
    return {
        "weighted_delta": weighted_delta.detach().cpu().numpy().astype(np.float32),
        "best_delta": best_delta.detach().cpu().numpy().astype(np.float32),
        "value_base": float(value[0].detach().cpu()),
        "value_weighted_expected": float(torch.sum(weights * value).detach().cpu()),
        "value_weighted_actual": float(weighted_value.detach().cpu()[0]),
        "value_best": float(value[best_idx].detach().cpu()),
        "beta_base": float(beta[0].detach().cpu()),
        "beta_weighted_actual": float(weighted_beta.detach().cpu()[0]),
        "beta_best": float(beta[best_idx].detach().cpu()),
        "score_max": float(torch.max(score).detach().cpu()),
        "score_mean": float(torch.mean(score).detach().cpu()),
        "score_std": float(torch.std(score, unbiased=False).detach().cpu()),
        "score_range": float((torch.max(score) - torch.min(score)).detach().cpu()),
        "weight_max": float(torch.max(weights).detach().cpu()),
        "weight_entropy": float((-torch.sum(weights * torch.log(torch.clamp(weights, min=1e-12)))).detach().cpu()),
        "best_index": best_idx,
    }


def row_metrics(
    state: dict[str, Any],
    setting: dict[str, float | int | str],
    grad: dict[str, Any],
    probe: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    grad_delta = np.asarray(grad["delta"], dtype=np.float32)
    weighted = np.asarray(probe["weighted_delta"], dtype=np.float32)
    best = np.asarray(probe["best_delta"], dtype=np.float32)
    grad_norm = float(np.linalg.norm(grad_delta))
    weighted_norm = float(np.linalg.norm(weighted))
    best_norm = float(np.linalg.norm(best))
    grad_step_norm = np.linalg.norm(grad_delta.reshape(grad_delta.shape[0], -1), axis=1)
    max_step = int(np.argmax(grad_step_norm)) if grad_step_norm.size else 0
    return {
        "state_id": int(state["state_id"]),
        "key": state["key"],
        "source_class": state["source_class"],
        "hist_end": int(state["hist_end"]),
        "length": int(state["length"]),
        "progress": float(state["progress"]),
        "time_bin": int(state["time_bin"]),
        "setting": setting["name"],
        "repeat": repeat,
        "chunk": int(setting["chunk"]),
        "samples": int(setting["samples"]),
        "trust": float(setting["trust"]),
        "sigma": float(setting["sigma"]),
        "temp": float(setting["temp"]),
        "grad_trust": float(np.max(np.abs(grad_delta))) if grad_delta.size else 0.0,
        "grad_clip_frac": float(grad["clip_frac"]),
        "grad_value_gain": float(grad["value_residual"] - grad["value_base"]),
        "weighted_value_gain": float(probe["value_weighted_actual"] - probe["value_base"]),
        "weighted_expected_value_gain": float(probe["value_weighted_expected"] - probe["value_base"]),
        "best_value_gain": float(probe["value_best"] - probe["value_base"]),
        "weighted_chunk_cos": cosine(grad_delta, weighted),
        "best_chunk_cos": cosine(grad_delta, best),
        "weighted_step_cos_mean": per_step_cosine_mean(grad_delta, weighted),
        "best_step_cos_mean": per_step_cosine_mean(grad_delta, best),
        "weighted_first_cos": cosine(grad_delta[0], weighted[0]),
        "best_first_cos": cosine(grad_delta[0], best[0]),
        "weighted_maxgradstep_cos": cosine(grad_delta[max_step], weighted[max_step]),
        "best_maxgradstep_cos": cosine(grad_delta[max_step], best[max_step]),
        "max_grad_step": max_step,
        "grad_norm": grad_norm,
        "weighted_norm": weighted_norm,
        "best_norm": best_norm,
        "weighted_norm_ratio": float(weighted_norm / max(grad_norm, 1e-12)),
        "best_norm_ratio": float(best_norm / max(grad_norm, 1e-12)),
        "weighted_l2_to_grad": float(np.linalg.norm(weighted - grad_delta)),
        "best_l2_to_grad": float(np.linalg.norm(best - grad_delta)),
        "mppi_weight_max": float(probe["weight_max"]),
        "mppi_weight_entropy": float(probe["weight_entropy"]),
        "mppi_score_max": float(probe["score_max"]),
        "mppi_score_mean": float(probe["score_mean"]),
        "mppi_score_std": float(probe["score_std"]),
        "mppi_score_range": float(probe["score_range"]),
        "mppi_best_index": int(probe["best_index"]),
    }


def load_wm(args: argparse.Namespace, device: torch.device) -> ThreePieceWM:
    wm = ThreePieceWM(args.vae_ckpt, history=args.history, embed_dim=args.embed_dim).to(device)
    ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    wm.load_state_dict(state, strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    return wm


def run_probe(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    settings = [parse_setting(x) for x in args.setting] if args.setting else default_settings()
    max_chunk = max(int(s["chunk"]) for s in settings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    demos = list_cache_demos(args.cache, args.history, max_chunk)
    states = sample_states(demos, args.states, args.history, max_chunk, args.time_bins, args.seed)
    class_counts = defaultdict(int)
    for s in states:
        class_counts[str(s["source_class"])] += 1

    print(f"[setup] out={out_dir}", flush=True)
    print(f"[setup] cache={args.cache}", flush=True)
    print(f"[setup] demos={len(demos)} sampled_states={len(states)} classes={dict(class_counts)}", flush=True)
    print(f"[setup] settings={len(settings)} max_chunk={max_chunk} device={device}", flush=True)

    wm = load_wm(args, device)
    normalizers = load_normalizers(args.normalizers)
    params = load_setpoint_params(args.params, device)

    rows: list[dict[str, Any]] = []
    with h5py.File(args.cache, "r") as cache_file:
        for idx, state in enumerate(states):
            tactile_mu, joint, action_hist, future = read_state(cache_file, state, args.history, max_chunk)
            grad_by_chunk: dict[int, dict[str, Any]] = {}
            for setting_i, setting in enumerate(settings):
                chunk = int(setting["chunk"])
                grad = grad_by_chunk.get(chunk)
                if grad is None:
                    grad = gradient_residual(
                        wm,
                        params,
                        tactile_mu,
                        joint,
                        action_hist,
                        future[:chunk],
                        normalizers,
                        device,
                        float(state["progress"]),
                        chunk,
                        args.grad_trust_delta,
                        args.qp_lambda_u,
                    )
                    grad_by_chunk[chunk] = grad
                for rep in range(args.repeats):
                    probe = mppi_residual(
                        wm,
                        params,
                        tactile_mu,
                        joint,
                        action_hist,
                        future[:chunk],
                        normalizers,
                        device,
                        float(state["progress"]),
                        chunk,
                        int(setting["samples"]),
                        float(setting["trust"]),
                        float(setting["sigma"]),
                        float(setting["temp"]),
                        args.mppi_action_l2,
                        args.mppi_smooth_l2,
                        args.mppi_batch_size,
                        args.seed + 1000003 * int(state["state_id"]) + 1009 * setting_i + rep,
                    )
                    rows.append(row_metrics(state, setting, grad, probe, rep))
            if (idx + 1) % max(args.progress_every, 1) == 0 or idx == len(states) - 1:
                print(f"[probe] states_done={idx + 1}/{len(states)} rows={len(rows)}", flush=True)

    meta = {
        "cache": str(Path(args.cache).resolve()),
        "wm_ckpt": str(Path(args.wm_ckpt).resolve()),
        "vae_ckpt": str(Path(args.vae_ckpt).resolve()),
        "normalizers": str(Path(args.normalizers).resolve()),
        "params": str(Path(args.params).resolve()),
        "states": len(states),
        "demos": len(demos),
        "class_counts": dict(class_counts),
        "settings": settings,
        "history": args.history,
        "max_chunk": max_chunk,
        "grad_trust_delta": args.grad_trust_delta,
        "qp_lambda_u": args.qp_lambda_u,
        "mppi_action_l2": args.mppi_action_l2,
        "mppi_smooth_l2": args.mppi_smooth_l2,
        "seed": args.seed,
        "device": str(device),
    }
    return rows, states, meta


def write_outputs(rows: list[dict[str, Any]], states: list[dict[str, Any]], meta: dict[str, Any], out_dir: Path) -> None:
    if not rows:
        raise RuntimeError("no probe rows collected")

    fields = list(rows[0].keys())
    with (out_dir / "probe_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "states.json").write_text(json.dumps(states, indent=2))
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[str(row["setting"])].append(row)

    summary = []
    metric_keys = [
        "weighted_chunk_cos",
        "best_chunk_cos",
        "weighted_step_cos_mean",
        "best_step_cos_mean",
        "weighted_first_cos",
        "best_first_cos",
        "weighted_maxgradstep_cos",
        "best_maxgradstep_cos",
        "weighted_norm_ratio",
        "best_norm_ratio",
        "grad_norm",
        "weighted_norm",
        "best_norm",
        "grad_value_gain",
        "weighted_value_gain",
        "weighted_expected_value_gain",
        "best_value_gain",
        "grad_clip_frac",
        "mppi_weight_max",
        "mppi_weight_entropy",
        "mppi_score_std",
        "mppi_score_range",
    ]
    for name, vals in sorted(by_setting.items()):
        item: dict[str, Any] = {
            "setting": name,
            "n": len(vals),
            "chunk": int(vals[0]["chunk"]),
            "samples": int(vals[0]["samples"]),
            "trust": float(vals[0]["trust"]),
            "sigma": float(vals[0]["sigma"]),
            "temp": float(vals[0]["temp"]),
        }
        for key in metric_keys:
            xs = [float(v[key]) for v in vals]
            item[f"{key}_mean"] = finite_mean(xs)
            item[f"{key}_median"] = finite_median(xs)
        for prefix in ("weighted_chunk_cos", "best_chunk_cos"):
            xs = [float(v[prefix]) for v in vals]
            item[f"{prefix}_frac_gt_0"] = finite_frac(xs, 0.0)
            item[f"{prefix}_frac_gt_03"] = finite_frac(xs, 0.3)
            item[f"{prefix}_frac_gt_05"] = finite_frac(xs, 0.5)
        summary.append(item)

    summary.sort(
        key=lambda x: (
            -(x["weighted_chunk_cos_mean"] if np.isfinite(x["weighted_chunk_cos_mean"]) else -999.0),
            -(x["best_chunk_cos_mean"] if np.isfinite(x["best_chunk_cos_mean"]) else -999.0),
            str(x["setting"]),
        )
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "| setting | n | chunk | samples | trust | sigma | temp | weighted chunk cos | best chunk cos | w cos>0.3 | w cos>0.5 | best cos>0.3 | best cos>0.5 | weighted norm/grad | best norm/grad | grad gain | weighted gain | best gain | w_max | entropy | score std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        md.append(
            f"| {s['setting']} | {s['n']} | {s['chunk']} | {s['samples']} | "
            f"{s['trust']:.3f} | {s['sigma']:.3f} | {s['temp']:.3f} | "
            f"{s['weighted_chunk_cos_mean']:.3f} | {s['best_chunk_cos_mean']:.3f} | "
            f"{s['weighted_chunk_cos_frac_gt_03']:.2f} | {s['weighted_chunk_cos_frac_gt_05']:.2f} | "
            f"{s['best_chunk_cos_frac_gt_03']:.2f} | {s['best_chunk_cos_frac_gt_05']:.2f} | "
            f"{s['weighted_norm_ratio_mean']:.2f} | {s['best_norm_ratio_mean']:.2f} | "
            f"{s['grad_value_gain_mean']:.4f} | {s['weighted_value_gain_mean']:.4f} | "
            f"{s['best_value_gain_mean']:.4f} | {s['mppi_weight_max_mean']:.3f} | "
            f"{s['mppi_weight_entropy_mean']:.3f} | {s['mppi_score_std_mean']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")


def parse_args() -> argparse.Namespace:
    base = DEXMG_ROOT / "outputs/threepiece_rate20_from_high200_old20_aligned_compare_20260808_1035"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default=str(base / "threepiece_official1006_plus_250s250f_rate20_last_vae8k_mu_cache_20260808_1035.hdf5"),
    )
    parser.add_argument("--wm-ckpt", default=str(base / "wm_100k_lr5e-5_sigreg0p09_oldvae8k/wm_best.pt"))
    parser.add_argument(
        "--vae-ckpt",
        default=str(
            POLICY_ROOT / "runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727/vae_step008000.pt"
        ),
    )
    parser.add_argument("--normalizers", default=str(base / "wm_100k_lr5e-5_sigreg0p09_oldvae8k/normalizers.npz"))
    parser.add_argument("--params", default=str(base / "fit_identity_late20_earlyneg/contrastive_params.npz"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mppi-batch-size", type=int, default=512)
    parser.add_argument("--mppi-action-l2", type=float, default=0.0)
    parser.add_argument("--mppi-smooth-l2", type=float, default=0.0)
    parser.add_argument("--grad-trust-delta", type=float, default=0.01)
    parser.add_argument("--qp-lambda-u", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=9810)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        help="Setting as name,chunk,samples,trust,sigma,temp. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows, states, meta = run_probe(args)
    write_outputs(rows, states, meta, out_dir)
    print(f"[done] rows={len(rows)} states={len(states)} out={out_dir}", flush=True)
    print(f"[done] summary={out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
