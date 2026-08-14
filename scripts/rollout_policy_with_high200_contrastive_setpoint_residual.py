#!/usr/bin/env python3
"""Roll out ThreePiece residual control against a 200 Hz tactile/joint WM.

The base policy still runs at the 20 Hz robosuite control rate. For residual
optimization, each candidate low-rate action chunk is expanded into ten 200 Hz
controller goals per low step using the same linear interpolation contract as
the generated high200 datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio  # noqa: F401  # keeps parity with older rollout env imports
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

import dexmimicgen  # noqa: E402,F401
import tactile_recollect.env as tactile_env  # noqa: E402
from mip.dataset_utils import RotationTransformer  # noqa: E402
from rollout_policy_with_q_video import (  # noqa: E402
    TrainingAgent,
    action20_to_action14,
    make_config,
    make_dataset,
    make_env,
    normalize_obs,
    obs_from_raw,
    stack_last,
)
from rollout_policy_with_contrastive_setpoint_residual import (  # noqa: E402
    load_normalizers,
    load_setpoint_params,
    normalize_action,
    pad_future,
    pad_slice,
    save_value_plot,
    setpoint_value_from_z,
)
from train_threepiece_high200_wm import ThreePieceWM  # noqa: E402


def robot_joint_pos(env) -> np.ndarray:
    pos = []
    for robot in getattr(env, "robots", []):
        if hasattr(robot, "_joint_positions"):
            pos.append(np.asarray(robot._joint_positions, dtype=np.float32).reshape(-1))
    if not pos:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(pos).astype(np.float32)


def high_sample(env, action: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "action": np.asarray(action, dtype=np.float32).copy(),
        "robot_joint_pos": robot_joint_pos(env),
        "tactile": tactile_env.read_tactile_image(env).astype(np.float32),
    }


def interp_low_to_high_torch(low_actions: torch.Tensor) -> torch.Tensor:
    low_actions = low_actions.reshape(-1, 14)
    if low_actions.shape[0] == 0:
        low_actions = torch.zeros((1, 14), dtype=torch.float32, device=low_actions.device)
    next_actions = torch.cat([low_actions[1:], low_actions[-1:]], dim=0)
    w = torch.linspace(0.0, 1.0, 10, dtype=low_actions.dtype, device=low_actions.device).view(1, 10, 1)
    high = low_actions[:, None, :] * (1.0 - w) + next_actions[:, None, :] * w
    return high.reshape(-1, 14)


def source_indices_for_rate(rate_hz: int) -> np.ndarray:
    if rate_hz % 20 != 0:
        raise ValueError(f"wm_rate_hz must be a multiple of 20, got {rate_hz}")
    samples_per_low = rate_hz // 20
    if samples_per_low < 1 or samples_per_low > 10 or 10 % samples_per_low != 0:
        raise ValueError(f"wm_rate_hz={rate_hz} is not aligned with 20 Hz control and 200 Hz source")
    return (np.floor((np.arange(samples_per_low, dtype=np.float64) + 1.0) * 10.0 / samples_per_low) - 1.0).astype(
        np.int64
    )


def interp_low_to_rate_torch(low_actions: torch.Tensor, rate_hz: int) -> torch.Tensor:
    high = interp_low_to_high_torch(low_actions).reshape(-1, 10, 14)
    idx = torch.as_tensor(source_indices_for_rate(rate_hz), dtype=torch.long, device=high.device)
    return high[:, idx].reshape(-1, 14)


def interp_low_to_rate_batch_torch(low_actions: torch.Tensor, rate_hz: int) -> torch.Tensor:
    low_actions = low_actions.reshape(low_actions.shape[0], -1, 14)
    if low_actions.shape[1] == 0:
        low_actions = torch.zeros((low_actions.shape[0], 1, 14), dtype=torch.float32, device=low_actions.device)
    next_actions = torch.cat([low_actions[:, 1:], low_actions[:, -1:]], dim=1)
    w = torch.linspace(0.0, 1.0, 10, dtype=low_actions.dtype, device=low_actions.device).view(1, 1, 10, 1)
    high = low_actions[:, :, None, :] * (1.0 - w) + next_actions[:, :, None, :] * w
    idx = torch.as_tensor(source_indices_for_rate(rate_hz), dtype=torch.long, device=high.device)
    return high[:, :, idx, :].reshape(low_actions.shape[0], -1, 14)


def interp_low_to_high_np(low_actions: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.asarray(low_actions, dtype=np.float32))
    return interp_low_to_high_torch(x).detach().cpu().numpy().astype(np.float32)


def step_with_high200_interp_minimal(
    env,
    action: np.ndarray,
    next_action: np.ndarray | None,
    low_step_index: int,
):
    if env.done:
        raise ValueError("executing action in terminated episode")

    env.timestep += 1
    start_time = float(env.cur_time)
    model_dt = float(env.model_timestep)
    control_dt = float(env.control_timestep)
    n_substeps = int(round(control_dt / model_dt))
    sample_dt = 0.005
    next_sample = 0
    samples = []

    high_actions = interp_low_to_high_np(
        np.stack(
            [
                np.asarray(action, dtype=np.float32),
                np.asarray(next_action if next_action is not None else action, dtype=np.float32),
            ],
            axis=0,
        )
    )[:10]
    last_bin = -1
    last_action = high_actions[0]

    for i in range(n_substeps):
        elapsed = i * model_dt
        high_bin = min(int((elapsed + 1e-12) / sample_dt), 9)
        last_action = high_actions[high_bin]
        if env.lite_physics:
            env.sim.step1()
        else:
            env.sim.forward()
        env._pre_action(last_action, policy_step=(high_bin != last_bin))
        if env.lite_physics:
            env.sim.step2()
        else:
            env.sim.step()
        env._update_observables()
        last_bin = high_bin

        actual_time = start_time + (i + 1) * model_dt
        while next_sample < 10:
            target_time = start_time + (next_sample + 1) * sample_dt
            if actual_time + 1e-12 < target_time:
                break
            samples.append(high_sample(env, high_actions[next_sample]))
            next_sample += 1

    while next_sample < 10:
        samples.append(high_sample(env, high_actions[next_sample]))
        next_sample += 1

    env.cur_time += env.control_timestep
    reward, done, info = env._post_action(last_action)
    observations = env.viewer._get_observations() if env.viewer_get_obs else env._get_observations()
    return observations, reward, done, info, samples


def setpoint_value_high200(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_low_raw: torch.Tensor,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    rate_hz: int,
    progress: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = tactile_hist_np.shape[0]
    tactile = torch.from_numpy(np.clip(tactile_hist_np, 0.0, 1.0)[None]).to(device=device, dtype=torch.float32)
    joint_np = (joint_hist_np.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
    joint = torch.from_numpy(joint_np[None]).to(device=device, dtype=torch.float32)
    action_hist = torch.from_numpy(action_hist_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    future_high_raw = interp_low_to_rate_torch(future_low_raw, rate_hz)
    action_full = torch.cat([action_hist, future_high_raw], dim=0)[None]
    action_norm = normalize_action(action_full, normalizers, device)

    z_window = wm.encode({"tactile": tactile, "joint": joint})
    for k in range(future_high_raw.shape[0]):
        a_win = wm.action(action_norm[:, k + 1 : k + 1 + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    value, beta = setpoint_value_from_z(z_window[:, -1], params, progress)
    return value.squeeze(0), beta.squeeze(0)


def setpoint_value_high200_batch(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_low_raw: torch.Tensor,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    rate_hz: int,
    progress: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    future_low_raw = future_low_raw.reshape(future_low_raw.shape[0], -1, 14)
    batch = int(future_low_raw.shape[0])
    tactile_one = torch.from_numpy(np.clip(tactile_hist_np, 0.0, 1.0)).to(device=device, dtype=torch.float32)
    tactile = tactile_one[None].expand(batch, -1, -1, -1, -1).contiguous()
    joint_np = (joint_hist_np.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
    joint_one = torch.from_numpy(joint_np).to(device=device, dtype=torch.float32)
    joint = joint_one[None].expand(batch, -1, -1).contiguous()
    action_hist_one = torch.from_numpy(action_hist_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    action_hist = action_hist_one[None].expand(batch, -1, -1).contiguous()

    future_high_raw = interp_low_to_rate_batch_torch(future_low_raw, rate_hz)
    action_full = torch.cat([action_hist, future_high_raw], dim=1)
    action_norm = normalize_action(action_full, normalizers, device)

    history = int(tactile_hist_np.shape[0])
    z_window = wm.encode({"tactile": tactile, "joint": joint})
    for k in range(future_high_raw.shape[1]):
        a_win = wm.action(action_norm[:, k + 1 : k + 1 + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    return setpoint_value_from_z(z_window[:, -1], params, progress)


def optimize_residual_high200(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_hist: deque,
    joint_hist: deque,
    action_hist: deque,
    future_low_actions_np: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk: int,
    trust_delta: float,
    qp_lambda_u: float,
    device: torch.device,
    rate_hz: int,
    progress: float,
) -> dict[str, np.ndarray | float]:
    tactile_hist_np = np.stack(list(tactile_hist)[-history:], axis=0).astype(np.float32)
    joint_hist_np = np.stack(list(joint_hist)[-history:], axis=0).astype(np.float32)
    action_hist_np = np.stack(list(action_hist)[-history:], axis=0).astype(np.float32)

    base_low = torch.from_numpy(pad_future(future_low_actions_np, chunk)).to(device=device, dtype=torch.float32)
    residual_low = torch.zeros_like(base_low, requires_grad=True)
    with torch.enable_grad():
        base_value, base_beta = setpoint_value_high200(
            wm,
            params,
            tactile_hist_np,
            joint_hist_np,
            action_hist_np,
            base_low + residual_low,
            normalizers,
            device,
            rate_hz,
            progress,
        )
        grad = torch.autograd.grad(base_value, residual_low, retain_graph=False, create_graph=False)[0]
    delta_low = torch.clamp(grad / max(float(qp_lambda_u), 1e-12), -float(trust_delta), float(trust_delta)).detach()
    with torch.no_grad():
        residual_value, residual_beta = setpoint_value_high200(
            wm,
            params,
            tactile_hist_np,
            joint_hist_np,
            action_hist_np,
            base_low + delta_low,
            normalizers,
            device,
            rate_hz,
            progress,
        )
    delta_np = delta_low.detach().cpu().numpy().astype(np.float32)
    grad_np = grad.detach().cpu().numpy().astype(np.float32)
    return {
        "delta_low": delta_np,
        "delta0": delta_np[0],
        "grad0": grad_np[0],
        "value_base": float(base_value.detach().cpu()),
        "value_residual": float(residual_value.detach().cpu()),
        "beta_base": float(base_beta.detach().cpu()),
        "beta_residual": float(residual_beta.detach().cpu()),
        "delta_abs_max": float(np.max(np.abs(delta_np[0]))),
        "delta_chunk_abs_max": float(np.max(np.abs(delta_np))),
        "delta_l2": float(np.linalg.norm(delta_np[0])),
        "grad_abs_max": float(np.max(np.abs(grad_np[0]))),
        "clip_frac": float(np.mean(np.abs(delta_np) >= float(trust_delta) - 1e-9)),
    }


def optimize_residual_high200_mppi(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_hist: deque,
    joint_hist: deque,
    action_hist: deque,
    future_low_actions_np: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk: int,
    trust_delta: float,
    device: torch.device,
    rate_hz: int,
    progress: float,
    samples: int,
    sigma: float,
    temperature: float,
    action_l2: float,
    smooth_l2: float,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    tactile_hist_np = np.stack(list(tactile_hist)[-history:], axis=0).astype(np.float32)
    joint_hist_np = np.stack(list(joint_hist)[-history:], axis=0).astype(np.float32)
    action_hist_np = np.stack(list(action_hist)[-history:], axis=0).astype(np.float32)

    samples = max(int(samples), 2)
    batch_size = max(int(batch_size), 1)
    base_low_np = pad_future(future_low_actions_np, chunk)
    base_low = torch.from_numpy(base_low_np).to(device=device, dtype=torch.float32)
    eps = torch.randn((samples, chunk, 14), device=device, dtype=torch.float32) * float(sigma)
    eps = torch.clamp(eps, -float(trust_delta), float(trust_delta))
    eps[0].zero_()
    candidates = base_low[None] + eps

    values = []
    betas = []
    with torch.no_grad():
        for start in range(0, samples, batch_size):
            cand = candidates[start : start + batch_size]
            value_b, beta_b = setpoint_value_high200_batch(
                wm,
                params,
                tactile_hist_np,
                joint_hist_np,
                action_hist_np,
                cand,
                normalizers,
                device,
                rate_hz,
                progress,
            )
            values.append(value_b.detach())
            betas.append(beta_b.detach())
    value = torch.cat(values, dim=0)
    beta = torch.cat(betas, dim=0)
    score = value.float()
    if float(action_l2) != 0.0:
        score = score - float(action_l2) * torch.sum(torch.square(eps), dim=(1, 2))
    if float(smooth_l2) != 0.0 and chunk > 1:
        score = score - float(smooth_l2) * torch.sum(torch.square(eps[:, 1:] - eps[:, :-1]), dim=(1, 2))
    weights = torch.softmax((score - torch.max(score)) / max(float(temperature), 1e-6), dim=0)
    delta_low = torch.sum(weights[:, None, None] * eps, dim=0)
    delta_low = torch.clamp(delta_low, -float(trust_delta), float(trust_delta)).detach()

    delta_np = delta_low.detach().cpu().numpy().astype(np.float32)
    score_np = score.detach().cpu().numpy().astype(np.float32)
    value_np = value.detach().cpu().numpy().astype(np.float32)
    beta_np = beta.detach().cpu().numpy().astype(np.float32)
    weight_np = weights.detach().cpu().numpy().astype(np.float32)
    return {
        "delta_low": delta_np,
        "delta0": delta_np[0],
        "grad0": np.zeros((14,), dtype=np.float32),
        "value_base": float(value_np[0]),
        "value_residual": float(torch.sum(weights * value).detach().cpu()),
        "beta_base": float(beta_np[0]),
        "beta_residual": float(torch.sum(weights * beta).detach().cpu()),
        "delta_abs_max": float(np.max(np.abs(delta_np[0]))),
        "delta_chunk_abs_max": float(np.max(np.abs(delta_np))),
        "delta_l2": float(np.linalg.norm(delta_np[0])),
        "grad_abs_max": 0.0,
        "clip_frac": float(np.mean(np.abs(delta_np) >= float(trust_delta) - 1e-9)),
        "mppi_score_max": float(np.max(score_np)),
        "mppi_score_mean": float(np.mean(score_np)),
        "mppi_value_max": float(np.max(value_np)),
        "mppi_value_mean": float(np.mean(value_np)),
        "mppi_best_index": int(np.argmax(score_np)),
        "mppi_weight_entropy": float(-np.sum(weight_np * np.log(np.clip(weight_np, 1e-12, 1.0)))),
        "mppi_weight_max": float(np.max(weight_np)),
    }


def optimize_residual_high200_annealed_mppi(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_hist: deque,
    joint_hist: deque,
    action_hist: deque,
    future_low_actions_np: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk: int,
    trust_delta: float,
    device: torch.device,
    rate_hz: int,
    progress: float,
    samples: int,
    sigma: float,
    temperature: float,
    action_l2: float,
    smooth_l2: float,
    batch_size: int,
    iterations: int,
    sigma_final_ratio: float,
    temperature_final_ratio: float,
    init_delta_np: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    tactile_hist_np = np.stack(list(tactile_hist)[-history:], axis=0).astype(np.float32)
    joint_hist_np = np.stack(list(joint_hist)[-history:], axis=0).astype(np.float32)
    action_hist_np = np.stack(list(action_hist)[-history:], axis=0).astype(np.float32)

    samples = max(int(samples), 2)
    batch_size = max(int(batch_size), 1)
    iterations = max(int(iterations), 1)
    sigma_final_ratio = max(float(sigma_final_ratio), 1e-6)
    temperature_final_ratio = max(float(temperature_final_ratio), 1e-6)

    base_low_np = pad_future(future_low_actions_np, chunk)
    base_low = torch.from_numpy(base_low_np).to(device=device, dtype=torch.float32)
    if init_delta_np is None:
        delta_mean = torch.zeros((chunk, 14), device=device, dtype=torch.float32)
    else:
        init_np = pad_future(np.asarray(init_delta_np, dtype=np.float32), chunk)
        delta_mean = torch.from_numpy(init_np).to(device=device, dtype=torch.float32)
        delta_mean = torch.clamp(delta_mean, -float(trust_delta), float(trust_delta))

    last_score = None
    last_value = None
    last_beta = None
    last_weights = None
    last_sigma = float(sigma)
    last_temperature = float(temperature)
    for it in range(iterations):
        frac = float(it) / max(float(iterations - 1), 1.0)
        sigma_i = float(sigma) * (sigma_final_ratio**frac)
        temp_i = float(temperature) * (temperature_final_ratio**frac)
        last_sigma = sigma_i
        last_temperature = temp_i

        eps = torch.randn((samples, chunk, 14), device=device, dtype=torch.float32) * sigma_i
        eps[0].zero_()
        cand_delta = torch.clamp(delta_mean[None] + eps, -float(trust_delta), float(trust_delta))
        cand_delta[0] = delta_mean
        candidates = base_low[None] + cand_delta

        values = []
        betas = []
        with torch.no_grad():
            for start in range(0, samples, batch_size):
                cand = candidates[start : start + batch_size]
                value_b, beta_b = setpoint_value_high200_batch(
                    wm,
                    params,
                    tactile_hist_np,
                    joint_hist_np,
                    action_hist_np,
                    cand,
                    normalizers,
                    device,
                    rate_hz,
                    progress,
                )
                values.append(value_b.detach())
                betas.append(beta_b.detach())
        value = torch.cat(values, dim=0)
        beta = torch.cat(betas, dim=0)
        score = value.float()
        if float(action_l2) != 0.0:
            score = score - float(action_l2) * torch.sum(torch.square(cand_delta), dim=(1, 2))
        if float(smooth_l2) != 0.0 and chunk > 1:
            score = score - float(smooth_l2) * torch.sum(torch.square(cand_delta[:, 1:] - cand_delta[:, :-1]), dim=(1, 2))

        weights = torch.softmax((score - torch.max(score)) / max(float(temp_i), 1e-6), dim=0)
        delta_mean = torch.sum(weights[:, None, None] * cand_delta, dim=0)
        delta_mean = torch.clamp(delta_mean, -float(trust_delta), float(trust_delta)).detach()
        last_score = score.detach()
        last_value = value.detach()
        last_beta = beta.detach()
        last_weights = weights.detach()

    with torch.no_grad():
        residual_value, residual_beta = setpoint_value_high200_batch(
            wm,
            params,
            tactile_hist_np,
            joint_hist_np,
            action_hist_np,
            (base_low + delta_mean)[None],
            normalizers,
            device,
            rate_hz,
            progress,
        )

    delta_np = delta_mean.detach().cpu().numpy().astype(np.float32)
    score_np = last_score.detach().cpu().numpy().astype(np.float32) if last_score is not None else np.zeros((0,), dtype=np.float32)
    value_np = last_value.detach().cpu().numpy().astype(np.float32) if last_value is not None else np.zeros((0,), dtype=np.float32)
    beta_np = last_beta.detach().cpu().numpy().astype(np.float32) if last_beta is not None else np.zeros((0,), dtype=np.float32)
    weight_np = (
        last_weights.detach().cpu().numpy().astype(np.float32) if last_weights is not None else np.zeros((0,), dtype=np.float32)
    )
    return {
        "delta_low": delta_np,
        "delta0": delta_np[0],
        "grad0": np.zeros((14,), dtype=np.float32),
        "value_base": float(value_np[0]) if value_np.size else float("nan"),
        "value_residual": float(residual_value.detach().cpu()[0]),
        "beta_base": float(beta_np[0]) if beta_np.size else float("nan"),
        "beta_residual": float(residual_beta.detach().cpu()[0]),
        "delta_abs_max": float(np.max(np.abs(delta_np[0]))),
        "delta_chunk_abs_max": float(np.max(np.abs(delta_np))),
        "delta_l2": float(np.linalg.norm(delta_np[0])),
        "grad_abs_max": 0.0,
        "clip_frac": float(np.mean(np.abs(delta_np) >= float(trust_delta) - 1e-9)),
        "mppi_score_max": float(np.max(score_np)) if score_np.size else float("nan"),
        "mppi_score_mean": float(np.mean(score_np)) if score_np.size else float("nan"),
        "mppi_value_max": float(np.max(value_np)) if value_np.size else float("nan"),
        "mppi_value_mean": float(np.mean(value_np)) if value_np.size else float("nan"),
        "mppi_best_index": int(np.argmax(score_np)) if score_np.size else -1,
        "mppi_weight_entropy": float(-np.sum(weight_np * np.log(np.clip(weight_np, 1e-12, 1.0)))) if weight_np.size else float("nan"),
        "mppi_weight_max": float(np.max(weight_np)) if weight_np.size else float("nan"),
        "mppi_iterations": float(iterations),
        "mppi_final_sigma": float(last_sigma),
        "mppi_final_temperature": float(last_temperature),
    }


@torch.no_grad()
def score_values_high200(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    high_tactile: np.ndarray,
    high_joint: np.ndarray,
    high_action: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk_low: int,
    rate_hz: int,
    device: torch.device,
    max_episode_steps: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    n_low = int(high_action.shape[0])
    if n_low == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    sample_indices = source_indices_for_rate(rate_hz)
    samples_per_low = int(sample_indices.size)
    tactile_flat = np.clip(high_tactile[:, sample_indices].reshape(-1, 4, 32, 32).astype(np.float32), 0.0, 1.0)
    joint_flat = high_joint[:, sample_indices].reshape(-1, high_joint.shape[-1]).astype(np.float32)
    action_flat = high_action[:, sample_indices].reshape(-1, high_action.shape[-1]).astype(np.float32)
    high_chunk = int(chunk_low * samples_per_low)

    values: list[float] = []
    betas: list[float] = []
    for low_start in range(0, n_low, batch_size):
        lows = list(range(low_start, min(low_start + batch_size, n_low)))
        tactile_batch = []
        joint_batch = []
        action_batch = []
        progress_batch = []
        for low_idx in lows:
            high_idx = low_idx * samples_per_low + samples_per_low - 1
            start = high_idx - history + 1
            tactile_batch.append(pad_slice(tactile_flat, start, history))
            joint_batch.append(pad_slice(joint_flat, start, history))
            action_batch.append(pad_slice(action_flat, start, history + high_chunk))
            progress_batch.append(min(1.0, max(0.0, float(low_idx + chunk_low) / max(float(max_episode_steps), 1.0))))
        tactile = np.stack(tactile_batch, axis=0)
        joint = np.stack(joint_batch, axis=0)
        action = np.stack(action_batch, axis=0)
        joint = (joint - normalizers["joint_mean"]) / normalizers["joint_std"]
        action = (action - normalizers["action_mean"]) / normalizers["action_std"]

        z_window = wm.encode(
            {
                "tactile": torch.from_numpy(tactile).to(device, non_blocking=True),
                "joint": torch.from_numpy(joint).to(device, non_blocking=True),
            }
        )
        action_t = torch.from_numpy(action).to(device, non_blocking=True)
        for k in range(high_chunk):
            a_win = wm.action(action_t[:, k + 1 : k + 1 + history])
            pred_seq = wm.predictor(z_window, a_win)
            next_z = pred_seq[:, -1]
            z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
        progress_t = torch.as_tensor(progress_batch, dtype=torch.float32, device=device)
        value, beta = setpoint_value_from_z(z_window[:, -1], params, progress_t)
        values.extend(value.detach().cpu().numpy().astype(np.float32).tolist())
        betas.extend(beta.detach().cpu().numpy().astype(np.float32).tolist())
    return np.asarray(values, dtype=np.float32), np.asarray(betas, dtype=np.float32)


def write_episode_csv(path: Path, episodes: list[dict]) -> None:
    fields = [
        "episode",
        "success",
        "steps",
        "reward_sum",
        "value_mean",
        "value_last",
        "beta_mean",
        "beta_last",
        "value_base_mean",
        "value_residual_mean",
        "delta_abs_max_mean",
        "delta_abs_max_max",
        "clip_frac_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ep in episodes:
            writer.writerow({k: int(ep[k]) if k == "success" else ep.get(k) for k in fields})


def collect_one_residual_rollout(env, config, dataset, agent, wm, params, normalizers, nfe: int, episode_idx: int, args):
    device = torch.device(config.optimization.device if torch.cuda.is_available() else "cpu")
    rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    sample_indices = source_indices_for_rate(args.wm_rate_hz)
    np.random.seed(args.seed + episode_idx)
    torch.manual_seed(args.seed + episode_idx)

    raw_obs = env.reset()
    obs_hist = deque(maxlen=config.task.obs_steps)
    first_obs = obs_from_raw(raw_obs, config, dataset)
    for _ in range(config.task.obs_steps):
        obs_hist.append(first_obs)

    init_high = high_sample(env, np.zeros(14, dtype=np.float32))
    tactile_hist = deque([init_high["tactile"]] * args.history, maxlen=args.history)
    joint_hist = deque([init_high["robot_joint_pos"]] * args.history, maxlen=args.history)
    action_hist = deque([init_high["action"]] * args.history, maxlen=args.history)

    low_actions, base_actions = [], []
    high_tactile, high_joint, high_action = [], [], []
    rewards, successes, residual_meta = [], [], []
    total_reward = 0.0
    success = False
    low_steps = 0
    anneal_delta_plan: np.ndarray | None = None

    while low_steps < config.task.max_episode_steps:
        obs_seq = stack_last(obs_hist, config.task.obs_steps)
        obs = normalize_obs(obs_seq, dataset, config.optimization.device)
        act_0 = torch.randn((1, config.task.horizon, config.task.act_dim), device=device, dtype=torch.float32)
        with torch.no_grad():
            act_normed = agent.sample(act_0=act_0, obs=obs, num_steps=nfe, use_ema=True)
        act20 = dataset.normalizer["action"].unnormalize(act_normed.detach().cpu().numpy())[0]
        start = config.task.obs_steps - 1
        end = start + config.task.act_steps
        act14_seq = action20_to_action14(act20[start:end], rotation_transformer)

        for j, base_act14 in enumerate(act14_seq):
            base_act14 = base_act14.astype(np.float32)
            future_low = act14_seq[j : j + args.chunk].astype(np.float32)
            value_progress = min(1.0, max(0.0, float(low_steps + args.chunk) / max(float(config.task.max_episode_steps), 1.0)))
            if args.optimizer == "mppi":
                opt = optimize_residual_high200_mppi(
                    wm,
                    params,
                    tactile_hist,
                    joint_hist,
                    action_hist,
                    future_low,
                    normalizers,
                    args.history,
                    args.chunk,
                    args.trust_delta,
                    device,
                    args.wm_rate_hz,
                    value_progress,
                    args.mppi_samples,
                    args.mppi_sigma,
                    args.mppi_temperature,
                    args.mppi_action_l2,
                    args.mppi_smooth_l2,
                    args.mppi_batch_size,
                )
            elif args.optimizer == "anneal_mppi":
                opt = optimize_residual_high200_annealed_mppi(
                    wm,
                    params,
                    tactile_hist,
                    joint_hist,
                    action_hist,
                    future_low,
                    normalizers,
                    args.history,
                    args.chunk,
                    args.trust_delta,
                    device,
                    args.wm_rate_hz,
                    value_progress,
                    args.mppi_samples,
                    args.mppi_sigma,
                    args.mppi_temperature,
                    args.mppi_action_l2,
                    args.mppi_smooth_l2,
                    args.mppi_batch_size,
                    args.mppi_iterations,
                    args.mppi_sigma_final_ratio,
                    args.mppi_temperature_final_ratio,
                    anneal_delta_plan if args.mppi_warm_start else None,
                )
            else:
                opt = optimize_residual_high200(
                    wm,
                    params,
                    tactile_hist,
                    joint_hist,
                    action_hist,
                    future_low,
                    normalizers,
                    args.history,
                    args.chunk,
                    args.trust_delta,
                    args.qp_lambda_u,
                    device,
                    args.wm_rate_hz,
                    value_progress,
                )
            future_base = pad_future(future_low, args.chunk)
            future_exec = (future_base + opt["delta_low"]).astype(np.float32)
            residual_act14 = future_exec[0]
            next_residual_act14 = future_exec[1] if future_exec.shape[0] > 1 else residual_act14
            if args.optimizer == "anneal_mppi" and args.mppi_warm_start:
                delta_low = np.asarray(opt["delta_low"], dtype=np.float32)
                if delta_low.shape[0] > 1:
                    anneal_delta_plan = np.concatenate([delta_low[1:], delta_low[-1:] * 0.0], axis=0)
                else:
                    anneal_delta_plan = delta_low * 0.0
            raw_obs, reward, done, _, samples = step_with_high200_interp_minimal(
                env,
                residual_act14,
                next_residual_act14,
                low_steps,
            )
            total_reward += float(reward)
            success = success or bool(env._check_success()) or float(reward) > 0.0

            low_actions.append(residual_act14)
            base_actions.append(base_act14)
            rewards.append(np.float32(reward))
            successes.append(np.float32(success))
            high_tactile.append(np.stack([s["tactile"] for s in samples], axis=0))
            high_joint.append(np.stack([s["robot_joint_pos"] for s in samples], axis=0))
            high_action.append(np.stack([s["action"] for s in samples], axis=0))
            for sample_i in sample_indices:
                s = samples[int(sample_i)]
                tactile_hist.append(s["tactile"])
                joint_hist.append(s["robot_joint_pos"])
                action_hist.append(s["action"])
            residual_meta.append({k: v for k, v in opt.items() if k not in ("delta_low", "delta0", "grad0")})

            if args.progress_every > 0 and low_steps % args.progress_every == 0:
                print(
                    f"[progress] ep={episode_idx} step={low_steps} success={int(success)} "
                    f"v_base={opt['value_base']:.3f} v_res={opt['value_residual']:.3f} "
                    f"delta_max={opt['delta_abs_max']:.4f}",
                    flush=True,
                )

            obs_hist.append(obs_from_raw(raw_obs, config, dataset))
            low_steps += 1
            if done or success or low_steps >= config.task.max_episode_steps:
                break
        if success or low_steps >= config.task.max_episode_steps:
            break

    return {
        "episode": episode_idx,
        "success": bool(success),
        "reward_sum": float(total_reward),
        "steps": int(low_steps),
        "actions": np.asarray(low_actions, dtype=np.float32),
        "base_actions": np.asarray(base_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "high_tactile": np.asarray(high_tactile, dtype=np.float32),
        "high_joint": np.asarray(high_joint, dtype=np.float32),
        "high_action": np.asarray(high_action, dtype=np.float32),
        "residual": residual_meta,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default=str(
            DEXMG_ROOT
            / "datasets/generated_tactile_actionrollout_virtual_s8fs25_shards/20260726_122743_s8fs25/two_arm_three_piece_assembly"
        ),
    )
    parser.add_argument("--env-dataset-path", default=str(DEXMG_ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    parser.add_argument("--task-config", default="dexmg_three_piece_image_tactile_cnn_virtual_s8fs25")
    parser.add_argument(
        "--model-path",
        default=str(
            POLICY_ROOT
            / "logs_dexmg_three_piece_tactile_cnn_virtual_s8fs25_flow_100k_20260726/models/model_step70000.pt"
        ),
    )
    parser.add_argument("--params", required=True)
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument(
        "--vae-ckpt",
        default=str(POLICY_ROOT / "runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727/vae_step008000.pt"),
    )
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--wm-rate-hz", type=int, default=200)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--optimizer", choices=["grad", "mppi", "anneal_mppi"], default="grad")
    parser.add_argument("--trust-delta", type=float, default=0.01)
    parser.add_argument("--qp-lambda-u", type=float, default=20.0)
    parser.add_argument("--mppi-samples", type=int, default=64)
    parser.add_argument("--mppi-sigma", type=float, default=0.01)
    parser.add_argument("--mppi-temperature", type=float, default=0.3)
    parser.add_argument("--mppi-action-l2", type=float, default=0.0)
    parser.add_argument("--mppi-smooth-l2", type=float, default=0.0)
    parser.add_argument("--mppi-batch-size", type=int, default=128)
    parser.add_argument("--mppi-iterations", type=int, default=1)
    parser.add_argument("--mppi-sigma-final-ratio", type=float, default=0.5)
    parser.add_argument("--mppi-temperature-final-ratio", type=float, default=1.0)
    parser.add_argument("--mppi-warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=8800)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class ConfigArgs:
        pass

    cfg_args = ConfigArgs()
    cfg_args.task_config = args.task_config
    cfg_args.model_path = args.model_path
    cfg_args.dataset_path = args.dataset_path
    cfg_args.max_episode_steps = args.max_episode_steps
    cfg_args.episodes = args.episodes
    config = make_config(cfg_args)
    if args.device is not None:
        config.optimization.device = args.device
    config.task.max_episode_steps = args.max_episode_steps
    device = torch.device(config.optimization.device if torch.cuda.is_available() else "cpu")

    sample_indices = source_indices_for_rate(args.wm_rate_hz)
    print(f"[setup] high200 policy_model={args.model_path}", flush=True)
    print(f"[setup] wm_rate_hz={args.wm_rate_hz} source_indices={sample_indices.tolist()}", flush=True)
    print(f"[setup] high200 wm_ckpt={args.wm_ckpt}", flush=True)
    print(f"[setup] params={args.params}", flush=True)
    print(f"[setup] optimizer={args.optimizer} trust_delta={args.trust_delta} qp_lambda_u={args.qp_lambda_u}", flush=True)
    if args.optimizer in ("mppi", "anneal_mppi"):
        print(
            f"[setup] mppi samples={args.mppi_samples} sigma={args.mppi_sigma} temp={args.mppi_temperature} "
            f"action_l2={args.mppi_action_l2} smooth_l2={args.mppi_smooth_l2} batch={args.mppi_batch_size}",
            flush=True,
        )
        if args.optimizer == "anneal_mppi":
            print(
                f"[setup] anneal_mppi iterations={args.mppi_iterations} sigma_final_ratio={args.mppi_sigma_final_ratio} "
                f"temperature_final_ratio={args.mppi_temperature_final_ratio} warm_start={args.mppi_warm_start}",
                flush=True,
            )

    env = make_env(args.env_dataset_path, enable_tactile=True)
    if abs(float(env.control_timestep) - 0.05) > 1e-9:
        raise RuntimeError(f"Expected 20 Hz control_dt=0.05, got {env.control_timestep}")
    if abs(float(env.model_timestep) - 0.002) > 1e-9:
        raise RuntimeError(f"Expected 500 Hz model_dt=0.002, got {env.model_timestep}")
    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(args.model_path, load_optimizer=False)
    agent.eval()

    wm = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    wm_ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    wm.load_state_dict(wm_ckpt["model"], strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    normalizers = load_normalizers(args.normalizers)
    params = load_setpoint_params(args.params, device)

    episodes = []
    for ep in range(args.episodes):
        print(f"[rollout] high200 {args.optimizer} residual episode={ep} nfe={args.nfe}", flush=True)
        data = collect_one_residual_rollout(env, config, dataset, agent, wm, params, normalizers, args.nfe, ep, args)
        values, betas = score_values_high200(
            wm,
            params,
            data["high_tactile"],
            data["high_joint"],
            data["high_action"],
            normalizers,
            args.history,
            args.chunk,
            args.wm_rate_hz,
            device,
            args.max_episode_steps,
        )
        deltas = [float(r["delta_abs_max"]) for r in data["residual"]]
        clips = [float(r["clip_frac"]) for r in data["residual"]]
        v_base = [float(r["value_base"]) for r in data["residual"]]
        v_res = [float(r["value_residual"]) for r in data["residual"]]
        ep_meta = {
            "episode": ep,
            "success": bool(data["success"]),
            "reward_sum": float(data["reward_sum"]),
            "steps": int(data["steps"]),
            "value_min": float(values.min()) if values.size else float("nan"),
            "value_max": float(values.max()) if values.size else float("nan"),
            "value_mean": float(values.mean()) if values.size else float("nan"),
            "value_first": float(values[0]) if values.size else float("nan"),
            "value_last": float(values[-1]) if values.size else float("nan"),
            "beta_mean": float(betas.mean()) if betas.size else float("nan"),
            "beta_last": float(betas[-1]) if betas.size else float("nan"),
            "value_base_mean": float(np.mean(v_base)) if v_base else float("nan"),
            "value_residual_mean": float(np.mean(v_res)) if v_res else float("nan"),
            "delta_abs_max_mean": float(np.mean(deltas)) if deltas else float("nan"),
            "delta_abs_max_max": float(np.max(deltas)) if deltas else float("nan"),
            "clip_frac_mean": float(np.mean(clips)) if clips else float("nan"),
            "values": values.astype(float).tolist(),
            "betas": betas.astype(float).tolist(),
            "residual": data["residual"],
        }
        episodes.append(ep_meta)
        compact = {k: v for k, v in ep_meta.items() if k not in ("values", "betas", "residual")}
        print(f"[rollout] {json.dumps(compact)}", flush=True)

    plot_path = out_dir / "value_curves.png"
    save_value_plot(episodes, plot_path)
    write_episode_csv(out_dir / "episodes.csv", episodes)
    success_count = int(sum(bool(ep["success"]) for ep in episodes))
    summary = {
        "plot": str(plot_path),
        "policy_model": args.model_path,
        "wm_ckpt": args.wm_ckpt,
        "params": args.params,
        "normalizers": args.normalizers,
        "nfe": args.nfe,
        "episodes_n": args.episodes,
        "success_count": success_count,
        "success_rate": success_count / max(args.episodes, 1),
        "optimizer": args.optimizer,
        "trust_delta": args.trust_delta,
        "qp_lambda_u": args.qp_lambda_u,
        "mppi_samples": args.mppi_samples,
        "mppi_sigma": args.mppi_sigma,
        "mppi_temperature": args.mppi_temperature,
        "mppi_action_l2": args.mppi_action_l2,
        "mppi_smooth_l2": args.mppi_smooth_l2,
        "mppi_batch_size": args.mppi_batch_size,
        "mppi_iterations": args.mppi_iterations,
        "mppi_sigma_final_ratio": args.mppi_sigma_final_ratio,
        "mppi_temperature_final_ratio": args.mppi_temperature_final_ratio,
        "mppi_warm_start": args.mppi_warm_start,
        "chunk_low20": args.chunk,
        "wm_rate_hz": args.wm_rate_hz,
        "source_high200_indices_per_low20": [int(x) for x in sample_indices.tolist()],
        "value_timebase": f"one value per 20Hz policy step; WM internally rolls chunk_low20 * {len(sample_indices)} target-rate steps",
        "residual_contract": "base policy 20Hz; residual optimized in raw 14D low20 action space; low20 chunk is expanded by linear interpolation then sampled at wm_rate_hz; environment execution still uses the full 200Hz interpolation",
        "episodes": episodes,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] success={success_count}/{args.episodes}", flush=True)
    print(f"[done] plot={plot_path}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
