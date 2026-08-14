#!/usr/bin/env python3
"""Roll out a tactile policy with a 20 Hz contrastive-setpoint residual.

The residual objective is:

    V_alpha(WM_rollout(z_hist, A_base + deltaU)) - lambda_u ||deltaU||^2

where V_alpha is loaded from contrastive_params.npz produced by
train_threepiece_contrastive_setpoint_value.py. The action semantics are raw
14-D low20 env delta commands; the WM sees them normalized by the 20 Hz WM
normalizers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
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

from mip.dataset_utils import RotationTransformer  # noqa: E402
from rollout_policy_with_q_video import (  # noqa: E402
    TrainingAgent,
    action20_to_action14,
    high_sample,
    make_config,
    make_dataset,
    make_env,
    normalize_obs,
    obs_from_raw,
    render_cameras,
    stack_last,
    step_with_high200_minimal,
)
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402

CAMERA_NAMES = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]


def pad_future(actions: np.ndarray, chunk: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 14)
    if actions.shape[0] == 0:
        actions = np.zeros((1, 14), dtype=np.float32)
    if actions.shape[0] < chunk:
        pad = np.repeat(actions[-1:], chunk - actions.shape[0], axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    return actions[:chunk]


def pad_slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    n = arr.shape[0]
    idx = np.arange(start, start + length)
    idx = np.clip(idx, 0, max(n - 1, 0))
    return arr[idx]


def load_normalizers(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k].astype(np.float32) for k in npz.files}


def load_setpoint_params(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    npz = np.load(path)
    required = ["P", "P_mean", "v", "beta_star", "sigma"]
    param_type_code = int(float(npz["param_type_code"])) if "param_type_code" in npz.files else 0
    missing = [k for k in required if k not in npz.files]
    if missing and param_type_code == 0:
        raise KeyError(f"contrastive params missing keys: {missing}")
    out = {k: torch.as_tensor(npz[k], dtype=torch.float32, device=device) for k in npz.files}
    out["bias"] = torch.as_tensor(npz["bias"] if "bias" in npz.files else np.array(0.0, dtype=np.float32), dtype=torch.float32, device=device)
    out["value_mode_code"] = torch.as_tensor(
        npz["value_mode_code"] if "value_mode_code" in npz.files else np.array(0.0, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    out["param_type_code"] = torch.as_tensor(np.array(param_type_code, dtype=np.float32), dtype=torch.float32, device=device)
    return out


def normalize_action(action: torch.Tensor, normalizers: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(normalizers["action_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(normalizers["action_std"], device=device, dtype=torch.float32)
    return (action - mean) / std


def _progress_tensor(progress: float | torch.Tensor | None, z: torch.Tensor) -> torch.Tensor:
    if progress is None:
        return torch.zeros((z.shape[0],), dtype=torch.float32, device=z.device)
    if isinstance(progress, torch.Tensor):
        tau = progress.to(device=z.device, dtype=torch.float32).reshape(-1)
        if tau.numel() == 1 and z.shape[0] != 1:
            tau = tau.expand(z.shape[0])
        return tau.clamp(0.0, 1.0)
    return torch.full((z.shape[0],), float(progress), dtype=torch.float32, device=z.device).clamp(0.0, 1.0)


def _interp_table(table: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    n = int(table.shape[0])
    if n <= 1:
        return table[0].expand(tau.shape[0], *table.shape[1:])
    x = tau.clamp(0.0, 1.0) * (n - 1)
    lo = torch.floor(x).long().clamp(0, n - 1)
    hi = torch.clamp(lo + 1, max=n - 1)
    w = (x - lo.float()).reshape(-1, *([1] * (table.dim() - 1)))
    return table[lo] * (1.0 - w) + table[hi] * w


def _rbf_features(tau: torch.Tensor, centers: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    width = torch.clamp(width.float(), min=1e-4)
    phi = torch.exp(-0.5 * torch.square((tau[:, None] - centers[None].float()) / width))
    return phi / torch.clamp(phi.sum(dim=-1, keepdim=True), min=1e-6)


def setpoint_value_from_z(
    z: torch.Tensor,
    params: dict[str, torch.Tensor],
    progress: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.float()
    param_type = int(float(params.get("param_type_code", torch.tensor(0.0, device=z.device)).detach().cpu()))
    if param_type == 10:
        tau = _progress_tensor(progress, z)
        v_table = params["v_table"].float()
        k = int(v_table.shape[0])
        idx = torch.clamp(torch.floor(tau * k).long(), 0, k - 1)
        v = v_table[idx]
        beta = torch.sum(z * v, dim=-1)
        beta_star = params["beta_star_table"].float()[idx]
        sigma = params["sigma_table"].float()[idx]
        value = torch.exp(-torch.square((beta_star - beta) / torch.clamp(sigma, min=1e-6)))
        return value, beta
    if param_type == 11:
        tau = _progress_tensor(progress, z)
        phi = _rbf_features(tau, params["basis_centers"], params["basis_width"])
        w = torch.einsum("bm,md->bd", phi, params["W_table"].float())
        bias = phi @ params["b_table"].float()
        beta = torch.sum(z * w, dim=-1) + bias
        return torch.sigmoid(beta), beta
    if param_type == 14:
        tau = _progress_tensor(progress, z)
        phi = _rbf_features(tau, params["basis_centers"], params["basis_width"])
        z_mean = params["z_mean"].float()
        z_std = torch.clamp(params["z_std"].float(), min=1e-6)
        z_bar = (z - z_mean) / z_std
        w = torch.einsum("bm,md->bd", phi, params["W_table"].float())
        bias = phi @ params["b_table"].float()
        beta = torch.sum(z_bar * w, dim=-1) + bias
        return torch.sigmoid(beta), beta
    if param_type == 12:
        tau = _progress_tensor(progress, z)
        beta = z @ params["v"].float()
        beta_star = _interp_table(params["beta_star_table"].float(), tau).reshape(-1)
        sigma = _interp_table(params["sigma_table"].float(), tau).reshape(-1)
        value = torch.exp(-torch.square((beta - beta_star) / torch.clamp(sigma, min=1e-6)))
        return value, beta
    if param_type == 13:
        tau = _progress_tensor(progress, z)
        phi = _rbf_features(tau, params["basis_centers"], params["basis_width"])
        z_mean = params["mlp_z_mean"].float()
        z_std = torch.clamp(params["mlp_z_std"].float(), min=1e-6)
        x = torch.cat([(z - z_mean) / z_std, phi], dim=-1)
        h = torch.nn.functional.gelu(x @ params["mlp_w0"].float().T + params["mlp_b0"].float())
        if "mlp_w1" in params and "mlp_b1" in params:
            h = torch.nn.functional.gelu(h @ params["mlp_w1"].float().T + params["mlp_b1"].float())
        beta = (h @ params["mlp_wout"].float().reshape(-1, 1)).reshape(-1) + params["mlp_bout"].float().reshape(())
        if "mlp_logit_temperature" in params:
            beta = beta / torch.clamp(params["mlp_logit_temperature"].float().reshape(()), min=1e-6)
        return torch.sigmoid(beta), beta

    pz = (z - params["P_mean"]) @ params["P"].T
    beta = pz @ params["v"] + params["bias"]
    mode_code = int(float(params["value_mode_code"].detach().cpu()))
    if mode_code == 2:
        value = torch.sigmoid(beta)
        return value, beta
    alpha = params["beta_star"] - beta
    if mode_code == 1:
        alpha = torch.clamp(alpha, min=0.0)
    value = torch.exp(-torch.square(alpha / torch.clamp(params["sigma"], min=1e-6)))
    return value, beta


def setpoint_value_20hz(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_low_raw: torch.Tensor,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    progress: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = tactile_hist_np.shape[0]
    tactile = torch.from_numpy(np.clip(tactile_hist_np, 0.0, 1.0)[None]).to(device=device, dtype=torch.float32)
    joint_np = (joint_hist_np.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
    joint = torch.from_numpy(joint_np[None]).to(device=device, dtype=torch.float32)
    action_hist = torch.from_numpy(action_hist_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    action_full = torch.cat([action_hist, future_low_raw], dim=0)[None]
    action_norm = normalize_action(action_full, normalizers, device)

    z_hist = wm.encode({"tactile": tactile, "joint": joint})
    z_window = z_hist
    chunk = future_low_raw.shape[0]
    for k in range(chunk):
        a_win = wm.action(action_norm[:, k + 1 : k + 1 + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    value, beta = setpoint_value_from_z(z_window[:, -1], params, progress)
    return value.squeeze(0), beta.squeeze(0)


def optimize_residual_20hz(
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
    progress: float,
) -> dict[str, np.ndarray | float]:
    tactile_hist_np = np.stack(list(tactile_hist)[-history:], axis=0).astype(np.float32)
    joint_hist_np = np.stack(list(joint_hist)[-history:], axis=0).astype(np.float32)
    if len(action_hist) >= history:
        action_hist_np = np.stack(list(action_hist)[-history:], axis=0).astype(np.float32)
    else:
        action_hist_np = np.zeros((history, 14), dtype=np.float32)

    base_low = torch.from_numpy(pad_future(future_low_actions_np, chunk)).to(device=device, dtype=torch.float32)
    residual_low = torch.zeros_like(base_low, requires_grad=True)
    with torch.enable_grad():
        base_value, base_beta = setpoint_value_20hz(
            wm,
            params,
            tactile_hist_np,
            joint_hist_np,
            action_hist_np,
            base_low + residual_low,
            normalizers,
            device,
            progress,
        )
        grad = torch.autograd.grad(base_value, residual_low, retain_graph=False, create_graph=False)[0]
    delta_low = torch.clamp(grad / max(float(qp_lambda_u), 1e-12), -float(trust_delta), float(trust_delta)).detach()
    with torch.no_grad():
        residual_value, residual_beta = setpoint_value_20hz(
            wm,
            params,
            tactile_hist_np,
            joint_hist_np,
            action_hist_np,
            base_low + delta_low,
            normalizers,
            device,
            progress,
        )
    delta0 = delta_low[0].detach().cpu().numpy().astype(np.float32)
    grad0 = grad[0].detach().cpu().numpy().astype(np.float32)
    return {
        "delta0": delta0,
        "grad0": grad0,
        "value_base": float(base_value.detach().cpu()),
        "value_residual": float(residual_value.detach().cpu()),
        "beta_base": float(base_beta.detach().cpu()),
        "beta_residual": float(residual_beta.detach().cpu()),
        "delta_abs_max": float(np.max(np.abs(delta0))),
        "delta_l2": float(np.linalg.norm(delta0)),
        "grad_abs_max": float(np.max(np.abs(grad0))),
        "clip_frac": float(np.mean(np.abs(delta_low.detach().cpu().numpy()) >= float(trust_delta) - 1e-9)),
    }


@torch.no_grad()
def score_values_20hz(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    high_tactile: np.ndarray,
    high_joint: np.ndarray,
    low_actions: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk: int,
    device: torch.device,
    max_episode_steps: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    n_low = int(low_actions.shape[0])
    if n_low == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    tactile_low = np.clip(high_tactile[:, -1].astype(np.float32), 0.0, 1.0)
    joint_low = high_joint[:, -1].astype(np.float32)
    action_low = low_actions.astype(np.float32)
    if joint_low.shape[-1] != 14:
        raise RuntimeError(f"expected 14-D robot_joint_pos, got {joint_low.shape}")

    values: list[float] = []
    betas: list[float] = []
    window = history + chunk
    for low_start in range(0, n_low, batch_size):
        lows = list(range(low_start, min(low_start + batch_size, n_low)))
        tactile_batch = []
        joint_batch = []
        action_batch = []
        progress_batch = []
        for low_idx in lows:
            start = low_idx - history + 1
            tactile_batch.append(pad_slice(tactile_low, start, history))
            joint_batch.append(pad_slice(joint_low, start, history))
            action_batch.append(pad_slice(action_low, start, window))
            progress_batch.append(min(1.0, max(0.0, float(low_idx + chunk) / max(float(max_episode_steps), 1.0))))
        tactile = np.stack(tactile_batch, axis=0)
        joint = np.stack(joint_batch, axis=0)
        action = np.stack(action_batch, axis=0)
        joint = (joint - normalizers["joint_mean"]) / normalizers["joint_std"]
        action = (action - normalizers["action_mean"]) / normalizers["action_std"]

        sample = {
            "tactile": torch.from_numpy(tactile).to(device, non_blocking=True),
            "joint": torch.from_numpy(joint).to(device, non_blocking=True),
            "action": torch.from_numpy(action).to(device, non_blocking=True),
        }
        z_hist = wm.encode({"tactile": sample["tactile"], "joint": sample["joint"]})
        z_window = z_hist
        for k in range(chunk):
            a_win = wm.action(sample["action"][:, k + 1 : k + 1 + history])
            pred_seq = wm.predictor(z_window, a_win)
            next_z = pred_seq[:, -1]
            z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
        progress_t = torch.as_tensor(progress_batch, dtype=torch.float32, device=device)
        value, beta = setpoint_value_from_z(z_window[:, -1], params, progress_t)
        values.extend(value.detach().cpu().numpy().astype(np.float32).tolist())
        betas.extend(beta.detach().cpu().numpy().astype(np.float32).tolist())
    return np.asarray(values, dtype=np.float32), np.asarray(betas, dtype=np.float32)


def collect_one_residual_rollout(env, config, dataset, agent, wm, params, normalizers, nfe: int, episode_idx: int, args):
    device = torch.device(config.optimization.device if torch.cuda.is_available() else "cpu")
    rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    np.random.seed(args.seed + episode_idx)
    torch.manual_seed(args.seed + episode_idx)

    raw_obs = env.reset()
    obs_hist = deque(maxlen=config.task.obs_steps)
    first_obs = obs_from_raw(raw_obs, config, dataset)
    for _ in range(config.task.obs_steps):
        obs_hist.append(first_obs)

    init_high = high_sample(env)
    tactile_hist = deque([init_high["tactile"]] * args.history, maxlen=args.history)
    joint_hist = deque([init_high["robot_joint_pos"]] * args.history, maxlen=args.history)
    action_hist = deque([np.zeros(14, dtype=np.float32)] * args.history, maxlen=args.history)

    rgb_frames, low_actions, base_actions = [], [], []
    rewards, successes = [], []
    high_tactile, high_joint = [], []
    residual_meta = []
    total_reward = 0.0
    success = False
    low_steps = 0

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
            opt = optimize_residual_20hz(
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
                value_progress,
            )
            residual_act14 = (base_act14 + opt["delta0"]).astype(np.float32)
            raw_obs, reward, done, _, samples = step_with_high200_minimal(env, residual_act14, low_steps)
            total_reward += float(reward)
            success = success or bool(env._check_success()) or float(reward) > 0.0

            if args.save_videos:
                rgb_frames.append(render_cameras(env, CAMERA_NAMES, args.video_height, args.video_width))
            low_actions.append(residual_act14)
            base_actions.append(base_act14)
            rewards.append(np.float32(reward))
            successes.append(np.float32(success))
            high_tactile.append(np.stack([s["tactile"] for s in samples], axis=0))
            high_joint.append(np.stack([s["robot_joint_pos"] for s in samples], axis=0))
            last = samples[-1]
            tactile_hist.append(last["tactile"])
            joint_hist.append(last["robot_joint_pos"])
            action_hist.append(residual_act14)
            residual_meta.append({k: v for k, v in opt.items() if k not in ("delta0", "grad0")})

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
        "rgb": np.asarray(rgb_frames, dtype=np.uint8) if args.save_videos else np.zeros((0,), dtype=np.uint8),
        "actions": np.asarray(low_actions, dtype=np.float32),
        "base_actions": np.asarray(base_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "high_tactile": np.asarray(high_tactile, dtype=np.float32),
        "high_joint": np.asarray(high_joint, dtype=np.float32),
        "residual": residual_meta,
    }


def save_value_plot(episodes: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=150)
    for ep in episodes:
        v = np.asarray(ep["values"], dtype=np.float32)
        x = np.arange(v.shape[0], dtype=np.float32) / 20.0
        label = (
            f"ep{ep['episode']} success={int(ep['success'])} steps={ep['steps']} "
            f"v_last={ep['value_last']:.3f} d={ep['delta_abs_max_mean']:.4f}"
        )
        ax.plot(x, v, linewidth=1.6, label=label)
    ax.set_title("20 Hz contrastive setpoint residual value")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("V_alpha")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def write_episode_video(path: Path, frames: np.ndarray, label: str, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(label_frame(frame, label))


def make_tactile_panel(tactile4: np.ndarray, width: int, height: int) -> np.ndarray:
    tactile4 = np.asarray(tactile4, dtype=np.float32)
    if tactile4.shape != (4, 32, 32):
        raise RuntimeError(f"expected tactile frame (4, 32, 32), got {tactile4.shape}")
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    gap = 8
    label_h = 20
    tile_w = max(1, (width - gap * 5) // 4)
    tile_h = max(1, height - label_h - 2 * gap)
    for idx in range(4):
        x0 = gap + idx * (tile_w + gap)
        y0 = label_h + gap
        img = np.clip(tactile4[idx], 0.0, 1.0)
        gray = (img * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = cv2.resize(color, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        panel[y0 : y0 + tile_h, x0 : x0 + tile_w] = color
        text = f"t{idx} max={float(img.max()):.2f}"
        cv2.putText(panel, text, (x0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, "tactile raw virtual value [0,1]", (8, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def make_value_panel(values: np.ndarray, betas: np.ndarray, idx: int, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 52, 14, 18, 34
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    cv2.rectangle(panel, (x0, y0), (x1, y1), (35, 35, 35), 1)
    for y_val in (0.25, 0.5, 0.75):
        y = int(y1 - y_val * (y1 - y0))
        cv2.line(panel, (x0, y), (x1, y), (215, 215, 215), 1)
        cv2.putText(panel, f"{y_val:.2f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    values = np.asarray(values, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)
    if values.size:
        end = min(idx + 1, values.size)
        xs = np.linspace(x0, x1, values.size)
        ys = y1 - np.clip(values, 0.0, 1.0) * (y1 - y0)
        pts = np.stack([xs[:end], ys[:end]], axis=1).astype(np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(panel, [pts], False, (20, 95, 210), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts[-1]), 4, (20, 95, 210), -1, cv2.LINE_AA)
        v_now = float(values[min(idx, values.size - 1)])
        beta_now = float(betas[min(idx, betas.size - 1)]) if betas.size else float("nan")
    else:
        v_now = float("nan")
        beta_now = float("nan")

    cv2.putText(panel, "V(z) realtime curve", (x0, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(panel, f"step={idx:03d}  V={v_now:.3f}  beta={beta_now:.3f}", (x0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(panel, "0", (x0 - 8, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{max(values.size - 1, 0)}", (x1 - 24, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
    return panel


def make_composite_frame(
    rgb: np.ndarray,
    tactile4: np.ndarray,
    values: np.ndarray,
    betas: np.ndarray,
    idx: int,
    label: str,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    width = int(rgb.shape[1])
    tactile_panel = make_tactile_panel(tactile4, width, max(112, rgb.shape[0] // 2))
    value_panel = make_value_panel(values, betas, idx, width, max(128, rgb.shape[0] // 2))
    top = label_frame(rgb, label)
    return np.concatenate([top, tactile_panel, value_panel], axis=0)


def write_composite_episode_video(
    path: Path,
    rgb_frames: np.ndarray,
    high_tactile: np.ndarray,
    values: np.ndarray,
    betas: np.ndarray,
    label: str,
    fps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(rgb_frames), len(high_tactile), len(values))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for idx in range(n):
            frame = make_composite_frame(rgb_frames[idx], high_tactile[idx, -1], values, betas, idx, label)
            writer.append_data(frame)


def write_episode_csv(path: Path, episodes: list[dict]) -> None:
    rows = []
    for ep in episodes:
        rows.append(
            {
                "episode": ep["episode"],
                "success": int(ep["success"]),
                "steps": ep["steps"],
                "reward_sum": ep["reward_sum"],
                "value_mean": ep["value_mean"],
                "value_last": ep["value_last"],
                "beta_mean": ep["beta_mean"],
                "beta_last": ep["beta_last"],
                "value_base_mean": ep["value_base_mean"],
                "value_residual_mean": ep["value_residual_mean"],
                "delta_abs_max_mean": ep["delta_abs_max_mean"],
                "delta_abs_max_max": ep["delta_abs_max_max"],
                "clip_frac_mean": ep["clip_frac_mean"],
            }
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument(
        "--params",
        required=True,
        help="contrastive_params.npz from train_threepiece_contrastive_setpoint_value.py fit",
    )
    parser.add_argument(
        "--wm-ckpt",
        default=str(DEXMG_ROOT / "outputs/threepiece_20hz_wm_vae8k_h4_attn_jointpos_bs512_520epoch_20260727/wm_best.pt"),
    )
    parser.add_argument(
        "--vae-ckpt",
        default=str(POLICY_ROOT / "runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727/vae_step008000.pt"),
    )
    parser.add_argument(
        "--normalizers",
        default=str(DEXMG_ROOT / "outputs/threepiece_20hz_wm_vae8k_h4_attn_jointpos_bs512_520epoch_20260727/normalizers.npz"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--trust-delta", type=float, default=0.005)
    parser.add_argument("--qp-lambda-u", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--video-height", type=int, default=384)
    parser.add_argument("--video-width", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-composite-videos", action="store_true")
    parser.add_argument("--seed", type=int, default=4700)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_composite_videos:
        args.save_videos = True
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

    print(f"[setup] policy_model={args.model_path}", flush=True)
    print(f"[setup] wm_ckpt={args.wm_ckpt}", flush=True)
    print(f"[setup] params={args.params}", flush=True)
    print(f"[setup] trust_delta={args.trust_delta} qp_lambda_u={args.qp_lambda_u}", flush=True)

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
    combined_path = out_dir / "combined_rollouts.mp4"
    combined_writer = (
        imageio.get_writer(combined_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
        if args.save_videos
        else None
    )
    composite_combined_path = out_dir / "combined_rollouts_rgb_tactile_vz.mp4"
    composite_combined_writer = (
        imageio.get_writer(composite_combined_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
        if args.save_composite_videos
        else None
    )
    for ep in range(args.episodes):
        print(f"[rollout] contrastive residual episode={ep} nfe={args.nfe}", flush=True)
        data = collect_one_residual_rollout(env, config, dataset, agent, wm, params, normalizers, args.nfe, ep, args)
        values, betas = score_values_20hz(
            wm,
            params,
            data["high_tactile"],
            data["high_joint"],
            data["actions"],
            normalizers,
            args.history,
            args.chunk,
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
        if args.save_videos:
            label = (
                f"ep={ep:02d} success={int(data['success'])} steps={data['steps']} "
                f"v_mean={ep_meta['value_mean']:.3f} d_mean={ep_meta['delta_abs_max_mean']:.4f}"
            )
            ep_video = out_dir / "videos" / f"episode_{ep:02d}.mp4"
            write_episode_video(ep_video, data["rgb"], label, args.fps)
            ep_meta["video"] = str(ep_video)
            assert combined_writer is not None
            for frame in data["rgb"]:
                combined_writer.append_data(label_frame(frame, label))
            if args.save_composite_videos:
                composite_video = out_dir / "videos" / f"episode_{ep:02d}_rgb_tactile_vz.mp4"
                write_composite_episode_video(
                    composite_video,
                    data["rgb"],
                    data["high_tactile"],
                    values,
                    betas,
                    label,
                    args.fps,
                )
                ep_meta["composite_video"] = str(composite_video)
                assert composite_combined_writer is not None
                n_composite = min(len(data["rgb"]), len(data["high_tactile"]), len(values))
                for idx in range(n_composite):
                    composite_combined_writer.append_data(
                        make_composite_frame(
                            data["rgb"][idx],
                            data["high_tactile"][idx, -1],
                            values,
                            betas,
                            idx,
                            label,
                        )
                    )
        episodes.append(ep_meta)
        print(f"[rollout] {json.dumps({k: v for k, v in ep_meta.items() if k not in ('values', 'betas', 'residual')})}", flush=True)

    if combined_writer is not None:
        combined_writer.close()
    if composite_combined_writer is not None:
        composite_combined_writer.close()

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
        "trust_delta": args.trust_delta,
        "qp_lambda_u": args.qp_lambda_u,
        "combined_video": str(combined_path) if args.save_videos else None,
        "composite_combined_video": str(composite_combined_path) if args.save_composite_videos else None,
        "value_timebase": "one V_alpha per 20Hz low-control step; tactile/joint use last high200 sample",
        "residual_contract": "base policy 20Hz; residual also low20 raw 14D env delta; WM actions normalized with 20Hz normalizers",
        "episodes": episodes,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] success={success_count}/{args.episodes}", flush=True)
    print(f"[done] plot={plot_path}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
