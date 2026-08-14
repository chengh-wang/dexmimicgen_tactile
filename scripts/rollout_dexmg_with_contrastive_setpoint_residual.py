#!/usr/bin/env python3
"""Roll out a DexMimicGen policy with a 20 Hz contrastive WM residual."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections import deque
from pathlib import Path

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
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TACTILE_RENDERER", "virtual")
os.environ.setdefault("TACTILE_VIRTUAL_SIGMA", "8.0")
os.environ.setdefault("TACTILE_VIRTUAL_FORCE_SCALE", "25.0")
os.environ.setdefault("TACTILE_VIRTUAL_MAX", "1.0")
os.environ.setdefault("TACTILE_VIRTUAL_CANONICAL", "1")
os.environ.setdefault("TACTILE_VIRTUAL_MAX_SURFACE_DIST", "0.015")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from mip.dataset_utils import RotationTransformer  # noqa: E402
from examples.eval_dexmg_rollout import (  # noqa: E402
    TrainingAgent,
    make_config,
    make_dataset,
    make_env,
    normalize_obs,
    obs_from_raw,
    policy_action_to_env_action,
    stack_last,
)
from collect_dexmg_wm_rollouts import high_sample, step_with_high200  # noqa: E402
from train_dexmg_20hz_wm import DexMGWM  # noqa: E402


POLICY_ACTION_SUPPORTS_LAYOUT = "action_layout" in inspect.signature(policy_action_to_env_action).parameters


def convert_policy_action_to_env(action, env_action_dim, rotation_transformer, action_layout):
    if POLICY_ACTION_SUPPORTS_LAYOUT:
        return policy_action_to_env_action(action, env_action_dim, rotation_transformer, action_layout)
    return policy_action_to_env_action(action, env_action_dim, rotation_transformer)


def pad_future(actions: np.ndarray, chunk: int, action_dim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, action_dim)
    if actions.shape[0] == 0:
        actions = np.zeros((1, action_dim), dtype=np.float32)
    if actions.shape[0] < chunk:
        actions = np.concatenate([actions, np.repeat(actions[-1:], chunk - actions.shape[0], axis=0)], axis=0)
    return actions[:chunk]


def load_normalizers(path: str) -> dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k].astype(np.float32) for k in npz.files}


def load_params(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    npz = np.load(path)
    out = {k: torch.as_tensor(npz[k], dtype=torch.float32, device=device) for k in npz.files}
    out["param_type_code"] = torch.as_tensor(
        npz["param_type_code"] if "param_type_code" in npz.files else np.array(0.0, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    out["value_mode_code"] = torch.as_tensor(
        npz["value_mode_code"] if "value_mode_code" in npz.files else np.array(0.0, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
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


def _rbf_features(tau: torch.Tensor, centers: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    width = torch.clamp(width.float(), min=1e-4)
    phi = torch.exp(-0.5 * torch.square((tau[:, None] - centers[None].float()) / width))
    return phi / torch.clamp(phi.sum(dim=-1, keepdim=True), min=1e-6)


def _interp_table(table: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    n = int(table.shape[0])
    if n <= 1:
        return table[0].expand(tau.shape[0], *table.shape[1:])
    x = tau.clamp(0.0, 1.0) * (n - 1)
    lo = torch.floor(x).long().clamp(0, n - 1)
    hi = torch.clamp(lo + 1, max=n - 1)
    w = (x - lo.float()).reshape(-1, *([1] * (table.dim() - 1)))
    return table[lo] * (1.0 - w) + table[hi] * w


def setpoint_value_from_z(z: torch.Tensor, params: dict[str, torch.Tensor], progress=None):
    z = z.float()
    param_type = int(float(params["param_type_code"].detach().cpu()))
    tau = _progress_tensor(progress, z)
    if param_type == 10:
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
        phi = _rbf_features(tau, params["basis_centers"], params["basis_width"])
        w = torch.einsum("bm,md->bd", phi, params["W_table"].float())
        bias = phi @ params["b_table"].float()
        beta = torch.sum(z * w, dim=-1) + bias
        return torch.sigmoid(beta), beta
    if param_type == 12:
        beta = z @ params["v"].float()
        beta_star = _interp_table(params["beta_star_table"].float(), tau).reshape(-1)
        sigma = _interp_table(params["sigma_table"].float(), tau).reshape(-1)
        value = torch.exp(-torch.square((beta - beta_star) / torch.clamp(sigma, min=1e-6)))
        return value, beta
    raise RuntimeError(f"unsupported contrastive param_type_code={param_type}")


def setpoint_value_20hz(
    wm: DexMGWM,
    params: dict[str, torch.Tensor],
    tactile_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_low_raw: torch.Tensor,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    progress: float,
    score_mode: str = "terminal",
    traj_gamma: float = 1.0,
    progress0: float | None = None,
    progress_step: float = 0.0,
):
    tactile = torch.from_numpy(np.clip(tactile_hist_np, 0.0, 1.0)[None]).to(device=device, dtype=torch.float32)
    joint_np = (joint_hist_np.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
    joint = torch.from_numpy(joint_np[None]).to(device=device, dtype=torch.float32)
    action_past = torch.from_numpy(action_hist_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    action_full = torch.cat([action_past, future_low_raw], dim=0)[None]
    action_norm = normalize_action(action_full, normalizers, device)

    z_window = wm.encode({"tactile": tactile, "joint": joint})
    history = tactile_hist_np.shape[0]
    expected_action_len = history - 1 + future_low_raw.shape[0]
    if action_norm.shape[1] != expected_action_len:
        raise RuntimeError(
            f"bad action context shape {tuple(action_norm.shape)}; "
            f"expected length {expected_action_len} = history-1 + future"
        )
    score = None
    beta_last = None
    for k in range(future_low_raw.shape[0]):
        a_win = wm.action(action_norm[:, k : k + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
        if score_mode == "trajectory":
            tau_k = progress
            if progress0 is not None:
                tau_k = min(1.0, float(progress0) + float(k + 1) * float(progress_step))
            value_k, beta_k = setpoint_value_from_z(z_window[:, -1], params, tau_k)
            weight = float(traj_gamma) ** float(k + 1)
            score = value_k * weight if score is None else score + value_k * weight
            beta_last = beta_k
    if score_mode == "trajectory":
        return score.squeeze(0), beta_last.squeeze(0)
    value, beta = setpoint_value_from_z(z_window[:, -1], params, progress)
    return value.squeeze(0), beta.squeeze(0)


def optimize_residual(
    wm,
    params,
    tactile_hist,
    joint_hist,
    action_hist,
    future_low_actions_np,
    normalizers,
    history,
    chunk,
    action_dim,
    trust_delta,
    qp_lambda_u,
    residual_optimizer,
    iter_steps,
    iter_lr,
    score_mode,
    traj_gamma,
    progress0,
    progress_step,
    device,
    progress,
):
    tactile_hist_np = np.stack(list(tactile_hist)[-history:], axis=0).astype(np.float32)
    joint_hist_np = np.stack(list(joint_hist)[-history:], axis=0).astype(np.float32)
    action_hist_np = np.zeros((max(history - 1, 0), action_dim), dtype=np.float32)
    if history > 1 and len(action_hist) > 0:
        tail = np.stack(list(action_hist)[-(history - 1) :], axis=0).astype(np.float32)
        action_hist_np[-tail.shape[0] :] = tail
    base_low = torch.from_numpy(pad_future(future_low_actions_np, chunk, action_dim)).to(device=device, dtype=torch.float32)
    residual_optimizer = str(residual_optimizer)
    if residual_optimizer == "closed_form":
        residual_low = torch.zeros_like(base_low, requires_grad=True)
        with torch.enable_grad():
            base_value, base_beta = setpoint_value_20hz(
                wm, params, tactile_hist_np, joint_hist_np, action_hist_np, base_low + residual_low,
                normalizers, device, progress, score_mode, traj_gamma, progress0, progress_step
            )
            grad = torch.autograd.grad(base_value, residual_low, retain_graph=False, create_graph=False)[0]
        delta_low = torch.clamp(grad / max(float(qp_lambda_u), 1e-12), -float(trust_delta), float(trust_delta)).detach()
    elif residual_optimizer in {"iter_value", "iter_reg"}:
        with torch.no_grad():
            base_value, base_beta = setpoint_value_20hz(
                wm, params, tactile_hist_np, joint_hist_np, action_hist_np, base_low,
                normalizers, device, progress, score_mode, traj_gamma, progress0, progress_step
            )
        delta_low = torch.zeros_like(base_low)
        grad = torch.zeros_like(base_low)
        steps = max(int(iter_steps), 1)
        lr = float(iter_lr)
        for _ in range(steps):
            residual_low = delta_low.detach().requires_grad_(True)
            with torch.enable_grad():
                value, _ = setpoint_value_20hz(
                    wm,
                    params,
                    tactile_hist_np,
                    joint_hist_np,
                    action_hist_np,
                    base_low + residual_low,
                    normalizers,
                    device,
                    progress,
                    score_mode,
                    traj_gamma,
                    progress0,
                    progress_step,
                )
                if residual_optimizer == "iter_reg":
                    objective = value - 0.5 * float(qp_lambda_u) * torch.sum(torch.square(residual_low))
                else:
                    objective = value
                grad = torch.autograd.grad(objective, residual_low, retain_graph=False, create_graph=False)[0]
            delta_low = torch.clamp(residual_low + lr * grad, -float(trust_delta), float(trust_delta)).detach()
    else:
        raise RuntimeError(f"unsupported residual_optimizer={residual_optimizer}")
    with torch.no_grad():
        residual_value, residual_beta = setpoint_value_20hz(
            wm, params, tactile_hist_np, joint_hist_np, action_hist_np, base_low + delta_low,
            normalizers, device, progress, score_mode, traj_gamma, progress0, progress_step
        )
    delta = delta_low.detach().cpu().numpy()
    return {
        "delta0": delta[0].astype(np.float32),
        "value_base": float(base_value.detach().cpu()),
        "value_residual": float(residual_value.detach().cpu()),
        "beta_base": float(base_beta.detach().cpu()),
        "beta_residual": float(residual_beta.detach().cpu()),
        "delta_abs_max": float(np.max(np.abs(delta[0]))),
        "delta_l2": float(np.linalg.norm(delta[0])),
        "grad_abs_max": float(np.max(np.abs(grad[0].detach().cpu().numpy()))),
        "clip_frac": float(np.mean(np.abs(delta) >= float(trust_delta) - 1e-9)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--env-dataset-path", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--trust-delta", type=float, default=0.01)
    parser.add_argument("--qp-lambda-u", type=float, default=20.0)
    parser.add_argument("--residual-optimizer", choices=["closed_form", "iter_value", "iter_reg"], default="closed_form")
    parser.add_argument("--iter-steps", type=int, default=1)
    parser.add_argument("--iter-lr", type=float, default=0.01)
    parser.add_argument("--score-mode", choices=["terminal", "trajectory"], default="terminal")
    parser.add_argument("--traj-gamma", type=float, default=0.97)
    parser.add_argument("--tactile-channels", type=int, default=12)
    parser.add_argument("--joint-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=24)
    parser.add_argument("--seed", type=int, default=5200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

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

    env = make_env(args.env_dataset_path, enable_tactile=True)
    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(args.model_path, load_optimizer=False)
    agent.eval()

    wm = DexMGWM(
        args.vae_ckpt,
        history=args.history,
        tactile_channels=args.tactile_channels,
        joint_dim=args.joint_dim,
        action_dim=args.action_dim,
    ).to(device)
    ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    normalizers = load_normalizers(args.normalizers)
    params = load_params(args.params, device)
    rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

    episodes = []
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        raw_obs = env.reset()
        obs_hist = deque(maxlen=config.task.obs_steps)
        first_obs = obs_from_raw(raw_obs, config, dataset)
        for _ in range(config.task.obs_steps):
            obs_hist.append(first_obs)
        init = high_sample(env, 0.0, 0.0, 0)
        tactile_hist = deque([init["tactile"]] * args.history, maxlen=args.history)
        joint_hist = deque([init["robot_joint_pos"]] * args.history, maxlen=args.history)
        action_hist = deque(
            [np.zeros(args.action_dim, dtype=np.float32)] * max(args.history - 1, 0),
            maxlen=max(args.history - 1, 1),
        )

        low_steps = 0
        success = False
        total_reward = 0.0
        residual_meta = []
        while low_steps < args.max_episode_steps:
            obs_seq = stack_last(obs_hist, config.task.obs_steps)
            obs = normalize_obs(obs_seq, dataset, config.optimization.device)
            act_0 = torch.randn((1, config.task.horizon, config.task.act_dim), device=device, dtype=torch.float32)
            with torch.no_grad():
                act_normed = agent.sample(act_0=act_0, obs=obs, num_steps=args.nfe, use_ema=True)
            policy_action = dataset.normalizer["action"].unnormalize(act_normed.detach().cpu().numpy())[0]
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act_seq = convert_policy_action_to_env(
                policy_action[start:end],
                env.action_dim,
                rotation_transformer,
                getattr(config.task, "dexmg_action_layout", "legacy"),
            ).astype(np.float32)
            for j, base_action in enumerate(act_seq):
                future = act_seq[j : j + args.chunk].astype(np.float32)
                progress = min(1.0, float(low_steps + args.chunk) / max(float(args.max_episode_steps), 1.0))
                progress0 = min(1.0, float(low_steps) / max(float(args.max_episode_steps), 1.0))
                progress_step = 1.0 / max(float(args.max_episode_steps), 1.0)
                opt = optimize_residual(
                    wm,
                    params,
                    tactile_hist,
                    joint_hist,
                    action_hist,
                    future,
                    normalizers,
                    args.history,
                    args.chunk,
                    args.action_dim,
                    args.trust_delta,
                    args.qp_lambda_u,
                    args.residual_optimizer,
                    args.iter_steps,
                    args.iter_lr,
                    args.score_mode,
                    args.traj_gamma,
                    progress0,
                    progress_step,
                    device,
                    progress,
                )
                action = (base_action + opt["delta0"]).astype(np.float32)
                raw_obs, reward, done, _, samples = step_with_high200(env, action, low_steps)
                total_reward += float(reward)
                success = success or bool(env._check_success()) or float(reward) > 0.0
                last = samples[-1]
                tactile_hist.append(last["tactile"])
                joint_hist.append(last["robot_joint_pos"])
                action_hist.append(action)
                residual_meta.append({k: v for k, v in opt.items() if k != "delta0"})
                obs_hist.append(obs_from_raw(raw_obs, config, dataset))
                if args.progress_every > 0 and low_steps % args.progress_every == 0:
                    print(
                        f"[progress] ep={ep} step={low_steps} success={int(success)} "
                        f"v_base={opt['value_base']:.3f} v_res={opt['value_residual']:.3f} "
                        f"delta={opt['delta_abs_max']:.4f}",
                        flush=True,
                    )
                low_steps += 1
                if done or success or low_steps >= args.max_episode_steps:
                    break
            if success or low_steps >= args.max_episode_steps:
                break
        deltas = [m["delta_abs_max"] for m in residual_meta]
        clips = [m["clip_frac"] for m in residual_meta]
        values = [m["value_residual"] for m in residual_meta]
        meta = {
            "episode": ep,
            "success": bool(success),
            "steps": int(low_steps),
            "reward_sum": float(total_reward),
            "value_mean": float(np.mean(values)) if values else float("nan"),
            "value_last": float(values[-1]) if values else float("nan"),
            "delta_abs_max_mean": float(np.mean(deltas)) if deltas else float("nan"),
            "delta_abs_max_max": float(np.max(deltas)) if deltas else float("nan"),
            "clip_frac_mean": float(np.mean(clips)) if clips else float("nan"),
        }
        episodes.append(meta)
        print("[episode] " + json.dumps(meta), flush=True)

    success_count = int(sum(e["success"] for e in episodes))
    summary = {
        "success_count": success_count,
        "episodes_n": args.episodes,
        "success_rate": success_count / max(args.episodes, 1),
        "nfe": args.nfe,
        "trust_delta": args.trust_delta,
        "qp_lambda_u": args.qp_lambda_u,
        "residual_optimizer": args.residual_optimizer,
        "iter_steps": args.iter_steps,
        "iter_lr": args.iter_lr,
        "score_mode": args.score_mode,
        "traj_gamma": args.traj_gamma,
        "chunk": args.chunk,
        "wm_ckpt": args.wm_ckpt,
        "params": args.params,
        "episodes": episodes,
    }
    out.write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps({k: v for k, v in summary.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
