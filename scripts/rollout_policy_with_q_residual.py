#!/usr/bin/env python3
"""Roll out tactile flow policy with a frozen WM/Q-head residual controller."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from rollout_policy_with_q_video import (
    CAMERA_NAMES,
    DEXMG_ROOT,
    POLICY_ROOT,
    QHead,
    ThreePieceWM,
    TrainingAgent,
    action20_to_action14,
    high_sample,
    make_config,
    make_dataset,
    make_env,
    make_video_frame,
    normalize_obs,
    obs_from_raw,
    render_cameras,
    score_q_values,
    stack_last,
)
from mip.dataset_utils import RotationTransformer


_COMPILED_Q_GRAD_CACHE = {}
_FAILED_Q_GRAD_COMPILE_KEYS = set()


def normalize_action(action: torch.Tensor, normalizers: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(normalizers["action_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(normalizers["action_std"], device=device, dtype=torch.float32)
    return (action - mean) / std


def resolve_ckpt_path(path: str) -> str:
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = DEXMG_ROOT / p
    return str(p.resolve())


def same_path(a: str, b: str) -> bool:
    return resolve_ckpt_path(a) == resolve_ckpt_path(b)


def q_logit_for_future(
    wm: ThreePieceWM,
    q_head: QHead,
    tactile_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_high_raw: torch.Tensor,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    history = int(tactile_hist_np.shape[0])
    chunk = int(future_high_raw.shape[0])
    tactile = torch.from_numpy(np.clip(tactile_hist_np, 0.0, 1.0)[None]).to(device=device, dtype=torch.float32)
    joint_np = (joint_hist_np.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
    joint = torch.from_numpy(joint_np[None]).to(device=device, dtype=torch.float32)
    action_hist = torch.from_numpy(action_hist_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    action_full = torch.cat([action_hist, future_high_raw], dim=0)[None]
    action_norm = normalize_action(action_full, normalizers, device)

    z_hist = wm.encode({"tactile": tactile, "joint": joint})
    z_window = z_hist
    for k in range(chunk):
        a_win = wm.action(action_norm[:, k : k + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    action_chunk = action_norm[:, history : history + chunk]
    return q_head(z_hist, action_chunk, z_window[:, -1]).squeeze(0)


def q_logit_for_future_tensors(
    wm: ThreePieceWM,
    q_head: QHead,
    tactile: torch.Tensor,
    joint_raw: torch.Tensor,
    action_hist_raw: torch.Tensor,
    future_high_raw: torch.Tensor,
    joint_mean: torch.Tensor,
    joint_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    joint = (joint_raw - joint_mean) / joint_std
    action_full = torch.cat([action_hist_raw, future_high_raw], dim=0)[None]
    action_norm = (action_full - action_mean) / action_std
    z_hist = wm.encode({"tactile": tactile[None], "joint": joint[None]})
    z_window = z_hist
    chunk = future_high_raw.shape[0]
    history = tactile.shape[0]
    for k in range(chunk):
        a_win = wm.action(action_norm[:, k : k + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    action_chunk = action_norm[:, history : history + chunk]
    return q_head(z_hist, action_chunk, z_window[:, -1]).squeeze(0)


def compiled_q_grad_value(
    wm: ThreePieceWM,
    q_head: QHead,
    tactile: torch.Tensor,
    joint_raw: torch.Tensor,
    action_hist_raw: torch.Tensor,
    base_low: torch.Tensor,
    residual_low: torch.Tensor,
    normalizer_tensors: dict[str, torch.Tensor],
    chunk: int,
    high_offset: int,
    compile_mode: str,
):
    low_horizon = int(base_low.shape[0])
    key = (int(chunk), int(high_offset), int(low_horizon), str(base_low.device), str(compile_mode))
    if key in _FAILED_Q_GRAD_COMPILE_KEYS:
        raise RuntimeError(f"compiled q grad disabled for key={key}")
    fn = _COMPILED_Q_GRAD_CACHE.get(key)
    if fn is None:
        def score_fn(tactile_t, joint_t, action_hist_t, base_low_t, residual_low_t):
            future_high = expand_low_to_high(base_low_t + residual_low_t, int(chunk), offset=int(high_offset))
            return q_logit_for_future_tensors(
                wm,
                q_head,
                tactile_t,
                joint_t,
                action_hist_t,
                future_high,
                normalizer_tensors["joint_mean"],
                normalizer_tensors["joint_std"],
                normalizer_tensors["action_mean"],
                normalizer_tensors["action_std"],
            ).sum()

        fn = torch.compile(
            torch.func.grad_and_value(score_fn, argnums=4),
            mode=compile_mode,
            fullgraph=False,
        )
        _COMPILED_Q_GRAD_CACHE[key] = fn
    return fn(tactile, joint_raw, action_hist_raw, base_low, residual_low)


def expand_low_to_high(low_actions: torch.Tensor, chunk: int, offset: int = 0) -> torch.Tensor:
    high = low_actions.repeat_interleave(10, dim=0)
    if high.shape[0] < chunk + offset:
        pad = high[-1:].expand(chunk + offset - high.shape[0], -1)
        high = torch.cat([high, pad], dim=0)
    return high[offset : offset + chunk]


def pad_high_actions(high_actions: np.ndarray, n_low: int) -> np.ndarray:
    high = np.asarray(high_actions, dtype=np.float32).reshape(-1, 14)
    target = n_low * 10
    if high.shape[0] == 0:
        return np.zeros((target, 14), dtype=np.float32)
    if high.shape[0] < target:
        pad = np.repeat(high[-1:], target - high.shape[0], axis=0)
        high = np.concatenate([high, pad], axis=0)
    return high[:target]


@torch.no_grad()
def score_q_values_high_actions(
    wm: ThreePieceWM,
    q_head: QHead,
    high_tactile: np.ndarray,
    high_joint: np.ndarray,
    high_actions: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    n_low = high_tactile.shape[0]
    if n_low == 0:
        return np.zeros((0,), dtype=np.float32)

    tactile_flat = high_tactile.reshape(-1, 4, 32, 32).astype(np.float32)
    tactile_flat = np.clip(tactile_flat, 0.0, 1.0)
    joint_flat = high_joint.reshape(-1, high_joint.shape[-1]).astype(np.float32)
    action_flat = pad_high_actions(high_actions, n_low)

    q_values = []
    window = history + chunk
    for low_start in range(0, n_low, batch_size):
        lows = list(range(low_start, min(low_start + batch_size, n_low)))
        tactile_batch, joint_batch, action_batch = [], [], []
        for low_idx in lows:
            high_idx = low_idx * 10 + 9
            start = high_idx - history + 1
            tactile_batch.append(score_pad_slice(tactile_flat, start, history))
            joint_batch.append(score_pad_slice(joint_flat, start, history))
            action_batch.append(score_pad_slice(action_flat, start, window))
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
            a_win = wm.action(sample["action"][:, k : k + history])
            pred_seq = wm.predictor(z_window, a_win)
            next_z = pred_seq[:, -1]
            z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
        action_chunk = sample["action"][:, history : history + chunk]
        logits = q_head(z_hist, action_chunk, z_window[:, -1])
        q_values.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).tolist())
    return np.asarray(q_values, dtype=np.float32)


def score_pad_slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    n = arr.shape[0]
    idx = np.arange(start, start + length)
    idx = np.clip(idx, 0, max(n - 1, 0))
    return arr[idx]


def optimize_residual(
    wm: ThreePieceWM,
    q_head: QHead,
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
    high_offset: int = 0,
    compile_q_grad: bool = False,
    normalizer_tensors: dict[str, torch.Tensor] | None = None,
    compile_mode: str = "reduce-overhead",
) -> dict[str, np.ndarray | float]:
    low_horizon = int(math.ceil((chunk + high_offset) / 10))
    future_low = np.asarray(future_low_actions_np, dtype=np.float32)
    if future_low.shape[0] < low_horizon:
        pad = np.repeat(future_low[-1:][None, 0], low_horizon - future_low.shape[0], axis=0)
        future_low = np.concatenate([future_low, pad], axis=0)
    future_low = future_low[:low_horizon]

    tactile_hist_np = np.stack(list(tactile_hist)[-history:], axis=0).astype(np.float32)
    joint_hist_np = np.stack(list(joint_hist)[-history:], axis=0).astype(np.float32)
    if len(action_hist) >= history:
        action_hist_np = np.stack(list(action_hist)[-history:], axis=0).astype(np.float32)
    else:
        action_hist_np = np.repeat(future_low[:1], history, axis=0).astype(np.float32)

    base_low = torch.from_numpy(future_low).to(device=device, dtype=torch.float32)
    residual_low = torch.zeros_like(base_low, requires_grad=True)

    compiled_ad_used = False
    if compile_q_grad:
        if normalizer_tensors is None:
            raise ValueError("compile_q_grad requires normalizer_tensors")
        tactile_t = torch.from_numpy(np.clip(tactile_hist_np, 0.0, 1.0)).to(device=device, dtype=torch.float32)
        joint_t = torch.from_numpy(joint_hist_np).to(device=device, dtype=torch.float32)
        action_hist_t = torch.from_numpy(action_hist_np).to(device=device, dtype=torch.float32)
        try:
            grad, logit = compiled_q_grad_value(
                wm,
                q_head,
                tactile_t,
                joint_t,
                action_hist_t,
                base_low.detach(),
                residual_low,
                normalizer_tensors,
                chunk,
                high_offset,
                compile_mode,
            )
            compiled_ad_used = True
        except Exception as exc:
            key = (int(chunk), int(high_offset), int(base_low.shape[0]), str(base_low.device), str(compile_mode))
            _FAILED_Q_GRAD_COMPILE_KEYS.add(key)
            print(f"[warn] compiled q grad failed, falling back to eager: {type(exc).__name__}: {exc}", flush=True)
            compile_q_grad = False

    if not compile_q_grad:
        future_high = expand_low_to_high(base_low + residual_low, chunk, offset=high_offset)
        with torch.enable_grad():
            logit = q_logit_for_future(
                wm,
                q_head,
                tactile_hist_np,
                joint_hist_np,
                action_hist_np,
                future_high,
                normalizers,
                device,
            )
            grad = torch.autograd.grad(logit, residual_low, retain_graph=False, create_graph=False)[0]
    delta_low = torch.clamp(grad / max(float(qp_lambda_u), 1e-12), -float(trust_delta), float(trust_delta)).detach()

    base_logit = logit.detach()
    residual_logit = torch.full_like(base_logit, float("nan"))

    delta0 = delta_low[0].detach().cpu().numpy().astype(np.float32)
    grad0 = grad[0].detach().cpu().numpy().astype(np.float32)
    return {
        "delta0": delta0,
        "grad0": grad0,
        "q_base": float(torch.sigmoid(base_logit).detach().cpu()),
        "q_residual": float(torch.sigmoid(residual_logit).detach().cpu()),
        "logit_base": float(base_logit.detach().cpu()),
        "logit_residual": float(residual_logit.detach().cpu()),
        "delta_abs_max": float(np.max(np.abs(delta0))),
        "delta_l2": float(np.linalg.norm(delta0)),
        "grad_abs_max": float(np.max(np.abs(grad0))),
        "clip_frac": float(np.mean(np.abs(delta_low.detach().cpu().numpy()) >= float(trust_delta) - 1e-9)),
        "compiled_ad_used": bool(compiled_ad_used),
    }


def step_one_model_tick(env, action: np.ndarray, policy_step: bool) -> None:
    if env.lite_physics:
        env.sim.step1()
    else:
        env.sim.forward()
    env._pre_action(action, policy_step)
    if env.lite_physics:
        env.sim.step2()
    else:
        env.sim.step()
    env._update_observables()


def finish_low_step(env, action: np.ndarray):
    env.cur_time += env.control_timestep
    reward, done, info = env._post_action(action)
    observations = env.viewer._get_observations() if env.viewer_get_obs else env._get_observations()
    return observations, reward, done, info


def run_low20_action(
    env,
    action: np.ndarray,
    low_step_index: int,
    tactile_hist: deque,
    joint_hist: deque,
    action_hist: deque,
    high_actions: list[np.ndarray],
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
    last_action = np.asarray(action, dtype=np.float32)

    for i in range(n_substeps):
        step_one_model_tick(env, last_action, policy_step=(i == 0))
        actual_time = start_time + (i + 1) * model_dt
        while next_sample < 10:
            target_time = start_time + (next_sample + 1) * sample_dt
            if actual_time + 1e-12 < target_time:
                break
            sample = high_sample(env)
            samples.append(sample)
            tactile_hist.append(sample["tactile"])
            joint_hist.append(sample["robot_joint_pos"])
            action_hist.append(last_action)
            high_actions.append(last_action.copy())
            next_sample += 1

    while next_sample < 10:
        sample = high_sample(env)
        samples.append(sample)
        tactile_hist.append(sample["tactile"])
        joint_hist.append(sample["robot_joint_pos"])
        action_hist.append(last_action)
        high_actions.append(last_action.copy())
        next_sample += 1

    raw_obs, reward, done, info = finish_low_step(env, last_action)
    return raw_obs, reward, done, info, samples


def run_high200_residual_action(
    env,
    base_act14: np.ndarray,
    future_low: np.ndarray,
    low_step_index: int,
    tactile_hist: deque,
    joint_hist: deque,
    action_hist: deque,
    high_actions: list[np.ndarray],
    wm,
    q_head,
    normalizers,
    normalizer_tensors,
    args,
    device: torch.device,
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
    residual_records = []
    last_action = np.asarray(base_act14, dtype=np.float32)
    last_delta = np.zeros_like(last_action, dtype=np.float32)
    last_opt = None

    for i in range(n_substeps):
        actual_time = start_time + i * model_dt
        target_time = start_time + next_sample * sample_dt
        if next_sample < 10 and actual_time + 1e-12 >= target_time:
            high_offset = int(next_sample)
            if high_offset % int(args.residual_solve_stride) == 0 or last_opt is None:
                opt = optimize_residual(
                    wm,
                    q_head,
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
                    high_offset=high_offset,
                    compile_q_grad=args.compile_q_grad,
                    normalizer_tensors=normalizer_tensors,
                    compile_mode=args.compile_mode,
                )
                last_delta = opt["delta0"].copy()
                last_opt = opt
            else:
                opt = dict(last_opt)
            # The base action is still a 20 Hz policy command; the residual goal is updated at 200 Hz.
            base_for_sample = future_low[min(high_offset // 10, future_low.shape[0] - 1)]
            last_action = (base_for_sample + last_delta).astype(np.float32)
            rec = {k: v for k, v in opt.items() if k not in ("delta0", "grad0")}
            rec["high_offset"] = high_offset
            rec["solved"] = high_offset % int(args.residual_solve_stride) == 0
            residual_records.append(rec)

        step_one_model_tick(env, last_action, policy_step=True)
        actual_time_after = start_time + (i + 1) * model_dt
        while next_sample < 10:
            target_time_after = start_time + (next_sample + 1) * sample_dt
            if actual_time_after + 1e-12 < target_time_after:
                break
            sample = high_sample(env)
            samples.append(sample)
            tactile_hist.append(sample["tactile"])
            joint_hist.append(sample["robot_joint_pos"])
            action_hist.append(last_action)
            high_actions.append(last_action.copy())
            next_sample += 1

    while next_sample < 10:
        sample = high_sample(env)
        samples.append(sample)
        tactile_hist.append(sample["tactile"])
        joint_hist.append(sample["robot_joint_pos"])
        action_hist.append(last_action)
        high_actions.append(last_action.copy())
        next_sample += 1

    raw_obs, reward, done, info = finish_low_step(env, last_action)
    return raw_obs, reward, done, info, samples, residual_records


def collect_one_residual_rollout(env, config, dataset, agent, wm, q_head, normalizers, normalizer_tensors, nfe: int, episode_idx: int, args):
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
    action_hist = deque(maxlen=args.history)

    rgb_frames, low_actions, base_actions = [], [], []
    rewards, successes = [], []
    high_tactile, high_joint = [], []
    residual_meta = []
    residual_high_meta = []
    high_actions = []
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
            future_low = act14_seq[j : j + int(math.ceil(args.chunk / 10))].astype(np.float32)
            if future_low.shape[0] == 0:
                future_low = base_act14[None]
            if args.residual_rate == "high200":
                raw_obs, reward, done, _, samples, hi_records = run_high200_residual_action(
                    env,
                    base_act14,
                    future_low,
                    low_steps,
                    tactile_hist,
                    joint_hist,
                    action_hist,
                    high_actions,
                    wm,
                    q_head,
                    normalizers,
                    normalizer_tensors,
                    args,
                    device,
                )
                residual_act14 = high_actions[-1].astype(np.float32)
                opt = dict(hi_records[-1]) if hi_records else {
                    "q_base": float("nan"),
                    "q_residual": float("nan"),
                    "delta_abs_max": 0.0,
                    "delta_l2": 0.0,
                    "grad_abs_max": 0.0,
                    "clip_frac": 0.0,
                }
                residual_high_meta.extend(hi_records)
                if args.progress_every > 0 and low_steps % args.progress_every == 0:
                    solved = sum(1 for r in hi_records if r.get("solved", False))
                    print(
                        f"[progress] ep={episode_idx} low_step={low_steps} "
                        f"success={int(success)} q_res={opt.get('q_residual', float('nan')):.3f} "
                        f"delta_max={opt.get('delta_abs_max', float('nan')):.4f} "
                        f"high_solves_this_step={solved}",
                        flush=True,
                    )
            else:
                opt = optimize_residual(
                    wm,
                    q_head,
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
                    compile_q_grad=args.compile_q_grad,
                    normalizer_tensors=normalizer_tensors,
                    compile_mode=args.compile_mode,
                )
                residual_act14 = (base_act14 + opt["delta0"]).astype(np.float32)
                raw_obs, reward, done, _, samples = run_low20_action(
                    env,
                    residual_act14,
                    low_steps,
                    tactile_hist,
                    joint_hist,
                    action_hist,
                    high_actions,
                )
                if args.progress_every > 0 and low_steps % args.progress_every == 0:
                    print(
                        f"[progress] ep={episode_idx} low_step={low_steps} "
                        f"success={int(success)} q_res={opt.get('q_residual', float('nan')):.3f} "
                        f"delta_max={opt.get('delta_abs_max', float('nan')):.4f}",
                        flush=True,
                    )
            total_reward += float(reward)
            success = success or bool(env._check_success()) or float(reward) > 0.0

            if not args.no_video:
                rgb_frames.append(render_cameras(env, CAMERA_NAMES, args.video_height, args.video_width))
            low_actions.append(residual_act14)
            base_actions.append(base_act14)
            rewards.append(np.float32(reward))
            successes.append(np.float32(success))
            high_tactile.append(np.stack([s["tactile"] for s in samples], axis=0))
            high_joint.append(np.stack([s["robot_joint_pos"] for s in samples], axis=0))
            residual_meta.append({k: v for k, v in opt.items() if k not in ("delta0", "grad0")})

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
        "rgb": np.asarray(rgb_frames, dtype=np.uint8),
        "actions": np.asarray(low_actions, dtype=np.float32),
        "base_actions": np.asarray(base_actions, dtype=np.float32),
        "high_actions": np.asarray(high_actions, dtype=np.float32).reshape(-1, 14),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "high_tactile": np.asarray(high_tactile, dtype=np.float32),
        "high_joint": np.asarray(high_joint, dtype=np.float32),
        "residual": residual_meta,
        "residual_high": residual_high_meta,
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
        default=str(POLICY_ROOT / "logs_dexmg_three_piece_tactile_cnn_virtual_s8fs25_flow_100k_20260726/models/model_step50000.pt"),
    )
    parser.add_argument(
        "--q-ckpt",
        default=str(DEXMG_ROOT / "outputs/threepiece_q_head_vae8k_wm021060_chunk20_stride10_10k_20260727/q_best.pt"),
    )
    parser.add_argument(
        "--wm-ckpt",
        default=str(DEXMG_ROOT / "outputs/threepiece_high200_wm_vae8k_h4_attn_jointpos_bs512_520epoch_evalfix_20260727/wm_step021060.pt"),
    )
    parser.add_argument(
        "--vae-ckpt",
        default=str(POLICY_ROOT / "runs/dexmg_shared_tactile_patch_vae_virtual_s8fs25_20260727/vae_step008000.pt"),
    )
    parser.add_argument(
        "--normalizers",
        default=str(DEXMG_ROOT / "outputs/threepiece_high200_wm_vae8k_h4_attn_jointpos_bs512_520epoch_evalfix_20260727/normalizers.npz"),
    )
    parser.add_argument("--output", default=str(DEXMG_ROOT / "outputs/qhead_policy_rollouts/flow50k_tactile_q_residual_5rollouts.mp4"))
    parser.add_argument("--summary", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--trust-delta", type=float, default=0.02)
    parser.add_argument("--qp-lambda-u", type=float, default=20.0)
    parser.add_argument("--residual-rate", choices=["low20", "high200"], default="high200")
    parser.add_argument("--residual-solve-stride", type=int, default=1, help="In high200 mode, solve every N high-rate ticks and hold the latest residual.")
    parser.add_argument("--compile-q-grad", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-height", type=int, default=384)
    parser.add_argument("--video-width", type=int, default=384)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--tactile-mode", choices=["max", "mean", "last"], default="max")
    parser.add_argument("--seed", type=int, default=3700)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary is None:
        args.summary = str(Path(args.output).with_suffix(".json"))

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

    print(f"[setup] policy_model={args.model_path}")
    print(f"[setup] wm_ckpt={args.wm_ckpt}")
    print(f"[setup] q_ckpt={args.q_ckpt}")
    print(f"[setup] residual trust_delta={args.trust_delta} qp_lambda_u={args.qp_lambda_u}", flush=True)
    print(f"[setup] residual_rate={args.residual_rate}", flush=True)
    if args.residual_solve_stride < 1:
        raise ValueError("--residual-solve-stride must be >= 1")
    print(f"[setup] residual_solve_stride={args.residual_solve_stride}", flush=True)
    print(f"[setup] compile_q_grad={int(args.compile_q_grad)} compile_mode={args.compile_mode}", flush=True)
    if args.compile_q_grad and device.type == "cuda":
        # torch.func.grad + torch.compile needs a differentiable attention backend.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    env = make_env(args.env_dataset_path, enable_tactile=True)
    if abs(float(env.control_timestep) - 0.05) > 1e-9:
        raise RuntimeError(f"Expected 20 Hz control_dt=0.05, got {env.control_timestep}")

    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(args.model_path, load_optimizer=False)
    agent.eval()

    wm = ThreePieceWM(args.vae_ckpt, history=args.history).to(device)
    wm_ckpt = torch.load(args.wm_ckpt, map_location=device)
    wm.load_state_dict(wm_ckpt["model"], strict=True)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    q_head = QHead(history=args.history, chunk=args.chunk).to(device)
    q_ckpt = torch.load(args.q_ckpt, map_location=device)
    q_cfg = q_ckpt.get("config", {})
    q_wm = q_cfg.get("wm_ckpt")
    if q_wm and not same_path(str(q_wm), args.wm_ckpt):
        raise RuntimeError(
            "Q-head / WM mismatch: "
            f"q checkpoint was trained with wm_ckpt={q_wm}, but --wm-ckpt={args.wm_ckpt}"
        )
    q_hist = int(q_cfg.get("history", args.history))
    q_chunk = int(q_cfg.get("chunk", args.chunk))
    if q_hist != int(args.history) or q_chunk != int(args.chunk):
        raise RuntimeError(
            "Q-head horizon mismatch: "
            f"checkpoint history/chunk={q_hist}/{q_chunk}, args history/chunk={args.history}/{args.chunk}"
        )
    q_head.load_state_dict(q_ckpt["q_head"], strict=True)
    q_head.eval()
    for p in q_head.parameters():
        p.requires_grad = False

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}
    normalizer_tensors = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in normalizers.items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_meta = []
    writer_ctx = (
        imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
        if not args.no_video
        else None
    )
    class NullWriter:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    with (writer_ctx if writer_ctx is not None else NullWriter()) as writer:
        for ep in range(args.episodes):
            print(f"[rollout] residual episode={ep} nfe={args.nfe}", flush=True)
            data = collect_one_residual_rollout(
                env,
                config,
                dataset,
                agent,
                wm,
                q_head,
                normalizers,
                normalizer_tensors,
                args.nfe,
                ep,
                args,
            )
            if args.residual_rate == "high200":
                q_values = score_q_values_high_actions(
                    wm,
                    q_head,
                    data["high_tactile"],
                    data["high_joint"],
                    data["high_actions"],
                    normalizers,
                    args.history,
                    args.chunk,
                    device,
                )
            else:
                q_values = score_q_values(
                    wm,
                    q_head,
                    data["high_tactile"],
                    data["high_joint"],
                    data["actions"],
                    normalizers,
                    args.history,
                    args.chunk,
                    device,
                )
            deltas = [r["delta_abs_max"] for r in data["residual"]]
            high_deltas = [r["delta_abs_max"] for r in data.get("residual_high", [])]
            q_base = [r["q_base"] for r in data["residual"]]
            q_res = [r["q_residual"] for r in data["residual"]]
            high_q_base = [r["q_base"] for r in data.get("residual_high", [])]
            high_q_res = [r["q_residual"] for r in data.get("residual_high", [])]
            meta = {
                "episode": ep,
                "success": data["success"],
                "reward_sum": data["reward_sum"],
                "steps": data["steps"],
                "q_applied_min": float(q_values.min()) if q_values.size else float("nan"),
                "q_applied_max": float(q_values.max()) if q_values.size else float("nan"),
                "q_base_mean": float(np.mean(q_base)) if q_base else float("nan"),
                "q_residual_mean": float(np.mean(q_res)) if q_res else float("nan"),
                "q_base_high_mean": float(np.mean(high_q_base)) if high_q_base else float("nan"),
                "q_residual_high_mean": float(np.mean(high_q_res)) if high_q_res else float("nan"),
                "delta_abs_max_mean": float(np.mean(deltas)) if deltas else float("nan"),
                "delta_abs_max_p95": float(np.percentile(deltas, 95)) if deltas else float("nan"),
                "delta_abs_max_high_mean": float(np.mean(high_deltas)) if high_deltas else float("nan"),
                "delta_abs_max_high_p95": float(np.percentile(high_deltas, 95)) if high_deltas else float("nan"),
                "high_residual_solves": len(high_deltas),
            }
            all_meta.append(meta)
            print(f"[rollout] {json.dumps(meta)}", flush=True)
            if writer is not None:
                for step in range(data["steps"]):
                    frame = make_video_frame(
                        data["rgb"][step],
                        data["high_tactile"][step],
                        q_values,
                        step,
                        ep,
                        data["success"],
                        data["reward_sum"],
                        args.tactile_mode,
                    )
                    writer.append_data(frame)

    summary = {
        "output": str(out),
        "policy_model": args.model_path,
        "q_ckpt": args.q_ckpt,
        "wm_ckpt": args.wm_ckpt,
        "nfe": args.nfe,
        "trust_delta": args.trust_delta,
        "qp_lambda_u": args.qp_lambda_u,
        "residual_rate": args.residual_rate,
        "residual_solve_stride": args.residual_solve_stride,
        "compile_q_grad": bool(args.compile_q_grad),
        "compile_mode": args.compile_mode,
        "no_video": bool(args.no_video),
        "episodes": all_meta,
        "residual_contract": (
            "base policy 20Hz; residual_rate=high200 updates raw 14D env delta command goal at 5ms "
            "sample boundaries using latest tactile+joint history; WM/Q training used low20 action repeated to high200, "
            "so high200-varying residual actions are out-of-distribution for the current WM"
            if args.residual_rate == "high200"
            else "base policy 20Hz; residual also low20; WM/Q scored with high200 tactile+joint and low20 action repeated to high200"
        ),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"[done] video={out}", flush=True)
    print(f"[done] summary={args.summary}", flush=True)


if __name__ == "__main__":
    main()
