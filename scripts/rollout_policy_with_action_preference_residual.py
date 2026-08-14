#!/usr/bin/env python3
"""Roll out a tactile policy with a 20 Hz action-preference residual controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio
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
from rollout_policy_with_20hz_q_residual import label_frame, write_episode_video  # noqa: E402
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402
from train_threepiece_action_preference_value import QHead  # noqa: E402

CAMERA_NAMES = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]


def pad_future(actions: np.ndarray, chunk: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 14)
    if actions.shape[0] == 0:
        actions = np.zeros((1, 14), dtype=np.float32)
    if actions.shape[0] < chunk:
        pad = np.repeat(actions[-1:], chunk - actions.shape[0], axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    return actions[:chunk]


def normalize_action(action: torch.Tensor, normalizers: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(normalizers["action_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(normalizers["action_std"], device=device, dtype=torch.float32)
    return (action - mean) / std


def q_logit_action_pref(
    wm: ThreePieceWM,
    q_head: QHead,
    tactile_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_context_np: np.ndarray,
    future_low_raw: torch.Tensor,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    history = tactile_hist_np.shape[0]
    if action_context_np.shape[0] != history - 1:
        raise RuntimeError(f"expected {history - 1} past actions, got {action_context_np.shape}")
    tactile = torch.from_numpy(np.clip(tactile_hist_np, 0.0, 1.0)[None]).to(device=device, dtype=torch.float32)
    joint_np = (joint_hist_np.astype(np.float32) - normalizers["joint_mean"]) / normalizers["joint_std"]
    joint = torch.from_numpy(joint_np[None]).to(device=device, dtype=torch.float32)
    action_context = torch.from_numpy(action_context_np.astype(np.float32)).to(device=device, dtype=torch.float32)
    action_full = torch.cat([action_context, future_low_raw], dim=0)[None]
    action_norm = normalize_action(action_full, normalizers, device)

    z_hist = wm.encode({"tactile": tactile, "joint": joint})
    z_window = z_hist
    chunk = future_low_raw.shape[0]
    for k in range(chunk):
        a_win = wm.action(action_norm[:, k : k + history])
        pred_seq = wm.predictor(z_window, a_win)
        next_z = pred_seq[:, -1]
        z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
    action_chunk = action_norm[:, history - 1 : history - 1 + chunk]
    return q_head(z_hist, action_chunk, z_window[:, -1]).squeeze(0)


def optimize_residual_action_pref(
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
    action_reg: float,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    tactile_hist_np = np.stack(list(tactile_hist)[-history:], axis=0).astype(np.float32)
    joint_hist_np = np.stack(list(joint_hist)[-history:], axis=0).astype(np.float32)
    if len(action_hist) >= history - 1:
        action_context_np = np.stack(list(action_hist)[-(history - 1) :], axis=0).astype(np.float32)
    else:
        action_context_np = np.zeros((history - 1, 14), dtype=np.float32)

    base_low = torch.from_numpy(pad_future(future_low_actions_np, chunk)).to(device=device, dtype=torch.float32)
    residual_low = torch.zeros_like(base_low, requires_grad=True)
    with torch.enable_grad():
        base_logit = q_logit_action_pref(
            wm,
            q_head,
            tactile_hist_np,
            joint_hist_np,
            action_context_np,
            base_low + residual_low,
            normalizers,
            device,
        )
        grad = torch.autograd.grad(base_logit, residual_low, retain_graph=False, create_graph=False)[0]
    delta_low = torch.clamp(grad / max(float(action_reg), 1e-12), -float(trust_delta), float(trust_delta)).detach()
    with torch.no_grad():
        residual_logit = q_logit_action_pref(
            wm,
            q_head,
            tactile_hist_np,
            joint_hist_np,
            action_context_np,
            base_low + delta_low,
            normalizers,
            device,
        )
    delta_np = delta_low.detach().cpu().numpy()
    delta0 = delta_np[0].astype(np.float32)
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
        "delta_chunk_abs_max": float(np.max(np.abs(delta_np))),
        "delta_chunk_l2": float(np.linalg.norm(delta_np)),
        "grad_abs_max": float(np.max(np.abs(grad0))),
        "clip_frac": float(np.mean(np.abs(delta_np) >= float(trust_delta) - 1e-9)),
    }


def pad_slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    n = arr.shape[0]
    if n == 0:
        raise RuntimeError("cannot pad empty array")
    idx = np.arange(start, start + length)
    idx = np.clip(idx, 0, n - 1)
    return arr[idx]


@torch.no_grad()
def score_q_values_action_pref(
    wm: ThreePieceWM,
    q_head: QHead,
    high_tactile: np.ndarray,
    high_joint: np.ndarray,
    low_actions: np.ndarray,
    normalizers: dict[str, np.ndarray],
    history: int,
    chunk: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    n_low = int(low_actions.shape[0])
    if n_low == 0:
        return np.zeros((0,), dtype=np.float32)
    tactile_low = np.clip(high_tactile[:, -1].astype(np.float32), 0.0, 1.0)
    joint_low = high_joint[:, -1].astype(np.float32)
    action_low = low_actions.astype(np.float32)
    q_values: list[float] = []
    action_len = history - 1 + chunk
    for low_start in range(0, n_low, batch_size):
        lows = list(range(low_start, min(low_start + batch_size, n_low)))
        tactile_batch, joint_batch, action_batch = [], [], []
        for low_idx in lows:
            hist_start = low_idx - history + 1
            tactile_batch.append(pad_slice(tactile_low, hist_start, history))
            joint_batch.append(pad_slice(joint_low, hist_start, history))
            action_batch.append(pad_slice(action_low, hist_start, action_len))
        tactile = np.stack(tactile_batch, axis=0)
        joint = np.stack(joint_batch, axis=0)
        action = np.stack(action_batch, axis=0)
        joint = (joint - normalizers["joint_mean"]) / normalizers["joint_std"]
        action = (action - normalizers["action_mean"]) / normalizers["action_std"]
        tactile_t = torch.from_numpy(tactile).to(device, non_blocking=True)
        joint_t = torch.from_numpy(joint).to(device, non_blocking=True)
        action_t = torch.from_numpy(action).to(device, non_blocking=True)
        z_hist = wm.encode({"tactile": tactile_t, "joint": joint_t})
        z_window = z_hist
        for k in range(chunk):
            a_win = wm.action(action_t[:, k : k + history])
            pred_seq = wm.predictor(z_window, a_win)
            next_z = pred_seq[:, -1]
            z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
        action_chunk = action_t[:, history - 1 : history - 1 + chunk]
        logits = q_head(z_hist, action_chunk, z_window[:, -1])
        q_values.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).tolist())
    return np.asarray(q_values, dtype=np.float32)


def collect_one_residual_rollout(env, config, dataset, agent, wm, q_head, normalizers, nfe: int, episode_idx: int, args):
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
    action_hist = deque([np.zeros(14, dtype=np.float32)] * (args.history - 1), maxlen=args.history - 1)

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
            opt = optimize_residual_action_pref(
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
                args.action_reg,
                device,
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
                    f"q_base={opt['q_base']:.3f} q_res={opt['q_residual']:.3f} "
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
        "rgb": np.asarray(rgb_frames, dtype=np.uint8),
        "actions": np.asarray(low_actions, dtype=np.float32),
        "base_actions": np.asarray(base_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "high_tactile": np.asarray(high_tactile, dtype=np.float32),
        "high_joint": np.asarray(high_joint, dtype=np.float32),
        "residual": residual_meta,
    }


def save_q_plot(episodes: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=150)
    for ep in episodes:
        q = np.asarray(ep["q_values"], dtype=np.float32)
        x = np.arange(q.shape[0], dtype=np.float32) / 20.0
        label = (
            f"ep{ep['episode']} s={int(ep['success'])} steps={ep['steps']} "
            f"q_last={ep['q_last']:.3f} d={ep['delta_abs_max_mean']:.4f}"
        )
        ax.plot(x, q, linewidth=1.5, label=label)
    ax.set_title("Action-preference residual value score")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sigmoid(value logit)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def aggregate_summary(episodes: list[dict[str, Any]]) -> dict[str, float]:
    successes = [bool(ep["success"]) for ep in episodes]
    rewards = [float(ep["reward_sum"]) for ep in episodes]
    deltas = [float(ep["delta_abs_max_mean"]) for ep in episodes if math.isfinite(float(ep["delta_abs_max_mean"]))]
    q_base = [float(ep["q_base_mean"]) for ep in episodes if math.isfinite(float(ep["q_base_mean"]))]
    q_res = [float(ep["q_residual_mean"]) for ep in episodes if math.isfinite(float(ep["q_residual_mean"]))]
    return {
        "success_count": int(sum(successes)),
        "episodes": int(len(episodes)),
        "success_rate": float(sum(successes) / max(len(successes), 1)),
        "mean_return_or_score": float(np.mean(rewards)) if rewards else float("nan"),
        "delta_abs_max_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "q_base_mean": float(np.mean(q_base)) if q_base else float("nan"),
        "q_residual_mean": float(np.mean(q_res)) if q_res else float("nan"),
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
    parser.add_argument("--q-ckpt", required=True)
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
    parser.add_argument("--setting", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--trust-delta", type=float, default=0.005)
    parser.add_argument("--action-reg", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--video-height", type=int, default=384)
    parser.add_argument("--video-width", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-video-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4700)
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

    print(f"[setup] policy_model={args.model_path}", flush=True)
    print(f"[setup] wm_ckpt={args.wm_ckpt}", flush=True)
    print(f"[setup] action_pref_q_ckpt={args.q_ckpt}", flush=True)
    print(f"[setup] trust_delta={args.trust_delta} action_reg={args.action_reg}", flush=True)

    env = make_env(args.env_dataset_path, enable_tactile=True)
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
    if q_cfg.get("objective") != "same-history counterfactual action preference":
        raise RuntimeError(f"refusing non-action-preference Q checkpoint: objective={q_cfg.get('objective')}")
    if int(q_cfg.get("history", args.history)) != args.history or int(q_cfg.get("chunk", args.chunk)) != args.chunk:
        raise RuntimeError(f"Q checkpoint history/chunk mismatch: {q_cfg}")
    q_head.load_state_dict(q_ckpt["q_head"], strict=True)
    q_head.eval()
    for p in q_head.parameters():
        p.requires_grad = False

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}

    episodes = []
    combined_path = out_dir / "combined_rollouts.mp4"
    combined_writer = (
        imageio.get_writer(combined_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
        if args.save_videos
        else None
    )
    episode_jsonl = out_dir / "episodes.jsonl"
    for local_ep in range(args.episodes):
        ep = args.episode_offset + local_ep
        print(f"[rollout] action_pref_residual episode={ep} nfe={args.nfe}", flush=True)
        data = collect_one_residual_rollout(env, config, dataset, agent, wm, q_head, normalizers, args.nfe, ep, args)
        q_values = score_q_values_action_pref(
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
        deltas = [float(r["delta_abs_max"]) for r in data["residual"]]
        q_base = [float(r["q_base"]) for r in data["residual"]]
        q_res = [float(r["q_residual"]) for r in data["residual"]]
        clip = [float(r["clip_frac"]) for r in data["residual"]]
        ep_meta = {
            "episode": ep,
            "success": bool(data["success"]),
            "reward_sum": float(data["reward_sum"]),
            "steps": int(data["steps"]),
            "q_min": float(q_values.min()) if q_values.size else float("nan"),
            "q_max": float(q_values.max()) if q_values.size else float("nan"),
            "q_mean": float(q_values.mean()) if q_values.size else float("nan"),
            "q_first": float(q_values[0]) if q_values.size else float("nan"),
            "q_last": float(q_values[-1]) if q_values.size else float("nan"),
            "q_base_mean": float(np.mean(q_base)) if q_base else float("nan"),
            "q_residual_mean": float(np.mean(q_res)) if q_res else float("nan"),
            "delta_abs_max_mean": float(np.mean(deltas)) if deltas else float("nan"),
            "delta_abs_max_max": float(np.max(deltas)) if deltas else float("nan"),
            "clip_frac_mean": float(np.mean(clip)) if clip else float("nan"),
            "q_values": q_values.astype(float).tolist(),
            "residual": data["residual"],
        }
        should_save_video = args.save_videos and (args.save_video_every <= 0 or ep % args.save_video_every == 0)
        if should_save_video:
            label = (
                f"{args.setting} ep={ep:02d} success={int(data['success'])} steps={data['steps']} "
                f"q_mean={ep_meta['q_mean']:.3f} d_mean={ep_meta['delta_abs_max_mean']:.4f}"
            )
            ep_video = out_dir / "videos" / f"episode_{ep:02d}.mp4"
            write_episode_video(ep_video, data["rgb"], label, args.fps)
            ep_meta["video"] = str(ep_video)
            assert combined_writer is not None
            for frame in data["rgb"]:
                combined_writer.append_data(label_frame(frame, label))
        episodes.append(ep_meta)
        with episode_jsonl.open("a") as f:
            f.write(json.dumps(ep_meta) + "\n")
        print(f"[rollout] {json.dumps({k: v for k, v in ep_meta.items() if k not in ('q_values', 'residual')})}", flush=True)

    if combined_writer is not None:
        combined_writer.close()

    plot_path = out_dir / "q_curves.png"
    save_q_plot(episodes, plot_path)
    agg = aggregate_summary(episodes)
    summary = {
        "method": "action_preference_value",
        "setting": args.setting,
        "plot": str(plot_path),
        "policy_model": args.model_path,
        "wm_ckpt": args.wm_ckpt,
        "head_ckpt": args.q_ckpt,
        "normalizers": args.normalizers,
        "nfe": args.nfe,
        "trust_delta": args.trust_delta,
        "action_reg": args.action_reg,
        "seed": args.seed,
        "combined_video": str(combined_path) if args.save_videos else None,
        "action_context": "history-1 past raw low20 actions + candidate future actions; normalized before WM/Q",
        "aggregate": agg,
        "episodes": episodes,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    with (out_dir / "summary_row.csv").open("w", newline="") as f:
        fieldnames = [
            "method",
            "head_ckpt",
            "setting",
            "success_count",
            "episodes",
            "success_rate",
            "mean_return_or_score",
            "trust/action_limit",
            "regularization",
            "seed",
            "output_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "method": "action_preference_value",
                "head_ckpt": args.q_ckpt,
                "setting": args.setting,
                "success_count": agg["success_count"],
                "episodes": agg["episodes"],
                "success_rate": agg["success_rate"],
                "mean_return_or_score": agg["mean_return_or_score"],
                "trust/action_limit": args.trust_delta,
                "regularization": args.action_reg,
                "seed": args.seed,
                "output_dir": str(out_dir),
            }
        )
    print(f"[done] plot={plot_path}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)
    print(f"[done] aggregate={json.dumps(agg)}", flush=True)


if __name__ == "__main__":
    main()
