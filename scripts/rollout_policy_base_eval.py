#!/usr/bin/env python3
"""No-residual ThreePiece tactile policy rollout eval."""

from __future__ import annotations

import argparse
import csv
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
    make_config,
    make_dataset,
    make_env,
    normalize_obs,
    obs_from_raw,
    stack_last,
)


def collect_one_base_rollout(env, config, dataset, agent, nfe: int, episode_idx: int, args) -> dict[str, float | int | bool]:
    device = torch.device(config.optimization.device if torch.cuda.is_available() else "cpu")
    rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    np.random.seed(args.seed + episode_idx)
    torch.manual_seed(args.seed + episode_idx)

    raw_obs = env.reset()
    obs_hist = deque(maxlen=config.task.obs_steps)
    first_obs = obs_from_raw(raw_obs, config, dataset)
    for _ in range(config.task.obs_steps):
        obs_hist.append(first_obs)

    total_reward = 0.0
    success = False
    low_steps = 0
    action_absmax = 0.0
    action_min = float("inf")
    action_max = float("-inf")

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

        for act14 in act14_seq:
            act14 = act14.astype(np.float32)
            action_min = min(action_min, float(np.min(act14)))
            action_max = max(action_max, float(np.max(act14)))
            action_absmax = max(action_absmax, float(np.max(np.abs(act14))))
            raw_obs, reward, done, _ = env.step(act14)
            total_reward += float(reward)
            success = success or bool(env._check_success()) or float(reward) > 0.0
            obs_hist.append(obs_from_raw(raw_obs, config, dataset))
            low_steps += 1
            if done or success or low_steps >= config.task.max_episode_steps:
                break
        if success or low_steps >= config.task.max_episode_steps:
            break

    return {
        "episode": int(episode_idx),
        "success": bool(success),
        "reward_sum": float(total_reward),
        "steps": int(low_steps),
        "action_min": float(action_min),
        "action_max": float(action_max),
        "action_absmax": float(action_absmax),
    }


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=5700)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    print(f"[setup] policy_model={args.model_path}", flush=True)
    print(f"[setup] no residual controller; raw policy actions only", flush=True)
    print(f"[setup] nfe={args.nfe} episodes={args.episodes} seed={args.seed}", flush=True)

    env = make_env(args.env_dataset_path, enable_tactile=True)
    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(args.model_path, load_optimizer=False)
    agent.eval()

    episodes = []
    for ep in range(args.episodes):
        data = collect_one_base_rollout(env, config, dataset, agent, args.nfe, ep, args)
        episodes.append(data)
        if args.progress_every > 0 and (ep % args.progress_every == 0 or ep == args.episodes - 1):
            print(f"[rollout] {json.dumps(data)}", flush=True)

    success_count = int(sum(bool(ep["success"]) for ep in episodes))
    summary = {
        "method": "base_policy_no_residual",
        "policy_model": args.model_path,
        "nfe": args.nfe,
        "episodes_n": args.episodes,
        "success_count": success_count,
        "success_rate": success_count / max(args.episodes, 1),
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "episodes": episodes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "episodes.csv").open("w", newline="") as f:
        fields = ["episode", "success", "reward_sum", "steps", "action_min", "action_max", "action_absmax"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)
    print(f"[done] success={success_count}/{args.episodes}", flush=True)
    print(f"[done] summary={out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
