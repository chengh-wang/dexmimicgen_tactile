#!/usr/bin/env python3
"""Roll out the tactile policy and plot 20 Hz WM/Q-head scores."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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

from rollout_policy_with_q_video import (  # noqa: E402
    TrainingAgent,
    collect_one_rollout,
    make_config,
    make_dataset,
    make_env,
)
from train_threepiece_20hz_q_head import QHead  # noqa: E402
from train_threepiece_20hz_wm import ThreePieceWM  # noqa: E402


def pad_slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    n = arr.shape[0]
    idx = np.arange(start, start + length)
    idx = np.clip(idx, 0, max(n - 1, 0))
    return arr[idx]


@torch.no_grad()
def score_q_values_20hz(
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
    if joint_low.shape[-1] != 14:
        raise RuntimeError(f"expected 14-D robot_joint_pos, got {joint_low.shape}")

    q_values: list[float] = []
    window = history + chunk
    for low_start in range(0, n_low, batch_size):
        lows = list(range(low_start, min(low_start + batch_size, n_low)))
        tactile_batch = []
        joint_batch = []
        action_batch = []
        for low_idx in lows:
            start = low_idx - history + 1
            tactile_batch.append(pad_slice(tactile_low, start, history))
            joint_batch.append(pad_slice(joint_low, start, history))
            action_batch.append(pad_slice(action_low, start, window))

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


def save_q_plot(episodes: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=150)
    for ep in episodes:
        q = np.asarray(ep["q_values"], dtype=np.float32)
        x = np.arange(q.shape[0], dtype=np.float32) / 20.0
        label = (
            f"ep{ep['episode']} success={int(ep['success'])} "
            f"steps={ep['steps']} q_last={ep['q_last']:.3f}"
        )
        ax.plot(x, q, linewidth=1.8, label=label)
        if q.size:
            ax.scatter([x[-1]], [q[-1]], s=22)
    ax.set_title("20 Hz Q-head score on policy rollouts")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Q score = sigmoid(logit)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


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
            / "logs_dexmg_three_piece_tactile_cnn_virtual_s8fs25_flow_100k_20260726/models/model_step50000.pt"
        ),
    )
    parser.add_argument(
        "--q-ckpt",
        default=str(DEXMG_ROOT / "outputs/threepiece_20hz_q_head_vae8k_wm_best_chunk20_stride10_10k_20260727/q_best.pt"),
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
    parser.add_argument("--out-dir", default=str(DEXMG_ROOT / "outputs/qhead_policy_rollouts/flow50k_tactile_20hz_q_plot_5rollouts_20260727"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--video-height", type=int, default=384)
    parser.add_argument("--video-width", type=int, default=384)
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
    print(f"[setup] q_ckpt={args.q_ckpt}", flush=True)
    print("[setup] q scoring: 20Hz tactile/joint last-pool, history=4, chunk=20", flush=True)

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
    if int(q_cfg.get("history", args.history)) != args.history or int(q_cfg.get("chunk", args.chunk)) != args.chunk:
        raise RuntimeError(f"Q checkpoint history/chunk mismatch: {q_cfg}")
    q_head.load_state_dict(q_ckpt["q_head"], strict=True)
    q_head.eval()
    for p in q_head.parameters():
        p.requires_grad = False

    norms_npz = np.load(args.normalizers)
    normalizers = {k: norms_npz[k].astype(np.float32) for k in norms_npz.files}

    episodes = []
    for ep in range(args.episodes):
        print(f"[rollout] episode={ep} nfe={args.nfe}", flush=True)
        data = collect_one_rollout(env, config, dataset, agent, args.nfe, ep, args)
        q_values = score_q_values_20hz(
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
            "q_values": q_values.astype(float).tolist(),
        }
        episodes.append(ep_meta)
        print(f"[rollout] {json.dumps({k: v for k, v in ep_meta.items() if k != 'q_values'})}", flush=True)

    plot_path = out_dir / "q_curves.png"
    save_q_plot(episodes, plot_path)
    summary = {
        "plot": str(plot_path),
        "policy_model": args.model_path,
        "wm_ckpt": args.wm_ckpt,
        "q_ckpt": args.q_ckpt,
        "normalizers": args.normalizers,
        "nfe": args.nfe,
        "q_timebase": "one score per 20Hz low-control step; tactile/joint pooled with last high200 sample",
        "episodes": episodes,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] plot={plot_path}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
