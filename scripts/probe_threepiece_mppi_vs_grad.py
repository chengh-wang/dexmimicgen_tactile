#!/usr/bin/env python3
"""Probe whether MPPI samples recover gradient/QP residual directions.

The probe fixes states encountered by the base tactile flow policy, computes the
gradient/QP residual for each state, then compares MPPI residuals from a grid of
hyperparameters against that gradient residual without re-running full rollouts
for each setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from rollout_policy_with_high200_contrastive_setpoint_residual import (  # noqa: E402
    DEXMG_ROOT,
    POLICY_ROOT,
    RotationTransformer,
    ThreePieceWM,
    TrainingAgent,
    action20_to_action14,
    high_sample,
    load_normalizers,
    load_setpoint_params,
    make_config,
    make_dataset,
    make_env,
    normalize_obs,
    pad_future,
    setpoint_value_high200_batch,
    source_indices_for_rate,
    stack_last,
    step_with_high200_interp_minimal,
    obs_from_raw,
    optimize_residual_high200,
)


def parse_setting(text: str) -> dict[str, float | int | str]:
    # name,chunk,samples,trust,sigma,temp
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
        "h500_s64_tr003_sig003_t003,10,64,0.03,0.03,0.03",
        "h500_s256_tr003_sig003_t003,10,256,0.03,0.03,0.03",
        "h500_s512_tr003_sig003_t003,10,512,0.03,0.03,0.03",
        "h500_s256_tr005_sig003_t003,10,256,0.05,0.03,0.03",
        "h500_s256_tr005_sig005_t003,10,256,0.05,0.05,0.03",
        "h1000_s64_tr003_sig003_t003,20,64,0.03,0.03,0.03",
        "h1000_s256_tr003_sig003_t003,20,256,0.03,0.03,0.03",
        "h1000_s512_tr003_sig003_t003,20,512,0.03,0.03,0.03",
        "h1000_s256_tr005_sig003_t003,20,256,0.05,0.03,0.03",
        "h1000_s256_tr005_sig005_t003,20,256,0.05,0.05,0.03",
    ]
    return [parse_setting(x) for x in raw]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def sign_agreement(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    mask = np.abs(a) > eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))


@torch.no_grad()
def mppi_probe(
    wm: ThreePieceWM,
    params: dict[str, torch.Tensor],
    tactile_hist_np: np.ndarray,
    joint_hist_np: np.ndarray,
    action_hist_np: np.ndarray,
    future_low_actions_np: np.ndarray,
    normalizers: dict[str, np.ndarray],
    device: torch.device,
    rate_hz: int,
    progress: float,
    chunk: int,
    samples: int,
    trust: float,
    sigma: float,
    temp: float,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray | float | int]:
    torch.manual_seed(int(seed))
    samples = max(int(samples), 2)
    batch_size = max(int(batch_size), 1)
    base_low = torch.from_numpy(pad_future(future_low_actions_np, chunk)).to(device=device, dtype=torch.float32)
    eps = torch.randn((samples, chunk, 14), device=device, dtype=torch.float32) * float(sigma)
    eps = torch.clamp(eps, -float(trust), float(trust))
    eps[0].zero_()
    candidates = base_low[None] + eps

    values = []
    betas = []
    for start in range(0, samples, batch_size):
        value_b, beta_b = setpoint_value_high200_batch(
            wm,
            params,
            tactile_hist_np,
            joint_hist_np,
            action_hist_np,
            candidates[start : start + batch_size],
            normalizers,
            device,
            rate_hz,
            progress,
        )
        values.append(value_b.detach())
        betas.append(beta_b.detach())
    value = torch.cat(values, dim=0).float()
    beta = torch.cat(betas, dim=0).float()
    score = value
    weights = torch.softmax((score - torch.max(score)) / max(float(temp), 1e-6), dim=0)
    weighted_delta = torch.sum(weights[:, None, None] * eps, dim=0)
    weighted_delta = torch.clamp(weighted_delta, -float(trust), float(trust)).detach()
    best_idx = int(torch.argmax(score).detach().cpu())
    best_delta = eps[best_idx].detach()
    return {
        "weighted_delta": weighted_delta.cpu().numpy().astype(np.float32),
        "best_delta": best_delta.cpu().numpy().astype(np.float32),
        "value_base": float(value[0].detach().cpu()),
        "value_weighted": float(torch.sum(weights * value).detach().cpu()),
        "value_best": float(value[best_idx].detach().cpu()),
        "beta_base": float(beta[0].detach().cpu()),
        "beta_best": float(beta[best_idx].detach().cpu()),
        "score_max": float(torch.max(score).detach().cpu()),
        "score_mean": float(torch.mean(score).detach().cpu()),
        "weight_max": float(torch.max(weights).detach().cpu()),
        "weight_entropy": float((-torch.sum(weights * torch.log(torch.clamp(weights, min=1e-12)))).detach().cpu()),
        "best_index": best_idx,
    }


def make_probe_config(args: argparse.Namespace):
    class ConfigArgs:
        pass

    cfg_args = ConfigArgs()
    cfg_args.task_config = args.task_config
    cfg_args.model_path = args.model_path
    cfg_args.dataset_path = args.dataset_path
    cfg_args.max_episode_steps = args.max_episode_steps
    cfg_args.episodes = args.episodes
    return make_config(cfg_args)


def run_probe(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    settings = [parse_setting(x) for x in args.setting] if args.setting else default_settings()
    max_chunk = max(int(s["chunk"]) for s in settings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = make_probe_config(args)
    if args.device is not None:
        config.optimization.device = args.device
    config.task.max_episode_steps = args.max_episode_steps
    device = torch.device(config.optimization.device if torch.cuda.is_available() else "cpu")
    sample_indices = source_indices_for_rate(args.wm_rate_hz)

    print(f"[setup] out={out_dir}", flush=True)
    print(f"[setup] wm_rate_hz={args.wm_rate_hz} source_indices={sample_indices.tolist()}", flush=True)
    print(f"[setup] settings={len(settings)} max_chunk={max_chunk}", flush=True)

    env = make_env(args.env_dataset_path, enable_tactile=True)
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
    rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

    rows: list[dict] = []
    state_rows: list[dict] = []
    state_id = 0
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        torch.manual_seed(args.seed + ep)
        raw_obs = env.reset()
        obs_hist = deque(maxlen=config.task.obs_steps)
        first_obs = obs_from_raw(raw_obs, config, dataset)
        for _ in range(config.task.obs_steps):
            obs_hist.append(first_obs)

        init_high = high_sample(env, np.zeros(14, dtype=np.float32))
        tactile_hist = deque([init_high["tactile"]] * args.history, maxlen=args.history)
        joint_hist = deque([init_high["robot_joint_pos"]] * args.history, maxlen=args.history)
        action_hist = deque([init_high["action"]] * args.history, maxlen=args.history)
        low_steps = 0
        while low_steps < config.task.max_episode_steps and state_id < args.probe_states:
            obs_seq = stack_last(obs_hist, config.task.obs_steps)
            obs = normalize_obs(obs_seq, dataset, config.optimization.device)
            act_0 = torch.randn((1, config.task.horizon, config.task.act_dim), device=device, dtype=torch.float32)
            with torch.no_grad():
                act_normed = agent.sample(act_0=act_0, obs=obs, num_steps=args.nfe, use_ema=True)
            act20 = dataset.normalizer["action"].unnormalize(act_normed.detach().cpu().numpy())[0]
            start = config.task.obs_steps - 1
            end = start + config.task.act_steps
            act14_seq = action20_to_action14(act20[start:end], rotation_transformer)

            for j, base_act14 in enumerate(act14_seq):
                if low_steps >= config.task.max_episode_steps or state_id >= args.probe_states:
                    break
                future_low_max = act14_seq[j : j + max_chunk].astype(np.float32)
                next_base = act14_seq[j + 1].astype(np.float32) if j + 1 < len(act14_seq) else base_act14.astype(np.float32)
                if low_steps % args.state_stride == 0:
                    tactile_hist_np = np.stack(list(tactile_hist)[-args.history:], axis=0).astype(np.float32)
                    joint_hist_np = np.stack(list(joint_hist)[-args.history:], axis=0).astype(np.float32)
                    action_hist_np = np.stack(list(action_hist)[-args.history:], axis=0).astype(np.float32)
                    progress = min(1.0, max(0.0, float(low_steps + max_chunk) / max(float(config.task.max_episode_steps), 1.0)))
                    state_rows.append({"state_id": state_id, "episode": ep, "low_step": low_steps, "progress": progress})
                    for si, setting in enumerate(settings):
                        chunk = int(setting["chunk"])
                        future_low = future_low_max[:chunk]
                        grad = optimize_residual_high200(
                            wm,
                            params,
                            tactile_hist,
                            joint_hist,
                            action_hist,
                            future_low,
                            normalizers,
                            args.history,
                            chunk,
                            args.grad_trust_delta,
                            args.qp_lambda_u,
                            device,
                            args.wm_rate_hz,
                            progress,
                        )
                        grad_delta = np.asarray(grad["delta_low"], dtype=np.float32)
                        for rep in range(args.repeats):
                            probe = mppi_probe(
                                wm,
                                params,
                                tactile_hist_np,
                                joint_hist_np,
                                action_hist_np,
                                future_low,
                                normalizers,
                                device,
                                args.wm_rate_hz,
                                progress,
                                chunk,
                                int(setting["samples"]),
                                float(setting["trust"]),
                                float(setting["sigma"]),
                                float(setting["temp"]),
                                args.seed + 100000 * state_id + 1000 * si + rep,
                                args.mppi_batch_size,
                            )
                            weighted = np.asarray(probe["weighted_delta"], dtype=np.float32)
                            best = np.asarray(probe["best_delta"], dtype=np.float32)
                            rows.append(
                                {
                                    "state_id": state_id,
                                    "episode": ep,
                                    "low_step": low_steps,
                                    "setting": setting["name"],
                                    "repeat": rep,
                                    "chunk": chunk,
                                    "samples": int(setting["samples"]),
                                    "trust": float(setting["trust"]),
                                    "sigma": float(setting["sigma"]),
                                    "temp": float(setting["temp"]),
                                    "grad_value_gain": float(grad["value_residual"] - grad["value_base"]),
                                    "mppi_weighted_value_gain": float(probe["value_weighted"] - probe["value_base"]),
                                    "mppi_best_value_gain": float(probe["value_best"] - probe["value_base"]),
                                    "weighted_first_cos": cosine(grad_delta[0], weighted[0]),
                                    "weighted_chunk_cos": cosine(grad_delta, weighted),
                                    "best_first_cos": cosine(grad_delta[0], best[0]),
                                    "best_chunk_cos": cosine(grad_delta, best),
                                    "weighted_first_l2": float(np.linalg.norm(grad_delta[0] - weighted[0])),
                                    "best_first_l2": float(np.linalg.norm(grad_delta[0] - best[0])),
                                    "grad_first_norm": float(np.linalg.norm(grad_delta[0])),
                                    "weighted_first_norm": float(np.linalg.norm(weighted[0])),
                                    "best_first_norm": float(np.linalg.norm(best[0])),
                                    "weighted_first_norm_ratio": float(np.linalg.norm(weighted[0]) / max(np.linalg.norm(grad_delta[0]), 1e-12)),
                                    "best_first_norm_ratio": float(np.linalg.norm(best[0]) / max(np.linalg.norm(grad_delta[0]), 1e-12)),
                                    "weighted_sign_agree": sign_agreement(grad_delta[0], weighted[0]),
                                    "best_sign_agree": sign_agreement(grad_delta[0], best[0]),
                                    "mppi_weight_max": float(probe["weight_max"]),
                                    "mppi_weight_entropy": float(probe["weight_entropy"]),
                                    "mppi_score_max": float(probe["score_max"]),
                                    "mppi_score_mean": float(probe["score_mean"]),
                                }
                            )
                    print(f"[probe] state={state_id} ep={ep} step={low_steps}", flush=True)
                    state_id += 1

                raw_obs, reward, done, _, samples = step_with_high200_interp_minimal(
                    env,
                    base_act14.astype(np.float32),
                    next_base,
                    low_steps,
                )
                obs_hist.append(obs_from_raw(raw_obs, config, dataset))
                for sample_i in sample_indices:
                    s = samples[int(sample_i)]
                    tactile_hist.append(s["tactile"])
                    joint_hist.append(s["robot_joint_pos"])
                    action_hist.append(s["action"])
                low_steps += 1
                if done or bool(env._check_success()) or float(reward) > 0.0:
                    break
            if done or bool(env._check_success()) or float(reward) > 0.0:
                break
    return rows, state_rows


def write_outputs(rows: list[dict], state_rows: list[dict], out_dir: Path) -> None:
    detail_path = out_dir / "probe_rows.csv"
    fields = list(rows[0].keys()) if rows else []
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "states.json").write_text(json.dumps(state_rows, indent=2))

    by_setting: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_setting[str(r["setting"])].append(r)

    summary = []
    for name, vals in sorted(by_setting.items()):
        def mean(key: str) -> float:
            xs = [float(v[key]) for v in vals if np.isfinite(float(v[key]))]
            return float(np.mean(xs)) if xs else float("nan")

        summary.append(
            {
                "setting": name,
                "n": len(vals),
                "chunk": int(vals[0]["chunk"]),
                "samples": int(vals[0]["samples"]),
                "trust": float(vals[0]["trust"]),
                "sigma": float(vals[0]["sigma"]),
                "temp": float(vals[0]["temp"]),
                "weighted_first_cos": mean("weighted_first_cos"),
                "weighted_chunk_cos": mean("weighted_chunk_cos"),
                "best_first_cos": mean("best_first_cos"),
                "best_chunk_cos": mean("best_chunk_cos"),
                "weighted_first_l2": mean("weighted_first_l2"),
                "best_first_l2": mean("best_first_l2"),
                "grad_first_norm": mean("grad_first_norm"),
                "weighted_first_norm": mean("weighted_first_norm"),
                "best_first_norm": mean("best_first_norm"),
                "weighted_first_norm_ratio": mean("weighted_first_norm_ratio"),
                "best_first_norm_ratio": mean("best_first_norm_ratio"),
                "weighted_sign_agree": mean("weighted_sign_agree"),
                "best_sign_agree": mean("best_sign_agree"),
                "grad_value_gain": mean("grad_value_gain"),
                "mppi_weighted_value_gain": mean("mppi_weighted_value_gain"),
                "mppi_best_value_gain": mean("mppi_best_value_gain"),
                "mppi_weight_max": mean("mppi_weight_max"),
                "mppi_weight_entropy": mean("mppi_weight_entropy"),
            }
        )
    summary.sort(key=lambda x: (-(x["weighted_first_cos"] if np.isfinite(x["weighted_first_cos"]) else -999), x["setting"]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "| setting | n | chunk | samples | trust | sigma | temp | weighted cos0 | best cos0 | weighted norm/grad | best norm/grad | grad gain | weighted gain | best gain | w_max | entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        md.append(
            f"| {s['setting']} | {s['n']} | {s['chunk']} | {s['samples']} | "
            f"{s['trust']:.3f} | {s['sigma']:.3f} | {s['temp']:.3f} | "
            f"{s['weighted_first_cos']:.3f} | {s['best_first_cos']:.3f} | "
            f"{s['weighted_first_norm_ratio']:.2f} | {s['best_first_norm_ratio']:.2f} | "
            f"{s['grad_value_gain']:.4f} | {s['mppi_weighted_value_gain']:.4f} | {s['mppi_best_value_gain']:.4f} | "
            f"{s['mppi_weight_max']:.3f} | {s['mppi_weight_entropy']:.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=str(DEXMG_ROOT / "datasets/generated_tactile_actionrollout_virtual_s8fs25_shards/20260726_122743_s8fs25/two_arm_three_piece_assembly"))
    parser.add_argument("--env-dataset-path", default=str(DEXMG_ROOT / "datasets/generated/two_arm_three_piece_assembly.hdf5"))
    parser.add_argument("--task-config", default="dexmg_three_piece_image_tactile_cnn_virtual_s8fs25")
    parser.add_argument("--model-path", default=str(POLICY_ROOT / "logs_dexmg_three_piece_tactile_cnn_virtual_s8fs25_flow_100k_20260726/models/model_step70000.pt"))
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--normalizers", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--wm-rate-hz", type=int, default=20)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--probe-states", type=int, default=12)
    parser.add_argument("--state-stride", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mppi-batch-size", type=int, default=256)
    parser.add_argument("--grad-trust-delta", type=float, default=0.01)
    parser.add_argument("--qp-lambda-u", type=float, default=20.0)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        help="Setting as name,chunk,samples,trust,sigma,temp. May be repeated. Defaults to a 10-setting sample-efficiency grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows, states = run_probe(args)
    if not rows:
        raise RuntimeError("no probe rows collected")
    write_outputs(rows, states, out_dir)
    print(f"[done] rows={len(rows)} states={len(states)} out={out_dir}", flush=True)
    print(f"[done] summary={out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
