#!/usr/bin/env python3
"""Aggregate latent-geometry diagnostics for LeWM PushT rollouts."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from eval_pusht_baseline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    PROJECT_ROOT,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from eval_pusht_sweep import analyze_offset, prepare_dataset_index
from three_cost_eval import expand_info_for_candidates, raw_to_blocked_normalized, tensor_clone_info


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "aggregate_latent_diagnostics.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LATENT_DIM = 192
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute temporal straightness, effective rank, covariance spectrum, "
            "and SIGReg-style normality diagnostics for real vs imagined PushT latents."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
    )
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Defaults to CPU so the analysis is sandbox-friendly.",
    )
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument(
        "--latent-steps",
        type=int,
        default=10,
        help="Number of predicted latent steps; each step consumes --action-block raw actions.",
    )
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--sigreg-directions", type=int, default=100)
    parser.add_argument(
        "--sigreg-frequencies",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated frequencies for the empirical-characteristic-function normality proxy.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_frequencies(raw: str) -> np.ndarray:
    values = [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one SIGReg frequency must be provided")
    return np.asarray(values, dtype=np.float64)


def make_policy_args(args: argparse.Namespace) -> argparse.Namespace:
    # These CEM fields are unused by this script, but build_policy constructs a solver.
    return argparse.Namespace(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        num_samples=1,
        var_scale=1.0,
        cem_iters=1,
        topk=1,
        seed=args.seed,
        horizon=args.latent_steps,
        receding_horizon=args.latent_steps,
        action_block=args.action_block,
        img_size=args.img_size,
    )


def sample_start_rows(dataset, *, num_trajectories: int, raw_steps: int, seed: int) -> tuple[np.ndarray, dict]:
    index = prepare_dataset_index(dataset)
    analysis = analyze_offset(index, raw_steps)
    valid_indices = analysis["valid_indices"]
    if len(valid_indices) < num_trajectories:
        raise ValueError(
            f"Requested {num_trajectories} trajectories, but only "
            f"{len(valid_indices)} valid start rows exist for raw_steps={raw_steps}."
        )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(valid_indices, size=num_trajectories, replace=False)
    return np.sort(sampled.astype(np.int64)), {
        "valid_start_points": int(len(valid_indices)),
        "total_rows": index.total_rows,
        "total_episodes": index.total_episodes,
        "mean_episode_length": float(np.mean(index.episode_lengths)),
        "median_episode_length": float(np.median(index.episode_lengths)),
        "min_episode_length": int(np.min(index.episode_lengths)),
        "max_episode_length": int(np.max(index.episode_lengths)),
    }


def collect_chunks(
    dataset,
    start_rows: np.ndarray,
    *,
    latent_steps: int,
    action_block: int,
    action_processor,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Collect block-time pixels and normalized action blocks.

    For 10 latent steps and action_block=5, pixels are taken at rows
    start + [0, 5, ..., 50], giving 11 latents including the initial point.
    Actions are the 50 raw env actions from start:start+50, reshaped into
    10 normalized model blocks of shape (10, 10).
    """
    raw_steps = latent_steps * action_block
    pixel_offsets = np.arange(latent_steps + 1, dtype=np.int64) * action_block
    action_col = dataset.get_col_data("action")
    episode_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = dataset.get_col_data(episode_col)
    steps = dataset.get_col_data("step_idx")

    pixels = []
    action_blocks = []
    records = []
    for traj_index, row in enumerate(start_rows):
        row = int(row)
        rows = row + pixel_offsets
        row_data = dataset.get_row_data(rows)
        raw_actions = np.asarray(action_col[row : row + raw_steps], dtype=np.float32)
        blocked = raw_to_blocked_normalized(
            raw_actions,
            action_processor=action_processor,
            action_block=action_block,
        )
        pixels.append(np.asarray(row_data["pixels"], dtype=np.uint8))
        action_blocks.append(blocked)
        records.append(
            {
                "trajectory_index": traj_index,
                "dataset_row": row,
                "episode_id": int(episodes[row]),
                "start_step": int(steps[row]),
                "end_step": int(steps[row] + raw_steps),
            }
        )
    return np.stack(pixels), np.stack(action_blocks), records


def encode_pixel_sequences(
    *,
    policy,
    model,
    pixel_sequences: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    encoded_batches = []
    for start in range(0, len(pixel_sequences), batch_size):
        batch = pixel_sequences[start : start + batch_size]
        info = policy._prepare_info({"pixels": batch})
        pixels = info["pixels"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            encoded = model.encode({"pixels": pixels})["emb"]
        encoded_batches.append(encoded.detach().cpu().numpy())
    return np.concatenate(encoded_batches, axis=0)


def rollout_imagined_latents(
    *,
    policy,
    model,
    initial_pixels: np.ndarray,
    action_blocks: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    imagined_batches = []
    for start in range(0, len(initial_pixels), batch_size):
        pixels_batch = initial_pixels[start : start + batch_size]
        actions_batch = action_blocks[start : start + batch_size]
        prepared = policy._prepare_info({"pixels": pixels_batch[:, None, ...]})
        expanded = expand_info_for_candidates(prepared, 1)
        for key, value in list(expanded.items()):
            if torch.is_tensor(value):
                expanded[key] = value.to(device=device, dtype=torch.float32)
        actions = torch.as_tensor(actions_batch[:, None, ...], dtype=torch.float32, device=device)
        with torch.no_grad():
            rolled = model.rollout(tensor_clone_info(expanded), actions)["predicted_emb"]
        imagined_batches.append(rolled[:, 0].detach().cpu().numpy())
    return np.concatenate(imagined_batches, axis=0)


def temporal_straightness(latents: np.ndarray) -> dict:
    velocities = np.diff(latents, axis=1)
    if velocities.shape[1] < 2:
        return {"mean": None, "std": None, "n": 0}
    left = velocities[:, :-1]
    right = velocities[:, 1:]
    denom = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    cos = np.sum(left * right, axis=-1) / np.maximum(denom, EPS)
    cos = cos[np.isfinite(cos)]
    return {
        "mean": float(np.mean(cos)) if len(cos) else None,
        "std": float(np.std(cos, ddof=1)) if len(cos) > 1 else 0.0,
        "n": int(len(cos)),
    }


def effective_rank(latents_matrix: np.ndarray) -> dict:
    matrix = np.asarray(latents_matrix, dtype=np.float64)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    total = float(np.sum(singular_values))
    if total <= EPS:
        return {"effective_rank": 0.0, "singular_values": singular_values.tolist()}
    probs = singular_values / total
    entropy = -float(np.sum(probs * np.log(probs + EPS)))
    return {
        "effective_rank": float(np.exp(entropy)),
        "entropy": entropy,
        "singular_values": singular_values.tolist(),
    }


def covariance_spectrum(latents_matrix: np.ndarray) -> dict:
    matrix = np.asarray(latents_matrix, dtype=np.float64)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    cov = matrix.T @ matrix / max(matrix.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    total = float(np.sum(eigvals))
    participation = (total * total / float(np.sum(eigvals * eigvals) + EPS)) if total > EPS else 0.0
    return {
        "eigenvalues": eigvals.tolist(),
        "trace": total,
        "top1_fraction": float(eigvals[0] / total) if total > EPS else 0.0,
        "top10_fraction": float(np.sum(eigvals[:10]) / total) if total > EPS else 0.0,
        "participation_rank": float(participation),
    }


def make_random_directions(dim: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, dim))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), EPS)
    return directions.astype(np.float64)


def sigreg_ecf_statistic(
    latents_matrix: np.ndarray,
    *,
    directions: np.ndarray,
    frequencies: np.ndarray,
) -> dict:
    """Empirical-characteristic-function normality proxy used as a SIGReg statistic.

    SIGReg encourages projected latent distributions to look isotropic Gaussian.
    For each random unit direction, we standardize the 1-D projection and compare
    its empirical characteristic function against the N(0,1) characteristic
    function exp(-t^2 / 2). Lower values are more Gaussian-like.
    """
    x = np.asarray(latents_matrix, dtype=np.float64)
    projections = x @ directions.T
    stats = []
    for col in range(projections.shape[1]):
        values = projections[:, col]
        std = float(np.std(values))
        if std <= EPS:
            stats.append(float("nan"))
            continue
        values = (values - np.mean(values)) / std
        discrepancy = 0.0
        for freq in frequencies:
            empirical = np.mean(np.exp(1j * freq * values))
            target = math.exp(-0.5 * freq * freq)
            discrepancy += float(np.abs(empirical - target) ** 2)
        stats.append(discrepancy / len(frequencies))
    stats_arr = np.asarray(stats, dtype=np.float64)
    stats_arr = stats_arr[np.isfinite(stats_arr)]
    return {
        "mean": float(np.mean(stats_arr)) if len(stats_arr) else None,
        "std": float(np.std(stats_arr, ddof=1)) if len(stats_arr) > 1 else 0.0,
        "n_directions": int(len(stats_arr)),
        "frequencies": frequencies.tolist(),
        "direction_statistics": stats_arr.tolist(),
        "interpretation": "lower is more Gaussian-like under random 1-D projections",
    }


def prediction_error(real: np.ndarray, imagined: np.ndarray) -> dict:
    diff = imagined[:, 1:] - real[:, 1:]
    l2 = np.linalg.norm(diff, axis=-1)
    mse = np.mean(diff * diff, axis=-1)
    by_step = []
    for idx in range(l2.shape[1]):
        by_step.append(
            {
                "step": idx + 1,
                "mean_l2": float(np.mean(l2[:, idx])),
                "std_l2": float(np.std(l2[:, idx], ddof=1)) if l2.shape[0] > 1 else 0.0,
                "mean_mse": float(np.mean(mse[:, idx])),
            }
        )
    return {
        "overall_mean_l2": float(np.mean(l2)),
        "overall_std_l2": float(np.std(l2, ddof=1)) if l2.size > 1 else 0.0,
        "overall_mean_mse": float(np.mean(mse)),
        "by_step": by_step,
    }


def horizon_metrics(
    real: np.ndarray,
    imagined: np.ndarray,
    *,
    directions: np.ndarray,
    frequencies: np.ndarray,
) -> list[dict]:
    rows = []
    steps = real.shape[1] - 1
    pred = prediction_error(real, imagined)["by_step"]
    for step in range(1, steps + 1):
        real_prefix = real[:, : step + 1]
        imagined_prefix = imagined[:, : step + 1]
        real_step = real[:, step]
        imagined_step = imagined[:, step]
        rows.append(
            {
                "step": step,
                "real_cumulative_straightness": temporal_straightness(real_prefix)["mean"],
                "imagined_cumulative_straightness": temporal_straightness(imagined_prefix)["mean"],
                "real_step_effective_rank": effective_rank(real_step)["effective_rank"],
                "imagined_step_effective_rank": effective_rank(imagined_step)["effective_rank"],
                "real_step_sigreg_ecf": sigreg_ecf_statistic(
                    real_step,
                    directions=directions,
                    frequencies=frequencies,
                )["mean"],
                "imagined_step_sigreg_ecf": sigreg_ecf_statistic(
                    imagined_step,
                    directions=directions,
                    frequencies=frequencies,
                )["mean"],
                "prediction_error_l2": pred[step - 1]["mean_l2"],
                "prediction_error_mse": pred[step - 1]["mean_mse"],
            }
        )
    return rows


def make_plots(results: dict, figures_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    real_eig = np.asarray(results["covariance_spectrum"]["real"]["eigenvalues"])
    imagined_eig = np.asarray(results["covariance_spectrum"]["imagined"]["eigenvalues"])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(np.arange(1, len(real_eig) + 1), real_eig + EPS, label="real encoder latents", linewidth=2)
    ax.plot(
        np.arange(1, len(imagined_eig) + 1),
        imagined_eig + EPS,
        label="imagined predictor latents",
        linewidth=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("covariance eigenvalue (log scale)")
    ax.set_title("Latent covariance spectrum")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figures_dir / "covariance_spectrum.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["covariance_spectrum"] = str(path)

    horizon = results["horizon_metrics"]
    steps = [row["step"] for row in horizon]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    ax = axes[0, 0]
    ax.plot(steps, [row["real_step_effective_rank"] for row in horizon], marker="o", label="real")
    ax.plot(steps, [row["imagined_step_effective_rank"] for row in horizon], marker="o", label="imagined")
    ax.set_title("Effective rank by step")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("effective rank")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(steps, [row["real_cumulative_straightness"] for row in horizon], marker="o", label="real")
    ax.plot(
        steps,
        [row["imagined_cumulative_straightness"] for row in horizon],
        marker="o",
        label="imagined",
    )
    ax.set_title("Cumulative temporal straightness")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("mean cosine(v_t, v_{t+1})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(steps, [row["prediction_error_l2"] for row in horizon], marker="o", color="#b23a48")
    ax.set_title("Prediction error")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("mean ||zhat_t - z_t||")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(steps, [row["real_step_sigreg_ecf"] for row in horizon], marker="o", label="real")
    ax.plot(steps, [row["imagined_step_sigreg_ecf"] for row in horizon], marker="o", label="imagined")
    ax.set_title("SIGReg ECF normality proxy")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("ECF discrepancy (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    path = figures_dir / "metrics_vs_horizon.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["metrics_vs_horizon"] = str(path)
    return paths


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


def build_report(results: dict) -> str:
    lines = []
    lines.append("Aggregate Latent Diagnostics")
    lines.append("=" * 30)
    lines.append(
        f"trajectories={results['metadata']['num_trajectories']} "
        f"latent_steps={results['metadata']['latent_steps']} "
        f"raw_steps={results['metadata']['raw_steps']} "
        f"device={results['metadata']['device']}"
    )
    lines.append("")
    lines.append("Temporal straightness:")
    lines.append(
        f"  real:     {fmt(results['temporal_straightness']['real']['mean'])} "
        f"+/- {fmt(results['temporal_straightness']['real']['std'])}"
    )
    lines.append(
        f"  imagined: {fmt(results['temporal_straightness']['imagined']['mean'])} "
        f"+/- {fmt(results['temporal_straightness']['imagined']['std'])}"
    )
    lines.append("")
    lines.append("Effective rank over all future steps:")
    lines.append(f"  real:     {fmt(results['effective_rank']['real']['effective_rank'], 2)}")
    lines.append(f"  imagined: {fmt(results['effective_rank']['imagined']['effective_rank'], 2)}")
    lines.append("")
    lines.append("Covariance spectrum:")
    lines.append(
        f"  real top1/top10:     "
        f"{fmt(results['covariance_spectrum']['real']['top1_fraction'])}/"
        f"{fmt(results['covariance_spectrum']['real']['top10_fraction'])}"
    )
    lines.append(
        f"  imagined top1/top10: "
        f"{fmt(results['covariance_spectrum']['imagined']['top1_fraction'])}/"
        f"{fmt(results['covariance_spectrum']['imagined']['top10_fraction'])}"
    )
    lines.append("")
    lines.append("SIGReg ECF normality proxy (lower is more Gaussian-like):")
    lines.append(f"  real:     {fmt(results['sigreg_ecf']['real']['mean'])}")
    lines.append(f"  imagined: {fmt(results['sigreg_ecf']['imagined']['mean'])}")
    lines.append("")
    pred = results["prediction_error"]
    lines.append("Prediction error:")
    lines.append(
        f"  mean L2 over future steps: {fmt(pred['overall_mean_l2'])} "
        f"+/- {fmt(pred['overall_std_l2'])}"
    )
    lines.append(f"  step-10 mean L2: {fmt(pred['by_step'][-1]['mean_l2'])}")
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"  JSON: {results['output_path']}")
    for path in results["figure_paths"].values():
        lines.append(f"  Figure: {path}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.time()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.results_path = args.results_path.expanduser().resolve()
    args.figures_dir = args.figures_dir.expanduser().resolve()
    args.device = resolve_device(args.device)
    frequencies = parse_frequencies(args.sigreg_frequencies)

    if args.latent_steps < 1:
        raise ValueError("--latent-steps must be positive")
    if args.action_block < 1:
        raise ValueError("--action-block must be positive")
    if args.num_trajectories < 2:
        raise ValueError("--num-trajectories must be at least 2")

    raw_steps = args.latent_steps * args.action_block
    print("== Aggregate latent diagnostics setup ==")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"results_path: {args.results_path}")
    print(f"figures_dir: {args.figures_dir}")
    print(f"device: {args.device}")
    print(f"num_trajectories: {args.num_trajectories}")
    print(f"latent_steps: {args.latent_steps}")
    print(f"raw_steps: {raw_steps}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action"])
    policy = build_policy(make_policy_args(args), process)
    model = policy.solver.model

    start_rows, dataset_stats = sample_start_rows(
        dataset,
        num_trajectories=args.num_trajectories,
        raw_steps=raw_steps,
        seed=args.seed,
    )
    pixel_sequences, action_blocks, records = collect_chunks(
        dataset,
        start_rows,
        latent_steps=args.latent_steps,
        action_block=args.action_block,
        action_processor=policy.process["action"],
    )
    print(f"collected pixels: {pixel_sequences.shape}")
    print(f"collected action blocks: {action_blocks.shape}")

    real_latents = encode_pixel_sequences(
        policy=policy,
        model=model,
        pixel_sequences=pixel_sequences,
        batch_size=args.encode_batch_size,
    )
    imagined_latents = rollout_imagined_latents(
        policy=policy,
        model=model,
        initial_pixels=pixel_sequences[:, 0],
        action_blocks=action_blocks,
        batch_size=args.encode_batch_size,
    )
    if real_latents.shape != imagined_latents.shape:
        raise RuntimeError(
            f"Real and imagined latent shapes differ: {real_latents.shape} vs {imagined_latents.shape}"
        )

    real_future = real_latents[:, 1:].reshape(-1, real_latents.shape[-1])
    imagined_future = imagined_latents[:, 1:].reshape(-1, imagined_latents.shape[-1])
    directions = make_random_directions(LATENT_DIM, args.sigreg_directions, args.seed + 991)

    results = {
        "output_path": str(args.results_path),
        "metadata": {
            "seed": args.seed,
            "device": args.device,
            "checkpoint_dir": str(args.checkpoint_dir),
            "cache_dir": str(args.cache_dir),
            "dataset_name": args.dataset_name,
            "num_trajectories": args.num_trajectories,
            "latent_steps": args.latent_steps,
            "action_block": args.action_block,
            "raw_steps": raw_steps,
            "latent_dim": int(real_latents.shape[-1]),
            "real_latent_shape": list(real_latents.shape),
            "imagined_latent_shape": list(imagined_latents.shape),
            "note": (
                "Latents include the initial block-time observation at index 0; "
                "aggregate metrics are reported over future rollout steps 1..latent_steps."
            ),
            "dataset_stats": dataset_stats,
            "elapsed_seconds": None,
        },
        "sampled_trajectories": records,
        "temporal_straightness": {
            "real": temporal_straightness(real_latents),
            "imagined": temporal_straightness(imagined_latents),
        },
        "effective_rank": {
            "real": effective_rank(real_future),
            "imagined": effective_rank(imagined_future),
        },
        "covariance_spectrum": {
            "real": covariance_spectrum(real_future),
            "imagined": covariance_spectrum(imagined_future),
        },
        "sigreg_ecf": {
            "real": sigreg_ecf_statistic(
                real_future,
                directions=directions,
                frequencies=frequencies,
            ),
            "imagined": sigreg_ecf_statistic(
                imagined_future,
                directions=directions,
                frequencies=frequencies,
            ),
        },
        "prediction_error": prediction_error(real_latents, imagined_latents),
    }
    results["horizon_metrics"] = horizon_metrics(
        real_latents,
        imagined_latents,
        directions=directions,
        frequencies=frequencies,
    )

    figure_paths = {}
    if not args.no_plots:
        figure_paths = make_plots(results, args.figures_dir)
    results["figure_paths"] = figure_paths
    results["metadata"]["elapsed_seconds"] = time.time() - started

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(to_jsonable(results), indent=2, allow_nan=False) + "\n")
    print()
    print(build_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
