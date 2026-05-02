#!/usr/bin/env python3
"""Evaluate P2-0 cost heads inside the LeWM CEM planning loop.

The metric-level P2-0 experiments rank stored terminal latents. This script
tests the stricter question: when CEM asks the frozen LeWM model to score action
candidates, does replacing the Euclidean terminal latent cost with a trained
cost head improve PushT rollout success?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import (  # noqa: E402
    block_pose_metrics,
    configure_goal_render,
)
from scripts.phase2.cost_head_model import LATENT_DIM, make_cost_head  # noqa: E402
from scripts.phase2.mahalanobis_baseline import load_mahalanobis_checkpoint  # noqa: E402
from scripts.phase2.splits import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    SPLIT3_TEST_PAIR_IDS,
    cell_by_pair_id,
    load_track_a_pairs,
    split1_random_holdout,
    split2_leave_one_cell_out,
    split3_hard_pair_holdout,
)


NUM_SAMPLES = 300
CEM_ITERS = 30
TOPK = 30
VAR_SCALE = 1.0
PLANNING_HORIZON = 5
RECEDING_HORIZON = 5
ACTION_BLOCK = 5
IMG_SIZE = 224
OFFSET = 50
EVAL_BUDGET = 100


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2" / "p2_0"
DEFAULT_RAW_MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "lewm-pusht"
DEFAULT_DATASET_NAME = "pusht_expert_train"


V1_CEM_LATE_SUCCESS_BY_CELL = {
    "D0xR0": 1.000,
    "D0xR1": 0.842,
    "D0xR2": 1.000,
    "D0xR3": 0.667,
    "D1xR0": 0.333,
    "D1xR1": 1.000,
    "D1xR2": 0.833,
    "D1xR3": 0.508,
    "D2xR0": 0.550,
    "D2xR1": 1.000,
    "D2xR2": 0.564,
    "D2xR3": 0.714,
    "D3xR0": 0.992,
    "D3xR1": 0.571,
    "D3xR2": 0.667,
    "D3xR3": 0.993,
}

LATENT_CEM_LATE_SUCCESS_BY_CELL = {
    "D0xR0": 1.000,
    "D0xR1": 0.917,
    "D0xR2": 0.667,
    "D0xR3": 0.167,
    "D1xR0": 0.508,
    "D1xR1": 0.167,
    "D1xR2": 0.500,
    "D1xR3": 0.000,
    "D2xR0": 0.408,
    "D2xR1": 0.467,
    "D2xR2": 0.021,
    "D2xR3": 0.007,
    "D3xR0": 0.167,
    "D3xR1": 0.371,
    "D3xR2": 0.092,
    "D3xR3": 0.071,
}


@dataclass(frozen=True)
class PairRows:
    """Initial and goal rows for one Track A pair."""

    initial: dict[str, Any]
    goal: dict[str, Any]


@dataclass(frozen=True)
class SingleEnvAdapter:
    """Minimal batched-env facade expected by WorldModelPolicy/CEMSolver."""

    action_space: gym.spaces.Box
    num_envs: int = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--model-type",
        choices=("mlp", "mahalanobis"),
        default="mlp",
        help="Cost model type to load for the non-Euclidean planner.",
    )
    parser.add_argument("--variant", choices=("small", "large"), default="small")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing the trained cost-head *_best.pt checkpoint.",
    )
    parser.add_argument(
        "--pairs-subset",
        default=None,
        help=(
            "Optional comma-separated pair IDs. When provided, these explicit "
            "pairs override the split test set for smoke tests."
        ),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--model-checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Converted stable-worldmodel LeWM checkpoint directory.",
    )
    parser.add_argument(
        "--raw-model-checkpoint-dir",
        type=Path,
        default=DEFAULT_RAW_MODEL_CHECKPOINT_DIR,
        help="Raw LeWM checkpoint directory retained for provenance.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    """Convert nested objects to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_pair_ids(raw: str) -> list[int]:
    """Parse a comma-separated pair-ID list."""
    pair_ids = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pair_id = int(chunk)
        if pair_id < 0:
            raise argparse.ArgumentTypeError("Pair IDs must be nonnegative")
        pair_ids.append(pair_id)
    if not pair_ids:
        raise argparse.ArgumentTypeError("--pairs-subset must include at least one ID")
    return list(dict.fromkeys(pair_ids))


def split_test_pair_ids(split: int, *, seed: int, pairs_path: Path) -> list[int]:
    """Return the test pair IDs for the requested split."""
    if split == 1:
        return split1_random_holdout(pairs_path, seed=seed)["test_pair_ids"]
    if split == 3:
        return split3_hard_pair_holdout(pairs_path, seed=seed)["test_pair_ids"]
    folds = split2_leave_one_cell_out(pairs_path, seed=seed)
    test_ids = []
    for fold in folds.values():
        test_ids.extend(fold["test_pair_ids"])
    return sorted(test_ids)


def load_pairs_by_id(path: Path) -> dict[int, dict]:
    """Load Track A pair metadata keyed by integer pair ID."""
    return {int(pair["pair_id"]): pair for pair in load_track_a_pairs(path)}


def make_policy_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the Namespace consumed by the existing Phase 1 policy helper."""
    return argparse.Namespace(
        checkpoint_dir=args.model_checkpoint_dir,
        device=args.device,
        num_samples=NUM_SAMPLES,
        var_scale=VAR_SCALE,
        cem_iters=CEM_ITERS,
        topk=TOPK,
        seed=args.seed,
        horizon=PLANNING_HORIZON,
        receding_horizon=RECEDING_HORIZON,
        action_block=ACTION_BLOCK,
        img_size=IMG_SIZE,
    )


def batched_action_adapter(env) -> SingleEnvAdapter:
    """Return a one-env batched action-space adapter for WorldModelPolicy."""
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"Expected Box action space, got {type(action_space)}")
    low = np.asarray(action_space.low, dtype=np.float32)[None, ...]
    high = np.asarray(action_space.high, dtype=np.float32)[None, ...]
    batched_space = gym.spaces.Box(
        low=low,
        high=high,
        shape=(1, *action_space.shape),
        dtype=np.float32,
    )
    return SingleEnvAdapter(action_space=batched_space)


def reset_policy_state(policy) -> None:
    """Clear receding-horizon warm-start state before a new episode."""
    if getattr(policy, "_action_buffer", None) is not None:
        policy._action_buffer.clear()
    if hasattr(policy, "action_buffer"):
        policy.action_buffer.clear()
    policy._next_init = None


def load_pair_rows(dataset, pair: dict) -> PairRows:
    """Load initial and goal rows using the exact Track A row indices."""
    start_row = int(pair["start_row"])
    goal_row = int(pair["goal_row"])
    rows = dataset.get_row_data([start_row, goal_row])
    return PairRows(
        initial={key: value[0] for key, value in rows.items()},
        goal={key: value[1] for key, value in rows.items()},
    )


def find_cost_head_checkpoint(
    checkpoint_dir: Path,
    *,
    split: int,
    variant: str,
) -> Path:
    """Find the trained cost-head checkpoint inside an output directory."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if checkpoint_dir.is_file():
        return checkpoint_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Cost-head checkpoint directory not found: {checkpoint_dir}")
    candidates = sorted(checkpoint_dir.glob(f"split{split}_*_{variant}_seed*_best.pt"))
    if not candidates:
        candidates = sorted(checkpoint_dir.glob(f"*_{variant}_seed*_best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No trained {variant!r} cost-head *_best.pt file found in {checkpoint_dir}"
        )
    if len(candidates) > 1 and split != 2:
        raise ValueError(
            "Expected one checkpoint for split "
            f"{split}, found {len(candidates)}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def load_cost_head(path: Path, *, variant: str, device: str):
    """Load a trained cost head from a Phase 2 checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_variant = checkpoint.get("variant")
    if saved_variant is not None and saved_variant != variant:
        raise ValueError(
            f"Checkpoint variant {saved_variant!r} does not match requested {variant!r}"
        )
    state_dict = checkpoint["model_state_dict"]
    checkpoint_args = checkpoint.get("args", {})
    use_temperature = bool(
        checkpoint.get("temperature")
        or checkpoint_args.get("temperature")
        or "log_temperature" in state_dict
    )
    temperature_init = float(checkpoint_args.get("temperature_init", 10.0))
    model = make_cost_head(
        variant,
        temperature=use_temperature,
        temperature_init=temperature_init,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model


def load_planning_cost_model(
    *,
    checkpoint_dir: Path,
    split: int,
    variant: str,
    model_type: str,
    device: str,
) -> tuple[Path, Any, str]:
    """Load the requested cost model for CEM-time planning evaluation."""
    if model_type == "mlp":
        checkpoint_path = find_cost_head_checkpoint(
            checkpoint_dir,
            split=split,
            variant=variant,
        )
        return checkpoint_path, load_cost_head(checkpoint_path, variant=variant, device=device), variant

    checkpoint_path, model, checkpoint = load_mahalanobis_checkpoint(
        checkpoint_dir,
        device=device,
    )
    saved_type = checkpoint.get("model_type")
    if saved_type is not None and saved_type != "mahalanobis":
        raise ValueError(
            f"Checkpoint model_type {saved_type!r} does not match requested 'mahalanobis'"
        )
    saved_variant = str(checkpoint.get("variant", "unknown"))
    return checkpoint_path, model, saved_variant


def make_cost_head_criterion(cost_head):
    """Return a LeWM criterion function backed by a trained cost head."""

    def criterion(info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict["predicted_emb"][..., -1, :]
        goal_emb = info_dict["goal_emb"]
        if goal_emb.ndim == 4:
            goal_emb = goal_emb[..., -1, :]
        elif goal_emb.ndim == 3:
            goal_emb = goal_emb[:, -1, :]
        elif goal_emb.ndim != 2:
            raise ValueError(f"Unexpected goal_emb shape: {tuple(goal_emb.shape)}")

        while goal_emb.ndim < pred_emb.ndim:
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb.expand_as(pred_emb)

        if pred_emb.shape[-1] != LATENT_DIM or goal_emb.shape[-1] != LATENT_DIM:
            raise ValueError(
                "Unexpected latent dimension in C_psi criterion: "
                f"pred={tuple(pred_emb.shape)}, goal={tuple(goal_emb.shape)}"
            )

        flat_pred = pred_emb.reshape(-1, pred_emb.shape[-1])
        flat_goal = goal_emb.reshape(-1, goal_emb.shape[-1])
        costs = cost_head(flat_pred, flat_goal)
        return costs.reshape(pred_emb.shape[:-1])

    return criterion


def build_planning_policy(
    *,
    policy_args: argparse.Namespace,
    process: dict,
    env_adapter: SingleEnvAdapter,
    cost_head=None,
):
    """Build a LeWM policy and optionally patch its model criterion."""
    policy = build_policy(policy_args, process)
    if cost_head is not None:
        policy.solver.model.criterion = make_cost_head_criterion(cost_head)
    policy.set_env(env_adapter)
    return policy


def rollout_pair(
    *,
    env,
    policy,
    pair_rows: PairRows,
    seed: int,
) -> dict:
    """Run one 100-step receding-horizon CEM rollout for one pair."""
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    initial_state = np.asarray(pair_rows.initial["state"], dtype=np.float32)
    goal_state = np.asarray(pair_rows.goal["state"], dtype=np.float32)
    goal_pixels = np.asarray(pair_rows.goal["pixels"], dtype=np.uint8)
    configure_goal_render(env_unwrapped, goal_state)
    env_unwrapped._set_state(initial_state)
    reset_policy_state(policy)

    env_success = False
    started = time.time()
    for _ in range(EVAL_BUDGET):
        pixels = np.asarray(env_unwrapped.render(), dtype=np.uint8)
        info = {
            "pixels": pixels[None, None, ...],
            "goal": goal_pixels[None, None, ...],
            "action": np.zeros((1, 1, 2), dtype=np.float32),
        }
        action = policy.get_action(info)[0]
        _, _, terminated, _, _ = env.step(np.asarray(action, dtype=np.float32))
        env_success = env_success or bool(terminated)

    terminal_state = np.asarray(env_unwrapped._get_obs(), dtype=np.float32)
    metrics = block_pose_metrics(terminal_state, goal_state)
    return {
        "success": bool(metrics["success"]),
        "env_success": bool(env_success),
        "block_pos_dist": float(metrics["block_pos_dist"]),
        "angle_dist": float(metrics["angle_dist"]),
        "c_real_state": float(metrics["c_real_state"]),
        "elapsed_seconds": float(time.time() - started),
    }


def mean_bool(values: list[bool]) -> float | None:
    """Return the mean of a boolean list, or None for empty input."""
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def per_cell_success(records: list[dict]) -> dict[str, dict]:
    """Compute per-cell Euclidean and C_psi success rates."""
    out = {}
    for cell in sorted({record["cell"] for record in records}):
        cell_records = [record for record in records if record["cell"] == cell]
        euc = [bool(record["euclidean"]["success"]) for record in cell_records]
        cpsi = [bool(record["cpsi"]["success"]) for record in cell_records]
        out[cell] = {
            "n_pairs": len(cell_records),
            "euclidean_success_rate": mean_bool(euc),
            "cpsi_success_rate": mean_bool(cpsi),
            "delta": (
                mean_bool(cpsi) - mean_bool(euc)
                if mean_bool(cpsi) is not None and mean_bool(euc) is not None
                else None
            ),
        }
    return out


def headroom_closure(per_cell: dict[str, dict]) -> dict:
    """Compute V1-oracle headroom closure on cells with at least 10pp headroom."""
    closures = {}
    values = []
    eps = 1e-8
    for cell, metrics in per_cell.items():
        v1_rate = V1_CEM_LATE_SUCCESS_BY_CELL.get(cell)
        latent_rate = LATENT_CEM_LATE_SUCCESS_BY_CELL.get(cell)
        if v1_rate is None or latent_rate is None:
            continue
        phase1_headroom = v1_rate - latent_rate
        if phase1_headroom < 0.10:
            continue
        euc_rate = metrics["euclidean_success_rate"]
        cpsi_rate = metrics["cpsi_success_rate"]
        if euc_rate is None or cpsi_rate is None:
            continue
        closure = (cpsi_rate - euc_rate) / (phase1_headroom + eps)
        closures[cell] = {
            "phase1_latent_cem_late_success_rate": latent_rate,
            "phase1_v1_cem_late_success_rate": v1_rate,
            "phase1_headroom": phase1_headroom,
            "matched_euclidean_success_rate": euc_rate,
            "matched_cpsi_success_rate": cpsi_rate,
            "headroom_closure": float(closure),
        }
        values.append(float(closure))
    return {
        "cells": closures,
        "mean_headroom_closure": float(np.mean(values)) if values else None,
    }


def split3_rescue_metrics(records: list[dict]) -> dict:
    """Compute rescue metrics on the all-fail + strong-rho Split 3 holdout."""
    split3_ids = set(SPLIT3_TEST_PAIR_IDS)
    hard_records = [record for record in records if int(record["pair_id"]) in split3_ids]
    euclidean_failures = [
        record for record in hard_records if not bool(record["euclidean"]["success"])
    ]
    rescues = [
        record
        for record in euclidean_failures
        if bool(record["cpsi"]["success"])
    ]
    return {
        "n_hard_pairs_evaluated": len(hard_records),
        "n_euclidean_failures": len(euclidean_failures),
        "n_cpsi_rescues": len(rescues),
        "rescue_rate_among_euclidean_failures": (
            len(rescues) / len(euclidean_failures) if euclidean_failures else None
        ),
        "cpsi_success_rate_on_hard_pairs": mean_bool(
            [bool(record["cpsi"]["success"]) for record in hard_records]
        ),
        "rescued_pair_ids": [int(record["pair_id"]) for record in rescues],
    }


def latent_favorable_degradation(per_cell: dict[str, dict]) -> dict:
    """Report cells where Phase 1 latent CEM was stronger than V1 oracle CEM."""
    out = {}
    for cell in ("D0xR1", "D1xR0"):
        if cell not in per_cell:
            continue
        metrics = per_cell[cell]
        out[cell] = {
            "euclidean_success_rate": metrics["euclidean_success_rate"],
            "cpsi_success_rate": metrics["cpsi_success_rate"],
            "delta": metrics["delta"],
            "phase1_latent_cem_late_success_rate": LATENT_CEM_LATE_SUCCESS_BY_CELL[cell],
            "phase1_v1_cem_late_success_rate": V1_CEM_LATE_SUCCESS_BY_CELL[cell],
        }
    return out


def aggregate(records: list[dict], *, split: int) -> dict:
    """Compute aggregate planning metrics from per-pair records."""
    euc_successes = [bool(record["euclidean"]["success"]) for record in records]
    cpsi_successes = [bool(record["cpsi"]["success"]) for record in records]
    euc_rate = mean_bool(euc_successes)
    cpsi_rate = mean_bool(cpsi_successes)
    per_cell = per_cell_success(records)
    return {
        "n_pairs": len(records),
        "euclidean_success_rate": euc_rate,
        "cpsi_success_rate": cpsi_rate,
        "delta_success_rate": (
            cpsi_rate - euc_rate if euc_rate is not None and cpsi_rate is not None else None
        ),
        "euclidean_env_success_rate": mean_bool(
            [bool(record["euclidean"]["env_success"]) for record in records]
        ),
        "cpsi_env_success_rate": mean_bool(
            [bool(record["cpsi"]["env_success"]) for record in records]
        ),
        "per_cell": per_cell,
        "headroom_closure": headroom_closure(per_cell),
        "split3_rescue": split3_rescue_metrics(records) if split == 3 else None,
        "latent_favorable_degradation": latent_favorable_degradation(per_cell),
    }


def print_summary(summary: dict) -> None:
    """Print a concise human-readable result summary."""
    aggregate_metrics = summary["aggregate"]
    print("\n== P2-0 planning summary ==")
    print(f"split: {summary['metadata']['split']}")
    print(f"model_type: {summary['metadata'].get('model_type', 'mlp')}")
    print(f"variant: {summary['metadata']['variant']}")
    print(f"pairs: {aggregate_metrics['n_pairs']}")
    print(
        "success_rate: "
        f"Euclidean={aggregate_metrics['euclidean_success_rate']:.4f} "
        f"C_psi={aggregate_metrics['cpsi_success_rate']:.4f} "
        f"delta={aggregate_metrics['delta_success_rate']:.4f}"
    )
    if aggregate_metrics["split3_rescue"] is not None:
        rescue = aggregate_metrics["split3_rescue"]
        print(
            "split3_rescue: "
            f"{rescue['n_cpsi_rescues']}/{rescue['n_euclidean_failures']} "
            f"rate={rescue['rescue_rate_among_euclidean_failures']}"
        )
    print("\nPer-pair:")
    for record in summary["records"]:
        euc = "Y" if record["euclidean"]["success"] else "N"
        cpsi = "Y" if record["cpsi"]["success"] else "N"
        print(
            f"pair={record['pair_id']:02d} cell={record['cell']} "
            f"Euclidean={euc} C_psi={cpsi} "
            f"dist=({record['euclidean']['block_pos_dist']:.2f},"
            f"{record['cpsi']['block_pos_dist']:.2f}) "
            f"angle=({record['euclidean']['angle_dist']:.3f},"
            f"{record['cpsi']['angle_dist']:.3f})"
        )


def run(args: argparse.Namespace) -> dict:
    """Run the matched Euclidean/C_psi planning evaluation."""
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.model_checkpoint_dir = args.model_checkpoint_dir.expanduser().resolve()
    args.raw_model_checkpoint_dir = args.raw_model_checkpoint_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()

    explicit_pair_ids = parse_pair_ids(args.pairs_subset) if args.pairs_subset else None
    selected_pair_ids = (
        explicit_pair_ids
        if explicit_pair_ids is not None
        else split_test_pair_ids(args.split, seed=args.seed, pairs_path=args.pairs_path)
    )

    pairs_by_id = load_pairs_by_id(args.pairs_path)
    missing = sorted(set(selected_pair_ids) - set(pairs_by_id))
    if missing:
        raise ValueError(f"Requested pair IDs are missing from Track A pairs: {missing}")
    selected_pairs = [pairs_by_id[pair_id] for pair_id in selected_pair_ids]

    checkpoint_path, cost_head, loaded_variant = load_planning_cost_model(
        checkpoint_dir=args.checkpoint_dir,
        split=args.split,
        variant=args.variant,
        model_type=args.model_type,
        device=args.device,
    )

    print("== P2-0 planning evaluation setup ==")
    print(f"device: {args.device}")
    print(f"model_type: {args.model_type}")
    print(f"model_checkpoint_dir: {args.model_checkpoint_dir}")
    print(f"cost_model_checkpoint: {checkpoint_path}")
    print(f"loaded_variant: {loaded_variant}")
    print(f"pairs: {selected_pair_ids}")
    print(f"output_path: {args.output_path}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    env = gym.make("swm/PushT-v1")
    env_adapter = batched_action_adapter(env)

    policy_args = make_policy_args(args)
    euclidean_policy = build_planning_policy(
        policy_args=policy_args,
        process=process,
        env_adapter=env_adapter,
        cost_head=None,
    )
    cpsi_policy = build_planning_policy(
        policy_args=policy_args,
        process=process,
        env_adapter=env_adapter,
        cost_head=cost_head,
    )

    started = time.time()
    records = []
    try:
        for idx, pair in enumerate(selected_pairs, start=1):
            pair_id = int(pair["pair_id"])
            cell = str(pair["cell"])
            pair_rows = load_pair_rows(dataset, pair)
            print(f"\n[{idx:02d}/{len(selected_pairs):02d}] pair={pair_id} cell={cell}")

            euclidean = rollout_pair(
                env=env,
                policy=euclidean_policy,
                pair_rows=pair_rows,
                seed=args.seed + pair_id * 1009,
            )
            print(
                "  Euclidean: "
                f"success={euclidean['success']} "
                f"dist={euclidean['block_pos_dist']:.2f} "
                f"angle={euclidean['angle_dist']:.3f} "
                f"elapsed={euclidean['elapsed_seconds']:.1f}s"
            )

            cpsi = rollout_pair(
                env=env,
                policy=cpsi_policy,
                pair_rows=pair_rows,
                seed=args.seed + pair_id * 1009,
            )
            print(
                "  C_psi:     "
                f"success={cpsi['success']} "
                f"dist={cpsi['block_pos_dist']:.2f} "
                f"angle={cpsi['angle_dist']:.3f} "
                f"elapsed={cpsi['elapsed_seconds']:.1f}s"
            )

            records.append(
                {
                    "pair_id": pair_id,
                    "cell": cell,
                    "episode_id": int(pair["episode_id"]),
                    "start_row": int(pair["start_row"]),
                    "goal_row": int(pair["goal_row"]),
                    "euclidean": euclidean,
                    "cpsi": cpsi,
                }
            )
    finally:
        env.close()

    elapsed = time.time() - started
    pair_cell_map = cell_by_pair_id(load_track_a_pairs(args.pairs_path))
    summary = {
        "metadata": {
            "split": args.split,
            "model_type": args.model_type,
            "variant": loaded_variant,
            "requested_variant": args.variant,
            "seed": args.seed,
            "device": args.device,
            "pairs_path": args.pairs_path,
            "pairs_subset_override": explicit_pair_ids is not None,
            "selected_pair_ids": selected_pair_ids,
            "pair_cell_map": {pair_id: pair_cell_map[pair_id] for pair_id in selected_pair_ids},
            "raw_model_checkpoint_dir": args.raw_model_checkpoint_dir,
            "model_checkpoint_dir": args.model_checkpoint_dir,
            "cost_head_checkpoint": checkpoint_path,
            "cost_model_checkpoint": checkpoint_path,
            "output_path": args.output_path,
            "cache_dir": args.cache_dir,
            "dataset_name": args.dataset_name,
            "cem_config": {
                "num_samples": NUM_SAMPLES,
                "iterations": CEM_ITERS,
                "topk": TOPK,
                "var_scale": VAR_SCALE,
                "planning_horizon": PLANNING_HORIZON,
                "receding_horizon": RECEDING_HORIZON,
                "action_block": ACTION_BLOCK,
                "eval_budget": EVAL_BUDGET,
                "offset": OFFSET,
            },
            "success_definition": (
                "final block_pos_dist < 20 and final angle_dist < pi/9; "
                "env_success records whether the environment ever terminated."
            ),
            "elapsed_seconds": elapsed,
        },
        "records": records,
        "aggregate": aggregate(records, split=args.split),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False) + "\n")
    return summary


def main() -> int:
    """CLI entry point."""
    summary = run(parse_args())
    print_summary(summary)
    print(f"\nsaved_results: {summary['metadata']['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
