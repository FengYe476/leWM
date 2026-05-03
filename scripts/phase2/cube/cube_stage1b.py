#!/usr/bin/env python3
"""Cube Stage 1B projected-cost re-rank CEM audit.

This is the Cube analogue of the PushT Stage 1B projected-cost planning audit,
but intentionally uses the efficient re-rank design: run default LeWM CEM once
per pair, simulator-score the final candidate pool, then re-rank that labelled
pool under fixed random projections.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_cube_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_NAME,
    get_dataset,
)
from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    build_policy,
    build_processors,
    resolve_device,
)
from scripts.phase2.cube.extract_cube_latents import (  # noqa: E402
    ACTION_BLOCK,
    CUBE_SUCCESS_THRESHOLD_M,
    IMG_SIZE,
    LATENT_DIM,
    NUM_SAMPLES,
    TOPK,
    VAR_SCALE,
    cube_metrics,
    infer_raw_action_dim,
    load_pair_rows,
    load_pairs,
    prepare_pair_info,
    set_cube_start_and_goal,
    terminal_cube_pose,
    tensor_clone_info,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    clean_float,
    deterministic_topk_indices,
    iso_now,
    jsonable,
    pairwise_accuracy,
    spearman_corr,
)


DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_pairs.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1b.json"
DEFAULT_DIMS = (1, 2, 4, 8, 16, 32, 64, 128, 192)
DEFAULT_PROJECTION_SEEDS = (0, 1, 2)
DEFAULT_CEM_ITERS = 30
ENDPOINT_TOPK_VALUES = (5, 10, 30)
FALSE_ELITE_K = 30


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--dimensions", type=parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--projection-seeds", type=parse_int_list, default=DEFAULT_PROJECTION_SEEDS)
    parser.add_argument("--candidates-per-pair", type=int, default=NUM_SAMPLES)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--action-block", type=int, default=ACTION_BLOCK)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--cem-iters", type=int, default=DEFAULT_CEM_ITERS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--var-scale", type=float, default=VAR_SCALE)
    args = parser.parse_args()

    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive when provided")
    if args.action_block <= 0:
        parser.error("--action-block must be positive")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.topk < 1 or args.topk > args.num_samples:
        parser.error("--topk must be in [1, --num-samples]")
    if args.cem_iters < 1:
        parser.error("--cem-iters must be positive")
    if args.candidates_per_pair < max(max(ENDPOINT_TOPK_VALUES), FALSE_ELITE_K):
        parser.error("--candidates-per-pair must be at least 30 for top-k and false-elite metrics")
    if args.candidates_per_pair > args.num_samples:
        parser.error("--candidates-per-pair must be <= --num-samples")
    if any(int(dim) <= 0 or int(dim) > LATENT_DIM for dim in args.dimensions):
        parser.error(f"--dimensions values must be in [1, {LATENT_DIM}]")
    if any(int(seed) < 0 for seed in args.projection_seeds):
        parser.error("--projection-seeds must be nonnegative integers")
    return args


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def scalar_summary(values: list[float | int | bool | None]) -> dict[str, Any]:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "mean": clean_float(arr.mean()) if len(arr) else None,
        "std": clean_float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "min": clean_float(arr.min()) if len(arr) else None,
        "max": clean_float(arr.max()) if len(arr) else None,
        "n": int(len(arr)),
        "ddof": 1,
    }


def nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def topk_overlap_for_pool(
    *,
    costs: np.ndarray,
    reference_costs: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> float | None:
    if int(costs.shape[0]) < int(k):
        return None
    mask = np.ones(int(costs.shape[0]), dtype=bool)
    top_cost = deterministic_topk_indices(costs, candidate_ids, mask, int(k))
    top_reference = deterministic_topk_indices(reference_costs, candidate_ids, mask, int(k))
    return clean_float(len(set(top_cost.tolist()) & set(top_reference.tolist())) / float(k))


def false_elite_rate_for_pool(
    *,
    costs: np.ndarray,
    success: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> float | None:
    if int(costs.shape[0]) < int(k):
        return None
    mask = np.ones(int(costs.shape[0]), dtype=bool)
    top_indices = deterministic_topk_indices(costs, candidate_ids, mask, int(k))
    return clean_float(float(np.mean(~success[top_indices])))


def endpoint_ranking_metrics(
    *,
    projected_costs: np.ndarray,
    default_costs: np.ndarray,
    v1_costs: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    candidate_ids: np.ndarray,
) -> dict[str, Any]:
    pair_ids = np.zeros(int(projected_costs.shape[0]), dtype=np.int64)
    pairwise = pairwise_accuracy(projected_costs, c_real_state, pair_ids)
    return {
        "n_candidates": int(projected_costs.shape[0]),
        "candidate_indices": candidate_ids,
        "endpoint_spearman": clean_float(spearman_corr(projected_costs, c_real_state)),
        "pairwise_accuracy": pairwise["value"],
        "pairwise_comparisons": int(pairwise["n_pairs_compared"]),
        "topk_overlap_lewm": {
            str(k): topk_overlap_for_pool(
                costs=projected_costs,
                reference_costs=default_costs,
                candidate_ids=candidate_ids,
                k=int(k),
            )
            for k in ENDPOINT_TOPK_VALUES
        },
        "topk_overlap_v1": {
            str(k): topk_overlap_for_pool(
                costs=projected_costs,
                reference_costs=v1_costs,
                candidate_ids=candidate_ids,
                k=int(k),
            )
            for k in ENDPOINT_TOPK_VALUES
        },
        "false_elite_rate": false_elite_rate_for_pool(
            costs=projected_costs,
            success=success,
            candidate_ids=candidate_ids,
            k=FALSE_ELITE_K,
        ),
        "false_elite_k": FALSE_ELITE_K,
        "projected_cost_min": clean_float(float(np.min(projected_costs))),
        "projected_cost_max": clean_float(float(np.max(projected_costs))),
        "projected_cost_dynamic_range": clean_float(float(np.max(projected_costs) - np.min(projected_costs))),
        "projected_top30_cost_std": clean_float(
            float(np.std(np.sort(projected_costs, kind="mergesort")[:FALSE_ELITE_K], ddof=0))
        ),
    }


def make_policy_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        cem_iters=args.cem_iters,
        topk=args.topk,
        seed=args.seed,
        horizon=args.horizon_blocks,
        receding_horizon=args.horizon_blocks,
        action_block=args.action_block,
        img_size=args.img_size,
    )


def move_info_to_model_device(info: dict[str, Any], model) -> dict[str, Any]:
    device = next(model.parameters()).device
    out: dict[str, Any] = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            if device.type == "mps" and value.is_floating_point():
                out[key] = value.to(device=device, dtype=torch.float32)
            else:
                out[key] = value.to(device)
        else:
            out[key] = value
    return out


def expand_info_for_candidates(info: dict[str, Any], num_samples: int) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            expanded[key] = value.unsqueeze(1).expand(
                value.shape[0],
                int(num_samples),
                *value.shape[1:],
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None, ...], int(num_samples), axis=1)
        else:
            expanded[key] = deepcopy(value)
    return expanded


def rollout_candidate_latents(
    model,
    prepared_info: dict[str, Any],
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    info = expand_info_for_candidates(prepared_info, int(candidates.shape[1]))
    info = tensor_clone_info(info)
    info = move_info_to_model_device(info, model)

    goal = {key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)}
    goal["pixels"] = goal["goal"]
    for key in list(goal.keys()):
        if key.startswith("goal_"):
            goal[key[len("goal_") :]] = goal.pop(key)
    goal.pop("action", None)

    with torch.inference_mode():
        goal_encoded = model.encode(goal)
        info["goal_emb"] = goal_encoded["emb"]
        info = model.rollout(info, candidates)
        z_pred = info["predicted_emb"][..., -1, :]
        z_goal = info["goal_emb"][:, -1, :]
    return z_pred, z_goal


def euclidean_costs(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    return torch.sum((z_pred - z_goal) ** 2, dim=-1)


def make_projection(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((LATENT_DIM, int(dim)), generator=generator, dtype=torch.float32) / math.sqrt(int(dim))


def projection_metadata(projections: dict[tuple[int, int], torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (dim, seed), projection in sorted(projections.items()):
        out[f"m={int(dim)},seed={int(seed)}"] = {
            "dimension": int(dim),
            "seed": int(seed),
            "shape": list(projection.shape),
            "scale": f"1/sqrt({int(dim)})",
            "mean": clean_float(float(projection.mean().item())),
            "std": clean_float(float(projection.std(unbiased=False).item())),
            "frobenius_norm": clean_float(float(torch.linalg.norm(projection).item())),
        }
    return out


def numpy_projected_costs(
    *,
    z_pred: np.ndarray,
    z_goal: np.ndarray,
    projection: torch.Tensor,
) -> np.ndarray:
    projection_np = projection.detach().cpu().numpy().astype(np.float32)
    if z_goal.ndim == 1:
        z_goal = np.repeat(z_goal[None, :], int(z_pred.shape[0]), axis=0)
    pred_proj = np.asarray(z_pred, dtype=np.float32) @ projection_np
    goal_proj = np.asarray(z_goal, dtype=np.float32) @ projection_np
    return np.sum((pred_proj - goal_proj) ** 2, axis=1).astype(np.float64)


def blocked_batch_to_raw(
    blocked: np.ndarray,
    *,
    action_processor: Any,
    action_block: int,
    raw_action_dim: int,
) -> np.ndarray:
    blocked = np.asarray(blocked, dtype=np.float32)
    blocked_dim = int(action_block) * int(raw_action_dim)
    if blocked.ndim != 3 or blocked.shape[-1] != blocked_dim:
        raise ValueError(f"Expected blocked shape (N,H,{blocked_dim}), got {blocked.shape}")
    n_candidates, horizon_blocks, _ = blocked.shape
    normalized = blocked.reshape(n_candidates * horizon_blocks * int(action_block), int(raw_action_dim))
    raw = action_processor.inverse_transform(normalized).astype(np.float32)
    return raw.reshape(n_candidates, horizon_blocks * int(action_block), int(raw_action_dim))


def run_default_cem(
    *,
    model,
    prepared_info: dict[str, Any],
    pair_id: int,
    seed: int,
    horizon_blocks: int,
    action_dim: int,
    num_samples: int,
    cem_iters: int,
    topk: int,
    var_scale: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed) + int(pair_id) * 1009)
    mean = torch.zeros((1, int(horizon_blocks), int(action_dim)), dtype=torch.float32, device=device)
    var = float(var_scale) * torch.ones_like(mean)
    final: dict[str, Any] | None = None
    started = time.time()

    for iter_idx in range(1, int(cem_iters) + 1):
        candidates = torch.randn(
            1,
            int(num_samples),
            int(horizon_blocks),
            int(action_dim),
            generator=generator,
            device=device,
        )
        candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        z_pred, z_goal = rollout_candidate_latents(model, prepared_info, candidates)
        default_cost = euclidean_costs(z_pred, z_goal)
        top_vals, top_inds = torch.topk(default_cost, k=int(topk), dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]

        if iter_idx == int(cem_iters):
            cost_np = default_cost[0].detach().cpu().numpy().astype(np.float64)
            top_np = top_vals[0].detach().cpu().numpy().astype(np.float64)
            final = {
                "blocked_candidates": candidates[0].detach().cpu().numpy().astype(np.float32),
                "rank1_blocked": elite_candidates[0, 0].detach().cpu().numpy().astype(np.float32),
                "rank1_candidate_index": int(top_inds[0, 0].detach().cpu().item()),
                "default_costs": cost_np,
                "z_pred": z_pred[0].detach().cpu().numpy().astype(np.float32),
                "z_goal": z_goal[0].detach().cpu().numpy().astype(np.float32),
                "top30_default_costs": top_np,
                "top30_default_cost_std": clean_float(float(np.std(top_np, ddof=0))),
                "default_cost_dynamic_range": clean_float(float(np.max(cost_np) - np.min(cost_np))),
                "default_cost_min": clean_float(float(np.min(cost_np))),
                "default_cost_max": clean_float(float(np.max(cost_np))),
            }

        mean = elite_candidates.mean(dim=1)
        var = elite_candidates.std(dim=1)

    if final is None:
        raise RuntimeError("CEM final iteration was not captured")
    final["wallclock_seconds"] = clean_float(time.time() - started)
    return final


def select_scored_candidate_indices(default_costs: np.ndarray, candidates_per_pair: int) -> np.ndarray:
    default_costs = np.asarray(default_costs, dtype=np.float64)
    if int(candidates_per_pair) >= int(default_costs.shape[0]):
        return np.arange(int(default_costs.shape[0]), dtype=np.int64)
    candidate_ids = np.arange(int(default_costs.shape[0]), dtype=np.int64)
    order = np.lexsort((candidate_ids, default_costs))
    return np.asarray(order[: int(candidates_per_pair)], dtype=np.int64)


def execute_raw_actions_no_render(
    env,
    *,
    initial: dict[str, Any],
    goal: dict[str, Any],
    raw_actions: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    set_cube_start_and_goal(env, initial=initial, goal=goal, seed=int(seed))
    env_success = False
    for action in np.asarray(raw_actions, dtype=np.float32):
        _, _, terminated, _, info = env.step(action)
        env_success = env_success or bool(terminated) or bool(info.get("success", False))
    terminal_pos, terminal_quat = terminal_cube_pose(env)
    return {
        "terminal_cube_pos": terminal_pos.astype(np.float64),
        "terminal_cube_quat": terminal_quat.astype(np.float64),
        "env_success": bool(env_success),
    }


def score_raw_actions(
    *,
    env,
    initial: dict[str, Any],
    goal: dict[str, Any],
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    raw_actions_batch: np.ndarray,
    candidate_indices: np.ndarray,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    raw_actions_batch = np.asarray(raw_actions_batch, dtype=np.float32)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.shape != (int(raw_actions_batch.shape[0]),):
        raise ValueError("candidate_indices must provide one index per candidate")

    v1_costs = np.empty((int(raw_actions_batch.shape[0]),), dtype=np.float64)
    c_real_state = np.empty_like(v1_costs)
    success = np.empty((int(raw_actions_batch.shape[0]),), dtype=bool)
    metrics: list[dict[str, Any]] = []

    for row_idx, (candidate_idx, raw_actions) in enumerate(zip(candidate_indices, raw_actions_batch, strict=True)):
        rollout = execute_raw_actions_no_render(
            env,
            initial=initial,
            goal=goal,
            raw_actions=raw_actions,
            seed=int(seed_base) + int(candidate_idx),
        )
        pose = cube_metrics(
            terminal_pos=rollout["terminal_cube_pos"],
            terminal_quat=rollout["terminal_cube_quat"],
            goal_pos=goal_pos,
            goal_quat=goal_quat,
        )
        c_real = float(pose["C_real_state"])
        v1_costs[row_idx] = c_real
        c_real_state[row_idx] = c_real
        success[row_idx] = bool(pose["success"])
        metrics.append(
            {
                "candidate_index": int(candidate_idx),
                "scored_pool_position": int(row_idx),
                "seed": int(seed_base) + int(candidate_idx),
                "v1_cost": clean_float(c_real),
                "c_real_state": clean_float(c_real),
                "cube_pos_dist": clean_float(float(pose["cube_pos_dist"])),
                "quat_angle_dist": clean_float(float(pose["quat_angle_dist"])),
                "success": bool(pose["success"]),
                "env_success": bool(rollout["env_success"]),
                "terminal_cube_pos": rollout["terminal_cube_pos"],
                "terminal_cube_quat": rollout["terminal_cube_quat"],
            }
        )

    return v1_costs, c_real_state, success, metrics


def build_default_baseline_record(
    *,
    pair_id: int,
    cell: str,
    cem_seed: int,
    baseline: dict[str, Any],
    candidate_indices: np.ndarray,
    raw_scored_candidates: np.ndarray,
    v1_costs: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    scored_metrics: list[dict[str, Any]],
    scoring_mode: str,
    original_n: int,
) -> dict[str, Any]:
    rank1_idx = int(baseline["rank1_candidate_index"])
    rank1_positions = np.flatnonzero(np.asarray(candidate_indices, dtype=np.int64) == rank1_idx)
    if len(rank1_positions) != 1:
        raise RuntimeError(f"Default rank-1 candidate {rank1_idx} was not scored")
    rank1_pos = int(rank1_positions[0])
    scored_default_costs = np.asarray(baseline["default_costs"], dtype=np.float64)[candidate_indices]
    return {
        "pair_id": int(pair_id),
        "cell": str(cell),
        "cem_sampling_seed": int(cem_seed),
        "candidate_scoring_mode": str(scoring_mode),
        "candidate_pool_original_n": int(original_n),
        "candidate_pool_scored_n": int(len(candidate_indices)),
        "rank1_candidate_index": rank1_idx,
        "rank1_scored_pool_position": rank1_pos,
        "rank1_success": bool(success[rank1_pos]),
        "rank1_v1_cost": clean_float(float(v1_costs[rank1_pos])),
        "rank1_c_real_state": clean_float(float(c_real_state[rank1_pos])),
        "rank1_state_metrics": scored_metrics[rank1_pos],
        "final_top30_default_cost_std": baseline["top30_default_cost_std"],
        "final_default_cost_dynamic_range": baseline["default_cost_dynamic_range"],
        "final_default_cost_min": baseline["default_cost_min"],
        "final_default_cost_max": baseline["default_cost_max"],
        "rank1_blocked_action": baseline["rank1_blocked"],
        "rank1_raw_action": raw_scored_candidates[rank1_pos],
        "candidate_pool": {
            "n_candidates": int(len(candidate_indices)),
            "original_n_candidates": int(original_n),
            "candidate_indices": candidate_indices,
            "default_costs": scored_default_costs,
            "v1_costs": v1_costs,
            "c_real_state": c_real_state,
            "success": success,
            "success_rate": clean_float(float(np.mean(success))),
            "v1_cost_min": clean_float(float(np.min(v1_costs))),
            "v1_cost_max": clean_float(float(np.max(v1_costs))),
            "c_real_state_min": clean_float(float(np.min(c_real_state))),
            "c_real_state_max": clean_float(float(np.max(c_real_state))),
        },
        "wallclock_seconds": baseline["wallclock_seconds"],
    }


def build_projected_record(
    *,
    pair_id: int,
    cell: str,
    dim: int,
    projection_seed: int,
    cem_seed: int,
    projection: torch.Tensor,
    baseline: dict[str, Any],
    candidate_indices: np.ndarray,
    raw_scored_candidates: np.ndarray,
    baseline_raw_rank1: np.ndarray,
    baseline_v1_costs: np.ndarray,
    baseline_c_real_state: np.ndarray,
    baseline_success: np.ndarray,
    baseline_metrics: list[dict[str, Any]],
    scoring_mode: str,
    original_n: int,
) -> dict[str, Any]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    projected_pool_costs = numpy_projected_costs(
        z_pred=np.asarray(baseline["z_pred"], dtype=np.float32)[candidate_indices],
        z_goal=np.asarray(baseline["z_goal"], dtype=np.float32),
        projection=projection,
    )
    default_pool_costs = np.asarray(baseline["default_costs"], dtype=np.float64)[candidate_indices]
    endpoint_metrics = endpoint_ranking_metrics(
        projected_costs=projected_pool_costs,
        default_costs=default_pool_costs,
        v1_costs=baseline_v1_costs,
        c_real_state=baseline_c_real_state,
        success=baseline_success,
        candidate_ids=candidate_indices,
    )
    mask = np.ones(int(len(candidate_indices)), dtype=bool)
    rank1_pos = int(deterministic_topk_indices(projected_pool_costs, candidate_indices, mask, 1)[0])
    rank1_idx = int(candidate_indices[rank1_pos])
    projected_blocked = np.asarray(baseline["blocked_candidates"], dtype=np.float32)[rank1_idx]
    projected_raw_rank1 = np.asarray(raw_scored_candidates, dtype=np.float32)[rank1_pos]

    return {
        "pair_id": int(pair_id),
        "cell": str(cell),
        "dimension": int(dim),
        "projection_seed": int(projection_seed),
        "cem_sampling_seed": int(cem_seed),
        "candidate_scoring_mode": str(scoring_mode),
        "candidate_pool_original_n": int(original_n),
        "candidate_pool_scored_n": int(len(candidate_indices)),
        "projection_matrix": {
            "shape": [LATENT_DIM, int(dim)],
            "scale": f"1/sqrt({int(dim)})",
            "fixed_scope": "fixed per (dimension, projection_seed) across pairs",
        },
        "cem_late_success": bool(baseline_success[rank1_pos]),
        "cem_late_v1_cost": clean_float(float(baseline_v1_costs[rank1_pos])),
        "cem_late_c_real_state": clean_float(float(baseline_c_real_state[rank1_pos])),
        "cem_late_state_metrics": baseline_metrics[rank1_pos],
        "endpoint_metrics_on_default_pool": endpoint_metrics,
        "projected_cem_diagnostics": {
            "planning_mode": "rerank_default_final_pool",
            "rank1_candidate_index": rank1_idx,
            "rank1_scored_pool_position": rank1_pos,
            "rank1_projected_cost": clean_float(float(projected_pool_costs[rank1_pos])),
            "rank1_default_latent_cost": clean_float(float(default_pool_costs[rank1_pos])),
            "final_top30_elite_cost_std": endpoint_metrics["projected_top30_cost_std"],
            "final_candidate_dynamic_range": endpoint_metrics["projected_cost_dynamic_range"],
            "final_projected_cost_min": endpoint_metrics["projected_cost_min"],
            "final_projected_cost_max": endpoint_metrics["projected_cost_max"],
            "final_default_latent_cost_min": clean_float(float(np.min(default_pool_costs))),
            "final_default_latent_cost_max": clean_float(float(np.max(default_pool_costs))),
            "final_default_latent_cost_dynamic_range": clean_float(
                float(np.max(default_pool_costs) - np.min(default_pool_costs))
            ),
            "action_l2_to_default_blocked": clean_float(
                float(np.linalg.norm(projected_blocked - np.asarray(baseline["rank1_blocked"], dtype=np.float32)))
            ),
            "action_l2_to_default_raw": clean_float(float(np.linalg.norm(projected_raw_rank1 - baseline_raw_rank1))),
            "wallclock_seconds": 0.0,
        },
        "rank1_blocked_action": projected_blocked,
        "rank1_raw_action": projected_raw_rank1,
    }


def aggregate_record_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "projected_success_rate": scalar_summary([record.get("cem_late_success") for record in records]),
        "cem_late_c_real_state": scalar_summary([record.get("cem_late_c_real_state") for record in records]),
        "endpoint_spearman": scalar_summary(
            [record["endpoint_metrics_on_default_pool"].get("endpoint_spearman") for record in records]
        ),
        "pairwise_accuracy": scalar_summary(
            [record["endpoint_metrics_on_default_pool"].get("pairwise_accuracy") for record in records]
        ),
        "false_elite_rate": scalar_summary(
            [record["endpoint_metrics_on_default_pool"].get("false_elite_rate") for record in records]
        ),
        "topk_overlap_lewm": {
            str(k): scalar_summary(
                [record["endpoint_metrics_on_default_pool"].get("topk_overlap_lewm", {}).get(str(k)) for record in records]
            )
            for k in ENDPOINT_TOPK_VALUES
        },
        "topk_overlap_v1": {
            str(k): scalar_summary(
                [record["endpoint_metrics_on_default_pool"].get("topk_overlap_v1", {}).get(str(k)) for record in records]
            )
            for k in ENDPOINT_TOPK_VALUES
        },
        "projected_elite_cost_std": scalar_summary(
            [record["projected_cem_diagnostics"].get("final_top30_elite_cost_std") for record in records]
        ),
        "projected_dynamic_range": scalar_summary(
            [record["projected_cem_diagnostics"].get("final_candidate_dynamic_range") for record in records]
        ),
        "action_l2_to_default_blocked": scalar_summary(
            [record["projected_cem_diagnostics"].get("action_l2_to_default_blocked") for record in records]
        ),
        "action_l2_to_default_raw": scalar_summary(
            [record["projected_cem_diagnostics"].get("action_l2_to_default_raw") for record in records]
        ),
    }


def seed_mean_summary(seed_groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "projected_success_rate": ("projected_success_rate", "mean"),
        "cem_late_c_real_state": ("cem_late_c_real_state", "mean"),
        "endpoint_spearman": ("endpoint_spearman", "mean"),
        "pairwise_accuracy": ("pairwise_accuracy", "mean"),
        "false_elite_rate": ("false_elite_rate", "mean"),
        "projected_elite_cost_std": ("projected_elite_cost_std", "mean"),
        "projected_dynamic_range": ("projected_dynamic_range", "mean"),
        "action_l2_to_default_blocked": ("action_l2_to_default_blocked", "mean"),
        "action_l2_to_default_raw": ("action_l2_to_default_raw", "mean"),
    }
    out = {
        name: scalar_summary([nested_get(group, path) for group in seed_groups.values()])
        for name, path in paths.items()
    }
    out["topk_overlap_lewm"] = {
        str(k): scalar_summary(
            [nested_get(group, ("topk_overlap_lewm", str(k), "mean")) for group in seed_groups.values()]
        )
        for k in ENDPOINT_TOPK_VALUES
    }
    out["topk_overlap_v1"] = {
        str(k): scalar_summary(
            [nested_get(group, ("topk_overlap_v1", str(k), "mean")) for group in seed_groups.values()]
        )
        for k in ENDPOINT_TOPK_VALUES
    }
    return out


def aggregate_default_baselines(default_baselines: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(default_baselines)),
        "rank1_success_rate": scalar_summary([record.get("rank1_success") for record in default_baselines]),
        "rank1_c_real_state": scalar_summary([record.get("rank1_c_real_state") for record in default_baselines]),
        "rank1_v1_cost": scalar_summary([record.get("rank1_v1_cost") for record in default_baselines]),
        "candidate_pool_success_rate": scalar_summary(
            [record.get("candidate_pool", {}).get("success_rate") for record in default_baselines]
        ),
        "candidate_pool_scored_n": scalar_summary([record.get("candidate_pool_scored_n") for record in default_baselines]),
        "final_top30_default_cost_std": scalar_summary(
            [record.get("final_top30_default_cost_std") for record in default_baselines]
        ),
        "final_default_cost_dynamic_range": scalar_summary(
            [record.get("final_default_cost_dynamic_range") for record in default_baselines]
        ),
    }


def aggregate_records(
    *,
    records: list[dict[str, Any]],
    default_baselines: list[dict[str, Any]],
    dimensions: tuple[int, ...],
    projection_seeds: tuple[int, ...],
    cells: tuple[str, ...],
) -> dict[str, Any]:
    by_dimension: dict[str, Any] = {}
    by_dimension_and_projection_seed: dict[str, dict[str, Any]] = {}
    for dim in dimensions:
        dim_records = [record for record in records if int(record["dimension"]) == int(dim)]
        seed_groups = {
            str(int(seed)): aggregate_record_group(
                [record for record in dim_records if int(record["projection_seed"]) == int(seed)]
            )
            for seed in projection_seeds
        }
        dim_group = aggregate_record_group(dim_records)
        dim_group["by_projection_seed"] = seed_groups
        dim_group["seed_mean_summary"] = seed_mean_summary(seed_groups)
        by_dimension[str(int(dim))] = dim_group
        by_dimension_and_projection_seed[str(int(dim))] = seed_groups

    return {
        "expected_projected_records": int(len(default_baselines) * len(dimensions) * len(projection_seeds)),
        "observed_projected_records": int(len(records)),
        "default_baselines": aggregate_default_baselines(default_baselines),
        "overall": aggregate_record_group(records),
        "by_dimension": by_dimension,
        "by_projection_seed": {
            str(int(seed)): aggregate_record_group(
                [record for record in records if int(record["projection_seed"]) == int(seed)]
            )
            for seed in projection_seeds
        },
        "by_dimension_and_projection_seed": by_dimension_and_projection_seed,
        "by_cell": {
            str(cell): aggregate_record_group([record for record in records if str(record["cell"]) == str(cell)])
            for cell in cells
        },
        "by_dimension_and_cell": {
            str(int(dim)): {
                str(cell): aggregate_record_group(
                    [
                        record
                        for record in records
                        if int(record["dimension"]) == int(dim) and str(record["cell"]) == str(cell)
                    ]
                )
                for cell in cells
            }
            for dim in dimensions
        },
    }


def print_summary_table(aggregate: dict[str, Any]) -> None:
    rows = []
    for dim, stats in sorted(aggregate["by_dimension"].items(), key=lambda item: int(item[0])):
        seed_stats = stats.get("seed_mean_summary", {})
        rows.append(
            [
                f"m={dim}",
                str(stats["n_records"]),
                fmt(nested_get(seed_stats, ("projected_success_rate", "mean"))),
                fmt(nested_get(seed_stats, ("endpoint_spearman", "mean"))),
                fmt(nested_get(seed_stats, ("pairwise_accuracy", "mean"))),
                fmt(nested_get(seed_stats, ("false_elite_rate", "mean"))),
                fmt(nested_get(seed_stats, ("action_l2_to_default_blocked", "mean"))),
            ]
        )
    headers = ["Dim", "N", "Success", "Spearman", "Pairwise", "FalseElite", "ActionL2"]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print("Cube Stage 1B re-rank projected CEM summary")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def seconds_to_hms(seconds: float) -> str:
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    pairs_data, requested_pairs = load_pairs(args.pairs_path, max_pairs=args.max_pairs, pair_ids=None)
    pairs_metadata = pairs_data.get("metadata", {})
    offset = int(pairs_metadata.get("offset", requested_pairs[0]["goal_row"] - requested_pairs[0]["start_row"]))
    if offset % int(args.action_block) != 0:
        raise ValueError("Cube offset must be divisible by --action-block")
    validate_requested_pair_offsets(requested_pairs, offset=offset)
    args.horizon_blocks = int(offset // int(args.action_block))

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    raw_action_dim = infer_raw_action_dim(dataset)
    if raw_action_dim != 5:
        raise ValueError(f"Expected Cube raw action dim 5, got {raw_action_dim}")
    process = build_processors(dataset, ["action"])
    policy = build_policy(make_policy_namespace(args), process)
    model = policy.solver.model
    action_processor = policy.process["action"]

    dimensions = tuple(int(dim) for dim in args.dimensions)
    projection_seeds = tuple(int(seed) for seed in args.projection_seeds)
    projections = {
        (int(dim), int(seed)): make_projection(int(dim), int(seed))
        for dim in dimensions
        for seed in projection_seeds
    }
    candidate_scoring_mode = "all_300" if int(args.candidates_per_pair) == int(args.num_samples) else "topk_by_default_cost"

    print("== Cube Stage 1B re-rank setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"pairs: {len(requested_pairs)}")
    print(f"offset: {offset}")
    print(f"horizon_blocks: {args.horizon_blocks}")
    print(f"raw_action_dim: {raw_action_dim}")
    print(f"blocked_action_dim: {raw_action_dim * args.action_block}")
    print(f"candidates_per_pair: {args.candidates_per_pair} ({candidate_scoring_mode})")
    print(f"dimensions: {list(dimensions)}")
    print(f"projection_seeds: {list(projection_seeds)}")
    print(f"expected_simulator_rollouts: {len(requested_pairs) * int(args.candidates_per_pair)}")
    print(f"expected_projected_records: {len(requested_pairs) * len(dimensions) * len(projection_seeds)}")

    default_baselines: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    total_started = time.time()
    env = gym.make(
        "swm/OGBCube-v0",
        env_type="single",
        ob_type="states",
        multiview=False,
        width=int(args.img_size),
        height=int(args.img_size),
        visualize_info=False,
        terminate_at_goal=True,
    )
    try:
        for pair_idx, pair_spec in enumerate(requested_pairs, start=1):
            pair_started = time.time()
            pair_id = int(pair_spec["pair_id"])
            cell = str(pair_spec["cell"])
            pair_rows = load_pair_rows(dataset, int(pair_spec["start_row"]), int(pair_spec["goal_row"]))
            initial = pair_rows["initial"]
            goal = pair_rows["goal"]
            goal_pos = np.asarray(goal["privileged_block_0_pos"], dtype=np.float64)
            goal_quat = np.asarray(goal["privileged_block_0_quat"], dtype=np.float64)
            prepared_info = prepare_pair_info(
                policy,
                initial["pixels"],
                goal["pixels"],
                raw_action_dim=raw_action_dim,
            )
            cem_seed = int(args.seed) + pair_id * 1009

            print(f"\n[{pair_idx}/{len(requested_pairs)}] pair_id={pair_id} cell={cell}: default CEM")
            baseline = run_default_cem(
                model=model,
                prepared_info=prepared_info,
                pair_id=pair_id,
                seed=int(args.seed),
                horizon_blocks=int(args.horizon_blocks),
                action_dim=int(args.action_block) * int(raw_action_dim),
                num_samples=int(args.num_samples),
                cem_iters=int(args.cem_iters),
                topk=int(args.topk),
                var_scale=float(args.var_scale),
            )
            candidate_indices = select_scored_candidate_indices(
                np.asarray(baseline["default_costs"], dtype=np.float64),
                int(args.candidates_per_pair),
            )
            print(
                f"  scoring {len(candidate_indices)}/{args.num_samples} final-pool candidates "
                f"({candidate_scoring_mode})"
            )
            raw_scored = blocked_batch_to_raw(
                np.asarray(baseline["blocked_candidates"], dtype=np.float32)[candidate_indices],
                action_processor=action_processor,
                action_block=int(args.action_block),
                raw_action_dim=int(raw_action_dim),
            )
            baseline_v1, baseline_real, baseline_success, baseline_metrics = score_raw_actions(
                env=env,
                initial=initial,
                goal=goal,
                goal_pos=goal_pos,
                goal_quat=goal_quat,
                raw_actions_batch=raw_scored,
                candidate_indices=candidate_indices,
                seed_base=int(args.seed) + pair_id * 100_000,
            )
            rank1_pos = int(np.flatnonzero(candidate_indices == int(baseline["rank1_candidate_index"]))[0])
            baseline_rank1_raw = raw_scored[rank1_pos]
            default_baselines.append(
                build_default_baseline_record(
                    pair_id=pair_id,
                    cell=cell,
                    cem_seed=cem_seed,
                    baseline=baseline,
                    candidate_indices=candidate_indices,
                    raw_scored_candidates=raw_scored,
                    v1_costs=baseline_v1,
                    c_real_state=baseline_real,
                    success=baseline_success,
                    scored_metrics=baseline_metrics,
                    scoring_mode=candidate_scoring_mode,
                    original_n=int(args.num_samples),
                )
            )

            for dim in dimensions:
                for projection_seed in projection_seeds:
                    projection = projections[(int(dim), int(projection_seed))]
                    records.append(
                        build_projected_record(
                            pair_id=pair_id,
                            cell=cell,
                            dim=int(dim),
                            projection_seed=int(projection_seed),
                            cem_seed=cem_seed,
                            projection=projection,
                            baseline=baseline,
                            candidate_indices=candidate_indices,
                            raw_scored_candidates=raw_scored,
                            baseline_raw_rank1=baseline_rank1_raw,
                            baseline_v1_costs=baseline_v1,
                            baseline_c_real_state=baseline_real,
                            baseline_success=baseline_success,
                            baseline_metrics=baseline_metrics,
                            scoring_mode=candidate_scoring_mode,
                            original_n=int(args.num_samples),
                        )
                    )

            print(
                f"  pair_id={pair_id} completed in {seconds_to_hms(time.time() - pair_started)}; "
                f"default_success={bool(default_baselines[-1]['rank1_success'])}; "
                f"projected_records={len(records)}"
            )
    finally:
        env.close()

    total_wallclock = time.time() - total_started
    cells = tuple(sorted({str(pair["cell"]) for pair in requested_pairs}))
    aggregate = aggregate_records(
        records=records,
        default_baselines=default_baselines,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        cells=cells,
    )
    estimated_full_seconds = (
        clean_float(total_wallclock * (100.0 / float(len(requested_pairs)))) if requested_pairs else None
    )
    output: dict[str, Any] = {
        "metadata": {
            "format": "cube_stage1b_rerank_projected_cem",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "planning_mode": "rerank_default_final_pool",
            "description": (
                "Default LeWM final CEM candidate pools are simulator-scored once per pair; "
                "fixed random projections re-rank only that labelled pool."
            ),
            "output": str(args.output),
            "pairs_path": str(args.pairs_path),
            "selected_pair_ids": [int(pair["pair_id"]) for pair in requested_pairs],
            "n_selected_pairs": int(len(requested_pairs)),
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_cache_dir": str(args.cache_dir),
            "dataset_name": args.dataset_name,
            "device": args.device,
            "seed": int(args.seed),
            "dimensions": [int(dim) for dim in dimensions],
            "projection_seeds": [int(seed) for seed in projection_seeds],
            "expected_projected_records": int(len(requested_pairs) * len(dimensions) * len(projection_seeds)),
            "expected_simulator_rollouts": {
                "default_final_candidate_pool_scored": int(len(requested_pairs) * int(args.candidates_per_pair)),
                "projected_rank1_extra": 0,
                "total": int(len(requested_pairs) * int(args.candidates_per_pair)),
            },
            "candidate_scoring_mode": candidate_scoring_mode,
            "candidate_pool_original_n": int(args.num_samples),
            "candidate_pool_scored_n": int(args.candidates_per_pair),
            "cem_config": {
                "samples_per_iter": int(args.num_samples),
                "iterations": int(args.cem_iters),
                "elites": int(args.topk),
                "planning_horizon_blocks": int(args.horizon_blocks),
                "action_block": int(args.action_block),
                "raw_action_dim": int(raw_action_dim),
                "blocked_action_dim": int(raw_action_dim * args.action_block),
                "raw_action_steps": int(args.horizon_blocks * args.action_block),
                "var_scale": float(args.var_scale),
                "img_size": int(args.img_size),
                "sampling_seed_rule": "base_seed + pair_id * 1009",
                "candidate_0_forced_to_search_mean": True,
            },
            "cube_pair_metadata": pairs_metadata,
            "success_definition": f"cube_pos_dist <= {CUBE_SUCCESS_THRESHOLD_M:.2f}m",
            "v1_cost_definition": "Cube v1_cost aliases C_real_state / cube_pos_dist.",
            "projection_config": {
                "matrix_shape": "[192, m]",
                "entry_distribution": "Gaussian N(0, 1) / sqrt(m)",
                "fixed_scope": "one matrix per (m, projection_seed), reused across all pairs",
                "matrices": projection_metadata(projections),
            },
            "endpoint_metric_pool": {
                "source": "scored subset of default LeWM final CEM iteration candidate pool",
                "topk_values": list(ENDPOINT_TOPK_VALUES),
                "false_elite_k": FALSE_ELITE_K,
            },
            "runtime": {
                "wallclock_seconds": clean_float(total_wallclock),
                "wallclock_hms": seconds_to_hms(total_wallclock),
                "estimated_full_100_pair_seconds_same_candidate_count": estimated_full_seconds,
                "estimated_full_100_pair_hms_same_candidate_count": (
                    seconds_to_hms(float(estimated_full_seconds)) if estimated_full_seconds is not None else None
                ),
            },
        },
        "default_baselines": default_baselines,
        "records": records,
        "aggregate": aggregate,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print()
    print_summary_table(aggregate)
    print(f"\nWallclock seconds: {total_wallclock:.3f} ({seconds_to_hms(total_wallclock)})")
    if estimated_full_seconds is not None:
        print(
            "Estimated full 100-pair runtime at same candidate count: "
            f"{estimated_full_seconds:.3f}s ({seconds_to_hms(float(estimated_full_seconds))})"
        )
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
