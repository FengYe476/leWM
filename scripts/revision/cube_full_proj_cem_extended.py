#!/usr/bin/env python3
"""Extended Cube full projected-CEM sweep with resumable full-pool scoring.

This revision runner extends the protocol-matched Cube full projected-CEM cell
from 25 pairs x 1 seed to 50 pairs x 3 seeds. Unlike the original rank-1-only
runner, each projected-CEM final pool is simulator-scored for all 300 candidates
and saved as a ``.pt`` artifact for later pool-ranking analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
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

from eval_cube_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_NAME,
    get_dataset,
)
from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    build_policy,
    build_processors,
)
from scripts.phase2.cube.cube_stage1b import (  # noqa: E402
    DEFAULT_CEM_ITERS,
    blocked_batch_to_raw,
    endpoint_ranking_metrics,
    make_policy_namespace,
    make_projection,
    projection_metadata,
    rollout_candidate_latents,
    score_raw_actions,
)
from scripts.phase2.cube.extract_cube_latents import (  # noqa: E402
    ACTION_BLOCK,
    CUBE_SUCCESS_THRESHOLD_M,
    IMG_SIZE,
    LATENT_DIM,
    NUM_SAMPLES,
    TOPK,
    VAR_SCALE,
    infer_raw_action_dim,
    load_pair_rows,
    load_pairs,
    prepare_pair_info,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    clean_float,
    iso_now,
    jsonable,
    spearman_corr,
)


DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_pairs.json"
DEFAULT_STAGE1B_PATH = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1b.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "cube_full_proj_cem_extended.json"
DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "revision" / "cube_full_proj_pools"
SMOKE_OUTPUT = PROJECT_ROOT / "results" / "revision" / "cube_full_proj_cem_extended_smoke.json"
SMOKE_POOL_DIR = PROJECT_ROOT / "results" / "revision" / "cube_full_proj_pools_smoke"

DEFAULT_DIMS = (1, 8, 32, 64, 192)
DEFAULT_PROJECTION_SEEDS = (0, 1, 2)
DEFAULT_PAIR_COUNT = 50
SMOKE_DIMS = (1, 8)
SMOKE_PROJECTION_SEEDS = (0,)
SMOKE_PAIR_COUNT = 2
ENDPOINT_TOPK_VALUES = (5, 10, 30)
FALSE_ELITE_K = 30


def parse_int_list(value: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    items = tuple(int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--stage1b-path", type=Path, default=DEFAULT_STAGE1B_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pool-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--dimensions", type=parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--projection-seeds", type=parse_int_list, default=DEFAULT_PROJECTION_SEEDS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--action-block", type=int, default=ACTION_BLOCK)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--cem-iters", type=int, default=DEFAULT_CEM_ITERS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--var-scale", type=float, default=VAR_SCALE)
    parser.add_argument("--resume", action="store_true", help="Explicitly enable resume behavior; this is the default.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.pair_count < 1:
        parser.error("--pair-count must be positive")
    if args.action_block <= 0:
        parser.error("--action-block must be positive")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.topk < 1 or args.topk > args.num_samples:
        parser.error("--topk must be in [1, --num-samples]")
    if args.cem_iters < 1:
        parser.error("--cem-iters must be positive")
    if any(int(dim) <= 0 or int(dim) > LATENT_DIM for dim in args.dimensions):
        parser.error(f"--dimensions values must be in [1, {LATENT_DIM}]")
    if any(int(seed) < 0 for seed in args.projection_seeds):
        parser.error("--projection-seeds must be nonnegative integers")
    return args


def resolve_planning_device(requested: str) -> str:
    requested = str(requested)
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def euclidean_costs(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    return torch.sum((z_pred - z_goal) ** 2, dim=-1)


def projected_costs(z_pred: torch.Tensor, z_goal: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    projection = projection.to(device=z_pred.device, dtype=z_pred.dtype)
    pred_proj = z_pred @ projection
    goal_proj = z_goal @ projection
    return torch.sum((pred_proj - goal_proj) ** 2, dim=-1)


def run_projected_cem(
    *,
    model,
    prepared_info: dict[str, Any],
    pair_id: int,
    seed: int,
    projection: torch.Tensor,
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
        select_cost = projected_costs(z_pred, z_goal, projection)
        top_vals, top_inds = torch.topk(select_cost, k=int(topk), dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]

        if iter_idx == int(cem_iters):
            select_np = select_cost[0].detach().cpu().numpy().astype(np.float64)
            default_np = default_cost[0].detach().cpu().numpy().astype(np.float64)
            top_np = top_vals[0].detach().cpu().numpy().astype(np.float64)
            top_default_np = default_np[top_inds[0].detach().cpu().numpy().astype(np.int64)]
            final = {
                "blocked_candidates": candidates[0].detach().cpu().numpy().astype(np.float32),
                "rank1_blocked": elite_candidates[0, 0].detach().cpu().numpy().astype(np.float32),
                "rank1_candidate_index": int(top_inds[0, 0].detach().cpu().item()),
                "select_costs": select_np,
                "default_costs": default_np,
                "z_pred": z_pred[0].detach().cpu().numpy().astype(np.float32),
                "z_goal": z_goal[0].detach().cpu().numpy().astype(np.float32),
                "top30_select_costs": top_np,
                "top30_default_costs_at_projected_elites": top_default_np.astype(np.float64),
                "top30_select_cost_std": clean_float(float(np.std(top_np, ddof=0))),
                "top30_default_cost_std_at_projected_elites": clean_float(float(np.std(top_default_np, ddof=0))),
                "select_cost_dynamic_range": clean_float(float(np.max(select_np) - np.min(select_np))),
                "select_cost_min": clean_float(float(np.min(select_np))),
                "select_cost_max": clean_float(float(np.max(select_np))),
                "default_cost_dynamic_range": clean_float(float(np.max(default_np) - np.min(default_np))),
                "default_cost_min": clean_float(float(np.min(default_np))),
                "default_cost_max": clean_float(float(np.max(default_np))),
                "default_cost_std": clean_float(float(np.std(default_np, ddof=0))),
            }

        mean = elite_candidates.mean(dim=1)
        var = elite_candidates.std(dim=1)

    if final is None:
        raise RuntimeError("Projected CEM final iteration was not captured")
    final["wallclock_seconds"] = clean_float(time.time() - started)
    return final


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_stage1b_reference(
    path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int, int], dict[str, Any]], dict[str, Any]]:
    data = load_json(path)
    if data.get("metadata", {}).get("format") != "cube_stage1b_rerank_projected_cem":
        raise RuntimeError(f"Unexpected Stage 1B reference format: {data.get('metadata', {}).get('format')!r}")
    baselines = {int(record["pair_id"]): record for record in data.get("default_baselines", [])}
    endpoint_records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for record in data.get("records", []):
        endpoint_records[record_key(record)] = record
    return baselines, endpoint_records, data.get("metadata", {})


def validate_reference_coverage(
    *,
    requested_pairs: list[dict[str, Any]],
    dimensions: tuple[int, ...],
    projection_seeds: tuple[int, ...],
    baselines: dict[int, dict[str, Any]],
    endpoint_records: dict[tuple[int, int, int], dict[str, Any]],
) -> None:
    missing_baselines = [int(pair["pair_id"]) for pair in requested_pairs if int(pair["pair_id"]) not in baselines]
    if missing_baselines:
        raise RuntimeError(f"Stage 1B reference is missing default baselines for pair_ids: {missing_baselines[:10]}")

    missing_endpoint = []
    for pair in requested_pairs:
        pair_id = int(pair["pair_id"])
        for dim in dimensions:
            for projection_seed in projection_seeds:
                key = (pair_id, int(dim), int(projection_seed))
                if key not in endpoint_records:
                    missing_endpoint.append(key)
    if missing_endpoint:
        raise RuntimeError(
            "Stage 1B reference is missing projection-specific endpoint metrics for "
            f"{len(missing_endpoint)} requested keys; examples={missing_endpoint[:10]}"
        )


def record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (int(record["pair_id"]), int(record["dimension"]), int(record["projection_seed"]))


def pool_filename(pair_id: int, dim: int, projection_seed: int) -> str:
    return f"pair_{int(pair_id):03d}_m{int(dim):03d}_seed_{int(projection_seed)}.pt"


def deterministic_argmin(costs: np.ndarray, candidate_ids: np.ndarray) -> int:
    order = np.lexsort((candidate_ids, costs))
    return int(order[0])


def save_pool_artifact(
    *,
    path: Path,
    pair_spec: dict[str, Any],
    dim: int,
    projection_seed: int,
    cem_sampling_seed: int,
    projected: dict[str, Any],
    raw_actions: np.ndarray,
    v1_costs: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    scored_metrics: list[dict[str, Any]],
    candidate_indices: np.ndarray,
    args: argparse.Namespace,
    raw_action_dim: int,
) -> None:
    rank1_idx = int(projected["rank1_candidate_index"])
    best_real_idx = deterministic_argmin(np.asarray(c_real_state, dtype=np.float64), candidate_indices)
    payload = {
        "metadata": {
            "format": "cube_full_projected_cem_extended_pool_v1",
            "created_at": iso_now(),
            "script_path": str(Path(__file__).resolve()),
            "pair_id": int(pair_spec["pair_id"]),
            "cell": str(pair_spec["cell"]),
            "dimension": int(dim),
            "projection_seed": int(projection_seed),
            "cem_sampling_seed": int(cem_sampling_seed),
            "candidate_pool_original_n": int(args.num_samples),
            "candidate_pool_scored_n": int(len(candidate_indices)),
            "success_definition": f"cube_pos_dist <= {CUBE_SUCCESS_THRESHOLD_M:.2f}m",
            "v1_cost_definition": "Cube v1_hinge_costs alias c_real_state / cube_pos_dist.",
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
                "sampling_seed_rule": "base_seed + pair_id * 1009",
                "candidate_0_forced_to_search_mean": True,
            },
        },
        "pair_spec": pair_spec,
        "z_pred": torch.from_numpy(np.asarray(projected["z_pred"], dtype=np.float32)),
        "z_goal": torch.from_numpy(np.asarray(projected["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.from_numpy(np.asarray(projected["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.from_numpy(np.asarray(raw_actions, dtype=np.float32)),
        "default_costs": torch.from_numpy(np.asarray(projected["default_costs"], dtype=np.float32)),
        "projected_costs": torch.from_numpy(np.asarray(projected["select_costs"], dtype=np.float32)),
        "v1_hinge_costs": torch.from_numpy(np.asarray(v1_costs, dtype=np.float32)),
        "v1_costs": torch.from_numpy(np.asarray(v1_costs, dtype=np.float32)),
        "c_real_state": torch.from_numpy(np.asarray(c_real_state, dtype=np.float32)),
        "cube_pos_dist": torch.from_numpy(np.asarray(c_real_state, dtype=np.float32)),
        "success": torch.from_numpy(np.asarray(success, dtype=bool)),
        "candidate_indices": torch.from_numpy(np.asarray(candidate_indices, dtype=np.int64)),
        "candidate_metrics": jsonable(scored_metrics),
        "rank1_candidate_index": rank1_idx,
        "oracle_best_candidate_index": best_real_idx,
        "rank1_c_real_state": clean_float(float(np.asarray(c_real_state, dtype=np.float64)[rank1_idx])),
        "oracle_best_c_real_state": clean_float(float(np.min(np.asarray(c_real_state, dtype=np.float64)))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_projected_record(
    *,
    pair_spec: dict[str, Any],
    dim: int,
    projection_seed: int,
    cem_seed: int,
    projected: dict[str, Any],
    raw_actions: np.ndarray,
    v1_costs: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    scored_metrics: list[dict[str, Any]],
    candidate_indices: np.ndarray,
    default_baseline: dict[str, Any],
    endpoint_reference_record: dict[str, Any],
    pool_path: Path,
) -> dict[str, Any]:
    pair_id = int(pair_spec["pair_id"])
    cell = str(pair_spec["cell"])
    projected_costs_np = np.asarray(projected["select_costs"], dtype=np.float64)
    default_costs_np = np.asarray(projected["default_costs"], dtype=np.float64)
    v1_costs_np = np.asarray(v1_costs, dtype=np.float64)
    c_real_state_np = np.asarray(c_real_state, dtype=np.float64)
    success_np = np.asarray(success, dtype=bool)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

    rank1_idx = int(projected["rank1_candidate_index"])
    rank1_pos = int(np.flatnonzero(candidate_indices == rank1_idx)[0])
    best_real_pos = deterministic_argmin(c_real_state_np, candidate_indices)
    top30_order = np.lexsort((candidate_indices, projected_costs_np))[:FALSE_ELITE_K]

    endpoint_metrics = endpoint_ranking_metrics(
        projected_costs=projected_costs_np,
        default_costs=default_costs_np,
        v1_costs=v1_costs_np,
        c_real_state=c_real_state_np,
        success=success_np,
        candidate_ids=candidate_indices,
    )
    rpool_cmodel = clean_float(spearman_corr(default_costs_np, c_real_state_np))
    rpool_v1 = clean_float(spearman_corr(v1_costs_np, c_real_state_np))
    rpool_projected = endpoint_metrics["endpoint_spearman"]

    default_blocked_rank1 = np.asarray(default_baseline["rank1_blocked_action"], dtype=np.float32)
    default_raw_rank1 = np.asarray(default_baseline["rank1_raw_action"], dtype=np.float32)
    projected_blocked_rank1 = np.asarray(projected["rank1_blocked"], dtype=np.float32)
    projected_raw_rank1 = np.asarray(raw_actions[rank1_pos], dtype=np.float32)
    endpoint_reference_metrics = endpoint_reference_record["endpoint_metrics_on_default_pool"]

    return {
        "pair_id": pair_id,
        "cell": cell,
        "dimension": int(dim),
        "projection_seed": int(projection_seed),
        "cem_sampling_seed": int(cem_seed),
        "candidate_scoring_mode": "full_projected_cem_all_300",
        "candidate_pool_original_n": int(len(projected_costs_np)),
        "candidate_pool_scored_n": int(len(candidate_indices)),
        "pool_path": str(pool_path),
        "projection_matrix": {
            "shape": [LATENT_DIM, int(dim)],
            "scale": f"1/sqrt({int(dim)})",
            "fixed_scope": "fixed per (dimension, projection_seed) across pairs and CEM iterations",
        },
        "default_rank1_success": bool(default_baseline["rank1_success"]),
        "default_rank1_v1_cost": default_baseline["rank1_v1_cost"],
        "default_rank1_c_real_state": default_baseline["rank1_c_real_state"],
        "rank1_candidate_index": rank1_idx,
        "rank1_scored_pool_position": rank1_pos,
        "rank1_success": bool(success_np[rank1_pos]),
        "planning_success": bool(success_np[rank1_pos]),
        "rank1_v1_cost": clean_float(float(v1_costs_np[rank1_pos])),
        "rank1_c_real_state": clean_float(float(c_real_state_np[rank1_pos])),
        "rank1_state_metrics": scored_metrics[rank1_pos],
        "oracle_best_candidate_index": int(candidate_indices[best_real_pos]),
        "oracle_best_scored_pool_position": int(best_real_pos),
        "oracle_best_c_real_state": clean_float(float(c_real_state_np[best_real_pos])),
        "selection_regret": clean_float(float(c_real_state_np[rank1_pos] - np.min(c_real_state_np))),
        "Rendpoint": endpoint_reference_metrics.get("endpoint_spearman"),
        "Rpool": rpool_projected,
        "Rpool_projected": rpool_projected,
        "Rpool_Cmodel": rpool_cmodel,
        "Rpool_V1": rpool_v1,
        "endpoint_metrics_on_default_pool": endpoint_reference_metrics,
        "pool_ranking_metrics_on_projected_pool": endpoint_metrics,
        "pool_diagnostics": {
            "pool_Creal_std": clean_float(float(np.std(c_real_state_np, ddof=0))),
            "pool_Creal_range": clean_float(float(np.max(c_real_state_np) - np.min(c_real_state_np))),
            "pool_Creal_min": clean_float(float(np.min(c_real_state_np))),
            "pool_Creal_max": clean_float(float(np.max(c_real_state_np))),
            "pool_success_mass": clean_float(float(np.mean(success_np))),
            "pool_Cmodel_std": clean_float(float(np.std(default_costs_np, ddof=0))),
            "pool_Cmodel_range": clean_float(float(np.max(default_costs_np) - np.min(default_costs_np))),
            "pool_projected_cost_std": clean_float(float(np.std(projected_costs_np, ddof=0))),
            "pool_projected_cost_range": clean_float(float(np.max(projected_costs_np) - np.min(projected_costs_np))),
            "top30_Creal_std": clean_float(float(np.std(c_real_state_np[top30_order], ddof=0))),
            "top30_projected_cost_std": clean_float(float(np.std(projected_costs_np[top30_order], ddof=0))),
            "top30_Cmodel_std": clean_float(float(np.std(default_costs_np[top30_order], ddof=0))),
        },
        "projected_cem_diagnostics": {
            "planning_mode": "full_projected_cem",
            "rank1_candidate_index": rank1_idx,
            "rank1_projected_cost": clean_float(float(projected_costs_np[rank1_idx])),
            "rank1_default_latent_cost": clean_float(float(default_costs_np[rank1_idx])),
            "default_rank1_candidate_index": int(default_baseline["rank1_candidate_index"]),
            "default_rank1_success": bool(default_baseline["rank1_success"]),
            "default_rank1_c_real_state": default_baseline["rank1_c_real_state"],
            "final_top30_elite_cost_std": projected["top30_select_cost_std"],
            "final_top30_default_cost_std_at_projected_elites": projected["top30_default_cost_std_at_projected_elites"],
            "final_candidate_dynamic_range": projected["select_cost_dynamic_range"],
            "final_projected_cost_min": projected["select_cost_min"],
            "final_projected_cost_max": projected["select_cost_max"],
            "final_default_latent_cost_min": projected["default_cost_min"],
            "final_default_latent_cost_max": projected["default_cost_max"],
            "final_default_latent_cost_dynamic_range": projected["default_cost_dynamic_range"],
            "final_default_latent_cost_std": projected["default_cost_std"],
            "action_l2_to_default_blocked": clean_float(
                float(np.linalg.norm(projected_blocked_rank1 - default_blocked_rank1))
            ),
            "action_l2_to_default_raw": clean_float(float(np.linalg.norm(projected_raw_rank1 - default_raw_rank1))),
            "wallclock_seconds": projected["wallclock_seconds"],
        },
        "rank1_blocked_action": projected_blocked_rank1,
        "rank1_raw_action": projected_raw_rank1,
    }


def aggregate_record_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "planning_success_rate": scalar_summary([record.get("planning_success") for record in records]),
        "rank1_c_real_state": scalar_summary([record.get("rank1_c_real_state") for record in records]),
        "selection_regret": scalar_summary([record.get("selection_regret") for record in records]),
        "Rendpoint": scalar_summary([record.get("Rendpoint") for record in records]),
        "Rpool": scalar_summary([record.get("Rpool") for record in records]),
        "Rpool_projected": scalar_summary([record.get("Rpool_projected") for record in records]),
        "Rpool_Cmodel": scalar_summary([record.get("Rpool_Cmodel") for record in records]),
        "Rpool_V1": scalar_summary([record.get("Rpool_V1") for record in records]),
        "pool_Creal_std": scalar_summary([nested_get(record, ("pool_diagnostics", "pool_Creal_std")) for record in records]),
        "pool_success_mass": scalar_summary(
            [nested_get(record, ("pool_diagnostics", "pool_success_mass")) for record in records]
        ),
        "pool_Cmodel_std": scalar_summary(
            [nested_get(record, ("pool_diagnostics", "pool_Cmodel_std")) for record in records]
        ),
        "pool_projected_cost_std": scalar_summary(
            [nested_get(record, ("pool_diagnostics", "pool_projected_cost_std")) for record in records]
        ),
        "top30_Creal_std": scalar_summary(
            [nested_get(record, ("pool_diagnostics", "top30_Creal_std")) for record in records]
        ),
        "top30_projected_cost_std": scalar_summary(
            [nested_get(record, ("pool_diagnostics", "top30_projected_cost_std")) for record in records]
        ),
        "wallclock_seconds": scalar_summary(
            [nested_get(record, ("projected_cem_diagnostics", "wallclock_seconds")) for record in records]
        ),
    }


def aggregate_default_baselines(default_baselines: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(default_baselines)),
        "rank1_success_rate": scalar_summary([record.get("rank1_success") for record in default_baselines]),
        "rank1_c_real_state": scalar_summary([record.get("rank1_c_real_state") for record in default_baselines]),
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
    expected_records: int,
) -> dict[str, Any]:
    return {
        "expected_projected_records": int(expected_records),
        "observed_projected_records": int(len(records)),
        "completed_fraction": clean_float(float(len(records) / expected_records)) if expected_records else None,
        "default_baselines": aggregate_default_baselines(default_baselines),
        "overall": aggregate_record_group(records),
        "by_dimension": {
            str(int(dim)): aggregate_record_group([record for record in records if int(record["dimension"]) == int(dim)])
            for dim in dimensions
        },
        "by_projection_seed": {
            str(int(seed)): aggregate_record_group(
                [record for record in records if int(record["projection_seed"]) == int(seed)]
            )
            for seed in projection_seeds
        },
        "by_dimension_and_projection_seed": {
            str(int(dim)): {
                str(int(seed)): aggregate_record_group(
                    [
                        record
                        for record in records
                        if int(record["dimension"]) == int(dim) and int(record["projection_seed"]) == int(seed)
                    ]
                )
                for seed in projection_seeds
            }
            for dim in dimensions
        },
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


def output_metadata(
    *,
    args: argparse.Namespace,
    selected_pairs: list[dict[str, Any]],
    dimensions: tuple[int, ...],
    projection_seeds: tuple[int, ...],
    projections: dict[tuple[int, int], torch.Tensor],
    pairs_metadata: dict[str, Any],
    stage1b_metadata: dict[str, Any],
    raw_action_dim: int,
    total_started: float,
    completed: bool,
) -> dict[str, Any]:
    n_pairs = len(selected_pairs)
    n_records = int(n_pairs * len(dimensions) * len(projection_seeds))
    return {
        "format": "cube_full_projected_cem_extended",
        "created_at": iso_now(),
        "git_commit": get_git_commit(),
        "script_path": str(Path(__file__).resolve()),
        "planning_mode": "full_projected_cem",
        "mode": "smoke" if args.smoke else "full_50pair_extension",
        "description": (
            "Projected costs are used for elite selection at every Cube CEM iteration. "
            "Each projected-CEM final pool is simulator-scored for all candidates and saved as a .pt file."
        ),
        "completed": bool(completed),
        "output": str(args.output),
        "pool_dir": str(args.pool_dir),
        "pairs_path": str(args.pairs_path),
        "stage1b_reference_path": str(args.stage1b_path),
        "stage1b_reference_format": stage1b_metadata.get("format"),
        "selected_pair_ids": [int(pair["pair_id"]) for pair in selected_pairs],
        "n_selected_pairs": int(n_pairs),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_cache_dir": str(args.cache_dir),
        "dataset_name": args.dataset_name,
        "device": args.device,
        "seed": int(args.seed),
        "dimensions": [int(dim) for dim in dimensions],
        "projection_seeds": [int(seed) for seed in projection_seeds],
        "expected_projected_records": n_records,
        "expected_simulator_rollouts": {
            "projected_final_candidate_pool_new": int(n_records * args.num_samples),
            "candidate_pool_scored_n_per_record": int(args.num_samples),
        },
        "candidate_scoring_mode": "full_projected_cem_all_300",
        "candidate_pool_original_n": int(args.num_samples),
        "candidate_pool_scored_n": int(args.num_samples),
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
            "fixed_scope": "one matrix per (m, projection_seed), reused across all pairs and CEM iterations",
            "matrices": projection_metadata(projections),
        },
        "endpoint_metric_pool": {
            "source": "loaded from cube_stage1b.json records for matching (pair, m, projection_seed)",
            "default_pool_source": "cube_stage1b.json default_baselines candidate_pool",
            "topk_values": list(ENDPOINT_TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
        },
        "runtime": {
            "wallclock_seconds": clean_float(time.time() - total_started),
            "wallclock_hms": seconds_to_hms(time.time() - total_started),
        },
    }


def load_existing_output(path: Path, *, resume: bool) -> dict[str, Any] | None:
    if not resume or not path.exists():
        return None
    data = load_json(path)
    if data.get("metadata", {}).get("format") != "cube_full_projected_cem_extended":
        raise RuntimeError(f"Cannot resume from unexpected output format: {data.get('metadata', {}).get('format')!r}")
    return data


def write_output(
    *,
    args: argparse.Namespace,
    selected_pairs: list[dict[str, Any]],
    dimensions: tuple[int, ...],
    projection_seeds: tuple[int, ...],
    projections: dict[tuple[int, int], torch.Tensor],
    pairs_metadata: dict[str, Any],
    stage1b_metadata: dict[str, Any],
    raw_action_dim: int,
    default_baselines: list[dict[str, Any]],
    records: list[dict[str, Any]],
    total_started: float,
    completed: bool,
) -> None:
    records = sorted(
        records,
        key=lambda record: (
            int(record["pair_id"]),
            int(record["dimension"]),
            int(record["projection_seed"]),
        ),
    )
    cells = tuple(sorted({str(pair["cell"]) for pair in selected_pairs}))
    expected_records = int(len(selected_pairs) * len(dimensions) * len(projection_seeds))
    output = {
        "metadata": output_metadata(
            args=args,
            selected_pairs=selected_pairs,
            dimensions=dimensions,
            projection_seeds=projection_seeds,
            projections=projections,
            pairs_metadata=pairs_metadata,
            stage1b_metadata=stage1b_metadata,
            raw_action_dim=raw_action_dim,
            total_started=total_started,
            completed=completed,
        ),
        "default_baselines": default_baselines,
        "records": records,
        "aggregate": aggregate_records(
            records=records,
            default_baselines=default_baselines,
            dimensions=dimensions,
            projection_seeds=projection_seeds,
            cells=cells,
            expected_records=expected_records,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")


def print_summary_table(aggregate: dict[str, Any]) -> None:
    rows = []
    for dim, stats in sorted(aggregate["by_dimension"].items(), key=lambda item: int(item[0])):
        rows.append(
            [
                f"m={dim}",
                str(stats["n_records"]),
                fmt(nested_get(stats, ("planning_success_rate", "mean"))),
                fmt(nested_get(stats, ("Rendpoint", "mean"))),
                fmt(nested_get(stats, ("Rpool", "mean"))),
                fmt(nested_get(stats, ("Rpool_Cmodel", "mean"))),
                fmt(nested_get(stats, ("pool_Creal_std", "mean"))),
                fmt(nested_get(stats, ("wallclock_seconds", "mean"))),
            ]
        )
    headers = ["Dim", "N", "Success", "Rendpoint", "Rpool", "Rpool(Cmodel)", "CRealStd", "CEMSec"]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print("Cube extended full projected-CEM summary")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output = args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)
    args.pool_dir = args.pool_dir or (SMOKE_POOL_DIR if args.smoke else DEFAULT_POOL_DIR)
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.stage1b_path = args.stage1b_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pool_dir = args.pool_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.device = resolve_planning_device(args.device)

    dimensions = SMOKE_DIMS if args.smoke else tuple(int(dim) for dim in args.dimensions)
    projection_seeds = SMOKE_PROJECTION_SEEDS if args.smoke else tuple(int(seed) for seed in args.projection_seeds)
    pair_count = SMOKE_PAIR_COUNT if args.smoke else int(args.pair_count)

    pairs_data, all_pairs = load_pairs(args.pairs_path, max_pairs=None, pair_ids=None)
    all_pairs = sorted(all_pairs, key=lambda pair: int(pair["pair_id"]))
    pairs_metadata = pairs_data.get("metadata", {})
    offset = int(pairs_metadata.get("offset", all_pairs[0]["goal_row"] - all_pairs[0]["start_row"]))
    if offset % int(args.action_block) != 0:
        raise ValueError("Cube offset must be divisible by --action-block")
    validate_requested_pair_offsets(all_pairs, offset=offset)
    args.horizon_blocks = int(offset // int(args.action_block))
    selected_pairs = all_pairs[:pair_count]

    stage1b_baselines, endpoint_reference_records, stage1b_metadata = load_stage1b_reference(args.stage1b_path)
    validate_reference_coverage(
        requested_pairs=selected_pairs,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        baselines=stage1b_baselines,
        endpoint_records=endpoint_reference_records,
    )
    default_baselines = [stage1b_baselines[int(pair["pair_id"])] for pair in selected_pairs]

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    raw_action_dim = infer_raw_action_dim(dataset)
    if raw_action_dim != 5:
        raise ValueError(f"Expected Cube raw action dim 5, got {raw_action_dim}")
    process = build_processors(dataset, ["action"])
    policy = build_policy(make_policy_namespace(args), process)
    model = policy.solver.model
    action_processor = policy.process["action"]

    projections = {
        (int(dim), int(seed)): make_projection(int(dim), int(seed))
        for dim in dimensions
        for seed in projection_seeds
    }

    expected_keys = {
        (int(pair["pair_id"]), int(dim), int(seed))
        for pair in selected_pairs
        for dim in dimensions
        for seed in projection_seeds
    }
    existing = load_existing_output(args.output, resume=not args.no_resume)
    records: list[dict[str, Any]] = list(existing.get("records", [])) if existing else []
    records = [record for record in records if record_key(record) in expected_keys]
    seen = {record_key(record) for record in records}

    print("== Cube extended full projected-CEM setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"stage1b_reference: {args.stage1b_path}")
    print(f"output: {args.output}")
    print(f"pool_dir: {args.pool_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"device: {args.device}")
    print(f"mode: {'smoke' if args.smoke else 'full_50pair_extension'}")
    print(f"seed: {args.seed}")
    print(f"pairs: {len(selected_pairs)}")
    print(f"offset: {offset}")
    print(f"horizon_blocks: {args.horizon_blocks}")
    print(f"raw_action_dim: {raw_action_dim}")
    print(f"blocked_action_dim: {raw_action_dim * args.action_block}")
    print(f"dimensions: {list(dimensions)}")
    print(f"projection_seeds: {list(projection_seeds)}")
    print(f"resume_records_loaded: {len(seen)}")
    print(f"expected_projected_records: {len(expected_keys)}")
    print(f"expected_new_simulator_rollouts_if_empty: {len(expected_keys) * int(args.num_samples)}")

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
        for pair_idx, pair_spec in enumerate(selected_pairs, start=1):
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
            default_baseline = stage1b_baselines[pair_id]

            print(f"\n[{pair_idx}/{len(selected_pairs)}] pair_id={pair_id} cell={cell}")
            pair_started_n = len(records)
            pair_started = time.time()
            for dim in dimensions:
                for projection_seed in projection_seeds:
                    key = (pair_id, int(dim), int(projection_seed))
                    if key in seen:
                        print(f"  m={dim} projection_seed={projection_seed}: loaded existing record")
                        continue

                    record_started = time.time()
                    print(f"  m={dim} projection_seed={projection_seed}: full projected CEM + score 300")
                    projected = run_projected_cem(
                        model=model,
                        prepared_info=prepared_info,
                        pair_id=pair_id,
                        seed=int(args.seed),
                        projection=projections[(int(dim), int(projection_seed))],
                        horizon_blocks=int(args.horizon_blocks),
                        action_dim=int(args.action_block) * int(raw_action_dim),
                        num_samples=int(args.num_samples),
                        cem_iters=int(args.cem_iters),
                        topk=int(args.topk),
                        var_scale=float(args.var_scale),
                    )
                    raw_candidates = blocked_batch_to_raw(
                        np.asarray(projected["blocked_candidates"], dtype=np.float32),
                        action_processor=action_processor,
                        action_block=int(args.action_block),
                        raw_action_dim=int(raw_action_dim),
                    )
                    candidate_indices = np.arange(int(raw_candidates.shape[0]), dtype=np.int64)
                    rollout_seed_base = int(args.seed) + pair_id * 100_000 + int(dim) * 1_000 + int(projection_seed)
                    v1_costs, c_real_state, success, scored_metrics = score_raw_actions(
                        env=env,
                        initial=initial,
                        goal=goal,
                        goal_pos=goal_pos,
                        goal_quat=goal_quat,
                        raw_actions_batch=raw_candidates,
                        candidate_indices=candidate_indices,
                        seed_base=rollout_seed_base,
                    )
                    pool_path = args.pool_dir / pool_filename(pair_id, int(dim), int(projection_seed))
                    save_pool_artifact(
                        path=pool_path,
                        pair_spec=pair_spec,
                        dim=int(dim),
                        projection_seed=int(projection_seed),
                        cem_sampling_seed=cem_seed,
                        projected=projected,
                        raw_actions=raw_candidates,
                        v1_costs=v1_costs,
                        c_real_state=c_real_state,
                        success=success,
                        scored_metrics=scored_metrics,
                        candidate_indices=candidate_indices,
                        args=args,
                        raw_action_dim=raw_action_dim,
                    )
                    record = build_projected_record(
                        pair_spec=pair_spec,
                        dim=int(dim),
                        projection_seed=int(projection_seed),
                        cem_seed=cem_seed,
                        projected=projected,
                        raw_actions=raw_candidates,
                        v1_costs=v1_costs,
                        c_real_state=c_real_state,
                        success=success,
                        scored_metrics=scored_metrics,
                        candidate_indices=candidate_indices,
                        default_baseline=default_baseline,
                        endpoint_reference_record=endpoint_reference_records[key],
                        pool_path=pool_path,
                    )
                    records.append(record)
                    seen.add(key)
                    write_output(
                        args=args,
                        selected_pairs=selected_pairs,
                        dimensions=dimensions,
                        projection_seeds=projection_seeds,
                        projections=projections,
                        pairs_metadata=pairs_metadata,
                        stage1b_metadata=stage1b_metadata,
                        raw_action_dim=raw_action_dim,
                        default_baselines=default_baselines,
                        records=records,
                        total_started=total_started,
                        completed=False,
                    )
                    print(
                        f"    success={bool(record['planning_success'])}; "
                        f"rank1_c_real={float(record['rank1_c_real_state']):.4f}; "
                        f"Rendpoint={fmt(record['Rendpoint'])}; "
                        f"Rpool={fmt(record['Rpool'])}; "
                        f"pool_Creal_std={fmt(record['pool_diagnostics']['pool_Creal_std'])}; "
                        f"elapsed={seconds_to_hms(time.time() - record_started)}"
                    )
            pair_new = len(records) - pair_started_n
            aggregate_so_far = aggregate_record_group(records)
            print(
                f"  pair complete: new_records={pair_new}; "
                f"pair_elapsed={seconds_to_hms(time.time() - pair_started)}; "
                f"overall_success={fmt(nested_get(aggregate_so_far, ('planning_success_rate', 'mean')))}; "
                f"overall_Rpool={fmt(nested_get(aggregate_so_far, ('Rpool', 'mean')))}"
            )
    finally:
        env.close()

    write_output(
        args=args,
        selected_pairs=selected_pairs,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        projections=projections,
        pairs_metadata=pairs_metadata,
        stage1b_metadata=stage1b_metadata,
        raw_action_dim=raw_action_dim,
        default_baselines=default_baselines,
        records=records,
        total_started=total_started,
        completed=len(seen) == len(expected_keys),
    )
    output = load_json(args.output)
    aggregate = output["aggregate"]
    print()
    print_summary_table(aggregate)
    print(f"\nWallclock seconds: {time.time() - total_started:.3f} ({seconds_to_hms(time.time() - total_started)})")
    print(f"Saved: {args.output}")
    print(f"Pools: {args.pool_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
