#!/usr/bin/env python3
"""Stage 1B-Smoke projected-cost CEM runner.

This diagnostic runs the normal LeWM CEM planner once per selected pair, scores
only that final 300-candidate pool with the simulator, and then evaluates
random-projection CEM planners against that same labelled pool. Projected CEM
success is measured by executing only the projected rank-1 final action.
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lewm_audit.diagnostics.three_cost import (  # noqa: E402
    block_pose_metrics,
    prepare_pair_info,
)
from lewm_audit.eval.oracle_cem import cost_v1_hinge, rollout_final_state  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    TOPK,
    VAR_SCALE,
)
from scripts.phase2.stage1.c6_audit import (  # noqa: E402
    audit_anchor_definitions,
    build_common,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    EXPECTED_RECORDS,
    FALSE_ELITE_K,
    LATENT_DIM,
    TOPK_VALUES,
    clean_float,
    compute_metrics,
    deterministic_topk_indices,
    iso_now,
    jsonable,
    load_latent_artifact,
    pairwise_accuracy,
    spearman_corr,
    squared_l2_torch,
    summary_row,
)
from scripts.phase2.track_b_common import (  # noqa: E402
    add_replay_args,
    build_replay_context,
)
from scripts.phase2.train_cem_aware import rollout_candidate_latents  # noqa: E402


DEFAULT_OUTPUT_FULL = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1b_smoke.json"
DEFAULT_OUTPUT_SANITY = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1b_sanity.json"
DEFAULT_DIMS = (8, 32, 64, 192)
DEFAULT_PROJECTION_SEEDS = (0, 1, 2)
SANITY_PAIR_IDS = (0, 1, 2, 3, 4)
ORDINARY_SAMPLE_SIZE = 10
ENDPOINT_TOPK_VALUES = (5, 10, 30)
PROJECTION_ELITE_K = 30


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args(parser)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--dimensions", type=parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--projection-seeds", type=parse_int_list, default=DEFAULT_PROJECTION_SEEDS)
    parser.add_argument("--sanity-success-tolerance", type=float, default=0.20)
    parser.add_argument("--sanity-elite-std-ratio-min", type=float, default=0.50)
    parser.add_argument("--sanity-elite-std-ratio-max", type=float, default=2.00)
    parser.add_argument("--sanity-action-l2-max", type=float, default=5.00)
    args = parser.parse_args()
    if any(int(dim) <= 0 or int(dim) > LATENT_DIM for dim in args.dimensions):
        parser.error(f"--dimensions values must be in [1, {LATENT_DIM}]")
    if any(int(seed) < 0 for seed in args.projection_seeds):
        parser.error("--projection-seeds must be nonnegative integers")
    if float(args.reference_atol) < 0:
        parser.error("--reference-atol must be nonnegative")
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


def print_summary_table(aggregate: dict[str, Any]) -> None:
    rows = []
    for dim, stats in sorted(aggregate["by_dimension"].items(), key=lambda item: int(item[0])):
        rows.append(
            [
                f"m={dim}",
                str(stats["n_records"]),
                fmt(stats["projected_success_rate"]["mean"]),
                fmt(stats["endpoint_spearman"]["mean"]),
                fmt(stats["pairwise_accuracy"]["mean"]),
                fmt(stats["false_elite_rate"]["mean"]),
                fmt(stats["action_l2_to_default_blocked"]["mean"]),
            ]
        )
    headers = ["Dim", "N", "Success", "Spearman", "Pairwise", "FalseElite", "ActionL2"]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print("Stage 1B-Smoke projected CEM summary")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


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
) -> dict[str, Any]:
    candidate_ids = np.arange(int(projected_costs.shape[0]), dtype=np.int64)
    pair_ids = np.zeros(int(projected_costs.shape[0]), dtype=np.int64)
    pairwise = pairwise_accuracy(projected_costs, c_real_state, pair_ids)
    return {
        "n_candidates": int(projected_costs.shape[0]),
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
            float(np.std(np.sort(projected_costs, kind="mergesort")[:PROJECTION_ELITE_K], ddof=0))
        ),
    }


def euclidean_costs(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    return torch.sum((z_pred - z_goal) ** 2, dim=-1)


def projected_costs(
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    projection = projection.to(device=z_pred.device, dtype=z_pred.dtype)
    pred_proj = z_pred @ projection
    goal_proj = z_goal @ projection
    return torch.sum((pred_proj - goal_proj) ** 2, dim=-1)


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


def make_projection(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((LATENT_DIM, int(dim)), generator=generator, dtype=torch.float32) / math.sqrt(int(dim))


def projection_metadata(projections: dict[tuple[int, int], torch.Tensor]) -> dict[str, Any]:
    out = {}
    for (dim, seed), projection in sorted(projections.items()):
        key = f"m={dim},seed={seed}"
        out[key] = {
            "dimension": int(dim),
            "seed": int(seed),
            "shape": list(projection.shape),
            "scale": f"1/sqrt({int(dim)})",
            "mean": clean_float(float(projection.mean().item())),
            "std": clean_float(float(projection.std(unbiased=False).item())),
            "frobenius_norm": clean_float(float(torch.linalg.norm(projection).item())),
        }
    return out


def blocked_batch_to_raw_fast(blocked: np.ndarray, *, action_processor: Any) -> np.ndarray:
    blocked = np.asarray(blocked, dtype=np.float32)
    if blocked.ndim != 3 or blocked.shape[-1] != ACTION_BLOCK * 2:
        raise ValueError(f"Expected blocked shape (N,H,{ACTION_BLOCK * 2}), got {blocked.shape}")
    n_candidates, horizon, _ = blocked.shape
    normalized = blocked.reshape(n_candidates * horizon * ACTION_BLOCK, 2)
    raw = action_processor.inverse_transform(normalized).astype(np.float32)
    return raw.reshape(n_candidates, horizon * ACTION_BLOCK, 2)


def score_raw_actions(
    *,
    env,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions_batch: np.ndarray,
    seed_base: int,
    seed_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    raw_actions_batch = np.asarray(raw_actions_batch, dtype=np.float32)
    if seed_indices is None:
        seed_indices = np.arange(int(raw_actions_batch.shape[0]), dtype=np.int64)
    else:
        seed_indices = np.asarray(seed_indices, dtype=np.int64)
        if seed_indices.shape != (int(raw_actions_batch.shape[0]),):
            raise ValueError("seed_indices must provide one seed offset per candidate")

    v1_costs = np.empty((int(raw_actions_batch.shape[0]),), dtype=np.float64)
    c_real_state = np.empty_like(v1_costs)
    success = np.empty((int(raw_actions_batch.shape[0]),), dtype=bool)
    metrics: list[dict[str, Any]] = []
    for idx, raw_actions in enumerate(raw_actions_batch):
        terminal_state = rollout_final_state(
            env,
            initial_state,
            goal_state,
            raw_actions,
            seed=int(seed_base) + int(seed_indices[idx]),
        )
        pose = block_pose_metrics(terminal_state, goal_state)
        v1_cost = float(cost_v1_hinge(terminal_state, goal_state))
        v1_costs[idx] = v1_cost
        c_real_state[idx] = float(pose["c_real_state"])
        success[idx] = bool(pose["success"])
        metrics.append(
            {
                "candidate_index": int(idx),
                "seed": int(seed_base) + int(seed_indices[idx]),
                "v1_cost": clean_float(v1_cost),
                "c_real_state": clean_float(float(pose["c_real_state"])),
                "block_pos_dist": clean_float(float(pose["block_pos_dist"])),
                "angle_dist": clean_float(float(pose["angle_dist"])),
                "success": bool(pose["success"]),
            }
        )
    return v1_costs, c_real_state, success, metrics


def run_cem(
    *,
    model,
    prepared_info: dict[str, Any],
    pair_id: int,
    seed: int,
    projection: torch.Tensor | None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed) + int(pair_id) * 1009)
    mean = torch.zeros((1, PLANNING_HORIZON, ACTION_BLOCK * 2), dtype=torch.float32, device=device)
    var = VAR_SCALE * torch.ones((1, PLANNING_HORIZON, ACTION_BLOCK * 2), dtype=torch.float32, device=device)
    final: dict[str, Any] | None = None
    started = time.time()

    for iter_idx in range(1, CEM_ITERS + 1):
        candidates = torch.randn(
            1,
            NUM_SAMPLES,
            PLANNING_HORIZON,
            ACTION_BLOCK * 2,
            generator=generator,
            device=device,
        )
        candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        z_pred, z_goal = rollout_candidate_latents(model, prepared_info, candidates)
        default_cost = euclidean_costs(z_pred, z_goal)
        select_cost = default_cost if projection is None else projected_costs(z_pred, z_goal, projection)
        top_vals, top_inds = torch.topk(select_cost, k=TOPK, dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]

        if iter_idx == CEM_ITERS:
            select_np = select_cost[0].detach().cpu().numpy().astype(np.float64)
            default_np = default_cost[0].detach().cpu().numpy().astype(np.float64)
            top_np = top_vals[0].detach().cpu().numpy().astype(np.float64)
            final = {
                "blocked_candidates": candidates[0].detach().cpu().numpy().astype(np.float32),
                "rank1_blocked": elite_candidates[0, 0].detach().cpu().numpy().astype(np.float32),
                "rank1_candidate_index": int(top_inds[0, 0].detach().cpu().item()),
                "select_costs": select_np,
                "default_costs": default_np,
                "z_pred": z_pred[0].detach().cpu().numpy().astype(np.float32),
                "z_goal": z_goal[0].detach().cpu().numpy().astype(np.float32),
                "top30_select_costs": top_np,
                "top30_select_cost_std": clean_float(float(np.std(top_np, ddof=0))),
                "select_cost_dynamic_range": clean_float(float(np.max(select_np) - np.min(select_np))),
                "select_cost_min": clean_float(float(np.min(select_np))),
                "select_cost_max": clean_float(float(np.max(select_np))),
            }

        mean = elite_candidates.mean(dim=1)
        var = elite_candidates.std(dim=1)

    if final is None:
        raise RuntimeError("CEM final iteration was not captured")
    final["wallclock_seconds"] = clean_float(time.time() - started)
    return final


def pair_subsets(
    *,
    pair_id: int,
    pair_subset_ids: dict[str, set[int]],
) -> list[str]:
    return [name for name, ids in sorted(pair_subset_ids.items()) if int(pair_id) in ids]


def select_pair_ids(
    *,
    args: argparse.Namespace,
    common: dict[str, Any],
) -> tuple[list[int], str]:
    if args.pair_ids is not None:
        return sorted(int(item) for item in args.pair_ids), "manual_cli_pair_ids"
    if bool(args.sanity):
        return list(SANITY_PAIR_IDS), "sanity_ordinary_fixed_5"

    pair_ids = common["pair_ids"]
    masks = common["anchor_masks"]
    selected: set[int] = set()
    for name in ("invisible_quadrant", "sign_reversal", "latent_favorable", "v1_favorable"):
        selected.update(int(item) for item in np.unique(pair_ids[masks[name]]).tolist())
    ordinary_pairs = sorted(int(item) for item in np.unique(pair_ids[masks["ordinary"]]).tolist())
    selected.update(ordinary_pairs[:ORDINARY_SAMPLE_SIZE])
    return sorted(selected), "full_anchor_union_plus_ordinary10"


def build_pair_subset_ids(common: dict[str, Any]) -> dict[str, set[int]]:
    pair_ids = common["pair_ids"]
    return {
        name: set(int(item) for item in np.unique(pair_ids[mask]).tolist())
        for name, mask in common["anchor_masks"].items()
    }


def build_default_baseline_record(
    *,
    pair_id: int,
    cell: str,
    subsets: list[str],
    cem_seed: int,
    baseline: dict[str, Any],
    v1_costs: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    scored_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    rank1_idx = int(baseline["rank1_candidate_index"])
    return {
        "pair_id": int(pair_id),
        "cell": str(cell),
        "subsets": subsets,
        "cem_sampling_seed": int(cem_seed),
        "rank1_candidate_index": rank1_idx,
        "rank1_success": bool(success[rank1_idx]),
        "rank1_v1_cost": clean_float(float(v1_costs[rank1_idx])),
        "rank1_c_real_state": clean_float(float(c_real_state[rank1_idx])),
        "rank1_state_metrics": scored_metrics[rank1_idx],
        "final_top30_default_cost_std": baseline["top30_select_cost_std"],
        "final_default_cost_dynamic_range": baseline["select_cost_dynamic_range"],
        "final_default_cost_min": baseline["select_cost_min"],
        "final_default_cost_max": baseline["select_cost_max"],
        "rank1_blocked_action": baseline["rank1_blocked"],
        "candidate_pool": {
            "n_candidates": int(NUM_SAMPLES),
            "default_costs": baseline["default_costs"],
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
    subsets: list[str],
    dim: int,
    projection_seed: int,
    cem_seed: int,
    projection: torch.Tensor,
    baseline: dict[str, Any],
    baseline_raw_rank1: np.ndarray,
    baseline_v1_costs: np.ndarray,
    baseline_c_real_state: np.ndarray,
    baseline_success: np.ndarray,
    projected: dict[str, Any],
    projected_rollout_metrics: dict[str, Any],
    projected_raw_rank1: np.ndarray,
) -> dict[str, Any]:
    projected_pool_costs = numpy_projected_costs(
        z_pred=baseline["z_pred"],
        z_goal=baseline["z_goal"],
        projection=projection,
    )
    endpoint_metrics = endpoint_ranking_metrics(
        projected_costs=projected_pool_costs,
        default_costs=baseline["default_costs"],
        v1_costs=baseline_v1_costs,
        c_real_state=baseline_c_real_state,
        success=baseline_success,
    )
    rank1_idx = int(projected["rank1_candidate_index"])
    projected_final_costs = np.asarray(projected["select_costs"], dtype=np.float64)
    projected_final_default_costs = np.asarray(projected["default_costs"], dtype=np.float64)
    return {
        "pair_id": int(pair_id),
        "cell": str(cell),
        "subsets": subsets,
        "dimension": int(dim),
        "projection_seed": int(projection_seed),
        "cem_sampling_seed": int(cem_seed),
        "projection_matrix": {
            "shape": [LATENT_DIM, int(dim)],
            "scale": f"1/sqrt({int(dim)})",
            "fixed_scope": "fixed per (dimension, projection_seed) across pairs and CEM iterations",
        },
        "cem_late_success": bool(projected_rollout_metrics["success"]),
        "cem_late_v1_cost": projected_rollout_metrics["v1_cost"],
        "cem_late_c_real_state": projected_rollout_metrics["c_real_state"],
        "cem_late_state_metrics": projected_rollout_metrics,
        "endpoint_metrics_on_default_pool": endpoint_metrics,
        "projected_cem_diagnostics": {
            "rank1_candidate_index": rank1_idx,
            "rank1_projected_cost": clean_float(float(projected_final_costs[rank1_idx])),
            "rank1_default_latent_cost": clean_float(float(projected_final_default_costs[rank1_idx])),
            "final_top30_elite_cost_std": projected["top30_select_cost_std"],
            "final_candidate_dynamic_range": projected["select_cost_dynamic_range"],
            "final_projected_cost_min": projected["select_cost_min"],
            "final_projected_cost_max": projected["select_cost_max"],
            "final_default_latent_cost_min": clean_float(float(np.min(projected_final_default_costs))),
            "final_default_latent_cost_max": clean_float(float(np.max(projected_final_default_costs))),
            "final_default_latent_cost_dynamic_range": clean_float(
                float(np.max(projected_final_default_costs) - np.min(projected_final_default_costs))
            ),
            "action_l2_to_default_blocked": clean_float(
                float(np.linalg.norm(projected["rank1_blocked"] - baseline["rank1_blocked"]))
            ),
            "action_l2_to_default_raw": clean_float(float(np.linalg.norm(projected_raw_rank1 - baseline_raw_rank1))),
            "wallclock_seconds": projected["wallclock_seconds"],
        },
        "rank1_blocked_action": projected["rank1_blocked"],
        "rank1_raw_action": projected_raw_rank1,
    }


def aggregate_record_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "projected_success_rate": scalar_summary([record.get("cem_late_success") for record in records]),
        "endpoint_spearman": scalar_summary(
            [
                record["endpoint_metrics_on_default_pool"].get("endpoint_spearman")
                for record in records
            ]
        ),
        "pairwise_accuracy": scalar_summary(
            [
                record["endpoint_metrics_on_default_pool"].get("pairwise_accuracy")
                for record in records
            ]
        ),
        "false_elite_rate": scalar_summary(
            [
                record["endpoint_metrics_on_default_pool"].get("false_elite_rate")
                for record in records
            ]
        ),
        "topk_overlap_lewm": {
            str(k): scalar_summary(
                [
                    record["endpoint_metrics_on_default_pool"].get("topk_overlap_lewm", {}).get(str(k))
                    for record in records
                ]
            )
            for k in ENDPOINT_TOPK_VALUES
        },
        "topk_overlap_v1": {
            str(k): scalar_summary(
                [
                    record["endpoint_metrics_on_default_pool"].get("topk_overlap_v1", {}).get(str(k))
                    for record in records
                ]
            )
            for k in ENDPOINT_TOPK_VALUES
        },
        "projected_elite_cost_std": scalar_summary(
            [
                record["projected_cem_diagnostics"].get("final_top30_elite_cost_std")
                for record in records
            ]
        ),
        "projected_dynamic_range": scalar_summary(
            [
                record["projected_cem_diagnostics"].get("final_candidate_dynamic_range")
                for record in records
            ]
        ),
        "action_l2_to_default_blocked": scalar_summary(
            [
                record["projected_cem_diagnostics"].get("action_l2_to_default_blocked")
                for record in records
            ]
        ),
        "action_l2_to_default_raw": scalar_summary(
            [
                record["projected_cem_diagnostics"].get("action_l2_to_default_raw")
                for record in records
            ]
        ),
    }


def aggregate_default_baselines(default_baselines: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(default_baselines)),
        "rank1_success_rate": scalar_summary([record.get("rank1_success") for record in default_baselines]),
        "rank1_c_real_state": scalar_summary([record.get("rank1_c_real_state") for record in default_baselines]),
        "rank1_v1_cost": scalar_summary([record.get("rank1_v1_cost") for record in default_baselines]),
        "candidate_pool_success_rate": scalar_summary(
            [record.get("candidate_pool", {}).get("success_rate") for record in default_baselines]
        ),
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
    pair_subset_ids: dict[str, set[int]],
) -> dict[str, Any]:
    by_dimension = {
        str(int(dim)): aggregate_record_group(
            [record for record in records if int(record["dimension"]) == int(dim)]
        )
        for dim in dimensions
    }
    by_projection_seed = {
        str(int(seed)): aggregate_record_group(
            [record for record in records if int(record["projection_seed"]) == int(seed)]
        )
        for seed in projection_seeds
    }
    by_subset = {
        name: aggregate_record_group(
            [record for record in records if int(record["pair_id"]) in pair_ids]
        )
        for name, pair_ids in sorted(pair_subset_ids.items())
    }
    by_dimension_and_subset = {
        str(int(dim)): {
            name: aggregate_record_group(
                [
                    record
                    for record in records
                    if int(record["dimension"]) == int(dim) and int(record["pair_id"]) in pair_ids
                ]
            )
            for name, pair_ids in sorted(pair_subset_ids.items())
        }
        for dim in dimensions
    }
    return {
        "expected_projected_records": int(len(default_baselines) * len(dimensions) * len(projection_seeds)),
        "observed_projected_records": int(len(records)),
        "default_baselines": aggregate_default_baselines(default_baselines),
        "overall": aggregate_record_group(records),
        "by_dimension": by_dimension,
        "by_projection_seed": by_projection_seed,
        "by_subset": by_subset,
        "by_dimension_and_subset": by_dimension_and_subset,
    }


def build_sanity_summary(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    default_baselines: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_success = float(np.mean([bool(record["rank1_success"]) for record in default_baselines]))
    projected_success = float(np.mean([bool(record["cem_late_success"]) for record in records]))
    success_delta = projected_success - baseline_success
    default_std_by_pair = {
        int(record["pair_id"]): float(record["final_top30_default_cost_std"] or 0.0)
        for record in default_baselines
    }
    ratios = []
    for record in records:
        default_std = default_std_by_pair[int(record["pair_id"])]
        projected_std = record["projected_cem_diagnostics"].get("final_top30_elite_cost_std")
        if projected_std is None or default_std <= 0:
            continue
        ratios.append(float(projected_std) / default_std)
    ratio_mean = float(np.mean(ratios)) if ratios else math.nan
    action_l2_values = [
        float(record["projected_cem_diagnostics"]["action_l2_to_default_blocked"])
        for record in records
        if record["projected_cem_diagnostics"].get("action_l2_to_default_blocked") is not None
    ]
    action_l2_mean = float(np.mean(action_l2_values)) if action_l2_values else math.nan
    success_pass = abs(success_delta) <= float(args.sanity_success_tolerance)
    ratio_pass = (
        math.isfinite(ratio_mean)
        and float(args.sanity_elite_std_ratio_min) <= ratio_mean <= float(args.sanity_elite_std_ratio_max)
    )
    action_pass = math.isfinite(action_l2_mean) and action_l2_mean <= float(args.sanity_action_l2_max)
    return {
        "criteria": {
            "success_delta_abs_max": float(args.sanity_success_tolerance),
            "elite_std_ratio_range": [
                float(args.sanity_elite_std_ratio_min),
                float(args.sanity_elite_std_ratio_max),
            ],
            "mean_blocked_action_l2_max": float(args.sanity_action_l2_max),
        },
        "baseline_success_rate": clean_float(baseline_success),
        "projected_success_rate": clean_float(projected_success),
        "success_delta": clean_float(success_delta),
        "success_pass": bool(success_pass),
        "elite_std_ratio_mean": clean_float(ratio_mean),
        "elite_std_ratio_values": ratios,
        "elite_std_ratio_pass": bool(ratio_pass),
        "blocked_action_l2_mean": clean_float(action_l2_mean),
        "blocked_action_l2_values": action_l2_values,
        "blocked_action_l2_pass": bool(action_pass),
        "pass": bool(success_pass and ratio_pass and action_pass),
    }


def metric_helper_names() -> list[str]:
    return [
        compute_metrics.__name__,
        summary_row.__name__,
        load_latent_artifact.__name__,
        squared_l2_torch.__name__,
        spearman_corr.__name__,
        pairwise_accuracy.__name__,
        deterministic_topk_indices.__name__,
        rollout_candidate_latents.__name__,
    ]


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.raw_checkpoint_dir = args.raw_checkpoint_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output = (
        (DEFAULT_OUTPUT_SANITY if args.sanity else DEFAULT_OUTPUT_FULL)
        if args.output is None
        else args.output
    )
    args.output = args.output.expanduser().resolve()

    latent_artifact = load_latent_artifact(args.latent_artifact)
    common = build_common(latent_artifact)
    selected_pair_ids, pair_selection_mode = select_pair_ids(args=args, common=common)
    if not args.sanity and args.pair_ids is None and len(selected_pair_ids) != 63:
        raise RuntimeError(f"Expected full projected-CEM pair set to contain 63 pairs, got {len(selected_pair_ids)}")

    if args.sanity and args.pair_ids is None:
        dimensions = (192,)
        projection_seeds = (0,)
    else:
        dimensions = tuple(int(dim) for dim in args.dimensions)
        projection_seeds = tuple(int(seed) for seed in args.projection_seeds)

    args.pair_ids = selected_pair_ids
    args.max_pairs = None
    ctx = build_replay_context(args)
    pair_subset_ids = build_pair_subset_ids(common)
    projections = {
        (int(dim), int(seed)): make_projection(int(dim), int(seed))
        for dim in dimensions
        for seed in projection_seeds
    }

    print(
        "Stage 1B-Smoke setup: "
        f"pairs={len(selected_pair_ids)} dims={list(dimensions)} seeds={list(projection_seeds)} "
        f"expected_projected_records={len(selected_pair_ids) * len(dimensions) * len(projection_seeds)}"
    )

    env = gym.make("swm/PushT-v1")
    default_baselines: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    started = time.time()
    try:
        for pair_idx, pair_spec in enumerate(ctx.requested_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            pair_started = time.time()
            row_data = ctx.dataset.get_row_data([int(pair_spec["start_row"]), int(pair_spec["goal_row"])])
            initial = {key: value[0] for key, value in row_data.items()}
            goal = {key: value[1] for key, value in row_data.items()}
            prepared_info = prepare_pair_info(ctx.policy, initial["pixels"], goal["pixels"])
            action_processor = ctx.policy.process["action"]
            initial_state = np.asarray(initial["state"], dtype=np.float32)
            goal_state = np.asarray(goal["state"], dtype=np.float32)
            subsets = pair_subsets(pair_id=pair_id, pair_subset_ids=pair_subset_ids)
            cell = str(pair_spec.get("cell", "unknown"))
            cem_seed = int(args.seed) + pair_id * 1009

            print(f"[{pair_idx}/{len(ctx.requested_pairs)}] pair_id={pair_id} cell={cell}: default CEM")
            baseline = run_cem(
                model=ctx.model,
                prepared_info=prepared_info,
                pair_id=pair_id,
                seed=int(args.seed),
                projection=None,
            )
            baseline_raw = blocked_batch_to_raw_fast(
                baseline["blocked_candidates"],
                action_processor=action_processor,
            )
            baseline_v1, baseline_real, baseline_success, baseline_metrics = score_raw_actions(
                env=env,
                initial_state=initial_state,
                goal_state=goal_state,
                raw_actions_batch=baseline_raw,
                seed_base=int(args.seed) + pair_id * 100_000,
            )
            baseline_rank1_raw = baseline_raw[int(baseline["rank1_candidate_index"])]
            default_baselines.append(
                build_default_baseline_record(
                    pair_id=pair_id,
                    cell=cell,
                    subsets=subsets,
                    cem_seed=cem_seed,
                    baseline=baseline,
                    v1_costs=baseline_v1,
                    c_real_state=baseline_real,
                    success=baseline_success,
                    scored_metrics=baseline_metrics,
                )
            )

            for dim in dimensions:
                for projection_seed in projection_seeds:
                    print(
                        f"  projected CEM m={int(dim)} projection_seed={int(projection_seed)} "
                        "(rank-1 simulator execution only)"
                    )
                    projection = projections[(int(dim), int(projection_seed))]
                    projected = run_cem(
                        model=ctx.model,
                        prepared_info=prepared_info,
                        pair_id=pair_id,
                        seed=int(args.seed),
                        projection=projection,
                    )
                    projected_raw = blocked_batch_to_raw_fast(
                        projected["rank1_blocked"][None, ...],
                        action_processor=action_processor,
                    )[0]
                    rollout_seed = int(args.seed) + pair_id * 100_000 + int(dim) * 1_000 + int(projection_seed)
                    proj_v1, proj_real, proj_success, proj_metrics = score_raw_actions(
                        env=env,
                        initial_state=initial_state,
                        goal_state=goal_state,
                        raw_actions_batch=projected_raw[None, ...],
                        seed_base=rollout_seed,
                        seed_indices=np.asarray([0], dtype=np.int64),
                    )
                    projected_rollout_metrics = {
                        **proj_metrics[0],
                        "v1_cost": clean_float(float(proj_v1[0])),
                        "c_real_state": clean_float(float(proj_real[0])),
                        "success": bool(proj_success[0]),
                    }
                    records.append(
                        build_projected_record(
                            pair_id=pair_id,
                            cell=cell,
                            subsets=subsets,
                            dim=int(dim),
                            projection_seed=int(projection_seed),
                            cem_seed=cem_seed,
                            projection=projection,
                            baseline=baseline,
                            baseline_raw_rank1=baseline_rank1_raw,
                            baseline_v1_costs=baseline_v1,
                            baseline_c_real_state=baseline_real,
                            baseline_success=baseline_success,
                            projected=projected,
                            projected_rollout_metrics=projected_rollout_metrics,
                            projected_raw_rank1=projected_raw,
                        )
                    )

            print(
                f"  pair_id={pair_id} completed in {time.time() - pair_started:.1f}s; "
                f"projected_records={len(records)}"
            )
    finally:
        env.close()

    aggregate = aggregate_records(
        records=records,
        default_baselines=default_baselines,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        pair_subset_ids=pair_subset_ids,
    )
    output: dict[str, Any] = {
        "metadata": {
            "format": "stage1b_smoke_projected_cem",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "mode": "sanity" if args.sanity else "full",
            "description": (
                "Projected-cost CEM smoke test. Default LeWM final candidate pools are simulator-scored "
                "once per pair; projected endpoint metrics reuse those labels; projected CEM executes "
                "only the rank-1 final action."
            ),
            "output": str(args.output),
            "pairs_path": str(args.pairs_path),
            "latent_artifact": str(args.latent_artifact),
            "pair_selection_mode": pair_selection_mode,
            "selected_pair_ids": selected_pair_ids,
            "n_selected_pairs": int(len(selected_pair_ids)),
            "dimensions": [int(dim) for dim in dimensions],
            "projection_seeds": [int(seed) for seed in projection_seeds],
            "expected_projected_records": int(len(selected_pair_ids) * len(dimensions) * len(projection_seeds)),
            "expected_simulator_rollouts": {
                "default_final_candidate_pool": int(len(selected_pair_ids) * NUM_SAMPLES),
                "projected_rank1": int(len(selected_pair_ids) * len(dimensions) * len(projection_seeds)),
                "total": int(len(selected_pair_ids) * NUM_SAMPLES + len(selected_pair_ids) * len(dimensions) * len(projection_seeds)),
            },
            "cem_config": {
                "samples_per_iter": NUM_SAMPLES,
                "iterations": CEM_ITERS,
                "elites": TOPK,
                "planning_horizon": PLANNING_HORIZON,
                "action_block": ACTION_BLOCK,
                "var_scale": VAR_SCALE,
                "img_size": IMG_SIZE,
                "sampling_seed_rule": "base_seed + pair_id * 1009",
                "candidate_0_forced_to_search_mean": True,
            },
            "projection_config": {
                "matrix_shape": "[192, m]",
                "entry_distribution": "Gaussian N(0, 1) / sqrt(m)",
                "fixed_scope": "one matrix per (m, projection_seed), reused across all pairs and CEM iterations",
                "matrices": projection_metadata(projections),
            },
            "endpoint_metric_pool": {
                "source": "default LeWM final CEM iteration 300-candidate pool",
                "simulator_scored_once_per_pair": True,
                "topk_values": list(ENDPOINT_TOPK_VALUES),
                "false_elite_k": FALSE_ELITE_K,
            },
            "stage1a_reference_constants": {
                "expected_records": EXPECTED_RECORDS,
                "latent_dim": LATENT_DIM,
                "stage1a_topk_values": list(TOPK_VALUES),
            },
            "anchor_definitions": audit_anchor_definitions(common["pair_ids"], common["cells"]),
            "metric_helpers_reused": metric_helper_names(),
            "wallclock_seconds": clean_float(time.time() - started),
        },
        "default_baselines": default_baselines,
        "records": records,
        "aggregate": aggregate,
    }
    if args.sanity:
        output["sanity"] = build_sanity_summary(
            args=args,
            records=records,
            default_baselines=default_baselines,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_table(aggregate)
    if args.sanity:
        print(f"Sanity pass: {output['sanity']['pass']}")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
