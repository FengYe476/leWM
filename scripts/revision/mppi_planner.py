#!/usr/bin/env python3
"""Reusable PushT MPPI planner utilities for Phase D revision experiments."""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lewm_audit.diagnostics.three_cost import prepare_pair_info  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    TOPK,
    VAR_SCALE,
    load_pairs,
)
from scripts.phase2.stage1.projected_cem import (  # noqa: E402
    blocked_batch_to_raw_fast,
    euclidean_costs,
    score_raw_actions,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    LATENT_DIM,
    clean_float,
    deterministic_topk_indices,
    iso_now,
    spearman_corr,
)
from scripts.phase2.train_cem_aware import rollout_candidate_latents  # noqa: E402


DEFAULT_RERANK_PATH = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
STAGE_A_PAIR_SELECTION_COUNTS = {
    "invisible_quadrant": 8,
    "ordinary": 12,
    "latent_favorable": 5,
    "v1_favorable": 5,
}
STAGE_A_SUBSET_ORDER = ("invisible_quadrant", "ordinary", "latent_favorable", "v1_favorable")
ALL_MEMBERSHIP_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)


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


def mppi_config_metadata(*, temperature: float, update_var: str = "fixed") -> dict[str, Any]:
    return {
        "samples_per_iter": int(NUM_SAMPLES),
        "iterations": int(CEM_ITERS),
        "planning_horizon_blocks": int(PLANNING_HORIZON),
        "action_block": int(ACTION_BLOCK),
        "raw_action_dim": 2,
        "blocked_action_dim": int(ACTION_BLOCK * 2),
        "raw_action_steps_executed": int(PLANNING_HORIZON * ACTION_BLOCK),
        "var_scale": float(VAR_SCALE),
        "temperature": clean_float(float(temperature)),
        "weight_rule": "softmax(-(cost - min(cost)) / temperature) over all 300 candidates",
        "var_update": str(update_var),
        "img_size": int(IMG_SIZE),
        "sampling_seed_rule": "base_seed + pair_id * 1009",
        "candidate_0_forced_to_search_mean": True,
        "cost": "squared Euclidean terminal latent distance",
    }


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


def load_pair_rows_direct(dataset, pair_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = dataset.get_row_data([int(pair_spec["start_row"]), int(pair_spec["goal_row"])])
    return {key: value[0] for key, value in rows.items()}, {key: value[1] for key, value in rows.items()}


def run_mppi(
    *,
    model,
    prepared_info: dict[str, Any],
    pair_id: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    """Run MPPI with the audit CEM action/cost path and capture the final pool.

    The final pool is the 300 candidates sampled on iteration 30. MPPI's actual
    final search mean after the last soft update is stored separately because it
    is a weighted average, not necessarily one of the sampled candidates.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature!r}")

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
        shifted_cost = default_cost - default_cost.min(dim=1, keepdim=True).values
        weights = torch.softmax(-shifted_cost / temperature, dim=1)
        next_mean = (weights.unsqueeze(-1).unsqueeze(-1) * candidates).sum(dim=1)
        top_vals, top_inds = torch.topk(default_cost, k=TOPK, dim=1, largest=False)

        if iter_idx == CEM_ITERS:
            costs_np = default_cost[0].detach().cpu().numpy().astype(np.float64)
            top_np = top_vals[0].detach().cpu().numpy().astype(np.float64)
            weights_np = weights[0].detach().cpu().numpy().astype(np.float64)
            rank1_idx = int(top_inds[0, 0].detach().cpu().item())
            final = {
                "blocked_candidates": candidates[0].detach().cpu().numpy().astype(np.float32),
                "rank1_blocked": candidates[0, rank1_idx].detach().cpu().numpy().astype(np.float32),
                "rank1_candidate_index": rank1_idx,
                "candidate_indices": np.arange(NUM_SAMPLES, dtype=np.int64),
                "select_costs": costs_np,
                "default_costs": costs_np,
                "z_pred": z_pred[0].detach().cpu().numpy().astype(np.float32),
                "z_goal": z_goal[0].detach().cpu().numpy().astype(np.float32),
                "top30_select_costs": top_np,
                "top30_select_cost_std": clean_float(float(np.std(top_np, ddof=0))),
                "select_cost_dynamic_range": clean_float(float(np.max(costs_np) - np.min(costs_np))),
                "select_cost_min": clean_float(float(np.min(costs_np))),
                "select_cost_max": clean_float(float(np.max(costs_np))),
                "select_cost_median": clean_float(float(np.median(costs_np))),
                "rank1_default_cost": clean_float(float(costs_np[rank1_idx])),
                "rank1_below_pool_median": bool(costs_np[rank1_idx] < float(np.median(costs_np))),
                "final_weights": weights_np,
                "final_weight_entropy": clean_float(
                    float(-np.sum(weights_np * np.log(np.clip(weights_np, 1e-300, 1.0))))
                ),
                "final_weight_effective_sample_size": clean_float(float(1.0 / np.sum(weights_np**2))),
                "mppi_mean_before_final_update": mean[0].detach().cpu().numpy().astype(np.float32),
                "mppi_mean_after_final_update": next_mean[0].detach().cpu().numpy().astype(np.float32),
                "temperature": clean_float(temperature),
            }

        mean = next_mean

    if final is None:
        raise RuntimeError("MPPI final iteration was not captured")
    final["wallclock_seconds"] = clean_float(time.time() - started)
    return final


def build_unscored_mppi_pool(
    *,
    pair_spec: dict[str, Any],
    dataset,
    policy,
    model,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    pair_id = int(pair_spec["pair_id"])
    started = time.time()
    initial, goal = load_pair_rows_direct(dataset, pair_spec)
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    result = run_mppi(
        model=model,
        prepared_info=prepared_info,
        pair_id=pair_id,
        seed=int(seed),
        temperature=float(temperature),
    )
    raw_actions = blocked_batch_to_raw_fast(
        np.asarray(result["blocked_candidates"], dtype=np.float32),
        action_processor=policy.process["action"],
    )
    default_costs = np.asarray(result["default_costs"], dtype=np.float64)
    rank1 = int(result["rank1_candidate_index"])
    return {
        "metadata": {
            "format": "pusht_mppi_pool_v1_unscored",
            "created_at": iso_now(),
            "pair_id": pair_id,
            "cell": str(pair_spec["cell"]),
            "start_row": int(pair_spec["start_row"]),
            "goal_row": int(pair_spec["goal_row"]),
            "seed": int(seed),
            "sampling_seed": int(seed) + pair_id * 1009,
            "mppi_config": mppi_config_metadata(temperature=float(temperature)),
            "wallclock_seconds": clean_float(time.time() - started),
        },
        "pair_spec": dict(pair_spec),
        "z_pred": torch.as_tensor(np.asarray(result["z_pred"], dtype=np.float32)),
        "z_goal": torch.as_tensor(np.asarray(result["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.as_tensor(np.asarray(result["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.as_tensor(raw_actions.astype(np.float32)),
        "candidate_indices": torch.arange(NUM_SAMPLES, dtype=torch.int64),
        "default_costs": torch.as_tensor(default_costs, dtype=torch.float64),
        "default_rank1_candidate_index": rank1,
        "default_top30_cost_std": result["top30_select_cost_std"],
        "default_cost_dynamic_range": result["select_cost_dynamic_range"],
        "default_cost_min": result["select_cost_min"],
        "default_cost_max": result["select_cost_max"],
        "default_cost_median": result["select_cost_median"],
        "rank1_default_cost": result["rank1_default_cost"],
        "rank1_below_pool_median": result["rank1_below_pool_median"],
        "final_weights": torch.as_tensor(np.asarray(result["final_weights"], dtype=np.float64)),
        "final_weight_entropy": result["final_weight_entropy"],
        "final_weight_effective_sample_size": result["final_weight_effective_sample_size"],
        "mppi_mean_before_final_update": torch.as_tensor(
            np.asarray(result["mppi_mean_before_final_update"], dtype=np.float32)
        ),
        "mppi_mean_after_final_update": torch.as_tensor(
            np.asarray(result["mppi_mean_after_final_update"], dtype=np.float32)
        ),
    }


def build_scored_mppi_pool(
    *,
    pair_spec: dict[str, Any],
    dataset,
    policy,
    model,
    env,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    pair_id = int(pair_spec["pair_id"])
    started = time.time()
    initial, goal = load_pair_rows_direct(dataset, pair_spec)
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    result = run_mppi(
        model=model,
        prepared_info=prepared_info,
        pair_id=pair_id,
        seed=int(seed),
        temperature=float(temperature),
    )
    raw_actions = blocked_batch_to_raw_fast(
        np.asarray(result["blocked_candidates"], dtype=np.float32),
        action_processor=policy.process["action"],
    )
    v1_costs, c_real_state, success, metrics = score_raw_actions(
        env=env,
        initial_state=np.asarray(initial["state"], dtype=np.float32),
        goal_state=np.asarray(goal["state"], dtype=np.float32),
        raw_actions_batch=raw_actions,
        seed_base=int(seed) + pair_id * 100_000,
    )
    block_pos_dist = np.asarray([float(item["block_pos_dist"]) for item in metrics], dtype=np.float64)
    angle_dist = np.asarray([float(item["angle_dist"]) for item in metrics], dtype=np.float64)
    default_costs = np.asarray(result["default_costs"], dtype=np.float64)
    rank1 = int(result["rank1_candidate_index"])
    oracle_best = int(
        deterministic_topk_indices(
            c_real_state,
            np.arange(NUM_SAMPLES, dtype=np.int64),
            np.ones(NUM_SAMPLES, dtype=bool),
            1,
        )[0]
    )

    return {
        "metadata": {
            "format": "pusht_mppi_pool_v1",
            "created_at": iso_now(),
            "pair_id": pair_id,
            "cell": str(pair_spec["cell"]),
            "start_row": int(pair_spec["start_row"]),
            "goal_row": int(pair_spec["goal_row"]),
            "seed": int(seed),
            "sampling_seed": int(seed) + pair_id * 1009,
            "mppi_config": mppi_config_metadata(temperature=float(temperature)),
            "wallclock_seconds": clean_float(time.time() - started),
        },
        "pair_spec": dict(pair_spec),
        "z_pred": torch.as_tensor(np.asarray(result["z_pred"], dtype=np.float32)),
        "z_goal": torch.as_tensor(np.asarray(result["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.as_tensor(np.asarray(result["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.as_tensor(raw_actions.astype(np.float32)),
        "candidate_indices": torch.arange(NUM_SAMPLES, dtype=torch.int64),
        "default_costs": torch.as_tensor(default_costs, dtype=torch.float64),
        "v1_hinge_costs": torch.as_tensor(v1_costs, dtype=torch.float64),
        "c_real_state": torch.as_tensor(c_real_state, dtype=torch.float64),
        "block_pos_dist": torch.as_tensor(block_pos_dist, dtype=torch.float64),
        "angle_dist": torch.as_tensor(angle_dist, dtype=torch.float64),
        "success": torch.as_tensor(success, dtype=torch.bool),
        "candidate_metrics": metrics,
        "default_rank1_candidate_index": rank1,
        "oracle_best_candidate_index": oracle_best,
        "default_top30_cost_std": result["top30_select_cost_std"],
        "default_cost_dynamic_range": result["select_cost_dynamic_range"],
        "default_cost_min": result["select_cost_min"],
        "default_cost_max": result["select_cost_max"],
        "default_cost_median": result["select_cost_median"],
        "rank1_default_cost": result["rank1_default_cost"],
        "rank1_below_pool_median": result["rank1_below_pool_median"],
        "final_weights": torch.as_tensor(np.asarray(result["final_weights"], dtype=np.float64)),
        "final_weight_entropy": result["final_weight_entropy"],
        "final_weight_effective_sample_size": result["final_weight_effective_sample_size"],
        "mppi_mean_before_final_update": torch.as_tensor(
            np.asarray(result["mppi_mean_before_final_update"], dtype=np.float32)
        ),
        "mppi_mean_after_final_update": torch.as_tensor(
            np.asarray(result["mppi_mean_after_final_update"], dtype=np.float32)
        ),
    }


def pool_summary_record(
    *,
    pool: dict[str, Any],
    primary_subset: str | None = None,
    subset_memberships: list[str] | None = None,
    pool_path: Path | None = None,
) -> dict[str, Any]:
    default_costs = torch.as_tensor(pool["default_costs"], dtype=torch.float64).detach().cpu().numpy()
    c_real_state = (
        torch.as_tensor(pool["c_real_state"], dtype=torch.float64).detach().cpu().numpy()
        if "c_real_state" in pool
        else None
    )
    success = (
        torch.as_tensor(pool["success"], dtype=torch.float32).detach().cpu().numpy()
        if "success" in pool
        else None
    )
    rank1 = int(pool["default_rank1_candidate_index"])
    record: dict[str, Any] = {
        "pair_id": int(pool["pair_spec"]["pair_id"]),
        "cell": str(pool["pair_spec"]["cell"]),
        "seed": int(pool["metadata"]["seed"]),
        "temperature": clean_float(float(pool["metadata"]["mppi_config"]["temperature"])),
        "sampling_seed": int(pool["metadata"]["sampling_seed"]),
        "primary_subset": primary_subset,
        "subset_memberships": subset_memberships or [],
        "pool_path": str(pool_path) if pool_path is not None else None,
        "rank1_candidate_index": rank1,
        "rank1_default_cost": clean_float(float(default_costs[rank1])),
        "pool_default_cost_median": clean_float(float(np.median(default_costs))),
        "rank1_below_pool_median": bool(default_costs[rank1] < float(np.median(default_costs))),
        "pool_Cmodel_std": clean_float(float(np.std(default_costs, ddof=0))),
        "top30_Cmodel_std": pool["default_top30_cost_std"],
        "default_cost_dynamic_range": pool["default_cost_dynamic_range"],
        "final_weight_entropy": pool["final_weight_entropy"],
        "final_weight_effective_sample_size": pool["final_weight_effective_sample_size"],
        "wallclock_seconds": pool["metadata"].get("wallclock_seconds"),
    }
    if c_real_state is not None and success is not None:
        oracle_best = int(pool["oracle_best_candidate_index"])
        record.update(
            {
                "rank1_success": bool(success[rank1]),
                "rank1_c_real_state": clean_float(float(c_real_state[rank1])),
                "oracle_best_candidate_index": oracle_best,
                "oracle_best_c_real_state": clean_float(float(c_real_state[oracle_best])),
                "selection_regret": clean_float(float(c_real_state[rank1] - c_real_state[oracle_best])),
                "pool_success_mass": clean_float(float(np.mean(success))),
                "pool_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
                "pool_Creal_range": clean_float(float(np.max(c_real_state) - np.min(c_real_state))),
                "Rpool_Cmodel": clean_float(spearman_corr(default_costs, c_real_state)),
            }
        )
    return record


def membership_map(anchor_definitions: dict[str, Any]) -> dict[int, list[str]]:
    memberships: dict[int, list[str]] = {}
    for name in ALL_MEMBERSHIP_ORDER:
        for pair_id in anchor_definitions.get(name, {}).get("pair_ids", []):
            memberships.setdefault(int(pair_id), []).append(name)
    return memberships


def load_anchor_definitions(rerank_path: Path) -> dict[str, Any]:
    import json

    data = json.loads(Path(rerank_path).read_text())
    anchor_definitions = data.get("metadata", {}).get("anchor_definitions")
    if not isinstance(anchor_definitions, dict):
        raise RuntimeError(f"Missing metadata.anchor_definitions in {rerank_path}")
    return anchor_definitions


def select_stage_a_pairs(
    *,
    pairs_path: Path,
    rerank_path: Path = DEFAULT_RERANK_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, list[str]], dict[str, Any]]:
    anchor_definitions = load_anchor_definitions(rerank_path)
    pairs_data, all_pairs = load_pairs(pairs_path, max_pairs=None, pair_ids=None)
    by_id = {int(pair["pair_id"]): pair for pair in all_pairs}
    memberships = membership_map(anchor_definitions)

    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for subset in STAGE_A_SUBSET_ORDER:
        pair_ids = [int(pair_id) for pair_id in anchor_definitions[subset]["pair_ids"]]
        chosen: list[int] = []
        for pair_id in pair_ids:
            if pair_id in used:
                continue
            chosen.append(pair_id)
            if len(chosen) == int(STAGE_A_PAIR_SELECTION_COUNTS[subset]):
                break
        if len(chosen) != int(STAGE_A_PAIR_SELECTION_COUNTS[subset]):
            raise RuntimeError(
                f"Could not select {STAGE_A_PAIR_SELECTION_COUNTS[subset]} pairs for {subset}; got {len(chosen)}"
            )
        for pair_id in chosen:
            if pair_id not in by_id:
                raise RuntimeError(f"Selected pair_id={pair_id} missing from {pairs_path}")
            used.add(pair_id)
            pair = dict(by_id[pair_id])
            pair["primary_subset"] = subset
            pair["subset_memberships"] = memberships.get(pair_id, [])
            selected.append(pair)

    if len(selected) != 30 or len({int(pair["pair_id"]) for pair in selected}) != 30:
        raise RuntimeError("Stage A pair selection must contain 30 unique pairs")
    return pairs_data, selected, memberships, anchor_definitions


def aggregate_run_records(records: list[dict[str, Any]], *, include_by_subset: bool = True) -> dict[str, Any]:
    subsets = sorted({str(record["primary_subset"]) for record in records if record.get("primary_subset")})
    summary = {
        "n_records": int(len(records)),
        "rank1_success": scalar_summary([record.get("rank1_success") for record in records]),
        "selection_regret": scalar_summary([record.get("selection_regret") for record in records]),
        "Rpool_Cmodel": scalar_summary([record.get("Rpool_Cmodel") for record in records]),
        "pool_success_mass": scalar_summary([record.get("pool_success_mass") for record in records]),
        "pool_Creal_std": scalar_summary([record.get("pool_Creal_std") for record in records]),
        "pool_Cmodel_std": scalar_summary([record.get("pool_Cmodel_std") for record in records]),
        "top30_Cmodel_std": scalar_summary([record.get("top30_Cmodel_std") for record in records]),
        "final_weight_effective_sample_size": scalar_summary(
            [record.get("final_weight_effective_sample_size") for record in records]
        ),
    }
    if include_by_subset:
        summary["by_subset"] = {
            subset: aggregate_run_records(
                [record for record in records if record.get("primary_subset") == subset],
                include_by_subset=False,
            )
            for subset in subsets
        }
    return summary
