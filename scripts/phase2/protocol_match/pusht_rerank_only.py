#!/usr/bin/env python3
"""PushT Block 1.1 re-rank-only projection audit.

This closes the PushT re-rank-only cell of the protocol grid. The expensive
phase runs default full-dimensional LeWM CEM once per Track A pair, simulator
scores the final 300-candidate pool, and saves that labelled pool. The cheap
phase re-ranks the saved pools under fixed Gaussian projections.
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

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import prepare_pair_info  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    DEFAULT_PAIRS_PATH,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    TOPK,
    VAR_SCALE,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.projected_cem import (  # noqa: E402
    blocked_batch_to_raw_fast,
    make_projection,
    numpy_projected_costs,
    projection_metadata,
    run_cem,
    score_raw_actions,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    ANCHOR_DEFINITIONS,
    LATENT_DIM,
    clean_float,
    deterministic_topk_indices,
    iso_now,
    jsonable,
    pairwise_accuracy,
    spearman_corr,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
DEFAULT_DIMS = (1, 2, 4, 8, 16, 32, 64, 128, 192)
DEFAULT_PROJECTION_SEEDS = (0, 1, 2)
SMOKE_DIMS = (8, 64, 192)
SMOKE_PROJECTION_SEEDS = (0,)
SMOKE_PAIR_COUNT = 5
ENDPOINT_TOPK_VALUES = (5, 10, 30)
FALSE_ELITE_K = 30
V1_FAVORABLE_CELLS = ("D3xR0", "D3xR3")
ANCHOR_SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
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


def cem_config_metadata() -> dict[str, Any]:
    return {
        "samples_per_iter": int(NUM_SAMPLES),
        "iterations": int(CEM_ITERS),
        "elites": int(TOPK),
        "planning_horizon_blocks": int(PLANNING_HORIZON),
        "action_block": int(ACTION_BLOCK),
        "raw_action_dim": 2,
        "blocked_action_dim": int(ACTION_BLOCK * 2),
        "raw_action_steps_executed": int(PLANNING_HORIZON * ACTION_BLOCK),
        "var_scale": float(VAR_SCALE),
        "img_size": int(IMG_SIZE),
        "sampling_seed_rule": "base_seed + pair_id * 1009",
        "candidate_0_forced_to_search_mean": True,
    }


def expected_pool_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "z_pred": (NUM_SAMPLES, LATENT_DIM),
        "z_goal": (LATENT_DIM,),
        "blocked_actions": (NUM_SAMPLES, PLANNING_HORIZON, ACTION_BLOCK * 2),
        "raw_actions": (NUM_SAMPLES, PLANNING_HORIZON * ACTION_BLOCK, 2),
        "default_costs": (NUM_SAMPLES,),
        "v1_hinge_costs": (NUM_SAMPLES,),
        "c_real_state": (NUM_SAMPLES,),
        "block_pos_dist": (NUM_SAMPLES,),
        "angle_dist": (NUM_SAMPLES,),
        "success": (NUM_SAMPLES,),
    }


def load_pair_rows_direct(dataset, pair_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = dataset.get_row_data([int(pair_spec["start_row"]), int(pair_spec["goal_row"])])
    return {key: value[0] for key, value in rows.items()}, {key: value[1] for key, value in rows.items()}


def pool_path(pool_dir: Path, pair_id: int) -> Path:
    return pool_dir / f"pair_{int(pair_id)}.pt"


def tensor_to_numpy(pool: dict[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    value = pool[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype)
    return np.asarray(value, dtype=dtype)


def tensor_to_bool_numpy(pool: dict[str, Any], key: str) -> np.ndarray:
    value = pool[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(bool)
    return np.asarray(value, dtype=bool)


def validate_pool_artifact(
    pool: dict[str, Any],
    *,
    pair_spec: dict[str, Any],
    seed: int,
) -> None:
    metadata = pool.get("metadata", {})
    if metadata.get("format") != "pusht_rerank_only_pool_v1":
        raise RuntimeError(f"Invalid pool format for pair {pair_spec['pair_id']}: {metadata.get('format')!r}")
    checks = {
        "pair_id": int(pair_spec["pair_id"]),
        "start_row": int(pair_spec["start_row"]),
        "goal_row": int(pair_spec["goal_row"]),
        "seed": int(seed),
    }
    for key, expected in checks.items():
        observed = int(metadata.get(key, -1))
        if observed != expected:
            raise RuntimeError(
                f"Pool metadata mismatch for pair {pair_spec['pair_id']} key={key}: "
                f"{observed} != {expected}"
            )
    if metadata.get("cem_config") != cem_config_metadata():
        raise RuntimeError(f"Pool CEM config mismatch for pair {pair_spec['pair_id']}")

    for key, shape in expected_pool_shapes().items():
        value = pool.get(key)
        if value is None:
            raise RuntimeError(f"Pool missing key {key!r} for pair {pair_spec['pair_id']}")
        observed_shape = tuple(value.shape) if torch.is_tensor(value) else tuple(np.asarray(value).shape)
        if observed_shape != shape:
            raise RuntimeError(
                f"Pool shape mismatch for pair {pair_spec['pair_id']} key={key}: "
                f"{observed_shape} != {shape}"
            )

    rank1 = int(pool.get("default_rank1_candidate_index", -1))
    if not 0 <= rank1 < NUM_SAMPLES:
        raise RuntimeError(f"Invalid default rank-1 index for pair {pair_spec['pair_id']}: {rank1}")


def load_valid_pool(path: Path, *, pair_spec: dict[str, Any], seed: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    pool = torch.load(path, map_location="cpu", weights_only=False)
    validate_pool_artifact(pool, pair_spec=pair_spec, seed=seed)
    return pool


def build_pool_for_pair(
    *,
    pair_spec: dict[str, Any],
    dataset,
    policy,
    model,
    env,
    seed: int,
) -> dict[str, Any]:
    pair_id = int(pair_spec["pair_id"])
    started = time.time()
    initial, goal = load_pair_rows_direct(dataset, pair_spec)
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    baseline = run_cem(
        model=model,
        prepared_info=prepared_info,
        pair_id=pair_id,
        seed=int(seed),
        projection=None,
    )
    action_processor = policy.process["action"]
    raw_actions = blocked_batch_to_raw_fast(
        np.asarray(baseline["blocked_candidates"], dtype=np.float32),
        action_processor=action_processor,
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
    default_costs = np.asarray(baseline["default_costs"], dtype=np.float64)
    default_rank1 = int(baseline["rank1_candidate_index"])
    oracle_best = int(deterministic_topk_indices(
        c_real_state,
        np.arange(NUM_SAMPLES, dtype=np.int64),
        np.ones(NUM_SAMPLES, dtype=bool),
        1,
    )[0])

    return {
        "metadata": {
            "format": "pusht_rerank_only_pool_v1",
            "created_at": iso_now(),
            "pair_id": pair_id,
            "cell": str(pair_spec["cell"]),
            "start_row": int(pair_spec["start_row"]),
            "goal_row": int(pair_spec["goal_row"]),
            "seed": int(seed),
            "cem_sampling_seed": int(seed) + pair_id * 1009,
            "cem_config": cem_config_metadata(),
            "wallclock_seconds": clean_float(time.time() - started),
        },
        "pair_spec": dict(pair_spec),
        "z_pred": torch.as_tensor(np.asarray(baseline["z_pred"], dtype=np.float32)),
        "z_goal": torch.as_tensor(np.asarray(baseline["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.as_tensor(np.asarray(baseline["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.as_tensor(raw_actions.astype(np.float32)),
        "default_costs": torch.as_tensor(default_costs, dtype=torch.float64),
        "v1_hinge_costs": torch.as_tensor(v1_costs, dtype=torch.float64),
        "c_real_state": torch.as_tensor(c_real_state, dtype=torch.float64),
        "block_pos_dist": torch.as_tensor(block_pos_dist, dtype=torch.float64),
        "angle_dist": torch.as_tensor(angle_dist, dtype=torch.float64),
        "success": torch.as_tensor(success, dtype=torch.bool),
        "candidate_metrics": metrics,
        "default_rank1_candidate_index": default_rank1,
        "oracle_best_candidate_index": oracle_best,
        "default_top30_cost_std": baseline["top30_select_cost_std"],
        "default_cost_dynamic_range": baseline["select_cost_dynamic_range"],
        "default_cost_min": baseline["select_cost_min"],
        "default_cost_max": baseline["select_cost_max"],
    }


def ensure_pair_pools(
    *,
    requested_pairs: list[dict[str, Any]],
    pool_dir: Path,
    dataset,
    policy,
    model,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    pools: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    started = time.time()
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair_spec in enumerate(requested_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            path = pool_path(pool_dir, pair_id)
            pool = load_valid_pool(path, pair_spec=pair_spec, seed=seed)
            if pool is not None:
                skipped += 1
                print(f"[{pair_idx}/{len(requested_pairs)}] pair_id={pair_id}: loaded existing pool")
                pools.append(pool)
                continue

            pair_started = time.time()
            print(
                f"[{pair_idx}/{len(requested_pairs)}] pair_id={pair_id} cell={pair_spec['cell']}: "
                "default CEM + score 300"
            )
            pool = build_pool_for_pair(
                pair_spec=pair_spec,
                dataset=dataset,
                policy=policy,
                model=model,
                env=env,
                seed=seed,
            )
            torch.save(pool, path)
            validate_pool_artifact(pool, pair_spec=pair_spec, seed=seed)
            generated += 1
            pools.append(pool)
            success_rate = float(torch.as_tensor(pool["success"], dtype=torch.float32).mean().item())
            print(
                f"  saved {path}; pool_success={success_rate:.4f}; "
                f"elapsed={seconds_to_hms(time.time() - pair_started)}"
            )
    finally:
        env.close()
    return pools, {
        "generated": int(generated),
        "skipped_existing": int(skipped),
        "pool_phase_wallclock_seconds": clean_float(time.time() - started),
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


def false_elite_rate_against_true_topk(
    *,
    projected_costs: np.ndarray,
    c_real_state: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
) -> float | None:
    if int(projected_costs.shape[0]) < int(k):
        return None
    mask = np.ones(int(projected_costs.shape[0]), dtype=bool)
    projected_top = deterministic_topk_indices(projected_costs, candidate_ids, mask, int(k))
    true_top = deterministic_topk_indices(c_real_state, candidate_ids, mask, int(k))
    return clean_float(len(set(projected_top.tolist()) - set(true_top.tolist())) / float(k))


def endpoint_metrics_for_pool(
    *,
    projected_costs: np.ndarray,
    default_costs: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    candidate_ids: np.ndarray,
) -> dict[str, Any]:
    pair_ids = np.zeros(int(projected_costs.shape[0]), dtype=np.int64)
    pairwise = pairwise_accuracy(projected_costs, c_real_state, pair_ids)
    return {
        "n_candidates": int(projected_costs.shape[0]),
        "candidate_indices": candidate_ids,
        "pool_spearman": clean_float(spearman_corr(projected_costs, c_real_state)),
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
                reference_costs=c_real_state,
                candidate_ids=candidate_ids,
                k=int(k),
            )
            for k in ENDPOINT_TOPK_VALUES
        },
        "topk_overlap_v1_definition": "oracle top-k is ranked by C_real_state for this protocol",
        "false_elite_rate": false_elite_rate_against_true_topk(
            projected_costs=projected_costs,
            c_real_state=c_real_state,
            candidate_ids=candidate_ids,
            k=FALSE_ELITE_K,
        ),
        "false_elite_definition": "fraction of projected top-30 not in true C_real_state top-30",
        "false_elite_k": FALSE_ELITE_K,
        "projected_cost_min": clean_float(float(np.min(projected_costs))),
        "projected_cost_max": clean_float(float(np.max(projected_costs))),
        "projected_cost_dynamic_range": clean_float(float(np.max(projected_costs) - np.min(projected_costs))),
        "projected_top30_cost_std": clean_float(
            float(np.std(np.sort(projected_costs, kind="mergesort")[:FALSE_ELITE_K], ddof=0))
        ),
        "pool_success_rate": clean_float(float(np.mean(success))),
    }


def build_default_baseline_record(pool: dict[str, Any]) -> dict[str, Any]:
    pair_spec = pool["pair_spec"]
    pair_id = int(pair_spec["pair_id"])
    candidate_ids = np.arange(NUM_SAMPLES, dtype=np.int64)
    default_costs = tensor_to_numpy(pool, "default_costs", dtype=np.float64)
    v1_costs = tensor_to_numpy(pool, "v1_hinge_costs", dtype=np.float64)
    c_real_state = tensor_to_numpy(pool, "c_real_state", dtype=np.float64)
    success = tensor_to_bool_numpy(pool, "success")
    blocked_actions = tensor_to_numpy(pool, "blocked_actions", dtype=np.float32)
    raw_actions = tensor_to_numpy(pool, "raw_actions", dtype=np.float32)
    rank1 = int(pool["default_rank1_candidate_index"])
    oracle_best = int(pool["oracle_best_candidate_index"])
    metrics = pool.get("candidate_metrics", [])
    return {
        "pair_id": pair_id,
        "cell": str(pair_spec["cell"]),
        "cem_sampling_seed": int(pool["metadata"]["cem_sampling_seed"]),
        "candidate_scoring_mode": "all_300",
        "candidate_pool_original_n": int(NUM_SAMPLES),
        "candidate_pool_scored_n": int(NUM_SAMPLES),
        "rank1_candidate_index": rank1,
        "rank1_scored_pool_position": rank1,
        "rank1_success": bool(success[rank1]),
        "rank1_v1_cost": clean_float(float(v1_costs[rank1])),
        "rank1_c_real_state": clean_float(float(c_real_state[rank1])),
        "rank1_state_metrics": metrics[rank1] if metrics else None,
        "oracle_best_candidate_index": oracle_best,
        "oracle_best_c_real_state": clean_float(float(c_real_state[oracle_best])),
        "oracle_best_success": bool(success[oracle_best]),
        "selection_regret": clean_float(float(c_real_state[rank1] - c_real_state[oracle_best])),
        "pool_success_mass": clean_float(float(np.mean(success))),
        "final_top30_default_cost_std": pool["default_top30_cost_std"],
        "final_default_cost_dynamic_range": pool["default_cost_dynamic_range"],
        "final_default_cost_min": pool["default_cost_min"],
        "final_default_cost_max": pool["default_cost_max"],
        "rank1_blocked_action": blocked_actions[rank1],
        "rank1_raw_action": raw_actions[rank1],
        "candidate_pool": {
            "n_candidates": int(NUM_SAMPLES),
            "original_n_candidates": int(NUM_SAMPLES),
            "candidate_indices": candidate_ids,
            "default_costs": default_costs,
            "v1_costs": v1_costs,
            "v1_hinge_costs": v1_costs,
            "c_real_state": c_real_state,
            "success": success,
            "success_rate": clean_float(float(np.mean(success))),
            "v1_hinge_cost_min": clean_float(float(np.min(v1_costs))),
            "v1_hinge_cost_max": clean_float(float(np.max(v1_costs))),
            "c_real_state_min": clean_float(float(np.min(c_real_state))),
            "c_real_state_max": clean_float(float(np.max(c_real_state))),
            "oracle_best_candidate_index": oracle_best,
            "oracle_best_c_real_state": clean_float(float(c_real_state[oracle_best])),
        },
        "wallclock_seconds": pool["metadata"].get("wallclock_seconds"),
    }


def build_projected_record(
    *,
    pool: dict[str, Any],
    dim: int,
    projection_seed: int,
    projection: torch.Tensor,
) -> dict[str, Any]:
    pair_spec = pool["pair_spec"]
    pair_id = int(pair_spec["pair_id"])
    candidate_ids = np.arange(NUM_SAMPLES, dtype=np.int64)
    default_costs = tensor_to_numpy(pool, "default_costs", dtype=np.float64)
    v1_costs = tensor_to_numpy(pool, "v1_hinge_costs", dtype=np.float64)
    c_real_state = tensor_to_numpy(pool, "c_real_state", dtype=np.float64)
    success = tensor_to_bool_numpy(pool, "success")
    blocked_actions = tensor_to_numpy(pool, "blocked_actions", dtype=np.float32)
    raw_actions = tensor_to_numpy(pool, "raw_actions", dtype=np.float32)
    projected_costs = numpy_projected_costs(
        z_pred=tensor_to_numpy(pool, "z_pred", dtype=np.float32),
        z_goal=tensor_to_numpy(pool, "z_goal", dtype=np.float32),
        projection=projection,
    )
    mask = np.ones(NUM_SAMPLES, dtype=bool)
    rank1 = int(deterministic_topk_indices(projected_costs, candidate_ids, mask, 1)[0])
    default_rank1 = int(pool["default_rank1_candidate_index"])
    oracle_best = int(pool["oracle_best_candidate_index"])
    endpoint_metrics = endpoint_metrics_for_pool(
        projected_costs=projected_costs,
        default_costs=default_costs,
        c_real_state=c_real_state,
        success=success,
        candidate_ids=candidate_ids,
    )
    regret = float(c_real_state[rank1] - c_real_state[oracle_best])
    if regret < -1e-9:
        raise RuntimeError(f"Negative selection regret for pair={pair_id} m={dim} seed={projection_seed}: {regret}")
    metrics = pool.get("candidate_metrics", [])
    return {
        "pair_id": pair_id,
        "cell": str(pair_spec["cell"]),
        "dimension": int(dim),
        "projection_seed": int(projection_seed),
        "cem_sampling_seed": int(pool["metadata"]["cem_sampling_seed"]),
        "candidate_scoring_mode": "all_300",
        "candidate_pool_original_n": int(NUM_SAMPLES),
        "candidate_pool_scored_n": int(NUM_SAMPLES),
        "projection_matrix": {
            "shape": [LATENT_DIM, int(dim)],
            "scale": f"1/sqrt({int(dim)})",
            "fixed_scope": "fixed per (dimension, projection_seed) across pairs",
        },
        "rank1_success": bool(success[rank1]),
        "rank1_v1_cost": clean_float(float(v1_costs[rank1])),
        "rank1_c_real_state": clean_float(float(c_real_state[rank1])),
        "rank1_state_metrics": metrics[rank1] if metrics else None,
        "cem_late_success": bool(success[rank1]),
        "cem_late_v1_cost": clean_float(float(v1_costs[rank1])),
        "cem_late_c_real_state": clean_float(float(c_real_state[rank1])),
        "cem_late_state_metrics": metrics[rank1] if metrics else None,
        "selection_regret": clean_float(max(regret, 0.0)),
        "oracle_best_candidate_index": oracle_best,
        "oracle_best_c_real_state": clean_float(float(c_real_state[oracle_best])),
        "endpoint_metrics_on_default_pool": endpoint_metrics,
        "projected_cem_diagnostics": {
            "planning_mode": "rerank_default_final_pool",
            "rank1_candidate_index": rank1,
            "rank1_scored_pool_position": rank1,
            "rank1_projected_cost": clean_float(float(projected_costs[rank1])),
            "rank1_default_latent_cost": clean_float(float(default_costs[rank1])),
            "rank1_c_real_state": clean_float(float(c_real_state[rank1])),
            "selection_regret": clean_float(max(regret, 0.0)),
            "default_rank1_candidate_index": default_rank1,
            "oracle_best_candidate_index": oracle_best,
            "final_top30_elite_cost_std": endpoint_metrics["projected_top30_cost_std"],
            "final_candidate_dynamic_range": endpoint_metrics["projected_cost_dynamic_range"],
            "final_projected_cost_min": endpoint_metrics["projected_cost_min"],
            "final_projected_cost_max": endpoint_metrics["projected_cost_max"],
            "final_default_latent_cost_min": clean_float(float(np.min(default_costs))),
            "final_default_latent_cost_max": clean_float(float(np.max(default_costs))),
            "final_default_latent_cost_dynamic_range": clean_float(float(np.max(default_costs) - np.min(default_costs))),
            "action_l2_to_default_blocked": clean_float(float(np.linalg.norm(blocked_actions[rank1] - blocked_actions[default_rank1]))),
            "action_l2_to_default_raw": clean_float(float(np.linalg.norm(raw_actions[rank1] - raw_actions[default_rank1]))),
            "wallclock_seconds": 0.0,
        },
        "rank1_blocked_action": blocked_actions[rank1],
        "rank1_raw_action": raw_actions[rank1],
    }


def anchor_definitions_from_pairs(all_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_by_id = {int(pair["pair_id"]): pair for pair in all_pairs}
    all_pair_ids = set(pair_by_id)
    invisible = set(int(item) for item in ANCHOR_DEFINITIONS["invisible_quadrant"]["pair_ids"])
    sign_reversal = set(int(item) for item in ANCHOR_DEFINITIONS["sign_reversal"]["pair_ids"])
    latent = {
        int(pair["pair_id"])
        for pair in all_pairs
        if str(pair["cell"]) in set(ANCHOR_DEFINITIONS["latent_favorable"]["cells"])
    }
    v1 = {int(pair["pair_id"]) for pair in all_pairs if str(pair["cell"]) in set(V1_FAVORABLE_CELLS)}
    ordinary = all_pair_ids - invisible - sign_reversal - latent - v1
    definitions = {
        "invisible_quadrant": {
            "description": ANCHOR_DEFINITIONS["invisible_quadrant"]["description"],
            "pair_ids": sorted(invisible),
        },
        "sign_reversal": {
            "description": ANCHOR_DEFINITIONS["sign_reversal"]["description"],
            "pair_ids": sorted(sign_reversal),
        },
        "latent_favorable": {
            "description": "Track A cells D0xR1 and D1xR0",
            "cells": list(ANCHOR_DEFINITIONS["latent_favorable"]["cells"]),
            "pair_ids": sorted(latent),
        },
        "v1_favorable": {
            "description": "Track A cells D3xR0 and D3xR3",
            "cells": list(V1_FAVORABLE_CELLS),
            "pair_ids": sorted(v1),
        },
        "ordinary": {
            "description": "Complement of invisible_quadrant, sign_reversal, latent_favorable, and v1_favorable",
            "excluded_anchors": ["invisible_quadrant", "sign_reversal", "latent_favorable", "v1_favorable"],
            "pair_ids": sorted(ordinary),
        },
    }
    for definition in definitions.values():
        definition["n_pairs"] = int(len(definition["pair_ids"]))
    return definitions


def assert_full_anchor_counts(anchor_definitions: dict[str, Any], *, full_mode: bool) -> None:
    if not full_mode:
        return
    expected = {
        "invisible_quadrant": 16,
        "sign_reversal": 21,
        "latent_favorable": 12,
        "v1_favorable": 13,
        "ordinary": 47,
    }
    observed = {name: int(anchor_definitions[name]["n_pairs"]) for name in expected}
    if observed != expected:
        raise RuntimeError(f"Unexpected anchor counts: {observed} != {expected}")


def aggregate_record_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "projected_success_rate": scalar_summary([record.get("rank1_success") for record in records]),
        "rank1_success_rate": scalar_summary([record.get("rank1_success") for record in records]),
        "rank1_c_real_state": scalar_summary([record.get("rank1_c_real_state") for record in records]),
        "selection_regret": scalar_summary([record.get("selection_regret") for record in records]),
        "pool_spearman": scalar_summary(
            [record["endpoint_metrics_on_default_pool"].get("pool_spearman") for record in records]
        ),
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
        "rank1_c_real_state": ("rank1_c_real_state", "mean"),
        "selection_regret": ("selection_regret", "mean"),
        "pool_spearman": ("pool_spearman", "mean"),
        "pairwise_accuracy": ("pairwise_accuracy", "mean"),
        "false_elite_rate": ("false_elite_rate", "mean"),
        "projected_elite_cost_std": ("projected_elite_cost_std", "mean"),
        "projected_dynamic_range": ("projected_dynamic_range", "mean"),
        "action_l2_to_default_blocked": ("action_l2_to_default_blocked", "mean"),
        "action_l2_to_default_raw": ("action_l2_to_default_raw", "mean"),
    }
    out = {name: scalar_summary([nested_get(group, path) for group in seed_groups.values()]) for name, path in paths.items()}
    out["topk_overlap_lewm"] = {
        str(k): scalar_summary([nested_get(group, ("topk_overlap_lewm", str(k), "mean")) for group in seed_groups.values()])
        for k in ENDPOINT_TOPK_VALUES
    }
    out["topk_overlap_v1"] = {
        str(k): scalar_summary([nested_get(group, ("topk_overlap_v1", str(k), "mean")) for group in seed_groups.values()])
        for k in ENDPOINT_TOPK_VALUES
    }
    return out


def aggregate_default_baselines(default_baselines: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(default_baselines)),
        "rank1_success_rate": scalar_summary([record.get("rank1_success") for record in default_baselines]),
        "rank1_c_real_state": scalar_summary([record.get("rank1_c_real_state") for record in default_baselines]),
        "rank1_v1_cost": scalar_summary([record.get("rank1_v1_cost") for record in default_baselines]),
        "selection_regret": scalar_summary([record.get("selection_regret") for record in default_baselines]),
        "oracle_best_c_real_state": scalar_summary([record.get("oracle_best_c_real_state") for record in default_baselines]),
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
    anchor_definitions: dict[str, Any],
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

    subset_pair_ids = {
        name: set(int(item) for item in anchor_definitions[name]["pair_ids"])
        for name in ANCHOR_SUBSET_ORDER
    }
    return {
        "expected_projected_records": int(len(default_baselines) * len(dimensions) * len(projection_seeds)),
        "observed_projected_records": int(len(records)),
        "default_baselines": aggregate_default_baselines(default_baselines),
        "overall": aggregate_record_group(records),
        "by_dimension": by_dimension,
        "by_projection_seed": {
            str(int(seed)): aggregate_record_group([record for record in records if int(record["projection_seed"]) == int(seed)])
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
                    [record for record in records if int(record["dimension"]) == int(dim) and str(record["cell"]) == str(cell)]
                )
                for cell in cells
            }
            for dim in dimensions
        },
        "by_subset": {
            name: aggregate_record_group([record for record in records if int(record["pair_id"]) in pair_ids])
            for name, pair_ids in subset_pair_ids.items()
        },
        "by_dimension_x_subset": {
            str(int(dim)): {
                name: aggregate_record_group(
                    [
                        record
                        for record in records
                        if int(record["dimension"]) == int(dim) and int(record["pair_id"]) in pair_ids
                    ]
                )
                for name, pair_ids in subset_pair_ids.items()
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
                f"{nested_get(seed_stats, ('projected_success_rate', 'mean'))}",
                f"{nested_get(seed_stats, ('pool_spearman', 'mean'))}",
                f"{nested_get(seed_stats, ('selection_regret', 'mean'))}",
                f"{nested_get(seed_stats, ('false_elite_rate', 'mean'))}",
            ]
        )
    headers = ["Dim", "N", "Success", "Spearman", "Regret", "FalseElite"]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print("PushT re-rank-only projection summary")
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
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pool_dir = args.pool_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    pairs_data, all_pairs = load_pairs(args.pairs_path, max_pairs=None, pair_ids=None)
    all_pairs = sorted(all_pairs, key=lambda pair: int(pair["pair_id"]))
    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    if offset % ACTION_BLOCK != 0:
        raise ValueError("Track A offset must be divisible by action_block=5")
    validate_requested_pair_offsets(all_pairs, offset=offset)
    requested_pairs = all_pairs[:SMOKE_PAIR_COUNT] if args.smoke else all_pairs
    dimensions = SMOKE_DIMS if args.smoke else DEFAULT_DIMS
    projection_seeds = SMOKE_PROJECTION_SEEDS if args.smoke else DEFAULT_PROJECTION_SEEDS

    anchor_definitions = anchor_definitions_from_pairs(all_pairs)
    assert_full_anchor_counts(anchor_definitions, full_mode=not args.smoke and len(all_pairs) == 100)

    dataset_path = Path(pair_metadata["dataset_path"])
    cache_dir = dataset_path.parent
    dataset_name = dataset_path.stem
    dataset = get_dataset(cache_dir, dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=int(args.seed),
        ),
        process,
    )
    model = policy.solver.model

    print("== PushT re-rank-only setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output: {args.output}")
    print(f"pool_dir: {args.pool_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"dataset_name: {dataset_name}")
    print(f"cache_dir: {cache_dir}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"mode: {'smoke' if args.smoke else 'full'}")
    print(f"pairs: {len(requested_pairs)}")
    print(f"dimensions: {list(dimensions)}")
    print(f"projection_seeds: {list(projection_seeds)}")
    print(f"expected_simulator_rollouts: {len(requested_pairs) * NUM_SAMPLES}")
    print(f"expected_projected_records: {len(requested_pairs) * len(dimensions) * len(projection_seeds)}")

    total_started = time.time()
    pools, pool_runtime = ensure_pair_pools(
        requested_pairs=requested_pairs,
        pool_dir=args.pool_dir,
        dataset=dataset,
        policy=policy,
        model=model,
        seed=int(args.seed),
    )

    projections = {
        (int(dim), int(seed)): make_projection(int(dim), int(seed))
        for dim in dimensions
        for seed in projection_seeds
    }
    default_baselines = [build_default_baseline_record(pool) for pool in pools]
    records: list[dict[str, Any]] = []
    offline_started = time.time()
    for pool in pools:
        for dim in dimensions:
            for projection_seed in projection_seeds:
                records.append(
                    build_projected_record(
                        pool=pool,
                        dim=int(dim),
                        projection_seed=int(projection_seed),
                        projection=projections[(int(dim), int(projection_seed))],
                    )
                )

    cells = tuple(sorted({str(pair["cell"]) for pair in requested_pairs}))
    aggregate = aggregate_records(
        records=records,
        default_baselines=default_baselines,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        cells=cells,
        anchor_definitions=anchor_definitions,
    )
    total_wallclock = time.time() - total_started
    output = {
        "metadata": {
            "format": "pusht_rerank_only_projected_cem",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "planning_mode": "rerank_default_final_pool",
            "mode": "smoke" if args.smoke else "full",
            "description": (
                "Default PushT LeWM final CEM candidate pools are simulator-scored once per pair; "
                "fixed random projections re-rank only those labelled pools."
            ),
            "protocol_note": (
                "PushT re-rank-only uses the existing PushT Stage 1B CEM horizon of 5 planning blocks. "
                "Cube Stage 1B uses 10 planning blocks because its Track A offset is converted to offset/action_block. "
                "This is intentional and cross-environment comparisons should account for it."
            ),
            "output": str(args.output),
            "pool_dir": str(args.pool_dir),
            "pairs_path": str(args.pairs_path),
            "selected_pair_ids": [int(pair["pair_id"]) for pair in requested_pairs],
            "n_selected_pairs": int(len(requested_pairs)),
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_cache_dir": str(cache_dir),
            "dataset_name": dataset_name,
            "track_a_pair_metadata": pair_metadata,
            "device": args.device,
            "seed": int(args.seed),
            "dimensions": [int(dim) for dim in dimensions],
            "projection_seeds": [int(seed) for seed in projection_seeds],
            "expected_projected_records": int(len(requested_pairs) * len(dimensions) * len(projection_seeds)),
            "expected_simulator_rollouts": {
                "default_final_candidate_pool_scored": int(len(requested_pairs) * NUM_SAMPLES),
                "projected_rank1_extra": 0,
                "total": int(len(requested_pairs) * NUM_SAMPLES),
            },
            "candidate_scoring_mode": "all_300",
            "candidate_pool_original_n": int(NUM_SAMPLES),
            "candidate_pool_scored_n": int(NUM_SAMPLES),
            "cem_config": cem_config_metadata(),
            "success_definition": "block_pos_dist < 20.0 and angle_dist < pi/9",
            "C_real_state_definition": "block_pos_dist + angle_dist",
            "v1_cost_definition": "PushT V1 hinge oracle cost from lewm_audit.eval.oracle_cem.cost_v1_hinge",
            "projection_config": {
                "matrix_shape": "[192, m]",
                "entry_distribution": "Gaussian N(0, 1) / sqrt(m)",
                "fixed_scope": "one matrix per (m, projection_seed), reused across all pairs",
                "matrices": projection_metadata(projections),
            },
            "endpoint_metric_pool": {
                "source": "saved default PushT final CEM iteration 300-candidate pool",
                "topk_values": list(ENDPOINT_TOPK_VALUES),
                "false_elite_k": FALSE_ELITE_K,
                "false_elite_definition": "fraction of projected top-30 not in true C_real_state top-30",
            },
            "anchor_definitions": anchor_definitions,
            "runtime": {
                **pool_runtime,
                "offline_rerank_wallclock_seconds": clean_float(time.time() - offline_started),
                "wallclock_seconds": clean_float(total_wallclock),
                "wallclock_hms": seconds_to_hms(total_wallclock),
            },
        },
        "default_baselines": default_baselines,
        "records": records,
        "aggregate": aggregate,
    }
    if int(aggregate["observed_projected_records"]) != int(aggregate["expected_projected_records"]):
        raise RuntimeError(
            f"Projected record count mismatch: {aggregate['observed_projected_records']} "
            f"!= {aggregate['expected_projected_records']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print()
    print_summary_table(aggregate)
    print(f"\nWallclock seconds: {total_wallclock:.3f} ({seconds_to_hms(total_wallclock)})")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
