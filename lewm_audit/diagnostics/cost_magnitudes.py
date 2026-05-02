"""Cost-magnitude summaries for Track A D-row supplementary analysis."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np


PUSHT_BLOCK_POSITION_SUCCESS_THRESHOLD_PX = 20.0
PUSHT_ANGLE_SUCCESS_THRESHOLD_RAD = math.pi / 9.0
PUSHT_C_REAL_STATE_SUCCESS_UPPER_BOUND = (
    PUSHT_BLOCK_POSITION_SUCCESS_THRESHOLD_PX + PUSHT_ANGLE_SUCCESS_THRESHOLD_RAD
)
COST_KEYS = ("C_real_z", "C_model", "C_real_state")


def _finite(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def summary_stats(values) -> dict:
    arr = _finite(values)
    if len(arr) == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def pearson_corr(x, y) -> float | None:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def infer_c_real_state_success_threshold(records_by_pair: Mapping[int, dict]) -> dict:
    """Return the scalar C_real_state threshold proxy used for magnitude counts.

    PushT success in the three-cost code is conjunctive:
    block_pos_dist < 20.0 and angle_dist < pi/9. The stored C_real_state is
    block_pos_dist + angle_dist, so no unique scalar C_real_state threshold is
    equivalent to success. We therefore expose the direct conjunctive criterion
    and use the max observed successful C_real_state as a finite upper-bound
    proxy for row-level counts.
    """
    successful_costs = []
    for pair in records_by_pair.values():
        for action in pair["actions"]:
            if bool(action["success"]):
                successful_costs.append(float(action["C_real_state"]))
    empirical = max(successful_costs) if successful_costs else None
    threshold = empirical if empirical is not None else PUSHT_C_REAL_STATE_SUCCESS_UPPER_BOUND
    return {
        "threshold": float(threshold),
        "method": "empirical_max_successful_C_real_state_upper_bound",
        "n_success_records": int(len(successful_costs)),
        "empirical_max_successful_C_real_state": None if empirical is None else float(empirical),
        "direct_success_criterion": {
            "source": "lewm_audit.diagnostics.three_cost.block_pose_metrics",
            "block_pos_dist_lt_px": PUSHT_BLOCK_POSITION_SUCCESS_THRESHOLD_PX,
            "angle_dist_lt_rad": PUSHT_ANGLE_SUCCESS_THRESHOLD_RAD,
            "c_real_state_definition": "block_pos_dist + angle_dist",
            "c_real_state_direct_upper_bound": PUSHT_C_REAL_STATE_SUCCESS_UPPER_BOUND,
        },
        "unique_scalar_threshold": False,
        "note": (
            "C_real_state is a sum while success is a conjunction, so this scalar "
            "threshold is an observed upper-bound proxy, not an equivalent success rule."
        ),
    }


def _selected_pairs(
    records_by_pair: Mapping[int, dict],
    row_filter_fn: Callable[[dict], bool],
) -> list[dict]:
    return [pair for pair in records_by_pair.values() if row_filter_fn(pair)]


def _all_actions(pairs: list[dict]) -> list[dict]:
    return [action for pair in pairs for action in pair["actions"]]


def _dynamic_range(values) -> float | None:
    arr = _finite(values)
    if len(arr) == 0:
        return None
    return float(np.max(arr) - np.min(arr))


def pairwise_cost_correlations(records_by_pair, row_filter_fn) -> dict:
    pairs = _selected_pairs(records_by_pair, row_filter_fn)
    actions = _all_actions(pairs)
    values = {key: [float(action[key]) for action in actions] for key in COST_KEYS}
    return {
        "pearson_C_model_vs_C_real_z": pearson_corr(values["C_model"], values["C_real_z"]),
        "pearson_C_real_z_vs_C_real_state": pearson_corr(
            values["C_real_z"],
            values["C_real_state"],
        ),
        "pearson_C_model_vs_C_real_state": pearson_corr(
            values["C_model"],
            values["C_real_state"],
        ),
    }


def per_row_cost_stats(
    records_by_pair,
    row_filter_fn,
    success_threshold: float | None = None,
) -> dict:
    pairs = _selected_pairs(records_by_pair, row_filter_fn)
    actions = _all_actions(pairs)
    cost_values = {key: [float(action[key]) for action in actions] for key in COST_KEYS}
    best_costs = [
        float(min(float(action["C_real_state"]) for action in pair["actions"]))
        for pair in pairs
        if pair.get("actions")
    ]
    if success_threshold is None:
        success_threshold = math.nan
    threshold_is_finite = math.isfinite(float(success_threshold))

    return {
        "n_pairs": int(len(pairs)),
        "n_records": int(len(actions)),
        "C_real_z": summary_stats(cost_values["C_real_z"]),
        "C_model": summary_stats(cost_values["C_model"]),
        "C_real_state": summary_stats(cost_values["C_real_state"]),
        "C_model_dynamic_range": _dynamic_range(cost_values["C_model"]),
        "C_real_z_dynamic_range": _dynamic_range(cost_values["C_real_z"]),
        "best_C_real_state_per_pair": best_costs,
        "median_best_C_real_state": (
            float(np.median(np.asarray(best_costs, dtype=np.float64))) if best_costs else None
        ),
        "n_pairs_with_min_C_real_state_below_success_threshold": (
            int(sum(value < float(success_threshold) for value in best_costs))
            if threshold_is_finite
            else 0
        ),
        "pairwise_pearson": pairwise_cost_correlations(records_by_pair, row_filter_fn),
    }
