"""DP1 and sign-reversal diagnostics for Track A three-cost outputs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


PHASE0_REFERENCE = {"mean": 0.353, "std": 0.486}


def _action_value(action: Mapping, key: str) -> float:
    if key in action:
        return float(action[key])
    lower = key[:1].lower() + key[1:]
    if lower in action:
        return float(action[lower])
    raise KeyError(f"Action record is missing {key!r}")


def _pair_id(pair_id: int | str) -> int | str:
    try:
        return int(pair_id)
    except (TypeError, ValueError):
        return pair_id


def _pair_actions(pair_or_actions) -> list[dict]:
    if isinstance(pair_or_actions, Mapping) and "actions" in pair_or_actions:
        return list(pair_or_actions["actions"])
    return list(pair_or_actions)


def _rankdata(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[sorter[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return _pearson_corr(_rankdata(x), _rankdata(y))


def per_pair_spearman(records_by_pair: Mapping[int | str, object]) -> dict[int | str, float | None]:
    """Compute Spearman corr(C_real_z, C_real_state) for each pair."""
    out = {}
    for pair_id, pair_or_actions in records_by_pair.items():
        actions = _pair_actions(pair_or_actions)
        c_real_z = np.asarray([_action_value(action, "C_real_z") for action in actions], dtype=np.float64)
        c_real_state = np.asarray(
            [_action_value(action, "C_real_state") for action in actions], dtype=np.float64
        )
        out[_pair_id(pair_id)] = spearman_corr(c_real_z, c_real_state)
    return out


def _finite_corr_values(per_pair_corrs: Mapping[int | str, float | None] | list[float]) -> np.ndarray:
    if isinstance(per_pair_corrs, Mapping):
        raw_values = per_pair_corrs.values()
    else:
        raw_values = per_pair_corrs
    values = np.asarray([value for value in raw_values if value is not None], dtype=np.float64)
    return values[np.isfinite(values)]


def dp1_test(
    per_pair_corrs: Mapping[int | str, float | None] | list[float],
    std_threshold: float = 0.3,
    n_bootstrap: int = 10000,
    ci_alpha: float = 0.05,
    rng_seed: int = 0,
) -> dict:
    """Bootstrap the per-pair correlation standard deviation for DP1."""
    values = _finite_corr_values(per_pair_corrs)
    if len(values) < 2:
        raise ValueError("DP1 requires at least two finite per-pair correlations")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not (0.0 < ci_alpha < 1.0):
        raise ValueError("ci_alpha must be between 0 and 1")

    rng = np.random.default_rng(rng_seed)
    sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot_values = values[sample_indices]
    boot_std = np.std(boot_values, axis=1, ddof=1)
    ci_low, ci_high = np.quantile(boot_std, [ci_alpha / 2.0, 1.0 - ci_alpha / 2.0])

    if ci_low >= std_threshold:
        verdict = "pass"
    elif ci_high < std_threshold:
        verdict = "fail"
    else:
        verdict = "ambiguous"

    return {
        "n_pairs_used": int(len(values)),
        "mean_rho": float(np.mean(values)),
        "std_rho": float(np.std(values, ddof=1)),
        "median_rho": float(np.median(values)),
        "min_rho": float(np.min(values)),
        "max_rho": float(np.max(values)),
        "ci_low_std": float(ci_low),
        "ci_high_std": float(ci_high),
        "threshold": float(std_threshold),
        "verdict": verdict,
        "phase0_reference": dict(PHASE0_REFERENCE),
    }


def sign_reversal_cluster(
    records_by_pair: Mapping[int | str, Mapping],
    neg_threshold: float = 0.0,
) -> list[dict]:
    """Return pairs whose per-pair Spearman correlation is below `neg_threshold`."""
    corr_by_pair = per_pair_spearman(records_by_pair)
    cluster = []
    for pair_id, rho in corr_by_pair.items():
        if rho is None or not np.isfinite(rho) or rho >= neg_threshold:
            continue
        pair = records_by_pair[pair_id]
        success_count = int(sum(bool(action["success"]) for action in pair["actions"]))
        cluster.append(
            {
                "pair_id": int(pair["pair_id"]),
                "cell": str(pair["cell"]),
                "rho": float(rho),
                "success_count": success_count,
                "block_displacement_px": float(pair["block_displacement_px"]),
                "required_rotation_rad": float(pair["required_rotation_rad"]),
                "episode_id": int(pair["episode_id"]),
                "start_row": int(pair["start_row"]),
                "goal_row": int(pair["goal_row"]),
            }
        )
    return sorted(cluster, key=lambda item: (item["rho"], item["pair_id"]))
