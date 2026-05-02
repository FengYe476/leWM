#!/usr/bin/env python3
"""Stage 1A random-geometry controls C0-C5 for Track A endpoint ranking."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LATENT_ARTIFACT = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_c0_c5.json"

EXPECTED_RECORDS = 8000
LATENT_DIM = 192
SEEDS = tuple(range(10))
C2_DIMS = (1, 2, 4, 8, 16, 32, 64, 128, 192)
C3_DIMS = (8, 16, 32, 64)
TOPK_VALUES = (5, 10, 20)
FALSE_ELITE_K = 30
C0_ATOL = 1e-3
C1_ATOL = 1e-4

ANCHOR_DEFINITIONS = {
    "invisible_quadrant": {
        "description": "all_fail + strong_rho pairs",
        "pair_ids": [25, 46, 60, 61, 67, 70, 71, 73, 78, 86, 87, 93, 94, 96, 97, 99],
    },
    "sign_reversal": {
        "description": "negative C_real_z vs C_real_state per-pair Spearman pairs",
        "pair_ids": [15, 17, 18, 20, 21, 22, 23, 29, 33, 37, 40, 42, 44, 45, 47, 49, 53, 62, 66, 69, 98],
    },
    "latent_favorable": {
        "description": "D0xR1 and D1xR0 latent-favorable cells",
        "pair_ids": list(range(6, 12)) + list(range(24, 30)),
        "cells": ["D0xR1", "D1xR0"],
    },
    "D0xR1": {
        "description": "latent-favorable D0xR1 cell",
        "pair_ids": list(range(6, 12)),
        "cells": ["D0xR1"],
    },
    "D1xR0": {
        "description": "latent-favorable D1xR0 cell",
        "pair_ids": list(range(24, 30)),
        "cells": ["D1xR0"],
    },
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def rankdata(values: np.ndarray) -> np.ndarray:
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return pearson_corr(rankdata(x), rankdata(y))


def squared_l2_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum((a - b) ** 2, dim=1)


def deterministic_topk_indices(
    costs: np.ndarray,
    action_ids: np.ndarray,
    mask: np.ndarray,
    k: int,
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    order = np.lexsort((action_ids[indices], costs[indices]))
    return indices[order[:k]]


def pairwise_accuracy(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> dict:
    n_correct = 0.0
    n_total = 0
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        pair_costs = costs[mask]
        pair_labels = labels[mask]
        n = len(pair_costs)
        if n < 2:
            continue
        label_diff = pair_labels[:, None] - pair_labels[None, :]
        cost_diff = pair_costs[:, None] - pair_costs[None, :]
        tri = np.triu(np.ones((n, n), dtype=bool), k=1)
        informative = tri & (label_diff != 0)
        n_total += int(informative.sum())
        if not informative.any():
            continue
        agreed = (label_diff[informative] * cost_diff[informative]) > 0
        tied_cost = cost_diff[informative] == 0
        n_correct += float(agreed.sum()) + 0.5 * float(tied_cost.sum())
    return {
        "value": clean_float(n_correct / n_total) if n_total else None,
        "n_pairs_compared": int(n_total),
    }


def per_pair_spearman(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> dict:
    by_pair = {}
    values = []
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        rho = spearman_corr(costs[mask], labels[mask])
        by_pair[int(pair_id)] = {"spearman": clean_float(rho), "n_records": int(mask.sum())}
        if rho is not None:
            values.append(float(rho))
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": clean_float(arr.mean()) if len(arr) else None,
        "std": clean_float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "n_pairs_used": int(len(arr)),
        "by_pair": by_pair,
    }


def topk_overlap(
    costs: np.ndarray,
    reference_costs: np.ndarray,
    action_ids: np.ndarray,
    pair_ids: np.ndarray,
    k_values: tuple[int, ...],
) -> dict:
    overlaps_by_k: dict[str, list[float]] = {str(k): [] for k in k_values}
    by_pair: dict[str, dict[str, float]] = {}
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        pair_result = {}
        for k in k_values:
            if int(mask.sum()) < k:
                continue
            top_control = deterministic_topk_indices(costs, action_ids, mask, k)
            top_reference = deterministic_topk_indices(reference_costs, action_ids, mask, k)
            overlap = len(set(top_control.tolist()) & set(top_reference.tolist())) / float(k)
            overlaps_by_k[str(k)].append(float(overlap))
            pair_result[str(k)] = float(overlap)
        by_pair[str(int(pair_id))] = pair_result
    return {
        str(k): clean_float(np.mean(overlaps_by_k[str(k)])) if overlaps_by_k[str(k)] else None
        for k in k_values
    } | {"by_pair": by_pair}


def false_elite_rate(
    costs: np.ndarray,
    action_ids: np.ndarray,
    success: np.ndarray,
    pair_ids: np.ndarray,
    k: int,
) -> dict:
    rates = []
    by_pair = {}
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        if int(mask.sum()) < k:
            continue
        top_indices = deterministic_topk_indices(costs, action_ids, mask, k)
        rate = float(np.mean(~success[top_indices]))
        rates.append(rate)
        by_pair[str(int(pair_id))] = {"false_elite_rate": rate, "k": int(k)}
    return {
        "value": clean_float(np.mean(rates)) if rates else None,
        "k": int(k),
        "n_pairs": int(len(rates)),
        "by_pair": by_pair,
    }


def summarize_mask(
    *,
    mask: np.ndarray,
    costs: np.ndarray,
    labels: np.ndarray,
    v1_cost: np.ndarray,
    c0_cost: np.ndarray,
    success: np.ndarray,
    pair_ids: np.ndarray,
    action_ids: np.ndarray,
) -> dict:
    masked_costs = costs[mask]
    masked_labels = labels[mask]
    masked_v1 = v1_cost[mask]
    masked_c0 = c0_cost[mask]
    masked_success = success[mask]
    masked_pair_ids = pair_ids[mask]
    masked_action_ids = action_ids[mask]
    pa = pairwise_accuracy(masked_costs, masked_labels, masked_pair_ids)
    per_pair = per_pair_spearman(masked_costs, masked_labels, masked_pair_ids)
    false_elite = false_elite_rate(
        masked_costs,
        masked_action_ids,
        masked_success,
        masked_pair_ids,
        FALSE_ELITE_K,
    )
    return {
        "n_records": int(mask.sum()),
        "n_pairs": int(len(np.unique(masked_pair_ids))) if mask.any() else 0,
        "global_spearman": clean_float(spearman_corr(masked_costs, masked_labels)),
        "per_pair_spearman": per_pair,
        "pairwise_accuracy": pa["value"],
        "pairwise_comparisons": pa["n_pairs_compared"],
        "topk_overlap_lewm": topk_overlap(
            masked_costs,
            masked_c0,
            masked_action_ids,
            masked_pair_ids,
            TOPK_VALUES,
        ),
        "topk_overlap_v1": topk_overlap(
            masked_costs,
            masked_v1,
            masked_action_ids,
            masked_pair_ids,
            TOPK_VALUES,
        ),
        "false_elite_rate": false_elite["value"],
        "false_elite_k": false_elite["k"],
        "false_elite_by_pair": false_elite["by_pair"],
    }


def compute_metrics(
    *,
    costs: np.ndarray,
    labels: np.ndarray,
    v1_cost: np.ndarray,
    c0_cost: np.ndarray,
    success: np.ndarray,
    pair_ids: np.ndarray,
    action_ids: np.ndarray,
    cells: np.ndarray,
    anchor_masks: dict[str, np.ndarray],
) -> dict:
    full_mask = np.ones(len(costs), dtype=bool)
    metrics = summarize_mask(
        mask=full_mask,
        costs=costs,
        labels=labels,
        v1_cost=v1_cost,
        c0_cost=c0_cost,
        success=success,
        pair_ids=pair_ids,
        action_ids=action_ids,
    )
    metrics["anchors"] = {
        name: summarize_mask(
            mask=mask,
            costs=costs,
            labels=labels,
            v1_cost=v1_cost,
            c0_cost=c0_cost,
            success=success,
            pair_ids=pair_ids,
            action_ids=action_ids,
        )
        for name, mask in anchor_masks.items()
    }
    metrics["per_cell"] = {}
    for cell in sorted(np.unique(cells).tolist()):
        cell_mask = cells == cell
        metrics["per_cell"][str(cell)] = summarize_mask(
            mask=cell_mask,
            costs=costs,
            labels=labels,
            v1_cost=v1_cost,
            c0_cost=c0_cost,
            success=success,
            pair_ids=pair_ids,
            action_ids=action_ids,
        )
    return metrics


def scalar_mean_std(values: list[float | None]) -> dict:
    arr = np.asarray([float(value) for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    return {
        "mean": clean_float(arr.mean()) if len(arr) else None,
        "std": clean_float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "n": int(len(arr)),
        "ddof": 1,
    }


def aggregate_metric_list(metrics_list: list[dict]) -> dict:
    aggregate = {
        "global_spearman": scalar_mean_std([item.get("global_spearman") for item in metrics_list]),
        "pairwise_accuracy": scalar_mean_std([item.get("pairwise_accuracy") for item in metrics_list]),
        "false_elite_rate": scalar_mean_std([item.get("false_elite_rate") for item in metrics_list]),
        "per_pair_spearman_mean": scalar_mean_std(
            [item.get("per_pair_spearman", {}).get("mean") for item in metrics_list]
        ),
        "per_pair_spearman_std": scalar_mean_std(
            [item.get("per_pair_spearman", {}).get("std") for item in metrics_list]
        ),
        "topk_overlap_lewm": {
            str(k): scalar_mean_std([item.get("topk_overlap_lewm", {}).get(str(k)) for item in metrics_list])
            for k in TOPK_VALUES
        },
        "topk_overlap_v1": {
            str(k): scalar_mean_std([item.get("topk_overlap_v1", {}).get(str(k)) for item in metrics_list])
            for k in TOPK_VALUES
        },
    }
    anchor_names = sorted(metrics_list[0].get("anchors", {}).keys()) if metrics_list else []
    aggregate["anchors"] = {
        name: aggregate_metric_list([item["anchors"][name] for item in metrics_list])
        for name in anchor_names
    }
    cell_names = sorted(metrics_list[0].get("per_cell", {}).keys()) if metrics_list else []
    aggregate["per_cell"] = {
        cell: aggregate_metric_list([item["per_cell"][cell] for item in metrics_list])
        for cell in cell_names
    }
    return aggregate


def summary_row(
    *,
    control: str,
    config: str,
    n_seeds: int,
    metrics: dict | None = None,
    aggregate: dict | None = None,
) -> dict:
    if metrics is not None:
        return {
            "control": control,
            "config": config,
            "n_seeds": int(n_seeds),
            "global_spearman_mean": metrics.get("global_spearman"),
            "global_spearman_std": None,
            "pairwise_accuracy_mean": metrics.get("pairwise_accuracy"),
            "pairwise_accuracy_std": None,
            "per_pair_rho_mean": metrics.get("per_pair_spearman", {}).get("mean"),
            "per_pair_rho_mean_std": None,
            "false_elite_rate_mean": metrics.get("false_elite_rate"),
            "false_elite_rate_std": None,
        }
    if aggregate is None:
        raise ValueError("Either metrics or aggregate is required")
    return {
        "control": control,
        "config": config,
        "n_seeds": int(n_seeds),
        "global_spearman_mean": aggregate["global_spearman"]["mean"],
        "global_spearman_std": aggregate["global_spearman"]["std"],
        "pairwise_accuracy_mean": aggregate["pairwise_accuracy"]["mean"],
        "pairwise_accuracy_std": aggregate["pairwise_accuracy"]["std"],
        "per_pair_rho_mean": aggregate["per_pair_spearman_mean"]["mean"],
        "per_pair_rho_mean_std": aggregate["per_pair_spearman_mean"]["std"],
        "false_elite_rate_mean": aggregate["false_elite_rate"]["mean"],
        "false_elite_rate_std": aggregate["false_elite_rate"]["std"],
    }


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def print_summary_table(rows: list[dict]) -> None:
    headers = [
        "Control",
        "Config",
        "Seeds",
        "Spearman",
        "Pairwise",
        "PerPairRho",
        "FalseElite",
    ]
    table = []
    for row in rows:
        table.append(
            [
                str(row["control"]),
                str(row["config"]),
                str(row["n_seeds"]),
                f"{fmt(row['global_spearman_mean'])}/{fmt(row['global_spearman_std'])}",
                f"{fmt(row['pairwise_accuracy_mean'])}/{fmt(row['pairwise_accuracy_std'])}",
                f"{fmt(row['per_pair_rho_mean'])}/{fmt(row['per_pair_rho_mean_std'])}",
                f"{fmt(row['false_elite_rate_mean'])}/{fmt(row['false_elite_rate_std'])}",
            ]
        )
    widths = [max(len(headers[i]), *(len(record[i]) for record in table)) for i in range(len(headers))]
    print("Stage 1A C0-C5 summary (mean/std for multi-seed controls)")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def load_latent_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing latent artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "z_terminal",
        "z_goal",
        "C_real_state",
        "C_real_z",
        "v1_cost",
        "success",
        "pair_id",
        "action_id",
        "source",
        "cell",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(f"Latent artifact missing required keys: {missing}")
    n_records = int(artifact["pair_id"].numel())
    if n_records != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} records, found {n_records}")
    if tuple(artifact["z_terminal"].shape) != (EXPECTED_RECORDS, LATENT_DIM):
        raise ValueError(f"Unexpected z_terminal shape: {tuple(artifact['z_terminal'].shape)}")
    if tuple(artifact["z_goal"].shape) != (EXPECTED_RECORDS, LATENT_DIM):
        raise ValueError(f"Unexpected z_goal shape: {tuple(artifact['z_goal'].shape)}")
    for key in ("C_real_state", "C_real_z", "v1_cost", "success", "action_id"):
        if int(artifact[key].numel()) != EXPECTED_RECORDS:
            raise ValueError(f"Unexpected {key} length: {int(artifact[key].numel())}")
    for key in ("source", "cell"):
        if len(artifact[key]) != EXPECTED_RECORDS:
            raise ValueError(f"Unexpected {key} length: {len(artifact[key])}")
    return artifact


def make_anchor_masks(pair_ids: np.ndarray, cells: np.ndarray) -> dict[str, np.ndarray]:
    masks = {}
    for name, definition in ANCHOR_DEFINITIONS.items():
        mask = np.ones(len(pair_ids), dtype=bool)
        if "pair_ids" in definition:
            mask &= np.isin(pair_ids, np.asarray(definition["pair_ids"], dtype=np.int64))
        if "cells" in definition:
            mask &= np.isin(cells, np.asarray(definition["cells"], dtype=object))
        masks[name] = mask
    return masks


def run_single_metrics(
    *,
    costs: torch.Tensor,
    labels: np.ndarray,
    v1_cost: np.ndarray,
    c0_cost: np.ndarray,
    success: np.ndarray,
    pair_ids: np.ndarray,
    action_ids: np.ndarray,
    cells: np.ndarray,
    anchor_masks: dict[str, np.ndarray],
) -> dict:
    return compute_metrics(
        costs=costs.detach().cpu().numpy().astype(np.float64),
        labels=labels,
        v1_cost=v1_cost,
        c0_cost=c0_cost,
        success=success,
        pair_ids=pair_ids,
        action_ids=action_ids,
        cells=cells,
        anchor_masks=anchor_masks,
    )


def orthogonal_matrix(dim: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((dim, dim), generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    artifact = load_latent_artifact(args.latent_artifact)
    z_terminal = artifact["z_terminal"].detach().cpu().to(dtype=torch.float32)
    z_goal = artifact["z_goal"].detach().cpu().to(dtype=torch.float32)
    z_terminal64 = z_terminal.to(dtype=torch.float64)
    z_goal64 = z_goal.to(dtype=torch.float64)
    labels = artifact["C_real_state"].detach().cpu().numpy().astype(np.float64)
    artifact_c_real_z = artifact["C_real_z"].detach().cpu().to(dtype=torch.float32)
    v1_cost = artifact["v1_cost"].detach().cpu().numpy().astype(np.float64)
    success = artifact["success"].detach().cpu().numpy().astype(bool)
    pair_ids = artifact["pair_id"].detach().cpu().numpy().astype(np.int64)
    action_ids = artifact["action_id"].detach().cpu().numpy().astype(np.int64)
    cells = np.asarray(artifact["cell"], dtype=object)
    anchor_masks = make_anchor_masks(pair_ids, cells)

    c0_cost_t = squared_l2_torch(z_terminal, z_goal)
    c0_cost64_t = squared_l2_torch(z_terminal64, z_goal64)
    c0_cost = c0_cost_t.detach().cpu().numpy().astype(np.float64)
    c0_diff = torch.abs(c0_cost_t - artifact_c_real_z)
    c0_validation = {
        "max_abs_diff": float(c0_diff.max().item()),
        "mean_abs_diff": float(c0_diff.mean().item()),
        "atol": C0_ATOL,
        "passed": bool(torch.allclose(c0_cost_t, artifact_c_real_z, atol=C0_ATOL, rtol=0.0)),
    }
    if not c0_validation["passed"]:
        raise RuntimeError(f"C0 validation failed: {c0_validation}")

    common = {
        "labels": labels,
        "v1_cost": v1_cost,
        "c0_cost": c0_cost,
        "success": success,
        "pair_ids": pair_ids,
        "action_ids": action_ids,
        "cells": cells,
        "anchor_masks": anchor_masks,
    }

    controls: dict[str, Any] = {}
    summary_rows: list[dict] = []

    print("Computing C0...")
    c0_metrics = run_single_metrics(costs=c0_cost_t, **common)
    controls["C0"] = {
        "name": "LeWM reference",
        "config": {"dim": LATENT_DIM},
        "validation": c0_validation,
        "metrics": c0_metrics,
    }
    summary_rows.append(summary_row(control="C0", config="lewm_192", n_seeds=1, metrics=c0_metrics))

    print("Computing C1 orthogonal controls...")
    c1_per_seed = []
    c1_validation = []
    for seed in SEEDS:
        q = orthogonal_matrix(LATENT_DIM, seed, dtype=torch.float64)
        cost = squared_l2_torch(z_terminal64 @ q, z_goal64 @ q)
        diff = torch.abs(cost - c0_cost64_t)
        validation = {
            "seed": int(seed),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "atol": C1_ATOL,
            "passed": bool(torch.allclose(cost, c0_cost64_t, atol=C1_ATOL, rtol=0.0)),
        }
        if not validation["passed"]:
            raise AssertionError(f"C1 must match C0; seed={seed} validation={validation}")
        c1_validation.append(validation)
        c1_per_seed.append({"seed": int(seed), "metrics": run_single_metrics(costs=cost, **common)})
    c1_aggregate = aggregate_metric_list([item["metrics"] for item in c1_per_seed])
    controls["C1"] = {
        "name": "same-dim orthogonal",
        "config": {"dim": LATENT_DIM, "n_seeds": len(SEEDS)},
        "validation": c1_validation,
        "per_seed": c1_per_seed,
        "aggregate": c1_aggregate,
    }
    summary_rows.append(summary_row(control="C1", config="orthogonal_192", n_seeds=len(SEEDS), aggregate=c1_aggregate))

    print("Computing C2 Gaussian dimension ladder...")
    controls["C2"] = {
        "name": "Gaussian projection dimension ladder",
        "config": {
            "dims": list(C2_DIMS),
            "n_seeds_per_dim": len(SEEDS),
            "scaling": "torch.randn(192, m) / sqrt(m), entry variance 1/m",
        },
        "by_dim": {},
    }
    for dim in C2_DIMS:
        per_seed = []
        for seed in SEEDS:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            projection = torch.randn((LATENT_DIM, dim), generator=generator, dtype=torch.float32) / math.sqrt(dim)
            cost = squared_l2_torch(z_terminal @ projection, z_goal @ projection)
            per_seed.append({"seed": int(seed), "metrics": run_single_metrics(costs=cost, **common)})
        aggregate = aggregate_metric_list([item["metrics"] for item in per_seed])
        controls["C2"]["by_dim"][str(dim)] = {
            "dim": int(dim),
            "per_seed": per_seed,
            "aggregate": aggregate,
        }
        summary_rows.append(summary_row(control="C2", config=f"gaussian_m={dim}", n_seeds=len(SEEDS), aggregate=aggregate))

    print("Computing C3 coordinate subset controls...")
    controls["C3"] = {
        "name": "coordinate subset",
        "config": {"dims": list(C3_DIMS), "n_seeds_per_dim": len(SEEDS)},
        "by_dim": {},
    }
    for dim in C3_DIMS:
        per_seed = []
        for seed in SEEDS:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            indices = torch.randperm(LATENT_DIM, generator=generator)[:dim]
            cost = squared_l2_torch(z_terminal[:, indices], z_goal[:, indices])
            per_seed.append(
                {
                    "seed": int(seed),
                    "indices": [int(item) for item in indices.tolist()],
                    "metrics": run_single_metrics(costs=cost, **common),
                }
            )
        aggregate = aggregate_metric_list([item["metrics"] for item in per_seed])
        controls["C3"]["by_dim"][str(dim)] = {
            "dim": int(dim),
            "per_seed": per_seed,
            "aggregate": aggregate,
        }
        summary_rows.append(summary_row(control="C3", config=f"coords_m={dim}", n_seeds=len(SEEDS), aggregate=aggregate))

    print("Computing C4 Gaussian null controls...")
    c4_per_seed = []
    for seed in SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        z_null_terminal = torch.randn((EXPECTED_RECORDS, LATENT_DIM), generator=generator, dtype=torch.float32)
        z_null_goal = torch.randn((EXPECTED_RECORDS, LATENT_DIM), generator=generator, dtype=torch.float32)
        cost = squared_l2_torch(z_null_terminal, z_null_goal)
        c4_per_seed.append({"seed": int(seed), "metrics": run_single_metrics(costs=cost, **common)})
    c4_aggregate = aggregate_metric_list([item["metrics"] for item in c4_per_seed])
    controls["C4"] = {
        "name": "Gaussian null",
        "config": {"dim": LATENT_DIM, "n_seeds": len(SEEDS)},
        "per_seed": c4_per_seed,
        "aggregate": c4_aggregate,
    }
    summary_rows.append(summary_row(control="C4", config="gaussian_null_192", n_seeds=len(SEEDS), aggregate=c4_aggregate))

    print("Computing C5 shuffled latent controls...")
    c5_per_seed = []
    for seed in SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        perm1 = torch.randperm(EXPECTED_RECORDS, generator=generator)
        perm2 = torch.randperm(EXPECTED_RECORDS, generator=generator)
        cost = squared_l2_torch(z_terminal[perm1], z_goal[perm2])
        c5_per_seed.append({"seed": int(seed), "metrics": run_single_metrics(costs=cost, **common)})
    c5_aggregate = aggregate_metric_list([item["metrics"] for item in c5_per_seed])
    controls["C5"] = {
        "name": "shuffled latent",
        "config": {"dim": LATENT_DIM, "n_seeds": len(SEEDS), "permutation_scope": "all 8000 rows, independent terminal and goal permutations"},
        "per_seed": c5_per_seed,
        "aggregate": c5_aggregate,
    }
    summary_rows.append(summary_row(control="C5", config="independent_row_shuffle", n_seeds=len(SEEDS), aggregate=c5_aggregate))

    output = {
        "metadata": {
            "format": "stage1a_c0_c5_controls",
            "created_at": iso_now(),
            "latent_artifact": str(args.latent_artifact),
            "n_records": EXPECTED_RECORDS,
            "latent_dim": LATENT_DIM,
            "seeds": list(SEEDS),
            "c2_dims": list(C2_DIMS),
            "c3_dims": list(C3_DIMS),
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "anchor_definitions": ANCHOR_DEFINITIONS,
            "tie_rules": {
                "pairwise_accuracy": "Skip C_real_state ties; score tied control costs as 0.5 when C_real_state differs.",
                "topk_and_false_elite": "Ascending cost ranking with deterministic action_id tie-break.",
            },
            "topk_overlap_definition": "|topk(control) intersection topk(reference)| / k, averaged over pairs.",
            "c2_scaling": "P = torch.randn(192, m) / sqrt(m), entry mean 0 and variance 1/m.",
            "aggregate_std": {"ddof": 1, "description": "sample standard deviation across seeds"},
            "c0_validation": c0_validation,
            "c1_validation": c1_validation,
        },
        "summary_table": summary_rows,
        "controls": controls,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_table(summary_rows)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
