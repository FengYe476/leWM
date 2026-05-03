#!/usr/bin/env python3
"""Cube Stage 1A random-geometry controls C0-C5.

This is the Cube analogue of ``scripts/phase2/stage1/stage1a_controls.py``.
It intentionally reuses the PushT Stage 1A metric helpers so ranking, tie, and
aggregation definitions remain comparable across environments.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    FALSE_ELITE_K,
    TOPK_VALUES,
    aggregate_metric_list,
    compute_metrics,
    iso_now,
    jsonable,
    orthogonal_matrix,
    squared_l2_torch,
    summary_row,
)


DEFAULT_LATENT_ARTIFACT = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_latents.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1a.json"

EXPECTED_RECORDS = 8000
EXPECTED_PAIRS = 100
LATENT_DIM = 192
SEEDS = tuple(range(10))
C2_DIMS = (1, 2, 4, 8, 16, 32, 64, 128, 192)
C3_DIMS = (8, 16, 32, 64)
C0_ATOL = 1e-3
C1_ATOL = 1e-4
ALIAS_ATOL = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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
    print("Cube Stage 1A C0-C5 summary (mean/std for multi-seed controls)")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def load_latent_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Cube latent artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "z_terminal",
        "z_goal",
        "C_real_state",
        "C_real_z",
        "cube_pos_dist",
        "v1_cost",
        "success",
        "pair_id",
        "action_id",
        "source",
        "cell",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(f"Cube latent artifact missing required keys: {missing}")

    n_records = int(artifact["pair_id"].numel())
    if n_records != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} records, found {n_records}")
    if tuple(artifact["z_terminal"].shape) != (EXPECTED_RECORDS, LATENT_DIM):
        raise ValueError(f"Unexpected z_terminal shape: {tuple(artifact['z_terminal'].shape)}")
    if tuple(artifact["z_goal"].shape) != (EXPECTED_RECORDS, LATENT_DIM):
        raise ValueError(f"Unexpected z_goal shape: {tuple(artifact['z_goal'].shape)}")
    for key in ("C_real_state", "C_real_z", "cube_pos_dist", "v1_cost", "success", "action_id"):
        if int(artifact[key].numel()) != EXPECTED_RECORDS:
            raise ValueError(f"Unexpected {key} length: {int(artifact[key].numel())}")
    for key in ("source", "cell"):
        if len(artifact[key]) != EXPECTED_RECORDS:
            raise ValueError(f"Unexpected {key} length: {len(artifact[key])}")

    n_pairs = len(torch.unique(artifact["pair_id"]))
    if n_pairs != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} pairs, found {n_pairs}")
    return artifact


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.to(dtype=torch.float64) - right.to(dtype=torch.float64))).item())


def validate_aliases(artifact: dict[str, Any]) -> dict[str, Any]:
    c_real_state = artifact["C_real_state"].detach().cpu()
    cube_pos_dist = artifact["cube_pos_dist"].detach().cpu()
    v1_cost = artifact["v1_cost"].detach().cpu()
    checks = {
        "v1_cost_vs_C_real_state": max_abs_diff(v1_cost, c_real_state),
        "v1_cost_vs_cube_pos_dist": max_abs_diff(v1_cost, cube_pos_dist),
        "C_real_state_vs_cube_pos_dist": max_abs_diff(c_real_state, cube_pos_dist),
    }
    print("Cube alias validation:")
    for name, diff in checks.items():
        print(f"  {name} max_abs_diff={diff:.12g}")
    passed = all(diff <= ALIAS_ATOL for diff in checks.values())
    validation = {"max_abs_diff": checks, "atol": ALIAS_ATOL, "passed": passed}
    if not passed:
        raise RuntimeError(f"Cube cost alias validation failed: {validation}")
    return validation


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


def cell_counts(cells: np.ndarray) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in Counter(cells.tolist()).items()))


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    artifact = load_latent_artifact(args.latent_artifact)
    alias_validation = validate_aliases(artifact)

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
    anchor_masks: dict[str, np.ndarray] = {}

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
    print(
        "C0 validation: "
        f"max_abs_diff={c0_validation['max_abs_diff']:.12g} "
        f"mean_abs_diff={c0_validation['mean_abs_diff']:.12g} "
        f"atol={C0_ATOL:g} passed={c0_validation['passed']}"
    )
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
        "name": "Cube LeWM reference",
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
        "config": {
            "dim": LATENT_DIM,
            "n_seeds": len(SEEDS),
            "permutation_scope": "all 8000 rows, independent terminal and goal permutations",
        },
        "per_seed": c5_per_seed,
        "aggregate": c5_aggregate,
    }
    summary_rows.append(summary_row(control="C5", config="independent_row_shuffle", n_seeds=len(SEEDS), aggregate=c5_aggregate))

    output = {
        "metadata": {
            "format": "cube_stage1a_c0_c5_controls",
            "created_at": iso_now(),
            "latent_artifact": str(args.latent_artifact),
            "n_records": EXPECTED_RECORDS,
            "n_pairs": EXPECTED_PAIRS,
            "latent_dim": LATENT_DIM,
            "target_metric": "C_real_state",
            "lower_cost_is_better": True,
            "seeds": list(SEEDS),
            "c2_dims": list(C2_DIMS),
            "c3_dims": list(C3_DIMS),
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "cell_counts": cell_counts(cells),
            "action_dim": {
                "raw": artifact.get("metadata", {}).get("raw_action_dim"),
                "blocked": artifact.get("metadata", {}).get("blocked_action_dim"),
                "action_block": artifact.get("metadata", {}).get("action_block"),
            },
            "success_definition": artifact.get("metadata", {}).get(
                "success_definition",
                "position-only: ||terminal_cube_pos - goal_cube_pos||_2 <= 0.04",
            ),
            "v1_cost_definition": artifact.get("metadata", {}).get(
                "v1_cost_definition",
                "Cube v1_cost aliases C_real_state / cube_pos_dist.",
            ),
            "alias_validation": alias_validation,
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
