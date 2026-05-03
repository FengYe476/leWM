#!/usr/bin/env python3
"""Build the Block 3 five-case taxonomy table from offline artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAP_TABLE = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "cem_gap_table.json"
DEFAULT_CUBE_STAGE1A = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1a.json"
DEFAULT_CUBE_FULL = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "cube_full_proj_cem.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "taxonomy_table.json"

ENDPOINT_HIGH = 0.30
ENDPOINT_LOW = 0.15
POOL_HIGH = 0.15
RANK1_HIGH = 0.50
POOL_SUCCESS_HIGH = 0.30
CASE_E_NEAR_ZERO_TOL = 0.16

CASE_DEFINITIONS = {
    "A": {
        "endpoint": "high",
        "pool": "high",
        "pool_success_mass": "any",
        "rank1": "high",
        "interpretation": "Endpoint geometry transfers to planning.",
    },
    "B": {
        "endpoint": "high",
        "pool": "low_or_near_zero",
        "pool_success_mass": "high",
        "rank1": "high",
        "interpretation": "Pool already good; ranking less important.",
    },
    "C": {
        "endpoint": "high",
        "pool": "low_or_near_zero",
        "pool_success_mass": "low",
        "rank1": "low",
        "interpretation": "Endpoint geometry fails as planning signal.",
    },
    "D": {
        "endpoint": "low",
        "pool": "high",
        "pool_success_mass": "any",
        "rank1": "high",
        "interpretation": "Local CEM geometry emerges despite poor global metric.",
    },
    "E": {
        "endpoint": "medium",
        "pool": "near_zero",
        "pool_success_mass": "locally_sufficient",
        "rank1": "high",
        "interpretation": "Rank-insensitive success in a locally sufficient pool.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-table", type=Path, default=DEFAULT_GAP_TABLE)
    parser.add_argument("--cube-stage1a", type=Path, default=DEFAULT_CUBE_STAGE1A)
    parser.add_argument("--cube-full", type=Path, default=DEFAULT_CUBE_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stat_mean(value: Any) -> float | None:
    if isinstance(value, dict):
        if "mean" in value:
            return clean_float(value.get("mean"))
        if "value" in value:
            return clean_float(value.get("value"))
        return None
    return clean_float(value)


def nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def load_stage1a_endpoint_by_dim(data: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for dim, group in data.get("controls", {}).get("C2", {}).get("by_dim", {}).items():
        out[int(dim)] = stat_mean(nested_get(group, ("aggregate", "global_spearman")))
        if out[int(dim)] is None:
            per_seed = group.get("per_seed", [])
            values = [stat_mean(nested_get(row, ("metrics", "global_spearman"))) for row in per_seed]
            values = [float(value) for value in values if value is not None]
            out[int(dim)] = float(np.mean(values)) if values else None
    missing = [dim for dim, value in out.items() if value is None]
    if missing:
        raise RuntimeError(f"Missing Stage 1A endpoint Spearman for dimensions: {missing}")
    return {dim: float(value) for dim, value in out.items() if value is not None}


def cube_full_gap_rows(cube_full: dict[str, Any], cube_endpoint: dict[int, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_pool_success = stat_mean(
        nested_get(cube_full, ("aggregate", "default_baselines", "candidate_pool_success_rate"))
    )
    for dim_str, group in sorted(
        cube_full.get("aggregate", {}).get("by_dimension", {}).items(),
        key=lambda item: int(item[0]),
    ):
        dim = int(dim_str)
        r_endpoint = cube_endpoint[dim]
        r_pool = stat_mean(group.get("endpoint_spearman"))
        delta = clean_float(float(r_endpoint) - float(r_pool)) if r_pool is not None else None
        rows.append(
            {
                "environment": "Cube",
                "protocol": "full_projected_cem",
                "dimension": dim,
                "n_records": int(group.get("n_records", 0)),
                "n_pairs": int(cube_full.get("metadata", {}).get("n_selected_pairs", 0)),
                "R_endpoint": clean_float(r_endpoint),
                "R_pool": clean_float(r_pool),
                "Delta_CEM": delta,
                "rho_ret": (
                    clean_float(float(r_pool) / float(r_endpoint))
                    if r_endpoint is not None and r_endpoint > 0.1 and r_pool is not None
                    else None
                ),
                "M_rank1": stat_mean(group.get("rank1_success_rate"))
                or stat_mean(group.get("projected_success_rate")),
                "M_pool_success": default_pool_success,
                "regret": None,
                "metric_notes": [
                    "Cube full projected CEM simulator-scored only rank-1 actions.",
                    "M_pool_success reuses the matched default final-pool labels from cube_stage1b.json.",
                ],
                "source_paths": {
                    "stage1a": "results/phase2/cube/cube_stage1a.json",
                    "planning": "results/phase2/protocol_match/cube_full_proj_cem.json",
                },
                "source": "derived_from_cube_full_projected_cem",
            }
        )
    return rows


def endpoint_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value > ENDPOINT_HIGH:
        return "high"
    if value < ENDPOINT_LOW:
        return "low"
    return "medium"


def pool_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "high" if value > POOL_HIGH else "low_or_near_zero"


def rank1_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "high" if value > RANK1_HIGH else "low"


def pool_success_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    return "high" if value > POOL_SUCCESS_HIGH else "low"


def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    e_band = endpoint_band(clean_float(row.get("R_endpoint")))
    p_band = pool_band(clean_float(row.get("R_pool")))
    r_band = rank1_band(clean_float(row.get("M_rank1")))
    s_band = pool_success_band(clean_float(row.get("M_pool_success")))
    reasons = [
        f"endpoint={e_band}",
        f"pool={p_band}",
        f"pool_success={s_band}",
        f"rank1={r_band}",
    ]

    if e_band == "high" and p_band == "high" and r_band == "high":
        return "A", reasons
    if e_band == "high" and p_band == "low_or_near_zero" and s_band == "high" and r_band == "high":
        return "B", reasons
    if e_band == "high" and p_band == "low_or_near_zero" and s_band == "low" and r_band == "low":
        return "C", reasons
    if e_band == "low" and p_band == "high" and r_band == "high":
        return "D", reasons
    if e_band == "medium" and p_band == "low_or_near_zero" and s_band == "high" and r_band == "high":
        return "E", reasons
    return "unclassified", reasons


def classify_rows(rows: list[dict[str, Any]], *, scope: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        case, reasons = classify(row)
        item = {
            **row,
            "scope": scope,
            "subset": row.get("subset", "overall"),
            "endpoint_band": endpoint_band(clean_float(row.get("R_endpoint"))),
            "pool_band": pool_band(clean_float(row.get("R_pool"))),
            "pool_success_band": pool_success_band(clean_float(row.get("M_pool_success"))),
            "rank1_band": rank1_band(clean_float(row.get("M_rank1"))),
            "case": case,
            "case_interpretation": CASE_DEFINITIONS.get(case, {}).get("interpretation"),
            "classification_reasons": reasons,
        }
        out.append(item)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["case"] for row in rows)
    by_env_protocol = defaultdict(Counter)
    by_scope = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_env_protocol[f"{row['environment']}::{row['protocol']}"][row["case"]] += 1
        by_scope[row["scope"]][row["case"]] += 1
        if row["case"] != "unclassified" and len(examples[row["case"]]) < 6:
            examples[row["case"]].append(
                {
                    "environment": row["environment"],
                    "protocol": row["protocol"],
                    "subset": row.get("subset", "overall"),
                    "dimension": row["dimension"],
                    "R_endpoint": row.get("R_endpoint"),
                    "R_pool": row.get("R_pool"),
                    "M_pool_success": row.get("M_pool_success"),
                    "M_rank1": row.get("M_rank1"),
                }
            )
    return {
        "n_cells": int(len(rows)),
        "case_distribution": dict(sorted(counts.items())),
        "case_distribution_by_env_protocol": {
            key: dict(sorted(counter.items())) for key, counter in sorted(by_env_protocol.items())
        },
        "case_distribution_by_scope": {
            key: dict(sorted(counter.items())) for key, counter in sorted(by_scope.items())
        },
        "examples_by_case": {key: value for key, value in sorted(examples.items())},
    }


def case_e_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = [
        row
        for row in rows
        if row["environment"] == "PushT"
        and row["protocol"] == "rerank_only"
        and row.get("subset") == "ordinary"
    ]
    ordinary = sorted(ordinary, key=lambda row: int(row["dimension"]))
    strict = [
        row
        for row in ordinary
        if clean_float(row.get("R_pool")) is not None
        and clean_float(row.get("R_pool")) <= POOL_HIGH
        and clean_float(row.get("M_pool_success")) is not None
        and clean_float(row.get("M_pool_success")) > POOL_SUCCESS_HIGH
        and clean_float(row.get("M_rank1")) is not None
        and clean_float(row.get("M_rank1")) > RANK1_HIGH
    ]
    tolerant = [
        row
        for row in ordinary
        if clean_float(row.get("R_pool")) is not None
        and clean_float(row.get("R_pool")) <= CASE_E_NEAR_ZERO_TOL
        and clean_float(row.get("M_pool_success")) is not None
        and clean_float(row.get("M_pool_success")) > POOL_SUCCESS_HIGH
        and clean_float(row.get("M_rank1")) is not None
        and clean_float(row.get("M_rank1")) > RANK1_HIGH
    ]
    m64 = next((row for row in ordinary if int(row["dimension"]) == 64), None)
    m64_pass = (
        m64 is not None
        and clean_float(m64.get("R_pool")) is not None
        and clean_float(m64.get("R_pool")) <= CASE_E_NEAR_ZERO_TOL
        and clean_float(m64.get("M_pool_success")) is not None
        and clean_float(m64.get("M_pool_success")) > POOL_SUCCESS_HIGH
        and clean_float(m64.get("M_rank1")) is not None
        and clean_float(m64.get("M_rank1")) > RANK1_HIGH
    )
    return {
        "decision_rule": (
            "PushT re-rank-only ordinary subset has near-zero pool Spearman, locally sufficient "
            "pool success mass, and high rank-1 success."
        ),
        "near_zero_pool_spearman_tolerance": CASE_E_NEAR_ZERO_TOL,
        "strict_threshold_pool_spearman": POOL_HIGH,
        "ordinary_subset_rows": [
            {
                "dimension": int(row["dimension"]),
                "R_endpoint": row.get("R_endpoint"),
                "R_pool": row.get("R_pool"),
                "M_pool_success": row.get("M_pool_success"),
                "M_rank1": row.get("M_rank1"),
                "strict_pass": row in strict,
                "tolerant_pass": row in tolerant,
            }
            for row in ordinary
        ],
        "strict_pass_dimensions": [int(row["dimension"]) for row in strict],
        "tolerant_pass_dimensions": [int(row["dimension"]) for row in tolerant],
        "m64_tolerant_pass": bool(m64_pass),
        "case_e_stable": bool(m64_pass and len(tolerant) >= 5),
        "note": (
            "The ordinary subset is the execution-plan Case E check. Some high-dimensional ordinary "
            "cells are threshold-classified as Case B because R_endpoint exceeds the high threshold; "
            "the stability flag tracks the locked qualitative decision rule separately."
        ),
    }


def print_summary(summary: dict[str, Any], stability: dict[str, Any]) -> None:
    print("Taxonomy case distribution")
    for case, count in summary["case_distribution"].items():
        print(f"  {case}: {count}")
    print("\nBy environment/protocol")
    for key, counts in summary["case_distribution_by_env_protocol"].items():
        parts = ", ".join(f"{case}={count}" for case, count in counts.items())
        print(f"  {key}: {parts}")
    print("\nCase E stability")
    print(f"  stable: {stability['case_e_stable']}")
    print(f"  strict pass dimensions: {stability['strict_pass_dimensions']}")
    print(f"  tolerant pass dimensions: {stability['tolerant_pass_dimensions']}")
    print(f"  m=64 tolerant pass: {stability['m64_tolerant_pass']}")


def main() -> int:
    args = parse_args()
    args.gap_table = args.gap_table.expanduser().resolve()
    args.cube_stage1a = args.cube_stage1a.expanduser().resolve()
    args.cube_full = args.cube_full.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    gap = load_json(args.gap_table)
    cube_stage1a = load_json(args.cube_stage1a)
    cube_full = load_json(args.cube_full)
    cube_endpoint = load_stage1a_endpoint_by_dim(cube_stage1a)

    gap_overall = list(gap.get("overall", []))
    augmented_overall = gap_overall + cube_full_gap_rows(cube_full, cube_endpoint)
    gap_subset = list(gap.get("by_subset", []))

    overall_rows = classify_rows(augmented_overall, scope="overall")
    subset_rows = classify_rows(gap_subset, scope="subset")
    all_rows = overall_rows + subset_rows
    summary = summarize(all_rows)
    stability = case_e_stability(subset_rows)

    output = {
        "metadata": {
            "format": "taxonomy_table_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "inputs": {
                "gap_table": str(args.gap_table.relative_to(PROJECT_ROOT)),
                "cube_stage1a": str(args.cube_stage1a.relative_to(PROJECT_ROOT)),
                "cube_full": str(args.cube_full.relative_to(PROJECT_ROOT)),
            },
            "thresholds": {
                "endpoint_high": ENDPOINT_HIGH,
                "endpoint_low": ENDPOINT_LOW,
                "pool_high": POOL_HIGH,
                "pool_low_or_near_zero_max": POOL_HIGH,
                "rank1_high": RANK1_HIGH,
                "pool_success_high": POOL_SUCCESS_HIGH,
                "case_e_near_zero_tolerance": CASE_E_NEAR_ZERO_TOL,
            },
            "case_definitions": CASE_DEFINITIONS,
            "notes": [
                "Rows from cem_gap_table.json are classified as-is.",
                "Cube full projected CEM overall rows are derived in-memory from cube_full_proj_cem.json.",
                "PushT subset rows are available for PushT protocols only.",
            ],
        },
        "summary": summary,
        "case_e_stability_check": stability,
        "overall": overall_rows,
        "by_subset": subset_rows,
        "all_rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary(summary, stability)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
