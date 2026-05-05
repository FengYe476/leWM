#!/usr/bin/env python3
"""Analyze Phase D MPPI final pools and compare against matched CEM pools."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.phase2.stage1.stage1a_controls import clean_float, iso_now, jsonable  # noqa: E402


DEFAULT_MPPI_RUN = PROJECT_ROOT / "results" / "revision" / "mppi_pusht_30pair.json"
DEFAULT_CEM_RPOOL = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "mppi_pool_analysis.json"
DEFAULT_MEMO = PROJECT_ROOT / "docs" / "revision" / "mppi_memo.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mppi-run", type=Path, default=DEFAULT_MPPI_RUN)
    parser.add_argument("--cem-rpool", type=Path, default=DEFAULT_CEM_RPOOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memo", type=Path, default=DEFAULT_MEMO)
    parser.add_argument("--success-debug-gap", type=float, default=0.10)
    parser.add_argument("--rpool-gap-threshold", type=float, default=0.10)
    parser.add_argument("--near-zero-rpool-threshold", type=float, default=0.10)
    return parser.parse_args()


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finite_mean(values: list[float | int | bool | None]) -> float | None:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return clean_float(float(arr.mean())) if len(arr) else None


def scalar_summary(values: list[float | int | bool | None]) -> dict[str, Any]:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "mean": clean_float(float(arr.mean())) if len(arr) else None,
        "std": clean_float(float(arr.std(ddof=1))) if len(arr) > 1 else None,
        "min": clean_float(float(arr.min())) if len(arr) else None,
        "max": clean_float(float(arr.max())) if len(arr) else None,
        "n": int(len(arr)),
        "ddof": 1,
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    value = spearmanr(x, y).correlation
    if value is None or not math.isfinite(float(value)):
        return None
    return clean_float(float(value))


def tensor_to_numpy(pool: dict[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    value = pool[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype)
    return np.asarray(value, dtype=dtype)


def pool_metrics(pool_path: Path, *, primary_subset: str | None, subset_memberships: list[str]) -> dict[str, Any]:
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    pair_spec = pool["pair_spec"]
    pair_id = int(pair_spec["pair_id"])
    seed = int(pool["metadata"]["seed"])
    default_costs = tensor_to_numpy(pool, "default_costs", dtype=np.float64)
    v1_costs = tensor_to_numpy(pool, "v1_hinge_costs", dtype=np.float64)
    c_real_state = tensor_to_numpy(pool, "c_real_state", dtype=np.float64)
    success = tensor_to_numpy(pool, "success", dtype=bool)
    rank1 = int(pool["default_rank1_candidate_index"])
    oracle_best = int(np.argmin(c_real_state))
    return {
        "pair_id": pair_id,
        "seed": seed,
        "cell": str(pair_spec["cell"]),
        "primary_subset": primary_subset,
        "subset_memberships": subset_memberships,
        "pool_path": str(pool_path),
        "rank1_index": rank1,
        "oracle_best_index": oracle_best,
        "Rpool_Cmodel": spearman(default_costs, c_real_state),
        "Rpool_V1": spearman(v1_costs, c_real_state),
        "pool_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
        "pool_Creal_range": clean_float(float(np.max(c_real_state) - np.min(c_real_state))),
        "pool_success_mass": clean_float(float(np.mean(success))),
        "planning_success": bool(success[rank1]),
        "selection_regret": clean_float(float(c_real_state[rank1] - c_real_state[oracle_best])),
        "rank1_c_real_state": clean_float(float(c_real_state[rank1])),
        "oracle_best_c_real_state": clean_float(float(c_real_state[oracle_best])),
        "rank1_default_cost": clean_float(float(default_costs[rank1])),
        "pool_Cmodel_std": clean_float(float(np.std(default_costs, ddof=0))),
        "final_weight_effective_sample_size": pool.get("final_weight_effective_sample_size"),
    }


def aggregate_seed_runs(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in per_seed:
        by_pair[int(record["pair_id"])].append(record)

    out = []
    for pair_id, records in sorted(by_pair.items()):
        first = records[0]
        out.append(
            {
                "pair_id": pair_id,
                "cell": first["cell"],
                "primary_subset": first.get("primary_subset"),
                "subset_memberships": first.get("subset_memberships", []),
                "n_seeds": int(len(records)),
                "seeds": [int(record["seed"]) for record in records],
                "planning_success": finite_mean([record["planning_success"] for record in records]),
                "selection_regret": finite_mean([record["selection_regret"] for record in records]),
                "Rpool_Cmodel": finite_mean([record["Rpool_Cmodel"] for record in records]),
                "Rpool_V1": finite_mean([record["Rpool_V1"] for record in records]),
                "pool_Creal_std": finite_mean([record["pool_Creal_std"] for record in records]),
                "pool_Creal_range": finite_mean([record["pool_Creal_range"] for record in records]),
                "pool_success_mass": finite_mean([record["pool_success_mass"] for record in records]),
                "rank1_c_real_state": finite_mean([record["rank1_c_real_state"] for record in records]),
                "pool_Cmodel_std": finite_mean([record["pool_Cmodel_std"] for record in records]),
                "final_weight_effective_sample_size": finite_mean(
                    [record["final_weight_effective_sample_size"] for record in records]
                ),
            }
        )
    return out


def aggregate_pairs(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(per_pair)),
        "planning_success": scalar_summary([record.get("planning_success") for record in per_pair]),
        "selection_regret": scalar_summary([record.get("selection_regret") for record in per_pair]),
        "Rpool_Cmodel": scalar_summary([record.get("Rpool_Cmodel") for record in per_pair]),
        "Rpool_V1": scalar_summary([record.get("Rpool_V1") for record in per_pair]),
        "pool_Creal_std": scalar_summary([record.get("pool_Creal_std") for record in per_pair]),
        "pool_Creal_range": scalar_summary([record.get("pool_Creal_range") for record in per_pair]),
        "pool_success_mass": scalar_summary([record.get("pool_success_mass") for record in per_pair]),
        "pool_Cmodel_std": scalar_summary([record.get("pool_Cmodel_std") for record in per_pair]),
        "final_weight_effective_sample_size": scalar_summary(
            [record.get("final_weight_effective_sample_size") for record in per_pair]
        ),
    }


def aggregate_by_subset(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    subsets = sorted({str(record["primary_subset"]) for record in per_pair if record.get("primary_subset")})
    return {
        subset: aggregate_pairs([record for record in per_pair if record.get("primary_subset") == subset])
        for subset in subsets
    }


def cem_matched_pairs(cem_data: dict[str, Any], pair_ids: set[int]) -> list[dict[str, Any]]:
    rows = []
    for row in cem_data["per_pair"]:
        if int(row["pair_id"]) not in pair_ids:
            continue
        rows.append(
            {
                "pair_id": int(row["pair_id"]),
                "cell": row.get("cell"),
                "subsets": row.get("subsets", []),
                "planning_success": bool(row.get("pool_success_rank1")),
                "selection_regret": row.get("selection_regret"),
                "Rpool_Cmodel": row.get("Rpool_Cmodel"),
                "Rpool_V1": row.get("Rpool_V1"),
                "pool_Creal_std": row.get("pool_Creal_std"),
                "pool_success_mass": row.get("pool_success_mass"),
                "pool_Cmodel_std": row.get("pool_Cmodel_std"),
            }
        )
    rows = sorted(rows, key=lambda item: item["pair_id"])
    if len(rows) != len(pair_ids):
        got = {int(row["pair_id"]) for row in rows}
        raise RuntimeError(f"Matched CEM rows missing pair IDs: {sorted(pair_ids - got)}")
    return rows


def comparison_row(label: str, key: str, cem: dict[str, Any], mppi: dict[str, Any], *, percent: bool = False) -> dict[str, Any]:
    cem_value = cem[key]["mean"]
    mppi_value = mppi[key]["mean"]
    return {
        "metric": label,
        "key": key,
        "cem_default": cem_value,
        "mppi_tau_1": mppi_value,
        "format": "percent" if percent else "float",
    }


def fmt(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.1f}%" if percent else f"{float(value):.3f}"


def decision(cem_summary: dict[str, Any], mppi_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cem_success = float(cem_summary["planning_success"]["mean"])
    mppi_success = float(mppi_summary["planning_success"]["mean"])
    cem_rpool = float(cem_summary["Rpool_Cmodel"]["mean"])
    mppi_rpool = float(mppi_summary["Rpool_Cmodel"]["mean"])
    success_gap = cem_success - mppi_success
    rpool_gap = mppi_rpool - cem_rpool
    if success_gap > float(args.success_debug_gap):
        verdict = "Debug needed, don't report"
        rationale = (
            "MPPI planning success is much lower than matched CEM "
            f"({fmt(mppi_success, percent=True)} vs {fmt(cem_success, percent=True)})."
        )
    elif rpool_gap > float(args.rpool_gap_threshold):
        verdict = "Decoupling is CEM hard-truncation specific"
        rationale = (
            "MPPI Rpool(C_model) exceeds matched CEM by more than the threshold "
            f"({mppi_rpool:.3f} vs {cem_rpool:.3f})."
        )
    elif abs(mppi_rpool) < float(args.near_zero_rpool_threshold):
        verdict = "Endpoint-planning decoupling generalizes beyond CEM to iterative optimizers"
        rationale = f"MPPI Rpool(C_model) remains near zero ({mppi_rpool:.3f})."
    else:
        verdict = "Mixed MPPI result"
        rationale = (
            "MPPI Rpool(C_model) is not near zero, but the gap to CEM does not exceed "
            f"the {args.rpool_gap_threshold:.2f} threshold."
        )
    return {
        "verdict": verdict,
        "rationale": rationale,
        "cem_planning_success": clean_float(cem_success),
        "mppi_planning_success": clean_float(mppi_success),
        "success_gap_cem_minus_mppi": clean_float(success_gap),
        "cem_Rpool_Cmodel": clean_float(cem_rpool),
        "mppi_Rpool_Cmodel": clean_float(mppi_rpool),
        "Rpool_gap_mppi_minus_cem": clean_float(rpool_gap),
        "thresholds": {
            "success_debug_gap": clean_float(args.success_debug_gap),
            "rpool_gap_threshold": clean_float(args.rpool_gap_threshold),
            "near_zero_rpool_threshold": clean_float(args.near_zero_rpool_threshold),
        },
    }


def make_memo(*, output: dict[str, Any]) -> str:
    rows = output["comparison_table"]
    lines = [
        "# MPPI Comparison Memo",
        "",
        "## Main Comparison",
        "",
        "| Metric | CEM default | MPPI (tau=1.0) |",
        "|---|---:|---:|",
    ]
    for row in rows:
        percent = row["format"] == "percent"
        lines.append(
            f"| {row['metric']} | {fmt(row['cem_default'], percent=percent)} | "
            f"{fmt(row['mppi_tau_1'], percent=percent)} |"
        )
    decision_block = output["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision_block['verdict']}**",
            "",
            decision_block["rationale"],
            "",
            "## Notes",
            "",
            "- MPPI metrics are computed per seed, averaged across the 3 seeds for each pair, then averaged across the 30 pairs.",
            "- CEM default metrics are the matched 30 pair rows from the Phase B Rpool(V1) artifact.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.mppi_run = args.mppi_run.expanduser().resolve()
    args.cem_rpool = args.cem_rpool.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.memo = args.memo.expanduser().resolve()

    mppi_run = load_json(args.mppi_run)
    cem_data = load_json(args.cem_rpool)
    records = mppi_run["records"]
    print("== MPPI pool analysis setup ==")
    print(f"mppi_run: {args.mppi_run}")
    print(f"cem_rpool: {args.cem_rpool}")
    print(f"mppi_records: {len(records)}")

    per_seed = []
    for idx, record in enumerate(records, start=1):
        pool_path = Path(record["pool_path"])
        if not pool_path.exists():
            raise FileNotFoundError(pool_path)
        if idx == 1 or idx % 10 == 0 or idx == len(records):
            print(f"[{idx}/{len(records)}] loading {pool_path.name}")
        per_seed.append(
            pool_metrics(
                pool_path,
                primary_subset=record.get("primary_subset"),
                subset_memberships=record.get("subset_memberships", []),
            )
        )

    mppi_per_pair = aggregate_seed_runs(per_seed)
    pair_ids = {int(record["pair_id"]) for record in mppi_per_pair}
    cem_per_pair = cem_matched_pairs(cem_data, pair_ids)
    mppi_summary = aggregate_pairs(mppi_per_pair)
    cem_summary = aggregate_pairs(cem_per_pair)
    comparison = [
        comparison_row("Planning success (30 pairs)", "planning_success", cem_summary, mppi_summary, percent=True),
        comparison_row("Rpool(C_model) mean", "Rpool_Cmodel", cem_summary, mppi_summary),
        comparison_row("Rpool(V1) mean", "Rpool_V1", cem_summary, mppi_summary),
        comparison_row("Pool C_real_state std mean", "pool_Creal_std", cem_summary, mppi_summary),
        comparison_row("Pool success mass mean", "pool_success_mass", cem_summary, mppi_summary, percent=True),
    ]
    decision_block = decision(cem_summary, mppi_summary, args)
    output = {
        "metadata": {
            "format": "pusht_mppi_pool_analysis_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "mppi_run": str(args.mppi_run),
            "cem_rpool": str(args.cem_rpool),
            "n_mppi_seed_runs": int(len(per_seed)),
            "n_pairs": int(len(mppi_per_pair)),
            "pair_ids": sorted(pair_ids),
            "aggregation": "MPPI per-seed metrics are averaged across seeds within pair, then across pairs.",
        },
        "mppi_per_seed": per_seed,
        "mppi_per_pair": mppi_per_pair,
        "cem_per_pair": cem_per_pair,
        "summary": {
            "cem_default": cem_summary,
            "mppi_tau_1": mppi_summary,
            "mppi_by_subset": aggregate_by_subset(mppi_per_pair),
        },
        "comparison_table": comparison,
        "decision": decision_block,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    args.memo.parent.mkdir(parents=True, exist_ok=True)
    args.memo.write_text(make_memo(output=output))

    print("\nMPPI vs CEM comparison")
    print("Metric | CEM default | MPPI (tau=1.0)")
    print("---|---:|---:")
    for row in comparison:
        percent = row["format"] == "percent"
        print(f"{row['metric']} | {fmt(row['cem_default'], percent=percent)} | {fmt(row['mppi_tau_1'], percent=percent)}")
    print(f"\nDecision: {decision_block['verdict']}")
    print(decision_block["rationale"])
    print(f"Saved: {args.output}")
    print(f"Memo: {args.memo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
