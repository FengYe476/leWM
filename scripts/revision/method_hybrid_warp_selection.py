#!/usr/bin/env python3
"""Hybrid CEM: Euclidean search with v3 warp-ensemble final selection."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REVISION_DIR = PROJECT_ROOT / "scripts" / "revision"
for path in (PROJECT_ROOT, SCRIPTS_DIR, REVISION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import prepare_pair_info  # noqa: E402
from method_cem_variants import (  # noqa: E402
    clean_float,
    deterministic_argmin,
    fmt_float,
    get_git_commit,
    jsonable,
    load_pair_rows_direct,
    scalar_summary,
    seconds_to_hms,
    spearman_corr,
)
from method_local_warp import parse_int_list, set_seed  # noqa: E402
from method_warp_ensemble_v3_cem_full import (  # noqa: E402
    DEFAULT_ENSEMBLE_DIR,
    DEFAULT_MPPI_ANALYSIS,
    DEFAULT_MPPI_RUN,
    DEFAULT_SEEDS,
    SUBSET_ORDER,
    load_json,
    load_mppi_summary,
    load_selected_pairs,
    load_warps,
    metric_mean,
    rank_average_score,
    run_cem,
    warp_cost_matrix,
    write_json_atomic,
)
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    NUM_SAMPLES,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.projected_cem import (  # noqa: E402
    blocked_batch_to_raw_fast,
    score_raw_actions,
)


DEFAULT_WARP_CEM_FULL = PROJECT_ROOT / "results" / "revision" / "warp_v3_cem_full_30pair.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "method_hybrid_warp_selection.json"
DEFAULT_POOL_ROOT = PROJECT_ROOT / "results" / "revision" / "hybrid_warp_pools"
VARIANTS = ("default_cem", "v3_warp_posthoc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--mppi-run", type=Path, default=DEFAULT_MPPI_RUN)
    parser.add_argument("--mppi-analysis", type=Path, default=DEFAULT_MPPI_ANALYSIS)
    parser.add_argument("--warp-cem-full", type=Path, default=DEFAULT_WARP_CEM_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def pool_path(pool_root: Path, *, pair_id: int, seed: int) -> Path:
    return pool_root / f"pair_{int(pair_id)}_seed_{int(seed)}.pt"


def run_key(record: dict[str, Any]) -> tuple[int, str, int]:
    return int(record["pair_id"]), str(record["variant"]), int(record["seed"])


def expected_record_keys(selected_pairs: list[dict[str, Any]], seeds: tuple[int, ...]) -> set[tuple[int, str, int]]:
    return {
        (int(pair["pair_id"]), variant, int(seed))
        for pair in selected_pairs
        for variant in VARIANTS
        for seed in seeds
    }


def load_existing_records(
    output_path: Path,
    *,
    resume: bool,
    expected: set[tuple[int, str, int]],
) -> list[dict[str, Any]]:
    if not resume or not output_path.exists():
        return []
    data = load_json(output_path)
    records_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for record in data.get("records", []):
        try:
            key = run_key(record)
        except (KeyError, TypeError, ValueError):
            continue
        if key not in expected:
            continue
        pool = record.get("pool_path")
        if pool and Path(pool).exists():
            records_by_key[key] = record
    return [records_by_key[key] for key in sorted(records_by_key)]


def run_complete(records: list[dict[str, Any]], *, pair_id: int, seed: int) -> bool:
    keys = {run_key(record) for record in records if int(record["pair_id"]) == int(pair_id) and int(record["seed"]) == int(seed)}
    return all((int(pair_id), variant, int(seed)) in keys for variant in VARIANTS)


def finite_mean(values: list[float | int | bool | None]) -> float | None:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return clean_float(float(arr.mean())) if len(arr) else None


def build_pool_and_records(
    *,
    pair_spec: dict[str, Any],
    initial: dict[str, Any],
    goal: dict[str, Any],
    policy,
    env,
    cem_result: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_id = int(pair_spec["pair_id"])
    raw_actions = blocked_batch_to_raw_fast(
        np.asarray(cem_result["blocked_candidates"], dtype=np.float32),
        action_processor=policy.process["action"],
    )
    v1_costs, c_real_state, success, metrics = score_raw_actions(
        env=env,
        initial_state=np.asarray(initial["state"], dtype=np.float32),
        goal_state=np.asarray(goal["state"], dtype=np.float32),
        raw_actions_batch=raw_actions,
        seed_base=int(seed) + pair_id * 100_000,
    )
    default_costs = np.asarray(cem_result["default_costs"], dtype=np.float64)
    warp_scores = np.asarray(cem_result["warp_rank_avg_costs"], dtype=np.float64)
    default_rank1 = int(deterministic_argmin(default_costs))
    warp_rank1 = int(deterministic_argmin(warp_scores))
    oracle = int(deterministic_argmin(c_real_state))
    rpool_cmodel = spearman_corr(default_costs, c_real_state)
    rpool_warp = spearman_corr(warp_scores, c_real_state)
    rpool_v1 = spearman_corr(v1_costs, c_real_state)
    pool_std = clean_float(float(np.std(c_real_state, ddof=0)))
    pool_success_mass = clean_float(float(np.mean(success)))
    pool = {
        "metadata": {
            "format": "pusht_hybrid_warp_selection_pool_v1",
            "created_at_unix": clean_float(time.time()),
            "pair_id": pair_id,
            "cell": str(pair_spec["cell"]),
            "start_row": int(pair_spec["start_row"]),
            "goal_row": int(pair_spec["goal_row"]),
            "seed": int(seed),
            "cem_sampling_seed": int(seed) + pair_id * 1009,
            "device": str(device),
            "search_variant": "default_cem",
            "selection_variants": list(VARIANTS),
            "primary_subset": str(pair_spec["primary_subset"]),
            "subset_memberships": list(pair_spec["subset_memberships"]),
            "wallclock_seconds": cem_result["wallclock_seconds"],
        },
        "pair_spec": dict(pair_spec),
        "z_pred": torch.as_tensor(np.asarray(cem_result["z_pred"], dtype=np.float32)),
        "z_goal": torch.as_tensor(np.asarray(cem_result["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.as_tensor(np.asarray(cem_result["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.as_tensor(raw_actions.astype(np.float32)),
        "default_costs": torch.as_tensor(default_costs, dtype=torch.float64),
        "warp_rank_avg_costs": torch.as_tensor(warp_scores, dtype=torch.float64),
        "individual_warp_costs": torch.as_tensor(cem_result["individual_warp_costs"], dtype=torch.float64),
        "v1_hinge_costs": torch.as_tensor(v1_costs, dtype=torch.float64),
        "c_real_state": torch.as_tensor(c_real_state, dtype=torch.float64),
        "success": torch.as_tensor(success, dtype=torch.bool),
        "candidate_metrics": metrics,
        "default_rank1_candidate_index": int(default_rank1),
        "warp_rank1_candidate_index": int(warp_rank1),
        "oracle_best_candidate_index": int(oracle),
        "elite_candidate_indices": torch.as_tensor(cem_result["elite_candidate_indices"], dtype=torch.int64),
        "elite_costs": torch.as_tensor(cem_result["elite_costs"], dtype=torch.float64),
        "elite_cost_std_final": cem_result["elite_cost_std"],
        "iteration_diagnostics": cem_result["iteration_diagnostics"],
    }

    def make_record(variant: str, rank1: int, active_rpool: float | None) -> dict[str, Any]:
        return {
            "pair_id": pair_id,
            "cell": str(pair_spec["cell"]),
            "seed": int(seed),
            "variant": variant,
            "search_variant": "default_cem",
            "selection_cost": "default_costs" if variant == "default_cem" else "warp_rank_avg_costs",
            "primary_subset": str(pair_spec["primary_subset"]),
            "subset_memberships": list(pair_spec["subset_memberships"]),
            "planning_success": bool(success[rank1]),
            "rank1_success": bool(success[rank1]),
            "rank1_candidate_index": int(rank1),
            "rank1_c_real": clean_float(float(c_real_state[rank1])),
            "oracle_best_candidate_index": int(oracle),
            "oracle_c_real": clean_float(float(c_real_state[oracle])),
            "selection_regret": clean_float(float(c_real_state[rank1] - c_real_state[oracle])),
            "Rpool_active": active_rpool,
            "Rpool_Cmodel": rpool_cmodel,
            "Rpool_warp_rank_avg": rpool_warp,
            "Rpool_V1": rpool_v1,
            "pool_Creal_std": pool_std,
            "pool_Cmodel_std": clean_float(float(np.std(default_costs, ddof=0))),
            "pool_success_mass": pool_success_mass,
            "elite_cost_std": cem_result["elite_cost_std"],
            "pool_path": None,
        }

    return pool, [
        make_record("default_cem", default_rank1, rpool_cmodel),
        make_record("v3_warp_posthoc", warp_rank1, rpool_warp),
    ]


def per_pair_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((int(record["pair_id"]), str(record["variant"])), []).append(record)
    out = []
    for (pair_id, variant), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        first = ordered[0]
        out.append(
            {
                "pair_id": int(pair_id),
                "cell": first["cell"],
                "variant": variant,
                "primary_subset": first["primary_subset"],
                "subset_memberships": first["subset_memberships"],
                "n_seeds": int(len(rows)),
                "seeds": [int(row["seed"]) for row in ordered],
                "planning_success": finite_mean([row["planning_success"] for row in rows]),
                "selection_regret": finite_mean([row["selection_regret"] for row in rows]),
                "Rpool_active": finite_mean([row["Rpool_active"] for row in rows]),
                "Rpool_Cmodel": finite_mean([row["Rpool_Cmodel"] for row in rows]),
                "Rpool_warp_rank_avg": finite_mean([row["Rpool_warp_rank_avg"] for row in rows]),
                "Rpool_V1": finite_mean([row["Rpool_V1"] for row in rows]),
                "pool_Creal_std": finite_mean([row["pool_Creal_std"] for row in rows]),
                "pool_Cmodel_std": finite_mean([row["pool_Cmodel_std"] for row in rows]),
                "pool_success_mass": finite_mean([row["pool_success_mass"] for row in rows]),
                "elite_cost_std": finite_mean([row["elite_cost_std"] for row in rows]),
            }
        )
    return out


def aggregate_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(rows)),
        "planning_success": scalar_summary([row.get("planning_success") for row in rows]),
        "selection_regret": scalar_summary([row.get("selection_regret") for row in rows]),
        "Rpool_active": scalar_summary([row.get("Rpool_active") for row in rows]),
        "Rpool_Cmodel": scalar_summary([row.get("Rpool_Cmodel") for row in rows]),
        "Rpool_warp_rank_avg": scalar_summary([row.get("Rpool_warp_rank_avg") for row in rows]),
        "Rpool_V1": scalar_summary([row.get("Rpool_V1") for row in rows]),
        "pool_Creal_std": scalar_summary([row.get("pool_Creal_std") for row in rows]),
        "pool_Cmodel_std": scalar_summary([row.get("pool_Cmodel_std") for row in rows]),
        "pool_success_mass": scalar_summary([row.get("pool_success_mass") for row in rows]),
        "elite_cost_std": scalar_summary([row.get("elite_cost_std") for row in rows]),
    }


def aggregate_by_variant(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        variant: aggregate_pairs([row for row in per_pair if row["variant"] == variant])
        for variant in VARIANTS
    }


def aggregate_by_subset(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in per_pair if row["variant"] == variant]
        out[variant] = {
            subset: aggregate_pairs([row for row in variant_rows if row.get("primary_subset") == subset])
            for subset in SUBSET_ORDER
        }
    return out


def prior_warp_cem_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    data = load_json(path)
    try:
        summary = data["summary_by_variant"]["v3_warp_cem"]
        subset = data["subset_by_variant"]["v3_warp_cem"]
    except KeyError as exc:
        raise ValueError(f"{path} is missing v3_warp_cem summary data") from exc
    pair30 = [row for row in data.get("records", []) if int(row.get("pair_id", -1)) == 30 and row.get("variant") == "v3_warp_cem"]
    return summary, subset, pair30


def mppi_pair30_records(mppi_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in mppi_summary.get("mppi_per_seed", []) if int(row.get("pair_id", -1)) == 30]


def fmt_metric(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.1f}%" if percent else f"{float(value):.3f}"


def make_comparison_table(
    summary_by_variant: dict[str, Any],
    warp_cem_summary: dict[str, Any],
    mppi_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        ("Planning success (30p, 3s)", "planning_success", True),
        ("Mean selection regret", "selection_regret", False),
        ("Rpool(active selector) mean", "Rpool_active", False),
        ("Rpool(C_model) mean", "Rpool_Cmodel", False),
        ("Rpool(warp rank avg) mean", "Rpool_warp_rank_avg", False),
        ("Pool C_real_state std", "pool_Creal_std", False),
    ]
    mppi = mppi_summary["summary"]["mppi_tau_1"] if "summary" in mppi_summary else mppi_summary["mppi_tau_1"]
    out = []
    for label, key, percent in rows:
        if key == "Rpool_active":
            v3_value = metric_mean(warp_cem_summary, "Rpool_warp_rank_avg")
            mppi_value = metric_mean(mppi, "Rpool_Cmodel")
        else:
            v3_value = metric_mean(warp_cem_summary, key)
            mppi_value = metric_mean(mppi, key)
        out.append(
            {
            "metric": label,
            "key": key,
            "default_cem": metric_mean(summary_by_variant["default_cem"], key),
            "v3_warp_posthoc": metric_mean(summary_by_variant["v3_warp_posthoc"], key),
            "v3_warp_cem": v3_value,
            "mppi_tau_1": mppi_value,
            "format": "percent" if percent else "float",
        }
        )
    return out


def make_subset_table(
    subset_by_variant: dict[str, Any],
    warp_cem_subset: dict[str, Any],
    mppi_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    mppi_by_subset = mppi_summary["summary"].get("mppi_by_subset", {}) if "summary" in mppi_summary else mppi_summary.get("mppi_by_subset", {})
    rows = []
    for subset in SUBSET_ORDER:
        for label, key, percent in (
            ("Planning success", "planning_success", True),
            ("Mean selection regret", "selection_regret", False),
            ("Rpool(active selector)", "Rpool_active", False),
            ("Rpool(C_model)", "Rpool_Cmodel", False),
            ("Pool C_real_state std", "pool_Creal_std", False),
        ):
            if key == "Rpool_active":
                v3_value = metric_mean(warp_cem_subset.get(subset, {}), "Rpool_warp_rank_avg")
                mppi_value = metric_mean(mppi_by_subset.get(subset, {}), "Rpool_Cmodel")
            else:
                v3_value = metric_mean(warp_cem_subset.get(subset, {}), key)
                mppi_value = metric_mean(mppi_by_subset.get(subset, {}), key)
            rows.append(
                {
                    "subset": subset,
                    "metric": label,
                    "key": key,
                    "default_cem": metric_mean(subset_by_variant["default_cem"][subset], key),
                    "v3_warp_posthoc": metric_mean(subset_by_variant["v3_warp_posthoc"][subset], key),
                    "v3_warp_cem": v3_value,
                    "mppi_tau_1": mppi_value,
                    "format": "percent" if percent else "float",
                }
            )
    return rows


def build_output(
    *,
    args: argparse.Namespace,
    selected_pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    mppi_full: dict[str, Any],
    warp_cem_summary: dict[str, Any],
    warp_cem_subset: dict[str, Any],
    warp_cem_pair30: list[dict[str, Any]],
    started: float,
    expected_count: int,
) -> dict[str, Any]:
    per_pair = per_pair_records(records)
    summary_by_variant = aggregate_by_variant(per_pair)
    subset_by_variant = aggregate_by_subset(per_pair)
    comparison = make_comparison_table(summary_by_variant, warp_cem_summary, mppi_full)
    subset_table = make_subset_table(subset_by_variant, warp_cem_subset, mppi_full)
    pair30_hybrid = [
        row for row in sorted(records, key=lambda item: (str(item["variant"]), int(item["seed"])))
        if int(row["pair_id"]) == 30
    ]
    pair30_mppi = mppi_pair30_records(mppi_full)
    return {
        "metadata": {
            "format": "pusht_hybrid_warp_selection_v1",
            "created_at_unix": clean_float(time.time()),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "pairs_path": str(args.pairs_path),
            "mppi_run": str(args.mppi_run),
            "mppi_analysis": str(args.mppi_analysis),
            "warp_cem_full": str(args.warp_cem_full),
            "output": str(args.output),
            "pool_root": str(args.pool_root),
            "checkpoint_dir": str(args.checkpoint_dir),
            "ensemble_dir": str(args.ensemble_dir),
            "device": str(args.device),
            "seeds": [int(seed) for seed in args.seeds],
            "pair_ids": [int(pair["pair_id"]) for pair in selected_pairs],
            "n_pairs": int(len(selected_pairs)),
            "variants": list(VARIANTS),
            "expected_records": int(expected_count),
            "n_records": int(len(records)),
            "expected_cem_runs": int(len(selected_pairs) * len(args.seeds)),
            "wallclock_seconds": clean_float(time.time() - started),
            "aggregation": "Seed runs are averaged within pair, then pair means are averaged across pairs.",
        },
        "records": sorted(records, key=lambda row: (int(row["pair_id"]), str(row["variant"]), int(row["seed"]))),
        "per_pair": per_pair,
        "summary_by_variant": summary_by_variant,
        "subset_by_variant": subset_by_variant,
        "v3_warp_cem_summary": warp_cem_summary,
        "v3_warp_cem_subset": warp_cem_subset,
        "mppi_summary": mppi_full,
        "comparison_table": comparison,
        "subset_breakdown_table": subset_table,
        "pair30_spotlight": {
            "hybrid_records": pair30_hybrid,
            "v3_warp_cem_records": warp_cem_pair30,
            "mppi_records": pair30_mppi,
        },
    }


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    table = []
    for row in rows:
        percent = row["format"] == "percent"
        table.append(
            [
                row["metric"],
                fmt_metric(row["default_cem"], percent=percent),
                fmt_metric(row["v3_warp_posthoc"], percent=percent),
                fmt_metric(row["v3_warp_cem"], percent=percent),
                fmt_metric(row["mppi_tau_1"], percent=percent),
            ]
        )
    print("\nFour-way aggregate comparison")
    print(
        tabulate(
            table,
            headers=["Metric", "Default CEM", "V3 Warp Post-hoc", "V3 Warp CEM", "MPPI (tau=1.0)"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def print_subset_table(rows: list[dict[str, Any]]) -> None:
    table = []
    for row in rows:
        percent = row["format"] == "percent"
        table.append(
            [
                row["subset"],
                row["metric"],
                fmt_metric(row["default_cem"], percent=percent),
                fmt_metric(row["v3_warp_posthoc"], percent=percent),
                fmt_metric(row["v3_warp_cem"], percent=percent),
                fmt_metric(row["mppi_tau_1"], percent=percent),
            ]
        )
    print("\nPer-subset breakdown")
    print(
        tabulate(
            table,
            headers=["Subset", "Metric", "Default CEM", "V3 Warp Post-hoc", "V3 Warp CEM", "MPPI"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def print_pair30_table(pair30: dict[str, Any]) -> None:
    rows = []
    for record in pair30["hybrid_records"]:
        rows.append(
            [
                record["seed"],
                record["variant"],
                fmt_float(record["Rpool_active"]),
                str(record["planning_success"]),
                fmt_float(record["selection_regret"]),
                fmt_float(record["pool_Creal_std"]),
            ]
        )
    for record in sorted(pair30["v3_warp_cem_records"], key=lambda row: int(row["seed"])):
        rows.append(
            [
                record["seed"],
                "v3_warp_cem",
                fmt_float(record.get("Rpool_warp_rank_avg")),
                str(record["planning_success"]),
                fmt_float(record["selection_regret"]),
                fmt_float(record["pool_Creal_std"]),
            ]
        )
    for record in sorted(pair30["mppi_records"], key=lambda row: int(row["seed"])):
        rows.append(
            [
                record["seed"],
                "mppi_tau_1",
                fmt_float(record.get("Rpool_Cmodel")),
                str(record["planning_success"]),
                fmt_float(record["selection_regret"]),
                fmt_float(record["pool_Creal_std"]),
            ]
        )
    print("\nPair 30 spotlight")
    print(
        tabulate(
            rows,
            headers=["Seed", "Variant", "Rpool(active)", "Success", "Regret", "Pool CReal Std"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def running_stats(records: list[dict[str, Any]]) -> str:
    if not records:
        return "no records yet"
    per_pair = per_pair_records(records)
    summary = aggregate_by_variant(per_pair)
    parts = []
    for variant in VARIANTS:
        stats = summary[variant]
        if stats["n_pairs"] == 0:
            continue
        parts.append(
            f"{variant}: success={fmt_metric(metric_mean(stats, 'planning_success'), percent=True)} "
            f"Ractive={fmt_metric(metric_mean(stats, 'Rpool_active'))} "
            f"regret={fmt_metric(metric_mean(stats, 'selection_regret'))}"
        )
    return " | ".join(parts) if parts else "no pair-complete records yet"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.mppi_run = args.mppi_run.expanduser().resolve()
    args.mppi_analysis = args.mppi_analysis.expanduser().resolve()
    args.warp_cem_full = args.warp_cem_full.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pool_root = args.pool_root.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.ensemble_dir = args.ensemble_dir.expanduser().resolve()
    args.device = resolve_device(args.device)
    args.seeds = tuple(int(seed) for seed in args.seeds)

    selected_pairs = load_selected_pairs(args.mppi_run, pair_limit=args.pair_limit)
    pairs_data, _ = load_pairs(args.pairs_path, max_pairs=None, pair_ids=None)
    pair_metadata = pairs_data["metadata"]
    validate_requested_pair_offsets(selected_pairs, offset=int(pair_metadata["offset"]))
    dataset_path = Path(pair_metadata["dataset_path"])
    mppi_full = load_json(args.mppi_analysis)
    load_mppi_summary(args.mppi_analysis)
    warp_cem_summary, warp_cem_subset, warp_cem_pair30 = prior_warp_cem_summary(args.warp_cem_full)

    expected = expected_record_keys(selected_pairs, args.seeds)
    records = load_existing_records(args.output, resume=not args.no_resume, expected=expected)

    print("== Hybrid CEM: Euclidean search + V3 warp final selection ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"mppi_run: {args.mppi_run}")
    print(f"mppi_analysis: {args.mppi_analysis}")
    print(f"warp_cem_full: {args.warp_cem_full}")
    print(f"output: {args.output}")
    print(f"pool_root: {args.pool_root}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"ensemble_dir: {args.ensemble_dir}")
    print(f"dataset_name: {dataset_path.stem}")
    print(f"cache_dir: {dataset_path.parent}")
    print(f"device: {args.device}")
    print(f"seeds: {[int(seed) for seed in args.seeds]}")
    print(f"pairs: {len(selected_pairs)}")
    print(f"selection_variants: {list(VARIANTS)}")
    print(f"expected_cem_runs: {len(selected_pairs) * len(args.seeds)}")
    print(f"expected_selection_records: {len(expected)}")
    print(f"expected_simulator_rollouts: {len(selected_pairs) * len(args.seeds) * NUM_SAMPLES}")
    print(f"resume_records: {len(records)}")

    set_seed(min(int(seed) for seed in args.seeds))
    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=min(int(seed) for seed in args.seeds),
        ),
        process,
    )
    model = policy.solver.model
    warps = load_warps(args.ensemble_dir, device=next(model.parameters()).device)

    total_started = time.time()
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair_spec in enumerate(selected_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            initial, goal = load_pair_rows_direct(dataset, pair_spec)
            prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
            pair_started = time.time()
            pair_completed_before = len(records)
            for seed_idx, seed in enumerate(args.seeds, start=1):
                seed = int(seed)
                if run_complete(records, pair_id=pair_id, seed=seed):
                    print(
                        f"[pair {pair_idx}/{len(selected_pairs)} seed {seed_idx}/{len(args.seeds)}] "
                        f"pair_id={pair_id} seed={seed}: resume"
                    )
                    continue
                records = [
                    record for record in records
                    if not (int(record["pair_id"]) == pair_id and int(record["seed"]) == seed)
                ]
                run_started = time.time()
                print(
                    f"[pair {pair_idx}/{len(selected_pairs)} seed {seed_idx}/{len(args.seeds)}] "
                    f"pair_id={pair_id} seed={seed}: default CEM + warp post-hoc selection"
                )
                cem_result = run_cem(
                    model=model,
                    prepared_info=prepared_info,
                    warps=warps,
                    pair_id=pair_id,
                    seed=seed,
                    variant="default_cem",
                )
                pool, new_records = build_pool_and_records(
                    pair_spec=pair_spec,
                    initial=initial,
                    goal=goal,
                    policy=policy,
                    env=env,
                    cem_result=cem_result,
                    seed=seed,
                    device=str(args.device),
                )
                path = pool_path(args.pool_root, pair_id=pair_id, seed=seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pool, path)
                for record in new_records:
                    record["pool_path"] = str(path)
                records.extend(new_records)

                output = build_output(
                    args=args,
                    selected_pairs=selected_pairs,
                    records=records,
                    mppi_full=mppi_full,
                    warp_cem_summary=warp_cem_summary,
                    warp_cem_subset=warp_cem_subset,
                    warp_cem_pair30=warp_cem_pair30,
                    started=total_started,
                    expected_count=len(expected),
                )
                write_json_atomic(args.output, output)
                default_record = [record for record in new_records if record["variant"] == "default_cem"][0]
                warp_record = [record for record in new_records if record["variant"] == "v3_warp_posthoc"][0]
                print(
                    f"  saved {path}; default success={default_record['planning_success']} "
                    f"regret={fmt_float(default_record['selection_regret'])}; "
                    f"posthoc success={warp_record['planning_success']} "
                    f"regret={fmt_float(warp_record['selection_regret'])} "
                    f"Rwarp={fmt_float(warp_record['Rpool_active'])} "
                    f"elapsed={seconds_to_hms(time.time() - run_started)}"
                )
                print(
                    f"  progress {len(records)}/{len(expected)} records; "
                    f"running {running_stats(records)}"
                )
            if len(records) > pair_completed_before:
                print(
                    f"[pair {pair_idx}/{len(selected_pairs)}] pair_id={pair_id} complete/update; "
                    f"elapsed={seconds_to_hms(time.time() - pair_started)}"
                )
    finally:
        if hasattr(env, "close"):
            env.close()

    output = build_output(
        args=args,
        selected_pairs=selected_pairs,
        records=records,
        mppi_full=mppi_full,
        warp_cem_summary=warp_cem_summary,
        warp_cem_subset=warp_cem_subset,
        warp_cem_pair30=warp_cem_pair30,
        started=total_started,
        expected_count=len(expected),
    )
    write_json_atomic(args.output, output)
    print_comparison_table(output["comparison_table"])
    print_subset_table(output["subset_breakdown_table"])
    print_pair30_table(output["pair30_spotlight"])
    print(f"\nWrote summary: {args.output}")
    print(f"Wrote pools under: {args.pool_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
