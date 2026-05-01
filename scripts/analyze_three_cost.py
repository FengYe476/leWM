#!/usr/bin/env python3
"""Analyze three-cost attribution results for LeWM PushT long-goal failures."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - exercised only outside the project env.
    scipy_stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "three_cost_offset50.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "three_cost_analysis.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
COST_KEYS = ("c_model", "c_real_z", "c_real_state")
METRIC_PAIRS = (
    ("c_real_z", "c_real_state"),
    ("c_model", "c_real_z"),
    ("c_model", "c_real_state"),
)
SOURCE_ORDER = ("data", "random", "cem_early", "cem_late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LeWM three-cost attribution JSON results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--low-corr-threshold",
        type=float,
        default=0.5,
        help="Correlation threshold used by the preliminary decision tree.",
    )
    parser.add_argument(
        "--planner-majority-threshold",
        type=float,
        default=0.5,
        help=(
            "Case D triggers when the fraction of pairs with zero successful "
            "actions is greater than this value."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args()


def load_records(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    records = []
    for pair_index, pair in enumerate(data.get("pairs", [])):
        for action_index, action in enumerate(pair.get("actions", [])):
            record = {
                "pair_index": pair_index,
                "action_index": action_index,
                "episode_id": pair.get("episode_id"),
                "start_step": pair.get("start_step"),
                "goal_step": pair.get("goal_step"),
                "source": action.get("source", "unknown"),
                "success": bool(action.get("success", False)),
                "env_success": bool(action.get("env_success", False)),
            }
            for key in COST_KEYS + ("block_pos_dist", "angle_dist"):
                record[key] = float(action[key])
            records.append(record)
    if not records:
        raise ValueError(f"No action records found in {path}")
    return data, records


def rankdata(values: np.ndarray) -> np.ndarray:
    if scipy_stats is not None:
        return scipy_stats.rankdata(values, method="average")

    values = np.asarray(values)
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # Ranks are one-indexed; ties receive the average occupied rank.
        ranks[sorter[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def finite_pair_arrays(records: Iterable[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([record[x_key] for record in records], dtype=np.float64)
    y = np.asarray([record[y_key] for record in records], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return pearson_corr(rankdata(x), rankdata(y))


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    if scipy_stats is not None:
        value = float(scipy_stats.kendalltau(x, y, nan_policy="omit").statistic)
        return value if math.isfinite(value) else None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def summarize_values(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def summarize_corr_values(values: Iterable[float | None]) -> dict:
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_correlations(records: list[dict]) -> dict:
    global_corrs = {}
    for x_key, y_key in METRIC_PAIRS:
        x, y = finite_pair_arrays(records, x_key, y_key)
        global_corrs[f"{x_key}_vs_{y_key}"] = {
            "pearson": pearson_corr(x, y),
            "spearman": spearman_corr(x, y),
            "n": int(len(x)),
        }

    records_by_pair = group_records(records, "pair_index")
    per_pair = []
    per_pair_summary = {}
    for pair_index, pair_records in sorted(records_by_pair.items()):
        entry = {"pair_index": int(pair_index), "n": len(pair_records)}
        for x_key, y_key in METRIC_PAIRS:
            x, y = finite_pair_arrays(pair_records, x_key, y_key)
            name = f"{x_key}_vs_{y_key}"
            entry[name] = {
                "pearson": pearson_corr(x, y),
                "spearman": spearman_corr(x, y),
            }
        per_pair.append(entry)

    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        per_pair_summary[name] = {
            "pearson": summarize_corr_values(entry[name]["pearson"] for entry in per_pair),
            "spearman": summarize_corr_values(entry[name]["spearman"] for entry in per_pair),
        }

    records_by_source = group_records(records, "source")
    per_source = {}
    for source in sorted(records_by_source, key=source_sort_key):
        per_source[source] = {}
        for x_key, y_key in METRIC_PAIRS:
            x, y = finite_pair_arrays(records_by_source[source], x_key, y_key)
            per_source[source][f"{x_key}_vs_{y_key}"] = {
                "pearson": pearson_corr(x, y),
                "spearman": spearman_corr(x, y),
                "n": int(len(x)),
            }

    return {
        "global": global_corrs,
        "per_pair": per_pair,
        "per_pair_summary": per_pair_summary,
        "per_source": per_source,
    }


def compute_planner_diagnostic(records: list[dict]) -> dict:
    records_by_pair = group_records(records, "pair_index")
    pair_entries = []
    for pair_index, pair_records in sorted(records_by_pair.items()):
        successes = int(sum(record["success"] for record in pair_records))
        env_successes = int(sum(record["env_success"] for record in pair_records))
        pair_entries.append(
            {
                "pair_index": int(pair_index),
                "actions": len(pair_records),
                "successful_actions": successes,
                "env_successful_actions": env_successes,
                "has_success": successes > 0,
                "has_env_success": env_successes > 0,
                "best_c_real_state": float(min(record["c_real_state"] for record in pair_records)),
                "best_block_pos_dist": float(min(record["block_pos_dist"] for record in pair_records)),
                "best_angle_dist": float(min(record["angle_dist"] for record in pair_records)),
            }
        )

    num_pairs = len(pair_entries)
    pairs_with_success = sum(entry["has_success"] for entry in pair_entries)
    zero_success_pairs = num_pairs - pairs_with_success
    return {
        "pairs": pair_entries,
        "num_pairs": num_pairs,
        "pairs_with_success": int(pairs_with_success),
        "pairs_with_zero_success": int(zero_success_pairs),
        "fraction_pairs_with_success": float(pairs_with_success / num_pairs),
        "fraction_pairs_with_zero_success": float(zero_success_pairs / num_pairs),
        "successful_actions_total": int(sum(entry["successful_actions"] for entry in pair_entries)),
        "mean_successful_actions_per_pair": float(
            np.mean([entry["successful_actions"] for entry in pair_entries])
        ),
    }


def compute_ranking_agreement(records: list[dict]) -> dict:
    records_by_pair = group_records(records, "pair_index")
    per_pair = []
    for pair_index, pair_records in sorted(records_by_pair.items()):
        entry = {"pair_index": int(pair_index), "n": len(pair_records)}
        for x_key, y_key in METRIC_PAIRS:
            x, y = finite_pair_arrays(pair_records, x_key, y_key)
            entry[f"{x_key}_vs_{y_key}"] = kendall_tau(x, y)
        per_pair.append(entry)

    summary = {}
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        summary[name] = summarize_corr_values(entry[name] for entry in per_pair)

    return {"per_pair": per_pair, "summary": summary}


def compute_summary_stats(records: list[dict]) -> dict:
    stats = {
        "overall": {
            key: summarize_values(record[key] for record in records)
            for key in COST_KEYS + ("block_pos_dist", "angle_dist")
        }
    }

    records_by_source = group_records(records, "source")
    stats["by_source"] = {}
    for source in sorted(records_by_source, key=source_sort_key):
        source_records = records_by_source[source]
        stats["by_source"][source] = {
            key: summarize_values(record[key] for record in source_records)
            for key in COST_KEYS + ("block_pos_dist", "angle_dist")
        }
        stats["by_source"][source]["success_rate"] = success_summary(source_records, "success")
        stats["by_source"][source]["env_success_rate"] = success_summary(source_records, "env_success")

    stats["success_rate"] = success_summary(records, "success")
    stats["env_success_rate"] = success_summary(records, "env_success")
    return stats


def success_summary(records: list[dict], key: str) -> dict:
    total = len(records)
    count = int(sum(record[key] for record in records))
    return {"count": count, "total": total, "rate": float(count / total) if total else None}


def classify_failure(results: dict, low_corr_threshold: float, planner_majority_threshold: float) -> dict:
    global_corr = results["correlations"]["global"]
    encoder_corr = global_corr["c_real_z_vs_c_real_state"]["spearman"]
    predictor_corr = global_corr["c_model_vs_c_real_z"]["spearman"]
    model_state_corr = global_corr["c_model_vs_c_real_state"]["spearman"]
    zero_success_fraction = results["planner_diagnostic"]["fraction_pairs_with_zero_success"]

    encoder_broken = encoder_corr is not None and encoder_corr < low_corr_threshold
    predictor_broken = (
        not encoder_broken
        and predictor_corr is not None
        and predictor_corr < low_corr_threshold
    )
    planner_problem = zero_success_fraction > planner_majority_threshold

    triggered = []
    if encoder_broken:
        triggered.append("Case B")
    if predictor_broken:
        triggered.append("Case C")
    if planner_problem:
        triggered.append("Case D")

    if len(triggered) == 1:
        primary = triggered[0]
    elif len(triggered) > 1:
        primary = "Case F"
    else:
        primary = "Case F"

    explanations = {
        "Case A": "LeWM does not fail; not applicable for this offset sweep context.",
        "Case B": (
            "Encoder goal geometry is suspect because corr(C_real_z, C_real_state) "
            f"is below {low_corr_threshold}."
        ),
        "Case C": (
            "Predictor/rollout is suspect because corr(C_model, C_real_z) is below "
            f"{low_corr_threshold} while encoder geometry is not classified as broken."
        ),
        "Case D": (
            "Planner/action-set coverage is suspect because most pairs have zero "
            "successful actions in the fixed evaluation set."
        ),
        "Case E": "Requires per-trajectory/event-localized diagnostics; not determined here.",
        "Case F": "No single dominant rule fired, or multiple rules fired simultaneously.",
    }

    return {
        "primary": primary,
        "triggered_cases": triggered,
        "thresholds": {
            "low_corr_threshold": low_corr_threshold,
            "planner_majority_threshold": planner_majority_threshold,
        },
        "metrics_used": {
            "spearman_c_real_z_vs_c_real_state": encoder_corr,
            "spearman_c_model_vs_c_real_z": predictor_corr,
            "spearman_c_model_vs_c_real_state": model_state_corr,
            "fraction_pairs_with_zero_success": zero_success_fraction,
        },
        "explanation": explanations[primary],
        "case_notes": explanations,
    }


def group_records(records: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return dict(grouped)


def source_sort_key(source: str) -> tuple[int, str]:
    try:
        return (SOURCE_ORDER.index(source), source)
    except ValueError:
        return (len(SOURCE_ORDER), source)


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "nan"
    return f"{value * 100:.{digits}f}%"


def fmt_mean_std(summary: dict, digits: int = 3) -> str:
    return f"{fmt_float(summary['mean'], digits)} +/- {fmt_float(summary['std'], digits)}"


def build_report(data: dict, results: dict) -> str:
    metadata = data.get("metadata", {})
    correlations = results["correlations"]
    stats = results["summary_statistics"]
    planner = results["planner_diagnostic"]
    ranking = results["ranking_agreement"]
    classification = results["classification"]

    lines = []
    lines.append("Three-Cost Attribution Analysis")
    lines.append("=" * 33)
    lines.append(
        "Input: "
        f"offset={data.get('offset')} pairs={results['counts']['pairs']} "
        f"actions={results['counts']['actions']} "
        f"actions_per_pair={results['counts']['actions_per_pair']}"
    )
    if metadata:
        lines.append(
            "Metadata: "
            f"seed={metadata.get('seed')} device={metadata.get('device')} "
            f"sources={', '.join(metadata.get('sources', []))}"
        )
    lines.append("")

    lines.append("1. Core Attribution Correlations")
    lines.append("Global correlations across all actions:")
    lines.append("  metric_pair                         pearson   spearman   n")
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        entry = correlations["global"][name]
        lines.append(
            f"  {name:<34} {fmt_float(entry['pearson']):>7}   "
            f"{fmt_float(entry['spearman']):>8}   {entry['n']}"
        )
    lines.append("")
    lines.append("Per-pair correlations, mean +/- std over pairs:")
    lines.append("  metric_pair                         pearson          spearman")
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        entry = correlations["per_pair_summary"][name]
        lines.append(
            f"  {name:<34} {fmt_mean_std(entry['pearson']):>15}   "
            f"{fmt_mean_std(entry['spearman']):>15}"
        )
    lines.append("")
    lines.append("Per-source correlations:")
    lines.append("  source      metric_pair                         pearson   spearman   n")
    for source, source_corr in correlations["per_source"].items():
        for x_key, y_key in METRIC_PAIRS:
            name = f"{x_key}_vs_{y_key}"
            entry = source_corr[name]
            lines.append(
                f"  {source:<10} {name:<34} {fmt_float(entry['pearson']):>7}   "
                f"{fmt_float(entry['spearman']):>8}   {entry['n']}"
            )
    lines.append("")

    lines.append("2. Planner Diagnostic")
    lines.append(
        "Pairs with at least one block-success action: "
        f"{planner['pairs_with_success']}/{planner['num_pairs']} "
        f"({fmt_pct(planner['fraction_pairs_with_success'])})"
    )
    lines.append(
        "Pairs with zero block-success actions: "
        f"{planner['pairs_with_zero_success']}/{planner['num_pairs']} "
        f"({fmt_pct(planner['fraction_pairs_with_zero_success'])})"
    )
    lines.append(
        "Successful actions total: "
        f"{planner['successful_actions_total']}/{results['counts']['actions']} "
        f"(mean per pair {planner['mean_successful_actions_per_pair']:.2f})"
    )
    lines.append("")

    lines.append("3. Cost Ranking Agreement")
    lines.append("Kendall tau by pair, mean +/- std:")
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        lines.append(f"  {name:<34} {fmt_mean_std(ranking['summary'][name])}")
    lines.append("")

    lines.append("4. Summary Statistics")
    lines.append("Overall cost and distance distributions:")
    lines.append("  metric             mean       std       min       p25    median       p75       max")
    for key in COST_KEYS + ("block_pos_dist", "angle_dist"):
        entry = stats["overall"][key]
        lines.append(
            f"  {key:<15} {fmt_float(entry['mean'], 3):>8} "
            f"{fmt_float(entry['std'], 3):>9} {fmt_float(entry['min'], 3):>9} "
            f"{fmt_float(entry['p25'], 3):>9} {fmt_float(entry['median'], 3):>9} "
            f"{fmt_float(entry['p75'], 3):>9} {fmt_float(entry['max'], 3):>9}"
        )
    lines.append("")
    lines.append("By-source means/stds and success:")
    lines.append(
        "  source      success  env_succ   c_model       c_real_z      "
        "c_real_state   block_pos     angle"
    )
    for source, source_stats in stats["by_source"].items():
        lines.append(
            f"  {source:<10} "
            f"{fmt_pct(source_stats['success_rate']['rate']):>7} "
            f"{fmt_pct(source_stats['env_success_rate']['rate']):>9} "
            f"{fmt_mean_std(source_stats['c_model']):>13} "
            f"{fmt_mean_std(source_stats['c_real_z']):>13} "
            f"{fmt_mean_std(source_stats['c_real_state']):>14} "
            f"{fmt_mean_std(source_stats['block_pos_dist']):>11} "
            f"{fmt_mean_std(source_stats['angle_dist']):>11}"
        )
    lines.append("")

    lines.append("5. Preliminary Decision Tree Classification")
    lines.append(f"Primary classification: {classification['primary']}")
    triggered = classification["triggered_cases"] or ["none"]
    lines.append(f"Triggered cases: {', '.join(triggered)}")
    for key, value in classification["metrics_used"].items():
        if "fraction" in key:
            lines.append(f"  {key}: {fmt_pct(value)}")
        else:
            lines.append(f"  {key}: {fmt_float(value)}")
    lines.append(f"Interpretation: {classification['explanation']}")
    lines.append("")
    lines.append("Outputs")
    lines.append(f"Detailed JSON: {results['output_path']}")
    if results["figure_paths"]:
        lines.append("Figures:")
        for path in results["figure_paths"]:
            lines.append(f"  {path}")
    else:
        lines.append("Figures: not generated")
    return "\n".join(lines)


def make_plots(records: list[dict], correlations: dict, figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    records_by_source = group_records(records, "source")
    colors = {
        "data": "#2f6f4e",
        "random": "#b45f06",
        "cem_early": "#31688e",
        "cem_late": "#7b3294",
    }

    scatter_specs = [
        ("c_model", "c_real_z", "scatter_cmodel_vs_crealz.png"),
        ("c_model", "c_real_state", "scatter_cmodel_vs_crealstate.png"),
        ("c_real_z", "c_real_state", "scatter_crealz_vs_crealstate.png"),
    ]
    paths = []
    for x_key, y_key, filename in scatter_specs:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        for source in sorted(records_by_source, key=source_sort_key):
            source_records = records_by_source[source]
            ax.scatter(
                [record[x_key] for record in source_records],
                [record[y_key] for record in source_records],
                s=18,
                alpha=0.55,
                color=colors.get(source, "#555555"),
                label=source,
                edgecolors="none",
            )
        name = f"{x_key}_vs_{y_key}"
        corr = correlations["global"][name]
        ax.set_title(f"{x_key} vs {y_key}")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.text(
            0.02,
            0.98,
            f"Pearson={fmt_float(corr['pearson'])}\nSpearman={fmt_float(corr['spearman'])}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#bbbbbb", "alpha": 0.85},
        )
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))

    pair_entries = correlations["per_pair"]
    pair_indices = np.asarray([entry["pair_index"] for entry in pair_entries])
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for offset_idx, (x_key, y_key) in enumerate(METRIC_PAIRS):
        name = f"{x_key}_vs_{y_key}"
        values = [
            np.nan if entry[name]["spearman"] is None else entry[name]["spearman"]
            for entry in pair_entries
        ]
        x_positions = pair_indices + (offset_idx - 1) * width
        ax.bar(x_positions, values, width=width, label=name)
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Per-pair Spearman correlations")
    ax.set_xlabel("pair_index")
    ax.set_ylabel("Spearman correlation")
    ax.set_xticks(pair_indices)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=1)
    fig.tight_layout()
    path = figures_dir / "correlation_by_pair.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))
    return paths


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    data, records = load_records(args.input)
    pair_count = len({record["pair_index"] for record in records})
    action_counts = [len(group) for group in group_records(records, "pair_index").values()]

    results = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "offset": data.get("offset"),
        "counts": {
            "pairs": pair_count,
            "actions": len(records),
            "actions_per_pair": {
                "mean": float(np.mean(action_counts)),
                "min": int(np.min(action_counts)),
                "max": int(np.max(action_counts)),
            },
            "by_source": {
                source: len(group)
                for source, group in sorted(group_records(records, "source").items(), key=lambda item: source_sort_key(item[0]))
            },
        },
    }
    results["correlations"] = compute_correlations(records)
    results["planner_diagnostic"] = compute_planner_diagnostic(records)
    results["ranking_agreement"] = compute_ranking_agreement(records)
    results["summary_statistics"] = compute_summary_stats(records)
    results["classification"] = classify_failure(
        results,
        low_corr_threshold=args.low_corr_threshold,
        planner_majority_threshold=args.planner_majority_threshold,
    )

    figure_paths = []
    if not args.no_plots:
        figure_paths = make_plots(records, results["correlations"], args.figures_dir)
    results["figure_paths"] = figure_paths

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(results), indent=2, allow_nan=False) + "\n")

    print(build_report(data, results))


if __name__ == "__main__":
    main()
