#!/usr/bin/env python3
"""Supplementary Track A fact analyses for F1 and F2."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from lewm_audit.diagnostics.cost_magnitudes import (
    COST_KEYS,
    infer_c_real_state_success_threshold,
    per_row_cost_stats,
)
from lewm_audit.diagnostics.dp1 import per_pair_spearman
from lewm_audit.diagnostics.failure_modes import (
    ENCODER_CLASSES,
    SUCCESS_CLASSES,
    all_fail_source_verification,
    classify_pairs,
    count_matrix,
    counts_by_cell_and_quadrant,
    quadrant_label,
    quadrant_table,
)


DEFAULT_THREE_COST_PATH = Path("results/phase1/track_a_three_cost.json")
DEFAULT_OUTPUT_DIR = Path("results/phase1/track_a_analysis")
DEFAULT_FIGURES_DIR = Path("results/phase1/figures/track_a")
REPORT_PATH = Path("docs/phase1/track_a_supplementary_findings.md")
DISPLACEMENT_GRID_LINES = [10.0, 50.0, 120.0]
ROTATION_GRID_LINES = [0.25, 0.75, 1.25]
CELL_ORDER = [f"D{d}xR{r}" for d in range(4) for r in range(4)]
ROW_ORDER = ["D0", "D1", "D2", "D3", "global"]


QUADRANT_COLORS = {
    "all_fail + neg_rho": "#8b1e3f",
    "all_fail + weak_rho": "#d95f02",
    "all_fail + strong_rho": "#7570b3",
    "some_succ + neg_rho": "#e7298a",
    "some_succ + weak_rho": "#66a61e",
    "some_succ + strong_rho": "#1b9e77",
    "all_succ + neg_rho": "#a6761d",
    "all_succ + weak_rho": "#666666",
    "all_succ + strong_rho": "#1f78b4",
}
SUCCESS_MARKERS = {"all_fail": "x", "some_succ": "o", "all_succ": "*"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-cost-path", default=str(DEFAULT_THREE_COST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_three_cost(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not data.get("pairs"):
        raise ValueError(f"No pairs found in {path}")
    return data


def records_by_pair(data: dict) -> dict[int, dict]:
    return {int(pair["pair_id"]): pair for pair in data["pairs"]}


def success_count_by_pair(records: dict[int, dict]) -> dict[int, int]:
    return {
        int(pair_id): int(sum(bool(action["success"]) for action in pair["actions"]))
        for pair_id, pair in records.items()
    }


def per_pair_records(records: dict[int, dict]) -> list[dict]:
    rhos = per_pair_spearman(records)
    rows = []
    for pair_id, pair in sorted(records.items()):
        rows.append(
            {
                "pair_id": int(pair_id),
                "cell": str(pair["cell"]),
                "rho": float(rhos[pair_id]),
                "total_actions": int(len(pair["actions"])),
                "block_displacement_px": float(pair["block_displacement_px"]),
                "required_rotation_rad": float(pair["required_rotation_rad"]),
            }
        )
    return rows


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None:
        return None
    return value


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, start=base)


def draw_grid_lines(ax) -> None:
    for value in DISPLACEMENT_GRID_LINES:
        ax.axvline(value, color="gray", linewidth=0.8, linestyle="--", zorder=0)
    for value in ROTATION_GRID_LINES:
        ax.axhline(value, color="gray", linewidth=0.8, linestyle="--", zorder=0)


def render_failure_mode_scatter(classified_df, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.4, 6.8), constrained_layout=True)
    for label, color in QUADRANT_COLORS.items():
        s_class, e_class = label.split(" + ")
        subset = classified_df[
            (classified_df["success_class"] == s_class)
            & (classified_df["encoder_class"] == e_class)
        ]
        if subset.empty:
            continue
        marker = SUCCESS_MARKERS[s_class]
        kwargs = {
            "label": label,
            "color": color,
            "marker": marker,
            "s": 70,
            "linewidth": 1.0,
            "alpha": 0.88,
        }
        if marker != "x":
            kwargs["edgecolor"] = "black"
        ax.scatter(
            subset["block_displacement_px"],
            subset["required_rotation_rad"],
            **kwargs,
        )
    draw_grid_lines(ax)
    ax.set_xlabel("Block displacement (px)")
    ax.set_ylabel("Required rotation (rad)")
    ax.set_title("Track A failure-mode quadrants")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_failure_mode_counts_grid(classified_df, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_by_cell = counts_by_cell_and_quadrant(classified_df)
    labels = [label for label in QUADRANT_COLORS if any(label in counts for counts in counts_by_cell.values())]
    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.set_xticks(np.arange(4) + 0.5, labels=["R0", "R1", "R2", "R3"])
    ax.set_yticks(np.arange(4) + 0.5, labels=["D0", "D1", "D2", "D3"])
    ax.set_title("Failure-mode counts by Track A cell")
    ax.set_xlabel("Rotation bin")
    ax.set_ylabel("Displacement bin")
    for d in range(4):
        for r in range(4):
            ax.add_patch(Rectangle((r, d), 1, 1, fill=False, edgecolor="#444444", linewidth=0.9))
            cell = f"D{d}xR{r}"
            counts = counts_by_cell.get(cell, {})
            total = sum(counts.values())
            if total == 0:
                ax.text(r + 0.5, d + 0.5, "0", ha="center", va="center")
                continue
            bar_x = r + 0.22
            bar_y = d + 0.16
            bar_w = 0.56
            bar_h = 0.52
            cursor = bar_y
            for label in labels:
                count = counts.get(label, 0)
                if count == 0:
                    continue
                height = bar_h * count / total
                ax.add_patch(
                    Rectangle(
                        (bar_x, cursor),
                        bar_w,
                        height,
                        facecolor=QUADRANT_COLORS[label],
                        edgecolor="white",
                        linewidth=0.4,
                    )
                )
                cursor += height
            ax.text(r + 0.5, d + 0.83, f"n={total}", ha="center", va="center", fontsize=9)
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=QUADRANT_COLORS[label], label=label)
        for label in labels
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def row_name(pair: dict) -> str:
    return str(pair["cell"]).split("x")[0]


def row_filter(name: str):
    if name == "global":
        return lambda pair: True
    return lambda pair: row_name(pair) == name


def render_cost_magnitude_by_row(records: dict[int, dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, key in zip(axes, COST_KEYS, strict=True):
        values = []
        for row in ROW_ORDER[:-1]:
            row_values = [
                float(action[key])
                for pair in records.values()
                if row_filter(row)(pair)
                for action in pair["actions"]
            ]
            values.append(row_values)
        ax.boxplot(values, tick_labels=ROW_ORDER[:-1], showfliers=False)
        ax.set_title(key)
        ax.set_xlabel("D-row")
        ax.set_ylabel("Cost value")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_best_cost_by_row(
    row_stats: dict[str, dict],
    threshold: float,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for idx, row in enumerate(ROW_ORDER[:-1]):
        values = row_stats[row]["best_C_real_state_per_pair"]
        if not values:
            continue
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(
            np.full(len(values), idx, dtype=np.float64) + jitter,
            values,
            s=42,
            color="#2f5597",
            alpha=0.86,
            edgecolor="black",
            linewidth=0.35,
        )
    ax.axhline(threshold, color="#b23a48", linewidth=1.4, linestyle="--", label="threshold proxy")
    ax.set_xticks(np.arange(4), labels=ROW_ORDER[:-1])
    ax.set_ylabel("Best C_real_state among 80 actions")
    ax.set_xlabel("D-row")
    ax.set_title("Best-of-80 C_real_state per pair by D-row")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_row_correlations(row_stats: dict[str, dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = [
        "pearson_C_model_vs_C_real_z",
        "pearson_C_real_z_vs_C_real_state",
        "pearson_C_model_vs_C_real_state",
    ]
    labels = ["C_model vs C_real_z", "C_real_z vs C_real_state", "C_model vs C_real_state"]
    x = np.arange(len(ROW_ORDER[:-1]))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    for offset, key, label in zip([-width, 0.0, width], metric_keys, labels, strict=True):
        values = [
            row_stats[row]["pairwise_pearson"][key]
            if row_stats[row]["pairwise_pearson"][key] is not None
            else np.nan
            for row in ROW_ORDER[:-1]
        ]
        ax.bar(x + offset, values, width=width, label=label)
    ax.set_xticks(x, labels=ROW_ORDER[:-1])
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Per-row pairwise cost correlations")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_failure_json(
    classified_df,
    q_table: dict,
    c_matrix: dict,
    verification: dict,
) -> dict:
    records = []
    for pair_id, row in classified_df.sort_index().iterrows():
        entry = {"pair_id": int(pair_id)}
        entry.update(row.to_dict())
        records.append(entry)
    return {
        "classified_pairs": to_jsonable(records),
        "quadrant_table": to_jsonable(q_table),
        "summary_count_matrix": to_jsonable(c_matrix),
        "all_fail_per_source_verification": to_jsonable(verification),
    }


def make_row_stats(records: dict[int, dict], threshold: float) -> dict:
    stats = {}
    for row in ROW_ORDER:
        stats[row] = per_row_cost_stats(records, row_filter(row), success_threshold=threshold)
    return stats


def mean_or_none(values) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else None


def write_report(
    *,
    report_path: Path,
    three_cost_path: Path,
    data: dict,
    failure_json_path: Path,
    row_json_path: Path,
    figure_paths: dict[str, Path],
    classified_df,
    q_table: dict,
    c_matrix: dict,
    verification: dict,
    row_stats: dict,
    threshold_info: dict,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    base = report_path.parent
    metadata = data["metadata"]

    lines = []
    lines.append("# Track A Supplementary Findings")
    lines.append("")
    lines.append("## 1. Provenance")
    lines.append("")
    lines.append(f"- Data file: `{three_cost_path}`")
    lines.append(f"- Three-cost git commit: `{metadata['git_commit']}`")
    lines.append(f"- Supplementary analysis git commit: `{git_commit()}`")
    lines.append(f"- Seed: `{metadata['seed']}`")
    lines.append("- This is a fact sheet for F1/F2 follow-up measurements, not interpretation.")
    lines.append(f"- Failure-mode JSON: `{failure_json_path}`")
    lines.append(f"- D-row cost JSON: `{row_json_path}`")
    lines.append("")

    lines.append("## 2. Failure-Mode Decomposition")
    lines.append("")
    lines.append("| Success class | neg_rho | weak_rho | strong_rho |")
    lines.append("|---|---:|---:|---:|")
    for s_class in SUCCESS_CLASSES:
        row = c_matrix[s_class]
        lines.append(
            f"| {s_class} | {row['neg_rho']} | {row['weak_rho']} | {row['strong_rho']} |"
        )
    lines.append("")
    lines.append("| Quadrant | n_pairs | Pair IDs | Mean displacement px | Mean rotation rad | Cells touched |")
    lines.append("|---|---:|---|---:|---:|---|")
    for s_class in SUCCESS_CLASSES:
        for e_class in ENCODER_CLASSES:
            entry = q_table[s_class][e_class]
            lines.append(
                f"| {s_class} + {e_class} | {entry['n_pairs']} | "
                f"{', '.join(map(str, entry['pair_ids'])) if entry['pair_ids'] else '-'} | "
                f"{fmt(entry['mean_displacement_px'], 2)} | {fmt(entry['mean_rotation_rad'], 3)} | "
                f"{', '.join(entry['cells_present']) if entry['cells_present'] else '-'} |"
            )
    lines.append("")
    lines.append(
        f"All-fail per-source verification: `{verification['all_source_rates_zero']}` "
        f"across `{verification['n_all_fail_pairs']}` all-fail pairs."
    )
    if verification["flagged_pairs"]:
        lines.append(f"Flagged all-fail pairs: `{verification['flagged_pairs']}`")
    lines.append("")
    lines.append(f"![Failure-mode quadrants]({relative(figure_paths['failure_mode_quadrants'], base)})")
    lines.append("")
    lines.append(f"![Failure-mode counts grid]({relative(figure_paths['failure_mode_counts_grid'], base)})")
    lines.append("")
    for s_class in SUCCESS_CLASSES:
        for e_class in ENCODER_CLASSES:
            entry = q_table[s_class][e_class]
            if entry["n_pairs"] == 0:
                continue
            lines.append(
                f"{s_class} + {e_class} contains {entry['n_pairs']} pairs; mean displacement "
                f"{fmt(entry['mean_displacement_px'], 2)} px and mean rotation "
                f"{fmt(entry['mean_rotation_rad'], 3)} rad. Cells touched: "
                f"{', '.join(entry['cells_present'])}."
            )
    lines.append("")

    lines.append("## 3. D3-Row Cost-Magnitude Diagnosis")
    lines.append("")
    lines.append("| Row | Cost | Mean | Std | Median | IQR | Min | Max |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in ROW_ORDER:
        stats = row_stats[row]
        for cost in COST_KEYS:
            cost_stats = stats[cost]
            lines.append(
                f"| {row} | {cost} | {fmt(cost_stats['mean'], 3)} | "
                f"{fmt(cost_stats['std'], 3)} | {fmt(cost_stats['median'], 3)} | "
                f"{fmt(cost_stats['iqr'], 3)} | {fmt(cost_stats['min'], 3)} | "
                f"{fmt(cost_stats['max'], 3)} |"
            )
    lines.append("")
    lines.append("| Row | Median best C_real_state | Pairs below threshold proxy | Pearson C_model/C_real_z | Pearson C_real_z/C_real_state | Pearson C_model/C_real_state |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in ROW_ORDER:
        stats = row_stats[row]
        corr = stats["pairwise_pearson"]
        lines.append(
            f"| {row} | {fmt(stats['median_best_C_real_state'], 3)} | "
            f"{stats['n_pairs_with_min_C_real_state_below_success_threshold']} | "
            f"{fmt(corr['pearson_C_model_vs_C_real_z'], 3)} | "
            f"{fmt(corr['pearson_C_real_z_vs_C_real_state'], 3)} | "
            f"{fmt(corr['pearson_C_model_vs_C_real_state'], 3)} |"
        )
    lines.append("")
    lines.append(
        f"Inferred scalar C_real_state threshold proxy: `{fmt(threshold_info['threshold'], 6)}`. "
        f"Method: `{threshold_info['method']}`. The direct PushT criterion is "
        "`block_pos_dist < 20.0 and angle_dist < pi/9`; C_real_state is their sum."
    )
    lines.append("")
    lines.append(f"![Cost magnitude by row]({relative(figure_paths['cost_magnitude_by_row'], base)})")
    lines.append("")
    lines.append("The cost-magnitude figure shows the distribution of each stored cost over all action records in D0-D3. Each subplot uses the same D-row grouping.")
    lines.append("")
    lines.append(f"![Best cost per pair by row]({relative(figure_paths['best_cost_per_pair_by_row'], base)})")
    lines.append("")
    lines.append("The best-cost figure shows the minimum C_real_state among 80 actions for each pair, split by D-row. The dashed line is the scalar threshold proxy described above.")
    lines.append("")
    lines.append(f"![Row Pearson correlations]({relative(figure_paths['row_pearson_correlations'], base)})")
    lines.append("")
    lines.append("The row-correlation figure shows the three pairwise Pearson correlations among C_real_z, C_model, and C_real_state for each D-row.")
    lines.append("")

    lines.append("## 4. Limitations")
    lines.append("")
    lines.append("- This report does not select a winning hypothesis among H_B1/B2/B3.")
    lines.append("- all_fail vs some_succ is binary at success_count=0; a pair with success_count=1 is treated very differently from one with 0, even though they may be physically similar.")
    lines.append("- encoder_class thresholds (0, 0.3) are inherited from DP1 and Phase 0; sensitivity to these thresholds is not tested here.")
    lines.append("- The scalar C_real_state threshold proxy is not equivalent to the conjunctive PushT success criterion.")
    lines.append("- This report does not update Phase 0 case classification, Failure Atlas pages, or Track B/C/D experiments.")
    lines.append("")
    report_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    three_cost_path = Path(args.three_cost_path)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_three_cost(three_cost_path)
    records = records_by_pair(data)
    success_counts = success_count_by_pair(records)
    classified = classify_pairs(per_pair_records(records), success_counts)
    q_table = quadrant_table(classified)
    c_matrix = count_matrix(classified)
    verification = all_fail_source_verification(records)
    threshold_info = infer_c_real_state_success_threshold(records)
    row_stats = make_row_stats(records, threshold_info["threshold"])

    figure_paths = {
        "failure_mode_quadrants": figures_dir / "failure_mode_quadrants.png",
        "failure_mode_counts_grid": figures_dir / "failure_mode_counts_grid.png",
        "cost_magnitude_by_row": figures_dir / "cost_magnitude_by_row.png",
        "best_cost_per_pair_by_row": figures_dir / "best_cost_per_pair_by_row.png",
        "row_pearson_correlations": figures_dir / "row_pearson_correlations.png",
    }
    render_failure_mode_scatter(classified, figure_paths["failure_mode_quadrants"])
    render_failure_mode_counts_grid(classified, figure_paths["failure_mode_counts_grid"])
    render_cost_magnitude_by_row(records, figure_paths["cost_magnitude_by_row"])
    render_best_cost_by_row(row_stats, threshold_info["threshold"], figure_paths["best_cost_per_pair_by_row"])
    render_row_correlations(row_stats, figure_paths["row_pearson_correlations"])

    failure_json_path = output_dir / "failure_mode_decomposition.json"
    row_json_path = output_dir / "d_row_cost_diagnosis.json"
    failure_json_path.write_text(
        json.dumps(
            make_failure_json(classified, q_table, c_matrix, verification),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    row_json_path.write_text(
        json.dumps(
            {
                "threshold_info": threshold_info,
                "row_stats": row_stats,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    write_report(
        report_path=REPORT_PATH,
        three_cost_path=three_cost_path,
        data=data,
        failure_json_path=failure_json_path,
        row_json_path=row_json_path,
        figure_paths=figure_paths,
        classified_df=classified,
        q_table=q_table,
        c_matrix=c_matrix,
        verification=verification,
        row_stats=row_stats,
        threshold_info=threshold_info,
    )

    all_fail_strong = q_table["all_fail"]["strong_rho"]
    low_row_corrs = {
        row: row_stats[row]["pairwise_pearson"]["pearson_C_model_vs_C_real_z"]
        for row in ROW_ORDER
    }
    print(
        json.dumps(
            {
                "summary_count_matrix": c_matrix,
                "all_fail_strong_rho": all_fail_strong,
                "all_fail_source_verification": {
                    "n_all_fail_pairs": verification["n_all_fail_pairs"],
                    "all_source_rates_zero": verification["all_source_rates_zero"],
                    "flagged_pairs": verification["flagged_pairs"],
                },
                "threshold_info": threshold_info,
                "row_model_realz_pearson": low_row_corrs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
