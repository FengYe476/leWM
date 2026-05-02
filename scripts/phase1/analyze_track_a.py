#!/usr/bin/env python3
"""Analyze Track A full three-cost run: DP1, heatmaps, and sign reversals."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lewm_audit.diagnostics.dp1 import dp1_test, per_pair_spearman, sign_reversal_cluster
from lewm_audit.diagnostics.heatmap import (
    cell_grid_from_records,
    matrix_from_cell_values,
    render_heatmap,
)


DEFAULT_DISPLACEMENT_EDGES = [0.0, 10.0, 50.0, 120.0, math.inf]
DEFAULT_ROTATION_EDGES = [0.0, 0.25, 0.75, 1.25, math.inf]
CELL_ORDER = [f"D{d}xR{r}" for d in range(4) for r in range(4)]
HEATMAP_FILES = {
    "success_rate": "success_rate_heatmap.png",
    "mean_pair_spearman": "mean_pair_spearman_heatmap.png",
    "std_pair_spearman": "std_pair_spearman_heatmap.png",
    "cell_model_realz_pearson": "cell_model_realz_pearson_heatmap.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--three-cost-path",
        default="results/phase1/track_a_three_cost.json",
        help="Full Track A three-cost JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase1/track_a_analysis/",
        help="Directory for machine-readable analysis outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/phase1/figures/track_a/",
        help="Directory for PNG figures.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-n", type=int, default=10000)
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


def flatten_actions(pairs: list[dict]) -> list[dict]:
    records = []
    for pair in pairs:
        for action in pair["actions"]:
            records.append(
                {
                    "pair_id": int(pair["pair_id"]),
                    "cell": str(pair["cell"]),
                    "source": str(action["source"]),
                    "C_real_z": float(action["C_real_z"]),
                    "C_model": float(action["C_model"]),
                    "C_real_state": float(action["C_real_state"]),
                    "success": bool(action["success"]),
                    "block_displacement_px": float(pair["block_displacement_px"]),
                    "required_rotation_rad": float(pair["required_rotation_rad"]),
                }
            )
    return records


def pair_records_for_grid(pairs: list[dict]) -> list[dict]:
    out = []
    for pair in pairs:
        out.append(
            {
                "pair_id": int(pair["pair_id"]),
                "cell": str(pair["cell"]),
                "block_displacement_px": float(pair["block_displacement_px"]),
                "required_rotation_rad": float(pair["required_rotation_rad"]),
            }
        )
    return out


def cell_indices(cell: str) -> tuple[int, int]:
    left, right = cell.split("x")
    return int(left[1:]), int(right[1:])


def finite_mean(values: list[float | None]) -> float | None:
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else None


def finite_std(values: list[float | None]) -> float | None:
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def compute_cell_table(
    pairs: list[dict],
    action_records: list[dict],
    pair_corrs: dict[int, float | None],
) -> tuple[list[dict], dict[str, list[float]]]:
    pairs_by_cell: dict[str, list[dict]] = defaultdict(list)
    actions_by_cell: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        pairs_by_cell[str(pair["cell"])].append(pair)
    for record in action_records:
        actions_by_cell[str(record["cell"])].append(record)

    cell_pair_corrs: dict[str, list[float]] = {}
    table = []
    for cell in CELL_ORDER:
        cell_pairs = pairs_by_cell.get(cell, [])
        cell_actions = actions_by_cell.get(cell, [])
        rhos = [pair_corrs[int(pair["pair_id"])] for pair in cell_pairs]
        finite_rhos = [float(rho) for rho in rhos if rho is not None and np.isfinite(rho)]
        cell_pair_corrs[cell] = finite_rhos
        successes = int(sum(record["success"] for record in cell_actions))
        c_model = np.asarray([record["C_model"] for record in cell_actions], dtype=np.float64)
        c_real_z = np.asarray([record["C_real_z"] for record in cell_actions], dtype=np.float64)
        table.append(
            {
                "cell": cell,
                "N_pairs": int(len(cell_pairs)),
                "N_records": int(len(cell_actions)),
                "success_count": successes,
                "success_rate": float(successes / len(cell_actions)) if cell_actions else None,
                "mean_pair_spearman_c_real_z_vs_c_real_state": finite_mean(finite_rhos),
                "std_pair_spearman_c_real_z_vs_c_real_state": finite_std(finite_rhos),
                "cell_pearson_c_model_vs_c_real_z": pearson_corr(c_model, c_real_z),
            }
        )
    return table, cell_pair_corrs


def matrix_for(table: list[dict], key: str) -> np.ndarray:
    values = {row["cell"]: row[key] for row in table}
    return matrix_from_cell_values(values, n_displacement_bins=4, n_rotation_bins=4)


def count_matrix(table: list[dict]) -> np.ndarray:
    return matrix_from_cell_values(
        {row["cell"]: row["N_pairs"] for row in table},
        n_displacement_bins=4,
        n_rotation_bins=4,
    )


def render_distribution_plot(
    *,
    pairs: list[dict],
    pair_corrs: dict[int, float | None],
    sign_reversal_pairs: list[dict],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite_items = [
        (pair, pair_corrs[int(pair["pair_id"])])
        for pair in pairs
        if pair_corrs[int(pair["pair_id"])] is not None and np.isfinite(pair_corrs[int(pair["pair_id"])])
    ]
    rhos = np.asarray([rho for _, rho in finite_items], dtype=np.float64)
    neg_rhos = np.asarray([item["rho"] for item in sign_reversal_pairs], dtype=np.float64)
    displacements = np.asarray([float(pair["block_displacement_px"]) for pair, _ in finite_items])
    rotations = np.asarray([float(pair["required_rotation_rad"]) for pair, _ in finite_items])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].hist(rhos, bins=np.linspace(-1, 1, 21), color="#9aa6b2", edgecolor="white", label="All pairs")
    if len(neg_rhos):
        axes[0].hist(
            neg_rhos,
            bins=np.linspace(-1, 0, 11),
            color="#b23a48",
            edgecolor="white",
            label="rho < 0",
        )
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_title("Per-pair Spearman rho distribution")
    axes[0].set_xlabel("rho(C_real_z, C_real_state)")
    axes[0].set_ylabel("Pair count")
    axes[0].legend()

    scatter = axes[1].scatter(
        displacements,
        rotations,
        c=rhos,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        s=52,
        edgecolor="black",
        linewidth=0.35,
    )
    axes[1].set_title("Per-pair rho over physical grid")
    axes[1].set_xlabel("Block displacement (px)")
    axes[1].set_ylabel("Required rotation (rad)")
    axes[1].axvline(10.0, color="gray", linewidth=0.7, linestyle="--")
    axes[1].axvline(50.0, color="gray", linewidth=0.7, linestyle="--")
    axes[1].axvline(120.0, color="gray", linewidth=0.7, linestyle="--")
    axes[1].axhline(0.25, color="gray", linewidth=0.7, linestyle="--")
    axes[1].axhline(0.75, color="gray", linewidth=0.7, linestyle="--")
    axes[1].axhline(1.25, color="gray", linewidth=0.7, linestyle="--")
    fig.colorbar(scatter, ax=axes[1], label="Spearman rho")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_cell_distribution_plot(
    *,
    cell_pair_corrs: dict[str, list[float]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = [cell_pair_corrs[cell] for cell in CELL_ORDER]
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    ax.boxplot(values, tick_labels=CELL_ORDER, showmeans=True, patch_artist=True)
    for idx, cell_values in enumerate(values, start=1):
        if not cell_values:
            continue
        x = np.full(len(cell_values), idx, dtype=np.float64)
        jitter = np.linspace(-0.08, 0.08, len(cell_values)) if len(cell_values) > 1 else np.zeros(1)
        ax.scatter(x + jitter, cell_values, color="#2f5597", s=24, zorder=3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Per-cell distribution of per-pair Spearman rho")
    ax.set_xlabel("Cell")
    ax.set_ylabel("rho(C_real_z, C_real_state)")
    ax.tick_params(axis="x", labelrotation=45)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, start=base)


def write_report(
    *,
    report_path: Path,
    three_cost_path: Path,
    data: dict,
    analysis_json_path: Path,
    sign_reversal_path: Path,
    figure_paths: dict[str, Path],
    dp1: dict,
    cell_table: list[dict],
    sign_pairs: list[dict],
    source_git_commit: str,
    seed: int,
    bootstrap_n: int,
) -> None:
    metadata = data["metadata"]
    action_counts = metadata["action_counts"]
    cem = metadata["cem_config"]
    cell_counts = Counter(item["cell"] for item in sign_pairs)
    phase0 = dp1["phase0_reference"]
    report_base = report_path.parent
    verdict_phrase = {
        "pass": "passes",
        "fail": "fails",
        "ambiguous": "is ambiguous",
    }[dp1["verdict"]]

    lines = []
    lines.append("# Track A Analysis Report")
    lines.append("")
    lines.append("## 1. Run Provenance")
    lines.append("")
    lines.append(f"- Data source: `{three_cost_path}`")
    lines.append(f"- Three-cost run git commit: `{metadata['git_commit']}`")
    lines.append(f"- Analysis git commit: `{source_git_commit}`")
    lines.append(f"- Three-cost seed: `{metadata['seed']}`")
    lines.append(f"- Analysis seed: `{seed}`")
    lines.append(f"- Bootstrap samples: `{bootstrap_n}`")
    lines.append(f"- Offset: `{metadata['offset_steps_at_runtime']}` raw steps")
    lines.append(
        f"- Pairs/actions: `{metadata['n_pairs_completed']}` pairs, "
        f"`{sum(action_counts.values())}` actions per pair "
        f"({action_counts['data']}/{action_counts['smooth_random']}/"
        f"{action_counts['CEM_early']}/{action_counts['CEM_late']})"
    )
    lines.append(
        f"- CEM: samples `{cem['samples_per_iter']}`, iterations `{cem['iterations']}`, "
        f"elites `{cem['elites']}`, planning horizon `{cem['planning_horizon']}`, "
        f"receding horizon `{cem['receding_horizon']}`, action block `{cem['action_block']}`"
    )
    lines.append(
        f"- Fixed sequence length: `{metadata['fixed_sequence_length_raw_steps']}` raw steps / "
        f"`{metadata['fixed_sequence_length_action_blocks']}` action blocks"
    )
    lines.append(f"- Machine-readable analysis: `{analysis_json_path}`")
    lines.append(f"- Sign-reversal JSON: `{sign_reversal_path}`")
    lines.append("")

    lines.append("## 2. DP1 Result")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")
    for key in (
        "n_pairs_used",
        "mean_rho",
        "std_rho",
        "median_rho",
        "min_rho",
        "max_rho",
        "ci_low_std",
        "ci_high_std",
        "threshold",
    ):
        value = dp1[key]
        lines.append(f"| {key} | {value if isinstance(value, int) else fmt(float(value), 6)} |")
    lines.append(f"| verdict | {dp1['verdict']} |")
    lines.append("")
    lines.append(
        f"DP1 {verdict_phrase} because the 95% bootstrap CI lower bound on the "
        f"per-pair Spearman standard deviation is {fmt(dp1['ci_low_std'], 3)}, "
        f"versus the threshold {fmt(dp1['threshold'], 3)}. Phase 0 reference values were "
        f"mean {fmt(phase0['mean'], 3)} and std {fmt(phase0['std'], 3)}; Track A measured "
        f"mean {fmt(dp1['mean_rho'], 3)} and std {fmt(dp1['std_rho'], 3)} across "
        f"{dp1['n_pairs_used']} pairs."
    )
    lines.append("")

    lines.append("## 3. Heatmaps")
    lines.append("")
    lines.append(f"![Success rate]({rel(figure_paths['success_rate'], report_base)})")
    lines.append("")
    lines.append(
        "The success-rate heatmap shows the empirical success fraction within each "
        "(displacement, rotation) cell over all mixed-source action sequences. "
        "The annotations give the cell value and number of sampled pairs."
    )
    lines.append("")
    lines.append(f"![Mean per-pair Spearman]({rel(figure_paths['mean_pair_spearman'], report_base)})")
    lines.append("")
    lines.append(
        "The mean per-pair Spearman heatmap summarizes corr(C_real_z, C_real_state) "
        "after first computing one correlation per pair. Negative cells are shown "
        "with the opposite side of the diverging color scale."
    )
    lines.append("")
    lines.append(f"![Std per-pair Spearman]({rel(figure_paths['std_pair_spearman'], report_base)})")
    lines.append("")
    lines.append(
        "The standard-deviation heatmap shows within-cell spread of the per-pair "
        "Spearman values. With 6 or 7 pairs per cell, these standard deviations are "
        "small-sample estimates."
    )
    lines.append("")
    lines.append(f"![Cell Pearson C_model vs C_real_z]({rel(figure_paths['cell_model_realz_pearson'], report_base)})")
    lines.append("")
    lines.append(
        "The model-vs-real-encoder heatmap computes one Pearson correlation per cell "
        "over the cell's mixed-source action records. It describes agreement between "
        "predicted and real encoder costs within each cell."
    )
    lines.append("")
    lines.append(f"![Per-cell rho distribution]({rel(figure_paths['per_cell_rho_distribution'], report_base)})")
    lines.append("")
    lines.append(
        "The per-cell distribution plot shows the individual per-pair Spearman values "
        "inside each cell. It is included to expose within-cell spread directly."
    )
    lines.append("")

    lines.append("## 4. Sign-Reversal Cluster")
    lines.append("")
    lines.append(f"- Pairs with rho < 0: `{len(sign_pairs)}`")
    if cell_counts:
        breakdown = ", ".join(f"{cell}: {cell_counts[cell]}" for cell in sorted(cell_counts))
    else:
        breakdown = "none"
    lines.append(f"- Cell breakdown: {breakdown}")
    lines.append("")
    lines.append(f"![Sign-reversal distribution]({rel(figure_paths['sign_reversal_distribution'], report_base)})")
    lines.append("")
    lines.append("| Pair | Cell | rho | Success count | Displacement px | Rotation rad |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for item in sign_pairs:
        lines.append(
            f"| {item['pair_id']} | {item['cell']} | {fmt(item['rho'], 3)} | "
            f"{item['success_count']} | {fmt(item['block_displacement_px'], 2)} | "
            f"{fmt(item['required_rotation_rad'], 3)} |"
        )
    lines.append("")
    if sign_pairs:
        lines.append(
            "The sign-reversal subset is distributed across "
            f"{len(cell_counts)} cells. The largest counts are "
            + ", ".join(
                f"{cell} ({count})" for cell, count in sorted(cell_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            )
            + ". This paragraph is a distributional description only."
        )
    else:
        lines.append("No pairs had rho < 0.")
    lines.append("")

    lines.append("## 5. Limitations / Open Questions")
    lines.append("")
    lines.append("- This does not show whether sign reversal is rotation-specific or angle-discontinuity specific.")
    lines.append("- Cell `N_pairs` is 6 or 7, so per-cell standard-deviation estimates are noisy.")
    lines.append("- Pearson `C_model` vs `C_real_z` aggregates across 80 mixed-source actions per pair and is not source-stratified.")
    lines.append("- This report does not update Phase 0's case classification or the Failure Atlas.")
    lines.append("- This report does not test Track B, Track C, or Track D hypotheses.")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    three_cost_path = Path(args.three_cost_path)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_three_cost(three_cost_path)
    pairs = data["pairs"]
    by_pair = records_by_pair(data)
    action_records = flatten_actions(pairs)
    grid = cell_grid_from_records(pair_records_for_grid(pairs), DEFAULT_DISPLACEMENT_EDGES, DEFAULT_ROTATION_EDGES)
    grid_counts = {cell: len(records) for cell, records in grid.items()}
    if sorted(grid_counts) != CELL_ORDER:
        raise ValueError("Unexpected Track A cell grid")

    pair_corrs = per_pair_spearman(by_pair)
    dp1 = dp1_test(pair_corrs, n_bootstrap=args.bootstrap_n, rng_seed=args.seed)
    cell_table, cell_pair_corrs = compute_cell_table(pairs, action_records, pair_corrs)
    sign_pairs = sign_reversal_cluster(by_pair)

    counts = count_matrix(cell_table)
    figure_paths = {
        "success_rate": figures_dir / HEATMAP_FILES["success_rate"],
        "mean_pair_spearman": figures_dir / HEATMAP_FILES["mean_pair_spearman"],
        "std_pair_spearman": figures_dir / HEATMAP_FILES["std_pair_spearman"],
        "cell_model_realz_pearson": figures_dir / HEATMAP_FILES["cell_model_realz_pearson"],
        "sign_reversal_distribution": figures_dir / "sign_reversal_distribution.png",
        "per_cell_rho_distribution": figures_dir / "per_cell_rho_distribution.png",
    }
    render_heatmap(
        100.0 * matrix_for(cell_table, "success_rate"),
        DEFAULT_DISPLACEMENT_EDGES,
        DEFAULT_ROTATION_EDGES,
        "Track A success rate by cell (%)",
        "viridis",
        figure_paths["success_rate"],
        value_fmt="{:.1f}",
        annotation_counts=counts,
    )
    render_heatmap(
        matrix_for(cell_table, "mean_pair_spearman_c_real_z_vs_c_real_state"),
        DEFAULT_DISPLACEMENT_EDGES,
        DEFAULT_ROTATION_EDGES,
        "Mean per-pair Spearman: C_real_z vs C_real_state",
        "RdBu_r",
        figure_paths["mean_pair_spearman"],
        value_fmt="{:.2f}",
        annotation_counts=counts,
    )
    render_heatmap(
        matrix_for(cell_table, "std_pair_spearman_c_real_z_vs_c_real_state"),
        DEFAULT_DISPLACEMENT_EDGES,
        DEFAULT_ROTATION_EDGES,
        "Std per-pair Spearman: C_real_z vs C_real_state",
        "RdBu_r",
        figure_paths["std_pair_spearman"],
        value_fmt="{:.2f}",
        annotation_counts=counts,
    )
    render_heatmap(
        matrix_for(cell_table, "cell_pearson_c_model_vs_c_real_z"),
        DEFAULT_DISPLACEMENT_EDGES,
        DEFAULT_ROTATION_EDGES,
        "Cell Pearson: C_model vs C_real_z",
        "magma",
        figure_paths["cell_model_realz_pearson"],
        value_fmt="{:.2f}",
        annotation_counts=counts,
    )
    render_distribution_plot(
        pairs=pairs,
        pair_corrs=pair_corrs,
        sign_reversal_pairs=sign_pairs,
        output_path=figure_paths["sign_reversal_distribution"],
    )
    render_cell_distribution_plot(
        cell_pair_corrs=cell_pair_corrs,
        output_path=figure_paths["per_cell_rho_distribution"],
    )

    analysis_json_path = output_dir / "track_a_analysis.json"
    sign_reversal_path = output_dir / "track_a_sign_reversal_pairs.json"
    analysis = {
        "metadata": {
            "three_cost_path": str(three_cost_path),
            "source_git_commit": data["metadata"]["git_commit"],
            "analysis_git_commit": git_commit(),
            "seed": int(args.seed),
            "bootstrap_n": int(args.bootstrap_n),
            "displacement_edges": [0.0, 10.0, 50.0, 120.0, "inf"],
            "rotation_edges": [0.0, 0.25, 0.75, 1.25, "inf"],
        },
        "dp1": dp1,
        "per_cell_table": cell_table,
        "cell_pair_corrs": cell_pair_corrs,
        "figures": {key: str(path) for key, path in figure_paths.items()},
    }
    analysis_json_path.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    sign_reversal_path.write_text(json.dumps(sign_pairs, indent=2, allow_nan=False) + "\n")

    write_report(
        report_path=Path("docs/phase1/track_a_analysis_report.md"),
        three_cost_path=three_cost_path,
        data=data,
        analysis_json_path=analysis_json_path,
        sign_reversal_path=sign_reversal_path,
        figure_paths=figure_paths,
        dp1=dp1,
        cell_table=cell_table,
        sign_pairs=sign_pairs,
        source_git_commit=git_commit(),
        seed=args.seed,
        bootstrap_n=args.bootstrap_n,
    )

    print(json.dumps({"dp1": dp1, "sign_reversal_count": len(sign_pairs)}, indent=2))


if __name__ == "__main__":
    main()
