#!/usr/bin/env python3
"""Render publication figures for the full Stage 1B dimension sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE1B_PATH = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1b_full.json"
DEFAULT_STAGE1A_PATH = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_full.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2" / "figures" / "stage1b"
DIMENSIONS = (1, 2, 4, 8, 16, 32, 64, 128, 192)

PLANNING_COLOR = "#1f77b4"
ENDPOINT_COLOR = "#d55e00"
GRAY = "#6f6f6f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1b-path", type=Path, default=DEFAULT_STAGE1B_PATH)
    parser.add_argument("--stage1a-path", type=Path, default=DEFAULT_STAGE1A_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object")
    return data


def metric_mean(bucket: dict[str, Any], metric: str) -> float:
    try:
        return float(bucket[metric]["mean"])
    except KeyError as exc:
        raise KeyError(f"Missing aggregate metric {metric!r}") from exc


def load_stage1b_series(path: Path) -> dict[str, np.ndarray | float]:
    data = load_json(path)
    by_dimension = data.get("aggregate", {}).get("by_dimension")
    if not isinstance(by_dimension, dict):
        raise ValueError(f"{path} is missing aggregate.by_dimension")

    records: dict[str, list[float]] = {
        "success": [],
        "planning_spearman": [],
        "pairwise": [],
        "false_elite": [],
        "action_l2": [],
    }
    for dim in DIMENSIONS:
        bucket = by_dimension.get(str(dim))
        if not isinstance(bucket, dict):
            raise ValueError(f"{path} is missing aggregate.by_dimension.{dim}")
        records["success"].append(metric_mean(bucket, "projected_success_rate"))
        records["planning_spearman"].append(metric_mean(bucket, "endpoint_spearman"))
        records["pairwise"].append(metric_mean(bucket, "pairwise_accuracy"))
        records["false_elite"].append(metric_mean(bucket, "false_elite_rate"))
        records["action_l2"].append(metric_mean(bucket, "action_l2_to_default_blocked"))

    return {
        "dims": np.asarray(DIMENSIONS, dtype=float),
        "success": np.asarray(records["success"], dtype=float),
        "planning_spearman": np.asarray(records["planning_spearman"], dtype=float),
        "pairwise": np.asarray(records["pairwise"], dtype=float),
        "false_elite": np.asarray(records["false_elite"], dtype=float),
        "action_l2": np.asarray(records["action_l2"], dtype=float),
        "default_success": records["success"][-1],
    }


def load_stage1a_endpoint_series(path: Path) -> dict[str, np.ndarray | float]:
    data = load_json(path)
    summary_table = data.get("summary_table")
    if not isinstance(summary_table, list):
        raise ValueError(f"{path} is missing summary_table")

    c2_by_dim: dict[int, float] = {}
    c0_spearman: float | None = None
    dim_pattern = re.compile(r"gaussian_m=(\d+)$")
    for row in summary_table:
        if not isinstance(row, dict):
            continue
        control = row.get("control")
        if control == "C0":
            c0_spearman = float(row["global_spearman_mean"])
        elif control == "C2":
            match = dim_pattern.match(str(row.get("config", "")))
            if match:
                c2_by_dim[int(match.group(1))] = float(row["global_spearman_mean"])

    missing = [dim for dim in DIMENSIONS if dim not in c2_by_dim]
    if missing:
        raise ValueError(f"{path} is missing C2 Gaussian rows for dimensions: {missing}")
    if c0_spearman is None:
        raise ValueError(f"{path} is missing the C0 trained LeWM summary row")

    return {
        "dims": np.asarray(DIMENSIONS, dtype=float),
        "c2_spearman": np.asarray([c2_by_dim[dim] for dim in DIMENSIONS], dtype=float),
        "c0_spearman": c0_spearman,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_dim_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMENSIONS)
    ax.set_xticklabels([str(dim) for dim in DIMENSIONS])
    ax.set_xlabel("Projection dimension m")
    ax.tick_params(axis="both", length=3.5, width=0.8)
    ax.grid(False)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_success(stage1b: dict[str, np.ndarray | float], output_dir: Path) -> list[Path]:
    dims = stage1b["dims"]
    success = stage1b["success"]
    default_success = float(stage1b["default_success"])

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axvspan(32, 64, color=GRAY, alpha=0.12, linewidth=0)
    ax.plot(
        dims,
        success,
        color=PLANNING_COLOR,
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        label="Projected CEM_late",
    )
    ax.axhline(
        default_success,
        color=GRAY,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        label=f"m=192 baseline ({default_success:.1%})",
    )
    ax.text(45.25, 0.135, "elbow\n32-64", ha="center", va="center", color=GRAY, fontsize=10)
    format_dim_axis(ax)
    ax.set_ylabel("CEM_late success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.0, max(0.38, float(np.max(success)) * 1.16))
    ax.set_title("Planning Success vs Projection Dimension")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig1_planning_success_vs_dimension")


def plot_decoupling(
    stage1b: dict[str, np.ndarray | float],
    stage1a: dict[str, np.ndarray | float],
    output_dir: Path,
) -> list[Path]:
    dims = stage1b["dims"]
    planning_spearman = stage1b["planning_spearman"]
    endpoint_spearman = stage1a["c2_spearman"]
    c0_spearman = float(stage1a["c0_spearman"])

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axhline(0.0, color="#b0b0b0", linewidth=0.9, zorder=0)
    ax.plot(
        dims,
        endpoint_spearman,
        color=ENDPOINT_COLOR,
        marker="s",
        linewidth=2.2,
        markersize=5.2,
        label="Stage 1A endpoint Spearman (C2)",
    )
    ax.plot(
        dims,
        planning_spearman,
        color=PLANNING_COLOR,
        marker="o",
        linewidth=2.2,
        markersize=5.2,
        label="Stage 1B planning Spearman",
    )
    ax.axhline(
        c0_spearman,
        color=ENDPOINT_COLOR,
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        alpha=0.75,
        label=f"C0 trained LeWM ({c0_spearman:.3f})",
    )
    format_dim_axis(ax)
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(-0.08, 0.56)
    ax.set_title("Endpoint Ranking Preserves, Planning Ranking Does Not")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig2_endpoint_planning_decoupling")


def plot_multimetric(
    stage1b: dict[str, np.ndarray | float],
    stage1a: dict[str, np.ndarray | float],
    output_dir: Path,
) -> list[Path]:
    dims = stage1b["dims"]
    success = stage1b["success"]
    planning_spearman = stage1b["planning_spearman"]
    endpoint_spearman = stage1a["c2_spearman"]
    action_l2 = stage1b["action_l2"]
    false_elite = stage1b["false_elite"]
    c0_spearman = float(stage1a["c0_spearman"])

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(dims, success, color=PLANNING_COLOR, marker="o", linewidth=2.0, markersize=4.8)
    ax.axvspan(32, 64, color=GRAY, alpha=0.10, linewidth=0)
    format_dim_axis(ax)
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Success")

    ax = axes[0, 1]
    ax.axhline(0.0, color="#b0b0b0", linewidth=0.9, zorder=0)
    ax.plot(
        dims,
        planning_spearman,
        color=PLANNING_COLOR,
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        label="Planning",
    )
    ax.plot(
        dims,
        endpoint_spearman,
        color=ENDPOINT_COLOR,
        marker="s",
        linewidth=2.0,
        markersize=4.8,
        label="Endpoint",
    )
    ax.axhline(c0_spearman, color=ENDPOINT_COLOR, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.65)
    format_dim_axis(ax)
    ax.set_ylabel("Spearman")
    ax.set_ylim(-0.08, 0.56)
    ax.set_title("Ranking")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    ax.plot(dims, action_l2, color=PLANNING_COLOR, marker="o", linewidth=2.0, markersize=4.8)
    format_dim_axis(ax)
    ax.set_ylabel("Blocked action L2")
    ax.set_title("Distance From Default Plan")

    ax = axes[1, 1]
    ax.plot(dims, false_elite, color=PLANNING_COLOR, marker="o", linewidth=2.0, markersize=4.8)
    format_dim_axis(ax)
    ax.set_ylabel("False elite rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.65, 0.71)
    ax.set_title("False Elite Rate")

    return save_figure(fig, output_dir, "fig3_stage1b_multimetric_panel")


def main() -> None:
    args = parse_args()
    configure_style()
    stage1b = load_stage1b_series(args.stage1b_path)
    stage1a = load_stage1a_endpoint_series(args.stage1a_path)

    saved_paths: list[Path] = []
    saved_paths.extend(plot_success(stage1b, args.output_dir))
    saved_paths.extend(plot_decoupling(stage1b, stage1a, args.output_dir))
    saved_paths.extend(plot_multimetric(stage1b, stage1a, args.output_dir))

    print("Saved Stage 1B figures:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
