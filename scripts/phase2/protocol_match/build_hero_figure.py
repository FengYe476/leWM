#!/usr/bin/env python3
"""Build the Block 3 four-panel protocol-matching hero figure."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAP_TABLE = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "cem_gap_table.json"
DEFAULT_CUBE_STAGE1A = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1a.json"
DEFAULT_CUBE_FULL = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "cube_full_proj_cem.json"
DEFAULT_OUTPUT_PNG = PROJECT_ROOT / "results" / "phase2" / "figures" / "hero_figure.png"
DEFAULT_OUTPUT_PDF = PROJECT_ROOT / "results" / "phase2" / "figures" / "hero_figure.pdf"

DIMENSIONS = (1, 2, 4, 8, 16, 32, 64, 128, 192)
CUBE_FULL_DIMS = (1, 8, 32, 64, 192)
SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)
SUBSET_LABELS = {
    "overall": "overall",
    "invisible_quadrant": "invisible",
    "sign_reversal": "sign-reversal",
    "latent_favorable": "latent-fav.",
    "v1_favorable": "V1-fav.",
    "ordinary": "ordinary",
}
SUBSET_COLORS = {
    "overall": "#222222",
    "invisible_quadrant": "#D55E00",
    "sign_reversal": "#CC79A7",
    "latent_favorable": "#0072B2",
    "v1_favorable": "#009E73",
    "ordinary": "#E69F00",
    "cube": "#4C72B0",
}
SUBSET_MARKERS = {
    "overall": "o",
    "invisible_quadrant": "s",
    "sign_reversal": "^",
    "latent_favorable": "D",
    "v1_favorable": "P",
    "ordinary": "o",
    "cube": "o",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-table", type=Path, default=DEFAULT_GAP_TABLE)
    parser.add_argument("--cube-stage1a", type=Path, default=DEFAULT_CUBE_STAGE1A)
    parser.add_argument("--cube-full", type=Path, default=DEFAULT_CUBE_FULL)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


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
        value = stat_mean(nested_get(group, ("aggregate", "global_spearman")))
        if value is None:
            per_seed = group.get("per_seed", [])
            values = [stat_mean(nested_get(row, ("metrics", "global_spearman"))) for row in per_seed]
            values = [float(item) for item in values if item is not None]
            value = float(np.mean(values)) if values else None
        if value is not None:
            out[int(dim)] = float(value)
    return out


def gap_rows(
    gap: dict[str, Any],
    *,
    environment: str,
    protocol: str,
    subset: str | None = None,
) -> list[dict[str, Any]]:
    container = gap.get("overall", []) if subset is None else gap.get("by_subset", [])
    rows = [
        row
        for row in container
        if row.get("environment") == environment and row.get("protocol") == protocol
    ]
    if subset is not None:
        rows = [row for row in rows if row.get("subset") == subset]
    return sorted(rows, key=lambda row: int(row["dimension"]))


def cube_full_rows(cube_full: dict[str, Any], cube_endpoint: dict[int, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim_str, group in sorted(
        cube_full.get("aggregate", {}).get("by_dimension", {}).items(),
        key=lambda item: int(item[0]),
    ):
        dim = int(dim_str)
        r_pool = stat_mean(group.get("endpoint_spearman"))
        r_endpoint = cube_endpoint[dim]
        rows.append(
            {
                "environment": "Cube",
                "protocol": "full_projected_cem",
                "dimension": dim,
                "R_endpoint": r_endpoint,
                "R_pool": r_pool,
                "Delta_CEM": (
                    clean_float(float(r_endpoint) - float(r_pool)) if r_pool is not None else None
                ),
                "M_rank1": stat_mean(group.get("rank1_success_rate"))
                or stat_mean(group.get("projected_success_rate")),
            }
        )
    return rows


def row_values(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = np.asarray([int(row["dimension"]) for row in rows], dtype=float)
    success = np.asarray([float(row["M_rank1"]) for row in rows], dtype=float)
    delta = np.asarray([float(row["Delta_CEM"]) for row in rows], dtype=float)
    return dims, success, delta


def plot_series(
    ax,
    ax_delta,
    rows: list[dict[str, Any]],
    *,
    label: str,
    color: str,
    marker: str,
    alpha: float,
    linewidth: float,
) -> None:
    dims, success, delta = row_values(rows)
    ax.plot(
        dims,
        success,
        color=color,
        marker=marker,
        markersize=3.8,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    ax_delta.plot(
        dims,
        delta,
        color=color,
        marker=marker,
        markersize=3.2,
        linewidth=max(0.9, linewidth - 0.35),
        linestyle="--",
        alpha=min(0.80, alpha + 0.05),
    )


def setup_axes(ax, ax_delta, *, title: str) -> None:
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, 230)
    ax.set_ylim(-0.02, 1.02)
    ax_delta.set_ylim(-0.02, 0.68)
    ax.set_xticks(DIMENSIONS)
    ax.set_xticklabels([str(dim) for dim in DIMENSIONS], fontsize=7)
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_ha("right")
    ax.tick_params(axis="y", labelsize=7, width=0.6, length=2.5)
    ax.tick_params(axis="x", labelsize=7, width=0.6, length=2.5)
    ax_delta.tick_params(axis="y", labelsize=7, width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    for spine in ax_delta.spines.values():
        spine.set_linewidth(0.6)


def main() -> int:
    args = parse_args()
    args.gap_table = args.gap_table.expanduser().resolve()
    args.cube_stage1a = args.cube_stage1a.expanduser().resolve()
    args.cube_full = args.cube_full.expanduser().resolve()
    args.output_png = args.output_png.expanduser().resolve()
    args.output_pdf = args.output_pdf.expanduser().resolve()

    gap = load_json(args.gap_table)
    cube_stage1a = load_json(args.cube_stage1a)
    cube_full = load_json(args.cube_full)
    cube_endpoint = load_stage1a_endpoint_by_dim(cube_stage1a)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.6), constrained_layout=False)
    panel_specs = [
        ("PushT", "full_projected_cem", "PushT: full projected CEM", axes[0, 0]),
        ("Cube", "full_projected_cem", "Cube: full projected CEM", axes[0, 1]),
        ("PushT", "rerank_only", "PushT: re-rank only", axes[1, 0]),
        ("Cube", "rerank_only", "Cube: re-rank only", axes[1, 1]),
    ]

    delta_axes = []
    for env, protocol, title, ax in panel_specs:
        ax_delta = ax.twinx()
        delta_axes.append(ax_delta)
        setup_axes(ax, ax_delta, title=title)
        if env == "PushT":
            overall = gap_rows(gap, environment=env, protocol=protocol)
            plot_series(
                ax,
                ax_delta,
                overall,
                label="overall",
                color=SUBSET_COLORS["overall"],
                marker=SUBSET_MARKERS["overall"],
                alpha=0.92,
                linewidth=1.65,
            )
            for subset in SUBSET_ORDER:
                rows = gap_rows(gap, environment=env, protocol=protocol, subset=subset)
                plot_series(
                    ax,
                    ax_delta,
                    rows,
                    label=SUBSET_LABELS[subset],
                    color=SUBSET_COLORS[subset],
                    marker=SUBSET_MARKERS[subset],
                    alpha=0.62,
                    linewidth=1.0,
                )
        elif protocol == "full_projected_cem":
            rows = cube_full_rows(cube_full, cube_endpoint)
            plot_series(
                ax,
                ax_delta,
                rows,
                label="Cube overall",
                color=SUBSET_COLORS["cube"],
                marker=SUBSET_MARKERS["cube"],
                alpha=0.94,
                linewidth=1.75,
            )
            ax.set_xticks(CUBE_FULL_DIMS)
            ax.set_xticklabels([str(dim) for dim in CUBE_FULL_DIMS], fontsize=7)
        else:
            rows = gap_rows(gap, environment=env, protocol=protocol)
            plot_series(
                ax,
                ax_delta,
                rows,
                label="Cube overall",
                color=SUBSET_COLORS["cube"],
                marker=SUBSET_MARKERS["cube"],
                alpha=0.94,
                linewidth=1.75,
            )

    axes[0, 0].set_ylabel(r"$M_{\mathrm{rank1}}$", fontsize=8.5)
    axes[1, 0].set_ylabel(r"$M_{\mathrm{rank1}}$", fontsize=8.5)
    axes[1, 0].set_xlabel("projection dimension $m$", fontsize=8.5)
    axes[1, 1].set_xlabel("projection dimension $m$", fontsize=8.5)
    axes[0, 1].text(
        1.19,
        0.50,
        r"$\Delta_{\mathrm{CEM}}$",
        rotation=90,
        transform=axes[0, 1].transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
    )
    axes[1, 1].text(
        1.19,
        0.50,
        r"$\Delta_{\mathrm{CEM}}$",
        rotation=90,
        transform=axes[1, 1].transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
    )

    subset_handles = [
        Line2D(
            [0],
            [0],
            color=SUBSET_COLORS[key],
            marker=SUBSET_MARKERS[key],
            linewidth=1.4,
            markersize=4,
            label=SUBSET_LABELS.get(key, key),
        )
        for key in ("overall", *SUBSET_ORDER, "cube")
    ]
    metric_handles = [
        Line2D([0], [0], color="#444444", linewidth=1.5, linestyle="-", label=r"$M_{\mathrm{rank1}}$"),
        Line2D([0], [0], color="#444444", linewidth=1.2, linestyle="--", label=r"$\Delta_{\mathrm{CEM}}$"),
    ]
    fig.legend(
        handles=subset_handles + metric_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.1,
        columnspacing=1.2,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.suptitle(
        "Endpoint-planning decoupling under matched projection protocols",
        fontsize=10.5,
        y=0.985,
    )
    fig.subplots_adjust(left=0.075, right=0.91, top=0.91, bottom=0.18, wspace=0.34, hspace=0.34)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300)
    fig.savefig(args.output_pdf)
    plt.close(fig)
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
