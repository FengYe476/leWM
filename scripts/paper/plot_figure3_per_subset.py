#!/usr/bin/env python3
"""Build Figure 4: PushT per-subset rank-1 success butterfly chart at m=64.

Data sources and values used:
- Left side, PushT full projected CEM:
  results/phase2/stage1/stage1b_full.json
  aggregate.by_dimension["64"].projected_success_rate.mean = 0.2962962962962963
  aggregate.by_dimension_and_subset["64"]:
  invisible_quadrant 0.041666666666666664, sign_reversal 0.047619047619047616,
  latent_favorable 0.5833333333333334, v1_favorable 0.1282051282051282,
  ordinary 0.8666666666666667
- Right side, PushT re-rank-only:
  results/phase2/protocol_match/pusht_rerank_only.json
  aggregate.by_dimension["64"].rank1_success_rate.mean = 0.35333333333333333
  aggregate.by_dimension_x_subset["64"]:
  invisible_quadrant 0.0, sign_reversal 0.047619047619047616,
  latent_favorable 0.75, v1_favorable 0.02564102564102564,
  ordinary 0.5531914893617021

Cube per-cell breakdowns are intentionally deferred to Appendix G.

Output:
- paper/figures/fig4_butterfly.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "paper" / "figures" / "fig4_butterfly.pdf"

PUSHT_FULL_PATH = ROOT / "results" / "phase2" / "stage1" / "stage1b_full.json"
PUSHT_RERANK_PATH = ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"

M_DIM = "64"
PUSHT_SUBSETS = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)
DISPLAY_SUBSETS = (
    "ordinary",
    "latent_favorable",
    "v1_favorable",
    "sign_reversal",
    "invisible_quadrant",
)

SUBSET_LABELS = {
    "invisible_quadrant": "Invisible quadrant",
    "sign_reversal": "Sign reversal",
    "latent_favorable": "Latent-favorable",
    "v1_favorable": "V1-favorable",
    "ordinary": "Ordinary",
}

SUBSET_COLORS = {
    "invisible_quadrant": "#d62728",
    "sign_reversal": "#ff7f0e",
    "latent_favorable": "#1f77b4",
    "v1_favorable": "#9467bd",
    "ordinary": "#2ca02c",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stat_mean(group: dict[str, Any], *metric_names: str) -> float:
    for metric_name in metric_names:
        value = group.get(metric_name)
        if isinstance(value, dict) and "mean" in value:
            return float(value["mean"])
        if value is not None and not isinstance(value, dict):
            return float(value)
    raise KeyError(f"None of {metric_names} found in group with keys {sorted(group)}")


def short_label(name: str) -> str:
    if name in SUBSET_LABELS:
        return SUBSET_LABELS[name]
    return name.replace("x", "")


def load_pusht_full() -> tuple[list[dict[str, Any]], float]:
    data = load_json(PUSHT_FULL_PATH)
    by_dim = data["aggregate"]["by_dimension"][M_DIM]
    aggregate = stat_mean(by_dim, "rank1_success_rate", "projected_success_rate")
    by_subset = data["aggregate"]["by_dimension_and_subset"][M_DIM]
    rows = []
    for subset in PUSHT_SUBSETS:
        group = by_subset[subset]
        rows.append(
            {
                "protocol": "PushT full projected CEM",
                "subset": subset,
                "label": short_label(subset),
                "m": int(M_DIM),
                "n": int(group["n_records"]),
                "success": stat_mean(group, "rank1_success_rate", "projected_success_rate"),
            }
        )
    return rows, aggregate


def load_pusht_rerank() -> tuple[list[dict[str, Any]], float]:
    data = load_json(PUSHT_RERANK_PATH)
    by_dim = data["aggregate"]["by_dimension"][M_DIM]
    aggregate = stat_mean(by_dim, "rank1_success_rate", "projected_success_rate")
    by_subset = data["aggregate"]["by_dimension_x_subset"][M_DIM]
    rows = []
    for subset in PUSHT_SUBSETS:
        group = by_subset[subset]
        rows.append(
            {
                "protocol": "PushT re-rank-only",
                "subset": subset,
                "label": short_label(subset),
                "m": int(M_DIM),
                "n": int(group["n_records"]),
                "success": stat_mean(group, "rank1_success_rate", "projected_success_rate"),
            }
        )
    return rows, aggregate


def rows_by_subset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["subset"]): row for row in rows}


def darken_hex(hex_color: str, factor: float = 0.62) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{max(0, min(255, int(channel * factor))):02x}" for channel in rgb)


def annotate_value(ax, x: float, y: float, value: float, side: str) -> None:
    offset = 5.0 if value < 5.0 else 3.0
    fontweight = "bold" if value > 50.0 else "normal"
    if side == "left":
        ax.text(
            x - offset,
            y,
            f"{value:.1f}%",
            ha="right",
            va="center",
            fontsize=9,
            fontweight=fontweight,
            color="0.15",
        )
    else:
        ax.text(
            x + offset,
            y,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight=fontweight,
            color="0.15",
        )


def plot_butterfly(
    ax,
    full_rows: list[dict[str, Any]],
    full_aggregate: float,
    rerank_rows: list[dict[str, Any]],
    rerank_aggregate: float,
) -> None:
    full_by_subset = rows_by_subset(full_rows)
    rerank_by_subset = rows_by_subset(rerank_rows)

    row_step = 0.8
    y = np.arange(len(DISPLAY_SUBSETS)) * row_step
    full_values = np.array(
        [float(full_by_subset[subset]["success"]) * 100.0 for subset in DISPLAY_SUBSETS]
    )
    rerank_values = np.array(
        [float(rerank_by_subset[subset]["success"]) * 100.0 for subset in DISPLAY_SUBSETS]
    )
    colors = [SUBSET_COLORS[subset] for subset in DISPLAY_SUBSETS]
    edge_colors = [darken_hex(color) for color in colors]

    ax.barh(
        y,
        -full_values,
        height=0.5,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        label="Full projected CEM",
    )
    ax.barh(
        y,
        rerank_values,
        height=0.5,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        label="Re-rank-only",
    )

    full_aggregate_pct = full_aggregate * 100.0
    rerank_aggregate_pct = rerank_aggregate * 100.0
    aggregate_color = "#666666"
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(
        -full_aggregate_pct,
        color=aggregate_color,
        linestyle="--",
        linewidth=1.1,
        alpha=0.7,
    )
    ax.axvline(
        rerank_aggregate_pct,
        color=aggregate_color,
        linestyle="--",
        linewidth=1.1,
        alpha=0.7,
    )

    ax.text(
        -full_aggregate_pct,
        -0.52,
        f"agg. {full_aggregate_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=aggregate_color,
    )
    ax.text(
        rerank_aggregate_pct,
        -0.52,
        f"agg. {rerank_aggregate_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=aggregate_color,
    )

    for idx, _subset in enumerate(DISPLAY_SUBSETS):
        annotate_value(ax, -full_values[idx], y[idx], full_values[idx], side="left")
        annotate_value(ax, rerank_values[idx], y[idx], rerank_values[idx], side="right")

    ax.set_title(
        r"PushT per-subset rank-1 success ($m = 64$)",
        fontsize=10,
        fontweight="normal",
        pad=14,
    )
    ax.set_xlabel("← Full projected CEM (%)    |    Re-rank-only (%) →")
    ax.set_xlim(-100, 100)
    ax.set_ylim(y[-1] + 0.48, -0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([SUBSET_LABELS[subset] for subset in DISPLAY_SUBSETS])
    for tick_label in ax.get_yticklabels():
        tick_label.set_horizontalalignment("right")
    ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(tick)) for tick in ticks])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def print_summary(panels: list[tuple[str, list[dict[str, Any]], float]]) -> None:
    print("Figure 4 butterfly values used")
    print("protocol\tsubset\tm\tn\tM_rank1")
    for protocol, rows, _aggregate in panels:
        for row in rows:
            print(
                f"{protocol}\t{row['subset']}\t{row['m']}\t{row['n']}\t"
                f"{float(row['success']) * 100.0:.6f}%"
            )
    print("protocol\taggregate_m64")
    for protocol, _rows, aggregate in panels:
        print(f"{protocol}\t{aggregate * 100.0:.6f}%")


def main() -> None:
    pusht_full_rows, pusht_full_aggregate = load_pusht_full()
    pusht_rerank_rows, pusht_rerank_aggregate = load_pusht_rerank()

    panels = [
        ("PushT full projected CEM", pusht_full_rows, pusht_full_aggregate),
        ("PushT re-rank-only", pusht_rerank_rows, pusht_rerank_aggregate),
    ]

    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_butterfly(
        ax,
        pusht_full_rows,
        pusht_full_aggregate,
        pusht_rerank_rows,
        pusht_rerank_aggregate,
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print_summary(panels)


if __name__ == "__main__":
    main()
