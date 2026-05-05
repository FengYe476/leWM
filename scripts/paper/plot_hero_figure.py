#!/usr/bin/env python3
"""Build the enhanced Section 4 endpoint-pool diagnostics figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURE_PATH = ROOT / "paper" / "figures" / "fig3_enhanced.pdf"

GAP_PATH = ROOT / "results" / "phase2" / "protocol_match" / "cem_gap_table.json"
TAXONOMY_PATH = ROOT / "results" / "phase2" / "protocol_match" / "taxonomy_table.json"
PUSHT_RERANK_PATH = ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
PUSHT_FULL_PATH = ROOT / "results" / "phase2" / "stage1" / "stage1b_full.json"
CUBE_FULL_PATH = ROOT / "results" / "revision" / "cube_full_proj_cem_extended.json"
CUBE_RERANK_PATH = ROOT / "results" / "phase2" / "cube" / "cube_stage1b.json"

COMMON_DIMS = [1, 8, 32, 64, 192]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_protocol(protocol: str) -> str:
    return protocol.replace("-", "_").lower()


def rows_from_consolidated(
    rows: list[dict],
    environment: str,
    protocol: str,
    dims: list[int],
) -> list[dict] | None:
    protocol_norm = normalize_protocol(protocol)
    matches = {}
    for row in rows:
        if row.get("environment") != environment:
            continue
        if normalize_protocol(str(row.get("protocol", ""))) != protocol_norm:
            continue
        dim = row.get("dimension", row.get("m"))
        if dim is None:
            continue
        dim = int(dim)
        if dim in dims:
            matches[dim] = row

    required = {"R_endpoint", "R_pool", "Delta_CEM", "M_rank1"}
    if not all(dim in matches for dim in dims):
        return None
    if not all(required.issubset(matches[dim]) for dim in dims):
        return None
    return [matches[dim] for dim in dims]


def stat_mean(group: dict, *metric_names: str) -> float:
    for metric_name in metric_names:
        value = group.get(metric_name)
        if isinstance(value, dict) and "mean" in value:
            return float(value["mean"])
        if value is not None and not isinstance(value, dict):
            return float(value)
    raise KeyError(f"None of {metric_names} found in group with keys {sorted(group)}")


def rows_from_cube_extended_splice(
    gap_rows: list[dict],
    cube_full: dict,
    dims: list[int],
) -> list[dict] | None:
    endpoint_rows = rows_from_consolidated(
        gap_rows,
        environment="Cube",
        protocol="rerank_only",
        dims=dims,
    )
    if endpoint_rows is None:
        return None

    by_dim = cube_full.get("aggregate", {}).get("by_dimension", {})
    output = []
    for endpoint_row in endpoint_rows:
        dim = int(endpoint_row["dimension"])
        stats = by_dim.get(str(dim))
        if stats is None:
            return None
        # R_endpoint intentionally comes from the Stage 1A endpoint reference
        # carried by the Cube re-rank-only rows. The extended Cube full-CEM
        # artifact supplies only planning-side R_pool and M_rank1 here.
        r_endpoint = float(endpoint_row["R_endpoint"])
        r_pool = stat_mean(stats, "Rpool", "Rpool_projected")
        m_rank1 = stat_mean(stats, "planning_success_rate", "rank1_success_rate")
        output.append(
            {
                "environment": "Cube",
                "protocol": "full_projected_cem",
                "dimension": dim,
                "R_endpoint": r_endpoint,
                "R_pool": r_pool,
                "Delta_CEM": r_endpoint - r_pool,
                "M_rank1": m_rank1,
            }
        )
    return output


def get_panel_rows(
    label: str,
    environment: str,
    protocol: str,
    gap_table: dict,
    taxonomy_table: dict,
    cube_full: dict,
) -> tuple[list[dict], str]:
    gap_rows = gap_table.get("overall", [])
    taxonomy_rows = taxonomy_table.get("overall", [])

    if environment == "Cube" and protocol == "full_projected_cem":
        rows = rows_from_cube_extended_splice(gap_rows, cube_full, COMMON_DIMS)
        if rows is not None:
            return rows, "cube_full_proj_cem_extended.json + Stage 1A endpoint rows"

    rows = rows_from_consolidated(gap_rows, environment, protocol, COMMON_DIMS)
    if rows is not None:
        return rows, "cem_gap_table.json"

    rows = rows_from_consolidated(taxonomy_rows, environment, protocol, COMMON_DIMS)
    if rows is not None:
        return rows, "taxonomy_table.json"

    raise RuntimeError(f"Could not assemble panel data for {label}")


def plot_panel(ax, rows: list[dict], title: str, panel_label: str, show_left_label: bool):
    dims = [int(row["dimension"]) for row in rows]
    x = np.arange(len(dims))
    r_endpoint = np.array([float(row["R_endpoint"]) for row in rows])
    r_pool = np.array([float(row["R_pool"]) for row in rows])
    success = np.array([float(row["M_rank1"]) * 100.0 for row in rows])

    fill = ax.fill_between(
        x,
        r_pool,
        r_endpoint,
        color="#ff7f0e",
        alpha=0.35,
        label=r"$\Delta_{\mathrm{CEM}}$",
    )
    line_endpoint, = ax.plot(
        x,
        r_endpoint,
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        label=r"$R_{\mathrm{endpoint}}$",
    )
    line_pool, = ax.plot(
        x,
        r_pool,
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label=r"$R_{\mathrm{pool}}$",
    )

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    bars = ax2.bar(
        x,
        success,
        width=0.48,
        color="#2ca02c",
        alpha=0.6,
        edgecolor="#2ca02c",
        linewidth=0.8,
        label=r"$M_{\mathrm{rank1}}$",
        zorder=0,
    )
    if 192 in dims:
        idx = dims.index(192)
        delta = r_endpoint[idx] - r_pool[idx]
        y = min(0.64, max(r_endpoint[idx], r_pool[idx]) + 0.035)
        ax.annotate(
            rf"$\Delta={delta:.3f}$",
            xy=(x[idx], y),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#7f3b08",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )

    ax.set_title(f"{panel_label} {title}", loc="left", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(dim) for dim in dims])
    ax.set_xlabel("projection dimension m")
    ax.set_ylim(-0.05, 0.68)
    ax2.set_ylim(0, 100)
    if show_left_label:
        ax.set_ylabel("Spearman rank correlation")
    ax2.set_ylabel("Rank-1 success (%)")

    ax.tick_params(axis="both", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(False)

    return [line_endpoint, line_pool, fill, bars]


def main() -> None:
    gap_table = load_json(GAP_PATH)
    taxonomy_table = load_json(TAXONOMY_PATH)

    # These raw files are read explicitly so the figure script validates every
    # experiment artifact named in the paper-generation plan.
    _pusht_rerank = load_json(PUSHT_RERANK_PATH)
    _pusht_full = load_json(PUSHT_FULL_PATH)
    cube_full = load_json(CUBE_FULL_PATH)
    _cube_rerank = load_json(CUBE_RERANK_PATH)

    panels = [
        ("PushT full projected CEM", "PushT", "full_projected_cem", "(a)"),
        ("PushT re-rank-only", "PushT", "rerank_only", "(b)"),
        ("Cube full projected CEM", "Cube", "full_projected_cem", "(c)"),
        ("Cube re-rank-only", "Cube", "rerank_only", "(d)"),
    ]

    panel_data = []
    for label, environment, protocol, panel_label in panels:
        rows, source = get_panel_rows(
            label,
            environment,
            protocol,
            gap_table,
            taxonomy_table,
            cube_full,
        )
        print(f"Panel {panel_label} {label}: {source}")
        panel_data.append((label, panel_label, rows, source))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.8), sharex=False, sharey=True)
    legend_handles = None
    for idx, (ax, (label, panel_label, rows, _source)) in enumerate(zip(axes.flat, panel_data)):
        handles = plot_panel(ax, rows, label, panel_label, show_left_label=idx in (0, 2))
        if legend_handles is None:
            legend_handles = handles

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    print(f"Saved {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
