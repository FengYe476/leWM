#!/usr/bin/env python3
"""Build Figure 5: four negative repair results from Section 6.

Data sources and values used:
- Panel A, learned cost heads:
  results/phase2/p2_0/planning_gap_diagnosis.json
    cost_scale.cem_euclidean[iteration=30].top30_std = 0.35356250405311584
    cost_scale.cem_cpsi[iteration=30].top30_std = 0.001487637055106461
  results/phase2/p2_0/split3_planning_large.json
    aggregate.split3_rescue.n_cpsi_rescues = 0 / 16 hard pairs
  results/phase2/p2_0/cem_aware_planning.json
    aggregate.split3_rescue.n_cpsi_rescues = 1 / 16 hard pairs
- Panel B, encoder replacement:
  results/phase2/stage1/stage1a_c0_c5.json
    controls.C0.metrics.global_spearman = 0.5064137547428476
  results/phase2/track_b/ranking_comparison.json
    DINOv2 mean-pool global_spearman = 0.26088428231965005
    DINOv2 CLS global_spearman = 0.23868904131770977
- Panel C, hybrid CEM oracle budget:
  results/phase2/p2_0/oracle_budget_cem/oracle_budget_cem_summary.json
    K=30: 0/16, K=60: 7/16, K=150: 12/16
- Panel D, Subspace-CEM regression:
  results/phase2/subspace_cem/stage_a_pusht.json
    invisible_quadrant default/subspace = 0.0 / 0.0
    ordinary default/subspace = 0.8333333333333334 / 0.75
    latent_favorable default/subspace = 1.0 / 0.9
    v1_favorable default/subspace = 0.4 / 0.2
    overall default/subspace = 0.5666666666666667 / 0.48333333333333334

Output:
- paper/figures/fig5_unified.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "paper" / "figures" / "fig5_unified.pdf"

PLANNING_GAP_PATH = ROOT / "results" / "phase2" / "p2_0" / "planning_gap_diagnosis.json"
LEARNED_COST_PATH = ROOT / "results" / "phase2" / "p2_0" / "split3_planning_large.json"
CEM_AWARE_PATH = ROOT / "results" / "phase2" / "p2_0" / "cem_aware_planning.json"
STAGE1A_C0_C5_PATH = ROOT / "results" / "phase2" / "stage1" / "stage1a_c0_c5.json"
TRACK_B_PATH = ROOT / "results" / "phase2" / "track_b" / "ranking_comparison.json"
ORACLE_BUDGET_PATH = (
    ROOT / "results" / "phase2" / "p2_0" / "oracle_budget_cem" / "oracle_budget_cem_summary.json"
)
SUBSPACE_STAGE_A_PATH = ROOT / "results" / "phase2" / "subspace_cem" / "stage_a_pusht.json"

SUBSPACE_SUBSETS = (
    "invisible_quadrant",
    "ordinary",
    "latent_favorable",
    "v1_favorable",
)

SUBSET_LABELS = {
    "invisible_quadrant": "Invisible",
    "ordinary": "Ordinary",
    "latent_favorable": "Latent fav.",
    "v1_favorable": "V1 fav.",
    "overall": "Overall",
}

COLORS = {
    "default": "#1f77b4",
    "learned": "#d62728",
    "oracle": "#2ca02c",
    "dinov2": "#ff7f0e",
    "subspace": "#9467bd",
}

SUBSET_COLORS = {
    "invisible_quadrant": "#d62728",
    "sign_reversal": "#ff7f0e",
    "latent_favorable": "#1f77b4",
    "v1_favorable": "#9467bd",
    "ordinary": "#2ca02c",
    "overall": COLORS["subspace"],
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def record_at_iteration(records: list[dict[str, Any]], iteration: int) -> dict[str, Any]:
    for record in records:
        if int(record["iteration"]) == iteration:
            return record
    raise KeyError(f"No record found at iteration {iteration}")


def load_panel_a() -> dict[str, Any]:
    planning_gap = load_json(PLANNING_GAP_PATH)
    cost_scale = planning_gap["cost_scale"]
    euclidean = record_at_iteration(cost_scale["cem_euclidean"], 30)
    cpsi = record_at_iteration(cost_scale["cem_cpsi"], 30)

    learned_cost = load_json(LEARNED_COST_PATH)["aggregate"]["split3_rescue"]
    cem_aware = load_json(CEM_AWARE_PATH)["aggregate"]["split3_rescue"]

    hard_pairs = int(learned_cost["n_hard_pairs_evaluated"])
    if hard_pairs != int(cem_aware["n_hard_pairs_evaluated"]):
        raise ValueError("Learned-cost and CEM-aware hard-pair counts differ")

    return {
        "euclidean_top30_std": float(euclidean["top30_std"]),
        "cpsi_top30_std": float(cpsi["top30_std"]),
        "learned_cost_successes": int(learned_cost["n_cpsi_rescues"]),
        "cem_aware_successes": int(cem_aware["n_cpsi_rescues"]),
        "hard_pairs": hard_pairs,
    }


def load_panel_b() -> dict[str, float]:
    stage1a = load_json(STAGE1A_C0_C5_PATH)
    lewm = float(stage1a["controls"]["C0"]["metrics"]["global_spearman"])

    track_b = load_json(TRACK_B_PATH)
    rows = {row["encoder"]: row for row in track_b["rows"]}
    mean_pool = float(rows["DINOv2 mean-pool"]["global_spearman"])
    cls = float(rows["DINOv2 CLS"]["global_spearman"])

    return {
        "LeWM": lewm,
        "DINOv2 mean": mean_pool,
        "DINOv2 CLS": cls,
    }


def load_panel_c() -> list[dict[str, Any]]:
    data = load_json(ORACLE_BUDGET_PATH)
    rows_by_k = {int(row["k"]): row for row in data["sweep"]["rows"]}
    rows = []
    for k in (30, 60, 150):
        row = rows_by_k[k]
        n_pairs = int(row["n_pairs"])
        rows.append(
            {
                "k": k,
                "budget_pct": k / 300.0 * 100.0,
                "successes": int(row["success_count"]),
                "n_pairs": n_pairs,
            }
        )
    return rows


def load_panel_d() -> list[dict[str, Any]]:
    data = load_json(SUBSPACE_STAGE_A_PATH)
    rows = []
    for subset in SUBSPACE_SUBSETS:
        group = data["summary"]["by_subset"][subset]
        rows.append(
            {
                "subset": subset,
                "label": SUBSET_LABELS[subset],
                "default": float(group["default_success"]),
                "subspace": float(group["subspace_success"]),
            }
        )
    overall = data["summary"]["overall"]
    rows.append(
        {
            "subset": "overall",
            "label": SUBSET_LABELS["overall"],
            "default": float(overall["default_success"]),
            "subspace": float(overall["subspace_success"]),
        }
    )
    return rows


def add_panel_labels(axes: Any) -> None:
    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"], strict=True):
        ax.text(
            -0.05,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
            clip_on=False,
        )


def plot_panel_a(ax, data: dict[str, Any]) -> None:
    euclidean = data["euclidean_top30_std"]
    cpsi = data["cpsi_top30_std"]
    compression = euclidean / cpsi

    ax.set_title("Cost head elite compression", fontsize=11, pad=10)
    ax.text(
        0.5,
        0.93,
        f"{data['learned_cost_successes']}/{data['hard_pairs']} hard-pair success",
        ha="center",
        fontsize=11,
        color=COLORS["learned"],
        fontweight="bold",
    )
    ax.add_patch(
        plt.Circle(
            (0.25, 0.55),
            0.13,
            fc=COLORS["default"],
            ec="black",
            lw=1.5,
            alpha=0.8,
        )
    )
    ax.text(
        0.25,
        0.55,
        f"{euclidean:.3f}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax.add_patch(
        plt.Circle((0.75, 0.55), 0.015, fc=COLORS["learned"], ec="black", lw=1.5)
    )
    ax.text(0.75, 0.42, f"{cpsi:.4f}", ha="center", va="top", fontsize=9)
    ax.annotate(
        "",
        xy=(0.60, 0.55),
        xytext=(0.40, 0.55),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "gray"},
    )
    ax.text(
        0.5,
        0.68,
        rf"{compression:.0f}$\times$ compression",
        ha="center",
        va="center",
        fontsize=9,
        color="0.25",
    )
    ax.text(0.25, 0.10, "Euclidean", ha="center", fontsize=10)
    ax.text(0.75, 0.10, r"Learned $C_\psi$", ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_panel_b(ax, data: dict[str, float]) -> None:
    models = ["DINOv2 CLS", "DINOv2 mean", "LeWM"]
    values = [data["DINOv2 CLS"], data["DINOv2 mean"], data["LeWM"]]
    colors = [COLORS["dinov2"], COLORS["dinov2"], COLORS["default"]]
    y_pos = list(range(len(models)))

    ax.set_title("Encoder replacement", fontsize=11, pad=10)
    ax.hlines(y=y_pos, xmin=0, xmax=values, colors=colors, linewidth=3, alpha=0.7)
    ax.scatter(
        values,
        y_pos,
        s=120,
        c=colors,
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
    )
    for i, value in enumerate(values):
        ax.text(value + 0.02, i, f"{value:.3f}", va="center", fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Spearman correlation", fontsize=10)
    ax.set_xlim(0, 0.6)
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel_c(ax, rows: list[dict[str, Any]]) -> None:
    budgets = [row["budget_pct"] for row in rows]
    successes = [row["successes"] for row in rows]

    ax.set_title("Hybrid CEM oracle budget", fontsize=11, pad=10)
    ax.fill_between(budgets, successes, alpha=0.3, color=COLORS["oracle"])
    ax.plot(
        budgets,
        successes,
        "o-",
        color=COLORS["oracle"],
        linewidth=2.5,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

    for budget, success in zip(budgets, successes):
        ax.text(
            budget,
            success + 0.8,
            f"{success}/16",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(16, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.text(50, 16.5, "16/16 max", fontsize=9, color="gray", ha="right")
    ax.set_xlabel("Oracle budget (%)", fontsize=10)
    ax.set_ylabel("Successes / 16 hard pairs", fontsize=10)
    ax.set_xlim(7, 53)
    ax.set_ylim(-0.5, 18)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{budget:.0f}%" for budget in budgets])
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel_d(ax, rows: list[dict[str, Any]]) -> None:
    subset_colors = {
        "invisible_quadrant": "#d62728",
        "ordinary": "#2ca02c",
        "latent_favorable": "#1f77b4",
        "v1_favorable": "#9467bd",
        "overall": "#666666",
    }

    ax.set_title("Subspace-CEM regression", fontsize=11, pad=10)
    right_labels = []
    for row in rows:
        default = row["default"] * 100.0
        subspace = row["subspace"] * 100.0
        color = subset_colors[row["subset"]]
        ax.plot([0, 1], [default, subspace], "-", color=color, linewidth=2, alpha=0.7)
        ax.scatter(
            0,
            default,
            s=100,
            c=color,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.scatter(
            1,
            subspace,
            s=100,
            c=color,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
            marker="s",
        )
        right_labels.append((subspace, row["label"], color))

    prev_y = None
    min_gap = 16.0
    for actual_y, name, color in sorted(right_labels, reverse=True):
        label_y = actual_y
        if prev_y is not None and prev_y - label_y < min_gap:
            label_y = prev_y - min_gap
        ax.text(
            1.05,
            label_y,
            f"{name}: {actual_y:.0f}%",
            va="center",
            fontsize=8,
            color=color,
        )
        prev_y = label_y

    ax.set_xlim(-0.3, 1.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Default CEM", "Subspace-CEM"], fontsize=10)
    ax.set_ylabel("Success rate (%)", fontsize=10)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def print_summary(
    panel_a: dict[str, Any],
    panel_b: dict[str, float],
    panel_c: list[dict[str, Any]],
    panel_d: list[dict[str, Any]],
) -> None:
    print("Figure 5 values used")
    print("Panel A: learned cost heads")
    print(f"  Euclidean top-30 std: {panel_a['euclidean_top30_std']:.12f}")
    print(f"  Learned C_psi top-30 std: {panel_a['cpsi_top30_std']:.12f}")
    print(
        "  Learned-cost CEM success: "
        f"{panel_a['learned_cost_successes']}/{panel_a['hard_pairs']}"
    )
    print(
        "  CEM-aware distillation success: "
        f"{panel_a['cem_aware_successes']}/{panel_a['hard_pairs']}"
    )
    print("Panel B: encoder replacement")
    for label, value in panel_b.items():
        print(f"  {label}: {value:.12f}")
    print("Panel C: hybrid CEM oracle budget")
    for row in panel_c:
        print(f"  K={row['k']} ({row['budget_pct']:.0f}%): {row['successes']}/{row['n_pairs']}")
    print("Panel D: Subspace-CEM regression")
    for row in panel_d:
        print(
            f"  {row['subset']}: default={row['default'] * 100.0:.6f}%, "
            f"subspace={row['subspace'] * 100.0:.6f}%"
        )


def main() -> None:
    panel_a = load_panel_a()
    panel_b = load_panel_b()
    panel_c = load_panel_c()
    panel_d = load_panel_d()

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

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        gridspec_kw={"hspace": 0.4, "wspace": 0.35},
    )
    plot_panel_a(axes[0, 0], panel_a)
    plot_panel_b(axes[0, 1], panel_b)
    plot_panel_c(axes[1, 0], panel_c)
    plot_panel_d(axes[1, 1], panel_d)
    add_panel_labels(axes)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.93)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print_summary(panel_a, panel_b, panel_c, panel_d)


if __name__ == "__main__":
    main()
