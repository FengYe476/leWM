#!/usr/bin/env python3
"""Build Figure 2 motivation panel from verified PushT result artifacts.

Data sources and values used:
- Panel A, PushT long-horizon degradation:
  - results/pusht_sweep_offset25.json: offset 25, success_rate 96.0
  - results/pusht_sweep_offset50.json: offset 50, success_rate 57.99999999999999
  - results/pusht_sweep_offset75.json: offset 75, success_rate 16.0
  - results/pusht_sweep_offset100.json: offset 100, success_rate 10.0
- Panel B, per-pair three-cost attribution at offset 50; dots show Table 11
  mean Spearman and error bars show +/-1 SEM over n=30 pairs:
  - results/three_cost_analysis.json,
    correlations.per_pair_summary.c_model_vs_c_real_z.spearman.mean/std/n:
    0.7789544538844492 / 0.16591767980802766 / 30
  - results/three_cost_analysis.json,
    correlations.per_pair_summary.c_real_z_vs_c_real_state.spearman.mean/std/n:
    0.3529711613920646 / 0.48648417517758613 / 30
  - results/three_cost_analysis.json,
    correlations.per_pair_summary.c_model_vs_c_real_state.spearman.mean/std/n:
    0.3319658869270196 / 0.4548996720632321 / 30

Output:
- paper/figures/fig2_enhanced.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "paper" / "figures" / "fig2_enhanced.pdf"

OFFSET_PATHS = {
    25: ROOT / "results" / "pusht_sweep_offset25.json",
    50: ROOT / "results" / "pusht_sweep_offset50.json",
    75: ROOT / "results" / "pusht_sweep_offset75.json",
    100: ROOT / "results" / "pusht_sweep_offset100.json",
}
THREE_COST_PATH = ROOT / "results" / "three_cost_analysis.json"
PANEL_B_TABLE_MEANS = [0.779, 0.353, 0.332]
PANEL_B_TABLE_SEMS = [0.030, 0.092, 0.085]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_offset_success() -> tuple[list[int], list[float]]:
    offsets = []
    success_rates = []
    for expected_offset, path in OFFSET_PATHS.items():
        data = load_json(path)
        actual_offset = int(data["goal_offset_steps"])
        if actual_offset != expected_offset:
            raise ValueError(f"{path} has offset {actual_offset}, expected {expected_offset}")
        offsets.append(actual_offset)
        success_rates.append(float(data["success_rate"]))
    return offsets, success_rates


def load_three_cost_spearman() -> tuple[list[float], list[float]]:
    data = load_json(THREE_COST_PATH)
    if int(data["offset"]) != 50:
        raise ValueError(f"{THREE_COST_PATH} has offset {data['offset']}, expected 50")

    per_pair = data["correlations"]["per_pair_summary"]
    n_pairs = int(per_pair["c_model_vs_c_real_z"]["spearman"]["n"])
    if n_pairs != int(data["counts"]["pairs"]):
        raise ValueError(
            f"per-pair summary n={n_pairs} does not match counts.pairs={data['counts']['pairs']}"
        )
    if n_pairs != 30:
        raise ValueError(f"Expected 30 pairs for Figure 2 Panel B, found {n_pairs}")

    keys = [
        "c_model_vs_c_real_z",
        "c_real_z_vs_c_real_state",
        "c_model_vs_c_real_state",
    ]
    artifact_means = [float(per_pair[key]["spearman"]["mean"]) for key in keys]
    for artifact_mean, table_mean in zip(artifact_means, PANEL_B_TABLE_MEANS):
        if abs(artifact_mean - table_mean) > 0.002:
            raise ValueError(
                f"{THREE_COST_PATH} mean {artifact_mean:.6f} does not match "
                f"Table 11 value {table_mean:.6f}"
            )
    return PANEL_B_TABLE_MEANS, PANEL_B_TABLE_SEMS


def annotate_bars(ax, bars, labels: list[str], dy: float) -> None:
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    offsets, success_rates = load_offset_success()
    correlations, correlation_sems = load_three_cost_spearman()

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    severity_colors = ["#9ecae1", "#fdbf6f", "#ef6548", "#99000d"]
    bars1 = ax1.bar([str(offset) for offset in offsets], success_rates, color=severity_colors)
    ax1.set_title("(a) PushT success vs. goal offset")
    ax1.set_xlabel("Goal offset (steps)")
    ax1.set_ylabel("Success rate (%)")
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    annotate_bars(ax1, bars1, [f"{rate:.0f}%" for rate in success_rates], dy=2.0)
    ax1.grid(axis="y", color="0.9", linewidth=0.8)
    ax1.set_axisbelow(True)

    attribution_labels = [
        "Predictor\nfidelity",
        "Encoder-physics\nalignment",
        "End-to-end\nalignment",
    ]
    attribution_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    x2 = list(range(len(attribution_labels)))
    ax2.errorbar(
        x2,
        correlations,
        yerr=correlation_sems,
        fmt="none",
        ecolor="0.25",
        elinewidth=1.1,
        capsize=4,
        capthick=1.1,
        zorder=2,
    )
    ax2.scatter(
        x2,
        correlations,
        s=82,
        color=attribution_colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax2.axhline(0.5, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
    ax2.set_title("(b) Per-pair three-cost attribution (offset 50, ±SEM)")
    ax2.set_ylabel("Spearman correlation")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(attribution_labels)
    for x, val, sem in zip(x2, correlations, correlation_sems):
        y_top = val + sem + 0.035
        ax2.text(x, y_top, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    bracket_y = 0.91
    bracket_drop = 0.035
    ax2.plot(
        [x2[0], x2[0], x2[1], x2[1]],
        [bracket_y - bracket_drop, bracket_y, bracket_y, bracket_y - bracket_drop],
        color="0.65",
        linewidth=1.1,
    )
    ax2.annotate(
        "gap",
        xy=((x2[0] + x2[1]) / 2, bracket_y),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax2.grid(axis="y", color="0.9", linewidth=0.8)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print("Panel A values:", dict(zip(offsets, success_rates)))
    print("Panel B Spearman values:", dict(zip(attribution_labels, correlations)))
    print("Panel B Spearman SEMs:", dict(zip(attribution_labels, correlation_sems)))


if __name__ == "__main__":
    main()
