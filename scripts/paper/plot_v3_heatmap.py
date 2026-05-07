#!/usr/bin/env python3
"""Build the V3 same-pool ranker decomposition heatmap for Section 4.4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V3_PATH = ROOT / "results" / "revision" / "v3_pool_analysis_pusht.json"
PHASE_B_PATH = ROOT / "results" / "revision" / "rpool_v1_pusht.json"
OUTPUT_PATH = ROOT / "paper" / "figures" / "fig_v3_heatmap.pdf"

SUBSETS = [
    ("invisible_quadrant", "Invisible quadrant"),
    ("sign_reversal", "Sign reversal"),
    ("latent_favorable", "Latent-favorable"),
    ("v1_favorable", "V1-favorable"),
    ("ordinary", "Ordinary"),
    ("overall", "Overall"),
]

RANKERS = [
    "$C_{\\mathrm{model}}$\n(predicted)",
    "$C_{V3}$\n(actual terminal)",
    "$C_{V1}$\n(oracle)",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def subset_rows(rows: list[dict[str, Any]], subset_key: str) -> list[dict[str, Any]]:
    if subset_key == "overall":
        return rows
    return [row for row in rows if subset_key in row.get("subsets", [])]


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise ValueError(f"No values found for {key}")
    return float(np.mean(values))


def build_matrix() -> tuple[np.ndarray, list[int]]:
    v3_data = load_json(V3_PATH)
    phase_b_data = load_json(PHASE_B_PATH)

    v3_by_pair = {int(row["pair_id"]): row for row in v3_data["per_pair"]}
    phase_b_rows = sorted(phase_b_data["per_pair"], key=lambda row: int(row["pair_id"]))

    missing = sorted({int(row["pair_id"]) for row in phase_b_rows} - set(v3_by_pair))
    if missing:
        raise ValueError(f"V3 rows missing pair IDs: {missing}")

    matrix_rows: list[list[float]] = []
    counts: list[int] = []
    for subset_key, _label in SUBSETS:
        pb_rows = subset_rows(phase_b_rows, subset_key)
        if not pb_rows:
            raise ValueError(f"No rows found for subset {subset_key}")
        v3_rows = [v3_by_pair[int(row["pair_id"])] for row in pb_rows]
        matrix_rows.append(
            [
                mean_value(pb_rows, "Rpool_Cmodel_effective"),
                mean_value(v3_rows, "Rpool_V3_effective"),
                mean_value(pb_rows, "Rpool_V1_effective"),
            ]
        )
        counts.append(len(pb_rows))

    return np.asarray(matrix_rows, dtype=float), counts


def plot_heatmap(data_matrix: np.ndarray, counts: list[int]) -> None:
    subset_labels = [
        label if key == "overall" else f"{label} (n={count})"
        for (key, label), count in zip(SUBSETS, counts, strict=True)
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    image = ax.imshow(data_matrix, cmap="YlOrBr", vmin=-0.2, vmax=1.0, aspect="auto")

    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            value = data_matrix[i, j]
            color = "white" if value > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
            if value < 0:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#d62728",
                        linewidth=2,
                    )
                )

    ax.set_xticks(range(len(RANKERS)))
    ax.set_xticklabels(RANKERS, fontsize=9)
    ax.set_yticks(range(len(subset_labels)))
    ax.set_yticklabels(subset_labels, fontsize=10)
    ax.tick_params(axis="x", pad=6)
    ax.axhline(4.5, color="white", linewidth=2)
    ax.set_title("Pool-level ranking by cost ranker and subset", fontsize=11, pad=10)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
    colorbar.set_label("$R_{\\mathrm{pool}}$", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)


def print_values(data_matrix: np.ndarray, counts: list[int]) -> None:
    print("V3 heatmap values")
    print("Subset | n | C_model | C_V3 | C_V1")
    for (_key, label), count, row in zip(SUBSETS, counts, data_matrix, strict=True):
        print(f"{label} | {count} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f}")


def main() -> None:
    data_matrix, counts = build_matrix()
    plot_heatmap(data_matrix, counts)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print_values(data_matrix, counts)


if __name__ == "__main__":
    main()
