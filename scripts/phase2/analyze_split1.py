#!/usr/bin/env python3
"""Compare Euclidean baseline vs trained C_psi on Split 1 test set."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.splits import split1_random_holdout  # noqa: E402

ARTIFACT_PATH = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
OUTPUT_PATH = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "split1_comparison.json"
METRIC_PATHS = {
    "Small C_psi": PROJECT_ROOT
    / "results"
    / "phase2"
    / "p2_0"
    / "split1_small"
    / "split1_random_70_15_15_small_seed0_test_metrics.json",
    "Large C_psi": PROJECT_ROOT
    / "results"
    / "phase2"
    / "p2_0"
    / "split1_large"
    / "split1_random_70_15_15_large_seed0_test_metrics.json",
}


def clean_float(value) -> float | None:
    """Convert scipy/numpy numbers to JSON-safe floats."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def pairwise_accuracy(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> float:
    """Return per-record pairwise ranking accuracy within each pair."""
    n_correct = 0
    n_total = 0
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        pair_costs = costs[mask]
        pair_labels = labels[mask]
        for i in range(len(pair_costs)):
            for j in range(i + 1, len(pair_costs)):
                if pair_labels[i] == pair_labels[j]:
                    continue
                n_total += 1
                if (pair_costs[i] < pair_costs[j]) == (pair_labels[i] < pair_labels[j]):
                    n_correct += 1
    return n_correct / max(n_total, 1)


def per_pair_spearman(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> dict:
    """Return per-pair Spearman summary."""
    values = []
    by_pair = {}
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        rho, _ = spearmanr(costs[mask], labels[mask])
        rho = clean_float(rho)
        by_pair[int(pair_id)] = {
            "spearman": rho,
            "n_records": int(mask.sum()),
        }
        if rho is not None:
            values.append(rho)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else None,
        "std": float(arr.std()) if len(arr) else None,
        "by_pair": by_pair,
    }


def load_model_metrics() -> list[dict]:
    """Load trained model test metrics from the standard Split 1 output paths."""
    rows = []
    for label, path in METRIC_PATHS.items():
        data = json.loads(path.read_text())
        rows.append(
            {
                "model": label,
                "global_spearman": data["test_spearman_C_psi_vs_C_v1"],
                "pairwise_accuracy": data["test_pairwise_accuracy"],
                "per_pair_spearman_mean": data["test_per_pair_spearman_mean"],
                "per_pair_spearman_std": data["test_per_pair_spearman_std"],
                "best_epoch": data["best_epoch"],
                "metrics_path": str(path),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    """Print a simple aligned comparison table."""
    headers = [
        "Model",
        "Global Spearman",
        "Pairwise Acc",
        "Per-Pair Spearman Mean",
        "Per-Pair Spearman Std",
        "Best Epoch",
    ]
    table = []
    for row in rows:
        table.append(
            [
                row["model"],
                f"{row['global_spearman']:.4f}",
                f"{row['pairwise_accuracy']:.4f}",
                f"{row['per_pair_spearman_mean']:.4f}",
                f"{row['per_pair_spearman_std']:.4f}",
                "-" if row.get("best_epoch") is None else str(row["best_epoch"]),
            ]
        )
    widths = [
        max(len(headers[idx]), *(len(record[idx]) for record in table))
        for idx in range(len(headers))
    ]
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[idx].ljust(widths[idx]) for idx in range(len(headers))))


def main() -> int:
    """Compute and print Split 1 comparison metrics."""
    artifact = torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=False)
    z_t = artifact["z_terminal"].numpy()
    z_g = artifact["z_goal"].numpy()
    v1 = artifact["v1_cost"].numpy()
    pair_ids = artifact["pair_id"].numpy()

    split = split1_random_holdout()
    test_ids = set(split["test_pair_ids"])
    mask = np.asarray([int(pair_id) in test_ids for pair_id in pair_ids], dtype=bool)
    z_t_test = z_t[mask]
    z_g_test = z_g[mask]
    v1_test = v1[mask]
    pair_ids_test = pair_ids[mask]

    c_euc = np.sum((z_t_test - z_g_test) ** 2, axis=1)
    rho_euc, _ = spearmanr(c_euc, v1_test)
    pair_summary = per_pair_spearman(c_euc, v1_test, pair_ids_test)
    rows = [
        {
            "model": "Euclidean",
            "global_spearman": clean_float(rho_euc),
            "pairwise_accuracy": pairwise_accuracy(c_euc, v1_test, pair_ids_test),
            "per_pair_spearman_mean": pair_summary["mean"],
            "per_pair_spearman_std": pair_summary["std"],
            "best_epoch": None,
        },
        *load_model_metrics(),
    ]

    print(f"Split 1 test pairs: {sorted(test_ids)}")
    print(f"Split 1 test records: {int(mask.sum())}")
    print_table(rows)

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "test_pair_ids": sorted(test_ids),
                "test_records": int(mask.sum()),
                "rows": rows,
                "euclidean_per_pair_spearman": pair_summary["by_pair"],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
