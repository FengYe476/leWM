#!/usr/bin/env python3
"""Unified P2-0 metric comparison across Split 1, Split 2, and Split 3."""

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

from scripts.phase2.cost_head_model import make_cost_head  # noqa: E402
from scripts.phase2.splits import (  # noqa: E402
    split1_random_holdout,
    split2_leave_one_cell_out,
    split3_hard_pair_holdout,
)


ARTIFACT_PATH = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
OUTPUT_PATH = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "all_splits_summary.json"
BASE_DIR = PROJECT_ROOT / "results" / "phase2" / "p2_0"
SPLIT1_METRICS = {
    "small": BASE_DIR
    / "split1_small"
    / "split1_random_70_15_15_small_seed0_test_metrics.json",
    "large": BASE_DIR
    / "split1_large"
    / "split1_random_70_15_15_large_seed0_test_metrics.json",
}
SPLIT3_METRICS = {
    "small": BASE_DIR
    / "split3_small"
    / "split3_all_fail_strong_rho_small_seed0_test_metrics.json",
    "large": BASE_DIR
    / "split3_large"
    / "split3_all_fail_strong_rho_large_seed0_test_metrics.json",
}


def clean_float(value) -> float | None:
    """Return a JSON-safe float or ``None``."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def fmt(value: float | None) -> str:
    """Format metric values for tables."""
    return "None" if value is None else f"{value:.4f}"


def load_json(path: Path) -> dict:
    """Load JSON from ``path``."""
    return json.loads(path.read_text())


def load_artifact() -> dict:
    """Load latent artifact and return numpy arrays."""
    artifact = torch.load(ARTIFACT_PATH, map_location="cpu", weights_only=False)
    return {
        "z_terminal": artifact["z_terminal"].numpy(),
        "z_goal": artifact["z_goal"].numpy(),
        "v1_cost": artifact["v1_cost"].numpy(),
        "pair_id": artifact["pair_id"].numpy(),
        "cell": np.asarray(artifact["cell"]),
    }


def pairwise_accuracy(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> float:
    """Compute within-pair pairwise ranking accuracy, skipping label ties."""
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


def per_pair_pairwise_accuracy(
    costs: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
) -> dict[int, float]:
    """Return pairwise accuracy for each pair ID."""
    out = {}
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        out[int(pair_id)] = pairwise_accuracy(costs[mask], labels[mask], pair_ids[mask])
    return out


def per_pair_spearman(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> dict:
    """Return per-pair Spearman details and summary statistics."""
    by_pair = {}
    values = []
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        rho, _ = spearmanr(costs[mask], labels[mask])
        rho = clean_float(rho)
        by_pair[int(pair_id)] = {"spearman": rho, "n_records": int(mask.sum())}
        if rho is not None:
            values.append(rho)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else None,
        "std": float(arr.std()) if len(arr) else None,
        "by_pair": by_pair,
    }


def subset(data: dict, pair_ids: list[int]) -> dict:
    """Filter artifact arrays to a pair-ID subset."""
    allowed = set(pair_ids)
    mask = np.asarray([int(pair_id) in allowed for pair_id in data["pair_id"]], dtype=bool)
    return {key: value[mask] for key, value in data.items()}


def euclidean_costs(data: dict) -> np.ndarray:
    """Return squared Euclidean latent distance."""
    return np.sum((data["z_terminal"] - data["z_goal"]) ** 2, axis=1)


def metrics_from_costs(costs: np.ndarray, data: dict) -> dict:
    """Compute global and per-pair metrics for scalar record costs."""
    rho, _ = spearmanr(costs, data["v1_cost"])
    per_pair = per_pair_spearman(costs, data["v1_cost"], data["pair_id"])
    return {
        "global_spearman": clean_float(rho),
        "pairwise_accuracy": pairwise_accuracy(costs, data["v1_cost"], data["pair_id"]),
        "per_pair_spearman_mean": per_pair["mean"],
        "per_pair_spearman_std": per_pair["std"],
        "per_pair_spearman": per_pair["by_pair"],
    }


def euclidean_metrics(data: dict, pair_ids: list[int]) -> dict:
    """Compute Euclidean baseline metrics for a split/fold."""
    split_data = subset(data, pair_ids)
    return metrics_from_costs(euclidean_costs(split_data), split_data)


def model_metrics_from_json(path: Path) -> dict:
    """Load trained model metrics from ``train_cost_head.py`` output."""
    raw = load_json(path)
    return {
        "global_spearman": raw["test_spearman_C_psi_vs_C_v1"],
        "pairwise_accuracy": raw["test_pairwise_accuracy"],
        "per_pair_spearman_mean": raw["test_per_pair_spearman_mean"],
        "per_pair_spearman_std": raw["test_per_pair_spearman_std"],
        "best_epoch": raw["best_epoch"],
        "metrics_path": str(path),
        "checkpoint_path": raw["checkpoint_path"],
    }


def with_delta(model_metrics: dict, baseline: dict) -> dict:
    """Add C_psi-minus-Euclidean deltas to trained model metrics."""
    return {
        **model_metrics,
        "delta_global_spearman": (
            model_metrics["global_spearman"] - baseline["global_spearman"]
            if model_metrics["global_spearman"] is not None
            and baseline["global_spearman"] is not None
            else None
        ),
        "delta_pairwise_accuracy": (
            model_metrics["pairwise_accuracy"] - baseline["pairwise_accuracy"]
        ),
        "delta_per_pair_spearman_mean": (
            model_metrics["per_pair_spearman_mean"] - baseline["per_pair_spearman_mean"]
            if model_metrics["per_pair_spearman_mean"] is not None
            and baseline["per_pair_spearman_mean"] is not None
            else None
        ),
    }


def predict_checkpoint_costs(
    *,
    checkpoint_path: Path,
    variant: str,
    data: dict,
    batch_size: int = 512,
) -> np.ndarray:
    """Predict C_psi costs for artifact records using a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_cost_head(variant)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preds = []
    with torch.inference_mode():
        for start in range(0, len(data["pair_id"]), batch_size):
            z = torch.as_tensor(data["z_terminal"][start : start + batch_size], dtype=torch.float32)
            z_g = torch.as_tensor(data["z_goal"][start : start + batch_size], dtype=torch.float32)
            preds.append(model(z, z_g).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def summarize_split(
    *,
    data: dict,
    pair_ids: list[int],
    metric_paths: dict[str, Path],
) -> dict:
    """Return Euclidean and trained model summary for one split."""
    baseline = euclidean_metrics(data, pair_ids)
    models = {
        variant: with_delta(model_metrics_from_json(path), baseline)
        for variant, path in metric_paths.items()
    }
    return {
        "test_pair_ids": list(pair_ids),
        "n_test_pairs": len(pair_ids),
        "n_test_records": len(pair_ids) * 80,
        "euclidean": baseline,
        "models": models,
    }


def split2_metric_path(cell: str) -> Path:
    """Return Split 2 metric path for one held-out cell."""
    return BASE_DIR / "split2_small" / f"split2_{cell}_small_seed0_test_metrics.json"


def summarize_split2(data: dict) -> dict:
    """Summarize all Split 2 LOCO folds."""
    folds = split2_leave_one_cell_out()
    cells = sorted(folds)
    rows = []
    for cell in cells:
        pair_ids = folds[cell]["test_pair_ids"]
        baseline = euclidean_metrics(data, pair_ids)
        model = model_metrics_from_json(split2_metric_path(cell))
        rows.append(
            {
                "cell": cell,
                "test_pair_ids": pair_ids,
                "euclidean": baseline,
                "small": with_delta(model, baseline),
            }
        )

    euc_pa = np.asarray([row["euclidean"]["pairwise_accuracy"] for row in rows], dtype=np.float64)
    cpsi_pa = np.asarray([row["small"]["pairwise_accuracy"] for row in rows], dtype=np.float64)
    return {
        "folds": rows,
        "mean_pairwise_accuracy": {
            "euclidean": float(euc_pa.mean()),
            "small": float(cpsi_pa.mean()),
            "delta": float(cpsi_pa.mean() - euc_pa.mean()),
        },
        "worst_pairwise_accuracy": {
            "euclidean": float(euc_pa.min()),
            "small": float(cpsi_pa.min()),
        },
        "best_pairwise_accuracy": {
            "euclidean": float(euc_pa.max()),
            "small": float(cpsi_pa.max()),
        },
    }


def split3_per_pair_breakdown(data: dict, split3: dict) -> list[dict]:
    """Return per-pair Split 3 PA breakdown for Euclidean, small, and large heads."""
    pair_ids = split3["test_pair_ids"]
    split_data = subset(data, pair_ids)
    euc_costs = euclidean_costs(split_data)
    euc_pa = per_pair_pairwise_accuracy(euc_costs, split_data["v1_cost"], split_data["pair_id"])

    small_metrics = model_metrics_from_json(SPLIT3_METRICS["small"])
    large_metrics = model_metrics_from_json(SPLIT3_METRICS["large"])
    small_costs = predict_checkpoint_costs(
        checkpoint_path=Path(small_metrics["checkpoint_path"]),
        variant="small",
        data=split_data,
    )
    large_costs = predict_checkpoint_costs(
        checkpoint_path=Path(large_metrics["checkpoint_path"]),
        variant="large",
        data=split_data,
    )
    small_pa = per_pair_pairwise_accuracy(small_costs, split_data["v1_cost"], split_data["pair_id"])
    large_pa = per_pair_pairwise_accuracy(large_costs, split_data["v1_cost"], split_data["pair_id"])
    cell_by_pair = {
        int(pair_id): str(split_data["cell"][idx])
        for idx, pair_id in enumerate(split_data["pair_id"])
    }
    return [
        {
            "pair_id": int(pair_id),
            "cell": cell_by_pair[int(pair_id)],
            "euclidean_pairwise_accuracy": euc_pa[int(pair_id)],
            "small_pairwise_accuracy": small_pa[int(pair_id)],
            "large_pairwise_accuracy": large_pa[int(pair_id)],
            "small_delta": small_pa[int(pair_id)] - euc_pa[int(pair_id)],
            "large_delta": large_pa[int(pair_id)] - euc_pa[int(pair_id)],
            "small_improved": small_pa[int(pair_id)] > euc_pa[int(pair_id)],
            "large_improved": large_pa[int(pair_id)] > euc_pa[int(pair_id)],
        }
        for pair_id in pair_ids
    ]


def print_split_summary(name: str, summary: dict) -> None:
    """Print compact split summary."""
    euc = summary["euclidean"]
    models = summary["models"]
    pieces = [
        f"Euclidean pairwise acc: {fmt(euc['pairwise_accuracy'])}",
        f"C_psi small: {fmt(models['small']['pairwise_accuracy'])}",
    ]
    if "large" in models:
        pieces.append(f"C_psi large: {fmt(models['large']['pairwise_accuracy'])}")
    print(f"{name}:")
    print("  " + "   ".join(pieces))
    rho_pieces = [
        f"Euclidean per-pair rho: {fmt(euc['per_pair_spearman_mean'])}",
        f"C_psi small: {fmt(models['small']['per_pair_spearman_mean'])}",
    ]
    if "large" in models:
        rho_pieces.append(f"C_psi large: {fmt(models['large']['per_pair_spearman_mean'])}")
    print("  " + "   ".join(rho_pieces))


def print_split2_table(split2: dict) -> None:
    """Print held-out cell PA table."""
    mean = split2["mean_pairwise_accuracy"]
    worst = split2["worst_pairwise_accuracy"]
    best = split2["best_pairwise_accuracy"]
    print("Split 2 (LOCO, 16 folds, small head):")
    print(
        "  Mean held-out cell pairwise acc:  "
        f"Euclidean: {fmt(mean['euclidean'])}   C_psi: {fmt(mean['small'])}"
    )
    print(
        "  Worst cell pairwise acc:          "
        f"Euclidean: {fmt(worst['euclidean'])}   C_psi: {fmt(worst['small'])}"
    )
    print(
        "  Best cell pairwise acc:           "
        f"Euclidean: {fmt(best['euclidean'])}   C_psi: {fmt(best['small'])}"
    )
    print("  Per-cell table:")
    print("  Cell  | Euclidean PA | C_psi PA | Delta")
    print("  ------+--------------+----------+-------")
    for row in split2["folds"]:
        euc_pa = row["euclidean"]["pairwise_accuracy"]
        cpsi_pa = row["small"]["pairwise_accuracy"]
        print(f"  {row['cell']:<5} | {fmt(euc_pa):>12} | {fmt(cpsi_pa):>8} | {fmt(cpsi_pa - euc_pa):>5}")


def print_split3_pairs(rows: list[dict]) -> None:
    """Print Split 3 per-pair PA breakdown."""
    print("Split 3 per-pair breakdown:")
    print("  Pair | Cell  | Euclidean | Small | Large | Small Delta | Large Delta")
    print("  -----+-------+-----------+-------+-------+-------------+------------")
    for row in rows:
        print(
            f"  {row['pair_id']:>4} | {row['cell']:<5} | "
            f"{fmt(row['euclidean_pairwise_accuracy']):>9} | "
            f"{fmt(row['small_pairwise_accuracy']):>5} | "
            f"{fmt(row['large_pairwise_accuracy']):>5} | "
            f"{fmt(row['small_delta']):>11} | {fmt(row['large_delta']):>10}"
        )
    small_improved = [row["pair_id"] for row in rows if row["small_improved"]]
    large_improved = [row["pair_id"] for row in rows if row["large_improved"]]
    print(f"  Small improved pairs: {small_improved}")
    print(f"  Large improved pairs: {large_improved}")


def main() -> int:
    """Run unified analysis and save JSON summary."""
    data = load_artifact()
    split1 = split1_random_holdout()
    split3 = split3_hard_pair_holdout()
    split1_summary = summarize_split(
        data=data,
        pair_ids=split1["test_pair_ids"],
        metric_paths=SPLIT1_METRICS,
    )
    split3_summary = summarize_split(
        data=data,
        pair_ids=split3["test_pair_ids"],
        metric_paths=SPLIT3_METRICS,
    )
    split2_summary = summarize_split2(data)
    split3_pairs = split3_per_pair_breakdown(data, split3)

    print("P2-0 Metric-Level Summary")
    print_split_summary("Split 1 (random holdout, 15 test pairs)", split1_summary)
    print_split_summary("Split 3 (all_fail+strong_rho, 16 test pairs)", split3_summary)
    print_split2_table(split2_summary)
    print_split3_pairs(split3_pairs)

    summary = {
        "split1": split1_summary,
        "split3": {
            **split3_summary,
            "per_pair_breakdown": split3_pairs,
        },
        "split2": split2_summary,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
