#!/usr/bin/env python3
"""Analyze Phase 2 Track B encoder coarse-ranking quality."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


PROJECT_ROOT_LOCAL = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_LOCAL))

from scripts.phase2.track_b_common import DEFAULT_LATENT_ARTIFACT, PROJECT_ROOT


DEFAULT_DINO = PROJECT_ROOT / "results" / "phase2" / "track_b" / "dinov2_features.pt"
DEFAULT_RANDOM = PROJECT_ROOT / "results" / "phase2" / "track_b" / "random_projection_features.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "track_b" / "ranking_comparison.json"


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def squared_l2_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum((a - b) ** 2, axis=1)


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


def per_pair_spearman(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> dict:
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
        "mean": clean_float(arr.mean()) if len(arr) else None,
        "std": clean_float(arr.std()) if len(arr) else None,
        "by_pair": by_pair,
    }


def evaluate_encoder(
    *,
    name: str,
    dim: int,
    costs: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
) -> dict:
    rho, _ = spearmanr(costs, labels)
    pair_summary = per_pair_spearman(costs, labels, pair_ids)
    return {
        "encoder": name,
        "dim": int(dim),
        "global_spearman": clean_float(rho),
        "pairwise_accuracy": clean_float(pairwise_accuracy(costs, labels, pair_ids)),
        "per_pair_rho_mean": pair_summary["mean"],
        "per_pair_rho_std": pair_summary["std"],
        "per_pair_spearman": pair_summary["by_pair"],
    }


def validate_feature_order(name: str, feature_artifact: dict, latent_artifact: dict) -> None:
    for key in ("pair_id", "action_id"):
        if key not in feature_artifact:
            continue
        if not torch.equal(feature_artifact[key].cpu(), latent_artifact[key].cpu()):
            raise RuntimeError(f"{name} {key} ordering does not match latent artifact")


def per_cell_pairwise(
    *,
    cells: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
    costs_by_name: dict[str, np.ndarray],
    best_dino_name: str | None,
) -> list[dict]:
    rows = []
    for cell in sorted(np.unique(cells).tolist()):
        mask = cells == cell
        row = {"cell": str(cell), "n_records": int(mask.sum())}
        for name, costs in costs_by_name.items():
            row[name] = clean_float(pairwise_accuracy(costs[mask], labels[mask], pair_ids[mask]))
        if best_dino_name is not None:
            row["best_dino"] = best_dino_name
            row["dino_delta_vs_lewm"] = clean_float(row[best_dino_name] - row["LeWM (SIGReg)"])
        rows.append(row)
    return rows


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def print_main_table(rows: list[dict]) -> None:
    headers = [
        "Encoder",
        "Dim",
        "Global Spearman",
        "Pairwise Acc",
        "Per-Pair Rho Mean",
        "Per-Pair Rho Std",
    ]
    table = [
        [
            row["encoder"],
            str(row["dim"]),
            fmt(row["global_spearman"]),
            fmt(row["pairwise_accuracy"]),
            fmt(row["per_pair_rho_mean"]),
            fmt(row["per_pair_rho_std"]),
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(record[i]) for record in table)) for i in range(len(headers))]
    print("Track B: Encoder Coarse Ranking Comparison")
    print("(100 pairs x 80 actions, ranking against V1 hinge cost)")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def print_cell_table(rows: list[dict], best_dino_name: str | None) -> None:
    if best_dino_name is None:
        return
    print("\nPer-cell pairwise accuracy")
    headers = ["Cell", "LeWM PA", f"{best_dino_name} PA", "Delta"]
    table = [
        [
            row["cell"],
            fmt(row["LeWM (SIGReg)"]),
            fmt(row[best_dino_name]),
            fmt(row["dino_delta_vs_lewm"]),
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(record[i]) for record in table)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument("--dino-features", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--random-features", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.dino_features = args.dino_features.expanduser().resolve()
    args.random_features = args.random_features.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    latent = torch.load(args.latent_artifact, map_location="cpu", weights_only=False)
    labels = latent["v1_cost"].cpu().numpy()
    pair_ids = latent["pair_id"].cpu().numpy()
    cells = np.asarray(latent["cell"])

    costs_by_name = {
        "LeWM (SIGReg)": squared_l2_cost(
            latent["z_terminal"].cpu().numpy(),
            latent["z_goal"].cpu().numpy(),
        )
    }
    dims = {"LeWM (SIGReg)": int(latent["z_terminal"].shape[1])}
    artifacts = {"latent_artifact": str(args.latent_artifact)}

    if args.dino_features.exists():
        dino = torch.load(args.dino_features, map_location="cpu", weights_only=False)
        validate_feature_order("DINOv2", dino, latent)
        costs_by_name["DINOv2 CLS"] = squared_l2_cost(
            dino["d_terminal_cls"].cpu().numpy(),
            dino["d_goal_cls"].cpu().numpy(),
        )
        costs_by_name["DINOv2 mean-pool"] = squared_l2_cost(
            dino["d_terminal_mean"].cpu().numpy(),
            dino["d_goal_mean"].cpu().numpy(),
        )
        dims["DINOv2 CLS"] = int(dino["d_terminal_cls"].shape[1])
        dims["DINOv2 mean-pool"] = int(dino["d_terminal_mean"].shape[1])
        artifacts["dino_features"] = str(args.dino_features)
    else:
        print(f"DINOv2 feature artifact not found, skipping: {args.dino_features}")

    if args.random_features.exists():
        random_artifact = torch.load(args.random_features, map_location="cpu", weights_only=False)
        validate_feature_order("Random projection", random_artifact, latent)
        costs_by_name["Random projection"] = squared_l2_cost(
            random_artifact["r_terminal"].cpu().numpy(),
            random_artifact["r_goal"].cpu().numpy(),
        )
        dims["Random projection"] = int(random_artifact["r_terminal"].shape[1])
        artifacts["random_features"] = str(args.random_features)

    rows = [
        evaluate_encoder(
            name=name,
            dim=dims[name],
            costs=costs,
            labels=labels,
            pair_ids=pair_ids,
        )
        for name, costs in costs_by_name.items()
    ]
    dino_rows = [row for row in rows if row["encoder"].startswith("DINOv2")]
    best_dino_name = None
    if dino_rows:
        best_dino_name = max(dino_rows, key=lambda row: row["pairwise_accuracy"])["encoder"]
    cell_rows = per_cell_pairwise(
        cells=cells,
        labels=labels,
        pair_ids=pair_ids,
        costs_by_name=costs_by_name,
        best_dino_name=best_dino_name,
    )

    print_main_table(rows)
    print_cell_table(cell_rows, best_dino_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "format": "phase2_track_b_ranking_comparison",
            "n_records": int(len(labels)),
            "n_pairs": int(len(np.unique(pair_ids))),
            "ranking_target": "V1 hinge cost",
            "pairwise_accuracy": "Within-pair pairwise accuracy, skipping V1 ties.",
            **artifacts,
        },
        "rows": rows,
        "best_dino_variant": best_dino_name,
        "per_cell_pairwise_accuracy": cell_rows,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
