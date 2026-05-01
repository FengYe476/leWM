"""Failure-mode decomposition helpers for Track A supplementary analysis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


SUCCESS_CLASSES = ("all_fail", "some_succ", "all_succ")
ENCODER_CLASSES = ("neg_rho", "weak_rho", "strong_rho")


def cell_indices(cell: str) -> tuple[int, int]:
    left, right = str(cell).split("x")
    return int(left[1:]), int(right[1:])


def success_class(success_count: int, total_actions: int = 80) -> str:
    success_count = int(success_count)
    if success_count == 0:
        return "all_fail"
    if success_count == int(total_actions):
        return "all_succ"
    return "some_succ"


def encoder_class(rho: float, neg_threshold: float = 0.0, weak_threshold: float = 0.3) -> str:
    rho = float(rho)
    if rho < neg_threshold:
        return "neg_rho"
    if rho < weak_threshold:
        return "weak_rho"
    return "strong_rho"


def classify_pairs(
    per_pair_records,
    success_count_by_pair,
    neg_threshold: float = 0.0,
    weak_threshold: float = 0.3,
) -> pd.DataFrame:
    """Classify pairs by success count and per-pair encoder/state rho."""
    rows = []
    for record in per_pair_records:
        pair_id = int(record["pair_id"])
        count = int(success_count_by_pair[pair_id])
        total_actions = int(record.get("total_actions", 80))
        cell = str(record["cell"])
        cell_d, cell_r = cell_indices(cell)
        rho = float(record["rho"])
        rows.append(
            {
                "pair_id": pair_id,
                "cell": cell,
                "success_count": count,
                "rho": rho,
                "success_class": success_class(count, total_actions=total_actions),
                "encoder_class": encoder_class(
                    rho,
                    neg_threshold=neg_threshold,
                    weak_threshold=weak_threshold,
                ),
                "cell_d": cell_d,
                "cell_r": cell_r,
                "block_displacement_px": float(record["block_displacement_px"]),
                "required_rotation_rad": float(record["required_rotation_rad"]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "cell",
                "success_count",
                "rho",
                "success_class",
                "encoder_class",
                "cell_d",
                "cell_r",
                "block_displacement_px",
                "required_rotation_rad",
            ]
        ).rename_axis("pair_id")
    return pd.DataFrame(rows).set_index("pair_id").sort_index()


def _quadrant_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n_pairs": 0,
            "pair_ids": [],
            "mean_displacement_px": None,
            "mean_rotation_rad": None,
            "cells_present": [],
        }
    return {
        "n_pairs": int(len(frame)),
        "pair_ids": [int(idx) for idx in frame.index.tolist()],
        "mean_displacement_px": float(np.mean(frame["block_displacement_px"].to_numpy())),
        "mean_rotation_rad": float(np.mean(frame["required_rotation_rad"].to_numpy())),
        "cells_present": sorted(frame["cell"].unique().tolist()),
    }


def quadrant_table(classified_df) -> dict:
    """Return nested quadrant summaries for every success/encoder class."""
    table = {}
    for s_class in SUCCESS_CLASSES:
        table[s_class] = {}
        for e_class in ENCODER_CLASSES:
            mask = (
                (classified_df["success_class"] == s_class)
                & (classified_df["encoder_class"] == e_class)
            )
            table[s_class][e_class] = _quadrant_summary(classified_df[mask])
    return table


def count_matrix(classified_df: pd.DataFrame) -> dict:
    counts = {
        s_class: {e_class: 0 for e_class in ENCODER_CLASSES}
        for s_class in SUCCESS_CLASSES
    }
    if classified_df.empty:
        return counts
    grouped = classified_df.groupby(["success_class", "encoder_class"]).size()
    for (s_class, e_class), value in grouped.items():
        counts[str(s_class)][str(e_class)] = int(value)
    return counts


def all_fail_source_verification(records_by_pair: dict[int, dict]) -> dict:
    """Verify all-fail pairs have zero successes in each source bucket."""
    by_pair = []
    flagged_pairs = []
    source_names = ["data", "smooth_random", "CEM_early", "CEM_late"]
    for pair_id, pair in sorted(records_by_pair.items()):
        actions = list(pair["actions"])
        success_count_total = int(sum(bool(action["success"]) for action in actions))
        if success_count_total != 0:
            continue
        rates = {}
        counts = {}
        for source in source_names:
            source_actions = [action for action in actions if action["source"] == source]
            successes = int(sum(bool(action["success"]) for action in source_actions))
            counts[source] = {
                "successes": successes,
                "n": int(len(source_actions)),
            }
            rates[source] = float(successes / len(source_actions)) if source_actions else None
        nonzero_sources = [
            source
            for source, rate in rates.items()
            if rate is not None and rate > 0.0
        ]
        if nonzero_sources:
            flagged_pairs.append({"pair_id": int(pair_id), "nonzero_sources": nonzero_sources})
        by_pair.append(
            {
                "pair_id": int(pair_id),
                "cell": str(pair["cell"]),
                "source_success_rates": rates,
                "source_success_counts": counts,
                "nonzero_source_success": bool(nonzero_sources),
            }
        )
    return {
        "n_all_fail_pairs": int(len(by_pair)),
        "all_source_rates_zero": not flagged_pairs,
        "flagged_pairs": flagged_pairs,
        "by_pair": by_pair,
    }


def quadrant_label(row: pd.Series | dict) -> str:
    return f"{row['success_class']} + {row['encoder_class']}"


def counts_by_cell_and_quadrant(classified_df: pd.DataFrame) -> dict:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    if classified_df.empty:
        return {}
    for _, row in classified_df.iterrows():
        cell = str(row["cell"])
        label = quadrant_label(row)
        grouped[cell][label] = grouped[cell].get(label, 0) + 1
    return {cell: dict(counts) for cell, counts in sorted(grouped.items())}
