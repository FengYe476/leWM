#!/usr/bin/env python3
"""Sample Track A PushT initial-goal pairs with fixed stratification bins."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lewm_audit.sampling.track_a_pairs import (
    assign_cells,
    enumerate_eligible_pool,
    inspect_h5_dataset,
    sample_stratified_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "stablewm_cache" / "pusht_expert_train.h5"
DEFAULT_BUDGET_PATH = PROJECT_ROOT / "configs" / "phase1" / "track_a_budget.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"


def parse_edges(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        if value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
            values.append(math.inf)
        else:
            values.append(float(value))
    if len(values) < 2:
        raise argparse.ArgumentTypeError("At least two bin edges are required")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("Bin edges must be strictly increasing")
    return values


def edge_to_json(value: float) -> int | float | str:
    if math.isinf(value):
        return "inf"
    if float(value).is_integer():
        return int(value)
    return float(value)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    return value


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


def load_budget(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    return {str(key): int(value) for key, value in data.items()}


def pair_records(sampled_pairs) -> list[dict]:
    records = []
    columns = [
        "pair_id",
        "cell",
        "episode_id",
        "start_row",
        "goal_row",
        "block_displacement_px",
        "required_rotation_rad",
        "physical_pose_distance",
    ]
    for row in sampled_pairs[columns].to_dict(orient="records"):
        records.append(
            {
                "pair_id": int(row["pair_id"]),
                "cell": row["cell"],
                "episode_id": int(row["episode_id"]),
                "start_row": int(row["start_row"]),
                "goal_row": int(row["goal_row"]),
                "block_displacement_px": float(row["block_displacement_px"]),
                "required_rotation_rad": float(row["required_rotation_rad"]),
                "physical_pose_distance": float(row["physical_pose_distance"]),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Track A stratified PushT pairs.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--offset", type=int, default=50)
    parser.add_argument(
        "--displacement-edges",
        type=parse_edges,
        default=parse_edges("0,10,50,120,inf"),
    )
    parser.add_argument(
        "--rotation-edges",
        type=parse_edges,
        default=parse_edges("0,0.25,0.75,1.25,inf"),
    )
    parser.add_argument("--budget-matrix-path", type=Path, default=DEFAULT_BUDGET_PATH)
    parser.add_argument("--per-episode-per-cell-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.budget_matrix_path = args.budget_matrix_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    budget_matrix = load_budget(args.budget_matrix_path)
    metadata = inspect_h5_dataset(args.dataset_path)
    pool = enumerate_eligible_pool(str(args.dataset_path), args.offset)
    pool_with_cells = assign_cells(
        pool,
        displacement_edges=args.displacement_edges,
        rotation_edges=args.rotation_edges,
    )
    sampled_pairs, sampling_report = sample_stratified_pairs(
        pool_with_cells,
        budget_matrix=budget_matrix,
        per_episode_per_cell_limit=args.per_episode_per_cell_limit,
        rng=np.random.default_rng(args.seed),
    )

    target_total = int(sum(budget_matrix.values()))
    actually_sampled_total = int(len(sampled_pairs))
    n_capped_cells = int(sum(1 for entry in sampling_report.values() if entry["capped"]))
    output = {
        "metadata": {
            "dataset_path": str(args.dataset_path),
            "dataset_n_rows": metadata["dataset_n_rows"],
            "dataset_n_episodes": metadata["dataset_n_episodes"],
            "offset": args.offset,
            "seed": args.seed,
            "displacement_edges": [edge_to_json(value) for value in args.displacement_edges],
            "rotation_edges": [edge_to_json(value) for value in args.rotation_edges],
            "per_episode_per_cell_limit": args.per_episode_per_cell_limit,
            "git_commit": get_git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "budget_matrix": budget_matrix,
        "sampling_report": sampling_report,
        "totals": {
            "target_total": target_total,
            "actually_sampled_total": actually_sampled_total,
            "n_capped_cells": n_capped_cells,
        },
        "pairs": pair_records(sampled_pairs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(output), indent=2, allow_nan=False) + "\n")
    print(f"eligible_pool_rows: {len(pool_with_cells)}")
    print(f"target_total: {target_total}")
    print(f"actually_sampled_total: {actually_sampled_total}")
    print(f"n_capped_cells: {n_capped_cells}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
