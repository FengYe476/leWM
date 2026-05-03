#!/usr/bin/env python3
"""Sample OGBench-Cube initial-goal pairs with fixed stratification bins."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from lewm_audit.sampling.track_a_pairs import sample_stratified_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "stablewm_cache" / "ogbench" / "cube_single_expert.h5"
DEFAULT_BUDGET_PATH = PROJECT_ROOT / "configs" / "phase2" / "cube_budget.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_pairs.json"

DEFAULT_DISPLACEMENT_EDGES = [0.0, 0.05, 0.16, 0.25, math.inf]
DEFAULT_ORIENTATION_EDGES = [0.0, 0.01, 0.03, 0.25, math.inf]
REQUIRED_H5_KEYS = (
    "ep_idx",
    "step_idx",
    "ep_len",
    "privileged_block_0_pos",
    "privileged_block_0_quat",
    "privileged_block_0_yaw",
)


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


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def require_h5_keys(handle: h5py.File, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in handle]
    if missing:
        raise KeyError(f"Cube dataset is missing required HDF5 keys: {missing}")


def expected_cells() -> list[str]:
    return [f"D{d}xR{r}" for d in range(4) for r in range(4)]


def load_budget(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    budget = {str(key): int(value) for key, value in data.items()}
    expected = set(expected_cells())
    missing = sorted(expected.difference(budget))
    extra = sorted(set(budget).difference(expected))
    if missing or extra:
        raise ValueError(
            f"Budget cells must be exactly D0xR0..D3xR3; missing={missing}, extra={extra}"
        )
    if any(value < 0 for value in budget.values()):
        raise ValueError("Budget values must be nonnegative")
    return budget


def inspect_h5_dataset(dataset_path: Path) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as handle:
        require_h5_keys(handle, REQUIRED_H5_KEYS)
        ep_idx = np.asarray(handle["ep_idx"][:], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"][:], dtype=np.int64)
        return {
            "dataset_n_rows": int(handle["ep_idx"].shape[0]),
            "dataset_n_episodes": int(len(np.unique(ep_idx))),
            "episode_length_rows": {
                "min": int(ep_len.min()),
                "mean": float(ep_len.mean()),
                "median": float(np.median(ep_len)),
                "max": int(ep_len.max()),
                "unique": sorted(int(item) for item in np.unique(ep_len).tolist()),
            },
            "fixed_episode_length_rows": int(ep_len[0]) if np.all(ep_len == ep_len[0]) else None,
            "fixed_episode_length_assumption_holds": bool(np.all(ep_len == ep_len[0])),
        }


def episode_lengths_for_rows(ep_idx: np.ndarray, ep_len: np.ndarray) -> np.ndarray:
    if ep_idx.min() >= 0 and ep_idx.max() < len(ep_len):
        return ep_len[ep_idx]
    unique_episodes = np.unique(ep_idx)
    if len(unique_episodes) != len(ep_len):
        raise ValueError(
            "Cannot map ep_len to ep_idx values: "
            f"{len(unique_episodes)} unique ep_idx values vs {len(ep_len)} ep_len rows"
        )
    length_by_episode = dict(zip(unique_episodes.tolist(), ep_len.tolist(), strict=True))
    return np.asarray([length_by_episode[int(ep)] for ep in ep_idx], dtype=np.int64)


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norm <= 0):
        bad = np.flatnonzero((norm <= 0).reshape(-1))[:5]
        raise ValueError(f"Encountered zero-norm Cube quaternions at local rows: {bad.tolist()}")
    return quat / norm


def quaternion_geodesic_angle(q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    q_start = normalize_quaternions(np.asarray(q_start, dtype=np.float64))
    q_goal = normalize_quaternions(np.asarray(q_goal, dtype=np.float64))
    dots = np.abs(np.sum(q_start * q_goal, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return (2.0 * np.arccos(dots)).astype(np.float64)


def wrapped_abs_angle_delta(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    delta = (np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi
    return np.abs(delta).astype(np.float64)


def enumerate_eligible_pool(dataset_path: str | Path, offset: int) -> pd.DataFrame:
    """Enumerate all valid Cube start/goal pairs for a fixed raw-step offset."""
    if offset <= 0:
        raise ValueError("offset must be positive")

    dataset_path = Path(dataset_path).expanduser()
    with h5py.File(dataset_path, "r") as handle:
        require_h5_keys(handle, REQUIRED_H5_KEYS)
        n_rows = int(handle["ep_idx"].shape[0])
        if offset >= n_rows:
            raise ValueError(f"offset={offset} is not smaller than dataset rows={n_rows}")

        ep_idx = np.asarray(handle["ep_idx"][:], dtype=np.int64)
        step_idx = np.asarray(handle["step_idx"][:], dtype=np.int64)
        ep_len = np.asarray(handle["ep_len"][:], dtype=np.int64)
        row_episode_lengths = episode_lengths_for_rows(ep_idx, ep_len)

        candidate_rows = np.arange(0, n_rows - offset, dtype=np.int64)
        step_valid = step_idx[:-offset] <= (row_episode_lengths[:-offset] - offset - 1)
        same_episode = ep_idx[:-offset] == ep_idx[offset:]
        valid_mask = step_valid & same_episode
        start_rows = candidate_rows[valid_mask]
        goal_rows = start_rows + offset

        start_pos = np.asarray(handle["privileged_block_0_pos"][start_rows], dtype=np.float64)
        goal_pos = np.asarray(handle["privileged_block_0_pos"][goal_rows], dtype=np.float64)
        start_quat = np.asarray(handle["privileged_block_0_quat"][start_rows], dtype=np.float64)
        goal_quat = np.asarray(handle["privileged_block_0_quat"][goal_rows], dtype=np.float64)
        start_yaw = np.asarray(handle["privileged_block_0_yaw"][start_rows], dtype=np.float64).reshape(-1)
        goal_yaw = np.asarray(handle["privileged_block_0_yaw"][goal_rows], dtype=np.float64).reshape(-1)

    displacement = np.linalg.norm(goal_pos - start_pos, axis=1).astype(np.float64)
    orientation = quaternion_geodesic_angle(start_quat, goal_quat)
    yaw_delta = wrapped_abs_angle_delta(start_yaw, goal_yaw)
    start_step_idx = step_idx[start_rows]
    goal_step_idx = step_idx[goal_rows]
    episode_start_rows = start_rows - start_step_idx

    return pd.DataFrame(
        {
            "episode_id": ep_idx[start_rows].astype(np.int64),
            "step_idx": start_step_idx.astype(np.int64),
            "goal_step_idx": goal_step_idx.astype(np.int64),
            "start_row": start_rows.astype(np.int64),
            "goal_row": goal_rows.astype(np.int64),
            "episode_start_row": episode_start_rows.astype(np.int64),
            "displacement_m": displacement,
            "orientation_rad": orientation,
            "yaw_delta_rad": yaw_delta,
            "start_cube_pos": list(start_pos.astype(np.float64)),
            "goal_cube_pos": list(goal_pos.astype(np.float64)),
            "start_cube_quat": list(start_quat.astype(np.float64)),
            "goal_cube_quat": list(goal_quat.astype(np.float64)),
            "start_cube_yaw": start_yaw.astype(np.float64),
            "goal_cube_yaw": goal_yaw.astype(np.float64),
        }
    )


def validate_edges(edges: list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError(f"{name} must contain at least two edges")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def assign_axis(values: np.ndarray, edges: np.ndarray, *, name: str) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    invalid = (indices < 0) | (indices >= len(edges) - 1)
    if np.any(invalid):
        bad = values[invalid][:5]
        raise ValueError(f"{name} values outside provided bin edges, examples={bad.tolist()}")
    return indices.astype(np.int64)


def assign_cells(
    pool: pd.DataFrame,
    *,
    displacement_edges: list[float],
    orientation_edges: list[float],
) -> pd.DataFrame:
    required = {"displacement_m", "orientation_rad"}
    missing = required - set(pool.columns)
    if missing:
        raise KeyError(f"Pool is missing required columns: {sorted(missing)}")

    d_edges = validate_edges(displacement_edges, name="displacement_edges")
    r_edges = validate_edges(orientation_edges, name="orientation_edges")
    out = pool.copy()
    cell_d = assign_axis(out["displacement_m"].to_numpy(), d_edges, name="displacement")
    cell_r = assign_axis(out["orientation_rad"].to_numpy(), r_edges, name="orientation")
    out["cell_d"] = cell_d
    out["cell_r"] = cell_r
    out["cell"] = [f"D{d}xR{r}" for d, r in zip(cell_d, cell_r, strict=True)]
    return out


def pair_records(sampled_pairs: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    columns = [
        "pair_id",
        "cell",
        "episode_id",
        "start_row",
        "goal_row",
        "step_idx",
        "goal_step_idx",
        "episode_start_row",
        "displacement_m",
        "orientation_rad",
        "yaw_delta_rad",
        "start_cube_pos",
        "goal_cube_pos",
        "start_cube_quat",
        "goal_cube_quat",
        "start_cube_yaw",
        "goal_cube_yaw",
    ]
    for row in sampled_pairs[columns].to_dict(orient="records"):
        records.append(
            {
                "pair_id": int(row["pair_id"]),
                "cell": str(row["cell"]),
                "episode_id": int(row["episode_id"]),
                "start_row": int(row["start_row"]),
                "goal_row": int(row["goal_row"]),
                "step_idx": int(row["step_idx"]),
                "goal_step_idx": int(row["goal_step_idx"]),
                "episode_start_row": int(row["episode_start_row"]),
                "displacement_m": float(row["displacement_m"]),
                "orientation_rad": float(row["orientation_rad"]),
                "yaw_delta_rad": float(row["yaw_delta_rad"]),
                "start_cube_pos": [float(item) for item in row["start_cube_pos"]],
                "goal_cube_pos": [float(item) for item in row["goal_cube_pos"]],
                "start_cube_quat": [float(item) for item in row["start_cube_quat"]],
                "goal_cube_quat": [float(item) for item in row["goal_cube_quat"]],
                "start_cube_yaw": float(row["start_cube_yaw"]),
                "goal_cube_yaw": float(row["goal_cube_yaw"]),
            }
        )
    return records


def print_sampling_grid(sampling_report: dict[str, dict[str, Any]]) -> None:
    print("Cube sampling grid (sampled/target):")
    print("      R0     R1     R2     R3")
    for d in range(4):
        cells = []
        for r in range(4):
            cell = f"D{d}xR{r}"
            entry = sampling_report[cell]
            cells.append(f"{entry['actually_sampled']:>2}/{entry['target']:<2}")
        print(f"D{d}  " + "  ".join(cells))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--offset", type=int, default=50)
    parser.add_argument(
        "--displacement-edges",
        type=parse_edges,
        default=DEFAULT_DISPLACEMENT_EDGES,
    )
    parser.add_argument(
        "--orientation-edges",
        type=parse_edges,
        default=DEFAULT_ORIENTATION_EDGES,
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
    pool = enumerate_eligible_pool(args.dataset_path, int(args.offset))
    pool_with_cells = assign_cells(
        pool,
        displacement_edges=args.displacement_edges,
        orientation_edges=args.orientation_edges,
    )
    sampled_pairs, sampling_report = sample_stratified_pairs(
        pool_with_cells,
        budget_matrix=budget_matrix,
        per_episode_per_cell_limit=int(args.per_episode_per_cell_limit),
        rng=np.random.default_rng(int(args.seed)),
    )

    target_total = int(sum(budget_matrix.values()))
    actually_sampled_total = int(len(sampled_pairs))
    n_capped_cells = int(sum(1 for entry in sampling_report.values() if entry["capped"]))
    output = {
        "metadata": {
            "format": "ogbench_cube_stratified_pairs",
            "dataset_path": str(args.dataset_path),
            "dataset_n_rows": metadata["dataset_n_rows"],
            "dataset_n_episodes": metadata["dataset_n_episodes"],
            "episode_length_rows": metadata["episode_length_rows"],
            "fixed_episode_length_rows": metadata["fixed_episode_length_rows"],
            "fixed_episode_length_assumption_holds": metadata["fixed_episode_length_assumption_holds"],
            "offset": int(args.offset),
            "seed": int(args.seed),
            "grid": "4x4 displacement_m x orientation_rad",
            "axis_definitions": {
                "displacement_m": "Euclidean distance between privileged_block_0_pos at goal and start, in meters.",
                "orientation_rad": "Quaternion geodesic angle 2*arccos(abs(dot(q_start, q_goal))).",
                "yaw_delta_rad": "Wrapped absolute privileged_block_0_yaw delta, recorded for interpretability only.",
            },
            "displacement_edges": [edge_to_json(value) for value in args.displacement_edges],
            "orientation_edges": [edge_to_json(value) for value in args.orientation_edges],
            "budget_matrix_path": str(args.budget_matrix_path),
            "budget_total": target_total,
            "per_episode_per_cell_limit": int(args.per_episode_per_cell_limit),
            "valid_start_rule": "step_idx <= ep_len - offset - 1 and goal_row remains in the same ep_idx",
            "eligible_pool_rows": int(len(pool_with_cells)),
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
    print_sampling_grid(sampling_report)
    print(f"eligible_pool_rows: {len(pool_with_cells)}")
    print(f"target_total: {target_total}")
    print(f"actually_sampled_total: {actually_sampled_total}")
    print(f"n_capped_cells: {n_capped_cells}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
