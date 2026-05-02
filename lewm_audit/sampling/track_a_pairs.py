"""Track A stratified PushT initial-goal pair sampler."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lewm_audit.diagnostics.per_pair import extract_physical_features


DEFAULT_DISPLACEMENT_EDGES = [0.0, 10.0, 50.0, 120.0, math.inf]
DEFAULT_ROTATION_EDGES = [0.0, 0.25, 0.75, 1.25, math.inf]


def _require_h5_keys(handle: h5py.File, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in handle]
    if missing:
        raise KeyError(f"Dataset is missing required HDF5 keys: {missing}")


def inspect_h5_dataset(dataset_path: str | Path) -> dict:
    """Return lightweight metadata used in sampler JSON headers."""
    with h5py.File(dataset_path, "r") as handle:
        _require_h5_keys(handle, ("state", "episode_idx"))
        return {
            "dataset_n_rows": int(handle["state"].shape[0]),
            "dataset_n_episodes": int(len(np.unique(handle["episode_idx"][:]))),
        }


def enumerate_eligible_pool(
    dataset_path: str,
    offset: int,
) -> pd.DataFrame:
    """Enumerate all valid PushT start/goal pairs for a fixed raw-step offset."""
    if offset <= 0:
        raise ValueError("offset must be positive")

    dataset_path = str(Path(dataset_path).expanduser())
    with h5py.File(dataset_path, "r") as handle:
        _require_h5_keys(handle, ("state", "episode_idx", "step_idx"))
        n_rows = int(handle["state"].shape[0])
        if offset >= n_rows:
            raise ValueError(f"offset={offset} is not smaller than dataset rows={n_rows}")

        episode_ids = handle["episode_idx"][:]
        step_idx = handle["step_idx"][:]
        candidate_rows = np.arange(0, n_rows - offset, dtype=np.int64)
        valid_mask = episode_ids[:-offset] == episode_ids[offset:]
        start_rows = candidate_rows[valid_mask]
        goal_rows = start_rows + offset

        start_states = np.asarray(handle["state"][start_rows], dtype=np.float64)
        goal_states = np.asarray(handle["state"][goal_rows], dtype=np.float64)

    block_displacement = np.empty(len(start_rows), dtype=np.float64)
    rotation_required = np.empty(len(start_rows), dtype=np.float64)
    physical_pose_distance = np.empty(len(start_rows), dtype=np.float64)
    for idx, (initial_state, goal_state) in enumerate(zip(start_states, goal_states, strict=True)):
        physical = extract_physical_features(initial_state, goal_state)
        block_displacement[idx] = physical["block_displacement"]
        rotation_required[idx] = physical["rotation_required"]
        physical_pose_distance[idx] = physical["physical_pose_distance"]

    start_step_idx = step_idx[start_rows]
    episode_start_rows = start_rows - start_step_idx
    return pd.DataFrame(
        {
            "episode_id": episode_ids[start_rows].astype(np.int64),
            "step_idx": start_step_idx.astype(np.int64),
            "start_row": start_rows.astype(np.int64),
            "goal_row": goal_rows.astype(np.int64),
            "episode_start_row": episode_start_rows.astype(np.int64),
            "block_displacement_px": block_displacement,
            "required_rotation_rad": rotation_required,
            "physical_pose_distance": physical_pose_distance,
        }
    )


def _validate_edges(edges: list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError(f"{name} must contain at least two edges")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _assign_axis(values: np.ndarray, edges: np.ndarray, *, name: str) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    invalid = (indices < 0) | (indices >= len(edges) - 1)
    if np.any(invalid):
        bad = values[invalid][:5]
        raise ValueError(f"{name} values outside provided bin edges, examples={bad.tolist()}")
    return indices.astype(np.int64)


def assign_cells(
    pool: pd.DataFrame,
    displacement_edges: list[float],
    rotation_edges: list[float],
) -> pd.DataFrame:
    """Add `cell`, `cell_d`, and `cell_r` labels to an eligible-pair pool."""
    required = {"block_displacement_px", "required_rotation_rad"}
    missing = required - set(pool.columns)
    if missing:
        raise KeyError(f"Pool is missing required columns: {sorted(missing)}")

    d_edges = _validate_edges(displacement_edges, name="displacement_edges")
    r_edges = _validate_edges(rotation_edges, name="rotation_edges")
    out = pool.copy()
    cell_d = _assign_axis(out["block_displacement_px"].to_numpy(), d_edges, name="displacement")
    cell_r = _assign_axis(out["required_rotation_rad"].to_numpy(), r_edges, name="rotation")
    out["cell_d"] = cell_d
    out["cell_r"] = cell_r
    out["cell"] = [f"D{d}xR{r}" for d, r in zip(cell_d, cell_r, strict=True)]
    return out


def _shuffle_take(group: pd.DataFrame, *, limit: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(group) <= limit:
        return group
    positions = rng.permutation(len(group))[:limit]
    return group.iloc[positions]


def _sample_rows(frame: pd.DataFrame, *, count: int, rng: np.random.Generator) -> pd.DataFrame:
    if count >= len(frame):
        return frame
    positions = rng.choice(len(frame), size=count, replace=False)
    return frame.iloc[positions]


def sample_stratified_pairs(
    pool_with_cells: pd.DataFrame,
    budget_matrix: dict[str, int],
    per_episode_per_cell_limit: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Sample pairs independently within each stratification cell."""
    if per_episode_per_cell_limit < 1:
        raise ValueError("per_episode_per_cell_limit must be at least 1")

    required = {"cell", "episode_id", "start_row"}
    missing = required - set(pool_with_cells.columns)
    if missing:
        raise KeyError(f"Pool is missing required columns: {sorted(missing)}")

    sampled_frames = []
    report = {}
    ordered_cells = sorted(
        budget_matrix,
        key=lambda cell: tuple(int(part[1:]) for part in cell.split("x")),
    )

    stable_pool = pool_with_cells.sort_values(["cell", "episode_id", "start_row"]).reset_index(
        drop=True
    )
    for cell in ordered_cells:
        target = int(budget_matrix[cell])
        if target < 0:
            raise ValueError(f"Budget for {cell} must be nonnegative")

        cell_pool = stable_pool[stable_pool["cell"] == cell].copy().reset_index(drop=True)
        eligible_pool_size = int(len(cell_pool))

        deduped_parts = []
        if eligible_pool_size:
            for _, group in cell_pool.groupby("episode_id", sort=True):
                deduped_parts.append(
                    _shuffle_take(
                        group.reset_index(drop=True),
                        limit=per_episode_per_cell_limit,
                        rng=rng,
                    )
                )
        deduped_pool = (
            pd.concat(deduped_parts, ignore_index=True)
            if deduped_parts
            else cell_pool.iloc[0:0].copy()
        )
        after_episode_dedup_size = int(len(deduped_pool))

        capped = False
        cap_reason = None
        if target == 0:
            selected = deduped_pool.iloc[0:0].copy()
        elif eligible_pool_size == 0:
            capped = True
            cap_reason = "empty_pool"
            selected = deduped_pool.copy()
        elif after_episode_dedup_size < target:
            capped = True
            cap_reason = (
                "fewer_in_pool_than_target"
                if eligible_pool_size < target
                else "fewer_after_episode_dedup_than_target"
            )
            selected = deduped_pool.copy()
        else:
            selected = _sample_rows(deduped_pool, count=target, rng=rng).copy()

        report[cell] = {
            "target": target,
            "eligible_pool_size": eligible_pool_size,
            "after_episode_dedup_size": after_episode_dedup_size,
            "actually_sampled": int(len(selected)),
            "capped": capped,
            "cap_reason": cap_reason,
        }
        if len(selected):
            sampled_frames.append(selected)

    sampled_pairs = (
        pd.concat(sampled_frames, ignore_index=True)
        if sampled_frames
        else pool_with_cells.iloc[0:0].copy()
    )
    sampled_pairs = sampled_pairs.reset_index(drop=True)
    sampled_pairs.insert(0, "pair_id", np.arange(len(sampled_pairs), dtype=np.int64))
    return sampled_pairs, report
