"""Shared PushT dataset-index helpers used by the audit scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetIndex:
    col_name: str
    episode_ids: np.ndarray
    step_idx: np.ndarray
    episode_indices: np.ndarray
    episode_inverse: np.ndarray
    episode_lengths: np.ndarray

    @property
    def total_episodes(self) -> int:
        return int(len(self.episode_indices))

    @property
    def total_rows(self) -> int:
        return int(len(self.step_idx))


def prepare_dataset_index(dataset) -> DatasetIndex:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    episode_indices, episode_inverse = np.unique(episode_ids, return_inverse=True)
    episode_lengths = np.zeros(len(episode_indices), dtype=np.int64)
    np.maximum.at(episode_lengths, episode_inverse, step_idx + 1)
    return DatasetIndex(
        col_name=col_name,
        episode_ids=episode_ids,
        step_idx=step_idx,
        episode_indices=episode_indices,
        episode_inverse=episode_inverse,
        episode_lengths=episode_lengths,
    )


def analyze_offset(index: DatasetIndex, offset: int) -> dict:
    max_start_per_episode = index.episode_lengths - offset - 1
    valid_mask = index.step_idx <= max_start_per_episode[index.episode_inverse]
    valid_indices = np.flatnonzero(valid_mask)
    eligible_episode_mask = index.episode_lengths > offset

    return {
        "offset": offset,
        "valid_indices": valid_indices,
        "valid_start_points": int(len(valid_indices)),
        "eligible_episodes": int(np.count_nonzero(eligible_episode_mask)),
        "ineligible_episodes": int(np.count_nonzero(~eligible_episode_mask)),
        "eligible_episode_fraction": float(np.mean(eligible_episode_mask)),
        "max_episode_length": int(index.episode_lengths.max()),
        "mean_episode_length": float(index.episode_lengths.mean()),
        "median_episode_length": float(np.median(index.episode_lengths)),
        "min_episode_length": int(index.episode_lengths.min()),
    }


def sample_eval_examples(
    dataset,
    index: DatasetIndex,
    *,
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    analysis = analyze_offset(index, goal_offset_steps)
    valid_indices = analysis["valid_indices"]

    if analysis["eligible_episodes"] < num_eval:
        raise ValueError(
            f"Offset {goal_offset_steps} only leaves "
            f"{analysis['eligible_episodes']} eligible episodes, which is fewer "
            f"than --num-eval={num_eval}."
        )

    if analysis["valid_start_points"] < num_eval:
        raise ValueError(
            f"Offset {goal_offset_steps} only leaves "
            f"{analysis['valid_start_points']} valid starting points, which is "
            f"fewer than --num-eval={num_eval}."
        )

    rng = np.random.default_rng(seed)
    population = len(valid_indices) - 1
    if population < num_eval:
        population = len(valid_indices)

    sampled_offsets = rng.choice(population, size=num_eval, replace=False)
    sampled_rows = np.sort(valid_indices[sampled_offsets])

    sampled = dataset.get_row_data(sampled_rows)
    eval_episodes = sampled[index.col_name]
    eval_start_steps = sampled["step_idx"]
    return sampled_rows, eval_episodes, eval_start_steps, analysis
