from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lewm_audit.sampling.track_a_pairs import assign_cells, sample_stratified_pairs


DISPLACEMENT_EDGES = [0.0, 10.0, 50.0, 120.0, float("inf")]
ROTATION_EDGES = [0.0, 0.25, 0.75, 1.25, float("inf")]
OFFSET = 50


def make_synthetic_pool() -> pd.DataFrame:
    displacement_centers = [5.0, 25.0, 80.0, 150.0]
    rotation_centers = [0.1, 0.5, 1.0, 1.5]
    rows = []
    for d_idx, displacement in enumerate(displacement_centers):
        for r_idx, rotation in enumerate(rotation_centers):
            for episode_slot in range(3):
                episode_id = d_idx * 1000 + r_idx * 100 + episode_slot
                episode_start = episode_id * 1000
                episode_end = episode_start + 240
                for local_row in range(2):
                    start_row = episode_start + local_row * 60
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "step_idx": local_row * 60,
                            "start_row": start_row,
                            "goal_row": start_row + OFFSET,
                            "episode_start_row": episode_start,
                            "episode_end_row": episode_end,
                            "block_displacement_px": displacement,
                            "required_rotation_rad": rotation,
                            "physical_pose_distance": displacement + rotation,
                        }
                    )
    return assign_cells(pd.DataFrame(rows), DISPLACEMENT_EDGES, ROTATION_EDGES)


class TrackASamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = make_synthetic_pool()
        self.budget = {f"D{d}xR{r}": 2 for d in range(4) for r in range(4)}

    def sample(self, seed: int):
        return sample_stratified_pairs(
            self.pool,
            budget_matrix=self.budget,
            per_episode_per_cell_limit=1,
            rng=np.random.default_rng(seed),
        )

    def test_same_seed_produces_identical_pair_mapping(self) -> None:
        first, _ = self.sample(seed=123)
        second, _ = self.sample(seed=123)
        first_mapping = first[["pair_id", "episode_id", "start_row"]].to_dict(orient="records")
        second_mapping = second[["pair_id", "episode_id", "start_row"]].to_dict(orient="records")
        self.assertEqual(first_mapping, second_mapping)

    def test_per_episode_per_cell_limit_is_respected(self) -> None:
        sampled, _ = self.sample(seed=456)
        counts = sampled.groupby(["cell", "episode_id"]).size()
        self.assertTrue((counts <= 1).all())

    def test_sampled_pairs_round_trip_to_labeled_cell(self) -> None:
        sampled, _ = self.sample(seed=789)
        reassigned = assign_cells(
            sampled.drop(columns=["cell", "cell_d", "cell_r"]),
            DISPLACEMENT_EDGES,
            ROTATION_EDGES,
        )
        self.assertEqual(sampled["cell"].tolist(), reassigned["cell"].tolist())

    def test_sampled_pairs_satisfy_offset_validity(self) -> None:
        sampled, _ = self.sample(seed=987)
        self.assertTrue(((sampled["goal_row"] - sampled["start_row"]) == OFFSET).all())
        self.assertTrue((sampled["goal_row"] <= sampled["episode_end_row"]).all())
        self.assertTrue((sampled["start_row"] >= sampled["episode_start_row"]).all())


if __name__ == "__main__":
    unittest.main()
