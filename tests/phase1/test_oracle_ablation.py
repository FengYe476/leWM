from __future__ import annotations

import unittest

import numpy as np

from lewm_audit.eval.oracle_cem import (
    ANGLE_SUCCESS_THRESHOLD_RAD,
    BLOCK_SUCCESS_THRESHOLD_PX,
    cem_with_oracle_cost,
    cost_v1_hinge,
    cost_v2_indicator,
    cost_v3_baseline,
)
from scripts.phase1.eval_d3_oracle_ablation import filter_pairs_by_cells


class TinyEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.state = np.zeros(7, dtype=np.float32)

    def reset(self, seed=None):
        self.state = np.zeros(7, dtype=np.float32)
        return self.state, {}

    def _set_state(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()

    def _get_obs(self):
        return self.state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.state[2] += action[0]
        return self.state, 0.0, False, False, {}

    def close(self):
        pass


class OracleAblationTest(unittest.TestCase):
    def test_cost_variants_on_hand_checked_states(self) -> None:
        goal = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0])
        success_state = np.array([0.0, 0.0, 15.0, 10.0, 0.1, 0.0, 0.0])
        outside_state = np.array(
            [
                0.0,
                0.0,
                10.0 + BLOCK_SUCCESS_THRESHOLD_PX + 1.0,
                10.0,
                ANGLE_SUCCESS_THRESHOLD_RAD + 0.1,
                0.0,
                0.0,
            ]
        )

        self.assertEqual(cost_v2_indicator(success_state, goal), 0.0)
        self.assertEqual(cost_v1_hinge(success_state, goal), 0.0)
        self.assertGreater(cost_v3_baseline(success_state, goal), 0.0)
        self.assertEqual(cost_v2_indicator(outside_state, goal), 1.0)
        self.assertGreater(cost_v1_hinge(outside_state, goal), 0.0)
        self.assertGreater(cost_v3_baseline(outside_state, goal), 0.0)

    def test_cem_with_oracle_cost_finds_synthetic_unimodal_optimum(self) -> None:
        init = np.zeros(7, dtype=np.float32)
        goal = np.zeros(7, dtype=np.float32)
        goal[2] = 1.0

        def cost_fn(state, goal_state):
            return float((state[2] - goal_state[2]) ** 2)

        result = cem_with_oracle_cost(
            TinyEnv,
            init,
            goal,
            cost_fn,
            n_samples=200,
            n_iters=8,
            n_elites=20,
            horizon=1,
            receding_horizon=1,
            action_block=1,
            rng=np.random.default_rng(0),
            action_dim=1,
        )

        self.assertLess(result["best_cost"], 0.01)
        self.assertGreater(result["best_action_seq"][0, 0], 0.9)
        self.assertLess(result["best_action_seq"][0, 0], 1.1)

    def test_cell_filter_respects_requested_cells(self) -> None:
        pairs = [
            {"pair_id": 0, "cell": "D2xR3"},
            {"pair_id": 1, "cell": "D3xR0"},
            {"pair_id": 2, "cell": "D3xR1"},
            {"pair_id": 3, "cell": "D0xR0"},
        ]

        selected = filter_pairs_by_cells(pairs, ["D3xR0", "D3xR1"])

        self.assertEqual([pair["pair_id"] for pair in selected], [1, 2])
        self.assertTrue(all(pair["cell"].startswith("D3") for pair in selected))


if __name__ == "__main__":
    unittest.main()
