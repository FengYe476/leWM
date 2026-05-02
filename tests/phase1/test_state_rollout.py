from __future__ import annotations

import unittest

import numpy as np

from lewm_audit.eval.oracle_cem import rollout_final_state
from lewm_audit.eval.state_rollout import rollout_with_state_capture


class TinyEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.state = np.zeros(7, dtype=np.float32)
        self.goal_state = np.zeros(7, dtype=np.float32)

    def reset(self, seed=None):
        self.state = np.zeros(7, dtype=np.float32)
        return self.state, {}

    def _set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state, dtype=np.float32).copy()

    def _set_state(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()

    def _get_obs(self):
        return self.state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.state[0:2] += action
        self.state[2:4] += 0.5 * action
        self.state[4] += 0.01 * float(action[0])
        return self.state.copy(), 0.0, False, False, {}

    def close(self):
        pass


class StateRolloutTest(unittest.TestCase):
    def test_rollout_shapes_include_initial_state(self) -> None:
        init = np.zeros(7, dtype=np.float32)
        goal = np.zeros(7, dtype=np.float32)
        actions = np.ones((4, 2), dtype=np.float32)

        result = rollout_with_state_capture(TinyEnv, init, goal, actions)

        self.assertEqual(result["states"].shape, (5, 7))
        self.assertEqual(result["block_xy"].shape, (5, 2))
        self.assertEqual(result["agent_xy"].shape, (5, 2))
        self.assertEqual(result["block_angle"].shape, (5,))
        self.assertEqual(result["step_success"].shape, (5,))
        self.assertIsInstance(result["final_success"], bool)

    def test_final_state_matches_existing_oracle_rollout_path(self) -> None:
        init = np.array([1.0, 2.0, 3.0, 4.0, 0.2, 0.0, 0.0], dtype=np.float32)
        goal = np.array([0.0, 0.0, 8.0, 9.0, 0.25, 0.0, 0.0], dtype=np.float32)
        actions = np.array([[1.0, 0.0], [0.0, -2.0], [0.5, 1.0]], dtype=np.float32)

        captured = rollout_with_state_capture(TinyEnv, init, goal, actions, seed=7)
        final = rollout_final_state(TinyEnv(), init, goal, actions, seed=7)

        np.testing.assert_allclose(captured["states"][-1], final)


if __name__ == "__main__":
    unittest.main()
