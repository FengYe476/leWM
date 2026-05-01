from __future__ import annotations

import unittest

from lewm_audit.diagnostics.dp1 import dp1_test, sign_reversal_cluster
from lewm_audit.diagnostics.heatmap import cell_grid_from_records


DISPLACEMENT_EDGES = [0.0, 10.0, 50.0, 120.0, float("inf")]
ROTATION_EDGES = [0.0, 0.25, 0.75, 1.25, float("inf")]


def make_pair(pair_id: int, cell: str, c_real_state: list[float]) -> dict:
    actions = [
        {"C_real_z": float(idx), "C_real_state": float(value), "success": False}
        for idx, value in enumerate(c_real_state)
    ]
    return {
        "pair_id": pair_id,
        "cell": cell,
        "episode_id": pair_id + 10,
        "start_row": pair_id * 100,
        "goal_row": pair_id * 100 + 50,
        "block_displacement_px": 25.0,
        "required_rotation_rad": 0.5,
        "actions": actions,
    }


class TrackAAnalysisTest(unittest.TestCase):
    def test_dp1_verdict_logic(self) -> None:
        passing = dp1_test([-1.0, -0.9, 0.9, 1.0] * 8, n_bootstrap=2000, rng_seed=1)
        failing = dp1_test([0.00, 0.01, 0.02, 0.03] * 8, n_bootstrap=2000, rng_seed=1)
        ambiguous = dp1_test([0.0, 0.0, 0.0, 1.0] * 2, n_bootstrap=2000, rng_seed=1)

        self.assertEqual(passing["verdict"], "pass")
        self.assertEqual(failing["verdict"], "fail")
        self.assertEqual(ambiguous["verdict"], "ambiguous")

    def test_cell_grid_from_records_assigns_expected_cells(self) -> None:
        records = [
            {"pair_id": 0, "block_displacement_px": 5.0, "required_rotation_rad": 0.1},
            {"pair_id": 1, "block_displacement_px": 25.0, "required_rotation_rad": 0.5},
            {"pair_id": 2, "block_displacement_px": 80.0, "required_rotation_rad": 1.0},
            {"pair_id": 3, "block_displacement_px": 150.0, "required_rotation_rad": 1.5},
        ]
        grid = cell_grid_from_records(records, DISPLACEMENT_EDGES, ROTATION_EDGES)

        self.assertEqual([record["pair_id"] for record in grid["D0xR0"]], [0])
        self.assertEqual([record["pair_id"] for record in grid["D1xR1"]], [1])
        self.assertEqual([record["pair_id"] for record in grid["D2xR2"]], [2])
        self.assertEqual([record["pair_id"] for record in grid["D3xR3"]], [3])

    def test_sign_reversal_cluster_sorted_by_rho(self) -> None:
        pairs = {
            0: make_pair(0, "D1xR1", [0.0, 1.0, 2.0, 3.0]),
            1: make_pair(1, "D1xR2", [3.0, 2.0, 1.0, 0.0]),
            2: make_pair(2, "D2xR3", [3.0, 1.0, 2.0, 0.0]),
        }

        cluster = sign_reversal_cluster(pairs)

        self.assertEqual([item["pair_id"] for item in cluster], [1, 2])
        self.assertLess(cluster[0]["rho"], cluster[1]["rho"])
        self.assertLess(cluster[1]["rho"], 0.0)


if __name__ == "__main__":
    unittest.main()
