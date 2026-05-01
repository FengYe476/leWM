from __future__ import annotations

import math
import unittest

from scripts.phase1.eval_d3_oracle_ablation import (
    filter_pairs_by_cells,
    relative_failure_reduction,
)


class D0OracleRunTest(unittest.TestCase):
    def test_relative_failure_reduction_formula(self) -> None:
        self.assertAlmostEqual(relative_failure_reduction(0.5, 0.75), 0.5)
        self.assertAlmostEqual(relative_failure_reduction(0.0, 1.0), 0.0)
        self.assertAlmostEqual(relative_failure_reduction(0.0, 0.0), 1.0)
        self.assertIsNone(relative_failure_reduction(1.0, 1.0))
        self.assertTrue(math.isinf(relative_failure_reduction(1.0, 0.9)))

    def test_cell_filter_restricts_to_d0_only(self) -> None:
        pairs = [
            {"pair_id": 0, "cell": "D0xR0"},
            {"pair_id": 1, "cell": "D0xR1"},
            {"pair_id": 2, "cell": "D3xR0"},
            {"pair_id": 3, "cell": "D1xR2"},
        ]

        selected = filter_pairs_by_cells(pairs, ["D0xR0", "D0xR1"])

        self.assertEqual([pair["pair_id"] for pair in selected], [0, 1])
        self.assertTrue(all(pair["cell"].startswith("D0") for pair in selected))


if __name__ == "__main__":
    unittest.main()
