from __future__ import annotations

import math
import unittest

from lewm_audit.diagnostics.cost_magnitudes import (
    infer_c_real_state_success_threshold,
    per_row_cost_stats,
)
from lewm_audit.diagnostics.failure_modes import classify_pairs


class TrackASupplementaryTest(unittest.TestCase):
    def test_classify_pairs_boundary_values(self) -> None:
        records = [
            {
                "pair_id": 0,
                "cell": "D0xR0",
                "rho": -0.1,
                "block_displacement_px": 5.0,
                "required_rotation_rad": 0.1,
            },
            {
                "pair_id": 1,
                "cell": "D1xR1",
                "rho": 0.0,
                "block_displacement_px": 25.0,
                "required_rotation_rad": 0.5,
            },
            {
                "pair_id": 2,
                "cell": "D2xR2",
                "rho": 0.3,
                "block_displacement_px": 80.0,
                "required_rotation_rad": 1.0,
            },
        ]
        counts = {0: 0, 1: 1, 2: 80}

        classified = classify_pairs(records, counts)

        self.assertEqual(classified.loc[0, "success_class"], "all_fail")
        self.assertEqual(classified.loc[0, "encoder_class"], "neg_rho")
        self.assertEqual(classified.loc[1, "success_class"], "some_succ")
        self.assertEqual(classified.loc[1, "encoder_class"], "weak_rho")
        self.assertEqual(classified.loc[2, "success_class"], "all_succ")
        self.assertEqual(classified.loc[2, "encoder_class"], "strong_rho")

    def test_per_row_cost_stats_empty_match_is_well_formed(self) -> None:
        records_by_pair = {
            0: {
                "cell": "D0xR0",
                "actions": [
                    {
                        "C_real_z": 1.0,
                        "C_model": 2.0,
                        "C_real_state": 3.0,
                        "success": False,
                    }
                ],
            }
        }

        stats = per_row_cost_stats(records_by_pair, lambda pair: pair["cell"].startswith("D9"))

        self.assertEqual(stats["n_pairs"], 0)
        self.assertEqual(stats["n_records"], 0)
        self.assertIsNone(stats["C_real_z"]["mean"])
        self.assertIsNone(stats["C_model"]["median"])
        self.assertIsNone(stats["C_real_state"]["iqr"])
        self.assertEqual(stats["best_C_real_state_per_pair"], [])
        self.assertEqual(stats["n_pairs_with_min_C_real_state_below_success_threshold"], 0)

    def test_inferred_success_threshold_is_finite_and_positive(self) -> None:
        records_by_pair = {
            0: {
                "actions": [
                    {"C_real_state": 2.5, "success": False},
                    {"C_real_state": 1.25, "success": True},
                ]
            },
            1: {
                "actions": [
                    {"C_real_state": 3.0, "success": True},
                ]
            },
        }

        threshold = infer_c_real_state_success_threshold(records_by_pair)

        self.assertTrue(math.isfinite(threshold["threshold"]))
        self.assertGreater(threshold["threshold"], 0.0)
        self.assertEqual(threshold["threshold"], 3.0)
        self.assertFalse(threshold["unique_scalar_threshold"])


if __name__ == "__main__":
    unittest.main()
