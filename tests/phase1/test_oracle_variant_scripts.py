from __future__ import annotations

import unittest

from scripts.phase1.analyze_oracle_full_variant_comparison import label_delta
from scripts.phase1.eval_oracle_cem_only_variant import (
    CEM_ONLY_ACTION_COUNTS,
    default_output_path,
    default_v3_source_path,
    row_from_cells,
)


class OracleVariantScriptTest(unittest.TestCase):
    def test_cem_only_action_counts_match_locked_subset(self) -> None:
        self.assertEqual(CEM_ONLY_ACTION_COUNTS["data"], 0)
        self.assertEqual(CEM_ONLY_ACTION_COUNTS["smooth_random"], 0)
        self.assertEqual(CEM_ONLY_ACTION_COUNTS["CEM_early"], 20)
        self.assertEqual(CEM_ONLY_ACTION_COUNTS["CEM_late"], 20)

    def test_row_paths_are_requested_outputs_and_existing_v3_sources(self) -> None:
        self.assertEqual(
            default_output_path("V1", "D2").as_posix().split("results/phase1/")[-1],
            "v1_oracle_ablation/v1_d2.json",
        )
        self.assertEqual(
            default_v3_source_path("D2").as_posix().split("results/phase1/")[-1],
            "d2_oracle_ablation/d2_oracle_V3.json",
        )

    def test_row_from_cells_requires_single_d_row(self) -> None:
        self.assertEqual(row_from_cells(["D1xR0", "D1xR3"]), "D1")
        with self.assertRaises(ValueError):
            row_from_cells(["D1xR0", "D2xR0"])

    def test_who_wins_threshold_labels(self) -> None:
        self.assertEqual(label_delta(0.20, "A", "B"), "B++ (+20.00 pp)")
        self.assertEqual(label_delta(0.05, "A", "B"), "B+ (+5.00 pp)")
        self.assertEqual(label_delta(0.049, "A", "B"), "tie (+4.90 pp)")
        self.assertEqual(label_delta(-0.05, "A", "B"), "A+ (-5.00 pp)")
        self.assertEqual(label_delta(-0.20, "A", "B"), "A++ (-20.00 pp)")


if __name__ == "__main__":
    unittest.main()
