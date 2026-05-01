#!/usr/bin/env python3
"""Analyze three-cost attribution results for LeWM PushT long-goal failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lewm_audit.diagnostics.three_cost import (
    analyze_three_cost_records,
    build_report,
    load_records,
    make_plots,
    to_jsonable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "three_cost_offset50.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "three_cost_analysis.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LeWM three-cost attribution JSON results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--low-corr-threshold",
        type=float,
        default=0.5,
        help="Correlation threshold used by the preliminary decision tree.",
    )
    parser.add_argument(
        "--planner-majority-threshold",
        type=float,
        default=0.5,
        help=(
            "Case D triggers when the fraction of pairs with zero successful "
            "actions is greater than this value."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, records = load_records(args.input)
    results = analyze_three_cost_records(
        data,
        records,
        input_path=args.input,
        output_path=args.output,
        low_corr_threshold=args.low_corr_threshold,
        planner_majority_threshold=args.planner_majority_threshold,
    )

    figure_paths = []
    if not args.no_plots:
        figure_paths = make_plots(records, results["correlations"], args.figures_dir)
    results["figure_paths"] = figure_paths

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(results), indent=2, allow_nan=False) + "\n")

    print(build_report(data, results))


if __name__ == "__main__":
    main()
