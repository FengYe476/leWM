#!/usr/bin/env python3
"""Combine Stage 1A C0-C5 and C6-C7 outputs into one summary artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.stage1.stage1a_c6_c7 import print_summary_rows  # noqa: E402
from scripts.phase2.stage1.stage1a_controls import iso_now, jsonable  # noqa: E402


DEFAULT_C0_C5 = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_c0_c5.json"
DEFAULT_C6_C7 = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_c6_c7.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_full.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-c5", type=Path, default=DEFAULT_C0_C5)
    parser.add_argument("--c6-c7", type=Path, default=DEFAULT_C6_C7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    return json.loads(path.read_text())


def parse_dim(config: str, prefix: str) -> int:
    if not config.startswith(prefix):
        return 10**9
    try:
        return int(config.split("=", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def row_order_key(row: dict[str, Any]) -> tuple[int, int, str]:
    control = str(row.get("control", ""))
    config = str(row.get("config", ""))
    if control == "C0":
        return (0, 0, config)
    if control == "C1":
        return (1, 0, config)
    if control == "C2":
        return (2, parse_dim(config, "gaussian_m="), config)
    if control == "C3":
        return (3, parse_dim(config, "coords_m="), config)
    if control == "C4":
        return (4, 0, config)
    if control == "C5":
        return (5, 0, config)
    if control == "C6":
        return (6, 0, config)
    if control == "C7_cls":
        return (7, 0, config)
    if control == "C7_mean":
        return (8, 0, config)
    return (99, 0, f"{control}:{config}")


def main() -> int:
    args = parse_args()
    args.c0_c5 = args.c0_c5.expanduser().resolve()
    args.c6_c7 = args.c6_c7.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    c0_c5 = load_json(args.c0_c5)
    c6_c7 = load_json(args.c6_c7)
    rows = list(c0_c5.get("summary_table", [])) + list(c6_c7.get("summary_table", []))
    rows = sorted(rows, key=row_order_key)

    controls = {}
    controls.update(c0_c5.get("controls", {}))
    controls.update(c6_c7.get("controls", {}))

    output = {
        "metadata": {
            "format": "stage1a_full_controls",
            "created_at": iso_now(),
            "inputs": {
                "c0_c5": str(args.c0_c5),
                "c6_c7": str(args.c6_c7),
            },
            "row_order": [
                "C0",
                "C1",
                "C2 gaussian_m=1",
                "C2 gaussian_m=2",
                "C2 gaussian_m=4",
                "C2 gaussian_m=8",
                "C2 gaussian_m=16",
                "C2 gaussian_m=32",
                "C2 gaussian_m=64",
                "C2 gaussian_m=128",
                "C2 gaussian_m=192",
                "C3 coords_m=8",
                "C3 coords_m=16",
                "C3 coords_m=32",
                "C3 coords_m=64",
                "C4",
                "C5",
                "C6",
                "C7_cls",
                "C7_mean",
            ],
            "c0_c5_metadata": c0_c5.get("metadata", {}),
            "c6_c7_metadata": c6_c7.get("metadata", {}),
        },
        "summary_table": rows,
        "controls": controls,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_rows(rows, title="Stage 1A full summary")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
