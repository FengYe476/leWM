#!/usr/bin/env python3
"""Phase D MPPI temperature sweep on five PushT Track A pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import prepare_pair_info  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.stage1a_controls import clean_float, iso_now, jsonable  # noqa: E402
from scripts.revision.mppi_planner import (  # noqa: E402
    NUM_SAMPLES,
    get_git_commit,
    mppi_config_metadata,
    run_mppi,
    scalar_summary,
    seconds_to_hms,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "mppi_temperature_sweep.json"
DEFAULT_PAIR_IDS = (0, 20, 40, 60, 80)
DEFAULT_TEMPERATURES = (0.01, 0.1, 1.0, 10.0)


def parse_float_list(value: str) -> tuple[float, ...]:
    items = tuple(float(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("Temperatures must be positive")
    return tuple(dict.fromkeys(items))


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pair-ids", type=parse_int_list, default=DEFAULT_PAIR_IDS)
    parser.add_argument("--temperatures", type=parse_float_list, default=DEFAULT_TEMPERATURES)
    return parser.parse_args()


def summarize_by_temperature(records: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for temperature in sorted({float(record["temperature"]) for record in records}):
        group = [record for record in records if float(record["temperature"]) == temperature]
        out[str(temperature)] = {
            "n_pairs": int(len(group)),
            "mean_rank1_default_cost": clean_float(np.mean([record["rank1_default_cost"] for record in group])),
            "rank1_default_cost": scalar_summary([record["rank1_default_cost"] for record in group]),
            "rank1_below_pool_median_rate": scalar_summary([record["rank1_below_pool_median"] for record in group]),
            "pool_default_cost_median": scalar_summary([record["pool_default_cost_median"] for record in group]),
            "pool_default_cost_std": scalar_summary([record["pool_default_cost_std"] for record in group]),
            "final_weight_effective_sample_size": scalar_summary(
                [record["final_weight_effective_sample_size"] for record in group]
            ),
        }
    return out


def print_summary(summary: dict[str, Any], selected_temperature: float) -> None:
    rows = []
    for temperature, stats in sorted(summary.items(), key=lambda item: float(item[0])):
        rows.append(
            [
                temperature,
                str(stats["n_pairs"]),
                f"{stats['mean_rank1_default_cost']:.6f}",
                f"{stats['rank1_below_pool_median_rate']['mean']:.3f}",
                f"{stats['final_weight_effective_sample_size']['mean']:.2f}",
            ]
        )
    headers = ["tau", "N", "mean_rank1_cost", "below_median", "ESS"]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("\nMPPI temperature sweep summary")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    print(f"\nselected_temperature: {selected_temperature}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    pairs_data, all_pairs = load_pairs(args.pairs_path, max_pairs=None, pair_ids=None)
    by_id = {int(pair["pair_id"]): pair for pair in all_pairs}
    requested_pairs = [dict(by_id[int(pair_id)]) for pair_id in args.pair_ids]
    validate_requested_pair_offsets(requested_pairs, offset=int(pairs_data["metadata"]["offset"]))
    dataset_path = Path(pairs_data["metadata"]["dataset_path"])

    print("== MPPI temperature sweep setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"pair_ids: {[int(pair['pair_id']) for pair in requested_pairs]}")
    print(f"temperatures: {[float(item) for item in args.temperatures]}")
    print(f"candidate_pool_size: {NUM_SAMPLES}")

    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=int(args.seed),
        ),
        process,
    )
    model = policy.solver.model

    records: list[dict[str, Any]] = []
    started = time.time()
    total = len(requested_pairs) * len(args.temperatures)
    completed = 0
    for temperature in args.temperatures:
        for pair_idx, pair_spec in enumerate(requested_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            completed += 1
            print(
                f"[{completed}/{total}] tau={float(temperature):g} "
                f"pair {pair_idx}/{len(requested_pairs)} pair_id={pair_id}"
            )
            rows = dataset.get_row_data([int(pair_spec["start_row"]), int(pair_spec["goal_row"])])
            initial = {key: value[0] for key, value in rows.items()}
            goal = {key: value[1] for key, value in rows.items()}
            prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
            result = run_mppi(
                model=model,
                prepared_info=prepared_info,
                pair_id=pair_id,
                seed=int(args.seed),
                temperature=float(temperature),
            )
            costs = np.asarray(result["default_costs"], dtype=np.float64)
            rank1 = int(result["rank1_candidate_index"])
            records.append(
                {
                    "pair_id": pair_id,
                    "cell": str(pair_spec["cell"]),
                    "temperature": clean_float(float(temperature)),
                    "seed": int(args.seed),
                    "sampling_seed": int(args.seed) + pair_id * 1009,
                    "rank1_candidate_index": rank1,
                    "rank1_default_cost": clean_float(float(costs[rank1])),
                    "final_pool_default_costs": costs.astype(float).tolist(),
                    "rank1_below_pool_median": bool(costs[rank1] < float(np.median(costs))),
                    "pool_default_cost_min": clean_float(float(np.min(costs))),
                    "pool_default_cost_median": clean_float(float(np.median(costs))),
                    "pool_default_cost_mean": clean_float(float(np.mean(costs))),
                    "pool_default_cost_std": clean_float(float(np.std(costs, ddof=0))),
                    "pool_default_cost_dynamic_range": result["select_cost_dynamic_range"],
                    "top30_default_cost_std": result["top30_select_cost_std"],
                    "final_weight_entropy": result["final_weight_entropy"],
                    "final_weight_effective_sample_size": result["final_weight_effective_sample_size"],
                    "wallclock_seconds": result["wallclock_seconds"],
                }
            )

    summary_by_temperature = summarize_by_temperature(records)
    selected_temperature = min(
        (float(temperature) for temperature in args.temperatures),
        key=lambda temperature: float(summary_by_temperature[str(float(temperature))]["mean_rank1_default_cost"]),
    )
    output = {
        "metadata": {
            "format": "pusht_mppi_temperature_sweep_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "pairs_path": str(args.pairs_path),
            "selected_pair_ids": [int(pair["pair_id"]) for pair in requested_pairs],
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_path": str(dataset_path),
            "device": args.device,
            "seed": int(args.seed),
            "temperatures": [float(item) for item in args.temperatures],
            "mppi_config_template": mppi_config_metadata(temperature=float(args.temperatures[0])),
            "selection_rule": "lowest mean rank-1 default latent cost over the five sweep pairs",
            "simulator_scoring": False,
            "runtime": {
                "wallclock_seconds": clean_float(time.time() - started),
                "wallclock_hms": seconds_to_hms(time.time() - started),
            },
        },
        "records": records,
        "summary": {
            "by_temperature": summary_by_temperature,
            "selected_temperature": clean_float(selected_temperature),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary(summary_by_temperature, selected_temperature)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
