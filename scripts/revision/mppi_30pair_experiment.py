#!/usr/bin/env python3
"""Phase D 30-pair simulator-scored PushT MPPI experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
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
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.stage1a_controls import iso_now, jsonable  # noqa: E402
from scripts.revision.mppi_planner import (  # noqa: E402
    DEFAULT_RERANK_PATH,
    NUM_SAMPLES,
    aggregate_run_records,
    build_scored_mppi_pool,
    get_git_commit,
    mppi_config_metadata,
    pool_summary_record,
    seconds_to_hms,
    select_stage_a_pairs,
)


DEFAULT_SWEEP_PATH = PROJECT_ROOT / "results" / "revision" / "mppi_temperature_sweep.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "mppi_pusht_30pair.json"
DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "revision" / "mppi_pusht_pools"
DEFAULT_SEEDS = (0, 1, 2)


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--rerank-path", type=Path, default=DEFAULT_RERANK_PATH)
    parser.add_argument("--sweep-path", type=Path, default=DEFAULT_SWEEP_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--list-pairs", action="store_true")
    return parser.parse_args()


def load_selected_temperature(*, sweep_path: Path, override: float | None) -> float:
    if override is not None:
        temperature = float(override)
    else:
        data = json.loads(sweep_path.read_text())
        temperature = float(data["summary"]["selected_temperature"])
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return temperature


def temperature_slug(temperature: float) -> str:
    return f"{float(temperature):g}".replace(".", "p").replace("-", "m")


def pool_path(pool_dir: Path, *, pair_id: int, seed: int, temperature: float) -> Path:
    return pool_dir / f"pair_{int(pair_id):03d}_seed_{int(seed)}_tau_{temperature_slug(temperature)}.pt"


def print_selected_pairs(selected_pairs: list[dict[str, Any]]) -> None:
    print("Selected MPPI 30-pair PushT subset")
    print("Pair | Cell  | Primary subset     | Memberships")
    print("-----+-------+--------------------+-------------------------------")
    for pair in selected_pairs:
        print(
            f"{int(pair['pair_id']):<4} | {str(pair['cell']):<5} | "
            f"{str(pair['primary_subset']):<18} | {','.join(pair['subset_memberships'])}"
        )


def load_existing_records(output_path: Path, *, resume: bool, expected_keys: set[tuple[int, int]]) -> list[dict[str, Any]]:
    if not resume or not output_path.exists():
        return []
    data = json.loads(output_path.read_text())
    records = []
    for record in data.get("records", []):
        key = (int(record.get("pair_id", -1)), int(record.get("seed", -1)))
        if key in expected_keys and record.get("pool_path") and Path(record["pool_path"]).exists():
            records.append(record)
    return records


def print_summary(summary: dict[str, Any]) -> None:
    print("\nMPPI 30-pair aggregate summary")
    print(f"records: {summary['n_records']}")
    print(f"rank1_success_mean: {summary['rank1_success']['mean']}")
    print(f"selection_regret_mean: {summary['selection_regret']['mean']}")
    print(f"Rpool_Cmodel_mean: {summary['Rpool_Cmodel']['mean']}")
    print(f"pool_success_mass_mean: {summary['pool_success_mass']['mean']}")
    if summary.get("by_subset"):
        print("\nBy subset")
        for subset, stats in summary["by_subset"].items():
            print(
                f"  {subset}: n={stats['n_records']} "
                f"success={stats['rank1_success']['mean']} "
                f"regret={stats['selection_regret']['mean']} "
                f"Rpool={stats['Rpool_Cmodel']['mean']}"
            )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.rerank_path = args.rerank_path.expanduser().resolve()
    args.sweep_path = args.sweep_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pool_dir = args.pool_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    pairs_data, selected_pairs, memberships, anchor_definitions = select_stage_a_pairs(
        pairs_path=args.pairs_path,
        rerank_path=args.rerank_path,
    )
    print_selected_pairs(selected_pairs)
    if args.list_pairs:
        return 0
    temperature = load_selected_temperature(sweep_path=args.sweep_path, override=args.temperature)

    pair_metadata = pairs_data["metadata"]
    validate_requested_pair_offsets(selected_pairs, offset=int(pair_metadata["offset"]))
    dataset_path = Path(pair_metadata["dataset_path"])
    expected_keys = {
        (int(pair["pair_id"]), int(seed))
        for pair in selected_pairs
        for seed in args.seeds
    }
    records = load_existing_records(args.output, resume=not args.no_resume, expected_keys=expected_keys)
    seen = {(int(record["pair_id"]), int(record["seed"])) for record in records}

    print("\n== MPPI 30-pair experiment setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"rerank_path: {args.rerank_path}")
    print(f"sweep_path: {args.sweep_path}")
    print(f"output: {args.output}")
    print(f"pool_dir: {args.pool_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"device: {args.device}")
    print(f"temperature: {temperature}")
    print(f"seeds: {[int(seed) for seed in args.seeds]}")
    print(f"pairs: {len(selected_pairs)}")
    print(f"expected_runs: {len(expected_keys)}")
    print(f"expected_simulator_rollouts: {len(expected_keys) * NUM_SAMPLES}")
    print(f"resume_records: {len(records)}")

    torch.manual_seed(min(int(seed) for seed in args.seeds))
    np.random.seed(min(int(seed) for seed in args.seeds))

    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        argparse.Namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            num_samples=NUM_SAMPLES,
            var_scale=1.0,
            cem_iters=30,
            topk=30,
            seed=min(int(seed) for seed in args.seeds),
            horizon=5,
            receding_horizon=5,
            action_block=5,
            img_size=224,
        ),
        process,
    )
    model = policy.solver.model

    args.pool_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.time()
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair_spec in enumerate(selected_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            for seed_idx, seed in enumerate(args.seeds, start=1):
                seed = int(seed)
                if (pair_id, seed) in seen:
                    print(f"[pair {pair_idx}/30 seed {seed_idx}/3] pair_id={pair_id} seed={seed}: resume")
                    continue
                started = time.time()
                print(
                    f"[pair {pair_idx}/30 seed {seed_idx}/3] "
                    f"pair_id={pair_id} seed={seed}: MPPI + score 300"
                )
                pool = build_scored_mppi_pool(
                    pair_spec=pair_spec,
                    dataset=dataset,
                    policy=policy,
                    model=model,
                    env=env,
                    seed=seed,
                    temperature=temperature,
                )
                path = pool_path(args.pool_dir, pair_id=pair_id, seed=seed, temperature=temperature)
                torch.save(pool, path)
                record = pool_summary_record(
                    pool=pool,
                    primary_subset=str(pair_spec["primary_subset"]),
                    subset_memberships=list(pair_spec["subset_memberships"]),
                    pool_path=path,
                )
                records.append(record)
                seen.add((pair_id, seed))
                success_mass = record.get("pool_success_mass")
                rank1_success = record.get("rank1_success")
                print(
                    f"  saved {path}; rank1_success={rank1_success}; "
                    f"pool_success={success_mass}; elapsed={seconds_to_hms(time.time() - started)}"
                )

                partial_summary = aggregate_run_records(records)
                output = {
                    "metadata": {
                        "format": "pusht_mppi_30pair_v1",
                        "created_at": iso_now(),
                        "git_commit": get_git_commit(),
                        "script_path": str(Path(__file__).resolve()),
                        "pairs_path": str(args.pairs_path),
                        "rerank_path": str(args.rerank_path),
                        "sweep_path": str(args.sweep_path),
                        "output": str(args.output),
                        "pool_dir": str(args.pool_dir),
                        "checkpoint_dir": str(args.checkpoint_dir),
                        "dataset_path": str(dataset_path),
                        "device": args.device,
                        "temperature": temperature,
                        "seeds": [int(item) for item in args.seeds],
                        "mppi_config": mppi_config_metadata(temperature=temperature),
                        "selected_pairs": selected_pairs,
                        "anchor_definitions": anchor_definitions,
                        "membership_map": memberships,
                        "candidate_pool_scored_n": NUM_SAMPLES,
                    },
                    "records": records,
                    "summary": partial_summary,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    finally:
        env.close()

    summary = aggregate_run_records(records)
    total_wallclock = time.time() - total_started
    output = {
        "metadata": {
            "format": "pusht_mppi_30pair_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "pairs_path": str(args.pairs_path),
            "rerank_path": str(args.rerank_path),
            "sweep_path": str(args.sweep_path),
            "output": str(args.output),
            "pool_dir": str(args.pool_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_path": str(dataset_path),
            "device": args.device,
            "temperature": temperature,
            "seeds": [int(item) for item in args.seeds],
            "mppi_config": mppi_config_metadata(temperature=temperature),
            "selected_pairs": selected_pairs,
            "anchor_definitions": anchor_definitions,
            "membership_map": memberships,
            "candidate_pool_scored_n": NUM_SAMPLES,
            "runtime": {
                "wallclock_seconds": total_wallclock,
                "wallclock_hms": seconds_to_hms(total_wallclock),
            },
        },
        "records": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary(summary)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
