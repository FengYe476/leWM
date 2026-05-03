#!/usr/bin/env python3
"""Cube full projected-CEM multi-seed anomaly sanity check.

This extends the Block 1.2 Cube full projected-CEM run with projection seeds 1
and 2 for the dimensions needed to judge the m=32 vs m=192 inverted-U.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_cube_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_NAME,
    get_dataset,
)
from eval_pusht_baseline import DEFAULT_CACHE_DIR, build_policy, build_processors, resolve_device  # noqa: E402
from scripts.phase2.cube.extract_cube_latents import (  # noqa: E402
    ACTION_BLOCK,
    IMG_SIZE,
    NUM_SAMPLES,
    TOPK,
    VAR_SCALE,
    infer_raw_action_dim,
    load_pair_rows,
    load_pairs,
    prepare_pair_info,
    validate_requested_pair_offsets,
)
from scripts.phase2.cube.cube_stage1b import DEFAULT_CEM_ITERS  # noqa: E402
from scripts.phase2.protocol_match.cube_full_proj_cem import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    DEFAULT_STAGE1B_PATH,
    aggregate_records,
    blocked_batch_to_raw,
    build_projected_record,
    load_json,
    load_stage1b_reference,
    make_policy_namespace,
    make_projection,
    projected_costs,  # noqa: F401 - imported to make the full-projected hook provenance explicit.
    record_key,
    run_projected_cem,
    score_raw_actions,
    seconds_to_hms,
)
from scripts.phase2.stage1.stage1a_controls import clean_float, jsonable  # noqa: E402


DEFAULT_BLOCK12_PATH = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "cube_full_proj_cem.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "subspace_cem" / "cube_anomaly_sanity.json"
DEFAULT_DIMS = (32, 64, 192)
DEFAULT_PROJECTION_SEEDS = (1, 2)
ALGORITHM = "cube_full_projected_cem"


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--stage1b-path", type=Path, default=DEFAULT_STAGE1B_PATH)
    parser.add_argument("--block12-path", type=Path, default=DEFAULT_BLOCK12_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=0, help="Base CEM sampling seed; projection seeds are separate.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimensions", type=parse_int_list, default=DEFAULT_DIMS)
    parser.add_argument("--projection-seeds", type=parse_int_list, default=DEFAULT_PROJECTION_SEEDS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--action-block", type=int, default=ACTION_BLOCK)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--cem-iters", type=int, default=DEFAULT_CEM_ITERS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--var-scale", type=float, default=VAR_SCALE)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if tuple(int(dim) for dim in args.dimensions) != DEFAULT_DIMS:
        parser.error(f"--dimensions is locked to {DEFAULT_DIMS} for this sanity check")
    if tuple(int(seed) for seed in args.projection_seeds) != DEFAULT_PROJECTION_SEEDS:
        parser.error(f"--projection-seeds is locked to {DEFAULT_PROJECTION_SEEDS} for this sanity check")
    if int(args.seed) != 0:
        parser.error("--seed is locked to 0 so only projection seeds vary from Block 1.2")
    if int(args.action_block) <= 0:
        parser.error("--action-block must be positive")
    if int(args.num_samples) <= 0:
        parser.error("--num-samples must be positive")
    if int(args.topk) < 1 or int(args.topk) > int(args.num_samples):
        parser.error("--topk must be in [1, --num-samples]")
    if int(args.cem_iters) <= 0:
        parser.error("--cem-iters must be positive")
    return args


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def load_existing_records(path: Path, *, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    data = load_json(path)
    if data.get("config", {}).get("algorithm") != ALGORITHM:
        raise RuntimeError(f"Cannot resume from unexpected output algorithm: {data.get('config', {}).get('algorithm')!r}")
    return list(data.get("records", []))


def select_block12_pairs(
    *,
    pairs_path: Path,
    block12_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    block12 = load_json(block12_path)
    pair_ids = [int(pair_id) for pair_id in block12.get("metadata", {}).get("selected_pair_ids", [])]
    if len(pair_ids) != 25:
        raise RuntimeError(f"Expected 25 Block 1.2 selected pairs, got {len(pair_ids)}")

    pairs_data, all_pairs = load_pairs(pairs_path, max_pairs=None, pair_ids=None)
    by_id = {int(pair["pair_id"]): pair for pair in all_pairs}
    missing = sorted(set(pair_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"Block 1.2 selected pair IDs missing from cube_pairs.json: {missing[:10]}")
    return pairs_data, [by_id[pair_id] for pair_id in pair_ids], block12


def add_aliases(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    record["m"] = int(record["dimension"])
    record["seed"] = int(record["projection_seed"])
    record["rank1_C_real_state"] = record["rank1_c_real_state"]
    return record


def success_rate(
    records: list[dict[str, Any]],
    *,
    dim: int,
    projection_seed: int,
    expected_n: int,
) -> tuple[float | None, int]:
    values = [
        bool(record["rank1_success"])
        for record in records
        if int(record["dimension"]) == int(dim) and int(record["projection_seed"]) == int(projection_seed)
    ]
    if len(values) != int(expected_n):
        return None, int(len(values))
    return clean_float(float(np.mean(values))), int(len(values))


def build_summary_and_decision(
    *,
    block12_records: list[dict[str, Any]],
    records: list[dict[str, Any]],
    dimensions: tuple[int, ...],
    projection_seeds: tuple[int, ...],
    n_pairs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary: dict[str, Any] = {}
    for dim in dimensions:
        dim_key = f"m_{int(dim)}"
        seed0, seed0_n = success_rate(block12_records, dim=int(dim), projection_seed=0, expected_n=n_pairs)
        seed_rates: dict[str, float | None] = {}
        seed_counts = {"seed_0": seed0_n}
        for seed in projection_seeds:
            rate, count = success_rate(records, dim=int(dim), projection_seed=int(seed), expected_n=n_pairs)
            seed_rates[f"seed_{int(seed)}"] = rate
            seed_counts[f"seed_{int(seed)}"] = count
        rates = [seed0, *seed_rates.values()]
        finite_rates = [float(rate) for rate in rates if rate is not None and math.isfinite(float(rate))]
        summary[dim_key] = {
            "seed_0": seed0,
            **seed_rates,
            "mean_with_seed0": clean_float(float(np.mean(finite_rates))) if len(finite_rates) == 3 else None,
            "n_seeds_in_mean": int(len(finite_rates)),
            "record_counts": seed_counts,
        }

    m32 = summary["m_32"]["mean_with_seed0"]
    m192 = summary["m_192"]["mean_with_seed0"]
    gap = None if m32 is None or m192 is None else clean_float(float(m32) - float(m192))
    if gap is None:
        verdict = None
        stable = None
    elif float(gap) >= 0.10:
        verdict = "STABLE"
        stable = True
    elif float(gap) < 0.05:
        verdict = "NOISE"
        stable = False
    else:
        verdict = "MIXED"
        stable = None

    decision = {
        "inverted_u_stable": stable,
        "m32_minus_m192_3seed_mean": gap,
        "verdict": verdict,
    }
    return summary, decision


def write_output(
    *,
    output_path: Path,
    dimensions: tuple[int, ...],
    projection_seeds: tuple[int, ...],
    selected_pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    decision: dict[str, Any],
    total_started: float,
) -> None:
    records = sorted(
        [add_aliases(record) for record in records],
        key=lambda record: (int(record["pair_id"]), int(record["dimension"]), int(record["projection_seed"])),
    )
    output = {
        "config": {
            "dims": [int(dim) for dim in dimensions],
            "seeds": [int(seed) for seed in projection_seeds],
            "n_pairs": int(len(selected_pairs)),
            "algorithm": ALGORITHM,
            "note": "Seeds 1,2 only. Seed 0 already in cube_full_proj_cem.json",
        },
        "records": records,
        "summary": summary,
        "decision": decision,
        "wall_clock_seconds": clean_float(time.time() - total_started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")


def print_decision_table(summary: dict[str, Any], decision: dict[str, Any]) -> None:
    rows = []
    for dim in DEFAULT_DIMS:
        item = summary[f"m_{dim}"]
        rows.append(
            [
                f"m={dim}",
                fmt(item.get("seed_0")),
                fmt(item.get("seed_1")),
                fmt(item.get("seed_2")),
                fmt(item.get("mean_with_seed0")),
            ]
        )
    headers = ["Dim", "Seed0", "Seed1", "Seed2", "3-seed mean"]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print("\nCube anomaly decision table")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print(f"\nm32_minus_m192_3seed_mean: {fmt(decision['m32_minus_m192_3seed_mean'])}")
    print(f"verdict: {decision['verdict']}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.stage1b_path = args.stage1b_path.expanduser().resolve()
    args.block12_path = args.block12_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.device = resolve_device(args.device)
    if args.device == "mps":
        raise RuntimeError("Cube anomaly sanity is intended for CUDA/RunPod runs; pass --device cuda.")

    dimensions = tuple(int(dim) for dim in args.dimensions)
    projection_seeds = tuple(int(seed) for seed in args.projection_seeds)

    pairs_data, selected_pairs, block12 = select_block12_pairs(
        pairs_path=args.pairs_path,
        block12_path=args.block12_path,
    )
    pairs_metadata = pairs_data.get("metadata", {})
    offset = int(pairs_metadata.get("offset", selected_pairs[0]["goal_row"] - selected_pairs[0]["start_row"]))
    if offset % int(args.action_block) != 0:
        raise ValueError("Cube offset must be divisible by --action-block")
    validate_requested_pair_offsets(selected_pairs, offset=offset)

    # CEM parameters intentionally mirror cube_full_proj_cem.py:
    # CLI defaults come from its lines 107-112, and it derives
    # horizon_blocks as offset // action_block on lines 751-755.
    args.horizon_blocks = int(offset // int(args.action_block))

    stage1b_baselines, endpoint_reference_records, stage1b_metadata = load_stage1b_reference(args.stage1b_path)
    default_baselines = [stage1b_baselines[int(pair["pair_id"])] for pair in selected_pairs]
    missing_refs = [
        (int(pair["pair_id"]), int(dim), int(seed))
        for pair in selected_pairs
        for dim in dimensions
        for seed in projection_seeds
        if (int(pair["pair_id"]), int(dim), int(seed)) not in endpoint_reference_records
    ]
    if missing_refs:
        raise RuntimeError(f"cube_stage1b.json is missing endpoint reference records: examples={missing_refs[:10]}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    raw_action_dim = infer_raw_action_dim(dataset)
    if raw_action_dim != 5:
        raise ValueError(f"Expected Cube raw action dim 5, got {raw_action_dim}")
    process = build_processors(dataset, ["action"])
    policy = build_policy(make_policy_namespace(args), process)
    model = policy.solver.model
    action_processor = policy.process["action"]

    projections = {
        (int(dim), int(seed)): make_projection(int(dim), int(seed))
        for dim in dimensions
        for seed in projection_seeds
    }

    expected_keys = {
        (int(pair["pair_id"]), int(dim), int(seed))
        for pair in selected_pairs
        for dim in dimensions
        for seed in projection_seeds
    }
    records = load_existing_records(args.output, resume=not args.no_resume)
    records = [record for record in records if record_key(record) in expected_keys]
    seen = {record_key(record) for record in records}
    duplicate_count = len(records) - len(seen)
    if duplicate_count:
        raise RuntimeError(f"Existing output contains {duplicate_count} duplicate record keys")

    block12_runtime = float(block12.get("metadata", {}).get("runtime", {}).get("wallclock_seconds", 316.09826016426086))
    block12_records = block12.get("records", [])
    estimate_seconds = block12_runtime * (len(expected_keys) / max(1, len(block12_records)))

    print("== Cube anomaly multi-seed sanity setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"block12_reference: {args.block12_path}")
    print(f"stage1b_reference: {args.stage1b_path}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"device: {args.device}")
    print(f"base_cem_seed: {args.seed}")
    print(f"pairs: {len(selected_pairs)}")
    print(f"offset: {offset}")
    print(f"horizon_blocks: {args.horizon_blocks}")
    print(f"action_block: {args.action_block}")
    print(f"raw_action_dim: {raw_action_dim}")
    print(f"num_samples: {args.num_samples}")
    print(f"cem_iters: {args.cem_iters}")
    print(f"topk: {args.topk}")
    print(f"var_scale: {args.var_scale}")
    print(f"dimensions: {list(dimensions)}")
    print(f"projection_seeds: {list(projection_seeds)}")
    print(f"resume_records_loaded: {len(seen)}")
    print(
        "runtime_estimate: Block 1.2 "
        f"{len(block12_records)} records in {seconds_to_hms(block12_runtime)}; "
        f"this run {len(expected_keys)} records ~= {seconds_to_hms(estimate_seconds)}"
    )

    total_started = time.time()
    env = gym.make(
        "swm/OGBCube-v0",
        env_type="single",
        ob_type="states",
        multiview=False,
        width=int(args.img_size),
        height=int(args.img_size),
        visualize_info=False,
        terminate_at_goal=True,
    )
    try:
        for pair_idx, pair_spec in enumerate(selected_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            pair_rows = load_pair_rows(dataset, int(pair_spec["start_row"]), int(pair_spec["goal_row"]))
            initial = pair_rows["initial"]
            goal = pair_rows["goal"]
            goal_pos = np.asarray(goal["privileged_block_0_pos"], dtype=np.float64)
            goal_quat = np.asarray(goal["privileged_block_0_quat"], dtype=np.float64)
            prepared_info = prepare_pair_info(
                policy,
                initial["pixels"],
                goal["pixels"],
                raw_action_dim=raw_action_dim,
            )
            default_baseline = stage1b_baselines[pair_id]
            cem_seed = int(args.seed) + pair_id * 1009

            for dim in dimensions:
                for projection_seed in projection_seeds:
                    key = (pair_id, int(dim), int(projection_seed))
                    if key in seen:
                        print(
                            f"Pair {pair_idx}/{len(selected_pairs)}, m={dim}, seed={projection_seed} "
                            "- loaded existing record"
                        )
                        continue

                    record_started = time.time()
                    projected = run_projected_cem(
                        model=model,
                        prepared_info=prepared_info,
                        pair_id=pair_id,
                        seed=int(args.seed),
                        projection=projections[(int(dim), int(projection_seed))],
                        horizon_blocks=int(args.horizon_blocks),
                        action_dim=int(args.action_block) * int(raw_action_dim),
                        num_samples=int(args.num_samples),
                        cem_iters=int(args.cem_iters),
                        topk=int(args.topk),
                        var_scale=float(args.var_scale),
                    )
                    projected_raw = blocked_batch_to_raw(
                        np.asarray(projected["rank1_blocked"], dtype=np.float32)[None, ...],
                        action_processor=action_processor,
                        action_block=int(args.action_block),
                        raw_action_dim=int(raw_action_dim),
                    )[0]
                    rollout_seed_base = int(args.seed) + pair_id * 100_000 + int(dim) * 1_000 + int(projection_seed)
                    proj_v1, proj_real, proj_success, proj_metrics = score_raw_actions(
                        env=env,
                        initial=initial,
                        goal=goal,
                        goal_pos=goal_pos,
                        goal_quat=goal_quat,
                        raw_actions_batch=projected_raw[None, ...],
                        candidate_indices=np.asarray([int(projected["rank1_candidate_index"])], dtype=np.int64),
                        seed_base=rollout_seed_base,
                    )
                    rollout_metrics = {
                        **proj_metrics[0],
                        "v1_cost": clean_float(float(proj_v1[0])),
                        "c_real_state": clean_float(float(proj_real[0])),
                        "success": bool(proj_success[0]),
                    }
                    record = build_projected_record(
                        pair_spec=pair_spec,
                        dim=int(dim),
                        projection_seed=int(projection_seed),
                        cem_seed=cem_seed,
                        projected=projected,
                        rollout_metrics=rollout_metrics,
                        projected_raw_rank1=projected_raw,
                        default_baseline=default_baseline,
                        endpoint_reference_record=endpoint_reference_records[key],
                    )
                    records.append(record)
                    seen.add(key)
                    summary, decision = build_summary_and_decision(
                        block12_records=block12_records,
                        records=records,
                        dimensions=dimensions,
                        projection_seeds=projection_seeds,
                        n_pairs=len(selected_pairs),
                    )
                    write_output(
                        output_path=args.output,
                        dimensions=dimensions,
                        projection_seeds=projection_seeds,
                        selected_pairs=selected_pairs,
                        records=records,
                        summary=summary,
                        decision=decision,
                        total_started=total_started,
                    )
                    print(
                        f"Pair {pair_idx}/{len(selected_pairs)}, m={dim}, seed={projection_seed} - "
                        f"success={bool(proj_success[0])}, C_real_state={float(proj_real[0]):.4f}, "
                        f"elapsed={seconds_to_hms(time.time() - record_started)}"
                    )
    finally:
        env.close()

    summary, decision = build_summary_and_decision(
        block12_records=block12_records,
        records=records,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        n_pairs=len(selected_pairs),
    )
    write_output(
        output_path=args.output,
        dimensions=dimensions,
        projection_seeds=projection_seeds,
        selected_pairs=selected_pairs,
        records=records,
        summary=summary,
        decision=decision,
        total_started=total_started,
    )

    if len(records) == len(expected_keys):
        aggregate = aggregate_records(
            records=records,
            default_baselines=default_baselines,
            dimensions=dimensions,
            projection_seeds=projection_seeds,
            cells=tuple(sorted({str(pair["cell"]) for pair in selected_pairs})),
        )
        observed = int(aggregate["observed_projected_records"])
        expected = int(aggregate["expected_projected_records"])
        if observed != expected:
            raise RuntimeError(f"Record count mismatch: {observed} != {expected}")

    print_decision_table(summary, decision)
    print(f"\nWallclock seconds: {time.time() - total_started:.3f} ({seconds_to_hms(time.time() - total_started)})")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
