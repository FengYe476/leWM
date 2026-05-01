#!/usr/bin/env python3
"""Evaluate Track A sampled pairs with the Phase 0 three-cost protocol."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.

from lewm_audit.diagnostics.three_cost import (
    SOURCE_CEM_EARLY,
    SOURCE_CEM_LATE,
    SOURCE_DATA,
    SOURCE_RANDOM,
    block_pose_metrics,
    compute_model_costs,
    encode_pixels,
    execute_raw_actions,
    generate_cem_action_sequences,
    load_pair_rows,
    prepare_pair_info,
    sample_data_action_sequences,
    sample_random_action_sequences,
    squared_l2,
    to_jsonable,
)
from lewm_audit.eval.pusht import analyze_offset, prepare_dataset_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)


DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
SOURCE_LABELS = {
    SOURCE_DATA: "data",
    SOURCE_RANDOM: "smooth_random",
    SOURCE_CEM_EARLY: "CEM_early",
    SOURCE_CEM_LATE: "CEM_late",
}
ACTION_SOURCE_ORDER = ("data", "smooth_random", "CEM_early", "CEM_late")
NUM_SAMPLES = 300
CEM_ITERS = 30
CEM_EARLY_ITERS = 3
CEM_LATE_ITERS = 30
TOPK = 30
VAR_SCALE = 1.0
PLANNING_HORIZON = 5
RECEDING_HORIZON = 5
ACTION_BLOCK = 5
IMG_SIZE = 224
RANDOM_WAYPOINTS = 5


def parse_action_counts(raw: str) -> dict[str, int]:
    chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if len(chunks) != 4:
        raise argparse.ArgumentTypeError(
            "--action-counts must contain four comma-separated integers"
        )
    values = [int(chunk) for chunk in chunks]
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("--action-counts values must be nonnegative")
    return dict(zip(ACTION_SOURCE_ORDER, values, strict=True))


def parse_pair_ids(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        pair_id = int(value)
        if pair_id < 0:
            raise argparse.ArgumentTypeError("--pair-ids must be nonnegative integers")
        values.append(pair_id)
    if not values:
        raise argparse.ArgumentTypeError("--pair-ids must include at least one integer")
    return list(dict.fromkeys(values))


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def load_pairs(
    path: Path,
    *,
    max_pairs: int | None,
    pair_ids: list[int] | None,
) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    pairs = sorted(data["pairs"], key=lambda pair: int(pair["pair_id"]))
    if pair_ids is not None:
        by_id = {int(pair["pair_id"]): pair for pair in pairs}
        missing = sorted(set(pair_ids) - set(by_id))
        if missing:
            raise ValueError(f"Requested pair_ids not found in pairs file: {missing}")
        pairs = [by_id[pair_id] for pair_id in sorted(pair_ids)]
    elif max_pairs is not None:
        pairs = pairs[:max_pairs]
    return data, pairs


def load_existing_output(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def make_policy_namespace(*, checkpoint_dir: Path, device: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=checkpoint_dir,
        device=device,
        num_samples=NUM_SAMPLES,
        var_scale=VAR_SCALE,
        cem_iters=CEM_ITERS,
        topk=TOPK,
        seed=seed,
        horizon=PLANNING_HORIZON,
        receding_horizon=RECEDING_HORIZON,
        action_block=ACTION_BLOCK,
        img_size=IMG_SIZE,
    )


def make_three_cost_namespace(
    *,
    checkpoint_dir: Path,
    cache_dir: Path,
    dataset_name: str,
    device: str,
    seed: int,
    offset: int,
    max_cem_count: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=device,
        seed=seed,
        offset=offset,
        num_per_source=max_cem_count,
        img_size=IMG_SIZE,
        action_block=ACTION_BLOCK,
        num_samples=NUM_SAMPLES,
        cem_early_iters=CEM_EARLY_ITERS,
        cem_late_iters=CEM_LATE_ITERS,
        topk=TOPK,
        var_scale=VAR_SCALE,
        random_waypoints=RANDOM_WAYPOINTS,
    )


def cem_config_json() -> dict:
    return {
        "samples_per_iter": NUM_SAMPLES,
        "iterations": CEM_ITERS,
        "elites": TOPK,
        "planning_horizon": PLANNING_HORIZON,
        "receding_horizon": RECEDING_HORIZON,
        "action_block": ACTION_BLOCK,
        "cem_early_iteration": CEM_EARLY_ITERS,
        "cem_late_iteration": CEM_LATE_ITERS,
    }


def build_output(
    *,
    pairs_path: Path,
    n_pairs_requested: int,
    device: str,
    seed: int,
    action_counts: dict[str, int],
    fixed_sequence_length_raw_steps: int,
    fixed_sequence_length_action_blocks: int,
    offset_steps_at_runtime: int,
    existing: dict | None,
) -> dict:
    sequence_metadata = {
        "fixed_sequence_length_raw_steps": fixed_sequence_length_raw_steps,
        "fixed_sequence_length_action_blocks": fixed_sequence_length_action_blocks,
        "offset_steps_at_runtime": offset_steps_at_runtime,
    }
    if existing is not None:
        existing["metadata"].update(
            {
                "pairs_path": str(pairs_path),
                "n_pairs_requested": n_pairs_requested,
                "device": device,
                "seed": seed,
                "action_counts": action_counts,
                "cem_config": cem_config_json(),
                "git_commit": get_git_commit(),
                "timestamp_finished": None,
                **sequence_metadata,
            }
        )
        return existing
    return {
        "metadata": {
            "pairs_path": str(pairs_path),
            "n_pairs_requested": n_pairs_requested,
            "n_pairs_completed": 0,
            "device": device,
            "seed": seed,
            "action_counts": action_counts,
            "cem_config": cem_config_json(),
            "git_commit": get_git_commit(),
            "timestamp_started": iso_now(),
            "timestamp_finished": None,
            **sequence_metadata,
        },
        "pairs": [],
    }


def validate_requested_pair_offsets(pairs: list[dict], *, offset: int) -> None:
    mismatches = [
        {
            "pair_id": int(pair["pair_id"]),
            "start_row": int(pair["start_row"]),
            "goal_row": int(pair["goal_row"]),
            "delta": int(pair["goal_row"]) - int(pair["start_row"]),
        }
        for pair in pairs
        if int(pair["goal_row"]) - int(pair["start_row"]) != offset
    ]
    if mismatches:
        raise ValueError(
            "Pair file offset mismatch at runtime. "
            f"Expected offset={offset}; examples={mismatches[:5]}"
        )


def write_output(path: Path, output: dict, *, finished: bool = False) -> None:
    output["pairs"] = sorted(output["pairs"], key=lambda pair: int(pair["pair_id"]))
    output["metadata"]["n_pairs_completed"] = len(output["pairs"])
    output["metadata"]["timestamp_finished"] = iso_now() if finished else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(output), indent=2, allow_nan=False) + "\n")


def select_action_sequences(
    *,
    dataset,
    valid_action_indices: np.ndarray,
    policy,
    model,
    prepared_info: dict,
    args: argparse.Namespace,
    pair_id: int,
    raw_steps: int,
    action_counts: dict[str, int],
) -> list[dict]:
    action_processor = policy.process["action"]
    rng = np.random.default_rng(args.seed + pair_id * 1_000_003)

    sequences = []
    if action_counts["data"]:
        sequences.extend(
            sample_data_action_sequences(
                dataset,
                valid_action_indices,
                count=action_counts["data"],
                raw_steps=raw_steps,
                action_processor=action_processor,
                action_block=args.action_block,
                rng=rng,
            )
        )
    if action_counts["smooth_random"]:
        sequences.extend(
            sample_random_action_sequences(
                count=action_counts["smooth_random"],
                raw_steps=raw_steps,
                waypoints=args.random_waypoints,
                action_processor=action_processor,
                action_block=args.action_block,
                rng=rng,
            )
        )
    max_cem_count = max(action_counts["CEM_early"], action_counts["CEM_late"])
    if max_cem_count:
        cem_sequences = generate_cem_action_sequences(
            model=model,
            prepared_info=prepared_info,
            args=args,
            horizon_blocks=raw_steps // args.action_block,
            action_dim=args.action_block * 2,
            action_processor=action_processor,
            pair_index=pair_id,
        )
        by_source = {SOURCE_CEM_EARLY: [], SOURCE_CEM_LATE: []}
        for sequence in cem_sequences:
            by_source[sequence["source"]].append(sequence)
        sequences.extend(by_source[SOURCE_CEM_EARLY][: action_counts["CEM_early"]])
        sequences.extend(by_source[SOURCE_CEM_LATE][: action_counts["CEM_late"]])
    return sequences


def evaluate_track_a_pair(
    *,
    pair_spec: dict,
    dataset,
    valid_action_indices: np.ndarray,
    policy,
    model,
    env,
    args: argparse.Namespace,
    action_counts: dict[str, int],
) -> dict:
    started = time.time()
    pair_id = int(pair_spec["pair_id"])
    row = int(pair_spec["start_row"])
    pair_rows = load_pair_rows(dataset, row, args.offset)
    initial = pair_rows["initial"]
    goal = pair_rows["goal"]
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    goal_emb = encode_pixels(policy, model, goal["pixels"])
    raw_steps = int(pair_spec["goal_row"]) - int(pair_spec["start_row"])

    action_sequences = select_action_sequences(
        dataset=dataset,
        valid_action_indices=valid_action_indices,
        policy=policy,
        model=model,
        prepared_info=prepared_info,
        args=args,
        pair_id=pair_id,
        raw_steps=raw_steps,
        action_counts=action_counts,
    )
    expected_actions = sum(action_counts.values())
    if len(action_sequences) != expected_actions:
        raise RuntimeError(
            f"Pair {pair_id} produced {len(action_sequences)} actions, expected {expected_actions}"
        )

    blocked = np.stack([sequence["blocked_normalized"] for sequence in action_sequences])
    model_costs = compute_model_costs(model, prepared_info, blocked)

    actions = []
    for action_id, (sequence, model_cost) in enumerate(zip(action_sequences, model_costs, strict=True)):
        rollout = execute_raw_actions(
            env,
            initial_state=np.asarray(initial["state"], dtype=np.float32),
            goal_state=np.asarray(goal["state"], dtype=np.float32),
            raw_actions=sequence["raw"],
            seed=args.seed + pair_id * 10_000 + action_id,
        )
        terminal_emb = encode_pixels(policy, model, rollout["terminal_pixels"])
        state_metrics = block_pose_metrics(
            rollout["terminal_state"],
            np.asarray(goal["state"], dtype=np.float32),
        )
        actions.append(
            {
                "action_id": action_id,
                "source": SOURCE_LABELS[sequence["source"]],
                "C_real_z": squared_l2(terminal_emb, goal_emb),
                "C_model": float(model_cost),
                "C_real_state": float(state_metrics["c_real_state"]),
                "success": bool(state_metrics["success"]),
            }
        )

    return {
        "pair_id": pair_id,
        "cell": pair_spec["cell"],
        "episode_id": int(pair_spec["episode_id"]),
        "start_row": int(pair_spec["start_row"]),
        "goal_row": int(pair_spec["goal_row"]),
        "block_displacement_px": float(pair_spec["block_displacement_px"]),
        "required_rotation_rad": float(pair_spec["required_rotation_rad"]),
        "wallclock_seconds": time.time() - started,
        "actions": actions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Track A pairs with three costs.")
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--action-counts",
        type=parse_action_counts,
        default=parse_action_counts("20,20,20,20"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.device = resolve_device(args.device)
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("--max-pairs must be positive when provided")
    if args.action_counts["CEM_early"] > TOPK or args.action_counts["CEM_late"] > TOPK:
        raise ValueError("CEM action counts must be <= TOPK=30")

    pairs_data, requested_pairs = load_pairs(
        args.pairs_path,
        max_pairs=args.max_pairs,
        pair_ids=args.pair_ids,
    )
    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    if offset % ACTION_BLOCK != 0:
        raise ValueError("Track A offset must be divisible by action_block=5")
    validate_requested_pair_offsets(requested_pairs, offset=offset)
    fixed_sequence_length_raw_steps = offset
    fixed_sequence_length_action_blocks = fixed_sequence_length_raw_steps // ACTION_BLOCK
    dataset_path = Path(pair_metadata["dataset_path"])
    cache_dir = dataset_path.parent
    dataset_name = dataset_path.stem
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR.expanduser().resolve()

    existing = load_existing_output(args.output) if args.resume else None
    output = build_output(
        pairs_path=args.pairs_path,
        n_pairs_requested=len(requested_pairs),
        device=args.device,
        seed=args.seed,
        action_counts=args.action_counts,
        fixed_sequence_length_raw_steps=fixed_sequence_length_raw_steps,
        fixed_sequence_length_action_blocks=fixed_sequence_length_action_blocks,
        offset_steps_at_runtime=offset,
        existing=existing,
    )
    completed = {int(pair["pair_id"]) for pair in output["pairs"]}

    print("== Track A three-cost eval setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output: {args.output}")
    print(f"dataset_name: {dataset_name}")
    print(f"cache_dir: {cache_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"device: {args.device}")
    print(f"offset: {offset}")
    print(f"fixed_sequence_length_raw_steps: {fixed_sequence_length_raw_steps}")
    print(f"fixed_sequence_length_action_blocks: {fixed_sequence_length_action_blocks}")
    print(f"action_counts: {args.action_counts}")
    print(f"resume_completed: {len(completed)}")

    dataset = get_dataset(cache_dir, dataset_name)
    index = prepare_dataset_index(dataset)
    analysis = analyze_offset(index, offset)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy_args = make_policy_namespace(
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )
    policy = build_policy(policy_args, process)
    model = policy.solver.model
    max_cem_count = max(args.action_counts["CEM_early"], args.action_counts["CEM_late"])
    cost_args = make_three_cost_namespace(
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        device=args.device,
        seed=args.seed,
        offset=offset,
        max_cem_count=max_cem_count,
    )

    env = gym.make("swm/PushT-v1")
    try:
        for pair_spec in requested_pairs:
            pair_id = int(pair_spec["pair_id"])
            if pair_id in completed:
                print(f"Skipping completed pair_id={pair_id}")
                continue
            pair_started = time.time()
            print(
                f"\n== pair_id={pair_id} cell={pair_spec['cell']} "
                f"start_row={pair_spec['start_row']} goal_row={pair_spec['goal_row']} =="
            )
            result = evaluate_track_a_pair(
                pair_spec=pair_spec,
                dataset=dataset,
                valid_action_indices=analysis["valid_indices"],
                policy=policy,
                model=model,
                env=env,
                args=cost_args,
                action_counts=args.action_counts,
            )
            output["pairs"].append(result)
            completed.add(pair_id)
            successes = sum(action["success"] for action in result["actions"])
            print(
                f"pair_successes: {successes}/{len(result['actions'])}; "
                f"elapsed_seconds: {time.time() - pair_started:.2f}"
            )
            write_output(args.output, output, finished=False)
    finally:
        env.close()

    write_output(args.output, output, finished=True)
    print("\n== Track A three-cost summary ==")
    print(f"pairs_completed: {output['metadata']['n_pairs_completed']}")
    print(f"actions_completed: {sum(len(pair['actions']) for pair in output['pairs'])}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
