#!/usr/bin/env python3
"""Three-cost attribution for LeWM PushT long-horizon planning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs

from lewm_audit.diagnostics.three_cost import (
    make_policy_args,
    run_three_cost_protocol,
    to_jsonable,
    validate_three_cost_args,
)

from eval_pusht_baseline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    PROJECT_ROOT,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "three_cost_offset50.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PushT three-cost attribution protocol at a fixed offset."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the converted PushT object checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
        help="stable-worldmodel cache directory containing pusht_expert_train.h5.",
    )
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offset", type=int, default=50)
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument(
        "--num-per-source",
        type=int,
        default=10,
        help="Number of action sequences to keep from each source.",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-early-iters", type=int, default=3)
    parser.add_argument("--cem-late-iters", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--random-waypoints", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.results_path = args.results_path.expanduser().resolve()
    args.device = resolve_device(args.device)
    validate_three_cost_args(args)

    horizon_blocks = args.offset // args.action_block
    raw_steps = horizon_blocks * args.action_block
    total_actions_per_pair = 4 * args.num_per_source

    print("== Three-cost PushT attribution setup ==")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"results_path: {args.results_path}")
    print(f"device: {args.device}")
    print(f"offset: {args.offset}")
    print(f"horizon_blocks: {horizon_blocks}")
    print(f"raw_steps: {raw_steps}")
    print(f"actions_per_pair: {total_actions_per_pair}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(make_policy_args(args, horizon_blocks), process)
    env = gym.make("swm/PushT-v1")
    try:
        output = run_three_cost_protocol(
            dataset=dataset,
            policy=policy,
            model=policy.solver.model,
            env=env,
            args=args,
        )
    finally:
        env.close()

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(to_jsonable(output), indent=2))
    print("\n== Three-cost summary ==")
    print(f"pairs: {len(output['pairs'])}")
    print(f"actions: {sum(len(pair['actions']) for pair in output['pairs'])}")
    print(f"elapsed_seconds: {output['metadata']['timing']['elapsed_seconds']:.2f}")
    print(f"saved_results: {args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
