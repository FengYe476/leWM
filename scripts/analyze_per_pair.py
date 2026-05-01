#!/usr/bin/env python3
"""Per-pair failure characterization for LeWM three-cost attribution results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.

from lewm_audit.diagnostics.per_pair import (
    build_report,
    load_results,
    make_analysis_args,
    run_per_pair_analysis,
    to_jsonable,
)
from lewm_audit.diagnostics.three_cost import make_policy_args

from eval_pusht_baseline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    PROJECT_ROOT,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)


DEFAULT_INPUT = PROJECT_ROOT / "results" / "three_cost_offset50.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "per_pair_analysis.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize pair-dependent three-cost failures for PushT."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
    )
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--var-scale", type=float, default=None)
    parser.add_argument("--cem-early-iters", type=int, default=3)
    parser.add_argument("--cem-late-iters", type=int, default=None)
    parser.add_argument(
        "--num-per-source",
        type=int,
        default=None,
        help="Number of CEM_late candidates to regenerate for representative rollouts.",
    )
    parser.add_argument(
        "--representatives-per-category",
        type=int,
        default=2,
        help="Number of Easy and Impossible pairs used for per-step analysis.",
    )
    parser.add_argument(
        "--skip-step-analysis",
        action="store_true",
        help="Skip model loading, CEM regeneration, latent plot, and per-step rollout.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.figures_dir = args.figures_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    data = load_results(args.input)
    offset = int(data["offset"])
    if offset % args.action_block != 0:
        raise ValueError("--action-block must divide the saved offset")
    horizon_blocks = offset // args.action_block

    metadata = data.get("metadata", {})
    if args.seed is None:
        args.seed = int(metadata.get("seed", 42))

    print("== Per-pair analysis setup ==")
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"figures_dir: {args.figures_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"device: {args.device}")
    print(f"offset: {offset}")
    print(f"horizon_blocks: {horizon_blocks}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    analysis_args = make_analysis_args(args, data, horizon_blocks)

    policy = None
    model = None
    env = None
    try:
        if not args.skip_step_analysis:
            process = build_processors(dataset, ["action", "proprio", "state"])
            policy = build_policy(make_policy_args(analysis_args, horizon_blocks), process)
            model = policy.solver.model
            env = gym.make("swm/PushT-v1")

        results = run_per_pair_analysis(
            data=data,
            dataset=dataset,
            args=args,
            policy=policy,
            model=model,
            env=env,
            analysis_args=analysis_args,
        )
    finally:
        if env is not None:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(results), indent=2, allow_nan=False) + "\n")

    print()
    print(build_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
