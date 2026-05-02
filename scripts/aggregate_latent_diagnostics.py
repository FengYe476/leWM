#!/usr/bin/env python3
"""Aggregate latent-geometry diagnostics for LeWM PushT rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lewm_audit.diagnostics.aggregate import (
    build_report,
    make_policy_args,
    run_aggregate_latent_diagnostics,
    to_jsonable,
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


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "aggregate_latent_diagnostics.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute temporal straightness, effective rank, covariance spectrum, "
            "and SIGReg-style normality diagnostics for real vs imagined PushT latents."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
    )
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Defaults to CPU so the analysis is sandbox-friendly.",
    )
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument(
        "--latent-steps",
        type=int,
        default=10,
        help="Number of predicted latent steps; each step consumes --action-block raw actions.",
    )
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--sigreg-directions", type=int, default=100)
    parser.add_argument(
        "--sigreg-frequencies",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated frequencies for the empirical-characteristic-function normality proxy.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.results_path = args.results_path.expanduser().resolve()
    args.figures_dir = args.figures_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    raw_steps = args.latent_steps * args.action_block
    print("== Aggregate latent diagnostics setup ==")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"results_path: {args.results_path}")
    print(f"figures_dir: {args.figures_dir}")
    print(f"device: {args.device}")
    print(f"num_trajectories: {args.num_trajectories}")
    print(f"latent_steps: {args.latent_steps}")
    print(f"raw_steps: {raw_steps}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action"])
    policy = build_policy(make_policy_args(args), process)
    results = run_aggregate_latent_diagnostics(
        dataset=dataset,
        policy=policy,
        model=policy.solver.model,
        args=args,
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(to_jsonable(results), indent=2, allow_nan=False) + "\n")
    print()
    print(build_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
