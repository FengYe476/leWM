#!/usr/bin/env python3
"""Sweep OGBench-Cube goal offsets using the LeWM evaluation path."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_cube_baseline import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_NAME,
    dataset_path,
    evaluate_chunk,
    get_dataset,
)
from eval_pusht_baseline import (
    DEFAULT_CACHE_DIR,
    PROJECT_ROOT,
    build_policy,
    build_processors,
    resolve_device,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
FIXED_CUBE_EPISODE_LENGTH = 201


@dataclass(frozen=True)
class DatasetIndex:
    col_name: str
    episode_ids: np.ndarray
    step_idx: np.ndarray
    episode_indices: np.ndarray
    episode_inverse: np.ndarray
    episode_lengths: np.ndarray

    @property
    def total_episodes(self) -> int:
        return int(len(self.episode_indices))

    @property
    def total_rows(self) -> int:
        return int(len(self.step_idx))


def parse_offsets(raw: str) -> list[int]:
    offsets = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        offset = int(value)
        if offset <= 0:
            raise argparse.ArgumentTypeError(
                f"Offsets must be positive integers, got {offset}."
            )
        offsets.append(offset)

    if not offsets:
        raise argparse.ArgumentTypeError("At least one offset must be provided.")

    return list(dict.fromkeys(offsets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable OGBench-Cube goal-offset sweep with the published "
            "LeWM stable-worldmodel planning loop."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing checkpoints/converted/lewm-cube/lewm_object.ckpt.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
        help=(
            "stable-worldmodel cache directory containing "
            "ogbench/cube_single_expert.h5. Defaults to STABLEWM_HOME if set, "
            "otherwise the project cache."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name without the .h5 suffix.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where per-offset JSON reports will be written.",
    )
    parser.add_argument("--offsets", default="25,50,75,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Device to run on. Defaults to MPS when available, otherwise CPU.",
    )
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=50,
        help=(
            "Parallel env batch size. Set lower than --num-eval on CPU to chunk "
            "evaluation without changing the sampled episode set."
        ),
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-iters", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rollout videos like the upstream eval script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run offsets even if their result JSON already exists.",
    )
    args = parser.parse_args()
    args.offset_values = parse_offsets(args.offsets)
    return args


def prepare_dataset_index(dataset) -> DatasetIndex:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    episode_indices, episode_inverse = np.unique(episode_ids, return_inverse=True)
    episode_lengths = np.zeros(len(episode_indices), dtype=np.int64)
    np.maximum.at(episode_lengths, episode_inverse, step_idx + 1)
    return DatasetIndex(
        col_name=col_name,
        episode_ids=episode_ids,
        step_idx=step_idx,
        episode_indices=episode_indices,
        episode_inverse=episode_inverse,
        episode_lengths=episode_lengths,
    )


def analyze_offset(index: DatasetIndex, offset: int) -> dict:
    # Cube episodes are fixed at 201 rows. We keep starts strictly before the
    # final goalable row so valid starts match 10000 * (201 - offset - 1).
    max_start_per_episode = index.episode_lengths - offset - 1
    valid_mask = index.step_idx < max_start_per_episode[index.episode_inverse]
    valid_indices = np.flatnonzero(valid_mask)
    eligible_episode_mask = index.episode_lengths > offset + 1

    expected_fixed_length_valid_starts = int(
        index.total_episodes * (FIXED_CUBE_EPISODE_LENGTH - offset - 1)
    )
    fixed_length_assumption_holds = bool(
        np.all(index.episode_lengths == FIXED_CUBE_EPISODE_LENGTH)
    )

    return {
        "offset": offset,
        "valid_indices": valid_indices,
        "valid_start_points": int(len(valid_indices)),
        "expected_fixed_length_valid_start_points": expected_fixed_length_valid_starts,
        "fixed_length_assumption_holds": fixed_length_assumption_holds,
        "eligible_episodes": int(np.count_nonzero(eligible_episode_mask)),
        "ineligible_episodes": int(np.count_nonzero(~eligible_episode_mask)),
        "eligible_episode_fraction": float(np.mean(eligible_episode_mask)),
        "max_episode_length": int(index.episode_lengths.max()),
        "mean_episode_length": float(index.episode_lengths.mean()),
        "median_episode_length": float(np.median(index.episode_lengths)),
        "min_episode_length": int(index.episode_lengths.min()),
    }


def sample_eval_examples(
    dataset,
    index: DatasetIndex,
    *,
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    analysis = analyze_offset(index, goal_offset_steps)
    valid_indices = analysis["valid_indices"]

    if analysis["valid_start_points"] != analysis[
        "expected_fixed_length_valid_start_points"
    ]:
        raise ValueError(
            f"Offset {goal_offset_steps} produced "
            f"{analysis['valid_start_points']} valid starts, but the fixed-length "
            "Cube formula expected "
            f"{analysis['expected_fixed_length_valid_start_points']}."
        )

    if analysis["eligible_episodes"] < num_eval:
        raise ValueError(
            f"Offset {goal_offset_steps} only leaves "
            f"{analysis['eligible_episodes']} eligible episodes, which is fewer "
            f"than --num-eval={num_eval}."
        )

    if analysis["valid_start_points"] < num_eval:
        raise ValueError(
            f"Offset {goal_offset_steps} only leaves "
            f"{analysis['valid_start_points']} valid starting points, which is "
            f"fewer than --num-eval={num_eval}."
        )

    rng = np.random.default_rng(seed)
    sampled_offsets = rng.choice(len(valid_indices), size=num_eval, replace=False)
    sampled_rows = np.sort(valid_indices[sampled_offsets])

    sampled = dataset.get_row_data(sampled_rows)
    eval_episodes = sampled[index.col_name]
    eval_start_steps = sampled["step_idx"]
    return sampled_rows, eval_episodes, eval_start_steps, analysis


def reset_policy_state(policy) -> None:
    if hasattr(policy, "_next_init"):
        policy._next_init = None


def build_result_path(results_dir: Path, offset: int) -> Path:
    return results_dir / f"cube_sweep_offset{offset}.json"


def load_existing_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not load existing result {path}: {exc}")
        return None


def evaluate_offset(
    *,
    args: argparse.Namespace,
    dataset,
    index: DatasetIndex,
    policy,
    offset: int,
    budget: int,
    results_path: Path,
) -> dict:
    sampled_rows, eval_episodes, eval_start_steps, analysis = sample_eval_examples(
        dataset,
        index,
        num_eval=args.num_eval,
        goal_offset_steps=offset,
        seed=args.seed,
    )

    all_records = []
    all_successes = []
    chunk_metrics = []
    start_time = time.time()

    for chunk_start in range(0, args.num_eval, args.num_envs):
        chunk_end = min(chunk_start + args.num_envs, args.num_eval)
        episodes_chunk = eval_episodes[chunk_start:chunk_end].tolist()
        start_steps_chunk = eval_start_steps[chunk_start:chunk_end].tolist()

        print(
            f"\n== Offset {offset}: Cube episodes {chunk_start + 1}-{chunk_end} "
            f"of {args.num_eval} (budget={budget}) =="
        )
        reset_policy_state(policy)
        chunk_args = argparse.Namespace(**vars(args))
        chunk_args.goal_offset_steps = offset
        chunk_args.eval_budget = budget
        metrics = evaluate_chunk(
            args=chunk_args,
            dataset=dataset,
            policy=policy,
            episodes_idx=episodes_chunk,
            start_steps=start_steps_chunk,
            video_path=results_path.parent,
        )
        chunk_metrics.append(
            {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "success_rate": metrics["success_rate"],
            }
        )

        successes = np.asarray(metrics["episode_successes"], dtype=bool)
        all_successes.extend(successes.tolist())

        for local_idx, success in enumerate(successes, start=chunk_start):
            record = {
                "eval_index": local_idx,
                "dataset_row": int(sampled_rows[local_idx]),
                "episode_idx": int(eval_episodes[local_idx]),
                "start_step": int(eval_start_steps[local_idx]),
                "goal_step": int(eval_start_steps[local_idx] + offset),
                "success": bool(success),
            }
            all_records.append(record)
            status = "SUCCESS" if success else "FAIL"
            print(
                f"[{local_idx + 1:02d}/{args.num_eval}] episode={record['episode_idx']} "
                f"start={record['start_step']} goal={record['goal_step']} -> {status}"
            )

    elapsed = time.time() - start_time
    successes = int(sum(all_successes))
    success_rate = float(np.mean(all_successes) * 100.0) if all_successes else 0.0

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_path": str(args.checkpoint_dir / "lewm_object.ckpt"),
        "cache_dir": str(args.cache_dir),
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path(args.cache_dir, args.dataset_name)),
        "env_name": "swm/OGBCube-v0",
        "seed": args.seed,
        "device": args.device,
        "num_eval": args.num_eval,
        "num_envs": args.num_envs,
        "goal_offset_steps": offset,
        "goal_offset_model_steps": offset / args.action_block,
        "eval_budget": budget,
        "plan_config": {
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "warm_start": True,
        },
        "solver": {
            "type": "CEMSolver",
            "num_samples": args.num_samples,
            "n_steps": args.cem_iters,
            "topk": args.topk,
            "var_scale": args.var_scale,
            "batch_size": 1,
        },
        "dataset_stats": {
            "total_rows": index.total_rows,
            "total_episodes": index.total_episodes,
            "fixed_episode_length_rows": FIXED_CUBE_EPISODE_LENGTH,
            "valid_start_formula": "total_episodes * (201 - offset - 1)",
            "valid_start_points": analysis["valid_start_points"],
            "expected_fixed_length_valid_start_points": analysis[
                "expected_fixed_length_valid_start_points"
            ],
            "fixed_length_assumption_holds": analysis[
                "fixed_length_assumption_holds"
            ],
            "eligible_episodes": analysis["eligible_episodes"],
            "ineligible_episodes": analysis["ineligible_episodes"],
            "eligible_episode_fraction": analysis["eligible_episode_fraction"],
            "episode_length_rows": {
                "min": analysis["min_episode_length"],
                "mean": analysis["mean_episode_length"],
                "median": analysis["median_episode_length"],
                "max": analysis["max_episode_length"],
            },
        },
        "success_rate": success_rate,
        "num_successes": successes,
        "success_criterion": {
            "source": "stable_worldmodel.envs.ogbench.cube_env.CubeEnv._compute_successes",
            "criterion": "cube position distance to target <= 0.04m",
        },
        "episode_successes": all_successes,
        "episodes": all_records,
        "chunk_metrics": chunk_metrics,
        "elapsed_seconds": elapsed,
    }
    results_path.write_text(json.dumps(summary, indent=2))
    return summary


def format_summary_table(rows: list[dict]) -> str:
    headers = [
        "Offset",
        "Budget",
        "Success Rate",
        "Episodes",
        "Valid Start Points",
    ]
    data_rows = []
    for row in rows:
        success_count = row.get("num_successes", sum(row["episode_successes"]))
        total = row["num_eval"]
        valid = row["dataset_stats"]["valid_start_points"]
        data_rows.append(
            [
                str(row["goal_offset_steps"]),
                str(row["eval_budget"]),
                f"{row['success_rate']:.2f}%",
                f"{success_count}/{total}",
                str(valid),
            ]
        )

    widths = [
        max(len(header), *(len(values[i]) for values in data_rows))
        for i, header in enumerate(headers)
    ]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[idx]) for idx, value in enumerate(values)
        )

    divider = "-+-".join("-" * width for width in widths)
    lines = [fmt_row(headers), divider]
    lines.extend(fmt_row(values) for values in data_rows)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    assert args.num_envs > 0, "--num-envs must be positive"
    assert args.num_eval > 0, "--num-eval must be positive"
    assert args.horizon > 0, "--horizon must be positive"
    assert args.receding_horizon > 0, "--receding-horizon must be positive"
    assert args.action_block > 0, "--action-block must be positive"

    budgets = {offset: offset * 2 for offset in args.offset_values}
    min_budget = args.horizon * args.action_block
    for offset, budget in budgets.items():
        if budget < min_budget:
            raise ValueError(
                f"Offset {offset} gives budget {budget}, but the configured "
                f"planning horizon requires at least {min_budget} steps."
            )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("== OGBench-Cube goal-offset sweep setup ==")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"checkpoint_path: {args.checkpoint_dir / 'lewm_object.ckpt'}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"dataset_path: {dataset_path(args.cache_dir, args.dataset_name)}")
    print(f"results_dir: {args.results_dir}")
    print(f"device: {args.device}")
    print(f"offsets: {args.offset_values}")
    print("budget_rule: offset * 2")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    index = prepare_dataset_index(dataset)
    print(
        "dataset rows/episodes: "
        f"{index.total_rows}/{index.total_episodes}"
    )
    print(
        "episode length stats (rows): "
        f"min={index.episode_lengths.min()} "
        f"mean={index.episode_lengths.mean():.2f} "
        f"median={np.median(index.episode_lengths):.2f} "
        f"max={index.episode_lengths.max()}"
    )

    process = build_processors(dataset, ["action"])
    policy = build_policy(args, process)

    summaries = []
    for offset in args.offset_values:
        budget = budgets[offset]
        results_path = build_result_path(args.results_dir, offset)
        existing = None if args.force else load_existing_result(results_path)

        if existing is not None:
            if existing.get("goal_offset_steps") != offset:
                print(
                    f"WARNING: {results_path} contains offset "
                    f"{existing.get('goal_offset_steps')} instead of {offset}; "
                    "re-running."
                )
            else:
                print(f"\n== Offset {offset}: using existing result {results_path} ==")
                summaries.append(existing)
                continue

        analysis = analyze_offset(index, offset)
        if analysis["valid_start_points"] <= 0:
            raise ValueError(
                f"Offset {offset} is not feasible: dataset max episode length is "
                f"{analysis['max_episode_length']} rows."
            )
        print(
            f"\n== Offset {offset}: {analysis['valid_start_points']} valid starts "
            f"(expected {analysis['expected_fixed_length_valid_start_points']}), "
            f"{analysis['eligible_episodes']}/{index.total_episodes} eligible "
            f"episodes, budget={budget} =="
        )
        summary = evaluate_offset(
            args=args,
            dataset=dataset,
            index=index,
            policy=policy,
            offset=offset,
            budget=budget,
            results_path=results_path,
        )
        summaries.append(summary)
        print(
            f"saved_results: {results_path}\n"
            f"success_rate: {summary['success_rate']:.2f}% "
            f"({summary['num_successes']}/{summary['num_eval']})"
        )

    print("\n== Cube sweep summary ==")
    print(format_summary_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
