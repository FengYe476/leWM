#!/usr/bin/env python3
"""Reproduce the published LeWM OGBench-Cube evaluation path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

from eval_pusht_baseline import (
    DEFAULT_CACHE_DIR,
    PROJECT_ROOT,
    build_policy,
    build_processors,
    get_episodes_length,
    resolve_device,
)


DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "converted" / "lewm-cube"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "cube_baseline_eval.json"
DEFAULT_DATASET_NAME = "ogbench/cube_single_expert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LeWM OGBench-Cube evaluation using the upstream cube.yaml "
            "settings. Use --num-eval 3 --num-envs 1 for a small CPU smoke test."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the converted Cube *_object.ckpt checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
        help=(
            "stable-worldmodel cache directory containing ogbench/cube_single_expert.h5. "
            "Defaults to STABLEWM_HOME if set, otherwise the project cache."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name without the .h5 suffix.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Where to save the evaluation JSON report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on. Defaults to MPS when available, otherwise CPU.",
    )
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=50,
        help="Parallel env batch size. CPU smoke tests should pass --num-envs 1.",
    )
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
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
    return parser.parse_args()


def dataset_path(cache_dir: Path, dataset_name: str) -> Path:
    return cache_dir / f"{dataset_name}.h5"


def ensure_dataset_exists(cache_dir: Path, dataset_name: str) -> Path:
    path = dataset_path(cache_dir, dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            "OGBench-Cube dataset not found.\n"
            f"Expected: {path}\n"
            "Download quentinll/lewm-cube, extract cube_single_expert.tar.zst, "
            "and place the extracted HDF5 file under the cache dir preserving "
            "the ogbench/ subdirectory."
        )
    return path


def get_dataset(cache_dir: Path, dataset_name: str):
    ensure_dataset_exists(cache_dir, dataset_name)
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action"],
        cache_dir=cache_dir,
    )


def sample_eval_examples(
    dataset,
    *,
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(col_name)
    ep_indices, _ = np.unique(episode_ids, return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }

    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_ids])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    if len(valid_indices) < num_eval:
        raise ValueError(
            f"Requested {num_eval} evaluation episodes, but only "
            f"{len(valid_indices)} valid starting points exist."
        )

    rng = np.random.default_rng(seed)
    sampled_offsets = rng.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    sampled_rows = np.sort(valid_indices[sampled_offsets])

    sampled = dataset.get_row_data(sampled_rows)
    analysis = {
        "valid_start_points": int(len(valid_indices)),
        "total_rows": int(len(dataset.get_col_data("step_idx"))),
        "total_episodes": int(len(ep_indices)),
        "mean_episode_length": float(np.mean(episode_len)),
        "median_episode_length": float(np.median(episode_len)),
        "min_episode_length": int(np.min(episode_len)),
        "max_episode_length": int(np.max(episode_len)),
    }
    return sampled_rows, sampled[col_name], sampled["step_idx"], analysis


def cube_callables() -> list[dict]:
    return [
        {
            "method": "set_state",
            "args": {
                "qpos": {"value": "qpos"},
                "qvel": {"value": "qvel"},
            },
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
    ]


def evaluate_chunk(
    *,
    args: argparse.Namespace,
    dataset,
    policy,
    episodes_idx: list[int],
    start_steps: list[int],
    video_path: Path,
) -> dict:
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=len(episodes_idx),
        max_episode_steps=2 * args.eval_budget,
        history_size=1,
        frame_skip=1,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=args.img_size,
        height=args.img_size,
        visualize_info=False,
        terminate_at_goal=True,
        image_shape=(args.img_size, args.img_size),
    )
    world.set_policy(policy)

    try:
        metrics = world.evaluate_from_dataset(
            dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset_steps=args.goal_offset_steps,
            eval_budget=args.eval_budget,
            callables=cube_callables(),
            save_video=args.save_video,
            video_path=video_path,
        )
    finally:
        if hasattr(world, "close"):
            world.close()
        elif hasattr(world, "envs"):
            world.envs.close()

    return metrics


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.results_path = args.results_path.expanduser().resolve()
    args.device = resolve_device(args.device)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)

    print("== OGBench-Cube baseline evaluation setup ==")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"dataset_path: {dataset_path(args.cache_dir, args.dataset_name)}")
    print(f"results_path: {args.results_path}")
    print(f"device: {args.device}")

    assert args.horizon * args.action_block <= args.eval_budget, (
        "Planning horizon must be <= eval budget"
    )
    assert args.num_envs > 0, "--num-envs must be positive"
    assert args.num_eval > 0, "--num-eval must be positive"

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action"])
    policy = build_policy(args, process)

    sampled_rows, eval_episodes, eval_start_steps, dataset_stats = sample_eval_examples(
        dataset,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset_steps,
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
            f"\n== Evaluating Cube episodes {chunk_start + 1}-{chunk_end} "
            f"of {args.num_eval} =="
        )
        metrics = evaluate_chunk(
            args=args,
            dataset=dataset,
            policy=policy,
            episodes_idx=episodes_chunk,
            start_steps=start_steps_chunk,
            video_path=args.results_path.parent,
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
                "goal_step": int(eval_start_steps[local_idx] + args.goal_offset_steps),
                "success": bool(success),
            }
            all_records.append(record)
            status = "SUCCESS" if success else "FAIL"
            print(
                f"[{local_idx + 1:02d}/{args.num_eval}] episode={record['episode_idx']} "
                f"start={record['start_step']} goal={record['goal_step']} -> {status}"
            )

    elapsed = time.time() - start_time
    success_rate = float(np.mean(all_successes) * 100.0) if all_successes else 0.0

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "cache_dir": str(args.cache_dir),
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path(args.cache_dir, args.dataset_name)),
        "dataset_stats": dataset_stats,
        "env_name": "swm/OGBCube-v0",
        "env_kwargs": {
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": args.img_size,
            "height": args.img_size,
            "visualize_info": False,
            "terminate_at_goal": True,
        },
        "success_criterion": {
            "source": "stable_worldmodel.envs.ogbench.cube_env.CubeEnv._compute_successes",
            "criterion": "cube position distance to target <= 0.04m",
            "note": "The eval callable also sets target_quat, but this wrapper's success check is position-only.",
        },
        "seed": args.seed,
        "device": args.device,
        "num_eval": args.num_eval,
        "num_envs": args.num_envs,
        "goal_offset_steps": args.goal_offset_steps,
        "eval_budget": args.eval_budget,
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
        "success_rate": success_rate,
        "episode_successes": all_successes,
        "episodes": all_records,
        "chunk_metrics": chunk_metrics,
        "elapsed_seconds": elapsed,
    }

    args.results_path.write_text(json.dumps(summary, indent=2))

    print("\n== Final Cube summary ==")
    print(f"success_rate: {success_rate:.2f}%")
    print(f"episodes_succeeded: {sum(all_successes)}/{len(all_successes)}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"saved_results: {args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
