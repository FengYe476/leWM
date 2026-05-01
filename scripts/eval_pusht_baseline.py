#!/usr/bin/env python3
"""Reproduce the published LeWM PushT evaluation protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import torch
from sklearn import preprocessing
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel, PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from torchvision.transforms import v2 as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_LEWM_ROOT = PROJECT_ROOT / "third_party" / "le-wm"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "converted" / "lewm-pusht"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "pusht_baseline_eval.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "stablewm_cache"

if str(THIRD_PARTY_LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_LEWM_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the published LeWM PushT evaluation with the stable-worldmodel "
            "planning loop."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the converted *_object.ckpt checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
        help=(
            "stable-worldmodel cache directory containing pusht_expert_train.h5. "
            "Defaults to STABLEWM_HOME if set, otherwise a project-local cache."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="pusht_expert_train",
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
        help=(
            "Parallel env batch size. Set lower than --num-eval on CPU to chunk "
            "evaluation without changing the episode set."
        ),
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


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_episodes_length(dataset, episodes: np.ndarray) -> np.ndarray:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.asarray(lengths)


def ensure_dataset_exists(cache_dir: Path, dataset_name: str) -> Path:
    dataset_path = cache_dir / f"{dataset_name}.h5"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "PushT dataset not found.\n"
            f"Expected: {dataset_path}\n"
            "Download the LeWM dataset archive and place the extracted "
            f"'{dataset_name}.h5' under the cache dir."
        )
    return dataset_path


def get_dataset(cache_dir: Path, dataset_name: str):
    ensure_dataset_exists(cache_dir, dataset_name)
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=cache_dir,
    )


def build_processors(dataset, keys_to_cache: list[str]) -> dict:
    process = {}
    for col in keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_eval_examples(
    dataset,
    *,
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }

    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    if len(valid_indices) < num_eval:
        raise ValueError(
            f"Requested {num_eval} evaluation episodes, but only "
            f"{len(valid_indices)} valid starting points exist."
        )

    rng = np.random.default_rng(seed)
    sampled_offsets = rng.choice(
        len(valid_indices) - 1, size=num_eval, replace=False
    )
    sampled_rows = np.sort(valid_indices[sampled_offsets])

    sampled = dataset.get_row_data(sampled_rows)
    eval_episodes = sampled[col_name]
    eval_start_idx = sampled["step_idx"]
    return sampled_rows, eval_episodes, eval_start_idx


def build_policy(args: argparse.Namespace, process: dict) -> WorldModelPolicy:
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Converted checkpoint directory not found: {checkpoint_dir}"
        )

    model = AutoCostModel(str(checkpoint_dir))
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    solver = CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        n_steps=args.cem_iters,
        topk=args.topk,
        device=args.device,
        seed=args.seed,
    )
    plan_config = PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        history_len=1,
        action_block=args.action_block,
        warm_start=True,
    )
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }
    return WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )


def evaluate_chunk(
    *,
    args: argparse.Namespace,
    dataset,
    policy: WorldModelPolicy,
    episodes_idx: list[int],
    start_steps: list[int],
    video_path: Path,
) -> dict:
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=len(episodes_idx),
        max_episode_steps=2 * args.eval_budget,
        history_size=1,
        frame_skip=1,
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
            callables=[
                {"method": "_set_state", "args": {"state": {"value": "state"}}},
                {
                    "method": "_set_goal_state",
                    "args": {"goal_state": {"value": "goal_state"}},
                },
            ],
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

    print("== PushT baseline evaluation setup ==")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"results_path: {args.results_path}")
    print(f"device: {args.device}")

    assert args.horizon * args.action_block <= args.eval_budget, (
        "Planning horizon must be <= eval budget"
    )
    assert args.num_envs > 0, "--num-envs must be positive"
    assert args.num_eval > 0, "--num-eval must be positive"

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(args, process)

    sampled_rows, eval_episodes, eval_start_steps = sample_eval_examples(
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
            f"\n== Evaluating episodes {chunk_start + 1}-{chunk_end} "
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
        "env_name": "swm/PushT-v1",
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

    print("\n== Final summary ==")
    print(f"success_rate: {success_rate:.2f}%")
    print(f"episodes_succeeded: {sum(all_successes)}/{len(all_successes)}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"saved_results: {args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
