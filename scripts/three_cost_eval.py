#!/usr/bin/env python3
"""Three-cost attribution for LeWM PushT long-horizon planning."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs
import torch

from eval_pusht_baseline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    PROJECT_ROOT,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from eval_pusht_sweep import prepare_dataset_index, sample_eval_examples


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "three_cost_offset50.json"
SOURCE_DATA = "data"
SOURCE_RANDOM = "random"
SOURCE_CEM_EARLY = "cem_early"
SOURCE_CEM_LATE = "cem_late"


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


def make_policy_args(args: argparse.Namespace, horizon_blocks: int) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        cem_iters=args.cem_late_iters,
        topk=args.topk,
        seed=args.seed,
        horizon=horizon_blocks,
        receding_horizon=horizon_blocks,
        action_block=args.action_block,
        img_size=args.img_size,
    )


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def tensor_clone_info(info: dict) -> dict:
    out = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = deepcopy(value)
    return out


def expand_info_for_candidates(info: dict, num_samples: int) -> dict:
    expanded = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            expanded[key] = value.unsqueeze(1).expand(
                value.shape[0], num_samples, *value.shape[1:]
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None, ...], num_samples, axis=1)
        else:
            expanded[key] = value
    return expanded


def prepare_pair_info(policy, initial_pixels: np.ndarray, goal_pixels: np.ndarray) -> dict:
    raw_info = {
        "pixels": initial_pixels[None, None, ...],
        "goal": goal_pixels[None, None, ...],
        # The model's get_cost path expects an action key while encoding the goal.
        # This placeholder is overwritten by JEPA.rollout before dynamics rollout.
        "action": np.zeros((1, 1, 2), dtype=np.float32),
    }
    return policy._prepare_info(raw_info)


def raw_to_blocked_normalized(
    raw_actions: np.ndarray,
    *,
    action_processor,
    action_block: int,
) -> np.ndarray:
    """Convert raw env actions (T, 2) to normalized model blocks (T/5, 10)."""
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 2:
        raise ValueError(f"Expected raw actions with shape (T, 2), got {raw_actions.shape}")
    if raw_actions.shape[0] % action_block != 0:
        raise ValueError(
            f"Raw action length {raw_actions.shape[0]} is not divisible by {action_block}"
        )

    normalized = action_processor.transform(raw_actions).astype(np.float32)
    return normalized.reshape(raw_actions.shape[0] // action_block, action_block * 2)


def blocked_normalized_to_raw(
    blocked_actions: np.ndarray,
    *,
    action_processor,
    action_block: int,
) -> np.ndarray:
    """Convert normalized model blocks (T/5, 10) back to raw env actions (T, 2)."""
    blocked_actions = np.asarray(blocked_actions, dtype=np.float32)
    if blocked_actions.ndim != 2 or blocked_actions.shape[1] != action_block * 2:
        raise ValueError(
            "Expected blocked actions with shape "
            f"(H, {action_block * 2}), got {blocked_actions.shape}"
        )

    normalized = blocked_actions.reshape(blocked_actions.shape[0] * action_block, 2)
    return action_processor.inverse_transform(normalized).astype(np.float32)


def sample_data_action_sequences(
    dataset,
    valid_indices: np.ndarray,
    *,
    count: int,
    raw_steps: int,
    action_processor,
    action_block: int,
    rng: np.random.Generator,
) -> list[dict]:
    action_col = dataset.get_col_data("action")
    sampled_rows = rng.choice(valid_indices, size=count, replace=False)
    sequences = []
    for source_index, row in enumerate(sampled_rows):
        row = int(row)
        raw_actions = np.asarray(action_col[row : row + raw_steps], dtype=np.float32)
        sequences.append(
            {
                "source": SOURCE_DATA,
                "source_index": source_index,
                "dataset_row": row,
                "blocked_normalized": raw_to_blocked_normalized(
                    raw_actions,
                    action_processor=action_processor,
                    action_block=action_block,
                ),
                "raw": raw_actions,
            }
        )
    return sequences


def catmull_rom_interpolate(waypoints: np.ndarray, num_steps: int) -> np.ndarray:
    positions = np.linspace(0, num_steps - 1, len(waypoints))
    xs = np.arange(num_steps, dtype=np.float32)
    out = np.empty((num_steps, waypoints.shape[1]), dtype=np.float32)

    for out_idx, x in enumerate(xs):
        seg = int(np.searchsorted(positions, x, side="right") - 1)
        seg = max(0, min(seg, len(waypoints) - 2))
        x0, x1 = positions[seg], positions[seg + 1]
        t = 0.0 if x1 == x0 else float((x - x0) / (x1 - x0))

        p0 = waypoints[max(seg - 1, 0)]
        p1 = waypoints[seg]
        p2 = waypoints[seg + 1]
        p3 = waypoints[min(seg + 2, len(waypoints) - 1)]

        t2 = t * t
        t3 = t2 * t
        out[out_idx] = 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def sample_random_action_sequences(
    *,
    count: int,
    raw_steps: int,
    waypoints: int,
    action_processor,
    action_block: int,
    rng: np.random.Generator,
) -> list[dict]:
    sequences = []
    for source_index in range(count):
        points = rng.uniform(-1.0, 1.0, size=(waypoints, 2)).astype(np.float32)
        raw_actions = catmull_rom_interpolate(points, raw_steps)
        sequences.append(
            {
                "source": SOURCE_RANDOM,
                "source_index": source_index,
                "blocked_normalized": raw_to_blocked_normalized(
                    raw_actions,
                    action_processor=action_processor,
                    action_block=action_block,
                ),
                "raw": raw_actions,
            }
        )
    return sequences


def run_instrumented_cem(
    *,
    model,
    prepared_info: dict,
    horizon_blocks: int,
    action_dim: int,
    num_samples: int,
    var_scale: float,
    topk: int,
    topn: int,
    capture_iters: tuple[int, int],
    device: str,
    seed: int,
) -> dict[int, dict[str, np.ndarray]]:
    if topn > topk:
        raise ValueError(f"topn={topn} must be <= topk={topk}")

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    max_iter = max(capture_iters)
    mean = torch.zeros((1, horizon_blocks, action_dim), device=torch_device)
    var = var_scale * torch.ones((1, horizon_blocks, action_dim), device=torch_device)
    expanded_info = expand_info_for_candidates(prepared_info, num_samples)
    captures: dict[int, dict[str, np.ndarray]] = {}

    for iter_idx in range(1, max_iter + 1):
        candidates = torch.randn(
            1,
            num_samples,
            horizon_blocks,
            action_dim,
            generator=generator,
            device=torch_device,
        )
        candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        with torch.inference_mode():
            costs = model.get_cost(tensor_clone_info(expanded_info), candidates)

        top_vals, top_inds = torch.topk(costs, k=topk, dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]

        if iter_idx in capture_iters:
            captures[iter_idx] = {
                "actions": elite_candidates[0, :topn].detach().cpu().numpy(),
                "costs": top_vals[0, :topn].detach().cpu().numpy(),
            }

        mean = elite_candidates.mean(dim=1)
        var = elite_candidates.std(dim=1)

    return captures


def generate_cem_action_sequences(
    *,
    model,
    prepared_info: dict,
    args: argparse.Namespace,
    horizon_blocks: int,
    action_dim: int,
    action_processor,
    pair_index: int,
) -> list[dict]:
    captures = run_instrumented_cem(
        model=model,
        prepared_info=prepared_info,
        horizon_blocks=horizon_blocks,
        action_dim=action_dim,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        topk=args.topk,
        topn=args.num_per_source,
        capture_iters=(args.cem_early_iters, args.cem_late_iters),
        device=args.device,
        seed=args.seed + pair_index * 1009,
    )

    sequences = []
    for source, iter_count in (
        (SOURCE_CEM_EARLY, args.cem_early_iters),
        (SOURCE_CEM_LATE, args.cem_late_iters),
    ):
        capture = captures[iter_count]
        for source_index, blocked_normalized in enumerate(capture["actions"]):
            raw_actions = blocked_normalized_to_raw(
                blocked_normalized,
                action_processor=action_processor,
                action_block=args.action_block,
            )
            sequences.append(
                {
                    "source": source,
                    "source_index": source_index,
                    "cem_iter": iter_count,
                    "cem_rank": source_index,
                    "cem_model_cost": float(capture["costs"][source_index]),
                    "blocked_normalized": blocked_normalized.astype(np.float32),
                    "raw": raw_actions,
                }
            )
    return sequences


def compute_model_costs(model, prepared_info: dict, blocked_actions: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    candidates = torch.as_tensor(
        blocked_actions[None, ...], dtype=torch.float32, device=device
    )
    expanded_info = expand_info_for_candidates(prepared_info, candidates.shape[1])
    with torch.inference_mode():
        costs = model.get_cost(tensor_clone_info(expanded_info), candidates)
    return costs[0].detach().cpu().numpy()


def encode_pixels(policy, model, pixels: np.ndarray) -> torch.Tensor:
    info = policy._prepare_info({"pixels": pixels[None, None, ...]})
    device = next(model.parameters()).device
    pixels_t = info["pixels"].to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        encoded = model.encode({"pixels": pixels_t})
    return encoded["emb"][:, -1, :]


def squared_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum((a - b) ** 2).detach().cpu().item())


def angular_distance(a: float, b: float) -> float:
    diff = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
    return abs(diff)


def block_pose_metrics(terminal_state: np.ndarray, goal_state: np.ndarray) -> dict:
    block_pos_dist = float(np.linalg.norm(terminal_state[2:4] - goal_state[2:4]))
    angle_dist = angular_distance(terminal_state[4], goal_state[4])
    success = block_pos_dist < 20.0 and angle_dist < math.pi / 9
    return {
        "block_pos_dist": block_pos_dist,
        "angle_dist": angle_dist,
        "c_real_state": block_pos_dist + angle_dist,
        "success": bool(success),
    }


def configure_goal_render(env_unwrapped, goal_state: np.ndarray) -> None:
    env_unwrapped._set_goal_state(goal_state)
    if hasattr(env_unwrapped, "goal_pose"):
        env_unwrapped.goal_pose = np.asarray(goal_state[2:5], dtype=np.float64)


def execute_raw_actions(
    env,
    *,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    seed: int,
) -> dict:
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    configure_goal_render(env_unwrapped, goal_state)
    env_unwrapped._set_state(initial_state)

    env_success = False
    for action in raw_actions:
        _, _, terminated, _, _ = env.step(np.asarray(action, dtype=np.float32))
        env_success = env_success or bool(terminated)

    terminal_state = env_unwrapped._get_obs()
    terminal_pixels = env_unwrapped.render()
    return {
        "terminal_state": np.asarray(terminal_state, dtype=np.float32),
        "terminal_pixels": np.asarray(terminal_pixels, dtype=np.uint8),
        "env_success": bool(env_success),
    }


def load_pair_rows(dataset, row: int, offset: int) -> dict:
    rows = dataset.get_row_data([row, row + offset])
    return {
        "initial": {key: value[0] for key, value in rows.items()},
        "goal": {key: value[1] for key, value in rows.items()},
    }


def make_action_records(
    *,
    action_sequences: list[dict],
    model_costs: np.ndarray,
    real_results: list[dict],
) -> list[dict]:
    records = []
    for idx, sequence in enumerate(action_sequences):
        real = real_results[idx]
        metrics = real["state_metrics"]
        record = {
            "source": sequence["source"],
            "source_index": int(sequence["source_index"]),
            "c_model": float(model_costs[idx]),
            "c_real_z": float(real["c_real_z"]),
            "c_real_state": float(metrics["c_real_state"]),
            "block_pos_dist": float(metrics["block_pos_dist"]),
            "angle_dist": float(metrics["angle_dist"]),
            "success": bool(metrics["success"]),
            "env_success": bool(real["env_success"]),
        }
        for optional_key in ("dataset_row", "cem_iter", "cem_rank", "cem_model_cost"):
            if optional_key in sequence:
                record[optional_key] = to_jsonable(sequence[optional_key])
        records.append(record)
    return records


def evaluate_pair(
    *,
    pair_index: int,
    dataset,
    valid_action_indices: np.ndarray,
    policy,
    model,
    env,
    args: argparse.Namespace,
    row: int,
    episode_id: int,
    start_step: int,
    rng: np.random.Generator,
    horizon_blocks: int,
    raw_steps: int,
) -> dict:
    pair_rows = load_pair_rows(dataset, row, args.offset)
    initial = pair_rows["initial"]
    goal = pair_rows["goal"]
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    goal_emb = encode_pixels(policy, model, goal["pixels"])
    action_processor = policy.process["action"]

    action_sequences = []
    action_sequences.extend(
        sample_data_action_sequences(
            dataset,
            valid_action_indices,
            count=args.num_per_source,
            raw_steps=raw_steps,
            action_processor=action_processor,
            action_block=args.action_block,
            rng=rng,
        )
    )
    action_sequences.extend(
        sample_random_action_sequences(
            count=args.num_per_source,
            raw_steps=raw_steps,
            waypoints=args.random_waypoints,
            action_processor=action_processor,
            action_block=args.action_block,
            rng=rng,
        )
    )
    action_sequences.extend(
        generate_cem_action_sequences(
            model=model,
            prepared_info=prepared_info,
            args=args,
            horizon_blocks=horizon_blocks,
            action_dim=args.action_block * 2,
            action_processor=action_processor,
            pair_index=pair_index,
        )
    )

    blocked = np.stack([seq["blocked_normalized"] for seq in action_sequences])
    model_costs = compute_model_costs(model, prepared_info, blocked)

    real_results = []
    for action_index, sequence in enumerate(action_sequences):
        rollout = execute_raw_actions(
            env,
            initial_state=np.asarray(initial["state"], dtype=np.float32),
            goal_state=np.asarray(goal["state"], dtype=np.float32),
            raw_actions=sequence["raw"],
            seed=args.seed + pair_index * 10_000 + action_index,
        )
        terminal_emb = encode_pixels(policy, model, rollout["terminal_pixels"])
        state_metrics = block_pose_metrics(
            rollout["terminal_state"],
            np.asarray(goal["state"], dtype=np.float32),
        )
        real_results.append(
            {
                "c_real_z": squared_l2(terminal_emb, goal_emb),
                "state_metrics": state_metrics,
                "env_success": rollout["env_success"],
            }
        )

    return {
        "episode_id": int(episode_id),
        "start_step": int(start_step),
        "goal_step": int(start_step + args.offset),
        "dataset_row": int(row),
        "actions": make_action_records(
            action_sequences=action_sequences,
            model_costs=model_costs,
            real_results=real_results,
        ),
    }


def main() -> int:
    args = parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.results_path = args.results_path.expanduser().resolve()
    args.device = resolve_device(args.device)

    if args.offset % args.action_block != 0:
        raise ValueError("--offset must be divisible by --action-block")
    if args.num_per_source <= 0:
        raise ValueError("--num-per-source must be positive")
    if args.random_waypoints < 2:
        raise ValueError("--random-waypoints must be at least 2")
    if args.topk > args.num_samples:
        raise ValueError("--topk must be <= --num-samples")
    if args.num_per_source > args.topk:
        raise ValueError("--num-per-source must be <= --topk")

    horizon_blocks = args.offset // args.action_block
    raw_steps = horizon_blocks * args.action_block
    action_dim = args.action_block * 2
    total_actions_per_pair = 4 * args.num_per_source
    started = time.time()

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
    index = prepare_dataset_index(dataset)
    sampled_rows, eval_episodes, eval_start_steps, analysis = sample_eval_examples(
        dataset,
        index,
        num_eval=args.num_pairs,
        goal_offset_steps=args.offset,
        seed=args.seed,
    )
    valid_action_indices = analysis["valid_indices"]

    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(make_policy_args(args, horizon_blocks), process)
    model = policy.solver.model
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1")

    pairs = []
    try:
        for pair_index, (row, episode_id, start_step) in enumerate(
            zip(sampled_rows, eval_episodes, eval_start_steps, strict=True)
        ):
            pair_start = time.time()
            print(
                f"\n== Pair {pair_index + 1}/{args.num_pairs}: "
                f"episode={int(episode_id)} start={int(start_step)} "
                f"goal={int(start_step + args.offset)} =="
            )
            pair = evaluate_pair(
                pair_index=pair_index,
                dataset=dataset,
                valid_action_indices=valid_action_indices,
                policy=policy,
                model=model,
                env=env,
                args=args,
                row=int(row),
                episode_id=int(episode_id),
                start_step=int(start_step),
                rng=rng,
                horizon_blocks=horizon_blocks,
                raw_steps=raw_steps,
            )
            pair["elapsed_seconds"] = time.time() - pair_start
            pairs.append(pair)
            successes = sum(1 for action in pair["actions"] if action["success"])
            print(
                f"pair_successes: {successes}/{len(pair['actions'])}; "
                f"elapsed_seconds: {pair['elapsed_seconds']:.2f}"
            )
    finally:
        env.close()

    elapsed = time.time() - started
    output = {
        "offset": args.offset,
        "metadata": {
            "seed": args.seed,
            "device": args.device,
            "checkpoint_dir": str(args.checkpoint_dir),
            "cache_dir": str(args.cache_dir),
            "dataset_name": args.dataset_name,
            "num_pairs": args.num_pairs,
            "num_per_source": args.num_per_source,
            "actions_per_pair": total_actions_per_pair,
            "horizon_blocks": horizon_blocks,
            "raw_steps": raw_steps,
            "action_block": args.action_block,
            "action_shapes": {
                "raw_env": [raw_steps, 2],
                "blocked_normalized_model": [horizon_blocks, action_dim],
            },
            "sources": [
                SOURCE_DATA,
                SOURCE_RANDOM,
                SOURCE_CEM_EARLY,
                SOURCE_CEM_LATE,
            ],
            "cem": {
                "num_samples": args.num_samples,
                "early_iters": args.cem_early_iters,
                "late_iters": args.cem_late_iters,
                "topk": args.topk,
                "var_scale": args.var_scale,
            },
            "dataset_stats": {
                "valid_start_points": analysis["valid_start_points"],
                "eligible_episodes": analysis["eligible_episodes"],
                "total_rows": index.total_rows,
                "total_episodes": index.total_episodes,
            },
            "timing": {"elapsed_seconds": elapsed},
        },
        "pairs": pairs,
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(to_jsonable(output), indent=2))
    print("\n== Three-cost summary ==")
    print(f"pairs: {len(pairs)}")
    print(f"actions: {sum(len(pair['actions']) for pair in pairs)}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"saved_results: {args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
