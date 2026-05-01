#!/usr/bin/env python3
"""Three-cost attribution for LeWM PushT long-horizon planning."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from lewm_audit.eval.pusht import prepare_dataset_index, sample_eval_examples

SOURCE_DATA = "data"
SOURCE_RANDOM = "random"
SOURCE_CEM_EARLY = "cem_early"
SOURCE_CEM_LATE = "cem_late"
COST_KEYS = ("c_model", "c_real_z", "c_real_state")
METRIC_PAIRS = (
    ("c_real_z", "c_real_state"),
    ("c_model", "c_real_z"),
    ("c_model", "c_real_state"),
)
SOURCE_ORDER = (SOURCE_DATA, SOURCE_RANDOM, SOURCE_CEM_EARLY, SOURCE_CEM_LATE)

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - exercised only outside the project env.
    scipy_stats = None


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


def validate_three_cost_args(args: argparse.Namespace) -> None:
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


def run_three_cost_protocol(
    *,
    dataset,
    policy,
    model,
    env,
    args: argparse.Namespace,
) -> dict:
    validate_three_cost_args(args)
    horizon_blocks = args.offset // args.action_block
    raw_steps = horizon_blocks * args.action_block
    action_dim = args.action_block * 2
    total_actions_per_pair = 4 * args.num_per_source
    started = time.time()

    index = prepare_dataset_index(dataset)
    sampled_rows, eval_episodes, eval_start_steps, analysis = sample_eval_examples(
        dataset,
        index,
        num_eval=args.num_pairs,
        goal_offset_steps=args.offset,
        seed=args.seed,
    )
    valid_action_indices = analysis["valid_indices"]
    rng = np.random.default_rng(args.seed)

    pairs = []
    for pair_index, (row, episode_id, start_step) in enumerate(
        zip(sampled_rows, eval_episodes, eval_start_steps, strict=True)
    ):
        pair_start = time.time()
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

    elapsed = time.time() - started
    return {
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


def load_records(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    records = []
    for pair_index, pair in enumerate(data.get("pairs", [])):
        for action_index, action in enumerate(pair.get("actions", [])):
            record = {
                "pair_index": pair_index,
                "action_index": action_index,
                "episode_id": pair.get("episode_id"),
                "start_step": pair.get("start_step"),
                "goal_step": pair.get("goal_step"),
                "source": action.get("source", "unknown"),
                "success": bool(action.get("success", False)),
                "env_success": bool(action.get("env_success", False)),
            }
            for key in COST_KEYS + ("block_pos_dist", "angle_dist"):
                record[key] = float(action[key])
            records.append(record)
    if not records:
        raise ValueError(f"No action records found in {path}")
    return data, records


def analyze_three_cost_records(
    data: dict,
    records: list[dict],
    *,
    input_path: Path,
    output_path: Path,
    low_corr_threshold: float,
    planner_majority_threshold: float,
) -> dict:
    pair_count = len({record["pair_index"] for record in records})
    action_counts = [len(group) for group in group_records(records, "pair_index").values()]

    results = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "offset": data.get("offset"),
        "counts": {
            "pairs": pair_count,
            "actions": len(records),
            "actions_per_pair": {
                "mean": float(np.mean(action_counts)),
                "min": int(np.min(action_counts)),
                "max": int(np.max(action_counts)),
            },
            "by_source": {
                source: len(group)
                for source, group in sorted(
                    group_records(records, "source").items(),
                    key=lambda item: source_sort_key(item[0]),
                )
            },
        },
    }
    results["correlations"] = compute_correlations(records)
    results["planner_diagnostic"] = compute_planner_diagnostic(records)
    results["ranking_agreement"] = compute_ranking_agreement(records)
    results["summary_statistics"] = compute_summary_stats(records)
    results["classification"] = classify_failure(
        results,
        low_corr_threshold=low_corr_threshold,
        planner_majority_threshold=planner_majority_threshold,
    )
    return results


def rankdata(values: np.ndarray) -> np.ndarray:
    if scipy_stats is not None:
        return scipy_stats.rankdata(values, method="average")

    values = np.asarray(values)
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # Ranks are one-indexed; ties receive the average occupied rank.
        ranks[sorter[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def finite_pair_arrays(records: Iterable[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([record[x_key] for record in records], dtype=np.float64)
    y = np.asarray([record[y_key] for record in records], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return pearson_corr(rankdata(x), rankdata(y))


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    if scipy_stats is not None:
        value = float(scipy_stats.kendalltau(x, y, nan_policy="omit").statistic)
        return value if math.isfinite(value) else None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def summarize_values(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def summarize_corr_values(values: Iterable[float | None]) -> dict:
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_correlations(records: list[dict]) -> dict:
    global_corrs = {}
    for x_key, y_key in METRIC_PAIRS:
        x, y = finite_pair_arrays(records, x_key, y_key)
        global_corrs[f"{x_key}_vs_{y_key}"] = {
            "pearson": pearson_corr(x, y),
            "spearman": spearman_corr(x, y),
            "n": int(len(x)),
        }

    records_by_pair = group_records(records, "pair_index")
    per_pair = []
    per_pair_summary = {}
    for pair_index, pair_records in sorted(records_by_pair.items()):
        entry = {"pair_index": int(pair_index), "n": len(pair_records)}
        for x_key, y_key in METRIC_PAIRS:
            x, y = finite_pair_arrays(pair_records, x_key, y_key)
            name = f"{x_key}_vs_{y_key}"
            entry[name] = {
                "pearson": pearson_corr(x, y),
                "spearman": spearman_corr(x, y),
            }
        per_pair.append(entry)

    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        per_pair_summary[name] = {
            "pearson": summarize_corr_values(entry[name]["pearson"] for entry in per_pair),
            "spearman": summarize_corr_values(entry[name]["spearman"] for entry in per_pair),
        }

    records_by_source = group_records(records, "source")
    per_source = {}
    for source in sorted(records_by_source, key=source_sort_key):
        per_source[source] = {}
        for x_key, y_key in METRIC_PAIRS:
            x, y = finite_pair_arrays(records_by_source[source], x_key, y_key)
            per_source[source][f"{x_key}_vs_{y_key}"] = {
                "pearson": pearson_corr(x, y),
                "spearman": spearman_corr(x, y),
                "n": int(len(x)),
            }

    return {
        "global": global_corrs,
        "per_pair": per_pair,
        "per_pair_summary": per_pair_summary,
        "per_source": per_source,
    }


def compute_planner_diagnostic(records: list[dict]) -> dict:
    records_by_pair = group_records(records, "pair_index")
    pair_entries = []
    for pair_index, pair_records in sorted(records_by_pair.items()):
        successes = int(sum(record["success"] for record in pair_records))
        env_successes = int(sum(record["env_success"] for record in pair_records))
        pair_entries.append(
            {
                "pair_index": int(pair_index),
                "actions": len(pair_records),
                "successful_actions": successes,
                "env_successful_actions": env_successes,
                "has_success": successes > 0,
                "has_env_success": env_successes > 0,
                "best_c_real_state": float(min(record["c_real_state"] for record in pair_records)),
                "best_block_pos_dist": float(min(record["block_pos_dist"] for record in pair_records)),
                "best_angle_dist": float(min(record["angle_dist"] for record in pair_records)),
            }
        )

    num_pairs = len(pair_entries)
    pairs_with_success = sum(entry["has_success"] for entry in pair_entries)
    zero_success_pairs = num_pairs - pairs_with_success
    return {
        "pairs": pair_entries,
        "num_pairs": num_pairs,
        "pairs_with_success": int(pairs_with_success),
        "pairs_with_zero_success": int(zero_success_pairs),
        "fraction_pairs_with_success": float(pairs_with_success / num_pairs),
        "fraction_pairs_with_zero_success": float(zero_success_pairs / num_pairs),
        "successful_actions_total": int(sum(entry["successful_actions"] for entry in pair_entries)),
        "mean_successful_actions_per_pair": float(
            np.mean([entry["successful_actions"] for entry in pair_entries])
        ),
    }


def compute_ranking_agreement(records: list[dict]) -> dict:
    records_by_pair = group_records(records, "pair_index")
    per_pair = []
    for pair_index, pair_records in sorted(records_by_pair.items()):
        entry = {"pair_index": int(pair_index), "n": len(pair_records)}
        for x_key, y_key in METRIC_PAIRS:
            x, y = finite_pair_arrays(pair_records, x_key, y_key)
            entry[f"{x_key}_vs_{y_key}"] = kendall_tau(x, y)
        per_pair.append(entry)

    summary = {}
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        summary[name] = summarize_corr_values(entry[name] for entry in per_pair)

    return {"per_pair": per_pair, "summary": summary}


def compute_summary_stats(records: list[dict]) -> dict:
    stats = {
        "overall": {
            key: summarize_values(record[key] for record in records)
            for key in COST_KEYS + ("block_pos_dist", "angle_dist")
        }
    }

    records_by_source = group_records(records, "source")
    stats["by_source"] = {}
    for source in sorted(records_by_source, key=source_sort_key):
        source_records = records_by_source[source]
        stats["by_source"][source] = {
            key: summarize_values(record[key] for record in source_records)
            for key in COST_KEYS + ("block_pos_dist", "angle_dist")
        }
        stats["by_source"][source]["success_rate"] = success_summary(source_records, "success")
        stats["by_source"][source]["env_success_rate"] = success_summary(source_records, "env_success")

    stats["success_rate"] = success_summary(records, "success")
    stats["env_success_rate"] = success_summary(records, "env_success")
    return stats


def success_summary(records: list[dict], key: str) -> dict:
    total = len(records)
    count = int(sum(record[key] for record in records))
    return {"count": count, "total": total, "rate": float(count / total) if total else None}


def classify_failure(results: dict, low_corr_threshold: float, planner_majority_threshold: float) -> dict:
    global_corr = results["correlations"]["global"]
    encoder_corr = global_corr["c_real_z_vs_c_real_state"]["spearman"]
    predictor_corr = global_corr["c_model_vs_c_real_z"]["spearman"]
    model_state_corr = global_corr["c_model_vs_c_real_state"]["spearman"]
    zero_success_fraction = results["planner_diagnostic"]["fraction_pairs_with_zero_success"]

    encoder_broken = encoder_corr is not None and encoder_corr < low_corr_threshold
    predictor_broken = (
        not encoder_broken
        and predictor_corr is not None
        and predictor_corr < low_corr_threshold
    )
    planner_problem = zero_success_fraction > planner_majority_threshold

    triggered = []
    if encoder_broken:
        triggered.append("Case B")
    if predictor_broken:
        triggered.append("Case C")
    if planner_problem:
        triggered.append("Case D")

    if len(triggered) == 1:
        primary = triggered[0]
    elif len(triggered) > 1:
        primary = "Case F"
    else:
        primary = "Case F"

    explanations = {
        "Case A": "LeWM does not fail; not applicable for this offset sweep context.",
        "Case B": (
            "Encoder goal geometry is suspect because corr(C_real_z, C_real_state) "
            f"is below {low_corr_threshold}."
        ),
        "Case C": (
            "Predictor/rollout is suspect because corr(C_model, C_real_z) is below "
            f"{low_corr_threshold} while encoder geometry is not classified as broken."
        ),
        "Case D": (
            "Planner/action-set coverage is suspect because most pairs have zero "
            "successful actions in the fixed evaluation set."
        ),
        "Case E": "Requires per-trajectory/event-localized diagnostics; not determined here.",
        "Case F": "No single dominant rule fired, or multiple rules fired simultaneously.",
    }

    return {
        "primary": primary,
        "triggered_cases": triggered,
        "thresholds": {
            "low_corr_threshold": low_corr_threshold,
            "planner_majority_threshold": planner_majority_threshold,
        },
        "metrics_used": {
            "spearman_c_real_z_vs_c_real_state": encoder_corr,
            "spearman_c_model_vs_c_real_z": predictor_corr,
            "spearman_c_model_vs_c_real_state": model_state_corr,
            "fraction_pairs_with_zero_success": zero_success_fraction,
        },
        "explanation": explanations[primary],
        "case_notes": explanations,
    }


def group_records(records: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return dict(grouped)


def source_sort_key(source: str) -> tuple[int, str]:
    try:
        return (SOURCE_ORDER.index(source), source)
    except ValueError:
        return (len(SOURCE_ORDER), source)


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "nan"
    return f"{value * 100:.{digits}f}%"


def fmt_mean_std(summary: dict, digits: int = 3) -> str:
    return f"{fmt_float(summary['mean'], digits)} +/- {fmt_float(summary['std'], digits)}"


def build_report(data: dict, results: dict) -> str:
    metadata = data.get("metadata", {})
    correlations = results["correlations"]
    stats = results["summary_statistics"]
    planner = results["planner_diagnostic"]
    ranking = results["ranking_agreement"]
    classification = results["classification"]

    lines = []
    lines.append("Three-Cost Attribution Analysis")
    lines.append("=" * 33)
    lines.append(
        "Input: "
        f"offset={data.get('offset')} pairs={results['counts']['pairs']} "
        f"actions={results['counts']['actions']} "
        f"actions_per_pair={results['counts']['actions_per_pair']}"
    )
    if metadata:
        lines.append(
            "Metadata: "
            f"seed={metadata.get('seed')} device={metadata.get('device')} "
            f"sources={', '.join(metadata.get('sources', []))}"
        )
    lines.append("")

    lines.append("1. Core Attribution Correlations")
    lines.append("Global correlations across all actions:")
    lines.append("  metric_pair                         pearson   spearman   n")
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        entry = correlations["global"][name]
        lines.append(
            f"  {name:<34} {fmt_float(entry['pearson']):>7}   "
            f"{fmt_float(entry['spearman']):>8}   {entry['n']}"
        )
    lines.append("")
    lines.append("Per-pair correlations, mean +/- std over pairs:")
    lines.append("  metric_pair                         pearson          spearman")
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        entry = correlations["per_pair_summary"][name]
        lines.append(
            f"  {name:<34} {fmt_mean_std(entry['pearson']):>15}   "
            f"{fmt_mean_std(entry['spearman']):>15}"
        )
    lines.append("")
    lines.append("Per-source correlations:")
    lines.append("  source      metric_pair                         pearson   spearman   n")
    for source, source_corr in correlations["per_source"].items():
        for x_key, y_key in METRIC_PAIRS:
            name = f"{x_key}_vs_{y_key}"
            entry = source_corr[name]
            lines.append(
                f"  {source:<10} {name:<34} {fmt_float(entry['pearson']):>7}   "
                f"{fmt_float(entry['spearman']):>8}   {entry['n']}"
            )
    lines.append("")

    lines.append("2. Planner Diagnostic")
    lines.append(
        "Pairs with at least one block-success action: "
        f"{planner['pairs_with_success']}/{planner['num_pairs']} "
        f"({fmt_pct(planner['fraction_pairs_with_success'])})"
    )
    lines.append(
        "Pairs with zero block-success actions: "
        f"{planner['pairs_with_zero_success']}/{planner['num_pairs']} "
        f"({fmt_pct(planner['fraction_pairs_with_zero_success'])})"
    )
    lines.append(
        "Successful actions total: "
        f"{planner['successful_actions_total']}/{results['counts']['actions']} "
        f"(mean per pair {planner['mean_successful_actions_per_pair']:.2f})"
    )
    lines.append("")

    lines.append("3. Cost Ranking Agreement")
    lines.append("Kendall tau by pair, mean +/- std:")
    for x_key, y_key in METRIC_PAIRS:
        name = f"{x_key}_vs_{y_key}"
        lines.append(f"  {name:<34} {fmt_mean_std(ranking['summary'][name])}")
    lines.append("")

    lines.append("4. Summary Statistics")
    lines.append("Overall cost and distance distributions:")
    lines.append("  metric             mean       std       min       p25    median       p75       max")
    for key in COST_KEYS + ("block_pos_dist", "angle_dist"):
        entry = stats["overall"][key]
        lines.append(
            f"  {key:<15} {fmt_float(entry['mean'], 3):>8} "
            f"{fmt_float(entry['std'], 3):>9} {fmt_float(entry['min'], 3):>9} "
            f"{fmt_float(entry['p25'], 3):>9} {fmt_float(entry['median'], 3):>9} "
            f"{fmt_float(entry['p75'], 3):>9} {fmt_float(entry['max'], 3):>9}"
        )
    lines.append("")
    lines.append("By-source means/stds and success:")
    lines.append(
        "  source      success  env_succ   c_model       c_real_z      "
        "c_real_state   block_pos     angle"
    )
    for source, source_stats in stats["by_source"].items():
        lines.append(
            f"  {source:<10} "
            f"{fmt_pct(source_stats['success_rate']['rate']):>7} "
            f"{fmt_pct(source_stats['env_success_rate']['rate']):>9} "
            f"{fmt_mean_std(source_stats['c_model']):>13} "
            f"{fmt_mean_std(source_stats['c_real_z']):>13} "
            f"{fmt_mean_std(source_stats['c_real_state']):>14} "
            f"{fmt_mean_std(source_stats['block_pos_dist']):>11} "
            f"{fmt_mean_std(source_stats['angle_dist']):>11}"
        )
    lines.append("")

    lines.append("5. Preliminary Decision Tree Classification")
    lines.append(f"Primary classification: {classification['primary']}")
    triggered = classification["triggered_cases"] or ["none"]
    lines.append(f"Triggered cases: {', '.join(triggered)}")
    for key, value in classification["metrics_used"].items():
        if "fraction" in key:
            lines.append(f"  {key}: {fmt_pct(value)}")
        else:
            lines.append(f"  {key}: {fmt_float(value)}")
    lines.append(f"Interpretation: {classification['explanation']}")
    lines.append("")
    lines.append("Outputs")
    lines.append(f"Detailed JSON: {results['output_path']}")
    if results["figure_paths"]:
        lines.append("Figures:")
        for path in results["figure_paths"]:
            lines.append(f"  {path}")
    else:
        lines.append("Figures: not generated")
    return "\n".join(lines)


def make_plots(records: list[dict], correlations: dict, figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    records_by_source = group_records(records, "source")
    colors = {
        "data": "#2f6f4e",
        "random": "#b45f06",
        "cem_early": "#31688e",
        "cem_late": "#7b3294",
    }

    scatter_specs = [
        ("c_model", "c_real_z", "scatter_cmodel_vs_crealz.png"),
        ("c_model", "c_real_state", "scatter_cmodel_vs_crealstate.png"),
        ("c_real_z", "c_real_state", "scatter_crealz_vs_crealstate.png"),
    ]
    paths = []
    for x_key, y_key, filename in scatter_specs:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        for source in sorted(records_by_source, key=source_sort_key):
            source_records = records_by_source[source]
            ax.scatter(
                [record[x_key] for record in source_records],
                [record[y_key] for record in source_records],
                s=18,
                alpha=0.55,
                color=colors.get(source, "#555555"),
                label=source,
                edgecolors="none",
            )
        name = f"{x_key}_vs_{y_key}"
        corr = correlations["global"][name]
        ax.set_title(f"{x_key} vs {y_key}")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.text(
            0.02,
            0.98,
            f"Pearson={fmt_float(corr['pearson'])}\nSpearman={fmt_float(corr['spearman'])}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#bbbbbb", "alpha": 0.85},
        )
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))

    pair_entries = correlations["per_pair"]
    pair_indices = np.asarray([entry["pair_index"] for entry in pair_entries])
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for offset_idx, (x_key, y_key) in enumerate(METRIC_PAIRS):
        name = f"{x_key}_vs_{y_key}"
        values = [
            np.nan if entry[name]["spearman"] is None else entry[name]["spearman"]
            for entry in pair_entries
        ]
        x_positions = pair_indices + (offset_idx - 1) * width
        ax.bar(x_positions, values, width=width, label=name)
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Per-pair Spearman correlations")
    ax.set_xlabel("pair_index")
    ax.set_ylabel("Spearman correlation")
    ax.set_xticks(pair_indices)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=1)
    fig.tight_layout()
    path = figures_dir / "correlation_by_pair.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))
    return paths


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
