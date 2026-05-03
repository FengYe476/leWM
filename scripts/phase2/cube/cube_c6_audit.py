#!/usr/bin/env python3
"""Cube C6 audit: random-init visual encoders and raw-pixel baselines.

This script replicates the PushT C6 audit on OGBench-Cube. It replays the Cube
Stage 1A action set to cache uint8 terminal/goal pixels once, then runs S1-S6
using the same Stage 1A metric definitions as the PushT audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/OGBCube-v0.
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cube.extract_cube_latents import (  # noqa: E402
    ACTION_BLOCK,
    DEFAULT_ACTION_COUNTS,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_NAME,
    DEFAULT_PAIRS_PATH,
    IMG_SIZE,
    CEM_EARLY_ITERS,
    CEM_LATE_ITERS,
    NUM_SAMPLES,
    RANDOM_WAYPOINTS,
    TOPK,
    VAR_SCALE,
    blocked_normalized_to_raw,
    build_policy,
    build_processors,
    cube_metrics,
    cube_valid_action_indices,
    execute_raw_actions,
    get_dataset,
    infer_raw_action_dim,
    load_pair_rows,
    load_pairs,
    make_policy_namespace,
    parse_action_counts,
    parse_pair_ids,
    prepare_pair_info,
    resolve_device,
    select_action_sequences,
    source_index_for_action,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.stage1a_c6_c7 import (  # noqa: E402
    build_random_model,
    pixels_to_policy_array,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    FALSE_ELITE_K,
    LATENT_DIM,
    TOPK_VALUES,
    aggregate_metric_list,
    clean_float,
    compute_metrics,
    iso_now,
    jsonable,
    squared_l2_torch,
    summary_row,
)


DEFAULT_LATENT_ARTIFACT = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_latents.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "lewm-cube" / "config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2" / "cube" / "c6_audit"
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_BATCH_SIZE = 64
DEFAULT_REPLAY_ATOL = 0.05
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreparedPixels:
    terminal_batches: list[torch.Tensor]
    goal_batches: list[torch.Tensor]
    goal_pair_ids: list[int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PixelCache:
    terminal_pixels: np.ndarray
    goal_pixels: np.ndarray
    goal_pair_ids: np.ndarray
    goal_index_for_row: np.ndarray
    terminal_cube_pos: np.ndarray
    terminal_cube_quat: np.ndarray
    goal_cube_pos: np.ndarray
    goal_cube_quat: np.ndarray
    pair_ids: np.ndarray
    action_ids: np.ndarray
    source_index: np.ndarray
    source: list[str]
    cell: list[str]
    replay_c_real_state: np.ndarray
    validation: dict[str, Any]
    metadata: dict[str, Any]


class SmallRandomCNN(torch.nn.Module):
    """PushT S6 matching random CNN with BatchNorm and 192-d output."""

    def __init__(self, output_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = torch.nn.Linear(192, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return self.projection(features)


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pixel-artifact",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/cube_c6_pixels.pt.",
    )
    parser.add_argument(
        "--action-artifact",
        type=Path,
        default=None,
        help="Optional artifact containing raw_actions or blocked_normalized_actions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--action-block", type=int, default=ACTION_BLOCK)
    parser.add_argument("--random-waypoints", type=int, default=RANDOM_WAYPOINTS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--cem-early-iters", type=int, default=CEM_EARLY_ITERS)
    parser.add_argument("--cem-late-iters", type=int, default=CEM_LATE_ITERS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--var-scale", type=float, default=VAR_SCALE)
    parser.add_argument("--action-counts", type=parse_action_counts, default=DEFAULT_ACTION_COUNTS)
    parser.add_argument("--seeds", type=parse_seed_list, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--arch-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--replay-atol", type=float, default=DEFAULT_REPLAY_ATOL)
    parser.add_argument("--force-replay", action="store_true")
    args = parser.parse_args()
    if args.max_pairs is not None and int(args.max_pairs) < 1:
        parser.error("--max-pairs must be positive")
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    if int(args.arch_batch_size) <= 0:
        parser.error("--arch-batch-size must be positive")
    if int(args.action_block) <= 0:
        parser.error("--action-block must be positive")
    if int(args.topk) > int(args.num_samples):
        parser.error("--topk must be <= --num-samples")
    return args


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def print_metric_table(rows: list[dict], *, title: str) -> None:
    headers = ["Control", "Config", "Seeds", "Spearman", "Pairwise", "PerPairRho", "FalseElite"]
    table = []
    for row in rows:
        table.append(
            [
                str(row["control"]),
                str(row["config"]),
                str(row["n_seeds"]),
                f"{fmt(row['global_spearman_mean'])}/{fmt(row['global_spearman_std'])}",
                f"{fmt(row['pairwise_accuracy_mean'])}/{fmt(row['pairwise_accuracy_std'])}",
                f"{fmt(row['per_pair_rho_mean'])}/{fmt(row['per_pair_rho_mean_std'])}",
                f"{fmt(row['false_elite_rate_mean'])}/{fmt(row['false_elite_rate_std'])}",
            ]
        )
    widths = [max(len(headers[i]), *(len(record[i]) for record in table)) for i in range(len(headers))]
    print(title)
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    headers = ["SubExp", "Metric", "PushT", "Cube", "Delta"]
    table = []
    for row in rows:
        pusht = row.get("pusht_spearman")
        cube = row.get("cube_spearman")
        delta = None if pusht is None or cube is None else float(cube) - float(pusht)
        table.append(
            [
                str(row["sub_experiment"]),
                str(row["metric"]),
                fmt(pusht),
                fmt(cube),
                fmt(delta),
            ]
        )
    widths = [max(len(headers[i]), *(len(record[i]) for record in table)) for i in range(len(headers))]
    print("PushT vs Cube C6 Spearman comparison")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def tensor_to_numpy(value: torch.Tensor, *, dtype: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype)


def load_cube_latent_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Cube latent artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "pair_id",
        "action_id",
        "source",
        "source_index",
        "cell",
        "z_terminal",
        "z_goal",
        "C_real_state",
        "cube_pos_dist",
        "v1_cost",
        "success",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(f"Cube latent artifact missing required keys: {missing}")
    return artifact


def subset_latent_artifact(artifact: dict[str, Any], requested_pair_ids: list[int]) -> dict[str, Any]:
    all_pair_ids = artifact["pair_id"].detach().cpu().numpy().astype(np.int64)
    mask_np = np.isin(all_pair_ids, np.asarray(requested_pair_ids, dtype=np.int64))
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool)
    subset: dict[str, Any] = {}
    n_records = int(len(all_pair_ids))
    for key, value in artifact.items():
        if torch.is_tensor(value) and int(value.shape[0]) == n_records:
            subset[key] = value[mask_t].detach().cpu()
        elif isinstance(value, list) and len(value) == n_records:
            subset[key] = [item for item, keep in zip(value, mask_np.tolist(), strict=True) if keep]
        else:
            subset[key] = value
    subset["metadata"] = dict(artifact.get("metadata", {}))
    subset["metadata"]["subset_pair_ids"] = [int(item) for item in requested_pair_ids]
    subset["metadata"]["subset_n_records"] = int(mask_np.sum())
    return subset


def build_common(artifact: dict[str, Any]) -> dict[str, Any]:
    z_terminal = artifact["z_terminal"].detach().cpu().to(dtype=torch.float32)
    z_goal = artifact["z_goal"].detach().cpu().to(dtype=torch.float32)
    c0_cost_t = squared_l2_torch(z_terminal, z_goal)
    pair_ids = tensor_to_numpy(artifact["pair_id"], dtype=np.int64)
    return {
        "labels": tensor_to_numpy(artifact["C_real_state"], dtype=np.float64),
        "v1_cost": tensor_to_numpy(artifact["v1_cost"], dtype=np.float64),
        "c0_cost": c0_cost_t.detach().cpu().numpy().astype(np.float64),
        "success": tensor_to_numpy(artifact["success"], dtype=bool),
        "pair_ids": pair_ids,
        "action_ids": tensor_to_numpy(artifact["action_id"], dtype=np.int64),
        "source_index": tensor_to_numpy(artifact["source_index"], dtype=np.int64),
        "source": list(artifact["source"]),
        "cells": np.asarray(artifact["cell"], dtype=object),
        "anchor_masks": {},
    }


def compute_cost_metrics(costs: np.ndarray | torch.Tensor, common: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if torch.is_tensor(costs):
        costs_np = costs.detach().cpu().numpy().astype(np.float64)
    else:
        costs_np = np.asarray(costs, dtype=np.float64)
    finite_mask = np.isfinite(costs_np)
    if not finite_mask.any():
        raise RuntimeError("No finite costs available for metric computation")
    metrics = compute_metrics(
        costs=costs_np[finite_mask],
        labels=common["labels"][finite_mask],
        v1_cost=common["v1_cost"][finite_mask],
        c0_cost=common["c0_cost"][finite_mask],
        success=common["success"][finite_mask],
        pair_ids=common["pair_ids"][finite_mask],
        action_ids=common["action_ids"][finite_mask],
        cells=common["cells"][finite_mask],
        anchor_masks={name: mask[finite_mask] for name, mask in common["anchor_masks"].items()},
    )
    return metrics, {
        "n_original": int(len(costs_np)),
        "n_finite": int(finite_mask.sum()),
        "n_nan": int((~finite_mask).sum()),
    }


def seed_validation(cost: torch.Tensor, z_terminal: torch.Tensor, z_goal: torch.Tensor) -> dict[str, Any]:
    return {
        "cost_shape": list(cost.shape),
        "cost_finite": bool(torch.isfinite(cost).all()),
        "cost_min": clean_float(cost.min().item()),
        "cost_max": clean_float(cost.max().item()),
        "cost_mean": clean_float(cost.mean().item()),
        "z_terminal_shape": list(z_terminal.shape),
        "z_goal_shape": list(z_goal.shape),
        "z_terminal_finite": bool(torch.isfinite(z_terminal).all()),
        "z_goal_finite": bool(torch.isfinite(z_goal).all()),
    }


def finite_stats(costs: np.ndarray) -> dict[str, Any]:
    costs = np.asarray(costs, dtype=np.float64)
    finite = costs[np.isfinite(costs)]
    return {
        "shape": list(costs.shape),
        "finite": bool(np.isfinite(costs).all()),
        "n_finite": int(len(finite)),
        "n_nan": int(np.isnan(costs).sum()),
        "min": clean_float(float(finite.min())) if len(finite) else None,
        "max": clean_float(float(finite.max())) if len(finite) else None,
        "mean": clean_float(float(finite.mean())) if len(finite) else None,
    }


def result_block_for_costs(
    *,
    name: str,
    costs: np.ndarray,
    common: dict[str, Any],
    cost_metadata: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics, filter_metadata = compute_cost_metrics(costs, common)
    return {
        "cost_metadata": {**cost_metadata, "finite_filter": filter_metadata, "name": name},
        "validation": validation or finite_stats(costs),
        "metrics": metrics,
    }


def result_block_for_embeddings(
    *,
    z_terminal: torch.Tensor,
    z_goal: torch.Tensor,
    encode_metadata: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    cost = squared_l2_torch(z_terminal, z_goal)
    metrics, _ = compute_cost_metrics(cost, common)
    return {
        "encode_metadata": {
            **encode_metadata,
            "embedding_shape": [int(z_terminal.shape[1])],
            "z_terminal_shape": list(z_terminal.shape),
            "z_goal_shape": list(z_goal.shape),
        },
        "validation": seed_validation(cost, z_terminal, z_goal),
        "metrics": metrics,
    }


def hwc_uint8(pixels: Any) -> np.ndarray:
    arr = pixels_to_policy_array(pixels)
    if arr.ndim != 3 or int(arr.shape[-1]) != 3:
        raise ValueError(f"Expected HWC RGB pixels, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def goal_index_for_rows(pair_ids: np.ndarray, goal_pair_ids: np.ndarray) -> np.ndarray:
    index_by_pair = {int(pair_id): idx for idx, pair_id in enumerate(goal_pair_ids.tolist())}
    return np.asarray([index_by_pair[int(pair_id)] for pair_id in pair_ids.tolist()], dtype=np.int64)


def pair_action_index(artifact: dict[str, Any]) -> dict[tuple[int, int], int]:
    pair_ids = artifact["pair_id"].detach().cpu().numpy().astype(np.int64)
    action_ids = artifact["action_id"].detach().cpu().numpy().astype(np.int64)
    return {
        (int(pair_id), int(action_id)): idx
        for idx, (pair_id, action_id) in enumerate(zip(pair_ids, action_ids, strict=True))
    }


def validate_replay_order(cache: PixelCache, artifact: dict[str, Any], *, atol: float) -> dict[str, Any]:
    expected_pair_ids = tensor_to_numpy(artifact["pair_id"], dtype=np.int64)
    expected_action_ids = tensor_to_numpy(artifact["action_id"], dtype=np.int64)
    expected_source_index = tensor_to_numpy(artifact["source_index"], dtype=np.int64)
    expected_source = list(artifact["source"])
    expected_cell = list(artifact["cell"])
    order_checks = {
        "pair_id_matches": bool(np.array_equal(cache.pair_ids, expected_pair_ids)),
        "action_id_matches": bool(np.array_equal(cache.action_ids, expected_action_ids)),
        "source_index_matches": bool(np.array_equal(cache.source_index, expected_source_index)),
        "source_matches": bool(cache.source == expected_source),
        "cell_matches": bool(cache.cell == expected_cell),
    }
    labels = tensor_to_numpy(artifact["C_real_state"], dtype=np.float64)
    diffs = np.abs(cache.replay_c_real_state.astype(np.float64) - labels)
    by_pair = {}
    for pair_id in np.unique(expected_pair_ids):
        mask = expected_pair_ids == pair_id
        by_pair[str(int(pair_id))] = {
            "max_abs_diff": clean_float(float(diffs[mask].max())),
            "mean_abs_diff": clean_float(float(diffs[mask].mean())),
            "n_records": int(mask.sum()),
        }
    validation = {
        "order": order_checks,
        "c_real_state": {
            "max_abs_diff": clean_float(float(diffs.max())) if len(diffs) else None,
            "mean_abs_diff": clean_float(float(diffs.mean())) if len(diffs) else None,
            "atol": float(atol),
            "passed": bool(len(diffs) > 0 and float(diffs.max()) <= float(atol)),
            "by_pair": by_pair,
        },
    }
    validation["passed"] = bool(all(order_checks.values()) and validation["c_real_state"]["passed"])
    return validation


def print_replay_validation(validation: dict[str, Any]) -> None:
    print("Cube C6 replay/cache validation:")
    for key, value in validation["order"].items():
        print(f"  {key}: {value}")
    c_real = validation["c_real_state"]
    print(
        "  C_real_state max_abs_diff="
        f"{c_real['max_abs_diff']:.6g} mean_abs_diff={c_real['mean_abs_diff']:.6g} "
        f"atol={c_real['atol']:.6g} passed={c_real['passed']}"
    )
    worst = sorted(
        c_real["by_pair"].items(),
        key=lambda item: float(item[1]["max_abs_diff"]),
        reverse=True,
    )[:10]
    print("  worst per-pair C_real_state diffs:")
    for pair_id, stats in worst:
        print(
            f"    pair_id={pair_id} max={stats['max_abs_diff']:.6g} "
            f"mean={stats['mean_abs_diff']:.6g} n={stats['n_records']}"
        )


def save_pixel_cache(path: Path, cache: PixelCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": cache.metadata,
        "validation": cache.validation,
        "pair_id": torch.as_tensor(cache.pair_ids, dtype=torch.long),
        "action_id": torch.as_tensor(cache.action_ids, dtype=torch.long),
        "source": list(cache.source),
        "source_index": torch.as_tensor(cache.source_index, dtype=torch.long),
        "cell": list(cache.cell),
        "terminal_pixels": torch.as_tensor(cache.terminal_pixels, dtype=torch.uint8),
        "goal_pair_id": torch.as_tensor(cache.goal_pair_ids, dtype=torch.long),
        "goal_pixels": torch.as_tensor(cache.goal_pixels, dtype=torch.uint8),
        "goal_index_for_row": torch.as_tensor(cache.goal_index_for_row, dtype=torch.long),
        "terminal_cube_pos": torch.as_tensor(cache.terminal_cube_pos, dtype=torch.float64),
        "terminal_cube_quat": torch.as_tensor(cache.terminal_cube_quat, dtype=torch.float64),
        "goal_cube_pos": torch.as_tensor(cache.goal_cube_pos, dtype=torch.float64),
        "goal_cube_quat": torch.as_tensor(cache.goal_cube_quat, dtype=torch.float64),
        "replay_c_real_state": torch.as_tensor(cache.replay_c_real_state, dtype=torch.float64),
    }
    torch.save(payload, path)


def load_pixel_cache(path: Path, artifact: dict[str, Any], *, atol: float) -> PixelCache | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = (
        "pair_id",
        "action_id",
        "source",
        "source_index",
        "cell",
        "terminal_pixels",
        "goal_pair_id",
        "goal_pixels",
        "terminal_cube_pos",
        "terminal_cube_quat",
        "goal_cube_pos",
        "goal_cube_quat",
        "replay_c_real_state",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Pixel cache missing required keys: {missing}")

    desired_keys = pair_action_index(artifact)
    cached_pair_ids = tensor_to_numpy(payload["pair_id"], dtype=np.int64)
    cached_action_ids = tensor_to_numpy(payload["action_id"], dtype=np.int64)
    cached_index = {
        (int(pair_id), int(action_id)): idx
        for idx, (pair_id, action_id) in enumerate(zip(cached_pair_ids, cached_action_ids, strict=True))
    }
    missing_keys = sorted(set(desired_keys).difference(cached_index))
    if missing_keys:
        raise RuntimeError(f"Pixel cache missing requested pair/action rows: {missing_keys[:10]}")
    take = np.asarray([cached_index[key] for key in desired_keys.keys()], dtype=np.int64)
    goal_pair_ids = tensor_to_numpy(payload["goal_pair_id"], dtype=np.int64)
    goal_idx = payload.get("goal_index_for_row")
    if goal_idx is None:
        goal_idx_np = goal_index_for_rows(cached_pair_ids, goal_pair_ids)
    else:
        goal_idx_np = tensor_to_numpy(goal_idx, dtype=np.int64)

    cache = PixelCache(
        terminal_pixels=payload["terminal_pixels"][take].detach().cpu().numpy().astype(np.uint8, copy=False),
        goal_pixels=payload["goal_pixels"].detach().cpu().numpy().astype(np.uint8, copy=False),
        goal_pair_ids=goal_pair_ids,
        goal_index_for_row=goal_idx_np[take],
        terminal_cube_pos=payload["terminal_cube_pos"][take].detach().cpu().numpy().astype(np.float64),
        terminal_cube_quat=payload["terminal_cube_quat"][take].detach().cpu().numpy().astype(np.float64),
        goal_cube_pos=payload["goal_cube_pos"].detach().cpu().numpy().astype(np.float64),
        goal_cube_quat=payload["goal_cube_quat"].detach().cpu().numpy().astype(np.float64),
        pair_ids=cached_pair_ids[take],
        action_ids=cached_action_ids[take],
        source_index=tensor_to_numpy(payload["source_index"], dtype=np.int64)[take],
        source=[payload["source"][int(idx)] for idx in take.tolist()],
        cell=[payload["cell"][int(idx)] for idx in take.tolist()],
        replay_c_real_state=tensor_to_numpy(payload["replay_c_real_state"], dtype=np.float64)[take],
        validation={},
        metadata={
            **payload.get("metadata", {}),
            "loaded_from_cache": True,
            "cache_path": str(path),
            "subset_from_cached_records": int(len(take)),
        },
    )
    validation = validate_replay_order(cache, artifact, atol=atol)
    return PixelCache(**{**cache.__dict__, "validation": validation})


def find_action_arrays(artifact: dict[str, Any]) -> tuple[Any | None, Any | None]:
    raw_keys = ("raw_actions", "action_raw", "raw_action_sequences", "actions_raw")
    blocked_keys = ("blocked_normalized_actions", "blocked_actions", "action_blocked", "blocked_normalized")
    raw = next((artifact[key] for key in raw_keys if key in artifact), None)
    blocked = next((artifact[key] for key in blocked_keys if key in artifact), None)
    return raw, blocked


def action_array_to_numpy(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_action_artifact(args: argparse.Namespace, latent_artifact: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [latent_artifact]
    if args.action_artifact is not None:
        if not args.action_artifact.exists():
            raise FileNotFoundError(f"Missing action artifact: {args.action_artifact}")
        candidates.append(torch.load(args.action_artifact, map_location="cpu", weights_only=False))
    for artifact in candidates:
        raw, blocked = find_action_arrays(artifact)
        raw_np = action_array_to_numpy(raw)
        blocked_np = action_array_to_numpy(blocked)
        if raw_np is None and blocked_np is None:
            continue
        if "pair_id" not in artifact or "action_id" not in artifact:
            raise KeyError("Action artifact must include pair_id and action_id")
        pair_ids = action_array_to_numpy(artifact["pair_id"]).astype(np.int64)
        action_ids = action_array_to_numpy(artifact["action_id"]).astype(np.int64)
        index = {
            (int(pair_id), int(action_id)): idx
            for idx, (pair_id, action_id) in enumerate(zip(pair_ids, action_ids, strict=True))
        }
        print("Using stored action sequences from artifact instead of regenerating actions.")
        return {
            "raw": raw_np,
            "blocked": blocked_np,
            "index": index,
            "metadata": artifact.get("metadata", {}),
        }
    return None


def actions_for_pair_from_cache(
    *,
    action_cache: dict[str, Any],
    artifact: dict[str, Any],
    pair_id: int,
    action_processor: Any,
    action_block: int,
    raw_action_dim: int,
) -> list[dict[str, Any]]:
    pair_ids = tensor_to_numpy(artifact["pair_id"], dtype=np.int64)
    action_ids = tensor_to_numpy(artifact["action_id"], dtype=np.int64)
    source_index = tensor_to_numpy(artifact["source_index"], dtype=np.int64)
    source = list(artifact["source"])
    rows = np.flatnonzero(pair_ids == int(pair_id))
    sequences = []
    for row in rows:
        action_id = int(action_ids[row])
        key = (int(pair_id), action_id)
        if key not in action_cache["index"]:
            raise RuntimeError(f"Stored action artifact missing pair/action {key}")
        idx = int(action_cache["index"][key])
        if action_cache["raw"] is not None:
            raw = np.asarray(action_cache["raw"][idx], dtype=np.float32)
        else:
            raw = blocked_normalized_to_raw(
                np.asarray(action_cache["blocked"][idx], dtype=np.float32),
                action_processor=action_processor,
                action_block=action_block,
                raw_action_dim=raw_action_dim,
            )
        sequences.append(
            {
                "source": str(source[row]),
                "source_index": int(source_index[row]),
                "raw": raw,
            }
        )
    return sequences


def replay_or_load_pixels(args: argparse.Namespace, latent_artifact: dict[str, Any], requested_pairs: list[dict]) -> PixelCache:
    if not args.force_replay:
        cached = load_pixel_cache(args.pixel_artifact, latent_artifact, atol=float(args.replay_atol))
        if cached is not None:
            print(f"Loaded Cube C6 pixel cache: {args.pixel_artifact}")
            print_replay_validation(cached.validation)
            if not cached.validation["passed"]:
                raise RuntimeError(f"Cached pixel replay validation failed: {cached.validation}")
            return cached

    print("Preparing Cube C6 pixels by simulator replay...")
    dataset = get_dataset(args.cache_dir, args.dataset_name)
    raw_action_dim = infer_raw_action_dim(dataset)
    if int(raw_action_dim) != 5:
        raise ValueError(f"Expected Cube raw action dim 5, got {raw_action_dim}")
    offset = int(requested_pairs[0]["goal_row"]) - int(requested_pairs[0]["start_row"])
    pairs_metadata = json.loads(args.pairs_path.read_text()).get("metadata", {})
    offset = int(pairs_metadata.get("offset", offset))
    if offset % int(args.action_block) != 0:
        raise ValueError("Cube offset must be divisible by --action-block")
    validate_requested_pair_offsets(requested_pairs, offset=offset)

    valid_action_indices = cube_valid_action_indices(dataset, offset=offset)
    process = build_processors(dataset, ["action"])
    policy = build_policy(make_policy_namespace(args), process)
    model = policy.solver.model
    action_cache = load_action_artifact(args, latent_artifact)
    expected_by_key = pair_action_index(latent_artifact)

    terminal_pixels = []
    terminal_cube_pos = []
    terminal_cube_quat = []
    replay_c_real_state = []
    pair_ids = []
    action_ids = []
    source = []
    source_index = []
    cell = []
    goal_pixels_by_pair: dict[int, np.ndarray] = {}
    goal_pos_by_pair: dict[int, np.ndarray] = {}
    goal_quat_by_pair: dict[int, np.ndarray] = {}

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
    started = time.time()
    try:
        for pair_idx, pair_spec in enumerate(requested_pairs):
            pair_started = time.time()
            pair_id = int(pair_spec["pair_id"])
            pair_rows = load_pair_rows(dataset, int(pair_spec["start_row"]), int(pair_spec["goal_row"]))
            initial = pair_rows["initial"]
            goal = pair_rows["goal"]
            goal_pixels_by_pair[pair_id] = hwc_uint8(goal["pixels"])
            goal_pos = np.asarray(goal["privileged_block_0_pos"], dtype=np.float64)
            goal_quat = np.asarray(goal["privileged_block_0_quat"], dtype=np.float64)
            goal_pos_by_pair[pair_id] = goal_pos
            goal_quat_by_pair[pair_id] = goal_quat

            if action_cache is not None:
                sequences = actions_for_pair_from_cache(
                    action_cache=action_cache,
                    artifact=latent_artifact,
                    pair_id=pair_id,
                    action_processor=policy.process["action"],
                    action_block=int(args.action_block),
                    raw_action_dim=raw_action_dim,
                )
                action_source = "stored_action_artifact"
            else:
                prepared_info = prepare_pair_info(
                    policy,
                    initial["pixels"],
                    goal["pixels"],
                    raw_action_dim=raw_action_dim,
                )
                sequences = select_action_sequences(
                    dataset=dataset,
                    valid_action_indices=valid_action_indices,
                    policy=policy,
                    model=model,
                    prepared_info=prepared_info,
                    args=args,
                    pair_id=pair_id,
                    raw_steps=offset,
                    action_counts=args.action_counts,
                    raw_action_dim=raw_action_dim,
                )
                action_source = "regenerated"

            expected_actions = int(np.sum(tensor_to_numpy(latent_artifact["pair_id"], dtype=np.int64) == pair_id))
            if len(sequences) != expected_actions:
                raise RuntimeError(f"Pair {pair_id} produced {len(sequences)} actions, expected {expected_actions}")

            pair_replay_costs = []
            counts: dict[str, int] = {}
            for action_id, sequence in enumerate(sequences):
                sequence_source = str(sequence["source"])
                if "source_index" in sequence:
                    sequence_source_index = int(sequence["source_index"])
                    counts[sequence_source] = max(counts.get(sequence_source, 0), sequence_source_index + 1)
                else:
                    sequence_source_index = source_index_for_action(counts, sequence_source)
                if (pair_id, action_id) not in expected_by_key:
                    raise RuntimeError(f"Unexpected regenerated pair/action {(pair_id, action_id)}")
                rollout = execute_raw_actions(
                    env,
                    initial=initial,
                    goal=goal,
                    raw_actions=np.asarray(sequence["raw"], dtype=np.float32),
                    seed=int(args.seed) + pair_id * 10_000 + action_id,
                )
                metrics = cube_metrics(
                    terminal_pos=rollout["terminal_cube_pos"],
                    terminal_quat=rollout["terminal_cube_quat"],
                    goal_pos=goal_pos,
                    goal_quat=goal_quat,
                )
                terminal_pixels.append(hwc_uint8(rollout["terminal_pixels"]))
                terminal_cube_pos.append(np.asarray(rollout["terminal_cube_pos"], dtype=np.float64))
                terminal_cube_quat.append(np.asarray(rollout["terminal_cube_quat"], dtype=np.float64))
                replay_cost = float(metrics["C_real_state"])
                replay_c_real_state.append(replay_cost)
                pair_replay_costs.append(replay_cost)
                pair_ids.append(pair_id)
                action_ids.append(int(action_id))
                source.append(sequence_source)
                source_index.append(sequence_source_index)
                cell.append(str(pair_spec["cell"]))

            expected_mask = tensor_to_numpy(latent_artifact["pair_id"], dtype=np.int64) == pair_id
            expected_labels = tensor_to_numpy(latent_artifact["C_real_state"], dtype=np.float64)[expected_mask]
            pair_diffs = np.abs(np.asarray(pair_replay_costs, dtype=np.float64) - expected_labels)
            print(
                f"Cube C6 replay pair {pair_idx + 1}/{len(requested_pairs)} "
                f"pair_id={pair_id} action_source={action_source} "
                f"max_C_real_diff={pair_diffs.max():.6g} elapsed={time.time() - pair_started:.2f}s"
            )
    finally:
        env.close()

    goal_pair_ids = np.asarray(sorted(goal_pixels_by_pair), dtype=np.int64)
    goal_pixels = np.stack([goal_pixels_by_pair[int(pair_id)] for pair_id in goal_pair_ids]).astype(np.uint8, copy=False)
    goal_cube_pos = np.stack([goal_pos_by_pair[int(pair_id)] for pair_id in goal_pair_ids]).astype(np.float64, copy=False)
    goal_cube_quat = np.stack([goal_quat_by_pair[int(pair_id)] for pair_id in goal_pair_ids]).astype(np.float64, copy=False)
    pair_ids_np = np.asarray(pair_ids, dtype=np.int64)
    cache = PixelCache(
        terminal_pixels=np.stack(terminal_pixels).astype(np.uint8, copy=False),
        goal_pixels=goal_pixels,
        goal_pair_ids=goal_pair_ids,
        goal_index_for_row=goal_index_for_rows(pair_ids_np, goal_pair_ids),
        terminal_cube_pos=np.stack(terminal_cube_pos).astype(np.float64, copy=False),
        terminal_cube_quat=np.stack(terminal_cube_quat).astype(np.float64, copy=False),
        goal_cube_pos=goal_cube_pos,
        goal_cube_quat=goal_cube_quat,
        pair_ids=pair_ids_np,
        action_ids=np.asarray(action_ids, dtype=np.int64),
        source_index=np.asarray(source_index, dtype=np.int64),
        source=source,
        cell=cell,
        replay_c_real_state=np.asarray(replay_c_real_state, dtype=np.float64),
        validation={},
        metadata={
            "format": "cube_c6_pixels",
            "created_at": iso_now(),
            "source": "simulator_replay",
            "action_source": "stored_action_artifact" if action_cache is not None else "regenerated",
            "pixel_dtype": "uint8",
            "terminal_pixel_shape": [len(terminal_pixels), int(args.img_size), int(args.img_size), 3],
            "goal_pixel_shape": list(goal_pixels.shape),
            "pairs_path": str(args.pairs_path),
            "latent_artifact": str(args.latent_artifact),
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_name": args.dataset_name,
            "cache_dir": str(args.cache_dir),
            "seed": int(args.seed),
            "device": args.device,
            "action_counts": dict(args.action_counts),
            "wallclock_seconds": clean_float(time.time() - started),
        },
    )
    validation = validate_replay_order(cache, latent_artifact, atol=float(args.replay_atol))
    cache = PixelCache(**{**cache.__dict__, "validation": validation})
    print_replay_validation(cache.validation)
    save_pixel_cache(args.pixel_artifact, cache)
    print(f"Saved Cube C6 pixel cache: {args.pixel_artifact}")
    if not cache.validation["passed"]:
        raise RuntimeError(f"Replay validation failed after saving cache: {cache.validation}")
    return cache


def prepare_policy_tensor_batch(*, policy: Any, pixels: np.ndarray) -> torch.Tensor:
    if int(pixels.shape[0]) == 0:
        return torch.empty((0,), dtype=torch.float32)
    batch = np.stack([pixels_to_policy_array(item) for item in pixels])
    prepared = policy._prepare_info({"pixels": batch[:, None, ...]})
    return prepared["pixels"].detach().cpu().to(dtype=torch.float32)


def prepare_vit_pixels(*, args: argparse.Namespace, cache: PixelCache) -> PreparedPixels:
    process = build_processors(get_dataset(args.cache_dir, args.dataset_name), ["action"])
    policy = build_policy(make_policy_namespace(args), process)
    terminal_batches = []
    for start in range(0, int(cache.terminal_pixels.shape[0]), int(args.batch_size)):
        end = min(start + int(args.batch_size), int(cache.terminal_pixels.shape[0]))
        terminal_batches.append(prepare_policy_tensor_batch(policy=policy, pixels=cache.terminal_pixels[start:end]))
    goal_batches = []
    for start in range(0, int(cache.goal_pixels.shape[0]), int(args.batch_size)):
        end = min(start + int(args.batch_size), int(cache.goal_pixels.shape[0]))
        goal_batches.append(prepare_policy_tensor_batch(policy=policy, pixels=cache.goal_pixels[start:end]))
    return PreparedPixels(
        terminal_batches=terminal_batches,
        goal_batches=goal_batches,
        goal_pair_ids=[int(item) for item in cache.goal_pair_ids.tolist()],
        metadata={
            "device": args.device,
            "prepared_once_before_seed_loop": True,
            "policy_tensor_dtype": "torch.float32",
            "terminal_batches": int(len(terminal_batches)),
            "goal_batches": int(len(goal_batches)),
            "batch_size": int(args.batch_size),
            "terminal_batch_shapes": [list(batch.shape) for batch in terminal_batches],
            "goal_batch_shapes": [list(batch.shape) for batch in goal_batches],
        },
    )


@torch.inference_mode()
def encode_prepared_batch(*, model: torch.nn.Module, pixels_t: torch.Tensor, device: str) -> torch.Tensor:
    encoded = model.encode({"pixels": pixels_t.to(device=device, dtype=torch.float32)})
    if "emb" not in encoded:
        raise KeyError('random_model.encode(...) did not return key "emb"')
    emb = encoded["emb"]
    if emb.ndim == 3:
        emb = emb[:, -1, :]
    elif emb.ndim != 2:
        raise ValueError(f"Unexpected encode emb shape: {tuple(emb.shape)}")
    if not torch.isfinite(emb).all():
        raise ValueError("Embeddings contain non-finite values")
    return emb.detach().cpu().to(dtype=torch.float32)


def goal_stack_from_unique(
    *,
    row_pair_ids: np.ndarray,
    goal_pair_ids: list[int],
    goal_emb_by_pair: dict[int, torch.Tensor],
) -> torch.Tensor:
    return torch.stack([goal_emb_by_pair[int(pair_id)] for pair_id in row_pair_ids.tolist()])


def encode_seed_embeddings(
    *,
    args: argparse.Namespace,
    seed: int,
    row_pair_ids: np.ndarray,
    prepared: PreparedPixels,
    mode: str = "eval",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    started = time.time()
    model = build_random_model(config_path=args.config_path, device=args.device, seed=int(seed))
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
        torch.manual_seed(int(seed))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    terminal_chunks = [
        encode_prepared_batch(model=model, pixels_t=batch, device=args.device)
        for batch in prepared.terminal_batches
    ]
    z_terminal = torch.cat(terminal_chunks, dim=0)
    goal_emb_by_pair: dict[int, torch.Tensor] = {}
    pair_offset = 0
    for batch in prepared.goal_batches:
        goal_emb = encode_prepared_batch(model=model, pixels_t=batch, device=args.device)
        batch_pair_ids = prepared.goal_pair_ids[pair_offset : pair_offset + int(goal_emb.shape[0])]
        for local_idx, pair_id in enumerate(batch_pair_ids):
            goal_emb_by_pair[int(pair_id)] = goal_emb[local_idx]
        pair_offset += int(goal_emb.shape[0])
    z_goal = goal_stack_from_unique(
        row_pair_ids=row_pair_ids,
        goal_pair_ids=prepared.goal_pair_ids,
        goal_emb_by_pair=goal_emb_by_pair,
    )
    metadata = {
        "seed": int(seed),
        "model_mode": mode,
        "checkpoint_weights_loaded": False,
        "config_path": str(args.config_path),
        "device": args.device,
        "pixel_source": "cube_c6_pixel_cache",
        "encode_path": 'model.encode({"pixels": pixels_t})["emb"]',
        "terminal_batches_encoded": int(len(prepared.terminal_batches)),
        "goal_batches_encoded": int(len(prepared.goal_batches)),
        "wallclock_seconds": clean_float(time.time() - started),
    }
    del model
    empty_accelerator_cache()
    return z_terminal, z_goal, metadata


@torch.inference_mode()
def encode_projector_variants_batch(
    *,
    model: torch.nn.Module,
    pixels_t: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pixels_t.ndim < 5:
        raise ValueError(f"Expected pixels shaped [B, T, ...], got {tuple(pixels_t.shape)}")
    batch_size = int(pixels_t.shape[0])
    time_steps = int(pixels_t.shape[1])
    pixels = pixels_t.to(device=device, dtype=torch.float32)
    flat_pixels = pixels.reshape(batch_size * time_steps, *pixels.shape[2:])
    encoder_out = model.encoder(flat_pixels, interpolate_pos_encoding=True)
    cls = encoder_out.last_hidden_state[:, 0]
    post = model.projector(cls)
    pre = cls.reshape(batch_size, time_steps, -1)[:, -1, :]
    post = post.reshape(batch_size, time_steps, -1)[:, -1, :]
    return pre.detach().cpu().to(dtype=torch.float32), post.detach().cpu().to(dtype=torch.float32)


def encode_projector_variants(
    *,
    args: argparse.Namespace,
    seed: int,
    row_pair_ids: np.ndarray,
    prepared: PreparedPixels,
    mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    started = time.time()
    model = build_random_model(config_path=args.config_path, device=args.device, seed=int(seed))
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
        torch.manual_seed(int(seed))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    terminal_pre_chunks = []
    terminal_post_chunks = []
    for batch in prepared.terminal_batches:
        pre, post = encode_projector_variants_batch(model=model, pixels_t=batch, device=args.device)
        terminal_pre_chunks.append(pre)
        terminal_post_chunks.append(post)

    goal_pre_by_pair: dict[int, torch.Tensor] = {}
    goal_post_by_pair: dict[int, torch.Tensor] = {}
    pair_offset = 0
    for batch in prepared.goal_batches:
        pre, post = encode_projector_variants_batch(model=model, pixels_t=batch, device=args.device)
        batch_pair_ids = prepared.goal_pair_ids[pair_offset : pair_offset + int(pre.shape[0])]
        for local_idx, pair_id in enumerate(batch_pair_ids):
            goal_pre_by_pair[int(pair_id)] = pre[local_idx]
            goal_post_by_pair[int(pair_id)] = post[local_idx]
        pair_offset += int(pre.shape[0])
    features = {
        "pre_projector_terminal": torch.cat(terminal_pre_chunks, dim=0),
        "pre_projector_goal": goal_stack_from_unique(
            row_pair_ids=row_pair_ids,
            goal_pair_ids=prepared.goal_pair_ids,
            goal_emb_by_pair=goal_pre_by_pair,
        ),
        "post_projector_terminal": torch.cat(terminal_post_chunks, dim=0),
        "post_projector_goal": goal_stack_from_unique(
            row_pair_ids=row_pair_ids,
            goal_pair_ids=prepared.goal_pair_ids,
            goal_emb_by_pair=goal_post_by_pair,
        ),
    }
    metadata = {
        "seed": int(seed),
        "model_mode": mode,
        "checkpoint_weights_loaded": False,
        "config_path": str(args.config_path),
        "device": args.device,
        "pixel_source": "cube_c6_pixel_cache",
        "encoder_call": "model.encoder(flat_pixels, interpolate_pos_encoding=True)",
        "pre_projector_feature": "encoder_out.last_hidden_state[:, 0]",
        "post_projector_feature": "model.projector(cls)",
        "terminal_batches_encoded": int(len(prepared.terminal_batches)),
        "goal_batches_encoded": int(len(prepared.goal_batches)),
        "wallclock_seconds": clean_float(time.time() - started),
    }
    del model
    empty_accelerator_cache()
    return features, metadata


def empty_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_s1(args: argparse.Namespace, artifact: dict[str, Any], common: dict[str, Any], prepared: PreparedPixels) -> dict[str, Any]:
    per_seed = []
    summary_rows = []
    for seed in args.seeds:
        print(f"Encoding Cube C6 S1 random-init LeWM seed={seed}...")
        z_terminal, z_goal, metadata = encode_seed_embeddings(
            args=args,
            seed=int(seed),
            row_pair_ids=common["pair_ids"],
            prepared=prepared,
            mode="eval",
        )
        block = result_block_for_embeddings(
            z_terminal=z_terminal,
            z_goal=z_goal,
            encode_metadata=metadata,
            common=common,
        )
        per_seed.append({"seed": int(seed), **block})
        summary_rows.append(summary_row(control="Cube_C6_S1", config=f"random_init_seed={int(seed)}", n_seeds=1, metrics=block["metrics"]))
        del z_terminal, z_goal

    aggregate = aggregate_metric_list([item["metrics"] for item in per_seed])
    negative_seeds = [
        int(item["seed"])
        for item in per_seed
        if item["metrics"].get("global_spearman") is not None and float(item["metrics"]["global_spearman"]) < 0
    ]
    summary_rows.append(summary_row(control="Cube_C6_S1", config="random_init_10seed", n_seeds=len(args.seeds), aggregate=aggregate))
    output = {
        "metadata": {
            "format": "cube_c6_audit_s1_random_init_10seed",
            "created_at": iso_now(),
            "sub_experiment": "S1",
            "latent_artifact": str(args.latent_artifact),
            "config_path": str(args.config_path),
            "checkpoint_weights_loaded": False,
            "seeds": [int(seed) for seed in args.seeds],
            "n_records": int(len(common["pair_ids"])),
            "n_pairs": int(len(np.unique(common["pair_ids"]))),
            "latent_dim": LATENT_DIM,
            "target_metric": "C_real_state",
            "negative_seed_count": int(len(negative_seeds)),
            "negative_seeds": negative_seeds,
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "pixel_preparation": prepared.metadata,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "summary_table": summary_rows,
    }
    return output


def run_s2_s3(args: argparse.Namespace, common: dict[str, Any], prepared: PreparedPixels) -> dict[str, Any]:
    print("Encoding Cube C6 S2 eval-mode pre/post projector features...")
    eval_features, eval_metadata = encode_projector_variants(
        args=args,
        seed=0,
        row_pair_ids=common["pair_ids"],
        prepared=prepared,
        mode="eval",
    )
    s2_pre = result_block_for_embeddings(
        z_terminal=eval_features["pre_projector_terminal"],
        z_goal=eval_features["pre_projector_goal"],
        encode_metadata={**eval_metadata, "sub_experiment": "S2", "variant": "pre_projector"},
        common=common,
    )
    s2_post = result_block_for_embeddings(
        z_terminal=eval_features["post_projector_terminal"],
        z_goal=eval_features["post_projector_goal"],
        encode_metadata={**eval_metadata, "sub_experiment": "S2", "variant": "post_projector"},
        common=common,
    )

    print("Encoding Cube C6 S3 train-mode post-projector features...")
    train_features, train_metadata = encode_projector_variants(
        args=args,
        seed=0,
        row_pair_ids=common["pair_ids"],
        prepared=prepared,
        mode="train",
    )
    s3_eval = {
        "encode_metadata": {**s2_post["encode_metadata"], "sub_experiment": "S3", "variant": "eval_mode", "reused_from": "S2.post_projector"},
        "validation": s2_post["validation"],
        "metrics": s2_post["metrics"],
    }
    s3_train = result_block_for_embeddings(
        z_terminal=train_features["post_projector_terminal"],
        z_goal=train_features["post_projector_goal"],
        encode_metadata={**train_metadata, "sub_experiment": "S3", "variant": "train_mode"},
        common=common,
    )
    summary_rows = [
        summary_row(control="Cube_C6_S2", config="pre_projector_seed0_eval", n_seeds=1, metrics=s2_pre["metrics"]),
        summary_row(control="Cube_C6_S2", config="post_projector_seed0_eval", n_seeds=1, metrics=s2_post["metrics"]),
        summary_row(control="Cube_C6_S3", config="eval_mode_seed0", n_seeds=1, metrics=s3_eval["metrics"]),
        summary_row(control="Cube_C6_S3", config="train_mode_seed0", n_seeds=1, metrics=s3_train["metrics"]),
    ]
    return {
        "metadata": {
            "format": "cube_c6_audit_s2_s3_results",
            "created_at": iso_now(),
            "sub_experiments": ["S2", "S3"],
            "latent_artifact": str(args.latent_artifact),
            "config_path": str(args.config_path),
            "checkpoint_weights_loaded": False,
            "seed": 0,
            "n_records": int(len(common["pair_ids"])),
            "latent_dim": LATENT_DIM,
            "target_metric": "C_real_state",
            "pixel_preparation": prepared.metadata,
            "s2_pre_projector_definition": "ViT CLS token before projector: model.encoder(...).last_hidden_state[:, 0]",
            "s2_post_projector_definition": "Projector output matching JEPA.encode: model.projector(cls)",
            "s3_train_mode_note": "model.train() is set before torch.inference_mode(); inference_mode disables gradients only.",
        },
        "S2": {"pre_projector": s2_pre, "post_projector": s2_post},
        "S3": {"eval_mode": s3_eval, "train_mode": s3_train},
        "summary_table": summary_rows,
    }


def raw_pixel_costs(cache: PixelCache, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    l2 = np.empty((int(cache.terminal_pixels.shape[0]),), dtype=np.float64)
    mean_abs = np.empty_like(l2)
    for start in range(0, len(l2), int(batch_size)):
        end = min(start + int(batch_size), len(l2))
        terminal = cache.terminal_pixels[start:end].astype(np.float32)
        goal = cache.goal_pixels[cache.goal_index_for_row[start:end]].astype(np.float32)
        diff = terminal - goal
        l2[start:end] = np.sum(diff * diff, axis=(1, 2, 3), dtype=np.float64)
        mean_abs[start:end] = np.mean(np.abs(diff), axis=(1, 2, 3), dtype=np.float64)
    return l2, mean_abs


def run_s4_s5(args: argparse.Namespace, cache: PixelCache, common: dict[str, Any]) -> dict[str, Any]:
    print("Computing Cube C6 S4 raw-pixel cost signals...")
    raw_l2, mean_rgb = raw_pixel_costs(cache, batch_size=int(args.arch_batch_size))
    s4 = {
        "raw_pixel_l2": result_block_for_costs(
            name="raw_pixel_l2",
            costs=raw_l2,
            common=common,
            cost_metadata={"cost": "sum((terminal_rgb - goal_rgb)^2)", "input_pixel_dtype": "uint8"},
        ),
        "mean_rgb_diff": result_block_for_costs(
            name="mean_rgb_diff",
            costs=mean_rgb,
            common=common,
            cost_metadata={"cost": "mean(abs(terminal_rgb - goal_rgb))", "input_pixel_dtype": "uint8"},
        ),
    }

    print("Computing Cube C6 S5 simulator-state cube-position feature...")
    goal_pos_rows = cache.goal_cube_pos[cache.goal_index_for_row]
    object_cost = np.linalg.norm(cache.terminal_cube_pos - goal_pos_rows, axis=1).astype(np.float64)
    diffs = np.abs(object_cost - common["labels"])
    s5_validation = {
        **finite_stats(object_cost),
        "matches_C_real_state": {
            "max_abs_diff": clean_float(float(diffs.max())),
            "mean_abs_diff": clean_float(float(diffs.mean())),
            "passed": bool(float(diffs.max()) <= float(args.replay_atol)),
            "atol": float(args.replay_atol),
        },
    }
    s5 = {
        "cube_position_distance": result_block_for_costs(
            name="cube_position_distance",
            costs=object_cost,
            common=common,
            cost_metadata={
                "cost": "euclidean_distance_between_replayed_terminal_cube_pos_and_goal_cube_pos",
                "uses_simulator_state": True,
            },
            validation=s5_validation,
        )
    }
    summary_rows = [
        summary_row(control="Cube_C6_S4", config="raw_pixel_l2", n_seeds=1, metrics=s4["raw_pixel_l2"]["metrics"]),
        summary_row(control="Cube_C6_S4", config="mean_rgb_diff", n_seeds=1, metrics=s4["mean_rgb_diff"]["metrics"]),
        summary_row(control="Cube_C6_S5", config="cube_position_distance", n_seeds=1, metrics=s5["cube_position_distance"]["metrics"]),
    ]
    return {
        "metadata": {
            "format": "cube_c6_audit_s4_s5_results",
            "created_at": iso_now(),
            "sub_experiments": ["S4", "S5"],
            "latent_artifact": str(args.latent_artifact),
            "pixel_artifact": str(args.pixel_artifact),
            "n_records": int(len(common["pair_ids"])),
            "target_metric": "C_real_state",
            "pixel_cache": cache.metadata,
            "uses_neural_network": False,
        },
        "S4": s4,
        "S5": s5,
        "summary_table": summary_rows,
    }


def build_arch_model(*, architecture: str, seed: int, device: str) -> torch.nn.Module:
    torch.manual_seed(int(seed))
    if architecture == "small_cnn":
        return SmallRandomCNN(output_dim=LATENT_DIM).to(device)
    if architecture == "resnet18":
        from torchvision.models import resnet18  # noqa: PLC0415

        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, LATENT_DIM)
        return model.to(device)
    raise ValueError(f"Unknown architecture: {architecture}")


def preprocess_arch_images(images: np.ndarray, *, architecture: str, device: str) -> torch.Tensor:
    if images.ndim != 4 or int(images.shape[-1]) != 3:
        raise ValueError(f"Expected raw HWC RGB images shaped [N,H,W,3], got {tuple(images.shape)}")
    tensor = torch.as_tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if tuple(tensor.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    if architecture == "resnet18":
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    elif architecture != "small_cnn":
        raise ValueError(f"Unknown architecture: {architecture}")
    return tensor.to(device=device, dtype=torch.float32)


@torch.inference_mode()
def encode_arch_images(
    *,
    model: torch.nn.Module,
    images: np.ndarray,
    architecture: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, int(images.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(images.shape[0]))
        batch = preprocess_arch_images(images[start:end], architecture=architecture, device=device)
        emb = model(batch)
        if emb.ndim != 2 or int(emb.shape[1]) != LATENT_DIM:
            raise ValueError(f"{architecture} embeddings must be [B,{LATENT_DIM}], got {tuple(emb.shape)}")
        chunks.append(emb.detach().cpu().to(dtype=torch.float32))
    return torch.cat(chunks, dim=0)


def run_architecture(
    *,
    args: argparse.Namespace,
    cache: PixelCache,
    common: dict[str, Any],
    architecture: str,
    seed: int,
) -> dict[str, Any]:
    print(f"Encoding Cube C6 S6 {architecture} eval_mode...")
    started = time.time()
    model = build_arch_model(architecture=architecture, seed=seed, device=args.device)
    model.eval()
    z_terminal = encode_arch_images(
        model=model,
        images=cache.terminal_pixels,
        architecture=architecture,
        device=args.device,
        batch_size=int(args.arch_batch_size),
    )
    z_goal_unique = encode_arch_images(
        model=model,
        images=cache.goal_pixels,
        architecture=architecture,
        device=args.device,
        batch_size=int(args.arch_batch_size),
    )
    z_goal = z_goal_unique[torch.as_tensor(cache.goal_index_for_row, dtype=torch.long)]
    metadata = {
        "seed": int(seed),
        "architecture": architecture,
        "mode": "eval_mode",
        "model_training_flag": bool(model.training),
        "checkpoint_weights_loaded": False,
        "pretrained_weights_loaded": False,
        "device": args.device,
        "pixel_source": "cube_c6_pixel_cache",
        "input_pixel_format": "uint8 HWC RGB",
        "normalization": "[0,1]" if architecture == "small_cnn" else "[0,1] plus ImageNet mean/std",
        "resize": f"bilinear_to_{IMAGE_SIZE}x{IMAGE_SIZE}_if_needed",
        "batch_size": int(args.arch_batch_size),
        "terminal_batches_encoded": int((int(cache.terminal_pixels.shape[0]) + int(args.arch_batch_size) - 1) // int(args.arch_batch_size)),
        "goal_unique_batches_encoded": int((int(cache.goal_pixels.shape[0]) + int(args.arch_batch_size) - 1) // int(args.arch_batch_size)),
        "wallclock_seconds": clean_float(time.time() - started),
    }
    del model
    empty_accelerator_cache()
    return result_block_for_embeddings(z_terminal=z_terminal, z_goal=z_goal, encode_metadata=metadata, common=common)


def run_s6(args: argparse.Namespace, cache: PixelCache, common: dict[str, Any]) -> dict[str, Any]:
    small_cnn = run_architecture(args=args, cache=cache, common=common, architecture="small_cnn", seed=0)
    resnet18 = run_architecture(args=args, cache=cache, common=common, architecture="resnet18", seed=0)
    summary_rows = [
        summary_row(control="Cube_C6_S6", config="small_cnn_seed0_eval", n_seeds=1, metrics=small_cnn["metrics"]),
        summary_row(control="Cube_C6_S6", config="resnet18_seed0_eval", n_seeds=1, metrics=resnet18["metrics"]),
    ]
    return {
        "metadata": {
            "format": "cube_c6_audit_s6_random_arch_results",
            "created_at": iso_now(),
            "sub_experiment": "S6",
            "latent_artifact": str(args.latent_artifact),
            "pixel_artifact": str(args.pixel_artifact),
            "seed": 0,
            "device": args.device,
            "batch_size": int(args.arch_batch_size),
            "n_records": int(len(common["pair_ids"])),
            "latent_dim": LATENT_DIM,
            "target_metric": "C_real_state",
            "preprocessing": {
                "input_pixel_format": "uint8 HWC RGB",
                "target_image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "small_cnn_normalization": "[0,1]",
                "resnet18_normalization": {"scale": "[0,1]", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            },
            "architectures": {
                "small_cnn": "Conv-BN-ReLU blocks 3->32->64->128->192, adaptive average pooling, Linear(192,192)",
                "resnet18": "torchvision.models.resnet18(weights=None) with fc replaced by Linear(in_features,192)",
            },
        },
        "small_cnn": {"eval_mode": small_cnn},
        "resnet18": {"eval_mode": resnet18},
        "summary_table": summary_rows,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n")
    print(f"Saved: {path}")


def metric_from_summary(summary_rows: list[dict[str, Any]], config: str) -> float | None:
    for row in summary_rows:
        if row.get("config") == config:
            value = row.get("global_spearman_mean")
            return None if value is None else float(value)
    return None


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_comparison_rows(cube_outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    push_root = PROJECT_ROOT / "results" / "phase2" / "stage1" / "c6_audit"
    push_s1 = load_json_if_exists(push_root / "s1_random_init_10seed.json")
    push_s2_s3 = load_json_if_exists(push_root / "s2_s3_results.json")
    push_s4_s5 = load_json_if_exists(push_root / "s4_s5_results.json")
    push_s6 = load_json_if_exists(push_root / "s6_random_arch_results.json")
    rows = []

    def add(sub_exp: str, metric: str, push_data: dict[str, Any] | None, cube_data: dict[str, Any], config: str, cube_config: str | None = None) -> None:
        rows.append(
            {
                "sub_experiment": sub_exp,
                "metric": metric,
                "pusht_spearman": metric_from_summary(push_data.get("summary_table", []) if push_data else [], config),
                "cube_spearman": metric_from_summary(cube_data.get("summary_table", []), cube_config or config),
            }
        )

    add("S1", "random_init_10seed", push_s1, cube_outputs["s1"], "random_init_10seed")
    add("S2", "pre_projector_seed0_eval", push_s2_s3, cube_outputs["s2_s3"], "pre_projector_seed0_eval")
    add("S2", "post_projector_seed0_eval", push_s2_s3, cube_outputs["s2_s3"], "post_projector_seed0_eval")
    add("S3", "eval_mode_seed0", push_s2_s3, cube_outputs["s2_s3"], "eval_mode_seed0")
    add("S3", "train_mode_seed0", push_s2_s3, cube_outputs["s2_s3"], "train_mode_seed0")
    add("S4", "raw_pixel_l2", push_s4_s5, cube_outputs["s4_s5"], "raw_pixel_l2")
    add("S4", "mean_rgb_diff", push_s4_s5, cube_outputs["s4_s5"], "mean_rgb_diff")
    add("S5", "object_feature", push_s4_s5, cube_outputs["s4_s5"], "block_center_distance", "cube_position_distance")
    add("S6", "small_cnn_seed0_eval", push_s6, cube_outputs["s6"], "small_cnn_seed0_eval")
    add("S6", "resnet18_seed0_eval", push_s6, cube_outputs["s6"], "resnet18_seed0_eval")
    return rows


def critical_verdict(cube_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    random_vit = metric_from_summary(cube_outputs["s2_s3"]["summary_table"], "eval_mode_seed0")
    raw_pixel = metric_from_summary(cube_outputs["s4_s5"]["summary_table"], "raw_pixel_l2")
    mean_rgb = metric_from_summary(cube_outputs["s4_s5"]["summary_table"], "mean_rgb_diff")
    verdict = {
        "cube_random_init_vit_eval_spearman": random_vit,
        "cube_raw_pixel_l2_spearman": raw_pixel,
        "cube_mean_rgb_diff_spearman": mean_rgb,
        "raw_pixels_positive": bool(raw_pixel is not None and raw_pixel > 0),
        "random_init_vit_negative": bool(random_vit is not None and random_vit < 0),
    }
    verdict["signal_inversion_generalizes"] = bool(verdict["raw_pixels_positive"] and verdict["random_init_vit_negative"])
    return verdict


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.config_path = args.config_path.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.pixel_artifact = (args.pixel_artifact or (args.output_dir / "cube_c6_pixels.pt")).expanduser().resolve()
    if args.action_artifact is not None:
        args.action_artifact = args.action_artifact.expanduser().resolve()
    args.device = resolve_device(str(args.device))

    print("== Cube C6 audit setup ==")
    print(f"latent_artifact: {args.latent_artifact}")
    print(f"pairs_path: {args.pairs_path}")
    print(f"config_path: {args.config_path}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"pixel_artifact: {args.pixel_artifact}")
    print(f"device: {args.device}")
    print(f"seeds: {[int(seed) for seed in args.seeds]}")

    pairs_data, requested_pairs = load_pairs(args.pairs_path, max_pairs=args.max_pairs, pair_ids=args.pair_ids)
    requested_pair_ids = [int(pair["pair_id"]) for pair in requested_pairs]
    latent_full = load_cube_latent_artifact(args.latent_artifact)
    latent_artifact = subset_latent_artifact(latent_full, requested_pair_ids)
    common = build_common(latent_artifact)
    if int(len(common["pair_ids"])) == 0:
        raise RuntimeError("No latent rows selected for Cube C6 audit")
    print(f"selected_pairs: {len(requested_pairs)} selected_records: {len(common['pair_ids'])}")

    cache = replay_or_load_pixels(args, latent_artifact, requested_pairs)
    prepared = prepare_vit_pixels(args=args, cache=cache)

    s1 = run_s1(args, latent_artifact, common, prepared)
    write_json(args.output_dir / "s1_random_init_10seed.json", s1)
    print_metric_table(s1["summary_table"], title="Cube C6 Audit S1 random-init LeWM summary")

    s2_s3 = run_s2_s3(args, common, prepared)
    write_json(args.output_dir / "s2_s3_results.json", s2_s3)
    print_metric_table(s2_s3["summary_table"], title="Cube C6 Audit S2/S3 summary")

    s4_s5 = run_s4_s5(args, cache, common)
    write_json(args.output_dir / "s4_s5_results.json", s4_s5)
    print_metric_table(s4_s5["summary_table"], title="Cube C6 Audit S4/S5 summary")

    s6 = run_s6(args, cache, common)
    write_json(args.output_dir / "s6_random_arch_results.json", s6)
    print_metric_table(s6["summary_table"], title="Cube C6 Audit S6 random-architecture summary")

    cube_outputs = {"s1": s1, "s2_s3": s2_s3, "s4_s5": s4_s5, "s6": s6}
    comparison_rows = build_comparison_rows(cube_outputs)
    verdict = critical_verdict(cube_outputs)
    summary = {
        "metadata": {
            "format": "cube_c6_audit_summary",
            "created_at": iso_now(),
            "latent_artifact": str(args.latent_artifact),
            "pairs_path": str(args.pairs_path),
            "output_dir": str(args.output_dir),
            "pixel_artifact": str(args.pixel_artifact),
            "n_pairs": int(len(requested_pairs)),
            "n_records": int(len(common["pair_ids"])),
            "device": args.device,
            "replay_validation": cache.validation,
        },
        "comparison_rows": comparison_rows,
        "critical_verdict": verdict,
        "summary_tables": {
            "S1": s1["summary_table"],
            "S2_S3": s2_s3["summary_table"],
            "S4_S5": s4_s5["summary_table"],
            "S6": s6["summary_table"],
        },
    }
    write_json(args.output_dir / "cube_c6_audit_summary.json", summary)
    print_comparison_table(comparison_rows)
    print("Cube C6 critical verdict:")
    print(f"  raw_pixel_l2_positive: {verdict['raw_pixels_positive']} ({fmt(verdict['cube_raw_pixel_l2_spearman'])})")
    print(f"  random_init_vit_eval_negative: {verdict['random_init_vit_negative']} ({fmt(verdict['cube_random_init_vit_eval_spearman'])})")
    print(f"  signal_inversion_generalizes: {verdict['signal_inversion_generalizes']}")
    print(f"\nSaved result directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
