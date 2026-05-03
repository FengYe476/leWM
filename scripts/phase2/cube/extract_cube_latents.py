#!/usr/bin/env python3
"""Replay Cube stratified pairs and save endpoint LeWM latents.

This is the Cube analogue of ``scripts/phase2/extract_latents.py`` for PushT.
It creates a Stage-1-ready artifact with the same core schema as
``track_a_latents.pt``, while using Cube-specific reset, action dimensions, and
success/cost definitions.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_cube_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATASET_NAME,
    get_dataset,
)
from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    build_policy,
    build_processors,
    resolve_device,
)


DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_pairs.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_latents.pt"

ACTION_SOURCE_ORDER = ("data", "smooth_random", "CEM_early", "CEM_late")
ACTION_BLOCK = 5
IMG_SIZE = 224
LATENT_DIM = 192
RANDOM_WAYPOINTS = 5
NUM_SAMPLES = 300
CEM_EARLY_ITERS = 3
CEM_LATE_ITERS = 30
TOPK = 30
VAR_SCALE = 1.0
CUBE_SUCCESS_THRESHOLD_M = 0.04


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


def parse_action_counts(raw: str) -> dict[str, int]:
    chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if len(chunks) != len(ACTION_SOURCE_ORDER):
        raise argparse.ArgumentTypeError(
            "--action-counts must contain four comma-separated integers"
        )
    values = [int(chunk) for chunk in chunks]
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("--action-counts values must be nonnegative")
    return dict(zip(ACTION_SOURCE_ORDER, values, strict=True))


DEFAULT_ACTION_COUNTS = parse_action_counts("20,20,20,20")


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_pairs(
    path: Path,
    *,
    max_pairs: int | None,
    pair_ids: list[int] | None,
) -> tuple[dict, list[dict]]:
    data = load_json(path)
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
            "Cube pair file offset mismatch. "
            f"Expected offset={offset}; examples={mismatches[:5]}"
        )


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
                value.shape[0],
                num_samples,
                *value.shape[1:],
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None, ...], num_samples, axis=1)
        else:
            expanded[key] = value
    return expanded


def infer_raw_action_dim(dataset) -> int:
    action_col = dataset.get_col_data("action")
    if action_col.ndim != 2:
        raise ValueError(f"Expected dataset action column to be 2D, got {action_col.shape}")
    return int(action_col.shape[1])


def prepare_pair_info(
    policy,
    initial_pixels: np.ndarray,
    goal_pixels: np.ndarray,
    *,
    raw_action_dim: int,
) -> dict:
    raw_info = {
        "pixels": initial_pixels[None, None, ...],
        "goal": goal_pixels[None, None, ...],
        # JEPA.get_cost expects an action key while encoding the goal; rollout
        # replaces it with blocked action candidates before dynamics prediction.
        "action": np.zeros((1, 1, raw_action_dim), dtype=np.float32),
    }
    return policy._prepare_info(raw_info)


def raw_to_blocked_normalized(
    raw_actions: np.ndarray,
    *,
    action_processor,
    action_block: int,
    raw_action_dim: int,
) -> np.ndarray:
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != raw_action_dim:
        raise ValueError(
            f"Expected raw actions with shape (T, {raw_action_dim}), got {raw_actions.shape}"
        )
    if raw_actions.shape[0] % action_block != 0:
        raise ValueError(
            f"Raw action length {raw_actions.shape[0]} is not divisible by {action_block}"
        )

    normalized = action_processor.transform(raw_actions).astype(np.float32)
    return normalized.reshape(
        raw_actions.shape[0] // action_block,
        action_block * raw_action_dim,
    )


def blocked_normalized_to_raw(
    blocked_actions: np.ndarray,
    *,
    action_processor,
    action_block: int,
    raw_action_dim: int,
) -> np.ndarray:
    blocked_actions = np.asarray(blocked_actions, dtype=np.float32)
    blocked_dim = action_block * raw_action_dim
    if blocked_actions.ndim != 2 or blocked_actions.shape[1] != blocked_dim:
        raise ValueError(
            f"Expected blocked actions with shape (H, {blocked_dim}), "
            f"got {blocked_actions.shape}"
        )

    normalized = blocked_actions.reshape(
        blocked_actions.shape[0] * action_block,
        raw_action_dim,
    )
    return action_processor.inverse_transform(normalized).astype(np.float32)


def sample_data_action_sequences(
    dataset,
    valid_indices: np.ndarray,
    *,
    count: int,
    raw_steps: int,
    action_processor,
    action_block: int,
    raw_action_dim: int,
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
                "source": "data",
                "source_index": source_index,
                "dataset_row": row,
                "blocked_normalized": raw_to_blocked_normalized(
                    raw_actions,
                    action_processor=action_processor,
                    action_block=action_block,
                    raw_action_dim=raw_action_dim,
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
    raw_action_dim: int,
    rng: np.random.Generator,
) -> list[dict]:
    sequences = []
    for source_index in range(count):
        points = rng.uniform(-1.0, 1.0, size=(waypoints, raw_action_dim)).astype(np.float32)
        raw_actions = catmull_rom_interpolate(points, raw_steps)
        sequences.append(
            {
                "source": "smooth_random",
                "source_index": source_index,
                "blocked_normalized": raw_to_blocked_normalized(
                    raw_actions,
                    action_processor=action_processor,
                    action_block=action_block,
                    raw_action_dim=raw_action_dim,
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
    pair_id: int,
    raw_action_dim: int,
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
        seed=args.seed + pair_id * 1009,
    )

    sequences = []
    for source, iter_count in (
        ("CEM_early", args.cem_early_iters),
        ("CEM_late", args.cem_late_iters),
    ):
        capture = captures[iter_count]
        for source_index, blocked_normalized in enumerate(capture["actions"]):
            raw_actions = blocked_normalized_to_raw(
                blocked_normalized,
                action_processor=action_processor,
                action_block=args.action_block,
                raw_action_dim=raw_action_dim,
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
    raw_action_dim: int,
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
                raw_action_dim=raw_action_dim,
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
                raw_action_dim=raw_action_dim,
                rng=rng,
            )
        )

    max_cem_count = max(action_counts["CEM_early"], action_counts["CEM_late"])
    if max_cem_count:
        cem_args = argparse.Namespace(**vars(args))
        cem_args.num_per_source = max_cem_count
        cem_sequences = generate_cem_action_sequences(
            model=model,
            prepared_info=prepared_info,
            args=cem_args,
            horizon_blocks=raw_steps // args.action_block,
            action_dim=args.action_block * raw_action_dim,
            action_processor=action_processor,
            pair_id=pair_id,
            raw_action_dim=raw_action_dim,
        )
        by_source = {"CEM_early": [], "CEM_late": []}
        for sequence in cem_sequences:
            by_source[sequence["source"]].append(sequence)
        sequences.extend(by_source["CEM_early"][: action_counts["CEM_early"]])
        sequences.extend(by_source["CEM_late"][: action_counts["CEM_late"]])
    return sequences


def compute_model_costs(model, prepared_info: dict, blocked_actions: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    candidates = torch.as_tensor(
        blocked_actions[None, ...],
        dtype=torch.float32,
        device=device,
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


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Encountered zero-norm Cube quaternion")
    return quat / norm


def quaternion_geodesic_angle(q_a: np.ndarray, q_b: np.ndarray) -> float:
    q_a = normalize_quaternion(q_a)
    q_b = normalize_quaternion(q_b)
    dot = float(abs(np.dot(q_a, q_b)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * math.acos(dot))


def cube_metrics(
    *,
    terminal_pos: np.ndarray,
    terminal_quat: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
) -> dict[str, float | bool]:
    cube_pos_dist = float(np.linalg.norm(np.asarray(terminal_pos) - np.asarray(goal_pos)))
    quat_angle_dist = quaternion_geodesic_angle(terminal_quat, goal_quat)
    return {
        "cube_pos_dist": cube_pos_dist,
        "quat_angle_dist": quat_angle_dist,
        "C_real_state": cube_pos_dist,
        "success": bool(cube_pos_dist <= CUBE_SUCCESS_THRESHOLD_M),
    }


def set_cube_start_and_goal(env, *, initial: dict, goal: dict, seed: int) -> None:
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    env_unwrapped.set_state(
        np.asarray(initial["qpos"], dtype=np.float64),
        np.asarray(initial["qvel"], dtype=np.float64),
    )
    env_unwrapped.set_target_pos(
        0,
        np.asarray(goal["privileged_block_0_pos"], dtype=np.float64),
        np.asarray(goal["privileged_block_0_quat"], dtype=np.float64),
    )


def terminal_cube_pose(env) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(env.unwrapped._data.joint("object_joint_0").qpos, dtype=np.float64)
    return qpos[:3].copy(), qpos[3:7].copy()


def execute_raw_actions(
    env,
    *,
    initial: dict,
    goal: dict,
    raw_actions: np.ndarray,
    seed: int,
) -> dict:
    set_cube_start_and_goal(env, initial=initial, goal=goal, seed=seed)

    env_success = False
    for action in raw_actions:
        _, _, terminated, _, info = env.step(np.asarray(action, dtype=np.float32))
        env_success = env_success or bool(terminated) or bool(info.get("success", False))

    terminal_pos, terminal_quat = terminal_cube_pose(env)
    terminal_pixels = env.unwrapped.render(camera="front_pixels")
    return {
        "terminal_cube_pos": terminal_pos.astype(np.float64),
        "terminal_cube_quat": terminal_quat.astype(np.float64),
        "terminal_pixels": np.asarray(terminal_pixels, dtype=np.uint8),
        "env_success": bool(env_success),
    }


def load_pair_rows(dataset, start_row: int, goal_row: int) -> dict[str, dict]:
    rows = dataset.get_row_data([start_row, goal_row])
    return {
        "initial": {key: value[0] for key, value in rows.items()},
        "goal": {key: value[1] for key, value in rows.items()},
    }


def cube_valid_action_indices(dataset, *, offset: int) -> np.ndarray:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = np.asarray(dataset.get_col_data(col_name), dtype=np.int64)
    step_idx = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    n_rows = int(len(step_idx))
    if offset <= 0 or offset >= n_rows:
        raise ValueError(f"Invalid offset={offset} for dataset rows={n_rows}")

    if "ep_len" in dataset.column_names:
        ep_len = np.asarray(dataset.get_col_data("ep_len"), dtype=np.int64)
        if episode_ids.min() >= 0 and episode_ids.max() < len(ep_len):
            row_episode_lengths = ep_len[episode_ids]
        else:
            unique_episodes = np.unique(episode_ids)
            if len(unique_episodes) != len(ep_len):
                raise ValueError("Cannot map ep_len rows to episode ids")
            length_by_episode = dict(zip(unique_episodes.tolist(), ep_len.tolist(), strict=True))
            row_episode_lengths = np.asarray(
                [length_by_episode[int(ep)] for ep in episode_ids],
                dtype=np.int64,
            )
    else:
        unique_episodes, inverse = np.unique(episode_ids, return_inverse=True)
        episode_lengths = np.zeros(len(unique_episodes), dtype=np.int64)
        np.maximum.at(episode_lengths, inverse, step_idx + 1)
        row_episode_lengths = episode_lengths[inverse]

    candidate_rows = np.arange(0, n_rows - offset, dtype=np.int64)
    step_valid = step_idx[:-offset] <= (row_episode_lengths[:-offset] - offset - 1)
    same_episode = episode_ids[:-offset] == episode_ids[offset:]
    return candidate_rows[step_valid & same_episode]


def make_policy_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        cem_iters=args.cem_late_iters,
        topk=args.topk,
        seed=args.seed,
        horizon=5,
        receding_horizon=5,
        action_block=args.action_block,
        img_size=args.img_size,
    )


def artifact_metadata(
    args: argparse.Namespace,
    *,
    pairs_metadata: dict,
    offset: int,
    n_pairs: int,
    raw_action_dim: int,
) -> dict:
    return {
        "format": "phase2_cube_latents",
        "created_at": iso_now(),
        "git_commit": get_git_commit(),
        "seed": int(args.seed),
        "device": args.device,
        "pairs_path": str(args.pairs_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_cache_dir": str(args.cache_dir),
        "dataset_name": args.dataset_name,
        "pairs_metadata": pairs_metadata,
        "offset": int(offset),
        "n_pairs_requested": int(n_pairs),
        "action_counts": dict(args.action_counts),
        "action_source_order": ACTION_SOURCE_ORDER,
        "latent_dim": LATENT_DIM,
        "image_size": int(args.img_size),
        "action_block": int(args.action_block),
        "raw_action_dim": int(raw_action_dim),
        "blocked_action_dim": int(raw_action_dim * args.action_block),
        "random_waypoints": int(args.random_waypoints),
        "cem_config": {
            "samples_per_iter": int(args.num_samples),
            "cem_early_iteration": int(args.cem_early_iters),
            "cem_late_iteration": int(args.cem_late_iters),
            "topk": int(args.topk),
            "var_scale": float(args.var_scale),
            "horizon_blocks": int(offset // args.action_block),
        },
        "success_definition": "cube_pos_dist <= 0.04m",
        "C_real_state_definition": (
            "Euclidean distance in meters between terminal object_joint_0 position "
            "and the goal row privileged_block_0_pos."
        ),
        "v1_cost_definition": (
            "Cube v1_cost aliases C_real_state because no PushT V1 hinge oracle exists."
        ),
        "diagnostic_fields": {
            "cube_pos_dist": "Same value as C_real_state, in meters.",
            "quat_angle_dist": (
                "Quaternion geodesic angle between terminal object_joint_0 quaternion "
                "and goal row privileged_block_0_quat; not used for success."
            ),
        },
        "goal_embedding_source": "goal row pixels from cube_pairs.json goal_row",
    }


def source_index_for_action(source_counts: dict[str, int], source: str) -> int:
    source_index = source_counts.get(source, 0)
    source_counts[source] = source_index + 1
    return source_index


def save_artifact(path: Path, metadata: dict, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda item: (item["pair_id"], item["action_id"]))
    if records:
        z_terminal = torch.stack([torch.as_tensor(item["z_terminal"]) for item in records])
        z_goal = torch.stack([torch.as_tensor(item["z_goal"]) for item in records])
        pair_id = torch.as_tensor([item["pair_id"] for item in records], dtype=torch.long)
        action_id = torch.as_tensor([item["action_id"] for item in records], dtype=torch.long)
        source_index = torch.as_tensor(
            [item["source_index"] for item in records],
            dtype=torch.long,
        )
        c_real_z = torch.as_tensor([item["C_real_z"] for item in records], dtype=torch.float32)
        c_model = torch.as_tensor([item["C_model"] for item in records], dtype=torch.float32)
        c_real_state = torch.as_tensor(
            [item["C_real_state"] for item in records],
            dtype=torch.float32,
        )
        cube_pos_dist = torch.as_tensor(
            [item["cube_pos_dist"] for item in records],
            dtype=torch.float32,
        )
        quat_angle_dist = torch.as_tensor(
            [item["quat_angle_dist"] for item in records],
            dtype=torch.float32,
        )
        v1_cost = torch.as_tensor([item["v1_cost"] for item in records], dtype=torch.float32)
        success = torch.as_tensor([item["success"] for item in records], dtype=torch.bool)
    else:
        z_terminal = torch.empty((0, LATENT_DIM), dtype=torch.float32)
        z_goal = torch.empty((0, LATENT_DIM), dtype=torch.float32)
        pair_id = torch.empty((0,), dtype=torch.long)
        action_id = torch.empty((0,), dtype=torch.long)
        source_index = torch.empty((0,), dtype=torch.long)
        c_real_z = torch.empty((0,), dtype=torch.float32)
        c_model = torch.empty((0,), dtype=torch.float32)
        c_real_state = torch.empty((0,), dtype=torch.float32)
        cube_pos_dist = torch.empty((0,), dtype=torch.float32)
        quat_angle_dist = torch.empty((0,), dtype=torch.float32)
        v1_cost = torch.empty((0,), dtype=torch.float32)
        success = torch.empty((0,), dtype=torch.bool)

    artifact = {
        "metadata": {
            **metadata,
            "n_records": int(len(records)),
            "n_pairs_completed": int(len({item["pair_id"] for item in records})),
            "updated_at": iso_now(),
        },
        "pair_id": pair_id,
        "action_id": action_id,
        "source": [item["source"] for item in records],
        "source_index": source_index,
        "action_key": [item["action_key"] for item in records],
        "cell": [item["cell"] for item in records],
        "z_terminal": z_terminal.to(dtype=torch.float32),
        "z_goal": z_goal.to(dtype=torch.float32),
        "C_real_z": c_real_z,
        "C_model": c_model,
        "C_real_state": c_real_state,
        "cube_pos_dist": cube_pos_dist,
        "quat_angle_dist": quat_angle_dist,
        "v1_cost": v1_cost,
        "success": success,
    }
    torch.save(artifact, path)


def load_existing_artifact(path: Path) -> tuple[dict | None, list[dict]]:
    if not path.exists():
        return None, []
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    records = []
    n_records = int(artifact["pair_id"].numel())
    cube_pos = artifact.get("cube_pos_dist", artifact["C_real_state"])
    quat_angle = artifact.get(
        "quat_angle_dist",
        torch.full((n_records,), float("nan"), dtype=torch.float32),
    )
    for idx in range(n_records):
        records.append(
            {
                "pair_id": int(artifact["pair_id"][idx]),
                "action_id": int(artifact["action_id"][idx]),
                "source": artifact["source"][idx],
                "source_index": int(artifact["source_index"][idx]),
                "action_key": artifact["action_key"][idx],
                "cell": artifact["cell"][idx],
                "z_terminal": artifact["z_terminal"][idx].cpu(),
                "z_goal": artifact["z_goal"][idx].cpu(),
                "C_real_z": float(artifact["C_real_z"][idx]),
                "C_model": float(artifact["C_model"][idx]),
                "C_real_state": float(artifact["C_real_state"][idx]),
                "cube_pos_dist": float(cube_pos[idx]),
                "quat_angle_dist": float(quat_angle[idx]),
                "v1_cost": float(artifact["v1_cost"][idx]),
                "success": bool(artifact["success"][idx]),
            }
        )
    return artifact.get("metadata", {}), records


def extract_pair_records(
    *,
    pair_spec: dict,
    dataset,
    valid_action_indices: np.ndarray,
    policy,
    model,
    env,
    args: argparse.Namespace,
    raw_action_dim: int,
) -> list[dict]:
    pair_id = int(pair_spec["pair_id"])
    start_row = int(pair_spec["start_row"])
    goal_row = int(pair_spec["goal_row"])
    raw_steps = goal_row - start_row
    if raw_steps % args.action_block != 0:
        raise ValueError(f"Pair {pair_id} raw_steps={raw_steps} not divisible by action_block")

    pair_rows = load_pair_rows(dataset, start_row, goal_row)
    initial = pair_rows["initial"]
    goal = pair_rows["goal"]
    goal_pos = np.asarray(goal["privileged_block_0_pos"], dtype=np.float64)
    goal_quat = np.asarray(goal["privileged_block_0_quat"], dtype=np.float64)

    prepared_info = prepare_pair_info(
        policy,
        initial["pixels"],
        goal["pixels"],
        raw_action_dim=raw_action_dim,
    )
    goal_emb_t = encode_pixels(policy, model, goal["pixels"])
    goal_emb = goal_emb_t.detach().cpu()[0].to(dtype=torch.float32)

    sequences = select_action_sequences(
        dataset=dataset,
        valid_action_indices=valid_action_indices,
        policy=policy,
        model=model,
        prepared_info=prepared_info,
        args=args,
        pair_id=pair_id,
        raw_steps=raw_steps,
        action_counts=args.action_counts,
        raw_action_dim=raw_action_dim,
    )
    expected_actions = sum(args.action_counts.values())
    if len(sequences) != expected_actions:
        raise RuntimeError(
            f"Pair {pair_id} produced {len(sequences)} actions, expected {expected_actions}"
        )

    blocked = np.stack([sequence["blocked_normalized"] for sequence in sequences])
    model_costs = compute_model_costs(model, prepared_info, blocked)
    source_counts: dict[str, int] = {}
    pair_records = []

    for action_id, (sequence, model_cost) in enumerate(zip(sequences, model_costs, strict=True)):
        source = sequence["source"]
        source_index = source_index_for_action(source_counts, source)
        rollout = execute_raw_actions(
            env,
            initial=initial,
            goal=goal,
            raw_actions=sequence["raw"],
            seed=args.seed + pair_id * 10_000 + action_id,
        )
        terminal_emb_t = encode_pixels(policy, model, rollout["terminal_pixels"])
        terminal_emb = terminal_emb_t.detach().cpu()[0].to(dtype=torch.float32)
        metrics = cube_metrics(
            terminal_pos=rollout["terminal_cube_pos"],
            terminal_quat=rollout["terminal_cube_quat"],
            goal_pos=goal_pos,
            goal_quat=goal_quat,
        )
        c_real_state = float(metrics["C_real_state"])
        record = {
            "pair_id": pair_id,
            "action_id": int(action_id),
            "source": source,
            "source_index": int(source_index),
            "action_key": f"{pair_id}:{source}:{source_index}",
            "cell": str(pair_spec["cell"]),
            "z_terminal": terminal_emb,
            "z_goal": goal_emb,
            "C_real_z": squared_l2(terminal_emb_t, goal_emb_t),
            "C_model": float(model_cost),
            "C_real_state": c_real_state,
            "cube_pos_dist": float(metrics["cube_pos_dist"]),
            "quat_angle_dist": float(metrics["quat_angle_dist"]),
            "v1_cost": c_real_state,
            "success": bool(metrics["success"]),
        }
        pair_records.append(record)
    return pair_records


def success_summary_by_source(records: list[dict]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for source in ACTION_SOURCE_ORDER:
        source_records = [record for record in records if record["source"] == source]
        total = len(source_records)
        successes = sum(bool(record["success"]) for record in source_records)
        summary[source] = {
            "successes": int(successes),
            "total": int(total),
            "success_rate": float(successes / total) if total else 0.0,
        }
    return summary


def print_success_summary(records: list[dict]) -> None:
    print("Success by source:")
    for source, row in success_summary_by_source(records).items():
        print(
            f"  {source}: {row['successes']}/{row['total']} "
            f"({100.0 * row['success_rate']:.2f}%)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--action-block", type=int, default=ACTION_BLOCK)
    parser.add_argument("--random-waypoints", type=int, default=RANDOM_WAYPOINTS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--cem-early-iters", type=int, default=CEM_EARLY_ITERS)
    parser.add_argument("--cem-late-iters", type=int, default=CEM_LATE_ITERS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--var-scale", type=float, default=VAR_SCALE)
    parser.add_argument(
        "--action-counts",
        type=parse_action_counts,
        default=DEFAULT_ACTION_COUNTS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("--max-pairs must be positive when provided")
    if args.action_block <= 0:
        raise ValueError("--action-block must be positive")
    if args.random_waypoints < 2:
        raise ValueError("--random-waypoints must be at least 2")
    if args.topk > args.num_samples:
        raise ValueError("--topk must be <= --num-samples")
    if args.action_counts["CEM_early"] > args.topk or args.action_counts["CEM_late"] > args.topk:
        raise ValueError("--action-counts CEM values must be <= --topk")

    pairs_data, requested_pairs = load_pairs(
        args.pairs_path,
        max_pairs=args.max_pairs,
        pair_ids=args.pair_ids,
    )
    pairs_metadata = pairs_data.get("metadata", {})
    offset = int(pairs_metadata.get("offset", requested_pairs[0]["goal_row"] - requested_pairs[0]["start_row"]))
    if offset % args.action_block != 0:
        raise ValueError("Cube offset must be divisible by --action-block")
    validate_requested_pair_offsets(requested_pairs, offset=offset)

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    raw_action_dim = infer_raw_action_dim(dataset)
    if raw_action_dim != 5:
        raise ValueError(f"Expected Cube raw action dim 5, got {raw_action_dim}")
    valid_action_indices = cube_valid_action_indices(dataset, offset=offset)
    process = build_processors(dataset, ["action"])
    policy = build_policy(make_policy_namespace(args), process)
    model = policy.solver.model

    existing_metadata, records = load_existing_artifact(args.output) if args.resume else (None, [])
    completed_pair_ids = {int(record["pair_id"]) for record in records}
    metadata = artifact_metadata(
        args,
        pairs_metadata=pairs_metadata,
        offset=offset,
        n_pairs=len(requested_pairs),
        raw_action_dim=raw_action_dim,
    )
    if existing_metadata:
        metadata.update(existing_metadata)
        metadata.update(
            artifact_metadata(
                args,
                pairs_metadata=pairs_metadata,
                offset=offset,
                n_pairs=len(requested_pairs),
                raw_action_dim=raw_action_dim,
            )
        )

    print("== Phase 2 Cube latent extraction ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"offset: {offset}")
    print(f"raw_action_dim: {raw_action_dim}")
    print(f"blocked_action_dim: {raw_action_dim * args.action_block}")
    print(f"action_counts: {args.action_counts}")
    print(f"resume_completed_pairs: {len(completed_pair_ids)}")

    total_started = time.time()
    env = gym.make(
        "swm/OGBCube-v0",
        env_type="single",
        ob_type="states",
        multiview=False,
        width=args.img_size,
        height=args.img_size,
        visualize_info=False,
        terminate_at_goal=True,
    )
    try:
        for pair_spec in requested_pairs:
            pair_id = int(pair_spec["pair_id"])
            if pair_id in completed_pair_ids:
                print(f"Skipping completed pair_id={pair_id}")
                continue
            pair_started = time.time()
            print(
                f"\n== pair_id={pair_id} cell={pair_spec['cell']} "
                f"start_row={pair_spec['start_row']} goal_row={pair_spec['goal_row']} =="
            )
            pair_records = extract_pair_records(
                pair_spec=pair_spec,
                dataset=dataset,
                valid_action_indices=valid_action_indices,
                policy=policy,
                model=model,
                env=env,
                args=args,
                raw_action_dim=raw_action_dim,
            )
            records.extend(pair_records)
            completed_pair_ids.add(pair_id)
            metadata["finished_at"] = None
            metadata["wallclock_seconds"] = time.time() - total_started
            save_artifact(args.output, metadata, records)
            successes = sum(bool(record["success"]) for record in pair_records)
            print(
                f"saved_pair_records={len(pair_records)} successes={successes}; "
                f"elapsed_seconds={time.time() - pair_started:.2f}"
            )
            print_success_summary(pair_records)
    finally:
        env.close()

    metadata["finished_at"] = iso_now()
    metadata["wallclock_seconds"] = time.time() - total_started
    save_artifact(args.output, metadata, records)
    print("\n== Cube latent extraction summary ==")
    print(f"pairs_completed: {len(completed_pair_ids)}")
    print(f"records: {len(records)}")
    print_success_summary(records)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
