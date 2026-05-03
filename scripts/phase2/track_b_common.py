#!/usr/bin/env python3
"""Shared replay helpers for Phase 2 Track B.

Track B needs the same 100 x 80 endpoint ordering as the P2-0 Track A latent
artifact. The Phase 1 outputs do not store raw action sequences or terminal
pixels, so these helpers reproduce the Track A replay path without modifying
Phase 1 code.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import (  # noqa: E402
    execute_raw_actions,
    load_pair_rows,
    prepare_pair_info,
)
from lewm_audit.eval.oracle_cem import cost_v1_hinge  # noqa: E402
from lewm_audit.eval.pusht import analyze_offset, prepare_dataset_index  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    ACTION_SOURCE_ORDER,
    DEFAULT_PAIRS_PATH,
    RANDOM_WAYPOINTS,
    SOURCE_LABELS,
    TOPK,
    load_pairs,
    make_policy_namespace,
    make_three_cost_namespace,
    parse_action_counts,
    parse_pair_ids,
    select_action_sequences,
    validate_requested_pair_offsets,
)
from scripts.phase2.extract_latents import (  # noqa: E402
    DEFAULT_ACTION_COUNTS,
    DEFAULT_RAW_CHECKPOINT_DIR,
    ensure_converted_checkpoint,
    source_index_for_action,
)


DEFAULT_LATENT_ARTIFACT = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"


@dataclass
class TrackAReplayContext:
    """Fully initialized state needed to replay Track A endpoints."""

    args: argparse.Namespace
    pairs_data: dict[str, Any]
    requested_pairs: list[dict[str, Any]]
    offset: int
    dataset: Any
    valid_action_indices: np.ndarray
    policy: Any
    model: torch.nn.Module
    cost_args: argparse.Namespace


def add_replay_args(parser: argparse.ArgumentParser) -> None:
    """Add common Track A replay arguments to a parser."""

    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--latent-artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument("--raw-checkpoint-dir", type=Path, default=DEFAULT_RAW_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    parser.add_argument(
        "--action-counts",
        type=parse_action_counts,
        default=DEFAULT_ACTION_COUNTS,
    )
    parser.add_argument("--reference-atol", type=float, default=1e-3)


def build_replay_context(args: argparse.Namespace) -> TrackAReplayContext:
    """Initialize the dataset, LeWM policy, and Track A action-selection state."""

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.raw_checkpoint_dir = args.raw_checkpoint_dir.expanduser().resolve()
    args.checkpoint_dir = ensure_converted_checkpoint(
        args.raw_checkpoint_dir,
        args.checkpoint_dir,
    )
    args.device = resolve_device(args.device)

    if args.action_counts["CEM_early"] > TOPK or args.action_counts["CEM_late"] > TOPK:
        raise ValueError("CEM action counts must be <= TOPK=30")

    pairs_data, requested_pairs = load_pairs(
        args.pairs_path,
        max_pairs=args.max_pairs,
        pair_ids=args.pair_ids,
    )
    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    if offset % ACTION_BLOCK != 0:
        raise ValueError("Track A offset must be divisible by action_block=5")
    validate_requested_pair_offsets(requested_pairs, offset=offset)

    dataset_path = Path(pair_metadata["dataset_path"])
    args.cache_dir = dataset_path.parent
    args.dataset_name = dataset_path.stem

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    index = prepare_dataset_index(dataset)
    analysis = analyze_offset(index, offset)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=int(args.seed),
        ),
        process,
    )
    model = policy.solver.model
    cost_args = make_three_cost_namespace(
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        device=args.device,
        seed=int(args.seed),
        offset=offset,
        max_cem_count=max(args.action_counts["CEM_early"], args.action_counts["CEM_late"]),
    )
    return TrackAReplayContext(
        args=args,
        pairs_data=pairs_data,
        requested_pairs=requested_pairs,
        offset=offset,
        dataset=dataset,
        valid_action_indices=analysis["valid_indices"],
        policy=policy,
        model=model,
        cost_args=cost_args,
    )


def prepare_pair_sequences(
    ctx: TrackAReplayContext,
    pair_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return initial row, goal row, and the exact Track A action sequences."""

    pair_id = int(pair_spec["pair_id"])
    pair_rows = load_pair_rows(ctx.dataset, int(pair_spec["start_row"]), ctx.offset)
    initial = pair_rows["initial"]
    goal = pair_rows["goal"]
    prepared_info = prepare_pair_info(ctx.policy, initial["pixels"], goal["pixels"])
    raw_steps = int(pair_spec["goal_row"]) - int(pair_spec["start_row"])
    sequences = select_action_sequences(
        dataset=ctx.dataset,
        valid_action_indices=ctx.valid_action_indices,
        policy=ctx.policy,
        model=ctx.model,
        prepared_info=prepared_info,
        args=ctx.cost_args,
        pair_id=pair_id,
        raw_steps=raw_steps,
        action_counts=ctx.args.action_counts,
    )
    expected_actions = sum(ctx.args.action_counts.values())
    if len(sequences) != expected_actions:
        raise RuntimeError(
            f"Pair {pair_id} produced {len(sequences)} actions, expected {expected_actions}"
        )
    return initial, goal, sequences


def load_latent_artifact(path: Path) -> dict[str, Any]:
    """Load the P2-0 Track A latent artifact for ordering and V1 validation."""

    if not path.exists():
        raise FileNotFoundError(f"Missing latent artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def latent_index_by_pair_action(artifact: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Map (pair_id, action_id) to row index in the latent artifact."""

    pair_ids = artifact["pair_id"].detach().cpu().numpy()
    action_ids = artifact["action_id"].detach().cpu().numpy()
    return {
        (int(pair_id), int(action_id)): idx
        for idx, (pair_id, action_id) in enumerate(zip(pair_ids, action_ids, strict=True))
    }


def validate_replayed_action(
    *,
    artifact: dict[str, Any],
    index_by_key: dict[tuple[int, int], int],
    pair_id: int,
    action_id: int,
    source: str,
    source_index: int,
    v1_cost: float,
    atol: float,
) -> int:
    """Validate replay ordering/source/V1 against the latent artifact and return row index."""

    key = (int(pair_id), int(action_id))
    if key not in index_by_key:
        raise KeyError(f"Missing pair/action in latent artifact: {key}")
    idx = index_by_key[key]
    expected_source = artifact["source"][idx]
    expected_source_index = int(artifact["source_index"][idx])
    expected_v1 = float(artifact["v1_cost"][idx])
    if source != expected_source:
        raise RuntimeError(
            f"Source mismatch at {key}: replay={source}, artifact={expected_source}"
        )
    if int(source_index) != expected_source_index:
        raise RuntimeError(
            f"Source-index mismatch at {key}: replay={source_index}, "
            f"artifact={expected_source_index}"
        )
    if abs(float(v1_cost) - expected_v1) > atol:
        raise RuntimeError(
            f"V1 mismatch at {key}: replay={float(v1_cost):.6f}, "
            f"artifact={expected_v1:.6f}"
        )
    return idx


def rollout_action(
    *,
    env,
    initial: dict[str, Any],
    goal: dict[str, Any],
    sequence: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], float]:
    """Execute one action sequence and return rollout data plus V1 hinge cost."""

    goal_state = np.asarray(goal["state"], dtype=np.float32)
    rollout = execute_raw_actions(
        env,
        initial_state=np.asarray(initial["state"], dtype=np.float32),
        goal_state=goal_state,
        raw_actions=sequence["raw"],
        seed=int(seed),
    )
    v1_cost = float(cost_v1_hinge(rollout["terminal_state"], goal_state))
    return rollout, v1_cost


def source_and_index(
    *,
    source_counts: dict[str, int],
    sequence: dict[str, Any],
) -> tuple[str, int]:
    """Return the canonical Track A source label and per-source index."""

    source = SOURCE_LABELS[sequence["source"]]
    source_index = source_index_for_action(source_counts, source)
    return source, int(source_index)


def main() -> int:
    """Sanity check helper imports and defaults."""

    parser = argparse.ArgumentParser(description="Sanity-check Track B replay helpers.")
    add_replay_args(parser)
    parser.add_argument("--skip-init", action="store_true")
    args = parser.parse_args()
    print(f"project_root: {PROJECT_ROOT}")
    print(f"default_pairs_path: {args.pairs_path}")
    print(f"default_latent_artifact: {args.latent_artifact}")
    print(f"action_source_order: {ACTION_SOURCE_ORDER}")
    if not args.skip_init:
        ctx = build_replay_context(args)
        print(f"pairs_loaded: {len(ctx.requested_pairs)}")
        print(f"offset: {ctx.offset}")
        print(f"device: {ctx.args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
