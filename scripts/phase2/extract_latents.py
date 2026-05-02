#!/usr/bin/env python3
"""Replay Track A and save terminal/goal latents for P2-0.

Phase 1 Track A JSONs store scalar costs only. They do not store action
sequences, terminal pixels, or terminal embeddings, so P2-0 must reproduce the
Track A action-generation pipeline and save an augmented latent artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
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
    block_pose_metrics,
    compute_model_costs,
    encode_pixels,
    execute_raw_actions,
    load_pair_rows,
    prepare_pair_info,
    squared_l2,
)
from lewm_audit.eval.oracle_cem import cost_v1_hinge  # noqa: E402
from lewm_audit.eval.pusht import analyze_offset, prepare_dataset_index  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    ACTION_SOURCE_ORDER,
    DEFAULT_PAIRS_PATH,
    IMG_SIZE,
    RANDOM_WAYPOINTS,
    SOURCE_LABELS,
    TOPK,
    get_git_commit,
    load_pairs,
    make_policy_namespace,
    make_three_cost_namespace,
    parse_action_counts,
    parse_pair_ids,
    select_action_sequences,
    validate_requested_pair_offsets,
)


DEFAULT_RAW_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "lewm-pusht"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
DEFAULT_REFERENCE = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
DEFAULT_ACTION_COUNTS = parse_action_counts("20,20,20,20")
LATENT_DIM = 192


def iso_now() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    return json.loads(path.read_text())


def ensure_converted_checkpoint(raw_checkpoint_dir: Path, converted_dir: Path) -> Path:
    """Return a stable-worldmodel object checkpoint directory.

    The raw LeWM checkpoint is ``weights.pt`` + ``config.json``. Phase 1 planning
    uses stable-worldmodel's ``AutoCostModel``, which expects a converted object
    checkpoint. If that converted checkpoint already exists, this function leaves
    it untouched; otherwise it creates it from the raw files.
    """
    raw_checkpoint_dir = raw_checkpoint_dir.expanduser().resolve()
    converted_dir = converted_dir.expanduser().resolve()
    weights_path = raw_checkpoint_dir / "weights.pt"
    config_path = raw_checkpoint_dir / "config.json"
    object_path = converted_dir / "lewm_object.ckpt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing raw LeWM weights: {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing raw LeWM config: {config_path}")
    if object_path.exists():
        return converted_dir

    from scripts.verify_checkpoint import build_model_from_config  # noqa: PLC0415

    cfg = load_json(config_path)
    model = build_model_from_config(cfg)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    converted_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, object_path)
    return converted_dir


def source_index_for_action(source_counts: dict[str, int], source: str) -> int:
    """Return and increment the inferred per-source action index."""
    source_index = source_counts.get(source, 0)
    source_counts[source] = source_index + 1
    return source_index


def reference_pairs_by_id(path: Path) -> dict[int, dict]:
    """Load the original Track A scalar-cost output keyed by pair ID."""
    data = load_json(path)
    return {int(pair["pair_id"]): pair for pair in data.get("pairs", [])}


def validate_against_reference(
    *,
    record: dict,
    reference_action: dict,
    c_real_z: float,
    c_model: float,
    c_real_state: float,
    success: bool,
    atol: float,
) -> dict:
    """Compare replayed scalar outputs against the Phase 1 reference record."""
    mismatches = {}
    if record["source"] != reference_action.get("source"):
        mismatches["source"] = (record["source"], reference_action.get("source"))
    if abs(c_real_z - float(reference_action["C_real_z"])) > atol:
        mismatches["C_real_z"] = (c_real_z, float(reference_action["C_real_z"]))
    if abs(c_model - float(reference_action["C_model"])) > atol:
        mismatches["C_model"] = (c_model, float(reference_action["C_model"]))
    if abs(c_real_state - float(reference_action["C_real_state"])) > atol:
        mismatches["C_real_state"] = (
            c_real_state,
            float(reference_action["C_real_state"]),
        )
    if bool(success) != bool(reference_action["success"]):
        mismatches["success"] = (bool(success), bool(reference_action["success"]))
    return mismatches


def empty_artifact_metadata(args: argparse.Namespace, *, offset: int, n_pairs: int) -> dict:
    """Return metadata for a new latent artifact."""
    return {
        "format": "p2_0_track_a_latents",
        "created_at": iso_now(),
        "git_commit": get_git_commit(),
        "seed": int(args.seed),
        "device": args.device,
        "pairs_path": str(args.pairs_path),
        "reference_three_cost_path": str(args.reference_three_cost_path),
        "raw_checkpoint_dir": str(args.raw_checkpoint_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_cache_dir": str(args.cache_dir),
        "dataset_name": args.dataset_name,
        "offset": int(offset),
        "n_pairs_requested": int(n_pairs),
        "action_counts": dict(args.action_counts),
        "action_source_order": ACTION_SOURCE_ORDER,
        "latent_dim": LATENT_DIM,
        "image_size": IMG_SIZE,
        "action_block": ACTION_BLOCK,
        "random_waypoints": RANDOM_WAYPOINTS,
        "scenario": (
            "Scenario A: Phase 1 did not store actions or terminal observations; "
            "this artifact was produced by replaying Track A with the original seeds."
        ),
    }


def save_artifact(path: Path, metadata: dict, records: list[dict]) -> None:
    """Serialize accumulated latent records to a torch artifact."""
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
        v1_cost = torch.as_tensor([item["v1_cost"] for item in records], dtype=torch.float32)
        success = torch.as_tensor([item["success"] for item in records], dtype=torch.bool)
        c_real_z = torch.as_tensor([item["C_real_z"] for item in records], dtype=torch.float32)
        c_model = torch.as_tensor([item["C_model"] for item in records], dtype=torch.float32)
        c_real_state = torch.as_tensor(
            [item["C_real_state"] for item in records],
            dtype=torch.float32,
        )
        block_pos_dist = torch.as_tensor(
            [item["block_pos_dist"] for item in records],
            dtype=torch.float32,
        )
        angle_dist = torch.as_tensor(
            [item["angle_dist"] for item in records],
            dtype=torch.float32,
        )
    else:
        z_terminal = torch.empty((0, LATENT_DIM), dtype=torch.float32)
        z_goal = torch.empty((0, LATENT_DIM), dtype=torch.float32)
        pair_id = torch.empty((0,), dtype=torch.long)
        action_id = torch.empty((0,), dtype=torch.long)
        source_index = torch.empty((0,), dtype=torch.long)
        v1_cost = torch.empty((0,), dtype=torch.float32)
        success = torch.empty((0,), dtype=torch.bool)
        c_real_z = torch.empty((0,), dtype=torch.float32)
        c_model = torch.empty((0,), dtype=torch.float32)
        c_real_state = torch.empty((0,), dtype=torch.float32)
        block_pos_dist = torch.empty((0,), dtype=torch.float32)
        angle_dist = torch.empty((0,), dtype=torch.float32)

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
        "v1_cost": v1_cost,
        "success": success,
        "C_real_z": c_real_z,
        "C_model": c_model,
        "C_real_state": c_real_state,
        "block_pos_dist": block_pos_dist,
        "angle_dist": angle_dist,
    }
    torch.save(artifact, path)


def load_existing_artifact(path: Path) -> tuple[dict | None, list[dict]]:
    """Load existing artifact records for resume, if present."""
    if not path.exists():
        return None, []
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    records = []
    n_records = int(artifact["pair_id"].numel())
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
                "v1_cost": float(artifact["v1_cost"][idx]),
                "success": bool(artifact["success"][idx]),
                "C_real_z": float(artifact["C_real_z"][idx]),
                "C_model": float(artifact["C_model"][idx]),
                "C_real_state": float(artifact["C_real_state"][idx]),
                "block_pos_dist": float(artifact["block_pos_dist"][idx]),
                "angle_dist": float(artifact["angle_dist"][idx]),
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
    cost_args: argparse.Namespace,
    reference_pair: dict | None,
) -> tuple[list[dict], list[dict]]:
    """Replay one pair and return latent records plus validation mismatches."""
    pair_id = int(pair_spec["pair_id"])
    pair_rows = load_pair_rows(dataset, int(pair_spec["start_row"]), cost_args.offset)
    initial = pair_rows["initial"]
    goal = pair_rows["goal"]
    goal_state = np.asarray(goal["state"], dtype=np.float32)
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    goal_emb_t = encode_pixels(policy, model, goal["pixels"])
    goal_emb = goal_emb_t.detach().cpu()[0].to(dtype=torch.float32)
    raw_steps = int(pair_spec["goal_row"]) - int(pair_spec["start_row"])

    sequences = select_action_sequences(
        dataset=dataset,
        valid_action_indices=valid_action_indices,
        policy=policy,
        model=model,
        prepared_info=prepared_info,
        args=cost_args,
        pair_id=pair_id,
        raw_steps=raw_steps,
        action_counts=args.action_counts,
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
    mismatches = []

    for action_id, (sequence, model_cost) in enumerate(zip(sequences, model_costs, strict=True)):
        source = SOURCE_LABELS[sequence["source"]]
        source_index = source_index_for_action(source_counts, source)
        rollout = execute_raw_actions(
            env,
            initial_state=np.asarray(initial["state"], dtype=np.float32),
            goal_state=goal_state,
            raw_actions=sequence["raw"],
            seed=args.seed + pair_id * 10_000 + action_id,
        )
        terminal_emb_t = encode_pixels(policy, model, rollout["terminal_pixels"])
        terminal_emb = terminal_emb_t.detach().cpu()[0].to(dtype=torch.float32)
        metrics = block_pose_metrics(rollout["terminal_state"], goal_state)
        c_real_z = squared_l2(terminal_emb_t, goal_emb_t)
        c_model = float(model_cost)
        c_real_state = float(metrics["c_real_state"])
        success = bool(metrics["success"])
        v1_cost = float(cost_v1_hinge(rollout["terminal_state"], goal_state))
        record = {
            "pair_id": pair_id,
            "action_id": int(action_id),
            "source": source,
            "source_index": int(source_index),
            "action_key": f"{pair_id}:{source}:{source_index}",
            "cell": str(pair_spec["cell"]),
            "z_terminal": terminal_emb,
            "z_goal": goal_emb,
            "v1_cost": v1_cost,
            "success": success,
            "C_real_z": c_real_z,
            "C_model": c_model,
            "C_real_state": c_real_state,
            "block_pos_dist": float(metrics["block_pos_dist"]),
            "angle_dist": float(metrics["angle_dist"]),
        }
        if reference_pair is not None:
            reference_actions = reference_pair.get("actions", [])
            if action_id >= len(reference_actions):
                mismatches.append({"pair_id": pair_id, "action_id": action_id, "missing": True})
            else:
                diff = validate_against_reference(
                    record=record,
                    reference_action=reference_actions[action_id],
                    c_real_z=c_real_z,
                    c_model=c_model,
                    c_real_state=c_real_state,
                    success=success,
                    atol=args.reference_atol,
                )
                if diff:
                    mismatches.append(
                        {"pair_id": pair_id, "action_id": action_id, "mismatches": diff}
                    )
        pair_records.append(record)
    return pair_records, mismatches


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--reference-three-cost-path", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--raw-checkpoint-dir", type=Path, default=DEFAULT_RAW_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-reference-validation", action="store_true")
    parser.add_argument("--reference-atol", type=float, default=1e-3)
    parser.add_argument(
        "--action-counts",
        type=parse_action_counts,
        default=DEFAULT_ACTION_COUNTS,
    )
    return parser.parse_args()


def main() -> int:
    """Run the Track A latent extraction."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.reference_three_cost_path = args.reference_three_cost_path.expanduser().resolve()
    args.raw_checkpoint_dir = args.raw_checkpoint_dir.expanduser().resolve()
    args.checkpoint_dir = ensure_converted_checkpoint(
        args.raw_checkpoint_dir,
        args.checkpoint_dir,
    )
    args.output = args.output.expanduser().resolve()
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

    existing_metadata, records = load_existing_artifact(args.output) if args.resume else (None, [])
    completed_pair_ids = {int(record["pair_id"]) for record in records}
    metadata = empty_artifact_metadata(args, offset=offset, n_pairs=len(requested_pairs))
    if existing_metadata:
        metadata.update(existing_metadata)
        metadata.update(empty_artifact_metadata(args, offset=offset, n_pairs=len(requested_pairs)))

    reference_by_pair = (
        reference_pairs_by_id(args.reference_three_cost_path)
        if not args.no_reference_validation
        else {}
    )

    print("== P2-0 Track A latent extraction ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"reference_three_cost_path: {args.reference_three_cost_path}")
    print(f"raw_checkpoint_dir: {args.raw_checkpoint_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"output: {args.output}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"action_counts: {args.action_counts}")
    print(f"resume_completed_pairs: {len(completed_pair_ids)}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    index = prepare_dataset_index(dataset)
    analysis = analyze_offset(index, offset)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=args.seed,
        ),
        process,
    )
    model = policy.solver.model
    cost_args = make_three_cost_namespace(
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        device=args.device,
        seed=args.seed,
        offset=offset,
        max_cem_count=max(args.action_counts["CEM_early"], args.action_counts["CEM_late"]),
    )

    total_started = time.time()
    all_mismatches = []
    env = gym.make("swm/PushT-v1")
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
            pair_records, mismatches = extract_pair_records(
                pair_spec=pair_spec,
                dataset=dataset,
                valid_action_indices=analysis["valid_indices"],
                policy=policy,
                model=model,
                env=env,
                args=args,
                cost_args=cost_args,
                reference_pair=reference_by_pair.get(pair_id),
            )
            records.extend(pair_records)
            completed_pair_ids.add(pair_id)
            all_mismatches.extend(mismatches)
            if mismatches:
                preview = mismatches[:3]
                raise RuntimeError(
                    f"Reference validation failed for pair {pair_id}; examples={preview}"
                )
            save_artifact(args.output, metadata, records)
            successes = sum(bool(record["success"]) for record in pair_records)
            print(
                f"saved_pair_records={len(pair_records)} successes={successes}; "
                f"elapsed_seconds={time.time() - pair_started:.2f}"
            )
    finally:
        env.close()

    metadata["finished_at"] = iso_now()
    metadata["wallclock_seconds"] = time.time() - total_started
    save_artifact(args.output, metadata, records)
    print("\n== P2-0 latent extraction summary ==")
    print(f"pairs_completed: {len(completed_pair_ids)}")
    print(f"records: {len(records)}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
