#!/usr/bin/env python3
"""Generate state-capture rollouts for locked or requested Track A anchor pairs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.

from lewm_audit.diagnostics.three_cost import (
    SOURCE_CEM_LATE,
    block_pose_metrics,
    blocked_normalized_to_raw,
    compute_model_costs,
    encode_pixels,
    execute_raw_actions,
    generate_cem_action_sequences,
    load_pair_rows,
    prepare_pair_info,
    squared_l2,
    to_jsonable,
)
from lewm_audit.eval.oracle_cem import (
    cost_v1_hinge,
    cost_v2_indicator,
    cost_v3_baseline,
    cem_with_oracle_cost,
)
from lewm_audit.eval.pusht import analyze_offset, prepare_dataset_index
from lewm_audit.eval.state_rollout import rollout_with_state_capture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)


DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
DEFAULT_TRACK_A_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1" / "per_trajectory"
LOCKED_PAIR_IDS = {
    "P_a": 80,
    "P_b": 74,
    "P_c": 6,
    "P_d": 93,
    "P_e": 20,
}
VARIANTS = ("latent", "V3", "V1", "V2")
ORACLE_COSTS = {
    "V3": cost_v3_baseline,
    "V1": cost_v1_hinge,
    "V2": cost_v2_indicator,
}
NUM_SAMPLES = 300
CEM_ITERS = 30
CEM_EARLY_ITERS = 3
CEM_LATE_ITERS = 30
TOPK = 30
VAR_SCALE = 1.0
PLANNING_HORIZON = 5
RECEDING_HORIZON = 5
ACTION_BLOCK = 5
IMG_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--track-a-path", type=Path, default=DEFAULT_TRACK_A_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_pair_ids(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise argparse.ArgumentTypeError("--pair-ids must include at least one id")
    return list(dict.fromkeys(values))


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


def make_policy_args(*, checkpoint_dir: Path, device: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=checkpoint_dir,
        device=device,
        num_samples=NUM_SAMPLES,
        var_scale=VAR_SCALE,
        cem_iters=CEM_ITERS,
        topk=TOPK,
        seed=seed,
        horizon=PLANNING_HORIZON,
        receding_horizon=RECEDING_HORIZON,
        action_block=ACTION_BLOCK,
        img_size=IMG_SIZE,
    )


def make_cem_args(*, checkpoint_dir: Path, device: str, seed: int, num_per_source: int) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=checkpoint_dir,
        device=device,
        seed=seed,
        offset=50,
        num_per_source=num_per_source,
        img_size=IMG_SIZE,
        action_block=ACTION_BLOCK,
        num_samples=NUM_SAMPLES,
        cem_early_iters=CEM_EARLY_ITERS,
        cem_late_iters=CEM_LATE_ITERS,
        topk=TOPK,
        var_scale=VAR_SCALE,
    )


def load_selected_pairs(path: Path, pair_ids: list[int] | None) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    by_id = {int(pair["pair_id"]): pair for pair in data["pairs"]}
    requested_ids = pair_ids if pair_ids is not None else list(LOCKED_PAIR_IDS.values())
    missing = sorted(set(requested_ids) - set(by_id))
    if missing:
        raise ValueError(f"Requested pair_ids not found in {path}: {missing}")
    pairs = [by_id[pair_id] for pair_id in requested_ids]
    return data, pairs


def validate_locked_selection(pairs: list[dict], all_pairs: list[dict]) -> None:
    by_id = {int(pair["pair_id"]): pair for pair in pairs}
    checks = [
        ("P_a", "D3xR1", min(int(p["pair_id"]) for p in all_pairs if p["cell"] == "D3xR1")),
        ("P_b", "D3xR0", min(int(p["pair_id"]) for p in all_pairs if p["cell"] == "D3xR0")),
        ("P_c", "D0xR1", min(int(p["pair_id"]) for p in all_pairs if p["cell"] == "D0xR1")),
        ("P_d", "D3xR3", min(int(p["pair_id"]) for p in all_pairs if p["cell"] == "D3xR3")),
        ("P_e", "D0xR3", 20),
    ]
    for label, cell, expected_pair_id in checks:
        actual = by_id[LOCKED_PAIR_IDS[label]]
        if actual["cell"] != cell or int(actual["pair_id"]) != expected_pair_id:
            raise ValueError(
                f"{label} selection mismatch: expected pair {expected_pair_id} in {cell}, "
                f"got pair {actual['pair_id']} in {actual['cell']}"
            )


def output_path(output_dir: Path, pair_id: int, cell: str, variant: str) -> Path:
    return output_dir / f"{pair_id}_{cell}_{variant}.json"


def oracle_reference_path(cell: str, variant: str) -> Path:
    row = cell.split("x", maxsplit=1)[0].lower()
    if variant == "V3":
        return PROJECT_ROOT / "results" / "phase1" / f"{row}_oracle_ablation" / f"{row}_oracle_V3.json"
    return PROJECT_ROOT / "results" / "phase1" / f"{variant.lower()}_oracle_ablation" / f"{variant.lower()}_{row}.json"


def reference_record(pair_id: int, cell: str, variant: str, track_a_data: dict) -> dict:
    if variant == "latent":
        pair = next(pair for pair in track_a_data["pairs"] if int(pair["pair_id"]) == pair_id)
        records = [action for action in pair["actions"] if action["source"] == "CEM_late"]
    else:
        data = json.loads(oracle_reference_path(cell, variant).read_text())
        pair = next(pair for pair in data["pairs"] if int(pair["pair_id"]) == pair_id)
        records = [
            action
            for action in pair["actions"]
            if action["source"] == f"CEM_late_{variant}" and int(action.get("source_index", -1)) == 0
        ]
    if not records:
        raise ValueError(f"No reference CEM_late record for pair {pair_id}, variant {variant}")
    return records[0]


def select_latent_late_sequence(
    *,
    model,
    prepared_info: dict,
    action_processor,
    pair_id: int,
    raw_steps: int,
    args: argparse.Namespace,
) -> dict:
    cem_args = make_cem_args(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
        num_per_source=TOPK,
    )
    sequences = generate_cem_action_sequences(
        model=model,
        prepared_info=prepared_info,
        args=cem_args,
        horizon_blocks=raw_steps // ACTION_BLOCK,
        action_dim=ACTION_BLOCK * 2,
        action_processor=action_processor,
        pair_index=pair_id,
    )
    late = [sequence for sequence in sequences if sequence["source"] == SOURCE_CEM_LATE]
    if len(late) != TOPK:
        raise RuntimeError(f"Expected {TOPK} latent late elites for pair {pair_id}, got {len(late)}")
    return min(late, key=lambda sequence: float(sequence["cem_model_cost"]))


def select_oracle_late_sequence(
    *,
    variant: str,
    env_factory,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    action_processor,
    pair_id: int,
    raw_steps: int,
    seed: int,
) -> dict:
    horizon_blocks = raw_steps // ACTION_BLOCK

    def action_transform(blocked: np.ndarray) -> np.ndarray:
        return blocked_normalized_to_raw(
            blocked,
            action_processor=action_processor,
            action_block=ACTION_BLOCK,
        )

    cem = cem_with_oracle_cost(
        env_factory,
        init_state,
        goal_state,
        ORACLE_COSTS[variant],
        n_samples=NUM_SAMPLES,
        n_iters=CEM_ITERS,
        n_elites=TOPK,
        horizon=horizon_blocks,
        receding_horizon=RECEDING_HORIZON,
        action_block=ACTION_BLOCK,
        rng=np.random.default_rng(seed + pair_id * 1009),
        action_dim=ACTION_BLOCK * 2,
        var_scale=VAR_SCALE,
        action_transform=action_transform,
    )
    late_iter = CEM_LATE_ITERS - 1
    candidate_idx = int(cem["elite_indices_per_iter"][late_iter][0])
    return {
        "source": f"CEM_late_{variant}",
        "source_index": 0,
        "cem_iter": CEM_LATE_ITERS,
        "cem_rank": 0,
        "cem_oracle_cost": float(cem["elite_costs_per_iter"][late_iter][0]),
        "blocked_normalized": cem["blocked_candidates_per_iter"][late_iter][candidate_idx].astype(np.float32),
        "raw": cem["candidates_per_iter"][late_iter][candidate_idx].astype(np.float32),
    }


def build_rollout_record(
    *,
    pair_spec: dict,
    variant: str,
    sequence: dict,
    reference: dict,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    env_factory,
    eval_env,
    policy,
    model,
    prepared_info: dict,
    goal_emb,
    git_commit: str,
    seed: int,
) -> tuple[dict, bool]:
    pair_id = int(pair_spec["pair_id"])
    reference_action_id = int(reference["action_id"])
    rollout_seed = seed + pair_id * 10_000 + reference_action_id
    captured = rollout_with_state_capture(
        env_factory,
        init_state,
        goal_state,
        sequence["raw"],
        seed=rollout_seed,
    )
    terminal = execute_raw_actions(
        eval_env,
        initial_state=init_state,
        goal_state=goal_state,
        raw_actions=sequence["raw"],
        seed=rollout_seed,
    )
    np.testing.assert_allclose(
        captured["states"][-1],
        terminal["terminal_state"],
        atol=1e-5,
        err_msg=f"state capture diverged for pair {pair_id}, variant {variant}",
    )
    terminal_emb = encode_pixels(policy, model, terminal["terminal_pixels"])
    blocked = sequence["blocked_normalized"][None, ...]
    model_cost = float(compute_model_costs(model, prepared_info, blocked)[0])
    metrics = block_pose_metrics(captured["states"][-1], goal_state)
    final_success = bool(metrics["success"])
    matches_reference = final_success == bool(reference["success"])
    record = {
        "pair_id": pair_id,
        "cell": pair_spec["cell"],
        "variant": variant,
        "init_state": init_state,
        "goal_state": goal_state,
        "action_seq": np.asarray(sequence["raw"], dtype=np.float32),
        "states": captured["states"],
        "block_xy": captured["block_xy"],
        "agent_xy": captured["agent_xy"],
        "block_angle": captured["block_angle"],
        "step_success": captured["step_success"],
        "final_success": final_success,
        "C_real_state_final": float(metrics["c_real_state"]),
        "C_real_z_final": squared_l2(terminal_emb, goal_emb),
        "block_pos_dist_final": float(metrics["block_pos_dist"]),
        "angle_dist_final": float(metrics["angle_dist"]),
        "C_model": model_cost,
        "reference_success": bool(reference["success"]),
        "reference_action_id": reference_action_id,
        "reference_C_real_state": float(reference["C_real_state"]),
        "git_commit": git_commit,
        "seed": seed,
    }
    for optional in ("cem_iter", "cem_rank", "cem_model_cost", "cem_oracle_cost"):
        if optional in sequence:
            record[optional] = sequence[optional]
    return record, matches_reference


def main() -> int:
    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.track_a_path = args.track_a_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.device = resolve_device(args.device)
    args.checkpoint_dir = DEFAULT_CHECKPOINT_DIR.expanduser().resolve()

    pairs_data, pairs = load_selected_pairs(args.pairs_path, args.pair_ids)
    if args.pair_ids is None:
        validate_locked_selection(pairs, pairs_data["pairs"])
    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    if offset != 50:
        raise ValueError(f"Expected locked Track A offset=50, got {offset}")
    dataset_path = Path(pair_metadata["dataset_path"])
    cache_dir = dataset_path.parent
    dataset_name = dataset_path.stem
    track_a_data = json.loads(args.track_a_path.read_text())

    print("== Per-trajectory rollout setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"track_a_path: {args.track_a_path}")
    print(f"output_dir: {args.output_dir}")
    print(f"dataset_name: {dataset_name}")
    print(f"cache_dir: {cache_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"device: {args.device}")
    label = "locked_pairs" if args.pair_ids is None else "requested_pairs"
    print(f"{label}:")
    for idx, pair in enumerate(pairs, start=1):
        print(f"  P{idx}: pair_id={pair['pair_id']} cell={pair['cell']}")

    dataset = get_dataset(cache_dir, dataset_name)
    index = prepare_dataset_index(dataset)
    analyze_offset(index, offset)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_args(checkpoint_dir=args.checkpoint_dir, device=args.device, seed=args.seed),
        process,
    )
    model = policy.solver.model
    action_processor = policy.process["action"]
    git_commit = get_git_commit()

    def env_factory():
        return gym.make("swm/PushT-v1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_env = gym.make("swm/PushT-v1")
    total_started = time.time()
    mismatches = []
    final_success = {}
    try:
        for pair_spec in pairs:
            pair_id = int(pair_spec["pair_id"])
            pair_rows = load_pair_rows(dataset, int(pair_spec["start_row"]), offset)
            initial = pair_rows["initial"]
            goal = pair_rows["goal"]
            init_state = np.asarray(initial["state"], dtype=np.float32)
            goal_state = np.asarray(goal["state"], dtype=np.float32)
            prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
            goal_emb = encode_pixels(policy, model, goal["pixels"])
            raw_steps = int(pair_spec["goal_row"]) - int(pair_spec["start_row"])
            final_success[pair_id] = {}
            print(f"\n== pair_id={pair_id} cell={pair_spec['cell']} ==")
            for variant in VARIANTS:
                path = output_path(args.output_dir, pair_id, pair_spec["cell"], variant)
                if path.exists() and not args.overwrite:
                    existing = json.loads(path.read_text())
                    final_success[pair_id][variant] = bool(existing["final_success"])
                    print(f"[{variant}] exists: {path}")
                    continue
                started = time.time()
                reference = reference_record(pair_id, pair_spec["cell"], variant, track_a_data)
                if variant == "latent":
                    sequence = select_latent_late_sequence(
                        model=model,
                        prepared_info=prepared_info,
                        action_processor=action_processor,
                        pair_id=pair_id,
                        raw_steps=raw_steps,
                        args=args,
                    )
                else:
                    sequence = select_oracle_late_sequence(
                        variant=variant,
                        env_factory=env_factory,
                        init_state=init_state,
                        goal_state=goal_state,
                        action_processor=action_processor,
                        pair_id=pair_id,
                        raw_steps=raw_steps,
                        seed=args.seed,
                    )
                record, matches_reference = build_rollout_record(
                    pair_spec=pair_spec,
                    variant=variant,
                    sequence=sequence,
                    reference=reference,
                    init_state=init_state,
                    goal_state=goal_state,
                    env_factory=env_factory,
                    eval_env=eval_env,
                    policy=policy,
                    model=model,
                    prepared_info=prepared_info,
                    goal_emb=goal_emb,
                    git_commit=git_commit,
                    seed=args.seed,
                )
                path.write_text(json.dumps(to_jsonable(record), indent=2, allow_nan=False) + "\n")
                final_success[pair_id][variant] = bool(record["final_success"])
                if not matches_reference:
                    mismatches.append(
                        {
                            "pair_id": pair_id,
                            "cell": pair_spec["cell"],
                            "variant": variant,
                            "reference_success": bool(reference["success"]),
                            "captured_success": bool(record["final_success"]),
                        }
                    )
                print(
                    f"[{variant}] final_success={record['final_success']} "
                    f"C_real_state={record['C_real_state_final']:.3f} "
                    f"elapsed={time.time() - started:.2f}s"
                )
    finally:
        eval_env.close()

    elapsed = time.time() - total_started
    print("\n== Per-trajectory rollout summary ==")
    print(f"wallclock_seconds: {elapsed:.2f}")
    print(f"json_files: {len(list(args.output_dir.glob('*.json')))}")
    print("final_success:")
    for pair_spec in pairs:
        pair_id = int(pair_spec["pair_id"])
        row = final_success.get(pair_id, {})
        values = " ".join(f"{variant}={row.get(variant)}" for variant in VARIANTS)
        print(f"  pair_id={pair_id} cell={pair_spec['cell']}: {values}")
    if mismatches:
        print("ROLL_OUT_REFERENCE_MISMATCHES:")
        print(json.dumps(mismatches, indent=2))
    else:
        print("ROLL_OUT_REFERENCE_MISMATCHES: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
