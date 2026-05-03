#!/usr/bin/env python3
"""Stage A Subspace-CEM smoke test for PushT.

Subspace-CEM uses projected latent cost to update the CEM proposal during
search, then re-ranks the final candidate pool with full-dimensional LeWM cost.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import prepare_pair_info  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    DEFAULT_PAIRS_PATH,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    TOPK,
    VAR_SCALE,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.projected_cem import (  # noqa: E402
    blocked_batch_to_raw_fast,
    euclidean_costs,
    make_projection,
    projected_costs,
    run_cem,
    score_raw_actions,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    ANCHOR_DEFINITIONS,
    LATENT_DIM,
    clean_float,
    deterministic_topk_indices,
    jsonable,
)
from scripts.phase2.train_cem_aware import rollout_candidate_latents  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "subspace_cem" / "stage_a_sanity.json"
DEFAULT_STAGE1B_FULL = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1b_full.json"
SMOKE_PAIR_IDS = (25, 50, 75)
DEFAULT_M = 64
DEFAULT_PROJECTION_SEED = 0
ALGORITHM = "subspace_cem_m64_search_full_rerank"
V1_FAVORABLE_CELLS = ("D3xR0", "D3xR3")
SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the locked 3-pair Stage A sanity check.")
    parser.add_argument("--pairs", type=parse_int_list, default=SMOKE_PAIR_IDS)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--stage1b-full", type=Path, default=DEFAULT_STAGE1B_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.smoke:
        parser.error("Only --smoke mode is implemented for this Stage A sanity script.")
    if int(args.m) <= 0 or int(args.m) > LATENT_DIM:
        parser.error(f"--m must be in [1, {LATENT_DIM}]")
    if int(args.seed) < 0:
        parser.error("--seed must be nonnegative")
    if tuple(int(pair_id) for pair_id in args.pairs) != SMOKE_PAIR_IDS:
        parser.error(f"--pairs is locked to {SMOKE_PAIR_IDS} for the requested smoke test")
    return args


def fmt(value: float | int | bool | None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "Y" if value else "N"
    value = float(value)
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def seconds_to_hms(seconds: float) -> str:
    seconds = float(seconds)
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def load_pair_rows_direct(dataset, pair_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = dataset.get_row_data([int(pair_spec["start_row"]), int(pair_spec["goal_row"])])
    return {key: value[0] for key, value in rows.items()}, {key: value[1] for key, value in rows.items()}


def pair_subsets(pair_spec: dict[str, Any]) -> list[str]:
    pair_id = int(pair_spec["pair_id"])
    cell = str(pair_spec.get("cell", ""))
    memberships: list[str] = []

    for name in ("invisible_quadrant", "sign_reversal", "latent_favorable"):
        definition = ANCHOR_DEFINITIONS[name]
        pair_ids = {int(item) for item in definition.get("pair_ids", [])}
        cells = {str(item) for item in definition.get("cells", [])}
        if pair_id in pair_ids or cell in cells:
            memberships.append(name)
    if cell in V1_FAVORABLE_CELLS:
        memberships.append("v1_favorable")
    if not memberships:
        memberships.append("ordinary")

    order = {name: idx for idx, name in enumerate(SUBSET_ORDER)}
    return sorted(dict.fromkeys(memberships), key=lambda item: order[item])


def run_subspace_cem(
    *,
    model,
    prepared_info: dict[str, Any],
    pair_id: int,
    seed: int,
    projection: torch.Tensor,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    # Projection matrices are generated by make_projection with an independent CPU
    # generator; this model-device generator controls only CEM candidate sampling.
    generator = torch.Generator(device=device).manual_seed(int(seed) + int(pair_id) * 1009)
    mean = torch.zeros((1, PLANNING_HORIZON, ACTION_BLOCK * 2), dtype=torch.float32, device=device)
    var = VAR_SCALE * torch.ones((1, PLANNING_HORIZON, ACTION_BLOCK * 2), dtype=torch.float32, device=device)
    final: dict[str, Any] | None = None
    started = time.time()

    for iter_idx in range(1, CEM_ITERS + 1):
        candidates = torch.randn(
            1,
            NUM_SAMPLES,
            PLANNING_HORIZON,
            ACTION_BLOCK * 2,
            generator=generator,
            device=device,
        )
        candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        z_pred, z_goal = rollout_candidate_latents(model, prepared_info, candidates)
        full_cost = euclidean_costs(z_pred, z_goal)
        select_cost = projected_costs(z_pred, z_goal, projection)
        top_vals, top_inds = torch.topk(select_cost, k=TOPK, dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]

        if iter_idx == CEM_ITERS:
            select_np = select_cost[0].detach().cpu().numpy().astype(np.float64)
            full_np = full_cost[0].detach().cpu().numpy().astype(np.float64)
            top_np = top_vals[0].detach().cpu().numpy().astype(np.float64)
            full_rank1 = int(torch.argmin(full_cost[0]).detach().cpu().item())
            projected_rank1 = int(top_inds[0, 0].detach().cpu().item())
            final = {
                "blocked_candidates": candidates[0].detach().cpu().numpy().astype(np.float32),
                "rank1_blocked": candidates[0, full_rank1].detach().cpu().numpy().astype(np.float32),
                "rank1_candidate_index": full_rank1,
                "projected_rank1_candidate_index": projected_rank1,
                "select_costs": select_np,
                "default_costs": full_np,
                "z_pred": z_pred[0].detach().cpu().numpy().astype(np.float32),
                "z_goal": z_goal[0].detach().cpu().numpy().astype(np.float32),
                "top30_select_costs": top_np,
                "top30_select_cost_std": clean_float(float(np.std(top_np, ddof=0))),
                "select_cost_dynamic_range": clean_float(float(np.max(select_np) - np.min(select_np))),
                "select_cost_min": clean_float(float(np.min(select_np))),
                "select_cost_max": clean_float(float(np.max(select_np))),
                "final_pool_full_cost_std": clean_float(float(np.std(full_np, ddof=0))),
                "full_cost_min": clean_float(float(np.min(full_np))),
                "full_cost_max": clean_float(float(np.max(full_np))),
                "full_cost_dynamic_range": clean_float(float(np.max(full_np) - np.min(full_np))),
                "rank1_full_cost": clean_float(float(full_np[full_rank1])),
                "rank1_projected_cost": clean_float(float(select_np[full_rank1])),
                "projected_rank1_projected_cost": clean_float(float(select_np[projected_rank1])),
            }

        mean = elite_candidates.mean(dim=1)
        var = elite_candidates.std(dim=1)

    if final is None:
        raise RuntimeError("Subspace-CEM final iteration was not captured")
    final["wallclock_seconds"] = clean_float(time.time() - started)
    return final


def score_final_pool(
    *,
    env,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    blocked_candidates: np.ndarray,
    action_processor: Any,
    seed_base: int,
) -> dict[str, Any]:
    raw_actions = blocked_batch_to_raw_fast(blocked_candidates, action_processor=action_processor)
    started = time.time()
    v1_costs, c_real_state, success, metrics = score_raw_actions(
        env=env,
        initial_state=initial_state,
        goal_state=goal_state,
        raw_actions_batch=raw_actions,
        seed_base=int(seed_base),
    )
    candidate_ids = np.arange(int(c_real_state.shape[0]), dtype=np.int64)
    oracle_best = int(
        deterministic_topk_indices(
            c_real_state,
            candidate_ids,
            np.ones(int(c_real_state.shape[0]), dtype=bool),
            1,
        )[0]
    )
    return {
        "raw_actions": raw_actions,
        "v1_costs": v1_costs,
        "c_real_state": c_real_state,
        "success": success,
        "metrics": metrics,
        "oracle_best_candidate_index": oracle_best,
        "simulator_scoring_seconds": clean_float(time.time() - started),
    }


def build_smoke_record(
    *,
    pair_id: int,
    cem_result: dict[str, Any],
    scored: dict[str, Any],
    default_rank1_blocked: np.ndarray,
    action_l2_from_default: float,
    elite_cost_std: float | None,
    cost_dynamic_range: float | None,
) -> dict[str, Any]:
    rank1 = int(cem_result["rank1_candidate_index"])
    oracle_best = int(scored["oracle_best_candidate_index"])
    c_real_state = np.asarray(scored["c_real_state"], dtype=np.float64)
    regret = float(c_real_state[rank1] - c_real_state[oracle_best])
    if regret < 0.0 and regret > -1e-9:
        regret = 0.0
    if regret < 0.0:
        raise RuntimeError(f"Negative selection regret for pair_id={pair_id}: {regret}")

    rank1_blocked = np.asarray(cem_result["rank1_blocked"], dtype=np.float32)
    expected_l2 = float(np.linalg.norm(rank1_blocked - default_rank1_blocked))
    if not math.isclose(float(action_l2_from_default), expected_l2, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError(
            f"Action L2 mismatch for pair_id={pair_id}: "
            f"{action_l2_from_default} != {expected_l2}"
        )

    return {
        "pair_id": int(pair_id),
        "rank1_success": bool(np.asarray(scored["success"], dtype=bool)[rank1]),
        "rank1_C_real_state": clean_float(float(c_real_state[rank1])),
        "elite_cost_std": clean_float(elite_cost_std),
        "cost_dynamic_range": clean_float(cost_dynamic_range),
        "action_l2_from_default": clean_float(float(action_l2_from_default)),
        "selection_regret": clean_float(float(regret)),
        "final_rerank_pool_size": int(c_real_state.shape[0]),
        "final_pool_full_cost_std": clean_float(float(cem_result["final_pool_full_cost_std"])),
    }


def default_pool_full_cost_std(cem_result: dict[str, Any]) -> float:
    full_costs = np.asarray(cem_result["default_costs"], dtype=np.float64)
    return float(np.std(full_costs, ddof=0))


def load_pure_projected_reference(
    path: Path,
    *,
    pair_ids: tuple[int, ...],
    m: int,
    projection_seed: int,
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    if not path.exists():
        return {}, list(pair_ids)
    data = json.loads(path.read_text())
    records: dict[int, dict[str, Any]] = {}
    for record in data.get("records", []):
        if int(record.get("dimension", -1)) != int(m):
            continue
        if int(record.get("projection_seed", -1)) != int(projection_seed):
            continue
        pair_id = int(record.get("pair_id", -1))
        if pair_id in pair_ids:
            if pair_id in records:
                raise RuntimeError(f"Duplicate pure projected record for pair_id={pair_id}, m={m}, seed={projection_seed}")
            records[pair_id] = record
    missing = [int(pair_id) for pair_id in pair_ids if int(pair_id) not in records]
    return records, missing


def pure_projected_cell(reference: dict[int, dict[str, Any]], pair_id: int) -> str:
    record = reference.get(int(pair_id))
    if record is None:
        return "missing"
    success = "Y" if bool(record.get("cem_late_success", False)) else "N"
    cost = fmt(record.get("cem_late_c_real_state"))
    return f"{cost}/{success}"


def print_summary_table(
    *,
    pair_specs: list[dict[str, Any]],
    pair_subset_map: dict[int, list[str]],
    default_records: list[dict[str, Any]],
    subspace_records: list[dict[str, Any]],
    pure_projected_reference: dict[int, dict[str, Any]],
) -> None:
    default_by_pair = {int(record["pair_id"]): record for record in default_records}
    subspace_by_pair = {int(record["pair_id"]): record for record in subspace_records}
    rows = []
    for pair_spec in pair_specs:
        pair_id = int(pair_spec["pair_id"])
        default = default_by_pair[pair_id]
        subspace = subspace_by_pair[pair_id]
        rows.append(
            [
                str(pair_id),
                str(pair_spec["cell"]),
                ",".join(pair_subset_map[pair_id]),
                f"{fmt(default['rank1_C_real_state'])}/{fmt(default['rank1_success'])}",
                f"{fmt(subspace['rank1_C_real_state'])}/{fmt(subspace['rank1_success'])}",
                fmt(subspace["selection_regret"]),
                fmt(subspace["action_l2_from_default"]),
                fmt(subspace["final_pool_full_cost_std"]),
                pure_projected_cell(pure_projected_reference, pair_id),
            ]
        )

    headers = [
        "Pair",
        "Cell",
        "Subsets",
        "Default C/succ",
        "Subspace C/succ",
        "Subspace regret",
        "Action L2",
        "Full cost std",
        "PureProj64 C/succ",
    ]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print("\nStage A Subspace-CEM smoke summary")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.stage1b_full = args.stage1b_full.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.device = resolve_device(args.device)

    pairs_data, requested_pairs = load_pairs(
        args.pairs_path,
        max_pairs=None,
        pair_ids=[int(pair_id) for pair_id in args.pairs],
    )
    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    if offset % ACTION_BLOCK != 0:
        raise ValueError("Track A offset must be divisible by action_block=5")
    validate_requested_pair_offsets(requested_pairs, offset=offset)

    dataset_path = Path(pair_metadata["dataset_path"])
    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
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

    pair_subset_map = {int(pair["pair_id"]): pair_subsets(pair) for pair in requested_pairs}
    pure_projected_reference, missing_pure_projected = load_pure_projected_reference(
        args.stage1b_full,
        pair_ids=tuple(int(pair_id) for pair_id in args.pairs),
        m=int(args.m),
        projection_seed=DEFAULT_PROJECTION_SEED,
    )

    # make_projection uses its own CPU generator seeded by projection_seed; the
    # CEM samplers below use independent model-device generators per pair.
    projection = make_projection(int(args.m), DEFAULT_PROJECTION_SEED)

    print("== Subspace-CEM Stage A smoke setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"dataset_name: {dataset_path.stem}")
    print(f"cache_dir: {dataset_path.parent}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"projection_m: {args.m}")
    print(f"projection_seed: {DEFAULT_PROJECTION_SEED}")
    print(f"cem_seed_rule: base_seed + pair_id * 1009")
    print(f"pairs: {list(args.pairs)}")
    for pair in requested_pairs:
        pair_id = int(pair["pair_id"])
        print(f"  pair_id={pair_id} cell={pair['cell']} subsets={','.join(pair_subset_map[pair_id])}")
    if missing_pure_projected:
        print(
            "WARNING: pure projected CEM m=64/seed=0 reference missing for "
            f"pair_ids={missing_pure_projected}; stdout table will show 'missing'."
        )

    default_records: list[dict[str, Any]] = []
    subspace_records: list[dict[str, Any]] = []
    total_started = time.time()
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair_spec in enumerate(requested_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            pair_started = time.time()
            initial, goal = load_pair_rows_direct(dataset, pair_spec)
            prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
            action_processor = policy.process["action"]
            initial_state = np.asarray(initial["state"], dtype=np.float32)
            goal_state = np.asarray(goal["state"], dtype=np.float32)

            print(
                f"\n[{pair_idx}/{len(requested_pairs)}] pair_id={pair_id} "
                f"cell={pair_spec['cell']} subsets={','.join(pair_subset_map[pair_id])}"
            )

            default_started = time.time()
            default_cem = run_cem(
                model=model,
                prepared_info=prepared_info,
                pair_id=pair_id,
                seed=int(args.seed),
                projection=None,
            )
            default_cem["final_pool_full_cost_std"] = clean_float(default_pool_full_cost_std(default_cem))
            print(f"  default CEM elapsed={seconds_to_hms(time.time() - default_started)}")

            default_scored = score_final_pool(
                env=env,
                initial_state=initial_state,
                goal_state=goal_state,
                blocked_candidates=np.asarray(default_cem["blocked_candidates"], dtype=np.float32),
                action_processor=action_processor,
                seed_base=int(args.seed) + pair_id * 100_000,
            )
            print(
                "  default simulator scoring: "
                f"300 candidates elapsed={seconds_to_hms(float(default_scored['simulator_scoring_seconds']))}"
            )

            default_rank1_blocked = np.asarray(default_cem["rank1_blocked"], dtype=np.float32)
            default_records.append(
                build_smoke_record(
                    pair_id=pair_id,
                    cem_result=default_cem,
                    scored=default_scored,
                    default_rank1_blocked=default_rank1_blocked,
                    action_l2_from_default=0.0,
                    elite_cost_std=default_cem["top30_select_cost_std"],
                    cost_dynamic_range=default_cem["select_cost_dynamic_range"],
                )
            )

            subspace_started = time.time()
            subspace_cem = run_subspace_cem(
                model=model,
                prepared_info=prepared_info,
                pair_id=pair_id,
                seed=int(args.seed),
                projection=projection,
            )
            print(f"  subspace CEM elapsed={seconds_to_hms(time.time() - subspace_started)}")

            subspace_scored = score_final_pool(
                env=env,
                initial_state=initial_state,
                goal_state=goal_state,
                blocked_candidates=np.asarray(subspace_cem["blocked_candidates"], dtype=np.float32),
                action_processor=action_processor,
                seed_base=int(args.seed) + pair_id * 100_000,
            )
            print(
                "  subspace simulator scoring: "
                f"300 candidates elapsed={seconds_to_hms(float(subspace_scored['simulator_scoring_seconds']))}"
            )

            subspace_l2 = float(np.linalg.norm(np.asarray(subspace_cem["rank1_blocked"], dtype=np.float32) - default_rank1_blocked))
            subspace_records.append(
                build_smoke_record(
                    pair_id=pair_id,
                    cem_result=subspace_cem,
                    scored=subspace_scored,
                    default_rank1_blocked=default_rank1_blocked,
                    action_l2_from_default=subspace_l2,
                    elite_cost_std=subspace_cem["top30_select_cost_std"],
                    cost_dynamic_range=subspace_cem["select_cost_dynamic_range"],
                )
            )

            print(f"  pair_id={pair_id} completed in {seconds_to_hms(time.time() - pair_started)}")
    finally:
        env.close()

    wall_clock = time.time() - total_started
    print_summary_table(
        pair_specs=requested_pairs,
        pair_subset_map=pair_subset_map,
        default_records=default_records,
        subspace_records=subspace_records,
        pure_projected_reference=pure_projected_reference,
    )

    output = {
        "config": {
            "m": int(args.m),
            "seed": int(args.seed),
            "pairs": [int(pair_id) for pair_id in args.pairs],
            "n_samples": int(NUM_SAMPLES),
            "n_iterations": int(CEM_ITERS),
            "n_elites": int(TOPK),
            "algorithm": ALGORITHM,
        },
        "subspace_cem": subspace_records,
        "default_cem": default_records,
        "wall_clock_seconds": clean_float(wall_clock),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print(f"\nSaved: {args.output}")
    print(f"Total wall clock: {seconds_to_hms(wall_clock)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

