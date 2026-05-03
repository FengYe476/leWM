#!/usr/bin/env python3
"""PushT Stage A Subspace-CEM runner.

Runs m=64 Subspace-CEM on the locked 30-pair Stage A subset and compares it
against default CEM plus available pure projected-CEM m=64 records.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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
    CEM_ITERS,
    DEFAULT_PAIRS_PATH,
    NUM_SAMPLES,
    TOPK,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.projected_cem import make_projection, run_cem  # noqa: E402
from scripts.phase2.stage1.stage1a_controls import clean_float, jsonable  # noqa: E402
from scripts.phase2.subspace_cem.subspace_cem import (  # noqa: E402
    ALGORITHM,
    DEFAULT_STAGE1B_FULL,
    blocked_batch_to_raw_fast,  # noqa: F401 - documents that action conversion path is inherited.
    build_smoke_record,
    default_pool_full_cost_std,
    load_pair_rows_direct,
    run_subspace_cem,
    score_final_pool,
    seconds_to_hms,
)


DEFAULT_RERANK_PATH = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "subspace_cem" / "stage_a_pusht.json"
STAGE_A_M = 64
STAGE_A_PROJECTION_SEEDS = (0, 1)
BASE_CEM_SEED = 0
PAIR_SELECTION_COUNTS = {
    "invisible_quadrant": 8,
    "ordinary": 12,
    "latent_favorable": 5,
    "v1_favorable": 5,
}
SUBSET_ORDER = ("invisible_quadrant", "ordinary", "latent_favorable", "v1_favorable")
ALL_MEMBERSHIP_ORDER = (
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
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--rerank-path", type=Path, default=DEFAULT_RERANK_PATH)
    parser.add_argument("--stage1b-full", type=Path, default=DEFAULT_STAGE1B_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m", type=int, default=STAGE_A_M)
    parser.add_argument("--projection-seeds", type=parse_int_list, default=STAGE_A_PROJECTION_SEEDS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--list-pairs", action="store_true", help="Print the derived Stage A pairs and exit.")
    args = parser.parse_args()
    if int(args.m) != STAGE_A_M:
        parser.error(f"--m is locked to {STAGE_A_M} for Stage A")
    if tuple(int(seed) for seed in args.projection_seeds) != STAGE_A_PROJECTION_SEEDS:
        parser.error(f"--projection-seeds is locked to {STAGE_A_PROJECTION_SEEDS}")
    return args


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


def fmt(value: float | int | bool | None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "Y" if value else "N"
    value = float(value)
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def membership_map(anchor_definitions: dict[str, Any]) -> dict[int, list[str]]:
    memberships: dict[int, list[str]] = {}
    for name in ALL_MEMBERSHIP_ORDER:
        for pair_id in anchor_definitions.get(name, {}).get("pair_ids", []):
            memberships.setdefault(int(pair_id), []).append(name)
    return memberships


def select_stage_a_pairs(
    *,
    pairs_path: Path,
    rerank_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, list[str]], dict[str, Any]]:
    rerank = load_json(rerank_path)
    anchor_definitions = rerank.get("metadata", {}).get("anchor_definitions")
    if not isinstance(anchor_definitions, dict):
        raise RuntimeError(f"Missing anchor_definitions in {rerank_path}")

    pairs_data, all_pairs = load_pairs(pairs_path, max_pairs=None, pair_ids=None)
    by_id = {int(pair["pair_id"]): pair for pair in all_pairs}
    memberships = membership_map(anchor_definitions)

    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for subset in SUBSET_ORDER:
        pair_ids = [int(pair_id) for pair_id in anchor_definitions[subset]["pair_ids"]]
        chosen: list[int] = []
        for pair_id in pair_ids:
            if pair_id in used:
                continue
            chosen.append(pair_id)
            if len(chosen) == int(PAIR_SELECTION_COUNTS[subset]):
                break
        if len(chosen) != int(PAIR_SELECTION_COUNTS[subset]):
            raise RuntimeError(
                f"Could not select {PAIR_SELECTION_COUNTS[subset]} pairs for {subset}; got {len(chosen)}"
            )
        for pair_id in chosen:
            if pair_id not in by_id:
                raise RuntimeError(f"Selected pair_id={pair_id} missing from {pairs_path}")
            used.add(pair_id)
            pair = dict(by_id[pair_id])
            pair["primary_subset"] = subset
            pair["subset_memberships"] = memberships.get(pair_id, [])
            selected.append(pair)

    if len(selected) != 30 or len({int(pair["pair_id"]) for pair in selected}) != 30:
        raise RuntimeError("Stage A pair selection must contain 30 unique pairs")
    return pairs_data, selected, memberships, anchor_definitions


def print_selected_pairs(selected_pairs: list[dict[str, Any]]) -> None:
    print("Selected Stage A PushT pairs")
    print("Pair | Cell  | Primary subset     | Memberships")
    print("-----+-------+--------------------+-------------------------------")
    for pair in selected_pairs:
        print(
            f"{int(pair['pair_id']):<4} | {str(pair['cell']):<5} | "
            f"{str(pair['primary_subset']):<18} | {','.join(pair['subset_memberships'])}"
        )


def load_existing_output(path: Path, *, resume: bool) -> dict[str, Any] | None:
    if not resume or not path.exists():
        return None
    data = load_json(path)
    if data.get("config", {}).get("algorithm") != ALGORITHM:
        raise RuntimeError(f"Cannot resume from unexpected algorithm: {data.get('config', {}).get('algorithm')!r}")
    return data


def is_complete_default_record(record: dict[str, Any]) -> bool:
    return "rank1_blocked_action" in record and "pair_id" in record


def decorate_record(
    record: dict[str, Any],
    *,
    pair_spec: dict[str, Any],
    projection_seed: int | None = None,
    rank1_blocked_action: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    out = dict(record)
    out["primary_subset"] = str(pair_spec["primary_subset"])
    out["subset_memberships"] = list(pair_spec["subset_memberships"])
    if projection_seed is not None:
        out["seed"] = int(projection_seed)
        out["projection_seed"] = int(projection_seed)
    if rank1_blocked_action is not None:
        out["rank1_blocked_action"] = np.asarray(rank1_blocked_action, dtype=np.float32)
    return out


def default_rank1_blocked(default_record: dict[str, Any]) -> np.ndarray:
    if "rank1_blocked_action" not in default_record:
        raise RuntimeError(f"Default record for pair_id={default_record.get('pair_id')} lacks rank1_blocked_action")
    return np.asarray(default_record["rank1_blocked_action"], dtype=np.float32)


def load_pure_projected_records(
    path: Path,
    *,
    selected_pairs: list[dict[str, Any]],
    m: int,
    projection_seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    data = load_json(path) if path.exists() else {"records": []}
    pair_by_id = {int(pair["pair_id"]): pair for pair in selected_pairs}
    stage_records = {
        (int(record.get("pair_id", -1)), int(record.get("projection_seed", -1))): record
        for record in data.get("records", [])
        if int(record.get("dimension", -1)) == int(m)
    }
    records: list[dict[str, Any]] = []
    missing: list[dict[str, int]] = []
    for pair in selected_pairs:
        pair_id = int(pair["pair_id"])
        for projection_seed in projection_seeds:
            source = stage_records.get((pair_id, int(projection_seed)))
            if source is None:
                missing.append({"pair_id": pair_id, "m": int(m), "seed": int(projection_seed)})
                continue
            records.append(
                {
                    "pair_id": pair_id,
                    "cell": str(pair["cell"]),
                    "primary_subset": str(pair_by_id[pair_id]["primary_subset"]),
                    "subset_memberships": list(pair_by_id[pair_id]["subset_memberships"]),
                    "m": int(m),
                    "seed": int(projection_seed),
                    "projection_seed": int(projection_seed),
                    "rank1_success": bool(source.get("rank1_success", source.get("cem_late_success"))),
                    "rank1_C_real_state": clean_float(
                        source.get("rank1_c_real_state", source.get("cem_late_c_real_state"))
                    ),
                    "rank1_c_real_state": clean_float(
                        source.get("rank1_c_real_state", source.get("cem_late_c_real_state"))
                    ),
                    "source": "stage1b_full",
                }
            )
    return records, missing


def scalar_summary(values: list[float | int | bool | None]) -> dict[str, Any]:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "mean": clean_float(float(arr.mean())) if len(arr) else None,
        "std": clean_float(float(arr.std(ddof=1))) if len(arr) > 1 else None,
        "n": int(len(arr)),
    }


def record_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "success": scalar_summary([record.get("rank1_success") for record in records]),
        "rank1_C_real_state": scalar_summary([record.get("rank1_C_real_state") for record in records]),
        "selection_regret": scalar_summary([record.get("selection_regret") for record in records]),
        "elite_cost_std": scalar_summary([record.get("elite_cost_std") for record in records]),
    }


def build_summary(
    *,
    selected_pairs: list[dict[str, Any]],
    subspace_records: list[dict[str, Any]],
    default_records: list[dict[str, Any]],
    pure_projected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "overall": {
            "subspace": record_group_summary(subspace_records),
            "default": record_group_summary(default_records),
            "pure_projected": record_group_summary(pure_projected_records),
            "subspace_success": record_group_summary(subspace_records)["success"]["mean"],
            "default_success": record_group_summary(default_records)["success"]["mean"],
            "pure_projected_success": record_group_summary(pure_projected_records)["success"]["mean"],
        },
        "by_subset": {},
    }
    for subset in SUBSET_ORDER:
        pair_ids = {int(pair["pair_id"]) for pair in selected_pairs if str(pair["primary_subset"]) == subset}
        subset_subspace = [record for record in subspace_records if int(record["pair_id"]) in pair_ids]
        subset_default = [record for record in default_records if int(record["pair_id"]) in pair_ids]
        subset_pure = [record for record in pure_projected_records if int(record["pair_id"]) in pair_ids]
        subset_summary = {
            "subspace": record_group_summary(subset_subspace),
            "default": record_group_summary(subset_default),
            "pure_projected": record_group_summary(subset_pure),
        }
        subset_summary["subspace_success"] = subset_summary["subspace"]["success"]["mean"]
        subset_summary["default_success"] = subset_summary["default"]["success"]["mean"]
        subset_summary["pure_projected_success"] = subset_summary["pure_projected"]["success"]["mean"]
        summary["by_subset"][subset] = subset_summary
    return summary


def success_or_floor(value: float | None) -> float:
    return float(value) if value is not None else float("-inf")


def build_decision(summary: dict[str, Any], *, completed: bool) -> dict[str, Any]:
    overall = summary["overall"]
    subspace_success = overall["subspace_success"]
    default_success = overall["default_success"]
    pure_success = overall["pure_projected_success"]
    max_baseline = max(success_or_floor(default_success), success_or_floor(pure_success))
    beats_by = None if subspace_success is None or not math.isfinite(max_baseline) else clean_float(subspace_success - max_baseline)

    subset_regressions: dict[str, float | None] = {}
    any_subset_regression = False
    for subset, subset_summary in summary["by_subset"].items():
        subset_subspace = subset_summary["subspace_success"]
        subset_baseline = max(
            success_or_floor(subset_summary["default_success"]),
            success_or_floor(subset_summary["pure_projected_success"]),
        )
        regression = None if subset_subspace is None or not math.isfinite(subset_baseline) else clean_float(
            subset_baseline - subset_subspace
        )
        subset_regressions[subset] = regression
        if regression is not None and float(regression) > 0.05:
            any_subset_regression = True

    subspace_regret = overall["subspace"]["selection_regret"]["mean"]
    default_regret = overall["default"]["selection_regret"]["mean"]
    regret_ok = None if subspace_regret is None or default_regret is None else bool(float(subspace_regret) <= float(default_regret))

    subspace_std = overall["subspace"]["elite_cost_std"]["mean"]
    default_std = overall["default"]["elite_cost_std"]["mean"]
    std_ratio = None if subspace_std is None or default_std in (None, 0) else clean_float(float(subspace_std) / float(default_std))
    std_ok = None if std_ratio is None else bool(float(std_ratio) >= 0.5)

    success_ok = None if beats_by is None else bool(float(beats_by) >= 0.03)
    subset_ok = None if not completed else bool(not any_subset_regression)
    pass_bits = [success_ok, subset_ok, regret_ok, std_ok]
    verdict = None if not completed or any(bit is None for bit in pass_bits) else ("PASS" if all(pass_bits) else "FAIL")
    return {
        "subspace_beats_max_baseline_by": beats_by,
        "max_baseline_success": clean_float(max_baseline) if math.isfinite(max_baseline) else None,
        "any_subset_regression_over_5pp": bool(any_subset_regression),
        "subset_regressions": subset_regressions,
        "selection_regret_nonincrease": regret_ok,
        "subspace_selection_regret_mean": subspace_regret,
        "default_selection_regret_mean": default_regret,
        "elite_std_not_collapsed": std_ok,
        "elite_std_ratio_subspace_over_default": std_ratio,
        "completed": bool(completed),
        "verdict": verdict,
    }


def write_output(
    *,
    path: Path,
    selected_pairs: list[dict[str, Any]],
    pure_projected_records: list[dict[str, Any]],
    missing_pure_projected: list[dict[str, int]],
    default_records: list[dict[str, Any]],
    subspace_records: list[dict[str, Any]],
    total_started: float,
) -> None:
    default_records = sorted(default_records, key=lambda record: int(record["pair_id"]))
    subspace_records = sorted(subspace_records, key=lambda record: (int(record["pair_id"]), int(record["seed"])))
    summary = build_summary(
        selected_pairs=selected_pairs,
        subspace_records=subspace_records,
        default_records=default_records,
        pure_projected_records=pure_projected_records,
    )
    completed = len(default_records) == len(selected_pairs) and len(subspace_records) == len(selected_pairs) * len(
        STAGE_A_PROJECTION_SEEDS
    )
    output = {
        "config": {
            "m": STAGE_A_M,
            "seeds": [int(seed) for seed in STAGE_A_PROJECTION_SEEDS],
            "base_cem_seed": BASE_CEM_SEED,
            "n_pairs": int(len(selected_pairs)),
            "pair_selection": dict(PAIR_SELECTION_COUNTS),
            "algorithm": ALGORITHM,
            "missing_baseline_policy": "continue_partial",
            "missing_pure_projected_keys": missing_pure_projected,
            "git_commit": get_git_commit(),
        },
        "selected_pairs": [
            {
                "pair_id": int(pair["pair_id"]),
                "cell": str(pair["cell"]),
                "primary_subset": str(pair["primary_subset"]),
                "subset_memberships": list(pair["subset_memberships"]),
            }
            for pair in selected_pairs
        ],
        "subspace_cem": subspace_records,
        "default_cem": default_records,
        "pure_projected_m64": pure_projected_records,
        "summary": summary,
        "stage_a_decision": build_decision(summary, completed=completed),
        "wall_clock_seconds": clean_float(time.time() - total_started),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")


def print_final_summary(summary: dict[str, Any], decision: dict[str, Any]) -> None:
    print("\nStage A PushT summary")
    print(
        "overall: "
        f"subspace={fmt(summary['overall']['subspace_success'])} "
        f"default={fmt(summary['overall']['default_success'])} "
        f"pure_projected={fmt(summary['overall']['pure_projected_success'])}"
    )
    print("by primary subset:")
    for subset in SUBSET_ORDER:
        row = summary["by_subset"][subset]
        print(
            f"  {subset}: subspace={fmt(row['subspace_success'])} "
            f"default={fmt(row['default_success'])} pure_projected={fmt(row['pure_projected_success'])}"
        )
    print(
        "decision: "
        f"beats_by={fmt(decision['subspace_beats_max_baseline_by'])} "
        f"regression_over_5pp={decision['any_subset_regression_over_5pp']} "
        f"regret_ok={decision['selection_regret_nonincrease']} "
        f"elite_std_ok={decision['elite_std_not_collapsed']} "
        f"verdict={decision['verdict']}"
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.rerank_path = args.rerank_path.expanduser().resolve()
    args.stage1b_full = args.stage1b_full.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    torch.manual_seed(BASE_CEM_SEED)
    np.random.seed(BASE_CEM_SEED)

    pairs_data, selected_pairs, _, _ = select_stage_a_pairs(
        pairs_path=args.pairs_path,
        rerank_path=args.rerank_path,
    )
    print_selected_pairs(selected_pairs)
    if args.list_pairs:
        return 0

    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    validate_requested_pair_offsets(selected_pairs, offset=offset)
    dataset_path = Path(pair_metadata["dataset_path"])

    pure_projected_records, missing_pure_projected = load_pure_projected_records(
        args.stage1b_full,
        selected_pairs=selected_pairs,
        m=STAGE_A_M,
        projection_seeds=STAGE_A_PROJECTION_SEEDS,
    )

    existing = load_existing_output(args.output, resume=not args.no_resume)
    default_records: list[dict[str, Any]] = []
    subspace_records: list[dict[str, Any]] = []
    if existing is not None:
        selected_ids = {int(pair["pair_id"]) for pair in selected_pairs}
        default_records = [
            record
            for record in existing.get("default_cem", [])
            if int(record.get("pair_id", -1)) in selected_ids and is_complete_default_record(record)
        ]
        expected_subspace = {
            (int(pair["pair_id"]), int(seed))
            for pair in selected_pairs
            for seed in STAGE_A_PROJECTION_SEEDS
        }
        subspace_records = [
            record
            for record in existing.get("subspace_cem", [])
            if (int(record.get("pair_id", -1)), int(record.get("seed", -1))) in expected_subspace
        ]
    default_by_pair = {int(record["pair_id"]): record for record in default_records}
    subspace_seen = {(int(record["pair_id"]), int(record["seed"])) for record in subspace_records}

    print("\n== PushT Subspace-CEM Stage A setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"rerank_path: {args.rerank_path}")
    print(f"stage1b_full: {args.stage1b_full}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"device: {args.device}")
    print(f"m: {STAGE_A_M}")
    print(f"projection_seeds: {list(STAGE_A_PROJECTION_SEEDS)}")
    print(f"base_cem_seed: {BASE_CEM_SEED}")
    print(f"resume_default_records: {len(default_by_pair)}")
    print(f"resume_subspace_records: {len(subspace_seen)}")
    print(f"pure_projected_records_available: {len(pure_projected_records)}")
    print(f"missing_pure_projected_keys: {missing_pure_projected}")
    print("runtime_estimate: 90 CEM runs total; expected roughly 10-15 minutes on CUDA/5090")

    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=BASE_CEM_SEED,
        ),
        process,
    )
    model = policy.solver.model
    action_processor = policy.process["action"]
    projections = {int(seed): make_projection(STAGE_A_M, int(seed)) for seed in STAGE_A_PROJECTION_SEEDS}

    total_started = time.time()
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair_spec in enumerate(selected_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            initial, goal = load_pair_rows_direct(dataset, pair_spec)
            prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
            initial_state = np.asarray(initial["state"], dtype=np.float32)
            goal_state = np.asarray(goal["state"], dtype=np.float32)

            if pair_id not in default_by_pair:
                started = time.time()
                default_cem = run_cem(
                    model=model,
                    prepared_info=prepared_info,
                    pair_id=pair_id,
                    seed=BASE_CEM_SEED,
                    projection=None,
                )
                default_cem["final_pool_full_cost_std"] = clean_float(default_pool_full_cost_std(default_cem))
                default_scored = score_final_pool(
                    env=env,
                    initial_state=initial_state,
                    goal_state=goal_state,
                    blocked_candidates=np.asarray(default_cem["blocked_candidates"], dtype=np.float32),
                    action_processor=action_processor,
                    seed_base=BASE_CEM_SEED + pair_id * 100_000,
                )
                default_record = build_smoke_record(
                    pair_id=pair_id,
                    cem_result=default_cem,
                    scored=default_scored,
                    default_rank1_blocked=np.asarray(default_cem["rank1_blocked"], dtype=np.float32),
                    action_l2_from_default=0.0,
                    elite_cost_std=default_cem["top30_select_cost_std"],
                    cost_dynamic_range=default_cem["select_cost_dynamic_range"],
                )
                default_record = decorate_record(
                    default_record,
                    pair_spec=pair_spec,
                    rank1_blocked_action=np.asarray(default_cem["rank1_blocked"], dtype=np.float32),
                )
                default_records.append(default_record)
                default_by_pair[pair_id] = default_record
                write_output(
                    path=args.output,
                    selected_pairs=selected_pairs,
                    pure_projected_records=pure_projected_records,
                    missing_pure_projected=missing_pure_projected,
                    default_records=default_records,
                    subspace_records=subspace_records,
                    total_started=total_started,
                )
                print(
                    f"Default {pair_idx}/{len(selected_pairs)} pair_id={pair_id} "
                    f"success={default_record['rank1_success']} "
                    f"C_real_state={default_record['rank1_C_real_state']:.4f} "
                    f"elapsed={seconds_to_hms(time.time() - started)}"
                )
            else:
                print(f"Default {pair_idx}/{len(selected_pairs)} pair_id={pair_id} - loaded existing record")

            default_blocked = default_rank1_blocked(default_by_pair[pair_id])
            for projection_seed in STAGE_A_PROJECTION_SEEDS:
                key = (pair_id, int(projection_seed))
                if key in subspace_seen:
                    print(
                        f"Subspace {pair_idx}/{len(selected_pairs)} pair_id={pair_id} seed={projection_seed} "
                        "- loaded existing record"
                    )
                    continue
                started = time.time()
                subspace_cem = run_subspace_cem(
                    model=model,
                    prepared_info=prepared_info,
                    pair_id=pair_id,
                    seed=BASE_CEM_SEED,
                    projection=projections[int(projection_seed)],
                )
                subspace_scored = score_final_pool(
                    env=env,
                    initial_state=initial_state,
                    goal_state=goal_state,
                    blocked_candidates=np.asarray(subspace_cem["blocked_candidates"], dtype=np.float32),
                    action_processor=action_processor,
                    seed_base=BASE_CEM_SEED + pair_id * 100_000,
                )
                action_l2 = float(np.linalg.norm(np.asarray(subspace_cem["rank1_blocked"], dtype=np.float32) - default_blocked))
                subspace_record = build_smoke_record(
                    pair_id=pair_id,
                    cem_result=subspace_cem,
                    scored=subspace_scored,
                    default_rank1_blocked=default_blocked,
                    action_l2_from_default=action_l2,
                    elite_cost_std=subspace_cem["top30_select_cost_std"],
                    cost_dynamic_range=subspace_cem["select_cost_dynamic_range"],
                )
                subspace_record = decorate_record(
                    subspace_record,
                    pair_spec=pair_spec,
                    projection_seed=int(projection_seed),
                )
                subspace_records.append(subspace_record)
                subspace_seen.add(key)
                write_output(
                    path=args.output,
                    selected_pairs=selected_pairs,
                    pure_projected_records=pure_projected_records,
                    missing_pure_projected=missing_pure_projected,
                    default_records=default_records,
                    subspace_records=subspace_records,
                    total_started=total_started,
                )
                print(
                    f"Subspace {pair_idx}/{len(selected_pairs)} pair_id={pair_id} seed={projection_seed} "
                    f"success={subspace_record['rank1_success']} "
                    f"C_real_state={subspace_record['rank1_C_real_state']:.4f} "
                    f"elapsed={seconds_to_hms(time.time() - started)}"
                )
    finally:
        env.close()

    write_output(
        path=args.output,
        selected_pairs=selected_pairs,
        pure_projected_records=pure_projected_records,
        missing_pure_projected=missing_pure_projected,
        default_records=default_records,
        subspace_records=subspace_records,
        total_started=total_started,
    )
    output = load_json(args.output)
    print_final_summary(output["summary"], output["stage_a_decision"])
    print(f"\nSaved: {args.output}")
    print(f"Wall clock: {seconds_to_hms(output['wall_clock_seconds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

