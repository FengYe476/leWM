#!/usr/bin/env python3
"""Run D3 oracle-CEM cost-criterion ablation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lewm_audit.eval.oracle_cem import (
    HINGE_ALPHA,
    cost_v1_hinge,
    cost_v2_indicator,
    cost_v3_baseline,
    cem_with_oracle_cost,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
DEFAULT_TRACK_A_THREE_COST_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1" / "d3_oracle_ablation"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "phase1" / "d3_oracle_ablation_report.md"
DEFAULT_CELL_FILTER = "D3xR0,D3xR1,D3xR2,D3xR3"
DEFAULT_VARIANTS = "V3,V1,V2"
ACTION_SOURCE_ORDER = ("data", "smooth_random", "CEM_early", "CEM_late")
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
RANDOM_WAYPOINTS = 5
TRACK_A_REFERENCE_SUCCESS_RATE = {
    "D3xR0": 38 / 480,
    "D3xR1": 64 / 560,
    "D3xR2": 13 / 480,
    "D3xR3": 10 / 560,
}
VARIANT_COSTS = {
    "V1": {
        "cost_fn": cost_v1_hinge,
        "formula": (
            "max(block_pos_dist - 20.0, 0.0) + "
            "(20.0 / (pi / 9.0)) * max(angle_dist - pi / 9.0, 0.0)"
        ),
        "alpha": HINGE_ALPHA,
    },
    "V2": {
        "cost_fn": cost_v2_indicator,
        "formula": "0.0 if success else 1.0",
        "alpha": None,
    },
    "V3": {
        "cost_fn": cost_v3_baseline,
        "formula": "block_pos_dist + angle_dist",
        "alpha": None,
    },
}


def parse_csv(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def parse_action_counts(raw: str) -> dict[str, int]:
    chunks = parse_csv(raw)
    if len(chunks) != 4:
        raise argparse.ArgumentTypeError("--action-counts must contain four integers")
    values = [int(chunk) for chunk in chunks]
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("--action-counts values must be nonnegative")
    return dict(zip(ACTION_SOURCE_ORDER, values, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cell-filter", default=DEFAULT_CELL_FILTER)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-counts", type=parse_action_counts, default=parse_action_counts("20,20,20,20"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pairs-per-cell", type=int, default=None)
    parser.add_argument("--track-a-three-cost-path", type=Path, default=DEFAULT_TRACK_A_THREE_COST_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-prefix", default="d3_oracle")
    return parser.parse_args()


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


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def filter_pairs_by_cells(
    pairs: list[dict],
    cells: list[str],
    max_pairs_per_cell: int | None = None,
) -> list[dict]:
    allowed = set(cells)
    selected = [pair for pair in sorted(pairs, key=lambda item: int(item["pair_id"])) if pair["cell"] in allowed]
    if max_pairs_per_cell is None:
        return selected
    if max_pairs_per_cell < 1:
        raise ValueError("--max-pairs-per-cell must be positive when provided")
    counts = {cell: 0 for cell in cells}
    capped = []
    for pair in selected:
        cell = pair["cell"]
        if counts[cell] >= max_pairs_per_cell:
            continue
        capped.append(pair)
        counts[cell] += 1
    return capped


def load_pairs(path: Path, cells: list[str], max_pairs_per_cell: int | None) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    return data, filter_pairs_by_cells(data["pairs"], cells, max_pairs_per_cell)


def cem_config_json() -> dict:
    return {
        "samples_per_iter": NUM_SAMPLES,
        "iterations": CEM_ITERS,
        "elites": TOPK,
        "planning_horizon": PLANNING_HORIZON,
        "receding_horizon": RECEDING_HORIZON,
        "action_block": ACTION_BLOCK,
        "cem_early_iteration": CEM_EARLY_ITERS,
        "cem_late_iteration": CEM_LATE_ITERS,
    }


def output_path_for_variant(output_dir: Path, variant: str, output_prefix: str = "d3_oracle") -> Path:
    return output_dir / f"{output_prefix}_{variant}.json"


def build_variant_output(
    *,
    variant: str,
    pairs_path: Path,
    cells: list[str],
    n_pairs: int,
    seed: int,
    device: str,
    action_counts: dict[str, int],
    existing: dict | None,
) -> dict:
    metadata_update = {
        "variant": variant,
        "cost_formula": VARIANT_COSTS[variant]["formula"],
        "alpha": VARIANT_COSTS[variant]["alpha"],
        "planner_cost_source": "oracle_real_state",
        "cells_evaluated": cells,
        "n_pairs": n_pairs,
        "cem_config": cem_config_json(),
        "seed": seed,
        "device": device,
        "action_counts": action_counts,
        "pairs_path": str(pairs_path),
        "git_commit": get_git_commit(),
        "fixed_sequence_length_raw_steps": 50,
        "fixed_sequence_length_action_blocks": 10,
        "data_random_storage": "duplicated_per_variant_for_easy_aggregation",
    }
    if existing is not None:
        existing["metadata"].update(metadata_update)
        existing["metadata"]["timestamp_finished"] = None
        return existing
    return {
        "metadata": {
            **metadata_update,
            "timestamp_started": iso_now(),
            "timestamp_finished": None,
            "wallclock_seconds": None,
        },
        "pairs": [],
    }


def write_output(path: Path, output: dict, *, finished: bool = False) -> None:
    output["pairs"] = sorted(output["pairs"], key=lambda pair: int(pair["pair_id"]))
    output["metadata"]["n_pairs_completed"] = len(output["pairs"])
    output["metadata"]["timestamp_finished"] = iso_now() if finished else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(output), indent=2, allow_nan=False) + "\n")


def select_shared_sequences(
    *,
    dataset,
    valid_action_indices: np.ndarray,
    action_processor,
    pair_id: int,
    raw_steps: int,
    action_counts: dict[str, int],
    seed: int,
) -> list[dict]:
    from lewm_audit.diagnostics.three_cost import (
        sample_data_action_sequences,
        sample_random_action_sequences,
    )

    rng = np.random.default_rng(seed + pair_id * 1_000_003)
    sequences = []
    if action_counts["data"]:
        sequences.extend(
            sample_data_action_sequences(
                dataset,
                valid_action_indices,
                count=action_counts["data"],
                raw_steps=raw_steps,
                action_processor=action_processor,
                action_block=ACTION_BLOCK,
                rng=rng,
            )
        )
    if action_counts["smooth_random"]:
        sequences.extend(
            sample_random_action_sequences(
                count=action_counts["smooth_random"],
                raw_steps=raw_steps,
                waypoints=RANDOM_WAYPOINTS,
                action_processor=action_processor,
                action_block=ACTION_BLOCK,
                rng=rng,
            )
        )
    for sequence in sequences:
        if sequence["source"] == "data":
            sequence["source_label"] = "data"
        elif sequence["source"] == "random":
            sequence["source_label"] = "smooth_random"
        else:
            sequence["source_label"] = str(sequence["source"])
    return sequences


def oracle_cem_sequences(
    *,
    variant: str,
    env_factory,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    action_processor,
    pair_id: int,
    raw_steps: int,
    action_counts: dict[str, int],
    seed: int,
) -> list[dict]:
    from lewm_audit.diagnostics.three_cost import blocked_normalized_to_raw

    max_count = max(action_counts["CEM_early"], action_counts["CEM_late"])
    if max_count == 0:
        return []
    cost_fn = VARIANT_COSTS[variant]["cost_fn"]
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
        cost_fn,
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

    sequences = []
    for bucket, iter_idx, count in (
        ("CEM_early", CEM_EARLY_ITERS, action_counts["CEM_early"]),
        ("CEM_late", CEM_LATE_ITERS, action_counts["CEM_late"]),
    ):
        if count == 0:
            continue
        iter_offset = iter_idx - 1
        raw_candidates = cem["candidates_per_iter"][iter_offset]
        blocked_candidates = cem["blocked_candidates_per_iter"][iter_offset]
        elite_indices = cem["elite_indices_per_iter"][iter_offset][:count]
        elite_costs = cem["elite_costs_per_iter"][iter_offset][:count]
        for source_index, (candidate_idx, planner_cost) in enumerate(
            zip(elite_indices, elite_costs, strict=True)
        ):
            sequences.append(
                {
                    "source_label": f"{bucket}_{variant}",
                    "source_index": source_index,
                    "cem_iter": iter_idx,
                    "cem_rank": source_index,
                    "cem_oracle_cost": float(planner_cost),
                    "blocked_normalized": blocked_candidates[int(candidate_idx)].astype(np.float32),
                    "raw": raw_candidates[int(candidate_idx)].astype(np.float32),
                }
            )
    return sequences


def evaluate_pair_variant(
    *,
    pair_spec: dict,
    variant: str,
    dataset,
    valid_action_indices: np.ndarray,
    policy,
    model,
    eval_env,
    env_factory,
    action_processor,
    offset: int,
    action_counts: dict[str, int],
    seed: int,
) -> dict:
    from lewm_audit.diagnostics.three_cost import (
        block_pose_metrics,
        compute_model_costs,
        encode_pixels,
        execute_raw_actions,
        load_pair_rows,
        prepare_pair_info,
        squared_l2,
    )

    started = time.time()
    pair_id = int(pair_spec["pair_id"])
    pair_rows = load_pair_rows(dataset, int(pair_spec["start_row"]), offset)
    initial = pair_rows["initial"]
    goal = pair_rows["goal"]
    init_state = np.asarray(initial["state"], dtype=np.float32)
    goal_state = np.asarray(goal["state"], dtype=np.float32)
    raw_steps = int(pair_spec["goal_row"]) - int(pair_spec["start_row"])
    if raw_steps != 50:
        raise ValueError(f"Expected D3 ablation raw_steps=50, got {raw_steps}")

    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    goal_emb = encode_pixels(policy, model, goal["pixels"])
    sequences = select_shared_sequences(
        dataset=dataset,
        valid_action_indices=valid_action_indices,
        action_processor=action_processor,
        pair_id=pair_id,
        raw_steps=raw_steps,
        action_counts=action_counts,
        seed=seed,
    )
    sequences.extend(
        oracle_cem_sequences(
            variant=variant,
            env_factory=env_factory,
            init_state=init_state,
            goal_state=goal_state,
            action_processor=action_processor,
            pair_id=pair_id,
            raw_steps=raw_steps,
            action_counts=action_counts,
            seed=seed,
        )
    )
    expected = sum(action_counts.values())
    if len(sequences) != expected:
        raise RuntimeError(f"Pair {pair_id} produced {len(sequences)} actions, expected {expected}")

    blocked = np.stack([sequence["blocked_normalized"] for sequence in sequences])
    model_costs = compute_model_costs(model, prepared_info, blocked)
    variant_cost_fn = VARIANT_COSTS[variant]["cost_fn"]
    actions = []
    for action_id, (sequence, model_cost) in enumerate(zip(sequences, model_costs, strict=True)):
        rollout = execute_raw_actions(
            eval_env,
            initial_state=init_state,
            goal_state=goal_state,
            raw_actions=sequence["raw"],
            seed=seed + pair_id * 10_000 + action_id,
        )
        terminal_emb = encode_pixels(policy, model, rollout["terminal_pixels"])
        state_metrics = block_pose_metrics(rollout["terminal_state"], goal_state)
        record = {
            "action_id": action_id,
            "source": sequence["source_label"],
            "source_index": int(sequence["source_index"]),
            "C_real_z": squared_l2(terminal_emb, goal_emb),
            "C_model": float(model_cost),
            "C_real_state": float(state_metrics["c_real_state"]),
            "C_variant": float(variant_cost_fn(rollout["terminal_state"], goal_state)),
            "success": bool(state_metrics["success"]),
        }
        for optional in ("dataset_row", "cem_iter", "cem_rank", "cem_oracle_cost"):
            if optional in sequence:
                record[optional] = to_jsonable(sequence[optional])
        actions.append(record)

    return {
        "pair_id": pair_id,
        "cell": pair_spec["cell"],
        "episode_id": int(pair_spec["episode_id"]),
        "start_row": int(pair_spec["start_row"]),
        "goal_row": int(pair_spec["goal_row"]),
        "block_displacement_px": float(pair_spec["block_displacement_px"]),
        "required_rotation_rad": float(pair_spec["required_rotation_rad"]),
        "physical_pose_distance": float(pair_spec["physical_pose_distance"]),
        "wallclock_seconds": time.time() - started,
        "actions": actions,
    }


def evaluate_variant(
    *,
    variant: str,
    args: argparse.Namespace,
    pairs_path: Path,
    pairs: list[dict],
    cells: list[str],
    dataset,
    valid_action_indices: np.ndarray,
    policy,
    model,
    action_processor,
    offset: int,
) -> dict:
    import gymnasium as gym

    output_path = output_path_for_variant(args.output_dir, variant, args.output_prefix)
    existing = json.loads(output_path.read_text()) if args.resume and output_path.exists() else None
    output = build_variant_output(
        variant=variant,
        pairs_path=pairs_path,
        cells=cells,
        n_pairs=len(pairs),
        seed=args.seed,
        device=args.device,
        action_counts=args.action_counts,
        existing=existing,
    )
    completed = {int(pair["pair_id"]) for pair in output["pairs"]}
    started = time.time()

    def env_factory():
        return gym.make("swm/PushT-v1")

    eval_env = gym.make("swm/PushT-v1")
    try:
        for pair_spec in pairs:
            pair_id = int(pair_spec["pair_id"])
            if pair_id in completed:
                print(f"[{variant}] skipping completed pair_id={pair_id}")
                continue
            pair_started = time.time()
            print(f"[{variant}] pair_id={pair_id} cell={pair_spec['cell']}")
            result = evaluate_pair_variant(
                pair_spec=pair_spec,
                variant=variant,
                dataset=dataset,
                valid_action_indices=valid_action_indices,
                policy=policy,
                model=model,
                eval_env=eval_env,
                env_factory=env_factory,
                action_processor=action_processor,
                offset=offset,
                action_counts=args.action_counts,
                seed=args.seed,
            )
            output["pairs"].append(result)
            completed.add(pair_id)
            successes = sum(action["success"] for action in result["actions"])
            print(
                f"[{variant}] pair_successes={successes}/{len(result['actions'])}; "
                f"elapsed={time.time() - pair_started:.2f}s"
            )
            output["metadata"]["wallclock_seconds"] = time.time() - started
            write_output(output_path, output, finished=False)
    finally:
        eval_env.close()

    output["metadata"]["wallclock_seconds"] = time.time() - started
    write_output(output_path, output, finished=True)
    return output


def load_track_a_references(path: Path, cells: list[str]) -> dict:
    if not path.exists():
        return {
            cell: {
                "success_rate": TRACK_A_REFERENCE_SUCCESS_RATE[cell],
                "cem_late_success_rate": None,
            }
            for cell in cells
        }
    data = json.loads(path.read_text())
    refs = {}
    for cell in cells:
        pairs = [pair for pair in data["pairs"] if pair["cell"] == cell]
        actions = [action for pair in pairs for action in pair["actions"]]
        late = [action for action in actions if action["source"] == "CEM_late"]
        refs[cell] = {
            "success_rate": sum(bool(action["success"]) for action in actions) / len(actions),
            "cem_late_success_rate": (
                sum(bool(action["success"]) for action in late) / len(late) if late else None
            ),
        }
    return refs


def success_rate_for_actions(actions: list[dict]) -> float | None:
    if not actions:
        return None
    return float(sum(bool(action["success"]) for action in actions) / len(actions))


def infer_overall_label(cells: list[str]) -> str:
    rows = {cell.split("x")[0] for cell in cells}
    if len(rows) == 1:
        return f"{next(iter(rows))}_overall"
    return "overall"


def variant_success_tables(outputs: dict[str, dict], cells: list[str], overall_label: str | None = None) -> dict:
    if overall_label is None:
        overall_label = infer_overall_label(cells)
    tables = {}
    for variant, output in outputs.items():
        per_cell = {}
        all_actions = []
        all_late = []
        for cell in cells:
            cell_pairs = [pair for pair in output["pairs"] if pair["cell"] == cell]
            actions = [action for pair in cell_pairs for action in pair["actions"]]
            late = [action for action in actions if action["source"] == f"CEM_late_{variant}"]
            all_actions.extend(actions)
            all_late.extend(late)
            per_cell[cell] = {
                "success_rate": success_rate_for_actions(actions),
                "n_records": len(actions),
                "cem_late_success_rate": success_rate_for_actions(late),
                "cem_late_records": len(late),
            }
        per_cell[overall_label] = {
            "success_rate": success_rate_for_actions(all_actions),
            "n_records": len(all_actions),
            "cem_late_success_rate": success_rate_for_actions(all_late),
            "cem_late_records": len(all_late),
        }
        tables[variant] = per_cell
    return tables


def pearson_corr(x, y) -> float | None:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def mean_std(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def cost_shape_table(outputs: dict[str, dict]) -> dict:
    table = {}
    for variant, output in outputs.items():
        actions = [action for pair in output["pairs"] for action in pair["actions"]]
        best = [
            min(float(action["C_real_state"]) for action in pair["actions"])
            for pair in output["pairs"]
        ]
        c_real_state = [action["C_real_state"] for action in actions]
        c_variant = [action["C_variant"] for action in actions]
        success = [1.0 if action["success"] else 0.0 for action in actions]
        table[variant] = {
            "C_real_state": mean_std(c_real_state),
            "C_variant": mean_std(c_variant),
            "pearson_C_variant_vs_C_real_state": pearson_corr(c_variant, c_real_state),
            "pearson_C_variant_vs_success": pearson_corr(c_variant, success),
            "best_of_80_median_C_real_state_per_pair": float(np.median(best)) if best else None,
        }
    return table


def sanity_gate(v3_output: dict, refs: dict, cells: list[str], tolerance_pp: float = 5.0) -> dict:
    tables = variant_success_tables({"V3": v3_output}, cells)["V3"]
    per_cell = {}
    passed = True
    for cell in cells:
        rate = tables[cell]["success_rate"]
        ref = refs[cell]["success_rate"]
        delta_pp = 100.0 * (rate - ref)
        cell_pass = abs(delta_pp) < tolerance_pp
        passed = passed and cell_pass
        per_cell[cell] = {
            "track_a_reference_success_rate": ref,
            "v3_success_rate": rate,
            "delta_pp": delta_pp,
            "passed": cell_pass,
        }
    return {"passed": passed, "tolerance_pp": tolerance_pp, "per_cell": per_cell}


def relative_failure_reduction(latent_success: float, oracle_success: float) -> float | None:
    """Return (1 - oracle_success) / (1 - latent_success).

    When latent success is already 1.0, the denominator is zero. If oracle is
    also 1.0, there are no latent failures to reduce, so the value is undefined
    and represented as None. If oracle is worse than 1.0, return infinity.
    """
    latent_success = float(latent_success)
    oracle_success = float(oracle_success)
    denom = 1.0 - latent_success
    numerator = 1.0 - oracle_success
    if denom == 0.0:
        return None if numerator == 0.0 else math.inf
    return float(numerator / denom)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def write_report(
    *,
    outputs: dict[str, dict],
    refs: dict,
    gate: dict,
    success_tables: dict,
    shape_table: dict,
    cells: list[str],
    args: argparse.Namespace,
) -> None:
    report_path = args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# D3 Oracle Cost-Criterion Ablation Report")
    lines.append("")
    lines.append("## 1. Provenance")
    lines.append("")
    lines.append(f"- Pair source: `{args.pairs_path}`")
    lines.append(f"- Track A reference source: `{args.track_a_three_cost_path}`")
    lines.append(f"- Git commit: `{get_git_commit()}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(f"- Cells: `{', '.join(cells)}`")
    for variant, output in outputs.items():
        lines.append(
            f"- {variant} wall-clock: `{fmt(output['metadata'].get('wallclock_seconds'), 2)}` seconds"
        )
    lines.append("")
    lines.append("| Cell | Track A ref | V3 oracle | Delta pp | Gate |")
    lines.append("|---|---:|---:|---:|---|")
    for cell in cells:
        row = gate["per_cell"][cell]
        lines.append(
            f"| {cell} | {pct(row['track_a_reference_success_rate'])} | "
            f"{pct(row['v3_success_rate'])} | {row['delta_pp']:.2f} | "
            f"{'pass' if row['passed'] else 'fail'} |"
        )
    lines.append(f"\nV3 sanity gate verdict: `{'pass' if gate['passed'] else 'fail'}`.")
    lines.append("")

    lines.append("## 2. Variant Headline Table")
    lines.append("")
    present = [variant for variant in ("V3", "V1", "V2") if variant in success_tables]
    lines.append("| Cell | " + " | ".join(f"{variant} success" for variant in present) + " |")
    lines.append("|---" + "|---:" * len(present) + "|")
    overall_label = infer_overall_label(cells)
    for cell in [*cells, overall_label]:
        lines.append(
            f"| {cell} | "
            + " | ".join(pct(success_tables[variant][cell]["success_rate"]) for variant in present)
            + " |"
        )
    lines.append("")
    lines.append("| Cell | " + " | ".join(f"{variant} CEM_late success" for variant in present) + " |")
    lines.append("|---" + "|---:" * len(present) + "|")
    for cell in [*cells, overall_label]:
        lines.append(
            f"| {cell} | "
            + " | ".join(pct(success_tables[variant][cell]["cem_late_success_rate"]) for variant in present)
            + " |"
        )
    lines.append("")

    lines.append("## 3. Cost-Shape Interpretation Table")
    lines.append("")
    lines.append("| Variant | C_real_state mean/std | C_variant mean/std | Pearson C_variant/C_real_state | Pearson C_variant/success | Median best C_real_state |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in present:
        row = shape_table[variant]
        lines.append(
            f"| {variant} | {fmt(row['C_real_state']['mean'])} / {fmt(row['C_real_state']['std'])} | "
            f"{fmt(row['C_variant']['mean'])} / {fmt(row['C_variant']['std'])} | "
            f"{fmt(row['pearson_C_variant_vs_C_real_state'])} | "
            f"{fmt(row['pearson_C_variant_vs_success'])} | "
            f"{fmt(row['best_of_80_median_C_real_state_per_pair'])} |"
        )
    lines.append("")

    lines.append("## 4. Headline Finding")
    lines.append("")
    if all(variant in success_tables for variant in ("V3", "V1", "V2")):
        lines.append("| Cell | V1 - V3 pp | V2 - V3 pp | V2 - V1 pp |")
        lines.append("|---|---:|---:|---:|")
        for cell in [*cells, overall_label]:
            v3 = success_tables["V3"][cell]["success_rate"]
            v1 = success_tables["V1"][cell]["success_rate"]
            v2 = success_tables["V2"][cell]["success_rate"]
            lines.append(
                f"| {cell} | {100.0 * (v1 - v3):.2f} | {100.0 * (v2 - v3):.2f} | "
                f"{100.0 * (v2 - v1):.2f} |"
            )
    else:
        lines.append("V1/V2 were not run because the V3 sanity gate did not pass.")
    lines.append("")

    lines.append("## 5. Limitations")
    lines.append("")
    lines.append("- This ablation only tests D3 row; D0/D1/D2 may behave differently.")
    lines.append("- Oracle CEM has access to ground-truth state, which the deployed system does not; V1/V2 are upper bounds.")
    lines.append("- alpha = 20 / (pi/9) is one specific choice; results may be sensitive to alpha.")
    lines.append("- V2 indicator cost has zero gradient inside the success region, so CEM may behave qualitatively differently.")
    lines.append("- Data and smooth_random records are duplicated per variant for easy per-variant aggregation.")
    lines.append("")
    report_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.track_a_three_cost_path = args.track_a_three_cost_path.expanduser().resolve()
    args.report_path = args.report_path.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = parse_csv(args.cell_filter)
    variants = parse_csv(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANT_COSTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if args.action_counts["CEM_early"] > TOPK or args.action_counts["CEM_late"] > TOPK:
        raise ValueError("CEM action counts must be <= TOPK=30")
    if "V3" not in variants:
        raise ValueError("V3 must be included so the sanity gate can run first")
    variants = ["V3", *[variant for variant in variants if variant != "V3"]]

    import gymnasium as gym  # noqa: F401
    import stable_worldmodel as swm  # noqa: F401
    from eval_pusht_baseline import (
        DEFAULT_CHECKPOINT_DIR,
        build_policy,
        build_processors,
        get_dataset,
        resolve_device,
    )
    from lewm_audit.eval.pusht import analyze_offset, prepare_dataset_index

    args.device = resolve_device(args.device)
    pairs_data, pairs = load_pairs(args.pairs_path, cells, args.max_pairs_per_cell)
    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    if offset != 50:
        raise ValueError(f"D3 oracle ablation expects offset=50, got {offset}")
    dataset_path = Path(pair_metadata["dataset_path"])
    cache_dir = dataset_path.parent
    dataset_name = dataset_path.stem

    print("== D3 oracle ablation setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output_dir: {args.output_dir}")
    print(f"cells: {cells}")
    print(f"variants: {variants}")
    print(f"n_pairs: {len(pairs)}")
    print(f"device: {args.device}")
    print(f"action_counts: {args.action_counts}")

    dataset = get_dataset(cache_dir, dataset_name)
    index = prepare_dataset_index(dataset)
    analysis = analyze_offset(index, offset)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy_args = argparse.Namespace(
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR.expanduser().resolve(),
        device=args.device,
        num_samples=NUM_SAMPLES,
        var_scale=VAR_SCALE,
        cem_iters=CEM_ITERS,
        topk=TOPK,
        seed=args.seed,
        horizon=PLANNING_HORIZON,
        receding_horizon=RECEDING_HORIZON,
        action_block=ACTION_BLOCK,
        img_size=IMG_SIZE,
    )
    policy = build_policy(policy_args, process)
    model = policy.solver.model
    action_processor = policy.process["action"]
    refs = load_track_a_references(args.track_a_three_cost_path, cells)
    outputs = {}

    v3_output = evaluate_variant(
        variant="V3",
        args=args,
        pairs_path=args.pairs_path,
        pairs=pairs,
        cells=cells,
        dataset=dataset,
        valid_action_indices=analysis["valid_indices"],
        policy=policy,
        model=model,
        action_processor=action_processor,
        offset=offset,
    )
    outputs["V3"] = v3_output
    gate = sanity_gate(v3_output, refs, cells)
    success_tables = variant_success_tables(outputs, cells)
    shape_table = cost_shape_table(outputs)
    if not gate["passed"]:
        write_report(
            outputs=outputs,
            refs=refs,
            gate=gate,
            success_tables=success_tables,
            shape_table=shape_table,
            cells=cells,
            args=args,
        )
        print(json.dumps({"v3_sanity_gate": gate, "report": str(args.report_path)}, indent=2))
        return 2

    for variant in variants:
        if variant == "V3":
            continue
        outputs[variant] = evaluate_variant(
            variant=variant,
            args=args,
            pairs_path=args.pairs_path,
            pairs=pairs,
            cells=cells,
            dataset=dataset,
            valid_action_indices=analysis["valid_indices"],
            policy=policy,
            model=model,
            action_processor=action_processor,
            offset=offset,
        )

    success_tables = variant_success_tables(outputs, cells)
    shape_table = cost_shape_table(outputs)
    write_report(
        outputs=outputs,
        refs=refs,
        gate=gate,
        success_tables=success_tables,
        shape_table=shape_table,
        cells=cells,
        args=args,
    )
    print(
        json.dumps(
            {
                "v3_sanity_gate": gate,
                "success_tables": success_tables,
                "cost_shape": shape_table,
                "report": str(args.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
