#!/usr/bin/env python3
"""Run a single oracle-CEM cost variant while storing only CEM actions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase1.eval_d3_oracle_ablation import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    CEM_LATE_ITERS,
    CEM_EARLY_ITERS,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    RECEDING_HORIZON,
    TOPK,
    VARIANT_COSTS,
    VAR_SCALE,
    build_variant_output,
    evaluate_pair_variant,
    load_pairs,
    parse_action_counts,
    parse_csv,
    to_jsonable,
    write_output,
)


DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
CEM_ONLY_ACTION_COUNTS = parse_action_counts("0,0,20,20")


def display_path(path: Path) -> str:
    path = path.expanduser().resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def row_from_cells(cells: list[str]) -> str:
    rows = {cell.split("x", maxsplit=1)[0] for cell in cells}
    if len(rows) != 1:
        raise ValueError(f"CEM-only row runs expect cells from one D row, got {cells}")
    return next(iter(rows))


def default_output_path(variant: str, row: str) -> Path:
    variant_lower = variant.lower()
    row_lower = row.lower()
    return PROJECT_ROOT / "results" / "phase1" / f"{variant_lower}_oracle_ablation" / f"{variant_lower}_{row_lower}.json"


def default_v3_source_path(row: str) -> Path:
    row_lower = row.lower()
    return PROJECT_ROOT / "results" / "phase1" / f"{row_lower}_oracle_ablation" / f"{row_lower}_oracle_V3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cell-filter", required=True)
    parser.add_argument("--variant", required=True, choices=("V1", "V2"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--data-smooth-random-source", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pairs-per-cell", type=int, default=None)
    return parser.parse_args()


def apply_cem_only_metadata(output: dict, *, data_smooth_random_source: Path) -> None:
    output["metadata"].update(
        {
            "actions_subset": "cem_only",
            "data_smooth_random_source": display_path(data_smooth_random_source),
            "data_random_storage": "omitted_for_cem_only_outputs",
        }
    )


def validate_cem_only_output(output: dict, *, variant: str) -> None:
    expected_sources = {f"CEM_early_{variant}", f"CEM_late_{variant}"}
    for pair in output["pairs"]:
        actions = pair["actions"]
        sources = {action["source"] for action in actions}
        if sources - expected_sources:
            raise RuntimeError(
                f"Pair {pair['pair_id']} contains non-CEM sources: {sorted(sources - expected_sources)}"
            )
        if len(actions) != 40:
            raise RuntimeError(f"Pair {pair['pair_id']} has {len(actions)} records, expected 40")
        for source in expected_sources:
            count = sum(action["source"] == source for action in actions)
            if count != 20:
                raise RuntimeError(f"Pair {pair['pair_id']} has {count} {source} records, expected 20")


def evaluate_cem_only_variant(
    *,
    variant: str,
    output_path: Path,
    pairs_path: Path,
    pairs: list[dict],
    cells: list[str],
    data_smooth_random_source: Path,
    args: argparse.Namespace,
    dataset,
    valid_action_indices: np.ndarray,
    policy,
    model,
    action_processor,
    offset: int,
) -> dict:
    import gymnasium as gym

    existing = json.loads(output_path.read_text()) if args.resume and output_path.exists() else None
    output = build_variant_output(
        variant=variant,
        pairs_path=pairs_path,
        cells=cells,
        n_pairs=len(pairs),
        seed=args.seed,
        device=args.device,
        action_counts=CEM_ONLY_ACTION_COUNTS,
        existing=existing,
    )
    apply_cem_only_metadata(output, data_smooth_random_source=data_smooth_random_source)
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
                action_counts=CEM_ONLY_ACTION_COUNTS,
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
            apply_cem_only_metadata(output, data_smooth_random_source=data_smooth_random_source)
            write_output(output_path, output, finished=False)
    finally:
        eval_env.close()

    output["metadata"]["wallclock_seconds"] = time.time() - started
    apply_cem_only_metadata(output, data_smooth_random_source=data_smooth_random_source)
    validate_cem_only_output(output, variant=variant)
    write_output(output_path, output, finished=True)
    return output


def main() -> int:
    args = parse_args()
    cells = parse_csv(args.cell_filter)
    row = row_from_cells(cells)
    output_path = (args.output_path or default_output_path(args.variant, row)).expanduser().resolve()
    data_smooth_random_source = (
        args.data_smooth_random_source or default_v3_source_path(row)
    ).expanduser().resolve()
    if not data_smooth_random_source.exists():
        raise FileNotFoundError(f"Missing V3 source for reused data/smooth_random records: {data_smooth_random_source}")
    args.pairs_path = args.pairs_path.expanduser().resolve()

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
        raise ValueError(f"Oracle ablation expects offset=50, got {offset}")
    dataset_path = Path(pair_metadata["dataset_path"])
    cache_dir = dataset_path.parent
    dataset_name = dataset_path.stem

    print("== oracle CEM-only variant setup ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"output_path: {output_path}")
    print(f"data_smooth_random_source: {data_smooth_random_source}")
    print(f"cells: {cells}")
    print(f"variant: {args.variant}")
    print(f"n_pairs: {len(pairs)}")
    print(f"device: {args.device}")
    print(f"action_counts: {CEM_ONLY_ACTION_COUNTS}")

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

    output = evaluate_cem_only_variant(
        variant=args.variant,
        output_path=output_path,
        pairs_path=args.pairs_path,
        pairs=pairs,
        cells=cells,
        data_smooth_random_source=data_smooth_random_source,
        args=args,
        dataset=dataset,
        valid_action_indices=analysis["valid_indices"],
        policy=policy,
        model=model,
        action_processor=action_processor,
        offset=offset,
    )
    print(
        json.dumps(
            {
                "variant": args.variant,
                "output_path": display_path(output_path),
                "n_pairs_completed": output["metadata"]["n_pairs_completed"],
                "wallclock_seconds": output["metadata"]["wallclock_seconds"],
                "actions_subset": output["metadata"]["actions_subset"],
                "data_smooth_random_source": output["metadata"]["data_smooth_random_source"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
