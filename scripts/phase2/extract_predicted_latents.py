#!/usr/bin/env python3
"""Extract predictor-imagined terminal latents for all Track A action records."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import prepare_pair_info  # noqa: E402
from lewm_audit.eval.pusht import analyze_offset, prepare_dataset_index  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    SOURCE_LABELS,
    load_pairs,
    make_policy_namespace,
    make_three_cost_namespace,
    parse_pair_ids,
    select_action_sequences,
    validate_requested_pair_offsets,
)
from scripts.phase2.dataloader import (  # noqa: E402
    DEFAULT_LATENT_ARTIFACT,
    DEFAULT_PREDICTED_LATENT_ARTIFACT,
)
from scripts.phase2.diagnose_planning_gap import (  # noqa: E402
    ACTION_COUNTS,
    rollout_predicted_latents,
    source_index_for_action,
)


LATENT_DIM = 192


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREDICTED_LATENT_ARTIFACT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    return parser.parse_args()


def iso_now() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def tensor_from_records(records: list[dict], key: str, dtype: torch.dtype) -> torch.Tensor:
    """Build a tensor from one scalar field across records."""
    return torch.as_tensor([record[key] for record in records], dtype=dtype)


def save_predicted_artifact(path: Path, metadata: dict, records: list[dict]) -> None:
    """Save predictor-latent records to a torch artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda item: (item["pair_id"], item["action_id"]))
    if records:
        z_predicted = torch.stack(
            [torch.as_tensor(record["z_predicted"], dtype=torch.float32) for record in records]
        )
        z_goal_pred = torch.stack(
            [torch.as_tensor(record["z_goal_pred"], dtype=torch.float32) for record in records]
        )
    else:
        z_predicted = torch.empty((0, LATENT_DIM), dtype=torch.float32)
        z_goal_pred = torch.empty((0, LATENT_DIM), dtype=torch.float32)

    artifact = {
        "metadata": {
            **metadata,
            "n_records": len(records),
            "n_pairs_completed": len({record["pair_id"] for record in records}),
            "updated_at": iso_now(),
        },
        "pair_id": tensor_from_records(records, "pair_id", torch.long),
        "action_id": tensor_from_records(records, "action_id", torch.long),
        "source": [record["source"] for record in records],
        "source_index": tensor_from_records(records, "source_index", torch.long),
        "action_key": [record["action_key"] for record in records],
        "cell": [record["cell"] for record in records],
        "z_predicted": z_predicted,
        "z_goal_pred": z_goal_pred,
    }
    torch.save(artifact, path)


def validate_against_real_artifact(records: list[dict], real_artifact: dict) -> None:
    """Assert extracted records match the real latent artifact ordering."""
    records = sorted(records, key=lambda item: (item["pair_id"], item["action_id"]))
    if len(records) != int(real_artifact["pair_id"].numel()):
        raise ValueError(
            f"Record count mismatch: predicted={len(records)} "
            f"real={int(real_artifact['pair_id'].numel())}"
        )
    for idx, record in enumerate(records):
        checks = {
            "pair_id": int(real_artifact["pair_id"][idx]),
            "action_id": int(real_artifact["action_id"][idx]),
            "source": str(real_artifact["source"][idx]),
            "source_index": int(real_artifact["source_index"][idx]),
        }
        for key, expected in checks.items():
            if record[key] != expected:
                raise ValueError(
                    f"Real artifact alignment mismatch at idx={idx} key={key}: "
                    f"{record[key]!r} != {expected!r}"
                )


def main() -> int:
    """Replay Track A action generation and save predictor terminal latents."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = resolve_device(args.device)
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.artifact = args.artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()

    pairs_data, requested_pairs = load_pairs(
        args.pairs_path,
        max_pairs=args.max_pairs,
        pair_ids=args.pair_ids,
    )
    pair_metadata = pairs_data["metadata"]
    offset = int(pair_metadata["offset"])
    validate_requested_pair_offsets(requested_pairs, offset=offset)

    dataset_path = Path(pair_metadata["dataset_path"])
    args.cache_dir = dataset_path.parent
    args.dataset_name = dataset_path.stem
    real_artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)

    print("== P2-0 predicted-latent extraction ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"real_artifact: {args.artifact}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")

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
        max_cem_count=max(ACTION_COUNTS["CEM_early"], ACTION_COUNTS["CEM_late"]),
    )

    records = []
    started = time.time()
    for pair_idx, pair_spec in enumerate(requested_pairs, start=1):
        pair_start = time.time()
        pair_id = int(pair_spec["pair_id"])
        rows = dataset.get_row_data([int(pair_spec["start_row"]), int(pair_spec["goal_row"])])
        initial = {key: value[0] for key, value in rows.items()}
        goal = {key: value[1] for key, value in rows.items()}
        prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
        raw_steps = int(pair_spec["goal_row"]) - int(pair_spec["start_row"])
        sequences = select_action_sequences(
            dataset=dataset,
            valid_action_indices=analysis["valid_indices"],
            policy=policy,
            model=model,
            prepared_info=prepared_info,
            args=cost_args,
            pair_id=pair_id,
            raw_steps=raw_steps,
            action_counts=ACTION_COUNTS,
        )
        blocked = np.stack([sequence["blocked_normalized"] for sequence in sequences])
        z_predicted, z_goal_pred = rollout_predicted_latents(model, prepared_info, blocked)
        source_counts: dict[str, int] = {}
        for action_id, sequence in enumerate(sequences):
            source = SOURCE_LABELS[sequence["source"]]
            source_index = source_index_for_action(source_counts, source)
            records.append(
                {
                    "pair_id": pair_id,
                    "action_id": int(action_id),
                    "source": source,
                    "source_index": int(source_index),
                    "action_key": f"{pair_id}:{source}:{source_index}",
                    "cell": str(pair_spec["cell"]),
                    "z_predicted": torch.as_tensor(z_predicted[action_id], dtype=torch.float32),
                    "z_goal_pred": torch.as_tensor(z_goal_pred, dtype=torch.float32),
                }
            )
        print(
            f"[{pair_idx:03d}/{len(requested_pairs):03d}] pair={pair_id} "
            f"cell={pair_spec['cell']} records={len(sequences)} "
            f"elapsed={time.time() - pair_start:.1f}s"
        )

    if args.max_pairs is None and args.pair_ids is None:
        validate_against_real_artifact(records, real_artifact)

    metadata = {
        "format": "p2_0_track_a_predicted_latents",
        "created_at": iso_now(),
        "seed": int(args.seed),
        "device": args.device,
        "pairs_path": str(args.pairs_path),
        "real_latent_artifact": str(args.artifact),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_cache_dir": str(args.cache_dir),
        "dataset_name": args.dataset_name,
        "offset": offset,
        "action_counts": ACTION_COUNTS,
        "latent_dim": LATENT_DIM,
        "wallclock_seconds": time.time() - started,
    }
    save_predicted_artifact(args.output, metadata, records)
    print(f"saved: {args.output}")
    print(f"records: {len(records)}")
    print(f"elapsed_seconds: {time.time() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
