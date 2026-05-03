#!/usr/bin/env python3
"""Replay Track A endpoints and save raw terminal/goal pixels for Track B.

This script exists because the P2-0 latent artifact does not contain pixels.
The DINOv2 extraction script can encode pixels on the fly, which is usually
preferable, but this artifact is useful for debugging and reproducibility.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch


PROJECT_ROOT_LOCAL = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_LOCAL))

from scripts.phase1.eval_track_a_three_cost import get_git_commit  # noqa: E402
from scripts.phase2.track_b_common import (  # noqa: E402
    PROJECT_ROOT,
    add_replay_args,
    build_replay_context,
    latent_index_by_pair_action,
    load_latent_artifact,
    prepare_pair_sequences,
    rollout_action,
    source_and_index,
    validate_replayed_action,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "track_b" / "track_a_pixels.pt"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pixels_to_chw_uint8(pixels: np.ndarray) -> torch.Tensor:
    """Convert HWC/CHW pixels to a contiguous uint8 CHW tensor."""

    arr = np.asarray(pixels)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")
    if arr.shape[-1] in (1, 3, 4):
        arr = arr[..., :3].transpose(2, 0, 1)
    elif arr.shape[0] in (1, 3, 4):
        arr = arr[:3]
    else:
        raise ValueError(f"Cannot infer channel axis for pixels with shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(arr))


def save_pixel_artifact(
    *,
    path: Path,
    records: list[dict],
    goals_by_pair: dict[int, torch.Tensor],
    metadata: dict,
) -> None:
    """Save replayed pixels in latent-artifact order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda item: (item["pair_id"], item["action_id"]))
    goal_items = sorted(goals_by_pair.items())
    artifact = {
        "metadata": {
            **metadata,
            "updated_at": iso_now(),
            "n_records": len(records),
            "n_goal_pairs": len(goal_items),
        },
        "pair_id": torch.as_tensor([item["pair_id"] for item in records], dtype=torch.long),
        "action_id": torch.as_tensor([item["action_id"] for item in records], dtype=torch.long),
        "source": [item["source"] for item in records],
        "source_index": torch.as_tensor(
            [item["source_index"] for item in records],
            dtype=torch.long,
        ),
        "cell": [item["cell"] for item in records],
        "terminal_pixels": torch.stack([item["terminal_pixels"] for item in records]),
        "goal_pair_id": torch.as_tensor([pair_id for pair_id, _ in goal_items], dtype=torch.long),
        "goal_pixels": torch.stack([pixels for _, pixels in goal_items]),
    }
    torch.save(artifact, path)


def load_existing(path: Path) -> tuple[list[dict], dict[int, torch.Tensor], dict]:
    """Load an existing pixel artifact for resume."""

    if not path.exists():
        return [], {}, {}
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    records = []
    for idx in range(int(artifact["pair_id"].numel())):
        records.append(
            {
                "pair_id": int(artifact["pair_id"][idx]),
                "action_id": int(artifact["action_id"][idx]),
                "source": artifact["source"][idx],
                "source_index": int(artifact["source_index"][idx]),
                "cell": artifact["cell"][idx],
                "terminal_pixels": artifact["terminal_pixels"][idx],
            }
        )
    goals = {
        int(pair_id): artifact["goal_pixels"][idx]
        for idx, pair_id in enumerate(artifact["goal_pair_id"].tolist())
    }
    return records, goals, artifact.get("metadata", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    ctx = build_replay_context(args)
    latent_artifact = load_latent_artifact(args.latent_artifact)
    index_by_key = latent_index_by_pair_action(latent_artifact)

    records, goals_by_pair, old_metadata = load_existing(args.output) if args.resume else ([], {}, {})
    completed_pair_ids = {int(record["pair_id"]) for record in records}
    metadata = {
        **old_metadata,
        "format": "phase2_track_b_track_a_pixels",
        "created_at": old_metadata.get("created_at", iso_now()),
        "git_commit": get_git_commit(),
        "seed": int(args.seed),
        "pairs_path": str(args.pairs_path),
        "latent_artifact": str(args.latent_artifact),
        "device": ctx.args.device,
        "action_counts": dict(args.action_counts),
        "note": "Pixels replayed from the exact Track A action-selection pipeline.",
    }

    print("== Phase 2 Track B pixel extraction ==")
    print(f"pairs: {len(ctx.requested_pairs)}")
    print(f"output: {args.output}")
    print(f"resume_completed_pairs: {len(completed_pair_ids)}")
    started = time.time()

    env = gym.make("swm/PushT-v1")
    try:
        for pair_spec in ctx.requested_pairs:
            pair_id = int(pair_spec["pair_id"])
            if pair_id in completed_pair_ids:
                print(f"Skipping completed pair_id={pair_id}")
                continue
            pair_started = time.time()
            initial, goal, sequences = prepare_pair_sequences(ctx, pair_spec)
            goals_by_pair[pair_id] = pixels_to_chw_uint8(goal["pixels"])
            source_counts: dict[str, int] = {}
            pair_records = []
            for action_id, sequence in enumerate(sequences):
                source, source_index = source_and_index(
                    source_counts=source_counts,
                    sequence=sequence,
                )
                rollout, v1_cost = rollout_action(
                    env=env,
                    initial=initial,
                    goal=goal,
                    sequence=sequence,
                    seed=int(args.seed) + pair_id * 10_000 + action_id,
                )
                validate_replayed_action(
                    artifact=latent_artifact,
                    index_by_key=index_by_key,
                    pair_id=pair_id,
                    action_id=action_id,
                    source=source,
                    source_index=source_index,
                    v1_cost=v1_cost,
                    atol=float(args.reference_atol),
                )
                pair_records.append(
                    {
                        "pair_id": pair_id,
                        "action_id": int(action_id),
                        "source": source,
                        "source_index": int(source_index),
                        "cell": str(pair_spec["cell"]),
                        "terminal_pixels": pixels_to_chw_uint8(rollout["terminal_pixels"]),
                    }
                )
            records.extend(pair_records)
            completed_pair_ids.add(pair_id)
            save_pixel_artifact(
                path=args.output,
                records=records,
                goals_by_pair=goals_by_pair,
                metadata=metadata,
            )
            print(
                f"pair_id={pair_id} saved_records={len(pair_records)} "
                f"elapsed={time.time() - pair_started:.2f}s"
            )
    finally:
        env.close()

    metadata["finished_at"] = iso_now()
    metadata["wallclock_seconds"] = time.time() - started
    save_pixel_artifact(
        path=args.output,
        records=records,
        goals_by_pair=goals_by_pair,
        metadata=metadata,
    )
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
