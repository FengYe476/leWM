#!/usr/bin/env python3
"""Extract DINOv2 endpoint features for Phase 2 Track B.

The P2-0 Track A latent artifact stores LeWM latents but not raw terminal
pixels. This script replays the exact Track A action records, encodes the
terminal and goal pixels with frozen DINOv2 ViT-B/14, and saves only features.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch
import torch.nn.functional as F


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


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "track_b" / "dinov2_features.pt"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_MODEL = "dinov2_vitb14"
DINO_REPO = "facebookresearch/dinov2"
DINO_DIM = 768


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_dino_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dinov2(requested_device: str) -> tuple[torch.nn.Module, str, str | None]:
    """Load DINOv2 and fall back to CPU if the requested accelerator fails."""

    first_device = resolve_dino_device(requested_device)
    model = torch.hub.load(DINO_REPO, DINO_MODEL).eval()
    failure: str | None = None
    for device in [first_device, "cpu"] if first_device != "cpu" else ["cpu"]:
        try:
            model = model.to(device)
            with torch.inference_mode():
                test = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
                out = model.forward_features(test)
                _ = out["x_norm_clstoken"]
            return model, device, failure
        except Exception as exc:  # pragma: no cover - hardware dependent.
            failure = f"{device}: {type(exc).__name__}: {exc}"
            model = model.to("cpu")
    raise RuntimeError(f"DINOv2 failed on requested device and CPU: {failure}")


def pixels_to_dino_tensor(pixels: list[np.ndarray], *, device: str) -> torch.Tensor:
    """Convert a list of uint8 HWC/CHW images to normalized DINOv2 input."""

    arrays = []
    for item in pixels:
        arr = np.asarray(item)
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
        arrays.append(np.ascontiguousarray(arr))

    batch = torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
    if float(batch.max().detach().cpu()) > 2.0:
        batch = batch / 255.0
    if tuple(batch.shape[-2:]) != (224, 224):
        batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


@torch.inference_mode()
def encode_pixels(
    dino: torch.nn.Module,
    pixels: list[np.ndarray],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DINOv2 CLS and mean-pooled patch features on CPU."""

    batch = pixels_to_dino_tensor(pixels, device=device)
    features = dino.forward_features(batch)
    cls = features["x_norm_clstoken"].detach().cpu().to(dtype=torch.float32)
    patch_tokens = features["x_norm_patchtokens"]
    mean = patch_tokens.mean(dim=1).detach().cpu().to(dtype=torch.float32)
    return cls, mean


def save_feature_artifact(*, path: Path, records: list[dict[str, Any]], metadata: dict) -> None:
    """Save DINOv2 features sorted in latent-artifact order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda item: (item["pair_id"], item["action_id"]))
    if records:
        d_terminal_cls = torch.stack([item["d_terminal_cls"] for item in records])
        d_goal_cls = torch.stack([item["d_goal_cls"] for item in records])
        d_terminal_mean = torch.stack([item["d_terminal_mean"] for item in records])
        d_goal_mean = torch.stack([item["d_goal_mean"] for item in records])
        pair_id = torch.as_tensor([item["pair_id"] for item in records], dtype=torch.long)
        action_id = torch.as_tensor([item["action_id"] for item in records], dtype=torch.long)
        source_index = torch.as_tensor(
            [item["source_index"] for item in records],
            dtype=torch.long,
        )
    else:
        d_terminal_cls = torch.empty((0, DINO_DIM), dtype=torch.float32)
        d_goal_cls = torch.empty((0, DINO_DIM), dtype=torch.float32)
        d_terminal_mean = torch.empty((0, DINO_DIM), dtype=torch.float32)
        d_goal_mean = torch.empty((0, DINO_DIM), dtype=torch.float32)
        pair_id = torch.empty((0,), dtype=torch.long)
        action_id = torch.empty((0,), dtype=torch.long)
        source_index = torch.empty((0,), dtype=torch.long)

    artifact = {
        "metadata": {
            **metadata,
            "updated_at": iso_now(),
            "n_records": int(len(records)),
        },
        "pair_id": pair_id,
        "action_id": action_id,
        "source": [item["source"] for item in records],
        "source_index": source_index,
        "cell": [item["cell"] for item in records],
        "d_terminal_cls": d_terminal_cls,
        "d_goal_cls": d_goal_cls,
        "d_terminal_mean": d_terminal_mean,
        "d_goal_mean": d_goal_mean,
    }
    torch.save(artifact, path)


def load_existing(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an existing DINOv2 feature artifact for resume."""

    if not path.exists():
        return [], {}
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
                "d_terminal_cls": artifact["d_terminal_cls"][idx],
                "d_goal_cls": artifact["d_goal_cls"][idx],
                "d_terminal_mean": artifact["d_terminal_mean"][idx],
                "d_goal_mean": artifact["d_goal_mean"][idx],
            }
        )
    return records, artifact.get("metadata", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dino-device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    ctx = build_replay_context(args)
    latent_artifact = load_latent_artifact(args.latent_artifact)
    index_by_key = latent_index_by_pair_action(latent_artifact)
    records, old_metadata = load_existing(args.output) if args.resume else ([], {})
    completed_pair_ids = {int(record["pair_id"]) for record in records}

    dino, dino_device, dino_device_fallback = load_dinov2(args.dino_device)
    metadata = {
        **old_metadata,
        "format": "phase2_track_b_dinov2_features",
        "created_at": old_metadata.get("created_at", iso_now()),
        "git_commit": get_git_commit(),
        "seed": int(args.seed),
        "pairs_path": str(args.pairs_path),
        "latent_artifact": str(args.latent_artifact),
        "lewm_device": ctx.args.device,
        "dino_repo": DINO_REPO,
        "dino_model": DINO_MODEL,
        "dino_dim": DINO_DIM,
        "dino_device": dino_device,
        "dino_device_fallback": dino_device_fallback,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "batch_size": int(args.batch_size),
        "action_counts": dict(args.action_counts),
        "note": "Terminal pixels were replayed and encoded in batches; raw pixels were not stored.",
    }

    print("== Phase 2 Track B DINOv2 feature extraction ==")
    print(f"DINOv2: {DINO_REPO}:{DINO_MODEL}")
    print(f"DINOv2 device: {dino_device}")
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
            goal_cls, goal_mean = encode_pixels(dino, [goal["pixels"]], device=dino_device)

            source_counts: dict[str, int] = {}
            pair_pending: list[dict[str, Any]] = []
            pending_pixels: list[np.ndarray] = []
            pair_records: list[dict[str, Any]] = []

            def flush_pending() -> None:
                if not pending_pixels:
                    return
                terminal_cls, terminal_mean = encode_pixels(
                    dino,
                    pending_pixels,
                    device=dino_device,
                )
                for local_idx, pending in enumerate(pair_pending):
                    pending["d_terminal_cls"] = terminal_cls[local_idx]
                    pending["d_terminal_mean"] = terminal_mean[local_idx]
                    pending["d_goal_cls"] = goal_cls[0]
                    pending["d_goal_mean"] = goal_mean[0]
                    pair_records.append(pending)
                pending_pixels.clear()
                pair_pending.clear()

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
                pair_pending.append(
                    {
                        "pair_id": pair_id,
                        "action_id": int(action_id),
                        "source": source,
                        "source_index": int(source_index),
                        "cell": str(pair_spec["cell"]),
                    }
                )
                pending_pixels.append(rollout["terminal_pixels"])
                if len(pending_pixels) >= int(args.batch_size):
                    flush_pending()
            flush_pending()

            records.extend(pair_records)
            completed_pair_ids.add(pair_id)
            save_feature_artifact(path=args.output, records=records, metadata=metadata)
            print(
                f"pair_id={pair_id} cell={pair_spec['cell']} "
                f"records={len(pair_records)} elapsed={time.time() - pair_started:.2f}s"
            )
    finally:
        env.close()

    metadata["finished_at"] = iso_now()
    metadata["wallclock_seconds"] = time.time() - started
    save_feature_artifact(path=args.output, records=records, metadata=metadata)

    saved = torch.load(args.output, map_location="cpu", weights_only=False)
    if not torch.equal(saved["pair_id"], latent_artifact["pair_id"][: saved["pair_id"].numel()]):
        raise RuntimeError("Saved DINOv2 pair_id ordering does not match latent artifact")
    print(f"Saved: {args.output}")
    print(f"records: {int(saved['pair_id'].numel())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
