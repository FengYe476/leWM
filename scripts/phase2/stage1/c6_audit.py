#!/usr/bin/env python3
"""C6 Audit S1: random-init LeWM encoder across 10 seeds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.stage1.stage1a_c6_c7 import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_PIXEL_ARTIFACT,
    add_replay_args_if_available,
    build_random_model,
    import_replay_helpers,
    load_cached_pixels,
    pixels_to_policy_array,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    ANCHOR_DEFINITIONS,
    EXPECTED_RECORDS,
    FALSE_ELITE_K,
    LATENT_DIM,
    TOPK_VALUES,
    aggregate_metric_list,
    clean_float,
    compute_metrics,
    iso_now,
    jsonable,
    load_latent_artifact,
    make_anchor_masks,
    run_single_metrics,
    squared_l2_torch,
    summary_row,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "stage1" / "c6_audit" / "s1_random_init_10seed.json"
DEFAULT_SEEDS = tuple(range(10))
V1_FAVORABLE_CELLS = ("D3xR0", "D3xR3")
ORDINARY_COMPLEMENT_ANCHORS = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
)


@dataclass(frozen=True)
class PreparedPixels:
    """Policy-preprocessed pixel tensors shared by every random seed."""

    terminal_batches: list[torch.Tensor]
    goal_pair_ids: list[int]
    goal_batches: list[torch.Tensor]
    metadata: dict[str, Any]


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args_if_available(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixel-artifact", type=Path, default=DEFAULT_PIXEL_ARTIFACT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seeds", type=parse_seed_list, default=DEFAULT_SEEDS)
    parser.add_argument("--c6-batch-size", type=int, default=64)
    args = parser.parse_args()
    if int(args.c6_batch_size) <= 0:
        parser.error("--c6-batch-size must be positive")
    return args


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def print_summary_table(rows: list[dict]) -> None:
    headers = [
        "Control",
        "Config",
        "Seeds",
        "Spearman",
        "Pairwise",
        "PerPairRho",
        "FalseElite",
    ]
    table = []
    for row in rows:
        table.append(
            [
                str(row["control"]),
                str(row["config"]),
                str(row["n_seeds"]),
                f"{fmt(row['global_spearman_mean'])}/{fmt(row['global_spearman_std'])}",
                f"{fmt(row['pairwise_accuracy_mean'])}/{fmt(row['pairwise_accuracy_std'])}",
                f"{fmt(row['per_pair_rho_mean'])}/{fmt(row['per_pair_rho_mean_std'])}",
                f"{fmt(row['false_elite_rate_mean'])}/{fmt(row['false_elite_rate_std'])}",
            ]
        )
    widths = [max(len(headers[i]), *(len(record[i]) for record in table)) for i in range(len(headers))]
    print("C6 Audit S1 random-init LeWM summary (mean/std for aggregate row)")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def tensor_to_numpy(value: torch.Tensor, *, dtype: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype)


def build_common(latent_artifact: dict[str, Any]) -> dict[str, Any]:
    z_terminal = latent_artifact["z_terminal"].detach().cpu().to(dtype=torch.float32)
    z_goal = latent_artifact["z_goal"].detach().cpu().to(dtype=torch.float32)
    c0_cost = squared_l2_torch(z_terminal, z_goal).detach().cpu().numpy().astype(np.float64)
    labels = tensor_to_numpy(latent_artifact["C_real_state"], dtype=np.float64)
    pair_ids = tensor_to_numpy(latent_artifact["pair_id"], dtype=np.int64)
    cells = np.asarray(latent_artifact["cell"], dtype=object)
    return {
        "labels": labels,
        "v1_cost": tensor_to_numpy(latent_artifact["v1_cost"], dtype=np.float64),
        "c0_cost": c0_cost,
        "success": tensor_to_numpy(latent_artifact["success"], dtype=bool),
        "pair_ids": pair_ids,
        "action_ids": tensor_to_numpy(latent_artifact["action_id"], dtype=np.int64),
        "cells": cells,
        "anchor_masks": make_audit_anchor_masks(pair_ids, cells),
    }


def make_audit_anchor_masks(pair_ids: np.ndarray, cells: np.ndarray) -> dict[str, np.ndarray]:
    masks = make_anchor_masks(pair_ids, cells)
    masks["v1_favorable"] = np.isin(cells, np.asarray(V1_FAVORABLE_CELLS, dtype=object))
    ordinary = np.ones(len(pair_ids), dtype=bool)
    for name in ORDINARY_COMPLEMENT_ANCHORS:
        ordinary &= ~masks[name]
    masks["ordinary"] = ordinary
    return masks


def audit_anchor_definitions(pair_ids: np.ndarray, cells: np.ndarray) -> dict[str, Any]:
    masks = make_audit_anchor_masks(pair_ids, cells)
    definitions = jsonable(ANCHOR_DEFINITIONS)
    definitions["v1_favorable"] = {
        "description": "V1-favorable D3xR0 and D3xR3 cells from Track A metadata",
        "pair_ids": sorted(set(int(item) for item in pair_ids[masks["v1_favorable"]])),
        "cells": list(V1_FAVORABLE_CELLS),
    }
    definitions["ordinary"] = {
        "description": "Complement of invisible_quadrant, sign_reversal, latent_favorable, and v1_favorable",
        "pair_ids": sorted(set(int(item) for item in pair_ids[masks["ordinary"]])),
        "excluded_anchors": list(ORDINARY_COMPLEMENT_ANCHORS),
    }
    return definitions


def stage1a_helper_names() -> list[str]:
    return [
        compute_metrics.__name__,
        aggregate_metric_list.__name__,
        summary_row.__name__,
        load_latent_artifact.__name__,
        make_anchor_masks.__name__,
        run_single_metrics.__name__,
        squared_l2_torch.__name__,
    ]


def prepare_policy_tensor_batch(*, policy: Any, pixels: list[Any]) -> torch.Tensor:
    if not pixels:
        return torch.empty((0,), dtype=torch.float32)
    batch = np.stack([pixels_to_policy_array(item) for item in pixels])
    prepared = policy._prepare_info({"pixels": batch[:, None, ...]})
    return prepared["pixels"].detach().cpu().to(dtype=torch.float32)


def chunked_prepare_policy_tensors(
    *,
    policy: Any,
    pixels: list[Any],
    batch_size: int,
) -> list[torch.Tensor]:
    batches = []
    for start in range(0, len(pixels), int(batch_size)):
        end = min(start + int(batch_size), len(pixels))
        batches.append(prepare_policy_tensor_batch(policy=policy, pixels=pixels[start:end]))
    return batches


def collect_cached_pixels_once(
    *,
    args: argparse.Namespace,
    cached_pixels: dict[str, Any],
) -> tuple[list[Any], dict[int, Any], dict[str, Any]]:
    terminal_pixels = [
        cached_pixels["terminal_pixels"][idx]
        for idx in range(int(cached_pixels["terminal_pixels"].shape[0]))
    ]
    goal_map = {
        int(pair_id): pixels
        for pair_id, pixels in cached_pixels["goal_map"].items()
    }
    metadata = {
        "source": "cached_pixel_artifact",
        "path": str(args.pixel_artifact),
        "terminal_records": int(len(terminal_pixels)),
        "goal_pairs": int(len(goal_map)),
        "note": "Pixels were loaded once before the random-seed encoding loop.",
    }
    return terminal_pixels, goal_map, metadata


def collect_replay_pixels_once(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
    helpers: dict[str, Any],
    ctx: Any,
) -> tuple[list[Any], dict[int, Any], dict[str, Any]]:
    replay_latent_artifact = helpers["load_replay_latent_artifact"](args.latent_artifact)
    index_by_key = helpers["latent_index_by_pair_action"](replay_latent_artifact)
    terminal_pixels: list[Any | None] = [None] * EXPECTED_RECORDS
    goal_map: dict[int, Any] = {}
    env = helpers["gym"].make("swm/PushT-v1")
    started = time.time()
    try:
        for pair_idx, pair_spec in enumerate(ctx.requested_pairs):
            pair_id = int(pair_spec["pair_id"])
            pair_started = time.time()
            initial, goal, sequences = helpers["prepare_pair_sequences"](ctx, pair_spec)
            goal_map[pair_id] = goal["pixels"]
            source_counts: dict[str, int] = {}
            for action_id, sequence in enumerate(sequences):
                source, source_index = helpers["source_and_index"](
                    source_counts=source_counts,
                    sequence=sequence,
                )
                rollout, v1_cost = helpers["rollout_action"](
                    env=env,
                    initial=initial,
                    goal=goal,
                    sequence=sequence,
                    seed=int(args.seed) + pair_id * 10_000 + action_id,
                )
                artifact_idx = helpers["validate_replayed_action"](
                    artifact=replay_latent_artifact,
                    index_by_key=index_by_key,
                    pair_id=pair_id,
                    action_id=action_id,
                    source=source,
                    source_index=source_index,
                    v1_cost=v1_cost,
                    atol=float(args.reference_atol),
                )
                terminal_pixels[int(artifact_idx)] = rollout["terminal_pixels"]
            print(
                f"C6 S1 pixel replay pair {pair_idx + 1}/{len(ctx.requested_pairs)} "
                f"pair_id={pair_id} elapsed={time.time() - pair_started:.2f}s"
            )
    finally:
        env.close()

    missing_indices = [idx for idx, pixels in enumerate(terminal_pixels) if pixels is None]
    if missing_indices:
        raise RuntimeError(f"Replay pixel collection left missing terminal pixels at rows: {missing_indices[:20]}")
    latent_pairs = sorted(set(int(item) for item in latent_artifact["pair_id"].detach().cpu().tolist()))
    missing_goals = sorted(set(latent_pairs).difference(goal_map))
    if missing_goals:
        raise RuntimeError(f"Replay pixel collection missing goal pixels for pairs: {missing_goals[:20]}")
    metadata = {
        "source": "simulator_replay",
        "cached_pixel_artifact_present": False,
        "cached_pixel_artifact_path": str(args.pixel_artifact),
        "terminal_records": EXPECTED_RECORDS,
        "goal_pairs": int(len(goal_map)),
        "wallclock_seconds": clean_float(time.time() - started),
        "note": "Simulator replay and V1 validation were performed once before the random-seed encoding loop.",
    }
    return [pixels for pixels in terminal_pixels if pixels is not None], goal_map, metadata


def prepare_pixels_once(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
) -> PreparedPixels:
    helpers = import_replay_helpers()
    ctx = helpers["build_replay_context"](args)
    cached = load_cached_pixels(
        pixel_artifact_path=args.pixel_artifact,
        latent_artifact=latent_artifact,
    )
    if cached is not None:
        cached_pixels, cached_metadata = cached
        terminal_pixels, goal_map, pixel_metadata = collect_cached_pixels_once(
            args=args,
            cached_pixels=cached_pixels,
        )
        pixel_metadata.update(
            {
                "artifact_metadata": cached_metadata.get("artifact_metadata", {}),
                "terminal_pixel_shape": cached_metadata.get("terminal_pixel_shape"),
                "goal_pixel_shape": cached_metadata.get("goal_pixel_shape"),
            }
        )
    else:
        terminal_pixels, goal_map, pixel_metadata = collect_replay_pixels_once(
            args=args,
            latent_artifact=latent_artifact,
            helpers=helpers,
            ctx=ctx,
        )

    unique_pair_ids = sorted(set(int(item) for item in latent_artifact["pair_id"].detach().cpu().tolist()))
    goal_pixels = [goal_map[pair_id] for pair_id in unique_pair_ids]
    prepare_started = time.time()
    terminal_batches = chunked_prepare_policy_tensors(
        policy=ctx.policy,
        pixels=terminal_pixels,
        batch_size=int(args.c6_batch_size),
    )
    goal_batches = chunked_prepare_policy_tensors(
        policy=ctx.policy,
        pixels=goal_pixels,
        batch_size=int(args.c6_batch_size),
    )
    metadata = {
        **pixel_metadata,
        "device": ctx.args.device,
        "prepared_once_before_seed_loop": True,
        "policy_tensor_dtype": "torch.float32",
        "terminal_batches": int(len(terminal_batches)),
        "goal_batches": int(len(goal_batches)),
        "batch_size": int(args.c6_batch_size),
        "terminal_batch_shapes": [list(batch.shape) for batch in terminal_batches],
        "goal_batch_shapes": [list(batch.shape) for batch in goal_batches],
        "policy_prepare_wallclock_seconds": clean_float(time.time() - prepare_started),
    }
    return PreparedPixels(
        terminal_batches=terminal_batches,
        goal_pair_ids=unique_pair_ids,
        goal_batches=goal_batches,
        metadata=metadata,
    )


@torch.inference_mode()
def encode_prepared_batch(
    *,
    model: torch.nn.Module,
    pixels_t: torch.Tensor,
    device: str,
) -> torch.Tensor:
    encoded = model.encode({"pixels": pixels_t.to(device=device, dtype=torch.float32)})
    if "emb" not in encoded:
        raise KeyError("random_model.encode(...) did not return key 'emb'")
    emb = encoded["emb"]
    if emb.ndim == 3:
        emb = emb[:, -1, :]
    elif emb.ndim != 2:
        raise ValueError(f"Unexpected random encode emb shape: {tuple(emb.shape)}")
    if tuple(emb.shape) != (int(pixels_t.shape[0]), LATENT_DIM):
        raise ValueError(
            f"Random encode emb shape must be ({int(pixels_t.shape[0])}, {LATENT_DIM}), "
            f"got {tuple(emb.shape)}"
        )
    if not torch.isfinite(emb).all():
        raise ValueError("Random encode embeddings contain non-finite values")
    return emb.detach().cpu().to(dtype=torch.float32)


def encode_seed_embeddings(
    *,
    args: argparse.Namespace,
    seed: int,
    latent_artifact: dict[str, Any],
    prepared_pixels: PreparedPixels,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model = build_random_model(
        config_path=args.config_path,
        device=str(prepared_pixels.metadata["device"]),
        seed=int(seed),
    )
    terminal_chunks = [
        encode_prepared_batch(
            model=model,
            pixels_t=batch,
            device=str(prepared_pixels.metadata["device"]),
        )
        for batch in prepared_pixels.terminal_batches
    ]
    z_terminal = torch.cat(terminal_chunks, dim=0)

    goal_emb_by_pair: dict[int, torch.Tensor] = {}
    pair_offset = 0
    for batch in prepared_pixels.goal_batches:
        goal_emb = encode_prepared_batch(
            model=model,
            pixels_t=batch,
            device=str(prepared_pixels.metadata["device"]),
        )
        batch_pair_ids = prepared_pixels.goal_pair_ids[pair_offset : pair_offset + int(goal_emb.shape[0])]
        for local_idx, pair_id in enumerate(batch_pair_ids):
            goal_emb_by_pair[int(pair_id)] = goal_emb[local_idx]
        pair_offset += int(goal_emb.shape[0])

    z_goal = torch.stack(
        [goal_emb_by_pair[int(pair_id)] for pair_id in latent_artifact["pair_id"].detach().cpu().tolist()]
    )
    metadata = {
        "seed": int(seed),
        "checkpoint_weights_loaded": False,
        "config_path": str(args.config_path),
        "encode_path": 'model.encode({"pixels": pixels_t})["emb"] with final token if 3D',
        "pixel_source": prepared_pixels.metadata["source"],
        "device": prepared_pixels.metadata["device"],
        "embedding_shape": [LATENT_DIM],
        "z_terminal_shape": list(z_terminal.shape),
        "z_goal_shape": list(z_goal.shape),
        "terminal_batches_encoded": int(len(prepared_pixels.terminal_batches)),
        "goal_batches_encoded": int(len(prepared_pixels.goal_batches)),
        "identical_prepared_pixel_tensors": True,
    }
    del model
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return z_terminal, z_goal, metadata


def compute_seed_metrics(
    *,
    costs: torch.Tensor,
    common: dict[str, Any],
) -> dict[str, Any]:
    return compute_metrics(
        costs=costs.detach().cpu().numpy().astype(np.float64),
        labels=common["labels"],
        v1_cost=common["v1_cost"],
        c0_cost=common["c0_cost"],
        success=common["success"],
        pair_ids=common["pair_ids"],
        action_ids=common["action_ids"],
        cells=common["cells"],
        anchor_masks=common["anchor_masks"],
    )


def seed_validation(cost: torch.Tensor, z_terminal: torch.Tensor, z_goal: torch.Tensor) -> dict[str, Any]:
    return {
        "cost_shape": list(cost.shape),
        "cost_finite": bool(torch.isfinite(cost).all()),
        "cost_min": clean_float(cost.min().item()),
        "cost_max": clean_float(cost.max().item()),
        "cost_mean": clean_float(cost.mean().item()),
        "z_terminal_shape": list(z_terminal.shape),
        "z_goal_shape": list(z_goal.shape),
        "z_terminal_finite": bool(torch.isfinite(z_terminal).all()),
        "z_goal_finite": bool(torch.isfinite(z_goal).all()),
        "random_embedding_dim": LATENT_DIM,
    }


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pixel_artifact = args.pixel_artifact.expanduser().resolve()
    args.config_path = args.config_path.expanduser().resolve()

    latent_artifact = load_latent_artifact(args.latent_artifact)
    common = build_common(latent_artifact)

    print("Preparing C6 S1 pixels once before seed loop...")
    prepared_pixels = prepare_pixels_once(args=args, latent_artifact=latent_artifact)

    per_seed = []
    summary_rows = []
    for seed in args.seeds:
        print(f"Encoding C6 S1 random-init LeWM seed={seed}...")
        started = time.time()
        z_terminal, z_goal, encode_metadata = encode_seed_embeddings(
            args=args,
            seed=int(seed),
            latent_artifact=latent_artifact,
            prepared_pixels=prepared_pixels,
        )
        cost = squared_l2_torch(z_terminal, z_goal)
        validation = seed_validation(cost, z_terminal, z_goal)
        metrics = compute_seed_metrics(costs=cost, common=common)
        encode_metadata["wallclock_seconds"] = clean_float(time.time() - started)
        per_seed.append(
            {
                "seed": int(seed),
                "encode_metadata": encode_metadata,
                "validation": validation,
                "metrics": metrics,
            }
        )
        summary_rows.append(
            summary_row(
                control="C6_S1",
                config=f"random_init_seed={int(seed)}",
                n_seeds=1,
                metrics=metrics,
            )
        )
        del z_terminal, z_goal, cost

    aggregate = aggregate_metric_list([item["metrics"] for item in per_seed])
    summary_rows.append(
        summary_row(
            control="C6_S1",
            config="random_init_10seed",
            n_seeds=len(args.seeds),
            aggregate=aggregate,
        )
    )
    output = {
        "metadata": {
            "format": "c6_audit_s1_random_init_10seed",
            "created_at": iso_now(),
            "sub_experiment": "S1",
            "description": "Random-init LeWM encoder across 10 seeds to test seed robustness of C6 anti-correlation.",
            "latent_artifact": str(args.latent_artifact),
            "output": str(args.output),
            "config_path": str(args.config_path),
            "checkpoint_weights_loaded": False,
            "seeds": [int(seed) for seed in args.seeds],
            "n_records": EXPECTED_RECORDS,
            "latent_dim": LATENT_DIM,
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "anchor_definitions": audit_anchor_definitions(common["pair_ids"], common["cells"]),
            "ordinary_definition": "Complement of invisible_quadrant, sign_reversal, latent_favorable, and v1_favorable.",
            "v1_favorable_source": "Derived from Track A cell labels D3xR0 and D3xR3 in the latent artifact metadata.",
            "tie_rules": {
                "pairwise_accuracy": "Skip C_real_state ties; score tied control costs as 0.5 when C_real_state differs.",
                "topk_and_false_elite": "Ascending cost ranking with deterministic action_id tie-break.",
            },
            "topk_overlap_definition": "|topk(control) intersection topk(reference)| / k, averaged over pairs.",
            "pixel_preparation": prepared_pixels.metadata,
            "metric_helpers_reused": stage1a_helper_names(),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "summary_table": summary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_table(summary_rows)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
