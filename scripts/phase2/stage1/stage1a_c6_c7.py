#!/usr/bin/env python3
"""Stage 1A C6 random-init LeWM and C7 DINOv2 endpoint controls."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    ANCHOR_DEFINITIONS,
    EXPECTED_RECORDS,
    FALSE_ELITE_K,
    LATENT_DIM,
    TOPK_VALUES,
    DEFAULT_LATENT_ARTIFACT,
    clean_float,
    iso_now,
    jsonable,
    load_latent_artifact,
    make_anchor_masks,
    run_single_metrics,
    squared_l2_torch,
)


DEFAULT_DINO_FEATURES = PROJECT_ROOT / "results" / "phase2" / "track_b" / "dinov2_features.pt"
DEFAULT_PIXEL_ARTIFACT = PROJECT_ROOT / "results" / "phase2" / "track_b" / "track_a_pixels.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "lewm-pusht" / "config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_c6_c7.json"
DEFAULT_C6_SEED = 0


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def print_summary_rows(rows: list[dict], *, title: str) -> None:
    headers = [
        "Control",
        "Config",
        "Seeds",
        "Spearman",
        "Pairwise",
        "PerPairRho",
        "FalseElite",
        "Status",
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
                str(row.get("status", "ok")),
            ]
        )
    widths = [
        max(len(headers[i]), *(len(record[i]) for record in table))
        for i in range(len(headers))
    ]
    print(title)
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def summary_row(
    *,
    control: str,
    config: str,
    n_seeds: int,
    metrics: dict | None = None,
    status: str = "ok",
) -> dict:
    if metrics is None:
        return {
            "control": control,
            "config": config,
            "n_seeds": int(n_seeds),
            "global_spearman_mean": None,
            "global_spearman_std": None,
            "pairwise_accuracy_mean": None,
            "pairwise_accuracy_std": None,
            "per_pair_rho_mean": None,
            "per_pair_rho_mean_std": None,
            "false_elite_rate_mean": None,
            "false_elite_rate_std": None,
            "status": status,
        }
    return {
        "control": control,
        "config": config,
        "n_seeds": int(n_seeds),
        "global_spearman_mean": metrics.get("global_spearman"),
        "global_spearman_std": None,
        "pairwise_accuracy_mean": metrics.get("pairwise_accuracy"),
        "pairwise_accuracy_std": None,
        "per_pair_rho_mean": metrics.get("per_pair_spearman", {}).get("mean"),
        "per_pair_rho_mean_std": None,
        "false_elite_rate_mean": metrics.get("false_elite_rate"),
        "false_elite_rate_std": None,
        "status": status,
    }


def add_replay_args_if_available(parser: argparse.ArgumentParser) -> None:
    try:
        from scripts.phase2.track_b_common import add_replay_args  # noqa: PLC0415

        add_replay_args(parser)
        parser.set_defaults(replay_arg_import_error=None)
    except Exception as exc:  # noqa: BLE001
        parser.add_argument("--latent-artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
        parser.add_argument("--device", default="auto")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--reference-atol", type=float, default=1e-3)
        parser.set_defaults(
            replay_arg_import_error=f"{type(exc).__name__}: {exc}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args_if_available(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dino-features", type=Path, default=DEFAULT_DINO_FEATURES)
    parser.add_argument("--pixel-artifact", type=Path, default=DEFAULT_PIXEL_ARTIFACT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--c6-seed", type=int, default=DEFAULT_C6_SEED)
    parser.add_argument("--c6-batch-size", type=int, default=64)
    parser.add_argument("--skip-c6", action="store_true")
    return parser.parse_args()


def tensor_to_numpy(value: torch.Tensor, *, dtype: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype)


def build_common(artifact: dict[str, Any]) -> dict[str, Any]:
    z_terminal = artifact["z_terminal"].detach().cpu().to(dtype=torch.float32)
    z_goal = artifact["z_goal"].detach().cpu().to(dtype=torch.float32)
    c0_cost_t = squared_l2_torch(z_terminal, z_goal)
    labels = tensor_to_numpy(artifact["C_real_state"], dtype=np.float64)
    pair_ids = tensor_to_numpy(artifact["pair_id"], dtype=np.int64)
    action_ids = tensor_to_numpy(artifact["action_id"], dtype=np.int64)
    cells = np.asarray(artifact["cell"], dtype=object)
    return {
        "labels": labels,
        "v1_cost": tensor_to_numpy(artifact["v1_cost"], dtype=np.float64),
        "c0_cost": c0_cost_t.detach().cpu().numpy().astype(np.float64),
        "success": tensor_to_numpy(artifact["success"], dtype=bool),
        "pair_ids": pair_ids,
        "action_ids": action_ids,
        "cells": cells,
        "anchor_masks": make_anchor_masks(pair_ids, cells),
        "c0_cost_t": c0_cost_t,
    }


def validate_feature_artifact_order(
    feature_artifact: dict[str, Any],
    latent_artifact: dict[str, Any],
) -> dict[str, Any]:
    validation = {}
    for key in ("pair_id", "action_id"):
        if key not in feature_artifact:
            raise KeyError(f"DINOv2 feature artifact missing required key: {key}")
        matches = torch.equal(
            feature_artifact[key].detach().cpu(),
            latent_artifact[key].detach().cpu(),
        )
        validation[f"{key}_matches"] = bool(matches)
        if not matches:
            raise RuntimeError(f"DINOv2 {key} ordering does not match latent artifact")
    for key in ("source", "cell"):
        if key in feature_artifact:
            matches = list(feature_artifact[key]) == list(latent_artifact[key])
            validation[f"{key}_matches"] = bool(matches)
    validation["passed"] = bool(
        validation.get("pair_id_matches") and validation.get("action_id_matches")
    )
    return validation


def require_feature_tensor(artifact: dict[str, Any], key: str) -> torch.Tensor:
    if key not in artifact:
        raise KeyError(f"DINOv2 feature artifact missing required key: {key}")
    tensor = artifact[key].detach().cpu().to(dtype=torch.float32)
    if tuple(tensor.shape) != (EXPECTED_RECORDS, 768):
        raise ValueError(f"Unexpected {key} shape: {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{key} contains non-finite values")
    return tensor


def compute_c7(
    *,
    dino_features_path: Path,
    latent_artifact: dict[str, Any],
    common: dict[str, Any],
) -> tuple[dict[str, Any], list[dict], dict[str, Any]]:
    if not dino_features_path.exists():
        raise FileNotFoundError(f"Missing DINOv2 feature artifact: {dino_features_path}")
    dino_artifact = torch.load(dino_features_path, map_location="cpu", weights_only=False)
    order_validation = validate_feature_artifact_order(dino_artifact, latent_artifact)
    terminal_cls = require_feature_tensor(dino_artifact, "d_terminal_cls")
    goal_cls = require_feature_tensor(dino_artifact, "d_goal_cls")
    terminal_mean = require_feature_tensor(dino_artifact, "d_terminal_mean")
    goal_mean = require_feature_tensor(dino_artifact, "d_goal_mean")

    controls = {}
    rows = []
    variants = {
        "C7_cls": ("dinov2_cls", terminal_cls, goal_cls),
        "C7_mean": ("dinov2_mean", terminal_mean, goal_mean),
    }
    for control_name, (config_name, terminal, goal) in variants.items():
        cost = squared_l2_torch(terminal, goal)
        cost_validation = {
            "shape": list(cost.shape),
            "finite": bool(torch.isfinite(cost).all()),
            "min": clean_float(cost.min().item()),
            "max": clean_float(cost.max().item()),
            "mean": clean_float(cost.mean().item()),
        }
        metrics = run_single_metrics(costs=cost, **without_private_common(common))
        controls[control_name] = {
            "name": "DINOv2 endpoint feature distance",
            "status": "ok",
            "config": {
                "variant": config_name,
                "feature_dim": 768,
                "cost": "squared_l2",
                "lower_is_better": True,
            },
            "validation": {
                "feature_order": order_validation,
                "cost": cost_validation,
            },
            "metrics": metrics,
        }
        rows.append(
            summary_row(
                control=control_name,
                config=config_name,
                n_seeds=1,
                metrics=metrics,
            )
        )
    metadata = {
        "path": str(dino_features_path),
        "artifact_metadata": dino_artifact.get("metadata", {}),
        "order_validation": order_validation,
    }
    return controls, rows, metadata


def without_private_common(common: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in common.items() if not key.endswith("_t")}


def pixels_to_policy_array(pixels: Any) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected one image with 3 dims, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = arr[:3].transpose(1, 2, 0)
    elif arr.shape[-1] in (1, 3, 4):
        arr = arr[..., :3]
    else:
        raise ValueError(f"Cannot infer channel axis for pixels with shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@torch.inference_mode()
def encode_pixel_batch(
    *,
    policy: Any,
    model: torch.nn.Module,
    pixels: list[Any],
    device: str,
) -> torch.Tensor:
    if not pixels:
        return torch.empty((0, LATENT_DIM), dtype=torch.float32)
    batch = np.stack([pixels_to_policy_array(item) for item in pixels])
    prepared = policy._prepare_info({"pixels": batch[:, None, ...]})
    pixels_t = prepared["pixels"].to(device=device, dtype=torch.float32)
    encoded = model.encode({"pixels": pixels_t})
    if "emb" not in encoded:
        raise KeyError("random_model.encode(...) did not return key 'emb'")
    emb = encoded["emb"]
    if emb.ndim == 3:
        emb = emb[:, -1, :]
    elif emb.ndim != 2:
        raise ValueError(f"Unexpected random encode emb shape: {tuple(emb.shape)}")
    if tuple(emb.shape) != (len(pixels), LATENT_DIM):
        raise ValueError(
            f"Random encode emb shape must be ({len(pixels)}, {LATENT_DIM}), "
            f"got {tuple(emb.shape)}"
        )
    if not torch.isfinite(emb).all():
        raise ValueError("Random encode embeddings contain non-finite values")
    return emb.detach().cpu().to(dtype=torch.float32)


def build_random_model(*, config_path: Path, device: str, seed: int) -> torch.nn.Module:
    from scripts.verify_checkpoint import build_model_from_config, load_json  # noqa: PLC0415

    if not config_path.exists():
        raise FileNotFoundError(f"Missing LeWM config: {config_path}")
    cfg = load_json(config_path)
    torch.manual_seed(int(seed))
    model = build_model_from_config(cfg)
    model.to(device)
    model.eval()
    return model


def load_cached_pixels(
    *,
    pixel_artifact_path: Path,
    latent_artifact: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not pixel_artifact_path.exists():
        return None
    artifact = torch.load(pixel_artifact_path, map_location="cpu", weights_only=False)
    required = ("pair_id", "action_id", "terminal_pixels", "goal_pair_id", "goal_pixels")
    missing = [key for key in required if key not in artifact]
    if missing:
        raise KeyError(f"Pixel artifact missing required keys: {missing}")
    if not torch.equal(artifact["pair_id"].detach().cpu(), latent_artifact["pair_id"].detach().cpu()):
        raise RuntimeError("Cached pixel artifact pair_id ordering does not match latent artifact")
    if not torch.equal(artifact["action_id"].detach().cpu(), latent_artifact["action_id"].detach().cpu()):
        raise RuntimeError("Cached pixel artifact action_id ordering does not match latent artifact")
    if int(artifact["terminal_pixels"].shape[0]) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} terminal pixels, "
            f"found {int(artifact['terminal_pixels'].shape[0])}"
        )
    goal_map = {
        int(pair_id): artifact["goal_pixels"][idx]
        for idx, pair_id in enumerate(artifact["goal_pair_id"].detach().cpu().tolist())
    }
    latent_pairs = set(int(item) for item in latent_artifact["pair_id"].detach().cpu().tolist())
    missing_goals = sorted(latent_pairs.difference(goal_map))
    if missing_goals:
        raise RuntimeError(f"Cached pixel artifact missing goal pixels for pairs: {missing_goals}")
    metadata = {
        "source": "cached_pixel_artifact",
        "path": str(pixel_artifact_path),
        "artifact_metadata": artifact.get("metadata", {}),
        "terminal_pixel_shape": list(artifact["terminal_pixels"].shape),
        "goal_pixel_shape": list(artifact["goal_pixels"].shape),
    }
    return {"terminal_pixels": artifact["terminal_pixels"], "goal_map": goal_map}, metadata


def import_replay_helpers() -> dict[str, Any]:
    import gymnasium as gym  # noqa: PLC0415
    import stable_worldmodel as swm  # noqa: F401, PLC0415
    from scripts.phase2.track_b_common import (  # noqa: PLC0415
        build_replay_context,
        latent_index_by_pair_action,
        load_latent_artifact as load_replay_latent_artifact,
        prepare_pair_sequences,
        rollout_action,
        source_and_index,
        validate_replayed_action,
    )

    return {
        "gym": gym,
        "build_replay_context": build_replay_context,
        "latent_index_by_pair_action": latent_index_by_pair_action,
        "load_replay_latent_artifact": load_replay_latent_artifact,
        "prepare_pair_sequences": prepare_pair_sequences,
        "rollout_action": rollout_action,
        "source_and_index": source_and_index,
        "validate_replayed_action": validate_replayed_action,
    }


def encode_c6_from_cached_pixels(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
    cached_pixels: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    helpers = import_replay_helpers()
    ctx = helpers["build_replay_context"](args)
    random_model = build_random_model(
        config_path=args.config_path,
        device=ctx.args.device,
        seed=int(args.c6_seed),
    )

    unique_pair_ids = sorted(set(int(item) for item in latent_artifact["pair_id"].tolist()))
    goal_emb_by_pair = {}
    for start in range(0, len(unique_pair_ids), int(args.c6_batch_size)):
        batch_pair_ids = unique_pair_ids[start : start + int(args.c6_batch_size)]
        goal_pixels = [cached_pixels["goal_map"][pair_id] for pair_id in batch_pair_ids]
        goal_emb = encode_pixel_batch(
            policy=ctx.policy,
            model=random_model,
            pixels=goal_pixels,
            device=ctx.args.device,
        )
        for local_idx, pair_id in enumerate(batch_pair_ids):
            goal_emb_by_pair[pair_id] = goal_emb[local_idx]

    terminal_emb_chunks = []
    terminal_pixels = cached_pixels["terminal_pixels"]
    for start in range(0, EXPECTED_RECORDS, int(args.c6_batch_size)):
        end = min(start + int(args.c6_batch_size), EXPECTED_RECORDS)
        terminal_emb_chunks.append(
            encode_pixel_batch(
                policy=ctx.policy,
                model=random_model,
                pixels=[terminal_pixels[idx] for idx in range(start, end)],
                device=ctx.args.device,
            )
        )
    z_terminal = torch.cat(terminal_emb_chunks, dim=0)
    z_goal = torch.stack(
        [goal_emb_by_pair[int(pair_id)] for pair_id in latent_artifact["pair_id"].tolist()]
    )
    metadata = {
        "pixel_source": "cached_pixel_artifact",
        "device": ctx.args.device,
        "z_terminal_shape": list(z_terminal.shape),
        "z_goal_shape": list(z_goal.shape),
    }
    return squared_l2_torch(z_terminal, z_goal), metadata


def encode_c6_from_replay(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    helpers = import_replay_helpers()
    ctx = helpers["build_replay_context"](args)
    replay_latent_artifact = helpers["load_replay_latent_artifact"](args.latent_artifact)
    index_by_key = helpers["latent_index_by_pair_action"](replay_latent_artifact)
    random_model = build_random_model(
        config_path=args.config_path,
        device=ctx.args.device,
        seed=int(args.c6_seed),
    )
    costs = torch.full((EXPECTED_RECORDS,), float("nan"), dtype=torch.float32)

    env = helpers["gym"].make("swm/PushT-v1")
    started = time.time()
    try:
        for pair_idx, pair_spec in enumerate(ctx.requested_pairs):
            pair_id = int(pair_spec["pair_id"])
            pair_started = time.time()
            initial, goal, sequences = helpers["prepare_pair_sequences"](ctx, pair_spec)
            goal_emb = encode_pixel_batch(
                policy=ctx.policy,
                model=random_model,
                pixels=[goal["pixels"]],
                device=ctx.args.device,
            )[0]

            source_counts: dict[str, int] = {}
            pending_pixels: list[Any] = []
            pending_indices: list[int] = []

            def flush_pending() -> None:
                if not pending_pixels:
                    return
                terminal_emb = encode_pixel_batch(
                    policy=ctx.policy,
                    model=random_model,
                    pixels=pending_pixels,
                    device=ctx.args.device,
                )
                goal_batch = goal_emb.unsqueeze(0).expand(len(pending_indices), -1)
                batch_cost = squared_l2_torch(terminal_emb, goal_batch)
                for local_idx, artifact_idx in enumerate(pending_indices):
                    costs[artifact_idx] = batch_cost[local_idx]
                pending_pixels.clear()
                pending_indices.clear()

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
                pending_pixels.append(rollout["terminal_pixels"])
                pending_indices.append(int(artifact_idx))
                if len(pending_pixels) >= int(args.c6_batch_size):
                    flush_pending()
            flush_pending()
            print(
                f"C6 replay pair {pair_idx + 1}/{len(ctx.requested_pairs)} "
                f"pair_id={pair_id} elapsed={time.time() - pair_started:.2f}s"
            )
    finally:
        env.close()

    if not torch.isfinite(costs).all():
        missing = torch.nonzero(~torch.isfinite(costs), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"C6 replay left non-finite costs at rows: {missing[:20]}")
    # Order validation is implicit in validate_replayed_action; keep this extra check for clarity.
    if int(costs.numel()) != int(latent_artifact["pair_id"].numel()):
        raise RuntimeError("C6 replay cost count does not match latent artifact")
    metadata = {
        "pixel_source": "simulator_replay",
        "device": ctx.args.device,
        "n_records_encoded": int(costs.numel()),
        "z_terminal_shape": [EXPECTED_RECORDS, LATENT_DIM],
        "z_goal_shape": [EXPECTED_RECORDS, LATENT_DIM],
        "goal_pairs_encoded": int(len(ctx.requested_pairs)),
        "wallclock_seconds": clean_float(time.time() - started),
    }
    return costs, metadata


def compute_c6(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
    common: dict[str, Any],
) -> tuple[dict[str, Any], list[dict], dict[str, Any]]:
    base_metadata = {
        "random_seed": int(args.c6_seed),
        "config_path": str(args.config_path),
        "checkpoint_weights_loaded": False,
        "encode_path": 'model.encode({"pixels": pixels_t})["emb"][:, -1, :]',
        "target_dim": LATENT_DIM,
    }
    if args.skip_c6:
        blocker = "Skipped by --skip-c6."
        control = blocked_c6_control(base_metadata=base_metadata, blocker=blocker)
        return {"C6": control}, [summary_row(control="C6", config="random_init_seed0", n_seeds=0, status="blocked")], {
            **base_metadata,
            "status": "blocked",
            "blocker": blocker,
        }

    try:
        cached = load_cached_pixels(
            pixel_artifact_path=args.pixel_artifact,
            latent_artifact=latent_artifact,
        )
        if cached is not None:
            cached_pixels, pixel_metadata = cached
            cost, encode_metadata = encode_c6_from_cached_pixels(
                args=args,
                latent_artifact=latent_artifact,
                cached_pixels=cached_pixels,
            )
        else:
            pixel_metadata = {
                "source": "simulator_replay",
                "cached_pixel_artifact_present": False,
                "cached_pixel_artifact_path": str(args.pixel_artifact),
                "note": "Terminal pixels were produced by simulator rollout because no cached pixel artifact was present.",
            }
            cost, encode_metadata = encode_c6_from_replay(args=args, latent_artifact=latent_artifact)

        metrics = run_single_metrics(costs=cost, **without_private_common(common))
        validation = {
            "cost_shape": list(cost.shape),
            "cost_finite": bool(torch.isfinite(cost).all()),
            "cost_min": clean_float(cost.min().item()),
            "cost_max": clean_float(cost.max().item()),
            "cost_mean": clean_float(cost.mean().item()),
            "random_embedding_dim": LATENT_DIM,
        }
        control = {
            "name": "random-init LeWM encoder",
            "status": "ok",
            "config": {
                "seed": int(args.c6_seed),
                "latent_dim": LATENT_DIM,
                "cost": "squared_l2",
                "lower_is_better": True,
                "checkpoint_weights_loaded": False,
            },
            "pixel_source": pixel_metadata,
            "encode_metadata": encode_metadata,
            "validation": validation,
            "metrics": metrics,
        }
        return {"C6": control}, [
            summary_row(control="C6", config="random_init_seed0", n_seeds=1, metrics=metrics)
        ], {
            **base_metadata,
            "status": "ok",
            "pixel_source": pixel_metadata,
            "encode_metadata": encode_metadata,
            "validation": validation,
        }
    except Exception as exc:  # noqa: BLE001
        blocker = f"{type(exc).__name__}: {exc}"
        control = blocked_c6_control(
            base_metadata=base_metadata,
            blocker=blocker,
            traceback_text=traceback.format_exc(),
        )
        return {"C6": control}, [
            summary_row(control="C6", config="random_init_seed0", n_seeds=0, status="blocked")
        ], {
            **base_metadata,
            "status": "blocked",
            "blocker": blocker,
            "traceback": traceback.format_exc(),
        }


def blocked_c6_control(
    *,
    base_metadata: dict[str, Any],
    blocker: str,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    control = {
        "name": "random-init LeWM encoder",
        "status": "blocked",
        "config": {
            "seed": int(base_metadata["random_seed"]),
            "latent_dim": LATENT_DIM,
            "cost": "squared_l2",
            "lower_is_better": True,
            "checkpoint_weights_loaded": False,
        },
        "blocker": blocker,
        "requirements": [
            "terminal pixels for all 8000 Track A action endpoints",
            "goal pixels for the 100 Track A goal rows",
            "PushT replay environment and stable-worldmodel dependencies",
            "LeWM config construction through scripts.verify_checkpoint.build_model_from_config",
        ],
        "metrics": None,
    }
    if traceback_text:
        control["traceback"] = traceback_text
    return control


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.dino_features = args.dino_features.expanduser().resolve()
    args.pixel_artifact = args.pixel_artifact.expanduser().resolve()
    args.config_path = args.config_path.expanduser().resolve()

    latent_artifact = load_latent_artifact(args.latent_artifact)
    common = build_common(latent_artifact)

    print("Computing C7 DINOv2 controls...")
    c7_controls, c7_rows, dino_metadata = compute_c7(
        dino_features_path=args.dino_features,
        latent_artifact=latent_artifact,
        common=common,
    )

    print("Computing C6 random-init LeWM control...")
    c6_controls, c6_rows, c6_metadata = compute_c6(
        args=args,
        latent_artifact=latent_artifact,
        common=common,
    )
    if c6_controls["C6"]["status"] == "blocked":
        print(f"C6 blocked: {c6_controls['C6']['blocker']}")

    controls = {**c6_controls, **c7_controls}
    summary_rows = c6_rows + c7_rows
    output = {
        "metadata": {
            "format": "stage1a_c6_c7_controls",
            "created_at": iso_now(),
            "latent_artifact": str(args.latent_artifact),
            "output": str(args.output),
            "n_records": EXPECTED_RECORDS,
            "target_metric": "C_real_state",
            "lower_cost_is_better": True,
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "anchor_definitions": ANCHOR_DEFINITIONS,
            "tie_rules": {
                "pairwise_accuracy": "Skip C_real_state ties; score tied control costs as 0.5 when C_real_state differs.",
                "topk_and_false_elite": "Ascending cost ranking with deterministic action_id tie-break.",
            },
            "topk_overlap_definition": "|topk(control) intersection topk(reference)| / k, averaged over pairs.",
            "c6": c6_metadata,
            "dino": dino_metadata,
            "replay_arg_import_error": getattr(args, "replay_arg_import_error", None),
        },
        "summary_table": summary_rows,
        "controls": controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_rows(summary_rows, title="Stage 1A C6/C7 summary")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
