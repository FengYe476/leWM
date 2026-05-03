#!/usr/bin/env python3
"""C6 Audit S2/S3: pre-projector and train-mode random-init LeWM checks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.stage1.c6_audit import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_PIXEL_ARTIFACT,
    add_replay_args_if_available,
    audit_anchor_definitions,
    build_common,
    build_random_model,
    compute_seed_metrics,
    prepare_pixels_once,
    seed_validation,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
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


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "stage1" / "c6_audit" / "s2_s3_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args_if_available(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixel-artifact", type=Path, default=DEFAULT_PIXEL_ARTIFACT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--c6-batch-size", type=int, default=64)
    args = parser.parse_args()
    if int(args.c6_batch_size) <= 0:
        parser.error("--c6-batch-size must be positive")
    if int(args.seed) != 0:
        parser.error("S2/S3 is locked to --seed 0")
    return args


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


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
    print("C6 Audit S2/S3 random-init LeWM summary")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


@torch.inference_mode()
def encode_projector_variants_batch(
    *,
    model: torch.nn.Module,
    pixels_t: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pixels_t.ndim < 5:
        raise ValueError(f"Expected pixels shaped [B, T, ...], got {tuple(pixels_t.shape)}")
    batch_size = int(pixels_t.shape[0])
    time_steps = int(pixels_t.shape[1])
    pixels = pixels_t.to(device=device, dtype=torch.float32)
    flat_pixels = pixels.reshape(batch_size * time_steps, *pixels.shape[2:])
    encoder_out = model.encoder(flat_pixels, interpolate_pos_encoding=True)
    cls = encoder_out.last_hidden_state[:, 0]
    post = model.projector(cls)
    pre = cls.reshape(batch_size, time_steps, -1)[:, -1, :]
    post = post.reshape(batch_size, time_steps, -1)[:, -1, :]
    for name, emb in (("pre_projector", pre), ("post_projector", post)):
        if emb.ndim != 2:
            raise ValueError(f"{name} embeddings must be rank 2, got {tuple(emb.shape)}")
        if not torch.isfinite(emb).all():
            raise ValueError(f"{name} embeddings contain non-finite values")
    return pre.detach().cpu().to(dtype=torch.float32), post.detach().cpu().to(dtype=torch.float32)


def goal_stack(
    *,
    latent_artifact: dict[str, Any],
    goal_emb_by_pair: dict[int, torch.Tensor],
) -> torch.Tensor:
    return torch.stack(
        [goal_emb_by_pair[int(pair_id)] for pair_id in latent_artifact["pair_id"].detach().cpu().tolist()]
    )


def encode_eval_projector_variants(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
    prepared_pixels: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = build_random_model(
        config_path=args.config_path,
        device=str(prepared_pixels.metadata["device"]),
        seed=int(args.seed),
    )
    model.eval()
    started = time.time()
    terminal_pre_chunks = []
    terminal_post_chunks = []
    for batch in prepared_pixels.terminal_batches:
        pre, post = encode_projector_variants_batch(
            model=model,
            pixels_t=batch,
            device=str(prepared_pixels.metadata["device"]),
        )
        terminal_pre_chunks.append(pre)
        terminal_post_chunks.append(post)
    z_terminal_pre = torch.cat(terminal_pre_chunks, dim=0)
    z_terminal_post = torch.cat(terminal_post_chunks, dim=0)

    goal_pre_by_pair: dict[int, torch.Tensor] = {}
    goal_post_by_pair: dict[int, torch.Tensor] = {}
    pair_offset = 0
    for batch in prepared_pixels.goal_batches:
        pre, post = encode_projector_variants_batch(
            model=model,
            pixels_t=batch,
            device=str(prepared_pixels.metadata["device"]),
        )
        batch_pair_ids = prepared_pixels.goal_pair_ids[pair_offset : pair_offset + int(pre.shape[0])]
        for local_idx, pair_id in enumerate(batch_pair_ids):
            goal_pre_by_pair[int(pair_id)] = pre[local_idx]
            goal_post_by_pair[int(pair_id)] = post[local_idx]
        pair_offset += int(pre.shape[0])

    features = {
        "pre_projector_terminal": z_terminal_pre,
        "pre_projector_goal": goal_stack(
            latent_artifact=latent_artifact,
            goal_emb_by_pair=goal_pre_by_pair,
        ),
        "post_projector_terminal": z_terminal_post,
        "post_projector_goal": goal_stack(
            latent_artifact=latent_artifact,
            goal_emb_by_pair=goal_post_by_pair,
        ),
    }
    metadata = {
        "seed": int(args.seed),
        "model_mode": "eval",
        "checkpoint_weights_loaded": False,
        "config_path": str(args.config_path),
        "device": prepared_pixels.metadata["device"],
        "pixel_source": prepared_pixels.metadata["source"],
        "encoder_call": "model.encoder(flat_pixels, interpolate_pos_encoding=True)",
        "pre_projector_feature": "encoder_out.last_hidden_state[:, 0]",
        "post_projector_feature": "model.projector(cls)",
        "terminal_batches_encoded": int(len(prepared_pixels.terminal_batches)),
        "goal_batches_encoded": int(len(prepared_pixels.goal_batches)),
        "identical_prepared_pixel_tensors": True,
        "wallclock_seconds": clean_float(time.time() - started),
    }
    del model
    empty_mps_cache()
    return features, metadata


def encode_train_post_projector(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
    prepared_pixels: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model = build_random_model(
        config_path=args.config_path,
        device=str(prepared_pixels.metadata["device"]),
        seed=int(args.seed),
    )
    model.train()
    torch.manual_seed(int(args.seed))
    started = time.time()
    terminal_chunks = []
    for batch in prepared_pixels.terminal_batches:
        _, post = encode_projector_variants_batch(
            model=model,
            pixels_t=batch,
            device=str(prepared_pixels.metadata["device"]),
        )
        terminal_chunks.append(post)
    z_terminal = torch.cat(terminal_chunks, dim=0)

    goal_post_by_pair: dict[int, torch.Tensor] = {}
    pair_offset = 0
    for batch in prepared_pixels.goal_batches:
        _, post = encode_projector_variants_batch(
            model=model,
            pixels_t=batch,
            device=str(prepared_pixels.metadata["device"]),
        )
        batch_pair_ids = prepared_pixels.goal_pair_ids[pair_offset : pair_offset + int(post.shape[0])]
        for local_idx, pair_id in enumerate(batch_pair_ids):
            goal_post_by_pair[int(pair_id)] = post[local_idx]
        pair_offset += int(post.shape[0])

    z_goal = goal_stack(
        latent_artifact=latent_artifact,
        goal_emb_by_pair=goal_post_by_pair,
    )
    metadata = {
        "seed": int(args.seed),
        "model_mode": "train",
        "checkpoint_weights_loaded": False,
        "config_path": str(args.config_path),
        "device": prepared_pixels.metadata["device"],
        "pixel_source": prepared_pixels.metadata["source"],
        "encoder_call": "model.encoder(flat_pixels, interpolate_pos_encoding=True)",
        "feature": "model.projector(encoder_out.last_hidden_state[:, 0])",
        "train_mode_set_before_inference_mode": True,
        "gradient_context": "torch.inference_mode() only disables gradients; model.train() controls BN/Dropout behavior",
        "terminal_batches_encoded": int(len(prepared_pixels.terminal_batches)),
        "goal_batches_encoded": int(len(prepared_pixels.goal_batches)),
        "identical_prepared_pixel_tensors": True,
        "wallclock_seconds": clean_float(time.time() - started),
    }
    del model
    empty_mps_cache()
    return z_terminal, z_goal, metadata


def empty_mps_cache() -> None:
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def result_block(
    *,
    z_terminal: torch.Tensor,
    z_goal: torch.Tensor,
    encode_metadata: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    cost = squared_l2_torch(z_terminal, z_goal)
    return {
        "encode_metadata": {
            **encode_metadata,
            "embedding_shape": [int(z_terminal.shape[1])],
            "z_terminal_shape": list(z_terminal.shape),
            "z_goal_shape": list(z_goal.shape),
        },
        "validation": seed_validation(cost, z_terminal, z_goal),
        "metrics": compute_seed_metrics(costs=cost, common=common),
    }


def reused_eval_block(post_projector: dict[str, Any]) -> dict[str, Any]:
    encode_metadata = dict(post_projector["encode_metadata"])
    encode_metadata["reused_from"] = "S2.post_projector"
    return {
        "encode_metadata": encode_metadata,
        "validation": post_projector["validation"],
        "metrics": post_projector["metrics"],
    }


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pixel_artifact = args.pixel_artifact.expanduser().resolve()
    args.config_path = args.config_path.expanduser().resolve()

    latent_artifact = load_latent_artifact(args.latent_artifact)
    common = build_common(latent_artifact)

    print("Preparing C6 S2/S3 pixels once before all variants...")
    prepared_pixels = prepare_pixels_once(args=args, latent_artifact=latent_artifact)

    print("Encoding S2 eval-mode pre/post projector features...")
    eval_features, eval_metadata = encode_eval_projector_variants(
        args=args,
        latent_artifact=latent_artifact,
        prepared_pixels=prepared_pixels,
    )
    s2_pre = result_block(
        z_terminal=eval_features["pre_projector_terminal"],
        z_goal=eval_features["pre_projector_goal"],
        encode_metadata={
            **eval_metadata,
            "sub_experiment": "S2",
            "variant": "pre_projector",
        },
        common=common,
    )
    s2_post = result_block(
        z_terminal=eval_features["post_projector_terminal"],
        z_goal=eval_features["post_projector_goal"],
        encode_metadata={
            **eval_metadata,
            "sub_experiment": "S2",
            "variant": "post_projector",
        },
        common=common,
    )

    print("Encoding S3 train-mode post-projector features...")
    train_terminal, train_goal, train_metadata = encode_train_post_projector(
        args=args,
        latent_artifact=latent_artifact,
        prepared_pixels=prepared_pixels,
    )
    s3_train = result_block(
        z_terminal=train_terminal,
        z_goal=train_goal,
        encode_metadata={
            **train_metadata,
            "sub_experiment": "S3",
            "variant": "train_mode",
        },
        common=common,
    )
    s3_eval = reused_eval_block(s2_post)
    s3_eval["encode_metadata"]["sub_experiment"] = "S3"
    s3_eval["encode_metadata"]["variant"] = "eval_mode"

    summary_rows = [
        summary_row(control="C6_S2", config="pre_projector_seed0_eval", n_seeds=1, metrics=s2_pre["metrics"]),
        summary_row(control="C6_S2", config="post_projector_seed0_eval", n_seeds=1, metrics=s2_post["metrics"]),
        summary_row(control="C6_S3", config="eval_mode_seed0", n_seeds=1, metrics=s3_eval["metrics"]),
        summary_row(control="C6_S3", config="train_mode_seed0", n_seeds=1, metrics=s3_train["metrics"]),
    ]

    output = {
        "metadata": {
            "format": "c6_audit_s2_s3_results",
            "created_at": iso_now(),
            "sub_experiments": ["S2", "S3"],
            "description": "C6 audit checks for pre-projector features and train-mode random-init LeWM behavior.",
            "latent_artifact": str(args.latent_artifact),
            "output": str(args.output),
            "config_path": str(args.config_path),
            "checkpoint_weights_loaded": False,
            "seed": int(args.seed),
            "n_records": EXPECTED_RECORDS,
            "latent_dim": LATENT_DIM,
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "anchor_definitions": audit_anchor_definitions(common["pair_ids"], common["cells"]),
            "pixel_preparation": prepared_pixels.metadata,
            "metric_helpers_reused": stage1a_helper_names(),
            "s2_pre_projector_definition": "ViT CLS token before projector: model.encoder(...).last_hidden_state[:, 0]",
            "s2_post_projector_definition": "Projector output matching JEPA.encode: model.projector(cls)",
            "s3_train_mode_note": "model.train() is set before torch.inference_mode(); inference_mode disables gradients only.",
        },
        "S2": {
            "pre_projector": s2_pre,
            "post_projector": s2_post,
        },
        "S3": {
            "eval_mode": s3_eval,
            "train_mode": s3_train,
        },
        "summary_table": summary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_table(summary_rows)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
