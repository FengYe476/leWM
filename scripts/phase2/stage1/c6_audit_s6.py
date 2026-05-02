#!/usr/bin/env python3
"""C6 Audit S6: random CNN and ResNet visual encoders."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.stage1.c6_audit import (  # noqa: E402
    DEFAULT_PIXEL_ARTIFACT,
    add_replay_args_if_available,
    audit_anchor_definitions,
    build_common,
    compute_seed_metrics,
    seed_validation,
)
from scripts.phase2.stage1.c6_audit_s4_s5 import prepare_raw_pixels_once  # noqa: E402
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    EXPECTED_RECORDS,
    FALSE_ELITE_K,
    LATENT_DIM,
    TOPK_VALUES,
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


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "stage1" / "c6_audit" / "s6_random_arch_results.json"
DEFAULT_SEED = 0
DEFAULT_BATCH_SIZE = 64
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SmallRandomCNN(torch.nn.Module):
    """Minimal random CNN baseline with BatchNorm and 192-d output."""

    def __init__(self, output_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = torch.nn.Linear(192, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return self.projection(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args_if_available(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixel-artifact", type=Path, default=DEFAULT_PIXEL_ARTIFACT)
    parser.add_argument("--s6-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--s6-seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if int(args.s6_batch_size) <= 0:
        parser.error("--s6-batch-size must be positive")
    if int(args.s6_seed) != DEFAULT_SEED:
        parser.error("S6 is locked to --s6-seed 0")
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
    print("C6 Audit S6 random-architecture summary")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stage1a_helper_names() -> list[str]:
    return [
        compute_metrics.__name__,
        summary_row.__name__,
        load_latent_artifact.__name__,
        make_anchor_masks.__name__,
        run_single_metrics.__name__,
        squared_l2_torch.__name__,
    ]


def build_small_cnn(*, seed: int, device: str) -> torch.nn.Module:
    torch.manual_seed(int(seed))
    model = SmallRandomCNN(output_dim=LATENT_DIM)
    return model.to(device)


def build_resnet18(*, seed: int, device: str) -> torch.nn.Module:
    try:
        from torchvision.models import resnet18  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "S6 ResNet18 requires torchvision in the runtime environment."
        ) from exc
    torch.manual_seed(int(seed))
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, LATENT_DIM)
    return model.to(device)


def empty_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def preprocess_images(
    images: np.ndarray,
    *,
    architecture: str,
    device: str,
) -> torch.Tensor:
    if images.ndim != 4 or int(images.shape[-1]) != 3:
        raise ValueError(f"Expected raw HWC RGB images shaped [N,H,W,3], got {tuple(images.shape)}")
    tensor = torch.as_tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if tuple(tensor.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    if architecture == "resnet18":
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    elif architecture != "small_cnn":
        raise ValueError(f"Unknown architecture: {architecture}")
    return tensor.to(device=device, dtype=torch.float32)


@torch.inference_mode()
def encode_batch(
    *,
    model: torch.nn.Module,
    images: np.ndarray,
    architecture: str,
    device: str,
) -> torch.Tensor:
    batch = preprocess_images(images, architecture=architecture, device=device)
    emb = model(batch)
    if emb.ndim != 2:
        raise ValueError(f"{architecture} embeddings must be rank 2, got {tuple(emb.shape)}")
    if int(emb.shape[1]) != LATENT_DIM:
        raise ValueError(f"{architecture} embedding dim must be {LATENT_DIM}, got {int(emb.shape[1])}")
    if not torch.isfinite(emb).all():
        raise ValueError(f"{architecture} embeddings contain non-finite values")
    return emb.detach().cpu().to(dtype=torch.float32)


def encode_images(
    *,
    model: torch.nn.Module,
    images: np.ndarray,
    architecture: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, int(images.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(images.shape[0]))
        chunks.append(
            encode_batch(
                model=model,
                images=images[start:end],
                architecture=architecture,
                device=device,
            )
        )
    z = torch.cat(chunks, dim=0)
    expected_shape = (int(images.shape[0]), LATENT_DIM)
    if tuple(z.shape) != expected_shape:
        raise ValueError(f"{architecture} embeddings must be shaped {expected_shape}, got {tuple(z.shape)}")
    return z


def build_model(*, architecture: str, seed: int, device: str) -> torch.nn.Module:
    if architecture == "small_cnn":
        return build_small_cnn(seed=seed, device=device)
    if architecture == "resnet18":
        return build_resnet18(seed=seed, device=device)
    raise ValueError(f"Unknown architecture: {architecture}")


def encode_variant(
    *,
    architecture: str,
    mode: str,
    raw_terminal: np.ndarray,
    raw_goal: np.ndarray,
    seed: int,
    device: str,
    batch_size: int,
    pixel_source: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    started = time.time()
    model = build_model(architecture=architecture, seed=seed, device=device)
    if mode == "eval_mode":
        model.eval()
        train_mode_set_before_inference_mode = False
    elif mode == "train_mode":
        model.train()
        train_mode_set_before_inference_mode = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    z_terminal = encode_images(
        model=model,
        images=raw_terminal,
        architecture=architecture,
        device=device,
        batch_size=batch_size,
    )
    z_goal = encode_images(
        model=model,
        images=raw_goal,
        architecture=architecture,
        device=device,
        batch_size=batch_size,
    )
    metadata = {
        "seed": int(seed),
        "architecture": architecture,
        "mode": mode,
        "model_training_flag": bool(model.training),
        "train_mode_set_before_inference_mode": train_mode_set_before_inference_mode,
        "gradient_context": "torch.inference_mode() disables gradients only; model.train()/eval() controls BN behavior",
        "checkpoint_weights_loaded": False,
        "pretrained_weights_loaded": False,
        "device": device,
        "pixel_source": pixel_source,
        "input_pixel_format": "uint8 HWC RGB",
        "normalization": "[0,1]" if architecture == "small_cnn" else "[0,1] plus ImageNet mean/std",
        "resize": f"bilinear_to_{IMAGE_SIZE}x{IMAGE_SIZE}_if_needed",
        "batch_size": int(batch_size),
        "terminal_batches_encoded": int((int(raw_terminal.shape[0]) + int(batch_size) - 1) // int(batch_size)),
        "goal_batches_encoded": int((int(raw_goal.shape[0]) + int(batch_size) - 1) // int(batch_size)),
        "identical_raw_pixel_arrays": True,
        "wallclock_seconds": clean_float(time.time() - started),
    }
    del model
    empty_accelerator_cache()
    return z_terminal, z_goal, metadata


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
            "embedding_shape": [LATENT_DIM],
            "z_terminal_shape": list(z_terminal.shape),
            "z_goal_shape": list(z_goal.shape),
        },
        "validation": seed_validation(cost, z_terminal, z_goal),
        "metrics": compute_seed_metrics(costs=cost, common=common),
    }


def run_architecture(
    *,
    architecture: str,
    raw_terminal: np.ndarray,
    raw_goal: np.ndarray,
    seed: int,
    device: str,
    batch_size: int,
    pixel_source: str,
    common: dict[str, Any],
) -> dict[str, Any]:
    results = {}
    for mode in ("eval_mode", "train_mode"):
        print(f"Encoding S6 {architecture} {mode}...")
        z_terminal, z_goal, metadata = encode_variant(
            architecture=architecture,
            mode=mode,
            raw_terminal=raw_terminal,
            raw_goal=raw_goal,
            seed=seed,
            device=device,
            batch_size=batch_size,
            pixel_source=pixel_source,
        )
        results[mode] = result_block(
            z_terminal=z_terminal,
            z_goal=z_goal,
            encode_metadata=metadata,
            common=common,
        )
    return results


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pixel_artifact = args.pixel_artifact.expanduser().resolve()

    latent_artifact = load_latent_artifact(args.latent_artifact)
    common = build_common(latent_artifact)

    print("Preparing raw C6 S6 pixels once before all architecture variants...")
    raw_pixels = prepare_raw_pixels_once(args=args, latent_artifact=latent_artifact)
    device = resolve_device(str(args.device))
    args.device = device
    if int(raw_pixels.terminal.shape[0]) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} terminal images, found {int(raw_pixels.terminal.shape[0])}")
    if raw_pixels.terminal.shape != raw_pixels.goal.shape:
        raise ValueError(f"Terminal/goal image shape mismatch: {raw_pixels.terminal.shape} vs {raw_pixels.goal.shape}")

    small_cnn = run_architecture(
        architecture="small_cnn",
        raw_terminal=raw_pixels.terminal,
        raw_goal=raw_pixels.goal,
        seed=int(args.s6_seed),
        device=device,
        batch_size=int(args.s6_batch_size),
        pixel_source=str(raw_pixels.metadata.get("source", "unknown")),
        common=common,
    )
    resnet18 = run_architecture(
        architecture="resnet18",
        raw_terminal=raw_pixels.terminal,
        raw_goal=raw_pixels.goal,
        seed=int(args.s6_seed),
        device=device,
        batch_size=int(args.s6_batch_size),
        pixel_source=str(raw_pixels.metadata.get("source", "unknown")),
        common=common,
    )

    summary_rows = [
        summary_row(control="C6_S6", config="small_cnn_seed0_eval", n_seeds=1, metrics=small_cnn["eval_mode"]["metrics"]),
        summary_row(control="C6_S6", config="small_cnn_seed0_train", n_seeds=1, metrics=small_cnn["train_mode"]["metrics"]),
        summary_row(control="C6_S6", config="resnet18_seed0_eval", n_seeds=1, metrics=resnet18["eval_mode"]["metrics"]),
        summary_row(control="C6_S6", config="resnet18_seed0_train", n_seeds=1, metrics=resnet18["train_mode"]["metrics"]),
    ]

    output = {
        "metadata": {
            "format": "c6_audit_s6_random_arch_results",
            "created_at": iso_now(),
            "sub_experiment": "S6",
            "description": "Random CNN and ResNet visual encoder baselines for the C6 audit.",
            "latent_artifact": str(args.latent_artifact),
            "output": str(args.output),
            "pixel_artifact": str(args.pixel_artifact),
            "seed": int(args.s6_seed),
            "device": device,
            "batch_size": int(args.s6_batch_size),
            "n_records": EXPECTED_RECORDS,
            "latent_dim": LATENT_DIM,
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "anchor_definitions": audit_anchor_definitions(common["pair_ids"], common["cells"]),
            "pixel_preparation": raw_pixels.metadata,
            "preprocessing": {
                "input_pixel_format": "uint8 HWC RGB",
                "target_image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "small_cnn_normalization": "[0,1]",
                "resnet18_normalization": {
                    "scale": "[0,1]",
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                },
                "resize": "bilinear only when raw pixels are not already 224x224",
            },
            "architectures": {
                "small_cnn": "Conv-BN-ReLU blocks 3->32->64->128->192, adaptive average pooling, Linear(192, 192)",
                "resnet18": "torchvision.models.resnet18(weights=None) with fc replaced by Linear(in_features, 192)",
            },
            "checkpoint_weights_loaded": False,
            "pretrained_weights_loaded": False,
            "metric_helpers_reused": stage1a_helper_names(),
        },
        "small_cnn": small_cnn,
        "resnet18": resnet18,
        "summary_table": summary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_table(summary_rows)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
