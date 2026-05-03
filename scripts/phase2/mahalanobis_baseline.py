#!/usr/bin/env python3
"""Train Mahalanobis latent-cost baselines for Phase 2 Track C L1.

The P2-0 MLP cost head improved offline ranking but did not transfer to CEM
planning. This script tests simpler structured quadratic costs

    C(z, z_g) = (z - z_g)^T M (z - z_g)

on the headline Split 3 holdout using the mixed encoder/predictor latent
artifact. Training uses the same pairwise hinge objective as P2-0, while the
reported test metrics are computed on predictor latents only, because those are
the latents seen by CEM.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cost_head_model import LATENT_DIM  # noqa: E402
from scripts.phase2.dataloader import (  # noqa: E402
    DEFAULT_LATENT_ARTIFACT,
    DEFAULT_PREDICTED_LATENT_ARTIFACT,
    LatentExample,
    PairwiseRankingDataset,
    collate_ranking_triples,
    latent_examples_from_artifact,
    load_latent_artifact,
    load_predicted_latent_artifact,
    mixed_latent_examples_from_artifacts,
    validate_predicted_artifact_alignment,
)
from scripts.phase2.splits import split3_hard_pair_holdout  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "mahalanobis"
MAHALANOBIS_VARIANTS = ("diagonal", "lowrank", "full")


def inverse_softplus(value: float) -> float:
    """Return x such that softplus(x) approximately equals ``value``."""
    return math.log(math.expm1(value))


class DiagonalMahalanobis(nn.Module):
    """Positive diagonal Mahalanobis metric, ``M = diag(d)``."""

    variant = "diagonal"

    def __init__(self, latent_dim: int = LATENT_DIM, diag_init: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.raw_d = nn.Parameter(
            torch.full((self.latent_dim,), inverse_softplus(float(diag_init)))
        )

    @property
    def diag(self) -> torch.Tensor:
        """Return the positive diagonal entries."""
        return F.softplus(self.raw_d)

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Compute scalar Mahalanobis costs."""
        diff = z - z_g
        return torch.sum(self.diag * diff.square(), dim=-1)


class LowRankMahalanobis(nn.Module):
    """Diagonal plus rank-r PSD correction, ``M = diag(d) + L L^T``."""

    variant = "lowrank"

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        rank: int = 16,
        diag_init: float = 1.0,
        lowrank_init_std: float = 1e-2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        self.raw_d = nn.Parameter(
            torch.full((self.latent_dim,), inverse_softplus(float(diag_init)))
        )
        self.L = nn.Parameter(
            torch.randn(self.latent_dim, self.rank) * float(lowrank_init_std)
        )

    @property
    def diag(self) -> torch.Tensor:
        """Return the positive diagonal entries."""
        return F.softplus(self.raw_d)

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Compute scalar Mahalanobis costs."""
        diff = z - z_g
        diag_cost = torch.sum(self.diag * diff.square(), dim=-1)
        lowrank_cost = torch.sum((diff @ self.L).square(), dim=-1)
        return diag_cost + lowrank_cost


class FullPSDMahalanobis(nn.Module):
    """Full PSD Mahalanobis metric, ``M = A^T A``."""

    variant = "full"

    def __init__(self, latent_dim: int = LATENT_DIM, init_scale: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.A = nn.Parameter(torch.eye(self.latent_dim) * float(init_scale))

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Compute scalar Mahalanobis costs."""
        diff = z - z_g
        projected = diff @ self.A.T
        return torch.sum(projected.square(), dim=-1)


def make_mahalanobis(variant: str) -> nn.Module:
    """Instantiate a Mahalanobis baseline by name."""
    normalized = variant.lower()
    if normalized == "diagonal":
        return DiagonalMahalanobis()
    if normalized == "lowrank":
        return LowRankMahalanobis()
    if normalized == "full":
        return FullPSDMahalanobis()
    raise ValueError(f"Unknown Mahalanobis variant: {variant!r}")


def infer_mahalanobis_variant_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Infer the Mahalanobis variant from a checkpoint state dict."""
    keys = set(state_dict)
    if "A" in keys:
        return "full"
    if "L" in keys:
        return "lowrank"
    if "raw_d" in keys:
        return "diagonal"
    raise ValueError(f"Cannot infer Mahalanobis variant from state keys: {sorted(keys)}")


def load_mahalanobis_checkpoint(path: Path, *, device: str = "cpu") -> tuple[Path, nn.Module, dict]:
    """Load a Mahalanobis checkpoint from a file or output directory."""
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*_best.pt"))
        if not candidates:
            candidates = sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No Mahalanobis checkpoint found in {path}")
        path = candidates[0]
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    variant = checkpoint.get("variant") or infer_mahalanobis_variant_from_state_dict(state_dict)
    model = make_mahalanobis(variant).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return path, model, checkpoint


@dataclass
class MahalanobisMetrics:
    """Saved metrics for one Mahalanobis run."""

    variant: str
    seed: int
    best_epoch: int
    train_examples: int
    val_examples: int
    test_predicted_examples: int
    train_triples: int
    val_triples: int
    test_predicted_triples: int
    test_predicted_loss: float | None
    test_predicted_pairwise_accuracy: float | None
    test_predicted_spearman: float | None
    test_predicted_per_pair_spearman_mean: float | None
    test_predicted_per_pair_spearman_std: float | None
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MAHALANOBIS_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument(
        "--predicted-artifact",
        type=Path,
        default=DEFAULT_PREDICTED_LATENT_ARTIFACT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--max-triples-per-pair", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""
    if device != "auto":
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def jsonable(value: Any) -> Any:
    """Convert nested objects to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, data: dict) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n")


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks with tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman correlation without requiring scipy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def filter_examples(examples: list[LatentExample], pair_ids: list[int]) -> list[LatentExample]:
    """Return examples for a pair-ID subset."""
    allowed = set(pair_ids)
    return [example for example in examples if example.pair_id in allowed]


def make_dataset(
    examples: list[LatentExample],
    pair_ids: list[int],
    *,
    seed: int,
    max_triples_per_pair: int | None,
) -> PairwiseRankingDataset:
    """Build a ranking dataset."""
    return PairwiseRankingDataset(
        examples,
        pair_ids=set(pair_ids),
        max_triples_per_pair=max_triples_per_pair,
        seed=seed,
    )


def make_loader(
    dataset: PairwiseRankingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic DataLoader."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ranking_triples,
        generator=generator,
    )


def ranking_loss(c_pos: torch.Tensor, c_neg: torch.Tensor, margin: float) -> torch.Tensor:
    """Return pairwise ranking hinge loss."""
    return torch.relu(margin + c_pos - c_neg).mean()


def move_batch(batch: dict, device: str) -> dict:
    """Move tensor batch values to the target device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    margin: float,
) -> float | None:
    """Train for one epoch and return mean loss."""
    if len(loader.dataset) == 0:
        return None
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        c_pos = model(batch["z_pos"], batch["z_g"])
        c_neg = model(batch["z_neg"], batch["z_g"])
        loss = ranking_loss(c_pos, c_neg, margin)
        loss.backward()
        optimizer.step()
        batch_size = int(batch["z_pos"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    return total_loss / total_count if total_count else None


@torch.inference_mode()
def evaluate_pairwise(
    model: nn.Module,
    dataset: PairwiseRankingDataset,
    *,
    batch_size: int,
    device: str,
    margin: float,
) -> dict[str, float | None]:
    """Evaluate hinge loss and pairwise accuracy on ranking triples."""
    if len(dataset) == 0:
        return {"loss": None, "pairwise_accuracy": None}
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        c_pos = model(batch["z_pos"], batch["z_g"])
        c_neg = model(batch["z_neg"], batch["z_g"])
        loss = ranking_loss(c_pos, c_neg, margin)
        batch_size_actual = int(batch["z_pos"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size_actual
        total_correct += int((c_pos < c_neg).detach().cpu().sum())
        total_count += batch_size_actual
    return {
        "loss": total_loss / total_count if total_count else None,
        "pairwise_accuracy": total_correct / total_count if total_count else None,
    }


@torch.inference_mode()
def predict_record_costs(
    model: nn.Module,
    examples: list[LatentExample],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Predict costs for individual latent examples."""
    if not examples:
        return np.empty((0,), dtype=np.float64)
    model.eval()
    preds = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        z = torch.as_tensor(
            np.stack([example.z for example in chunk]),
            dtype=torch.float32,
            device=device,
        )
        z_g = torch.as_tensor(
            np.stack([example.z_g for example in chunk]),
            dtype=torch.float32,
            device=device,
        )
        preds.append(model(z, z_g).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def spearman_metrics(
    model: nn.Module,
    examples: list[LatentExample],
    *,
    batch_size: int,
    device: str,
) -> dict:
    """Compute global and per-pair Spearman metrics."""
    preds = predict_record_costs(model, examples, batch_size=batch_size, device=device)
    labels = np.asarray([example.v1_cost for example in examples], dtype=np.float64)
    pair_ids = np.asarray([example.pair_id for example in examples], dtype=np.int64)
    by_pair = {}
    per_pair = []
    for pair_id in sorted(set(int(pid) for pid in pair_ids)):
        mask = pair_ids == pair_id
        rho = spearman(preds[mask], labels[mask])
        by_pair[pair_id] = {
            "spearman": rho,
            "n_records": int(np.count_nonzero(mask)),
        }
        if rho is not None:
            per_pair.append(rho)
    values = np.asarray(per_pair, dtype=np.float64)
    return {
        "spearman": spearman(preds, labels),
        "per_pair_spearman": by_pair,
        "per_pair_spearman_mean": float(values.mean()) if len(values) else None,
        "per_pair_spearman_std": float(values.std()) if len(values) else None,
    }


def predictor_examples_from_artifacts(
    real_artifact: dict,
    pred_artifact: dict,
    *,
    pair_ids: list[int],
) -> list[LatentExample]:
    """Create predictor-latent examples aligned to the real artifact labels."""
    validate_predicted_artifact_alignment(real_artifact, pred_artifact)
    joined = dict(real_artifact)
    joined["z_predicted"] = pred_artifact["z_predicted"]
    return latent_examples_from_artifact(
        joined,
        pair_ids=set(pair_ids),
        latent_key="z_predicted",
        latent_type="predictor",
    )


def run(args: argparse.Namespace) -> dict:
    """Train and evaluate one Mahalanobis variant on Split 3."""
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    args.artifact = args.artifact.expanduser().resolve()
    args.predicted_artifact = args.predicted_artifact.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_artifact = load_latent_artifact(args.artifact)
    pred_artifact = load_predicted_latent_artifact(args.predicted_artifact)
    all_examples = mixed_latent_examples_from_artifacts(real_artifact, pred_artifact)
    split = split3_hard_pair_holdout(seed=args.seed)
    train_examples = filter_examples(all_examples, split["train_pair_ids"])
    val_examples = filter_examples(all_examples, split["val_pair_ids"])
    test_pred_examples = predictor_examples_from_artifacts(
        real_artifact,
        pred_artifact,
        pair_ids=split["test_pair_ids"],
    )

    train_dataset = make_dataset(
        train_examples,
        split["train_pair_ids"],
        seed=args.seed,
        max_triples_per_pair=args.max_triples_per_pair,
    )
    val_dataset = make_dataset(
        val_examples,
        split["val_pair_ids"],
        seed=args.seed,
        max_triples_per_pair=args.max_triples_per_pair,
    )
    test_pred_dataset = make_dataset(
        test_pred_examples,
        split["test_pair_ids"],
        seed=args.seed,
        max_triples_per_pair=args.max_triples_per_pair,
    )

    model = make_mahalanobis(args.variant).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    prefix = args.output_dir / f"split3_{args.variant}_mahalanobis_seed{args.seed}"
    checkpoint_path = prefix.with_name(prefix.name + "_best.pt")
    log_path = prefix.with_name(prefix.name + "_train_log.json")
    metrics_path = prefix.with_name(prefix.name + "_test_metrics.json")
    summary_path = args.output_dir / "summary.json"

    print("== Mahalanobis Split 3 training ==")
    print(f"variant: {args.variant}")
    print(f"device: {args.device}")
    print(f"artifact: {args.artifact}")
    print(f"predicted_artifact: {args.predicted_artifact}")
    print(f"output_dir: {args.output_dir}")
    print(
        "examples: "
        f"train={len(train_examples)} val={len(val_examples)} "
        f"test_predicted={len(test_pred_examples)}"
    )
    print(
        "triples: "
        f"train={len(train_dataset)} val={len(val_dataset)} "
        f"test_predicted={len(test_pred_dataset)}"
    )

    best_val_accuracy = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    log = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            margin=args.margin,
        )
        val_pairwise = evaluate_pairwise(
            model,
            val_dataset,
            batch_size=args.batch_size,
            device=args.device,
            margin=args.margin,
        )
        val_spearman = spearman_metrics(
            model,
            val_examples,
            batch_size=args.batch_size,
            device=args.device,
        )["spearman"]
        val_accuracy = val_pairwise["pairwise_accuracy"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_pairwise["loss"],
            "val_pairwise_accuracy": val_accuracy,
            "val_spearman": val_spearman,
        }
        log.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={train_loss} "
            f"val_loss={val_pairwise['loss']} "
            f"val_acc={val_accuracy} val_spearman={val_spearman}"
        )

        improved = False
        if val_accuracy is not None and val_accuracy > best_val_accuracy + 1e-12:
            improved = True
            best_val_accuracy = float(val_accuracy)
        elif best_epoch == 0:
            improved = True

        if improved:
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": "mahalanobis",
                    "variant": args.variant,
                    "split": 3,
                    "epoch": epoch,
                    "val_pairwise_accuracy": val_accuracy,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    save_json(
        log_path,
        {
            "args": vars(args),
            "split": 3,
            "train_pair_ids": split["train_pair_ids"],
            "val_pair_ids": split["val_pair_ids"],
            "test_pair_ids": split["test_pair_ids"],
            "epochs": log,
        },
    )

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_pairwise = evaluate_pairwise(
        model,
        test_pred_dataset,
        batch_size=args.batch_size,
        device=args.device,
        margin=args.margin,
    )
    test_spearman = spearman_metrics(
        model,
        test_pred_examples,
        batch_size=args.batch_size,
        device=args.device,
    )
    metrics = MahalanobisMetrics(
        variant=args.variant,
        seed=args.seed,
        best_epoch=best_epoch,
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        test_predicted_examples=len(test_pred_examples),
        train_triples=len(train_dataset),
        val_triples=len(val_dataset),
        test_predicted_triples=len(test_pred_dataset),
        test_predicted_loss=test_pairwise["loss"],
        test_predicted_pairwise_accuracy=test_pairwise["pairwise_accuracy"],
        test_predicted_spearman=test_spearman["spearman"],
        test_predicted_per_pair_spearman_mean=test_spearman["per_pair_spearman_mean"],
        test_predicted_per_pair_spearman_std=test_spearman["per_pair_spearman_std"],
        checkpoint_path=str(checkpoint_path),
    )
    metrics_payload = {
        **asdict(metrics),
        "test_predicted_per_pair_spearman": test_spearman["per_pair_spearman"],
    }
    save_json(metrics_path, metrics_payload)
    save_json(
        summary_path,
        {
            "args": vars(args),
            "metrics": metrics_payload,
            "checkpoint_path": str(checkpoint_path),
            "train_log_path": str(log_path),
            "test_metrics_path": str(metrics_path),
        },
    )

    print("== Mahalanobis test metrics on Split 3 predicted latents ==")
    print(f"variant: {args.variant}")
    print(f"PA_z_pred: {metrics.test_predicted_pairwise_accuracy}")
    print(f"Spearman_z_pred: {metrics.test_predicted_spearman}")
    print(f"checkpoint: {checkpoint_path}")
    return metrics_payload


def main() -> int:
    """CLI entry point."""
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
