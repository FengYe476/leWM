#!/usr/bin/env python3
"""Train Phase 2 P2-0 cost heads from extracted Track A latents."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cost_head_model import make_cost_head  # noqa: E402
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
)
from scripts.phase2.splits import (  # noqa: E402
    split1_random_holdout,
    split2_leave_one_cell_out,
    split3_hard_pair_holdout,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2" / "p2_0"


@dataclass
class RunMetrics:
    """Metrics from one train/val/test run."""

    split: int
    fold: str
    variant: str
    seed: int
    best_epoch: int
    train_examples: int
    val_examples: int
    test_examples: int
    train_triples: int
    val_triples: int
    test_triples: int
    test_loss: float | None
    test_pairwise_accuracy: float | None
    test_spearman_C_psi_vs_C_v1: float | None
    test_per_pair_spearman_mean: float | None
    test_per_pair_spearman_std: float | None
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--variant", choices=("small", "large"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument(
        "--predicted-artifact",
        type=Path,
        default=DEFAULT_PREDICTED_LATENT_ARTIFACT,
    )
    parser.add_argument(
        "--mixed-latents",
        action="store_true",
        help="Train on both encoder terminal latents and predictor rollout latents.",
    )
    parser.add_argument(
        "--temperature",
        action="store_true",
        help="Wrap the cost head with a learnable positive output temperature.",
    )
    parser.add_argument("--temperature-init", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
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


def jsonable(value):
    """Convert nested metrics to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float | None:
    """Compute Spearman correlation with a small dependency-free fallback."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return None
    rx = rankdata(x_arr)
    ry = rankdata(y_arr)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def make_dataset(
    examples: list[LatentExample],
    pair_ids: list[int],
    *,
    seed: int,
    max_triples_per_pair: int | None,
) -> PairwiseRankingDataset:
    """Build a ranking dataset for a pair-ID subset."""
    return PairwiseRankingDataset(
        examples,
        pair_ids=set(pair_ids),
        max_triples_per_pair=max_triples_per_pair,
        seed=seed,
    )


def examples_for_pair_ids(examples: list[LatentExample], pair_ids: list[int]) -> list[LatentExample]:
    """Filter latent examples to a sorted pair-ID subset."""
    allowed = set(pair_ids)
    return [example for example in examples if example.pair_id in allowed]


def make_loader(
    dataset: PairwiseRankingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic DataLoader for ranking triples."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ranking_triples,
        generator=generator,
    )


def ranking_loss(c_pos: torch.Tensor, c_neg: torch.Tensor, margin: float) -> torch.Tensor:
    """Return ranking hinge loss for predicted positive/negative costs."""
    return torch.relu(margin + c_pos - c_neg).mean()


def move_batch(batch: dict, device: str) -> dict:
    """Move tensor batch values to the target device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    margin: float,
) -> float | None:
    """Run one training epoch and return mean loss."""
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
    model,
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
    model,
    examples: list[LatentExample],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Predict scalar costs for individual latent examples."""
    if not examples:
        return np.empty((0,), dtype=np.float64)
    model.eval()
    preds = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        z = torch.as_tensor(np.stack([example.z for example in chunk]), dtype=torch.float32, device=device)
        z_g = torch.as_tensor(
            np.stack([example.z_g for example in chunk]),
            dtype=torch.float32,
            device=device,
        )
        preds.append(model(z, z_g).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def record_spearman_metrics(
    model,
    examples: list[LatentExample],
    *,
    batch_size: int,
    device: str,
) -> dict:
    """Compute overall and per-pair Spearman metrics against V1 costs."""
    preds = predict_record_costs(model, examples, batch_size=batch_size, device=device)
    labels = np.asarray([example.v1_cost for example in examples], dtype=np.float64)
    overall = spearman(preds, labels)
    by_pair: dict[int, dict[str, float | int | None]] = {}
    per_pair_values = []
    for pair_id in sorted({example.pair_id for example in examples}):
        indices = [idx for idx, example in enumerate(examples) if example.pair_id == pair_id]
        rho = spearman(preds[indices], labels[indices])
        by_pair[pair_id] = {"spearman": rho, "n_records": len(indices)}
        if rho is not None:
            per_pair_values.append(rho)
    per_pair = np.asarray(per_pair_values, dtype=np.float64)
    return {
        "spearman": overall,
        "per_pair_spearman": by_pair,
        "per_pair_spearman_mean": float(per_pair.mean()) if len(per_pair) else None,
        "per_pair_spearman_std": float(per_pair.std()) if len(per_pair) else None,
    }


def output_prefix(args: argparse.Namespace, *, split: int, fold: str) -> Path:
    """Return output path prefix for a run."""
    safe_fold = fold.replace("/", "_").replace("x", "x")
    return args.output_dir / f"split{split}_{safe_fold}_{args.variant}_seed{args.seed}"


def save_json(path: Path, data: dict) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n")


def run_one_fold(
    *,
    args: argparse.Namespace,
    artifact: dict,
    predicted_artifact: dict | None,
    split: int,
    fold: str,
    split_def: dict[str, list[int]],
) -> RunMetrics:
    """Train and evaluate one split/fold."""
    if args.mixed_latents:
        if predicted_artifact is None:
            raise ValueError("--mixed-latents requires --predicted-artifact")
        all_examples = mixed_latent_examples_from_artifacts(artifact, predicted_artifact)
    else:
        all_examples = latent_examples_from_artifact(artifact)
    train_examples = examples_for_pair_ids(all_examples, split_def["train_pair_ids"])
    val_examples = examples_for_pair_ids(all_examples, split_def["val_pair_ids"])
    test_examples = examples_for_pair_ids(all_examples, split_def["test_pair_ids"])

    train_dataset = make_dataset(
        train_examples,
        split_def["train_pair_ids"],
        seed=args.seed,
        max_triples_per_pair=args.max_triples_per_pair,
    )
    val_dataset = make_dataset(
        val_examples,
        split_def["val_pair_ids"],
        seed=args.seed,
        max_triples_per_pair=args.max_triples_per_pair,
    )
    test_dataset = make_dataset(
        test_examples,
        split_def["test_pair_ids"],
        seed=args.seed,
        max_triples_per_pair=args.max_triples_per_pair,
    )

    prefix = output_prefix(args, split=split, fold=fold)
    checkpoint_path = prefix.with_name(prefix.name + "_best.pt")
    log_path = prefix.with_name(prefix.name + "_train_log.json")
    metrics_path = prefix.with_name(prefix.name + "_test_metrics.json")

    model = make_cost_head(
        args.variant,
        temperature=args.temperature,
        temperature_init=args.temperature_init,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
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
        val_spearman = record_spearman_metrics(
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
            "val_spearman_C_psi_vs_C_v1": val_spearman,
        }
        log.append(epoch_record)
        print(
            f"[split={split} fold={fold}] epoch={epoch} "
            f"train_loss={train_loss} val_loss={val_pairwise['loss']} "
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
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": args.variant,
                    "split": split,
                    "fold": fold,
                    "epoch": epoch,
                    "val_pairwise_accuracy": val_accuracy,
                    "args": vars(args),
                    "temperature": args.temperature,
                    "temperature_init": args.temperature_init,
                    "mixed_latents": args.mixed_latents,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[split={split} fold={fold}] early stopping at epoch {epoch}")
                break

    save_json(
        log_path,
        {
            "args": vars(args),
            "split": split,
            "fold": fold,
            "train_pair_ids": split_def["train_pair_ids"],
            "val_pair_ids": split_def["val_pair_ids"],
            "test_pair_ids": split_def["test_pair_ids"],
            "epochs": log,
        },
    )

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_pairwise = evaluate_pairwise(
        model,
        test_dataset,
        batch_size=args.batch_size,
        device=args.device,
        margin=args.margin,
    )
    test_spearman = record_spearman_metrics(
        model,
        test_examples,
        batch_size=args.batch_size,
        device=args.device,
    )
    metrics = RunMetrics(
        split=split,
        fold=fold,
        variant=args.variant,
        seed=args.seed,
        best_epoch=best_epoch,
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        test_examples=len(test_examples),
        train_triples=len(train_dataset),
        val_triples=len(val_dataset),
        test_triples=len(test_dataset),
        test_loss=test_pairwise["loss"],
        test_pairwise_accuracy=test_pairwise["pairwise_accuracy"],
        test_spearman_C_psi_vs_C_v1=test_spearman["spearman"],
        test_per_pair_spearman_mean=test_spearman["per_pair_spearman_mean"],
        test_per_pair_spearman_std=test_spearman["per_pair_spearman_std"],
        checkpoint_path=str(checkpoint_path),
    )
    save_json(
        metrics_path,
        {
            **asdict(metrics),
            "test_per_pair_spearman": test_spearman["per_pair_spearman"],
        },
    )
    return metrics


def split_definitions(args: argparse.Namespace) -> dict[str, dict[str, list[int]]]:
    """Return fold definitions for the requested split."""
    if args.split == 1:
        return {"random_70_15_15": split1_random_holdout(seed=args.seed)}
    if args.split == 2:
        return split2_leave_one_cell_out(seed=args.seed)
    return {"all_fail_strong_rho": split3_hard_pair_holdout(seed=args.seed)}


def main() -> int:
    """Train one split or all split-2 folds."""
    args = parse_args()
    set_seed(args.seed)
    args.artifact = args.artifact.expanduser().resolve()
    args.predicted_artifact = args.predicted_artifact.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.device = resolve_device(args.device)
    artifact = load_latent_artifact(args.artifact)
    predicted_artifact = (
        load_predicted_latent_artifact(args.predicted_artifact)
        if args.mixed_latents
        else None
    )
    folds = split_definitions(args)

    print("== P2-0 cost-head training ==")
    print(f"artifact: {args.artifact}")
    print(f"output_dir: {args.output_dir}")
    print(f"split: {args.split}")
    print(f"variant: {args.variant}")
    print(f"mixed_latents: {args.mixed_latents}")
    print(f"temperature: {args.temperature}")
    if args.mixed_latents:
        print(f"predicted_artifact: {args.predicted_artifact}")
    print(f"device: {args.device}")
    print(f"folds: {list(folds)}")

    summaries = []
    for fold, split_def in folds.items():
        summaries.append(
            asdict(
                run_one_fold(
                    args=args,
                    artifact=artifact,
                    predicted_artifact=predicted_artifact,
                    split=args.split,
                    fold=fold,
                    split_def=split_def,
                )
            )
        )

    save_json(
        args.output_dir / f"split{args.split}_{args.variant}_seed{args.seed}_summary.json",
        {"runs": summaries},
    )
    print("== training complete ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
