#!/usr/bin/env python3
"""Train a v3 hard-negative warp ensemble and evaluate rank averaging."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REVISION_DIR = PROJECT_ROOT / "scripts" / "revision"
for path in (PROJECT_ROOT, SCRIPTS_DIR, REVISION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_pusht_baseline import resolve_device  # noqa: E402
from method_cem_variants import (  # noqa: E402
    EPS,
    clean_float,
    deterministic_argmin,
    fmt_float,
    get_git_commit,
    scalar_summary,
    seconds_to_hms,
    spearman_corr,
)
from method_local_warp import (  # noqa: E402
    DEFAULT_POOL_ROOT,
    LATENT_DIM,
    load_pool_data,
    set_seed,
    warped_costs,
    write_json_atomic,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "method_warp_ensemble_v3.json"
DEFAULT_ENSEMBLE_DIR = PROJECT_ROOT / "results" / "revision" / "warp_ensemble_v3"
DEFAULT_V2_REFERENCE = PROJECT_ROOT / "results" / "revision" / "method_warp_ensemble_safe.json"
DEFAULT_SUBSET_PATH = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_PAIR_IDS = tuple(range(100))
FOLD_SIZE = 10
SUBSET_PRIORITY = ("invisible_quadrant", "sign_reversal", "latent_favorable", "v1_favorable", "ordinary")
V3_CONFIG = {
    "name": "v3_hard_top50",
    "hidden": 32,
    "dropout": 0.1,
    "scale": 0.05,
    "identity_reg": 3e-2,
    "epochs": 300,
    "lr": 1e-3,
    "hard_topk": 50,
    "pair_samples_per_pool": 500,
}
STRATEGIES = ("euclidean", "v3_rank_avg", "v3_rank_avg_top30pct", "v2_reference")


class LocalWarpV3(nn.Module):
    def __init__(self, dim: int = LATENT_DIM, hidden: int = 32, scale: float = 0.05, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.scale = float(scale)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.scale * self.net(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--v2-reference", type=Path, default=DEFAULT_V2_REFERENCE)
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    return parser.parse_args()


def make_warp(device: torch.device | str) -> LocalWarpV3:
    return LocalWarpV3(
        dim=LATENT_DIM,
        hidden=int(V3_CONFIG["hidden"]),
        scale=float(V3_CONFIG["scale"]),
        dropout=float(V3_CONFIG["dropout"]),
    ).to(device=device)


def cpu_state_dict(warp: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in warp.state_dict().items()}


def save_warp(path: Path, warp: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": cpu_state_dict(warp),
            "config": dict(V3_CONFIG),
            "metadata": dict(metadata),
        },
        path,
    )


def fold_pairs(pair_ids: list[int], fold_size: int = FOLD_SIZE) -> list[list[int]]:
    return [pair_ids[start : start + fold_size] for start in range(0, len(pair_ids), fold_size)]


def fold_for_pair(pair_id: int) -> int:
    return int(pair_id) // FOLD_SIZE


def zscore_torch(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if float(std.detach().cpu().item()) < EPS:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + EPS)


def train_warp_v3(
    *,
    warp: LocalWarpV3,
    train_data: list[dict[str, Any]],
    epochs: int,
    pair_samples_per_pool: int,
    hard_topk: int,
    lr: float,
    weight_decay: float,
    identity_reg: float,
    seed: int,
    log_every: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.Adam(warp.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    device = next(warp.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed) + 301)
    order_rng = np.random.default_rng(int(seed) + 719)
    train_log: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(epochs) + 1):
        warp.train()
        order = order_rng.permutation(len(train_data)).tolist()
        epoch_losses: list[float] = []
        epoch_rank_losses: list[float] = []
        epoch_reg_losses: list[float] = []
        epoch_pair_acc: list[float] = []
        for data_idx in order:
            item = train_data[int(data_idx)]
            z_pred = item["z_pred"]
            z_goal = item["z_goal"]
            c_real = item["c_real_state"]
            c_model = item["default_costs"]
            topk = min(int(hard_topk), int(z_pred.shape[0]))
            top_idx = torch.argsort(c_model)[:topk]
            z_top = z_pred[top_idx]
            c_real_top = c_real[top_idx]
            costs = zscore_torch(warped_costs(warp, z_top, z_goal))

            idx_i = torch.randint(0, topk, (int(pair_samples_per_pool),), generator=generator, device=device)
            idx_j = torch.randint(0, topk, (int(pair_samples_per_pool),), generator=generator, device=device)
            target = torch.sign(c_real_top[idx_j] - c_real_top[idx_i])
            mask = target != 0
            if bool(mask.any().detach().cpu().item()):
                cost_delta = costs[idx_j] - costs[idx_i]
                rank_loss = F.softplus(-target[mask] * cost_delta[mask]).mean()
                pair_acc = (target[mask] * cost_delta[mask] > 0).to(dtype=torch.float32).mean()
            else:
                rank_loss = costs.sum() * 0.0
                pair_acc = torch.tensor(float("nan"), device=device)

            displacement = (warp(z_top) - z_top).pow(2).mean() + (warp(z_goal.unsqueeze(0)) - z_goal).pow(2).mean()
            reg_loss = float(identity_reg) * displacement
            loss = rank_loss + reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_rank_losses.append(float(rank_loss.detach().cpu().item()))
            epoch_reg_losses.append(float(reg_loss.detach().cpu().item()))
            if torch.isfinite(pair_acc):
                epoch_pair_acc.append(float(pair_acc.detach().cpu().item()))

        log_row = {
            "epoch": int(epoch),
            "loss": clean_float(float(np.mean(epoch_losses))),
            "rank_loss": clean_float(float(np.mean(epoch_rank_losses))),
            "identity_reg_loss": clean_float(float(np.mean(epoch_reg_losses))),
            "sampled_pair_accuracy": clean_float(float(np.mean(epoch_pair_acc))) if epoch_pair_acc else None,
            "elapsed_seconds": clean_float(time.time() - started),
        }
        train_log.append(log_row)
        if epoch == 1 or epoch == int(epochs) or (int(log_every) > 0 and epoch % int(log_every) == 0):
            print(
                f"Epoch {epoch:03d}/{int(epochs)}: "
                f"loss={fmt_float(log_row['loss'], 4)} "
                f"rank={fmt_float(log_row['rank_loss'], 4)} "
                f"reg={fmt_float(log_row['identity_reg_loss'], 6)} "
                f"pair_acc={fmt_float(log_row['sampled_pair_accuracy'])}"
            )
    return train_log


@torch.no_grad()
def warp_cost_matrix(warps: list[nn.Module], z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    costs = []
    for warp in warps:
        warp.eval()
        costs.append(warped_costs(warp, z_pred, z_goal).detach())
    return torch.stack(costs, dim=0)


def rank_average_score(cost_matrix: torch.Tensor) -> torch.Tensor:
    ranks = torch.argsort(torch.argsort(cost_matrix, dim=1), dim=1).to(dtype=torch.float32)
    avg_rank = ranks.mean(dim=0)
    avg_cost = cost_matrix.mean(dim=0)
    norm_cost = (avg_cost - avg_cost.min()) / (avg_cost.max() - avg_cost.min() + EPS)
    return avg_rank + 1e-6 * norm_cost


def topk_indices(score: torch.Tensor, k: int) -> set[int]:
    idx = torch.topk(score, k=min(int(k), int(score.numel())), largest=False).indices
    return {int(item) for item in idx.detach().cpu().tolist()}


def metric_record(
    *,
    pair_id: int,
    strategy: str,
    score: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    selected_index: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rank1 = int(deterministic_argmin(score)) if selected_index is None else int(selected_index)
    oracle = int(deterministic_argmin(c_real_state))
    record = {
        "pair_id": int(pair_id),
        "strategy": str(strategy),
        "Rpool": spearman_corr(score, c_real_state),
        "rank1_candidate_index": rank1,
        "rank1_c_real": clean_float(float(c_real_state[rank1])),
        "rank1_success": bool(success[rank1]),
        "oracle_best_candidate_index": oracle,
        "oracle_c_real": clean_float(float(c_real_state[oracle])),
        "selection_regret": clean_float(float(c_real_state[rank1] - c_real_state[oracle])),
    }
    if extra:
        record.update(extra)
    return record


@torch.no_grad()
def evaluate_pair_v3(
    *,
    item: dict[str, Any],
    warps: list[nn.Module],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_id = int(item["pair_id"])
    heldout_idx = fold_for_pair(pair_id)
    train_indices = [idx for idx in range(len(warps)) if idx != heldout_idx]
    z_pred = item["z_pred"]
    z_goal = item["z_goal"]
    default = item["default_costs"].detach().cpu().numpy().astype(np.float64)
    c_real = item["c_real_state"].detach().cpu().numpy().astype(np.float64)
    success = item["success"].detach().cpu().numpy().astype(bool)
    cost_matrix = warp_cost_matrix(warps, z_pred, z_goal)
    train_score = rank_average_score(cost_matrix[train_indices])
    heldout_cost = cost_matrix[heldout_idx]
    heldout_rank1 = int(deterministic_argmin(heldout_cost.detach().cpu().numpy()))
    train_rank1 = int(deterministic_argmin(train_score.detach().cpu().numpy()))
    eu_rank1 = int(deterministic_argmin(default))
    agrees_top30pct = heldout_rank1 in topk_indices(train_score, 90)
    safe_score = train_score if agrees_top30pct else item["default_costs"].to(device=train_score.device, dtype=torch.float32)
    records = [
        metric_record(
            pair_id=pair_id,
            strategy="euclidean",
            score=default,
            c_real_state=c_real,
            success=success,
            extra={"heldout_fold": int(heldout_idx + 1), "used_fallback": False, "selected_source": "euclidean"},
        ),
        metric_record(
            pair_id=pair_id,
            strategy="v3_rank_avg",
            score=train_score.detach().cpu().numpy().astype(np.float64),
            c_real_state=c_real,
            success=success,
            extra={
                "heldout_fold": int(heldout_idx + 1),
                "heldout_rank1": heldout_rank1,
                "ensemble_rank1": train_rank1,
                "euclidean_rank1": eu_rank1,
                "used_fallback": False,
                "selected_source": "rank_avg",
            },
        ),
        metric_record(
            pair_id=pair_id,
            strategy="v3_rank_avg_top30pct",
            score=safe_score.detach().cpu().numpy().astype(np.float64),
            c_real_state=c_real,
            success=success,
            extra={
                "heldout_fold": int(heldout_idx + 1),
                "heldout_rank1": heldout_rank1,
                "ensemble_rank1": train_rank1,
                "euclidean_rank1": eu_rank1,
                "used_fallback": not bool(agrees_top30pct),
                "selected_source": "rank_avg" if agrees_top30pct else "euclidean_fallback",
            },
        ),
    ]
    spotlight = {
        "pair_id": pair_id,
        "heldout_fold": int(heldout_idx + 1),
        "heldout_rank1": heldout_rank1,
        "ensemble_rank1": train_rank1,
        "euclidean_rank1": eu_rank1,
        "heldout_in_top30pct": bool(agrees_top30pct),
        "records": records,
    }
    return records, spotlight


def load_subset_map(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing subset metadata: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in data.get("per_pair", []):
        pair_id = int(row["pair_id"])
        subsets = list(row.get("subsets", []))
        primary = "ordinary"
        for candidate in SUBSET_PRIORITY:
            if candidate in subsets:
                primary = candidate
                break
        out[pair_id] = {"subsets": subsets, "primary_subset": primary}
    return out


def load_v2_reference(path: Path, pair_ids: list[int]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing v2 reference: {path}. Run method_warp_ensemble_safe.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    wanted = set(int(pair_id) for pair_id in pair_ids)
    records = []
    for row in data.get("posthoc_records", []):
        if str(row.get("strategy")) != "safe_top30pct" or int(row["pair_id"]) not in wanted:
            continue
        copied = dict(row)
        copied["strategy"] = "v2_reference"
        records.append(copied)
    if len(records) != len(wanted):
        raise RuntimeError(f"Expected {len(wanted)} v2 reference rows, found {len(records)}")
    return records


def aggregate_records(records: list[dict[str, Any]], strategies: tuple[str, ...]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_strategy.setdefault(str(record["strategy"]), []).append(record)
    summary = {}
    for strategy in strategies:
        rows = by_strategy.get(strategy, [])
        fallback_count = sum(1 for row in rows if bool(row.get("used_fallback", False)))
        summary[strategy] = {
            "n": int(len(rows)),
            "Rpool": scalar_summary([row.get("Rpool") for row in rows]),
            "rank1_success_rate": scalar_summary([row.get("rank1_success") for row in rows]),
            "rank1_c_real": scalar_summary([row.get("rank1_c_real") for row in rows]),
            "selection_regret": scalar_summary([row.get("selection_regret") for row in rows]),
            "fallback_count": int(fallback_count),
            "fallback_rate": clean_float(float(fallback_count) / float(len(rows))) if rows else None,
        }
    return summary


def subset_breakdown(records: list[dict[str, Any]], subset_map: dict[int, dict[str, Any]]) -> dict[str, Any]:
    subsets = list(SUBSET_PRIORITY)
    out = {}
    for subset in subsets:
        subset_records = [
            record for record in records if subset_map.get(int(record["pair_id"]), {}).get("primary_subset") == subset
        ]
        if subset_records:
            out[subset] = aggregate_records(subset_records, STRATEGIES)
    return out


def print_summary(title: str, summary: dict[str, Any]) -> None:
    rows = []
    for strategy in STRATEGIES:
        stats = summary[strategy]
        rows.append(
            [
                strategy,
                stats["n"],
                fmt_float(stats["Rpool"]["mean"]),
                fmt_float(stats["rank1_success_rate"]["mean"]),
                fmt_float(stats["selection_regret"]["mean"]),
                fmt_float(stats["rank1_c_real"]["mean"]),
                stats["fallback_count"],
                fmt_float(stats["fallback_rate"]),
            ]
        )
    print(f"\n{title}")
    print(
        tabulate(
            rows,
            headers=["Strategy", "n", "mean_Rpool", "success_rate", "mean_regret", "mean_c_real", "fallbacks", "fallback_rate"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def print_subset_summary(breakdown: dict[str, Any]) -> None:
    rows = []
    for subset, summary in breakdown.items():
        for strategy in STRATEGIES:
            stats = summary[strategy]
            rows.append(
                [
                    subset,
                    strategy,
                    stats["n"],
                    fmt_float(stats["Rpool"]["mean"]),
                    fmt_float(stats["rank1_success_rate"]["mean"]),
                    fmt_float(stats["selection_regret"]["mean"]),
                    stats["fallback_count"],
                ]
            )
    print("\nPer-subset breakdown")
    print(
        tabulate(
            rows,
            headers=["Subset", "Strategy", "n", "mean_Rpool", "success_rate", "mean_regret", "fallbacks"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def print_pair30(pair30_records: list[dict[str, Any]]) -> None:
    rows = []
    for record in pair30_records:
        rows.append(
            [
                record["strategy"],
                record["rank1_candidate_index"],
                fmt_float(record["Rpool"]),
                str(record["rank1_success"]),
                fmt_float(record["selection_regret"]),
                fmt_float(record["rank1_c_real"]),
                str(record.get("used_fallback", "")),
                record.get("selected_source", ""),
                record.get("heldout_rank1", ""),
                record.get("ensemble_rank1", ""),
            ]
        )
    print("\nPair 30 spotlight")
    print(
        tabulate(
            rows,
            headers=["Strategy", "rank1", "Rpool", "success", "regret", "c_real", "fallback", "source", "heldout", "ensemble"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pool_root = args.pool_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.ensemble_dir = args.ensemble_dir.expanduser().resolve()
    args.v2_reference = args.v2_reference.expanduser().resolve()
    args.subset_path = args.subset_path.expanduser().resolve()
    args.device = resolve_device(str(args.device))
    set_seed(int(args.seed))
    started = time.time()

    pair_ids = list(DEFAULT_PAIR_IDS)
    if args.max_pairs is not None:
        pair_ids = pair_ids[: int(args.max_pairs)]
    folds = fold_pairs(pair_ids, FOLD_SIZE)
    epochs = int(args.epochs_override) if args.epochs_override is not None else int(V3_CONFIG["epochs"])
    config = dict(V3_CONFIG, epochs=epochs)

    print("== Warp Ensemble v3: hard negatives + rank averaging ==")
    print(f"pool_root: {args.pool_root}")
    print(f"output: {args.output}")
    print(f"ensemble_dir: {args.ensemble_dir}")
    print(f"v2_reference: {args.v2_reference}")
    print(f"device: {args.device}")
    print(f"pairs: n={len(pair_ids)} folds={len(folds)}")
    print(f"config: {config}")

    loaded = load_pool_data(pool_root=args.pool_root, pair_ids=pair_ids, device=args.device)
    by_pair = {int(item["pair_id"]): item for item in loaded}
    subset_map = load_subset_map(args.subset_path)
    payload: dict[str, Any] = {
        "metadata": {
            "format": "pusht_warp_ensemble_v3",
            "created_at_unix": clean_float(time.time()),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "pool_root": str(args.pool_root),
            "output": str(args.output),
            "ensemble_dir": str(args.ensemble_dir),
            "v2_reference": str(args.v2_reference),
            "subset_path": str(args.subset_path),
            "device": str(args.device),
            "seed": int(args.seed),
            "pair_ids": pair_ids,
            "folds": folds,
            "config": config,
        },
        "fold_training": [],
        "posthoc_records": [],
        "posthoc_summary": {},
        "subset_breakdown": {},
        "pair30_spotlight": [],
    }

    warps: list[nn.Module] = []
    fold_training: list[dict[str, Any]] = []
    for fold_idx, heldout_pairs in enumerate(folds, start=1):
        fold_started = time.time()
        train_pairs = [pair_id for pair_id in pair_ids if pair_id not in set(heldout_pairs)]
        train_data = [by_pair[pair_id] for pair_id in train_pairs]
        print(f"\n[Fold {fold_idx}/{len(folds)}] train_n={len(train_pairs)} heldout={heldout_pairs[0]}-{heldout_pairs[-1]}")
        warp = make_warp(args.device)
        train_log = train_warp_v3(
            warp=warp,
            train_data=train_data,
            epochs=epochs,
            pair_samples_per_pool=int(V3_CONFIG["pair_samples_per_pool"]),
            hard_topk=int(V3_CONFIG["hard_topk"]),
            lr=float(V3_CONFIG["lr"]),
            weight_decay=float(args.weight_decay),
            identity_reg=float(V3_CONFIG["identity_reg"]),
            seed=int(args.seed) + fold_idx * 1000,
            log_every=int(args.log_every),
        )
        model_path = args.ensemble_dir / f"warp_fold_{fold_idx}.pt"
        final_log = train_log[-1]
        row = {
            "fold": int(fold_idx),
            "model_path": str(model_path),
            "train_pairs": train_pairs,
            "heldout_pairs": heldout_pairs,
            "final_loss": final_log.get("loss"),
            "final_rank_loss": final_log.get("rank_loss"),
            "final_identity_reg_loss": final_log.get("identity_reg_loss"),
            "final_sampled_pair_accuracy": final_log.get("sampled_pair_accuracy"),
            "elapsed_seconds": clean_float(time.time() - fold_started),
            "train_log_tail": train_log[-5:],
        }
        save_warp(model_path, warp, row)
        warps.append(warp)
        fold_training.append(row)
        payload["fold_training"] = fold_training
        payload["metadata"]["wallclock_seconds_so_far"] = clean_float(time.time() - started)
        write_json_atomic(args.output, payload)
        print(
            f"[Fold done] {fold_idx} loss={fmt_float(row['final_loss'], 4)} "
            f"rank={fmt_float(row['final_rank_loss'], 4)} "
            f"pair_acc={fmt_float(row['final_sampled_pair_accuracy'])} "
            f"elapsed={seconds_to_hms(row['elapsed_seconds'] or 0.0)}"
        )

    posthoc_records: list[dict[str, Any]] = []
    pair30_records: list[dict[str, Any]] = []
    print("\n== Post-hoc v3 evaluation ==")
    for idx, pair_id in enumerate(pair_ids, start=1):
        records, spotlight = evaluate_pair_v3(item=by_pair[pair_id], warps=warps)
        posthoc_records.extend(records)
        if pair_id == 30:
            pair30_records.extend(records)
        if idx % 10 == 0 or idx == len(pair_ids):
            print(f"  evaluated {idx}/{len(pair_ids)} pairs")

    v2_records = load_v2_reference(args.v2_reference, pair_ids)
    posthoc_records.extend(v2_records)
    pair30_records.extend([record for record in v2_records if int(record["pair_id"]) == 30])
    posthoc_summary = aggregate_records(posthoc_records, STRATEGIES)
    breakdown = subset_breakdown(posthoc_records, subset_map)
    print_summary("Post-hoc 100-pair v3 summary", posthoc_summary)
    print_subset_summary(breakdown)
    print_pair30(pair30_records)

    payload["posthoc_records"] = posthoc_records
    payload["posthoc_summary"] = posthoc_summary
    payload["subset_breakdown"] = breakdown
    payload["pair30_spotlight"] = pair30_records
    payload["metadata"]["wallclock_seconds"] = clean_float(time.time() - started)
    write_json_atomic(args.output, payload)
    print(f"\nWrote summary: {args.output}")
    print(f"Total elapsed: {seconds_to_hms(time.time() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
