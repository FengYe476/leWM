#!/usr/bin/env python3
"""Strict held-out Warp Ensemble Selection evaluation and baselines.

This script trains WES only on PushT pairs 0--69, evaluates post-hoc selection
on pairs 70--99 that are unseen by every warp, and then runs default CEM on the
same held-out pairs with Euclidean versus WES final selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REVISION_DIR = PROJECT_ROOT / "scripts" / "revision"
for path in (PROJECT_ROOT, SCRIPTS_DIR, REVISION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import prepare_pair_info  # noqa: E402
from method_cem_variants import (  # noqa: E402
    EPS,
    clean_float,
    deterministic_argmin,
    fmt_float,
    get_git_commit,
    jsonable,
    load_pair_rows_direct,
    scalar_summary,
    seconds_to_hms,
    spearman_corr,
)
from method_local_warp import (  # noqa: E402
    DEFAULT_POOL_ROOT,
    LATENT_DIM,
    parse_int_list,
    set_seed,
    warped_costs,
)
from method_warp_ensemble_v3 import (  # noqa: E402
    LocalWarpV3,
    V3_CONFIG,
    save_warp,
    train_warp_v3,
    warp_cost_matrix,
    zscore_torch,
)
from method_warp_ensemble_v3_cem_full import rank_average_score, run_cem, write_json_atomic  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    NUM_SAMPLES,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.projected_cem import (  # noqa: E402
    blocked_batch_to_raw_fast,
    score_raw_actions,
)


TRAIN_PAIRS = tuple(range(70))
TEST_PAIRS = tuple(range(70, 100))
FOLD_SIZE = 7
N_FOLDS = 10
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "wes_strict_holdout.json"
DEFAULT_STRICT_DIR = PROJECT_ROOT / "results" / "revision" / "warp_ensemble_strict"
DEFAULT_CEM_POOL_ROOT = PROJECT_ROOT / "results" / "revision" / "wes_strict_cem_pools"
DEFAULT_SEEDS = (0, 1, 2)
POSTHOC_STRATEGIES = (
    "euclidean",
    "wes_rank_avg",
    "wes_vote_top10",
    "single_warp_best",
    "mahalanobis",
    "linear_metric",
    "scalar_mlp",
    "random_top30",
    "v1_oracle",
)
CEM_VARIANTS = ("default_cem", "wes_posthoc")
BOOTSTRAP_SAMPLES = 10_000


class LinearMetric(nn.Module):
    """A PSD linear metric: cost(z, g) = ||(z-g)(I + scale W)||^2."""

    def __init__(self, dim: int = LATENT_DIM, scale: float = 0.05):
        super().__init__()
        self.dim = int(dim)
        self.scale = float(scale)
        self.W = nn.Parameter(torch.zeros(self.dim, self.dim))

    def transformed_diff(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        while z_goal.ndim < z_pred.ndim:
            z_goal = z_goal.unsqueeze(-2)
        diff = z_pred - z_goal.expand_as(z_pred)
        return diff + self.scale * (diff @ self.W)

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        diff = self.transformed_diff(z_pred, z_goal)
        return torch.sum(diff.pow(2), dim=-1)

    def regularization(self) -> torch.Tensor:
        return self.W.pow(2).mean()


class ScalarCostMLP(nn.Module):
    """Small scalar cost head over (z, z_goal) used as a baseline."""

    def __init__(self, dim: int = LATENT_DIM, hidden: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.dim * 4, self.hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden, 1),
        )
        # Unlike residual warps, a scalar head has no identity cost path.  A
        # tiny non-zero final layer avoids a constant score vector, which would
        # make the z-scored ranking loss flat at initialization.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        while z_goal.ndim < z_pred.ndim:
            z_goal = z_goal.unsqueeze(-2)
        goal = z_goal.expand_as(z_pred)
        diff = z_pred - goal
        features = torch.cat([z_pred, goal, diff, diff.pow(2)], dim=-1)
        return self.net(features).squeeze(-1)

    def regularization(self) -> torch.Tensor:
        return sum(param.pow(2).mean() for param in self.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-dir", type=Path, default=DEFAULT_STRICT_DIR)
    parser.add_argument("--cem-pool-root", type=Path, default=DEFAULT_CEM_POOL_ROOT)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-cem", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def fold_splits() -> list[list[int]]:
    return [list(range(start, start + FOLD_SIZE)) for start in range(0, len(TRAIN_PAIRS), FOLD_SIZE)]


def make_warp(device: torch.device | str) -> LocalWarpV3:
    return LocalWarpV3(
        dim=LATENT_DIM,
        hidden=int(V3_CONFIG["hidden"]),
        scale=float(V3_CONFIG["scale"]),
        dropout=float(V3_CONFIG["dropout"]),
    ).to(device=device)


def load_warp(path: Path, *, device: torch.device | str) -> LocalWarpV3:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    warp = LocalWarpV3(
        dim=LATENT_DIM,
        hidden=int(config.get("hidden", V3_CONFIG["hidden"])),
        scale=float(config.get("scale", V3_CONFIG["scale"])),
        dropout=float(config.get("dropout", V3_CONFIG["dropout"])),
    ).to(device=device)
    warp.load_state_dict(payload["model_state_dict"])
    warp.eval()
    return warp


def pool_path(pair_id: int, pool_root: Path) -> Path:
    return pool_root / f"pair_{int(pair_id)}.pt"


def load_pool_items(pair_ids: tuple[int, ...], *, pool_root: Path, device: torch.device | str) -> list[dict[str, Any]]:
    items = []
    for pair_id in pair_ids:
        path = pool_path(pair_id, pool_root)
        if not path.exists():
            raise FileNotFoundError(f"Missing pool for pair {pair_id}: {path}")
        pool = torch.load(path, map_location="cpu", weights_only=False)
        items.append(
            {
                "pair_id": int(pair_id),
                "path": str(path),
                "z_pred": pool["z_pred"].detach().to(device=device, dtype=torch.float32),
                "z_goal": pool["z_goal"].detach().to(device=device, dtype=torch.float32),
                "c_real_state": pool["c_real_state"].detach().to(device=device, dtype=torch.float32),
                "default_costs": pool["default_costs"].detach().to(device=device, dtype=torch.float32),
                "v1_hinge_costs": pool["v1_hinge_costs"].detach().to(device=device, dtype=torch.float32),
                "success": pool["success"].detach().cpu().to(dtype=torch.bool),
            }
        )
    return items


@torch.no_grad()
def validation_ranking_loss(cost_fn, data: list[dict[str, Any]], *, hard_topk: int) -> dict[str, Any]:
    losses: list[float] = []
    accs: list[float] = []
    for item in data:
        z_pred = item["z_pred"]
        z_goal = item["z_goal"]
        c_real = item["c_real_state"]
        c_model = item["default_costs"]
        topk = min(int(hard_topk), int(z_pred.shape[0]))
        top_idx = torch.argsort(c_model)[:topk]
        costs = zscore_torch(cost_fn(z_pred[top_idx], z_goal))
        c_top = c_real[top_idx]
        i_idx, j_idx = torch.meshgrid(
            torch.arange(topk, device=costs.device),
            torch.arange(topk, device=costs.device),
            indexing="ij",
        )
        target = torch.sign(c_top[j_idx.reshape(-1)] - c_top[i_idx.reshape(-1)])
        mask = target != 0
        if not bool(mask.any().detach().cpu().item()):
            continue
        delta = costs[j_idx.reshape(-1)] - costs[i_idx.reshape(-1)]
        losses.append(float(F.softplus(-target[mask] * delta[mask]).mean().detach().cpu().item()))
        accs.append(float((target[mask] * delta[mask] > 0).to(dtype=torch.float32).mean().detach().cpu().item()))
    return {
        "loss": clean_float(float(np.mean(losses))) if losses else None,
        "pair_accuracy": clean_float(float(np.mean(accs))) if accs else None,
        "n_pools": int(len(data)),
    }


def train_metric_model(
    *,
    model: nn.Module,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed) + 917)
    order_rng = np.random.default_rng(int(seed) + 431)
    log: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        rank_losses: list[float] = []
        reg_losses: list[float] = []
        accs: list[float] = []
        for data_idx in order_rng.permutation(len(train_data)).tolist():
            item = train_data[int(data_idx)]
            z_pred = item["z_pred"]
            z_goal = item["z_goal"]
            c_real = item["c_real_state"]
            c_model = item["default_costs"]
            topk = min(int(hard_topk), int(z_pred.shape[0]))
            top_idx = torch.argsort(c_model)[:topk]
            costs = zscore_torch(model(z_pred[top_idx], z_goal))
            c_top = c_real[top_idx]
            idx_i = torch.randint(0, topk, (int(pair_samples_per_pool),), generator=generator, device=device)
            idx_j = torch.randint(0, topk, (int(pair_samples_per_pool),), generator=generator, device=device)
            target = torch.sign(c_top[idx_j] - c_top[idx_i])
            mask = target != 0
            if bool(mask.any().detach().cpu().item()):
                delta = costs[idx_j] - costs[idx_i]
                rank_loss = F.softplus(-target[mask] * delta[mask]).mean()
                pair_acc = (target[mask] * delta[mask] > 0).to(dtype=torch.float32).mean()
            else:
                rank_loss = costs.sum() * 0.0
                pair_acc = torch.tensor(float("nan"), device=device)
            reg_loss = float(identity_reg) * model.regularization()
            loss = rank_loss + reg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            rank_losses.append(float(rank_loss.detach().cpu().item()))
            reg_losses.append(float(reg_loss.detach().cpu().item()))
            if torch.isfinite(pair_acc):
                accs.append(float(pair_acc.detach().cpu().item()))
        row = {
            "epoch": int(epoch),
            "loss": clean_float(float(np.mean(losses))),
            "rank_loss": clean_float(float(np.mean(rank_losses))),
            "reg_loss": clean_float(float(np.mean(reg_losses))),
            "sampled_pair_accuracy": clean_float(float(np.mean(accs))) if accs else None,
            "elapsed_seconds": clean_float(time.time() - started),
        }
        log.append(row)
        if epoch == 1 or epoch == int(epochs) or (int(log_every) > 0 and epoch % int(log_every) == 0):
            print(
                f"Epoch {epoch:03d}/{int(epochs)}: loss={fmt_float(row['loss'], 4)} "
                f"rank={fmt_float(row['rank_loss'], 4)} reg={fmt_float(row['reg_loss'], 6)} "
                f"pair_acc={fmt_float(row['sampled_pair_accuracy'])}"
            )
    model.eval()
    return log


def save_model(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "metadata": dict(metadata),
        },
        path,
    )


def load_linear_metric(path: Path, *, device: torch.device | str) -> LinearMetric:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = LinearMetric().to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def load_scalar_mlp(path: Path, *, device: torch.device | str) -> ScalarCostMLP:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ScalarCostMLP().to(device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def mahalanobis_cost(z_pred: torch.Tensor, z_goal: torch.Tensor) -> np.ndarray:
    z = z_pred.detach().cpu().numpy().astype(np.float64)
    g = z_goal.detach().cpu().numpy().astype(np.float64)
    cov = np.cov(z, rowvar=False)
    reg = 0.01 * float(np.trace(cov)) / float(cov.shape[0]) + 1e-6
    inv = np.linalg.pinv(cov + reg * np.eye(cov.shape[0], dtype=np.float64), hermitian=True)
    diff = z - g[None, :]
    return np.einsum("ij,jk,ik->i", diff, inv, diff).astype(np.float64)


def numpy_cost(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def metric_record(
    *,
    pair_id: int,
    strategy: str,
    score: np.ndarray,
    c_real: np.ndarray,
    success: np.ndarray,
    selected_index: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rank1 = int(deterministic_argmin(score)) if selected_index is None else int(selected_index)
    oracle = int(deterministic_argmin(c_real))
    row = {
        "pair_id": int(pair_id),
        "strategy": str(strategy),
        "Rpool": spearman_corr(score, c_real),
        "rank1_candidate_index": int(rank1),
        "rank1_c_real": clean_float(float(c_real[rank1])),
        "rank1_success": bool(success[rank1]),
        "oracle_best_candidate_index": int(oracle),
        "oracle_c_real": clean_float(float(c_real[oracle])),
        "selection_regret": clean_float(float(c_real[rank1] - c_real[oracle])),
    }
    if extra:
        row.update(extra)
    return row


@torch.no_grad()
def evaluate_posthoc_pair(
    *,
    item: dict[str, Any],
    warps: list[nn.Module],
    best_warp_index: int,
    linear_metric: LinearMetric,
    scalar_mlp: ScalarCostMLP,
    random_seed: int,
) -> list[dict[str, Any]]:
    pair_id = int(item["pair_id"])
    z_pred = item["z_pred"]
    z_goal = item["z_goal"]
    c_real = numpy_cost(item["c_real_state"])
    success = item["success"].detach().cpu().numpy().astype(bool)
    default = numpy_cost(item["default_costs"])
    v1 = numpy_cost(item["v1_hinge_costs"])
    cost_matrix = warp_cost_matrix(warps, z_pred, z_goal)
    rank_avg = rank_average_score(cost_matrix).detach().cpu().numpy().astype(np.float64)
    votes = torch.zeros(z_pred.shape[0], dtype=torch.float32, device=z_pred.device)
    for row in cost_matrix:
        top = torch.topk(row, k=10, largest=False).indices
        votes[top] += 1.0
    default_norm = (item["default_costs"] - item["default_costs"].min()) / (
        item["default_costs"].max() - item["default_costs"].min() + EPS
    )
    vote_score = (-(votes) + 1e-6 * default_norm).detach().cpu().numpy().astype(np.float64)
    best_cost = cost_matrix[int(best_warp_index)].detach().cpu().numpy().astype(np.float64)
    maha = mahalanobis_cost(z_pred, z_goal)
    linear = linear_metric(z_pred, z_goal).detach().cpu().numpy().astype(np.float64)
    scalar = scalar_mlp(z_pred, z_goal).detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(int(random_seed) + pair_id * 104_729)
    random_score = rng.random(int(z_pred.shape[0])).astype(np.float64)
    random_rank1 = int(rng.choice(np.argsort(default)[:30]))
    return [
        metric_record(pair_id=pair_id, strategy="euclidean", score=default, c_real=c_real, success=success),
        metric_record(pair_id=pair_id, strategy="wes_rank_avg", score=rank_avg, c_real=c_real, success=success),
        metric_record(pair_id=pair_id, strategy="wes_vote_top10", score=vote_score, c_real=c_real, success=success),
        metric_record(
            pair_id=pair_id,
            strategy="single_warp_best",
            score=best_cost,
            c_real=c_real,
            success=success,
            extra={"best_warp_index": int(best_warp_index + 1)},
        ),
        metric_record(pair_id=pair_id, strategy="mahalanobis", score=maha, c_real=c_real, success=success),
        metric_record(pair_id=pair_id, strategy="linear_metric", score=linear, c_real=c_real, success=success),
        metric_record(pair_id=pair_id, strategy="scalar_mlp", score=scalar, c_real=c_real, success=success),
        metric_record(
            pair_id=pair_id,
            strategy="random_top30",
            score=random_score,
            c_real=c_real,
            success=success,
            selected_index=random_rank1,
        ),
        metric_record(pair_id=pair_id, strategy="v1_oracle", score=v1, c_real=c_real, success=success),
    ]


def exact_binomial_ci(successes: int, n: int) -> dict[str, Any]:
    if n <= 0:
        return {"successes": 0, "n": 0, "rate": None, "ci_low": None, "ci_high": None}
    ci = binomtest(int(successes), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return {
        "successes": int(successes),
        "n": int(n),
        "rate": clean_float(float(successes) / float(n)),
        "ci_low": clean_float(float(ci.low)),
        "ci_high": clean_float(float(ci.high)),
    }


def bootstrap_mean_ci(values: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        means[idx] = float(arr[rng.integers(0, len(arr), size=len(arr))].mean())
    return {
        "mean": clean_float(float(arr.mean())),
        "ci_low": clean_float(float(np.percentile(means, 2.5))),
        "ci_high": clean_float(float(np.percentile(means, 97.5))),
        "n": int(len(arr)),
        "n_bootstrap": int(n_boot),
    }


def paired_bootstrap_diff(
    records: list[dict[str, Any]],
    *,
    a: str,
    b: str,
    metric: str,
    key_fields: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in records:
        if row.get("strategy") not in (a, b) and row.get("variant") not in (a, b):
            continue
        name = str(row.get("strategy", row.get("variant")))
        key = tuple(row[field] for field in key_fields)
        by_key.setdefault(key, {})[name] = row
    diffs = []
    for row_map in by_key.values():
        if a in row_map and b in row_map:
            diffs.append(float(row_map[b][metric]) - float(row_map[a][metric]))
    arr = np.asarray(diffs, dtype=np.float64)
    return bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed) | {"a": a, "b": b, "metric": metric}


def mcnemar_exact(records: list[dict[str, Any]], *, a: str, b: str, key_fields: tuple[str, ...]) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], dict[str, bool]] = {}
    for row in records:
        name = str(row.get("strategy", row.get("variant")))
        if name not in (a, b):
            continue
        key = tuple(row[field] for field in key_fields)
        by_key.setdefault(key, {})[name] = bool(row.get("rank1_success", row.get("planning_success")))
    a_only = 0
    b_only = 0
    both = 0
    neither = 0
    for row_map in by_key.values():
        if a not in row_map or b not in row_map:
            continue
        av = row_map[a]
        bv = row_map[b]
        if av and bv:
            both += 1
        elif av and not bv:
            a_only += 1
        elif (not av) and bv:
            b_only += 1
        else:
            neither += 1
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(a_only, b_only), discordant, p=0.5).pvalue)
    return {
        "a": a,
        "b": b,
        "both_success": int(both),
        "both_failure": int(neither),
        "a_success_b_failure": int(a_only),
        "a_failure_b_success": int(b_only),
        "discordant": int(discordant),
        "p_value": clean_float(p_value),
    }


def summarize_strategy_records(records: list[dict[str, Any]], strategies: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for strategy in strategies:
        rows = [row for row in records if row["strategy"] == strategy]
        successes = sum(1 for row in rows if bool(row["rank1_success"]))
        regrets = np.asarray([float(row["selection_regret"]) for row in rows], dtype=np.float64)
        out[strategy] = {
            "n": int(len(rows)),
            "success": exact_binomial_ci(successes, len(rows)),
            "selection_regret": bootstrap_mean_ci(regrets, n_boot=BOOTSTRAP_SAMPLES, seed=17_000 + len(strategy)),
            "Rpool": scalar_summary([row.get("Rpool") for row in rows]),
            "rank1_c_real": scalar_summary([row.get("rank1_c_real") for row in rows]),
        }
    return out


def format_ci_rate(ci: dict[str, Any]) -> str:
    if ci["rate"] is None:
        return "NA"
    return f"{100.0 * ci['rate']:.1f}% [{100.0 * ci['ci_low']:.1f}, {100.0 * ci['ci_high']:.1f}]"


def format_ci_float(ci: dict[str, Any]) -> str:
    if ci["mean"] is None:
        return "NA"
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def print_posthoc_table(summary: dict[str, Any]) -> None:
    rows = []
    for strategy in POSTHOC_STRATEGIES:
        stats = summary[strategy]
        rows.append(
            [
                strategy,
                format_ci_rate(stats["success"]),
                format_ci_float(stats["selection_regret"]),
                fmt_float(stats["Rpool"]["mean"]),
            ]
        )
    print("\nStrict held-out post-hoc comparison (pairs 70--99)")
    print(tabulate(rows, headers=["Strategy", "Success 95% CI", "Regret 95% CI", "Rpool"], tablefmt="github"))


def make_cem_pool_path(root: Path, pair_id: int, seed: int) -> Path:
    return root / f"pair_{int(pair_id)}_seed_{int(seed)}.pt"


def load_existing_cem_records(output: Path, *, pool_root: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not output.exists():
        return []
    data = json.loads(output.read_text(encoding="utf-8"))
    records = []
    for row in data.get("cem_records", []):
        pool = row.get("pool_path")
        if pool and Path(pool).exists() and Path(pool).is_relative_to(pool_root):
            records.append(row)
        elif pool and Path(pool).exists():
            records.append(row)
    return records


def cem_run_complete(records: list[dict[str, Any]], pair_id: int, seed: int) -> bool:
    names = {
        row["variant"]
        for row in records
        if int(row["pair_id"]) == int(pair_id) and int(row["seed"]) == int(seed)
    }
    return all(name in names for name in CEM_VARIANTS)


def cem_pool_and_records(
    *,
    pair_spec: dict[str, Any],
    initial: dict[str, Any],
    goal: dict[str, Any],
    policy,
    env,
    cem_result: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_id = int(pair_spec["pair_id"])
    raw_actions = blocked_batch_to_raw_fast(
        np.asarray(cem_result["blocked_candidates"], dtype=np.float32),
        action_processor=policy.process["action"],
    )
    v1_costs, c_real, success, metrics = score_raw_actions(
        env=env,
        initial_state=np.asarray(initial["state"], dtype=np.float32),
        goal_state=np.asarray(goal["state"], dtype=np.float32),
        raw_actions_batch=raw_actions,
        seed_base=int(seed) + pair_id * 100_000,
    )
    default_costs = np.asarray(cem_result["default_costs"], dtype=np.float64)
    wes_costs = np.asarray(cem_result["warp_rank_avg_costs"], dtype=np.float64)
    default_rank1 = int(deterministic_argmin(default_costs))
    wes_rank1 = int(deterministic_argmin(wes_costs))
    oracle = int(deterministic_argmin(c_real))
    pool = {
        "metadata": {
            "format": "wes_strict_default_cem_pool_v1",
            "pair_id": int(pair_id),
            "seed": int(seed),
            "cem_sampling_seed": int(seed) + int(pair_id) * 1009,
            "device": str(device),
            "search_variant": "default_cem",
            "selection_variants": list(CEM_VARIANTS),
            "wallclock_seconds": cem_result.get("wallclock_seconds"),
        },
        "pair_spec": dict(pair_spec),
        "z_pred": torch.as_tensor(np.asarray(cem_result["z_pred"], dtype=np.float32)),
        "z_goal": torch.as_tensor(np.asarray(cem_result["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.as_tensor(np.asarray(cem_result["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.as_tensor(raw_actions.astype(np.float32)),
        "default_costs": torch.as_tensor(default_costs, dtype=torch.float64),
        "wes_rank_avg_costs": torch.as_tensor(wes_costs, dtype=torch.float64),
        "individual_warp_costs": torch.as_tensor(cem_result["individual_warp_costs"], dtype=torch.float64),
        "v1_hinge_costs": torch.as_tensor(v1_costs, dtype=torch.float64),
        "c_real_state": torch.as_tensor(c_real, dtype=torch.float64),
        "success": torch.as_tensor(success, dtype=torch.bool),
        "candidate_metrics": metrics,
        "default_rank1_candidate_index": int(default_rank1),
        "wes_rank1_candidate_index": int(wes_rank1),
        "oracle_best_candidate_index": int(oracle),
        "elite_candidate_indices": torch.as_tensor(cem_result["elite_candidate_indices"], dtype=torch.int64),
        "elite_costs": torch.as_tensor(cem_result["elite_costs"], dtype=torch.float64),
        "elite_cost_std_final": cem_result.get("elite_cost_std"),
        "iteration_diagnostics": cem_result.get("iteration_diagnostics"),
    }

    def record(variant: str, rank1: int, score: np.ndarray) -> dict[str, Any]:
        return {
            "pair_id": int(pair_id),
            "seed": int(seed),
            "variant": variant,
            "rank1_candidate_index": int(rank1),
            "rank1_success": bool(success[rank1]),
            "planning_success": bool(success[rank1]),
            "rank1_c_real": clean_float(float(c_real[rank1])),
            "oracle_best_candidate_index": int(oracle),
            "oracle_c_real": clean_float(float(c_real[oracle])),
            "selection_regret": clean_float(float(c_real[rank1] - c_real[oracle])),
            "Rpool": spearman_corr(score, c_real),
            "Rpool_Cmodel": spearman_corr(default_costs, c_real),
            "Rpool_WES": spearman_corr(wes_costs, c_real),
            "Rpool_V1": spearman_corr(v1_costs, c_real),
            "pool_Creal_std": clean_float(float(np.std(c_real, ddof=0))),
            "pool_success_mass": clean_float(float(np.mean(success))),
            "elite_cost_std": cem_result.get("elite_cost_std"),
            "pool_path": None,
        }

    return pool, [
        record("default_cem", default_rank1, default_costs),
        record("wes_posthoc", wes_rank1, wes_costs),
    ]


def summarize_cem_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for variant in CEM_VARIANTS:
        rows = [row for row in records if row["variant"] == variant]
        successes = sum(1 for row in rows if bool(row["planning_success"]))
        regrets = np.asarray([float(row["selection_regret"]) for row in rows], dtype=np.float64)
        out[variant] = {
            "n": int(len(rows)),
            "success": exact_binomial_ci(successes, len(rows)),
            "selection_regret": bootstrap_mean_ci(regrets, n_boot=BOOTSTRAP_SAMPLES, seed=31_000 + len(variant)),
            "Rpool": scalar_summary([row.get("Rpool") for row in rows]),
            "pool_Creal_std": scalar_summary([row.get("pool_Creal_std") for row in rows]),
        }
    return out


def print_cem_table(summary: dict[str, Any]) -> None:
    rows = []
    for variant in CEM_VARIANTS:
        stats = summary[variant]
        rows.append([variant, format_ci_rate(stats["success"]), format_ci_float(stats["selection_regret"]), fmt_float(stats["Rpool"]["mean"])])
    print("\nStrict held-out CEM comparison (pairs 70--99, seeds 0--2)")
    print(tabulate(rows, headers=["Variant", "Success 95% CI", "Regret 95% CI", "Rpool"], tablefmt="github"))


def update_payload(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def build_base_payload(args: argparse.Namespace, *, epochs: int, started: float) -> dict[str, Any]:
    return {
        "metadata": {
            "format": "wes_strict_holdout_v1",
            "created_at_unix": clean_float(time.time()),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "train_pairs": list(TRAIN_PAIRS),
            "test_pairs": list(TEST_PAIRS),
            "folds_within_train": fold_splits(),
            "epochs": int(epochs),
            "config": dict(V3_CONFIG, epochs=int(epochs)),
            "pool_root": str(args.pool_root),
            "strict_dir": str(args.strict_dir),
            "cem_pool_root": str(args.cem_pool_root),
            "device": str(args.device),
            "seed": int(args.seed),
            "wallclock_seconds": clean_float(time.time() - started),
        },
        "fold_training": [],
        "best_warp": {},
        "baseline_training": {},
        "posthoc_records": [],
        "posthoc_summary": {},
        "posthoc_paired_tests": {},
        "cem_records": [],
        "cem_summary": {},
        "cem_paired_tests": {},
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pool_root = args.pool_root.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.strict_dir = args.strict_dir.expanduser().resolve()
    args.cem_pool_root = args.cem_pool_root.expanduser().resolve()
    args.device = resolve_device(str(args.device))
    args.seeds = tuple(int(seed) for seed in args.seeds)
    epochs = int(args.epochs_override) if args.epochs_override is not None else int(V3_CONFIG["epochs"])
    set_seed(int(args.seed))
    started = time.time()
    payload = build_base_payload(args, epochs=epochs, started=started)

    print("== Strict held-out WES evaluation ==")
    print(f"train_pairs: {TRAIN_PAIRS[0]}-{TRAIN_PAIRS[-1]} ({len(TRAIN_PAIRS)})")
    print(f"test_pairs: {TEST_PAIRS[0]}-{TEST_PAIRS[-1]} ({len(TEST_PAIRS)})")
    print(f"device: {args.device}")
    print(f"epochs: {epochs}")
    print(f"output: {args.output}")

    train_items = load_pool_items(TRAIN_PAIRS, pool_root=args.pool_root, device=args.device)
    test_items = load_pool_items(TEST_PAIRS, pool_root=args.pool_root, device=args.device)
    by_train = {item["pair_id"]: item for item in train_items}

    warps: list[nn.Module] = []
    fold_rows: list[dict[str, Any]] = []
    if args.skip_training:
        print("Loading strict WES warps (--skip-training)")
        for fold_idx in range(1, N_FOLDS + 1):
            warps.append(load_warp(args.strict_dir / f"warp_fold_{fold_idx}.pt", device=args.device))
    else:
        for fold_idx, val_pairs in enumerate(fold_splits(), start=1):
            model_path = args.strict_dir / f"warp_fold_{fold_idx}.pt"
            if model_path.exists() and not args.no_resume:
                print(f"[Fold {fold_idx}/10] loading existing {model_path}")
                warp = load_warp(model_path, device=args.device)
                val_data = [by_train[pair_id] for pair_id in val_pairs]
                val_stats = validation_ranking_loss(lambda z, g, w=warp: warped_costs(w, z, g), val_data, hard_topk=int(V3_CONFIG["hard_topk"]))
                row = {"fold": fold_idx, "model_path": str(model_path), "validation_pairs": val_pairs, "loaded_existing": True, "validation": val_stats}
            else:
                train_pairs = [pair_id for pair_id in TRAIN_PAIRS if pair_id not in set(val_pairs)]
                train_data = [by_train[pair_id] for pair_id in train_pairs]
                val_data = [by_train[pair_id] for pair_id in val_pairs]
                print(f"\n[Fold {fold_idx}/10] train_n={len(train_data)} val={val_pairs[0]}-{val_pairs[-1]}")
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
                    seed=int(args.seed) + fold_idx * 10_000,
                    log_every=int(args.log_every),
                )
                val_stats = validation_ranking_loss(lambda z, g, w=warp: warped_costs(w, z, g), val_data, hard_topk=int(V3_CONFIG["hard_topk"]))
                row = {
                    "fold": fold_idx,
                    "model_path": str(model_path),
                    "train_pairs": train_pairs,
                    "validation_pairs": val_pairs,
                    "final_train_log": train_log[-1],
                    "validation": val_stats,
                }
                save_warp(model_path, warp, row)
            warps.append(warp)
            fold_rows.append(row)
            payload["fold_training"] = fold_rows
            payload["metadata"]["wallclock_seconds"] = clean_float(time.time() - started)
            update_payload(args.output, payload)
            print(f"[Fold {fold_idx}] val_loss={fmt_float(row['validation']['loss'], 4)} val_acc={fmt_float(row['validation']['pair_accuracy'])}")

    if not fold_rows:
        for fold_idx in range(1, N_FOLDS + 1):
            val_pairs = fold_splits()[fold_idx - 1]
            val_data = [by_train[pair_id] for pair_id in val_pairs]
            val_stats = validation_ranking_loss(
                lambda z, g, w=warps[fold_idx - 1]: warped_costs(w, z, g),
                val_data,
                hard_topk=int(V3_CONFIG["hard_topk"]),
            )
            fold_rows.append({"fold": fold_idx, "model_path": str(args.strict_dir / f"warp_fold_{fold_idx}.pt"), "validation_pairs": val_pairs, "validation": val_stats})

    best_warp_index = int(np.nanargmin([row["validation"]["loss"] if row["validation"]["loss"] is not None else np.inf for row in fold_rows]))
    payload["fold_training"] = fold_rows
    payload["best_warp"] = {"fold": int(best_warp_index + 1), "validation_loss": fold_rows[best_warp_index]["validation"]["loss"]}

    linear_path = args.strict_dir / "linear_metric.pt"
    scalar_path = args.strict_dir / "scalar_mlp.pt"
    if args.skip_training and linear_path.exists() and scalar_path.exists():
        linear_metric = load_linear_metric(linear_path, device=args.device)
        scalar_mlp = load_scalar_mlp(scalar_path, device=args.device)
    else:
        if linear_path.exists() and not args.no_resume:
            print(f"Loading existing linear metric: {linear_path}")
            linear_metric = load_linear_metric(linear_path, device=args.device)
        else:
            print("\n== Training linear metric baseline ==")
            linear_metric = LinearMetric().to(device=args.device)
            linear_log = train_metric_model(
                model=linear_metric,
                train_data=train_items,
                epochs=epochs,
                pair_samples_per_pool=int(V3_CONFIG["pair_samples_per_pool"]),
                hard_topk=int(V3_CONFIG["hard_topk"]),
                lr=float(V3_CONFIG["lr"]),
                weight_decay=float(args.weight_decay),
                identity_reg=float(V3_CONFIG["identity_reg"]),
                seed=int(args.seed) + 88_000,
                log_every=int(args.log_every),
            )
            save_model(linear_path, linear_metric, {"train_pairs": list(TRAIN_PAIRS), "final_train_log": linear_log[-1]})
            payload["baseline_training"]["linear_metric"] = {"model_path": str(linear_path), "final_train_log": linear_log[-1]}
        if scalar_path.exists() and not args.no_resume:
            print(f"Loading existing scalar MLP: {scalar_path}")
            scalar_mlp = load_scalar_mlp(scalar_path, device=args.device)
        else:
            print("\n== Training scalar MLP baseline ==")
            scalar_mlp = ScalarCostMLP().to(device=args.device)
            scalar_log = train_metric_model(
                model=scalar_mlp,
                train_data=train_items,
                epochs=epochs,
                pair_samples_per_pool=int(V3_CONFIG["pair_samples_per_pool"]),
                hard_topk=int(V3_CONFIG["hard_topk"]),
                lr=float(V3_CONFIG["lr"]),
                weight_decay=float(args.weight_decay),
                identity_reg=float(V3_CONFIG["identity_reg"]),
                seed=int(args.seed) + 99_000,
                log_every=int(args.log_every),
            )
            save_model(scalar_path, scalar_mlp, {"train_pairs": list(TRAIN_PAIRS), "final_train_log": scalar_log[-1]})
            payload["baseline_training"]["scalar_mlp"] = {"model_path": str(scalar_path), "final_train_log": scalar_log[-1]}

    print("\n== Strict held-out post-hoc evaluation ==")
    posthoc_records: list[dict[str, Any]] = []
    for item in test_items:
        rows = evaluate_posthoc_pair(
            item=item,
            warps=warps,
            best_warp_index=best_warp_index,
            linear_metric=linear_metric,
            scalar_mlp=scalar_mlp,
            random_seed=int(args.seed),
        )
        posthoc_records.extend(rows)
        print(
            f"pair {item['pair_id']}: eu_regret={fmt_float(rows[0]['selection_regret'])} "
            f"wes_regret={fmt_float(rows[1]['selection_regret'])} "
            f"eu_success={rows[0]['rank1_success']} wes_success={rows[1]['rank1_success']}"
        )
    posthoc_summary = summarize_strategy_records(posthoc_records, POSTHOC_STRATEGIES)
    posthoc_paired = {
        "success_diff_wes_minus_euclidean": paired_bootstrap_diff(
            posthoc_records,
            a="euclidean",
            b="wes_rank_avg",
            metric="rank1_success",
            key_fields=("pair_id",),
            n_boot=BOOTSTRAP_SAMPLES,
            seed=123,
        ),
        "regret_diff_wes_minus_euclidean": paired_bootstrap_diff(
            posthoc_records,
            a="euclidean",
            b="wes_rank_avg",
            metric="selection_regret",
            key_fields=("pair_id",),
            n_boot=BOOTSTRAP_SAMPLES,
            seed=124,
        ),
        "mcnemar_wes_vs_euclidean": mcnemar_exact(posthoc_records, a="euclidean", b="wes_rank_avg", key_fields=("pair_id",)),
    }
    payload["posthoc_records"] = posthoc_records
    payload["posthoc_summary"] = posthoc_summary
    payload["posthoc_paired_tests"] = posthoc_paired
    payload["metadata"]["wallclock_seconds"] = clean_float(time.time() - started)
    update_payload(args.output, payload)
    print_posthoc_table(posthoc_summary)
    print("\nPost-hoc paired WES-vs-Euclidean")
    print(json.dumps(jsonable(posthoc_paired), indent=2))

    cem_records = load_existing_cem_records(args.output, pool_root=args.cem_pool_root, resume=not args.no_resume)
    if not args.skip_cem:
        print("\n== Strict held-out CEM evaluation ==")
        pairs_data, pair_specs = load_pairs(args.pairs_path, max_pairs=None, pair_ids=list(TEST_PAIRS))
        pair_metadata = pairs_data["metadata"]
        validate_requested_pair_offsets(pair_specs, offset=int(pair_metadata["offset"]))
        dataset_path = Path(pair_metadata["dataset_path"])
        dataset = get_dataset(dataset_path.parent, dataset_path.stem)
        process = build_processors(dataset, ["action", "proprio", "state"])
        policy = build_policy(
            make_policy_namespace(checkpoint_dir=args.checkpoint_dir, device=args.device, seed=min(args.seeds)),
            process,
        )
        model = policy.solver.model
        model_device = next(model.parameters()).device
        for warp in warps:
            warp.to(device=model_device)
            warp.eval()
        env = gym.make("swm/PushT-v1")
        try:
            by_spec = {int(pair["pair_id"]): pair for pair in pair_specs}
            for pair_idx, pair_id in enumerate(TEST_PAIRS, start=1):
                pair_spec = by_spec[int(pair_id)]
                initial, goal = load_pair_rows_direct(dataset, pair_spec)
                prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
                for seed_idx, seed in enumerate(args.seeds, start=1):
                    if cem_run_complete(cem_records, pair_id, seed):
                        print(f"[pair {pair_idx}/30 seed {seed_idx}/3] pair={pair_id} seed={seed}: resume")
                        continue
                    cem_records = [row for row in cem_records if not (int(row["pair_id"]) == int(pair_id) and int(row["seed"]) == int(seed))]
                    run_started = time.time()
                    print(f"[pair {pair_idx}/30 seed {seed_idx}/3] pair={pair_id} seed={seed}: default CEM search")
                    cem_result = run_cem(
                        model=model,
                        prepared_info=prepared_info,
                        warps=warps,
                        pair_id=int(pair_id),
                        seed=int(seed),
                        variant="default_cem",
                    )
                    pool, rows = cem_pool_and_records(
                        pair_spec=pair_spec,
                        initial=initial,
                        goal=goal,
                        policy=policy,
                        env=env,
                        cem_result=cem_result,
                        seed=int(seed),
                        device=str(args.device),
                    )
                    path = make_cem_pool_path(args.cem_pool_root, pair_id, seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(pool, path)
                    for row in rows:
                        row["pool_path"] = str(path)
                    cem_records.extend(rows)
                    cem_summary = summarize_cem_records(cem_records)
                    payload["cem_records"] = sorted(cem_records, key=lambda row: (int(row["pair_id"]), str(row["variant"]), int(row["seed"])))
                    payload["cem_summary"] = cem_summary
                    payload["metadata"]["wallclock_seconds"] = clean_float(time.time() - started)
                    update_payload(args.output, payload)
                    print(
                        f"  default regret={fmt_float(rows[0]['selection_regret'])} success={rows[0]['planning_success']}; "
                        f"wes regret={fmt_float(rows[1]['selection_regret'])} success={rows[1]['planning_success']} "
                        f"elapsed={seconds_to_hms(time.time() - run_started)}"
                    )
        finally:
            if hasattr(env, "close"):
                env.close()

    if cem_records:
        cem_summary = summarize_cem_records(cem_records)
        cem_paired = {
            "success_diff_wes_minus_default": paired_bootstrap_diff(
                cem_records,
                a="default_cem",
                b="wes_posthoc",
                metric="planning_success",
                key_fields=("pair_id", "seed"),
                n_boot=BOOTSTRAP_SAMPLES,
                seed=223,
            ),
            "regret_diff_wes_minus_default": paired_bootstrap_diff(
                cem_records,
                a="default_cem",
                b="wes_posthoc",
                metric="selection_regret",
                key_fields=("pair_id", "seed"),
                n_boot=BOOTSTRAP_SAMPLES,
                seed=224,
            ),
            "mcnemar_wes_vs_default": mcnemar_exact(cem_records, a="default_cem", b="wes_posthoc", key_fields=("pair_id", "seed")),
        }
        payload["cem_records"] = sorted(cem_records, key=lambda row: (int(row["pair_id"]), str(row["variant"]), int(row["seed"])))
        payload["cem_summary"] = cem_summary
        payload["cem_paired_tests"] = cem_paired
        print_cem_table(cem_summary)
        print("\nCEM paired WES-vs-default")
        print(json.dumps(jsonable(cem_paired), indent=2))

    payload["metadata"]["wallclock_seconds"] = clean_float(time.time() - started)
    update_payload(args.output, payload)
    print(f"\nWrote results: {args.output}")
    print(f"Wrote strict warps: {args.strict_dir}")
    print(f"Wrote CEM pools: {args.cem_pool_root}")
    print(f"Elapsed: {seconds_to_hms(time.time() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
