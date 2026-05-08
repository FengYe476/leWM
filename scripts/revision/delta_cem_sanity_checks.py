#!/usr/bin/env python3
"""Matched-distribution sanity checks for PushT Delta_CEM.

The checks in this script address the range-restriction objection: R_pool may
be low merely because final CEM pools have compressed physical-cost support.
We therefore test ranking quality on large physical gaps, top-k retrieval, and
endpoint correlations after matching the endpoint set to each pool's physical
cost range.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LATENT_DIM = 192
N_EXPECTED_PAIRS = 100
N_CANDIDATES = 300
PAIRWISE_THRESHOLDS = (5.0, 10.0, 20.0)
NDCG_KS = (10, 30)
SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)
ENDPOINT_C2_DIM = 64
ENDPOINT_C2_SEEDS = tuple(range(10))

DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
DEFAULT_RERANK_PATH = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_ENDPOINT_LATENTS = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
DEFAULT_STAGE1A_PATH = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_full.json"
DEFAULT_RPOOL_PATH = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "delta_cem_sanity_checks.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--rerank-path", type=Path, default=DEFAULT_RERANK_PATH)
    parser.add_argument("--endpoint-latents", type=Path, default=DEFAULT_ENDPOINT_LATENTS)
    parser.add_argument("--stage1a-path", type=Path, default=DEFAULT_STAGE1A_PATH)
    parser.add_argument("--rpool-path", type=Path, default=DEFAULT_RPOOL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairwise-pairs", type=int, default=1000)
    parser.add_argument("--random-ndcg-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260508)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def tensor_to_numpy(pool: dict[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    value = pool[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[sorter[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    rho = float(np.corrcoef(x, y)[0, 1])
    return clean_float(rho)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return pearson_corr(rankdata(x), rankdata(y))


def scalar_summary(values: list[float | int | None], *, none_as_zero: bool = False) -> dict[str, Any]:
    valid = []
    n_missing = 0
    for value in values:
        number = clean_float(value)
        if number is None:
            n_missing += 1
            if none_as_zero:
                valid.append(0.0)
        else:
            valid.append(float(number))
    arr = np.asarray(valid, dtype=np.float64)
    return {
        "mean": clean_float(arr.mean()) if len(arr) else None,
        "std": clean_float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "min": clean_float(arr.min()) if len(arr) else None,
        "max": clean_float(arr.max()) if len(arr) else None,
        "n": int(len(arr)),
        "n_missing": int(0 if none_as_zero else n_missing),
        "none_as_zero": bool(none_as_zero),
        "ddof": 1,
    }


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return clean_float(float(numerator) / float(denominator))


def sample_pair_indices(*, n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    i = rng.integers(0, n, size=int(count), endpoint=False)
    j = rng.integers(0, n, size=int(count), endpoint=False)
    equal = i == j
    while bool(equal.any()):
        j[equal] = rng.integers(0, n, size=int(equal.sum()), endpoint=False)
        equal = i == j
    return i, j


def pairwise_accuracy_for_scores(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    real_diff = labels[i] - labels[j]
    score_diff = scores[i] - scores[j]
    mask = (
        np.isfinite(real_diff)
        & np.isfinite(score_diff)
        & (np.abs(real_diff) > float(threshold))
        & (real_diff != 0.0)
    )
    n = int(mask.sum())
    if n == 0:
        return {"accuracy": None, "n_comparisons": 0, "n_correct": 0}
    correct = np.sign(score_diff[mask]) == np.sign(real_diff[mask])
    return {
        "accuracy": clean_float(float(np.mean(correct))),
        "n_comparisons": n,
        "n_correct": int(correct.sum()),
    }


def ndcg_at(*, scores: np.ndarray, labels: np.ndarray, k: int) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    mask = np.isfinite(scores) & np.isfinite(labels)
    if int(mask.sum()) == 0:
        return None
    scores = scores[mask]
    labels = labels[mask]
    k_eff = min(int(k), len(labels))
    relevance = float(np.max(labels)) - labels
    if not np.any(relevance > 0.0):
        return None
    indices = np.arange(len(labels))
    order = np.lexsort((indices, scores))[:k_eff]
    ideal_order = np.lexsort((indices, labels))[:k_eff]
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2, dtype=np.float64))
    dcg = float(np.sum(relevance[order] * discounts))
    idcg = float(np.sum(relevance[ideal_order] * discounts))
    if idcg <= 0.0:
        return None
    return clean_float(dcg / idcg)


def random_ndcg_at(
    *,
    labels: np.ndarray,
    k: int,
    trials: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = []
    for _ in range(int(trials)):
        scores = rng.random(len(labels))
        values.append(ndcg_at(scores=scores, labels=labels, k=k))
    return scalar_summary(values)


def load_subset_ids(path: Path) -> dict[str, list[int]]:
    data = load_json(path)
    anchors = data.get("metadata", {}).get("anchor_definitions")
    if not isinstance(anchors, dict):
        raise ValueError(f"{path} is missing metadata.anchor_definitions")
    subset_ids: dict[str, list[int]] = {}
    for subset in SUBSET_ORDER:
        entry = anchors.get(subset)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing subset anchor definition: {subset}")
        subset_ids[subset] = [int(pair_id) for pair_id in entry.get("pair_ids", [])]
    return subset_ids


def pool_path(pool_dir: Path, pair_id: int) -> Path:
    return pool_dir / f"pair_{int(pair_id)}.pt"


def validate_pool(pool: dict[str, Any], *, pair_id: int) -> None:
    expected = {
        "default_costs": (N_CANDIDATES,),
        "v1_hinge_costs": (N_CANDIDATES,),
        "c_real_state": (N_CANDIDATES,),
    }
    for key, shape in expected.items():
        if key not in pool:
            raise KeyError(f"pair_{pair_id}.pt is missing key {key!r}")
        value = pool[key]
        if not torch.is_tensor(value):
            raise TypeError(f"pair_{pair_id}.pt key {key!r} is not a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"pair_{pair_id}.pt key {key!r} has shape {tuple(value.shape)} != {shape}")


def analyze_pool(
    *,
    pool: dict[str, Any],
    pair_id: int,
    pairwise_pairs: int,
    random_ndcg_trials: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    validate_pool(pool, pair_id=pair_id)
    c_model = tensor_to_numpy(pool, "default_costs", dtype=np.float64)
    v1 = tensor_to_numpy(pool, "v1_hinge_costs", dtype=np.float64)
    c_real = tensor_to_numpy(pool, "c_real_state", dtype=np.float64)
    random_scores = rng.random(len(c_real))
    sample_i, sample_j = sample_pair_indices(n=len(c_real), count=pairwise_pairs, rng=rng)

    pairwise: dict[str, Any] = {}
    for threshold in PAIRWISE_THRESHOLDS:
        pairwise[str(int(threshold))] = {
            "threshold": float(threshold),
            "C_model": pairwise_accuracy_for_scores(
                scores=c_model,
                labels=c_real,
                i=sample_i,
                j=sample_j,
                threshold=threshold,
            ),
            "V1_oracle": pairwise_accuracy_for_scores(
                scores=v1,
                labels=c_real,
                i=sample_i,
                j=sample_j,
                threshold=threshold,
            ),
            "Random": pairwise_accuracy_for_scores(
                scores=random_scores,
                labels=c_real,
                i=sample_i,
                j=sample_j,
                threshold=threshold,
            ),
        }

    ndcg: dict[str, Any] = {}
    for k in NDCG_KS:
        ndcg[str(k)] = {
            "k": int(k),
            "C_model": ndcg_at(scores=c_model, labels=c_real, k=k),
            "V1_oracle": ndcg_at(scores=v1, labels=c_real, k=k),
            "Random": random_ndcg_at(labels=c_real, k=k, trials=random_ndcg_trials, rng=rng),
        }

    c_model_range = clean_float(float(np.nanmax(c_model) - np.nanmin(c_model)))
    c_real_range = clean_float(float(np.nanmax(c_real) - np.nanmin(c_real)))
    v1_range = clean_float(float(np.nanmax(v1) - np.nanmin(v1)))
    top30 = np.argsort(c_model, kind="mergesort")[:30]

    return {
        "pair_id": int(pair_id),
        "cell": str(pool.get("metadata", {}).get("cell", pool.get("pair_spec", {}).get("cell", ""))),
        "pairwise_accuracy": pairwise,
        "ndcg": ndcg,
        "r_pool": {
            "C_model": spearman_corr(c_model, c_real),
            "V1_oracle": spearman_corr(v1, c_real),
            "Random": spearman_corr(random_scores, c_real),
        },
        "compression": {
            "C_model_dynamic_range": c_model_range,
            "V1_dynamic_range": v1_range,
            "C_real_dynamic_range": c_real_range,
            "C_model_to_C_real_range_ratio": (
                ratio(c_model_range, c_real_range) if c_model_range is not None and c_real_range is not None else None
            ),
            "V1_to_C_real_range_ratio": (
                ratio(v1_range, c_real_range) if v1_range is not None and c_real_range is not None else None
            ),
            "C_model_std": clean_float(float(np.nanstd(c_model))),
            "C_real_std": clean_float(float(np.nanstd(c_real))),
            "top30_C_model_std": clean_float(float(np.nanstd(c_model[top30]))),
            "top30_C_real_std": clean_float(float(np.nanstd(c_real[top30]))),
            "C_real_min": clean_float(float(np.nanmin(c_real))),
            "C_real_max": clean_float(float(np.nanmax(c_real))),
        },
    }


def make_endpoint_projection_costs(endpoint: dict[str, Any]) -> dict[str, Any]:
    z_terminal = endpoint["z_terminal"].detach().cpu().to(dtype=torch.float32)
    z_goal = endpoint["z_goal"].detach().cpu().to(dtype=torch.float32)
    labels = endpoint["C_real_state"].detach().cpu().numpy().astype(np.float64)
    c0_cost = torch.sum((z_terminal - z_goal) ** 2, dim=1).detach().cpu().numpy().astype(np.float64)
    c2_costs = []
    for seed in ENDPOINT_C2_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        projection = torch.randn(
            (LATENT_DIM, ENDPOINT_C2_DIM),
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(ENDPOINT_C2_DIM)
        costs = torch.sum((z_terminal @ projection - z_goal @ projection) ** 2, dim=1)
        c2_costs.append(costs.detach().cpu().numpy().astype(np.float64))
    return {
        "labels": labels,
        "C0_lewm_cost": c0_cost,
        "C2_m64_costs": c2_costs,
    }


def endpoint_restricted_records(
    *,
    pool_records: list[dict[str, Any]],
    endpoint_costs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = endpoint_costs["labels"]
    c0_cost = endpoint_costs["C0_lewm_cost"]
    c2_costs = endpoint_costs["C2_m64_costs"]

    full_c2_seed_values = [spearman_corr(costs, labels) for costs in c2_costs]
    full = {
        "C0_lewm": spearman_corr(c0_cost, labels),
        "C2_m64": {
            "mean": scalar_summary(full_c2_seed_values),
            "by_seed": {str(seed): full_c2_seed_values[idx] for idx, seed in enumerate(ENDPOINT_C2_SEEDS)},
        },
    }

    records = []
    for pool_record in pool_records:
        compression = pool_record["compression"]
        lo = float(compression["C_real_min"])
        hi = float(compression["C_real_max"])
        mask = np.isfinite(labels) & (labels >= lo) & (labels <= hi)
        n = int(mask.sum())
        c2_seed_values = [spearman_corr(costs[mask], labels[mask]) for costs in c2_costs]
        records.append(
            {
                "pair_id": int(pool_record["pair_id"]),
                "pool_C_real_min": lo,
                "pool_C_real_max": hi,
                "n_endpoint_records": n,
                "C0_lewm": spearman_corr(c0_cost[mask], labels[mask]),
                "C2_m64": {
                    "mean": scalar_summary(c2_seed_values),
                    "by_seed": {str(seed): c2_seed_values[idx] for idx, seed in enumerate(ENDPOINT_C2_SEEDS)},
                },
            }
        )

    summary = {
        "full": full,
        "pool_range_restricted": {
            "C0_lewm": scalar_summary([record["C0_lewm"] for record in records]),
            "C2_m64": scalar_summary([record["C2_m64"]["mean"]["mean"] for record in records]),
            "n_endpoint_records": scalar_summary([record["n_endpoint_records"] for record in records]),
        },
    }
    return records, summary


def load_reference_metrics(*, rpool_path: Path, stage1a_path: Path) -> dict[str, Any]:
    references: dict[str, Any] = {}
    if rpool_path.exists():
        rpool = load_json(rpool_path)
        overall = rpool.get("summary", {}).get("overall", {})
        references["rpool_v1_pusht"] = {
            "R_pool_C_model": overall.get("Rpool_Cmodel", {}).get("mean"),
            "R_pool_V1_effective": overall.get("Rpool_V1_effective", {}).get("mean"),
        }
    if stage1a_path.exists():
        stage1a = load_json(stage1a_path)
        c2_64 = stage1a.get("controls", {}).get("C2", {}).get("by_dim", {}).get(str(ENDPOINT_C2_DIM), {})
        references["stage1a_C2_m64"] = {
            "R_endpoint": c2_64.get("aggregate", {}).get("global_spearman", {}).get("mean"),
            "source": str(stage1a_path.relative_to(PROJECT_ROOT)),
        }
        c0 = stage1a.get("controls", {}).get("C0", {})
        references["stage1a_C0_lewm"] = {
            "R_endpoint": c0.get("metrics", {}).get("global_spearman"),
            "source": str(stage1a_path.relative_to(PROJECT_ROOT)),
        }
    return references


def summarize_pool_records(
    *,
    records: list[dict[str, Any]],
    subset_ids: dict[str, list[int]],
    endpoint_summary: dict[str, Any],
    references: dict[str, Any],
) -> dict[str, Any]:
    def summarize_selected(selected: list[dict[str, Any]]) -> dict[str, Any]:
        pairwise: dict[str, Any] = {}
        for threshold in PAIRWISE_THRESHOLDS:
            key = str(int(threshold))
            pairwise[key] = {
                "threshold": float(threshold),
                "C_model": scalar_summary([record["pairwise_accuracy"][key]["C_model"]["accuracy"] for record in selected]),
                "V1_oracle": scalar_summary(
                    [record["pairwise_accuracy"][key]["V1_oracle"]["accuracy"] for record in selected]
                ),
                "Random": scalar_summary([record["pairwise_accuracy"][key]["Random"]["accuracy"] for record in selected]),
                "n_comparisons": scalar_summary(
                    [record["pairwise_accuracy"][key]["C_model"]["n_comparisons"] for record in selected]
                ),
            }
        ndcg: dict[str, Any] = {}
        for k in NDCG_KS:
            key = str(k)
            ndcg[key] = {
                "C_model": scalar_summary([record["ndcg"][key]["C_model"] for record in selected]),
                "V1_oracle": scalar_summary([record["ndcg"][key]["V1_oracle"] for record in selected]),
                "Random": scalar_summary([record["ndcg"][key]["Random"]["mean"] for record in selected]),
            }
        compression = {
            "C_model_dynamic_range": scalar_summary(
                [record["compression"]["C_model_dynamic_range"] for record in selected]
            ),
            "V1_dynamic_range": scalar_summary([record["compression"]["V1_dynamic_range"] for record in selected]),
            "C_real_dynamic_range": scalar_summary(
                [record["compression"]["C_real_dynamic_range"] for record in selected]
            ),
            "C_model_to_C_real_range_ratio": scalar_summary(
                [record["compression"]["C_model_to_C_real_range_ratio"] for record in selected]
            ),
            "V1_to_C_real_range_ratio": scalar_summary(
                [record["compression"]["V1_to_C_real_range_ratio"] for record in selected]
            ),
            "C_model_std": scalar_summary([record["compression"]["C_model_std"] for record in selected]),
            "C_real_std": scalar_summary([record["compression"]["C_real_std"] for record in selected]),
            "top30_C_model_std": scalar_summary([record["compression"]["top30_C_model_std"] for record in selected]),
            "top30_C_real_std": scalar_summary([record["compression"]["top30_C_real_std"] for record in selected]),
        }
        r_pool = {
            "C_model": scalar_summary([record["r_pool"]["C_model"] for record in selected]),
            "V1_oracle": scalar_summary([record["r_pool"]["V1_oracle"] for record in selected]),
            "V1_oracle_effective": scalar_summary(
                [record["r_pool"]["V1_oracle"] for record in selected],
                none_as_zero=True,
            ),
            "Random": scalar_summary([record["r_pool"]["Random"] for record in selected]),
        }
        return {
            "n_pairs": int(len(selected)),
            "pairwise_accuracy": pairwise,
            "ndcg": ndcg,
            "r_pool": r_pool,
            "compression": compression,
        }

    by_id = {int(record["pair_id"]): record for record in records}
    by_subset = {}
    for subset in SUBSET_ORDER:
        selected = [by_id[pair_id] for pair_id in subset_ids[subset] if pair_id in by_id]
        by_subset[subset] = summarize_selected(selected)
        by_subset[subset]["pair_ids"] = [int(pair_id) for pair_id in subset_ids[subset]]

    return {
        "overall": summarize_selected(records),
        "by_subset": by_subset,
        "endpoint_range_match": endpoint_summary,
        "references": references,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def make_summary_table(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    endpoint = summary["endpoint_range_match"]
    refs = summary["references"]
    rows = [
        "| Metric | C_model | V1 oracle | Random |",
        "|---|---:|---:|---:|",
    ]
    for threshold in PAIRWISE_THRESHOLDS:
        key = str(int(threshold))
        row = overall["pairwise_accuracy"][key]
        rows.append(
            f"| Pairwise accuracy (gap>{int(threshold)}) | "
            f"{pct(row['C_model']['mean'])} | {pct(row['V1_oracle']['mean'])} | {pct(row['Random']['mean'])} |"
        )
    for k in NDCG_KS:
        key = str(k)
        row = overall["ndcg"][key]
        rows.append(
            f"| NDCG@{k} | {fmt(row['C_model']['mean'])} | {fmt(row['V1_oracle']['mean'])} | "
            f"{fmt(row['Random']['mean'])} |"
        )
    endpoint_full = refs.get("stage1a_C2_m64", {}).get("R_endpoint")
    if endpoint_full is None:
        endpoint_full = endpoint["full"]["C2_m64"]["mean"]["mean"]
    rows.append(f"| R_endpoint (full) | {fmt(endpoint_full)} | - | - |")
    rows.append(
        "| R_endpoint (pool-range restricted) | "
        f"{fmt(endpoint['pool_range_restricted']['C2_m64']['mean'])} | - | - |"
    )
    rows.append(
        f"| R_pool | {fmt(overall['r_pool']['C_model']['mean'])} | "
        f"{fmt(overall['r_pool']['V1_oracle_effective']['mean'])} | "
        f"{fmt(overall['r_pool']['Random']['mean'])} |"
    )
    return "\n".join(rows)


def print_subset_pairwise(summary: dict[str, Any]) -> None:
    print("\nPairwise accuracy by subset")
    header = "subset | n | gap>5 C_model/V1 | gap>10 C_model/V1 | gap>20 C_model/V1"
    print(header)
    print("-" * len(header))
    for subset in SUBSET_ORDER:
        item = summary["by_subset"][subset]
        chunks = []
        for threshold in PAIRWISE_THRESHOLDS:
            key = str(int(threshold))
            row = item["pairwise_accuracy"][key]
            chunks.append(f"{pct(row['C_model']['mean'])}/{pct(row['V1_oracle']['mean'])}")
        print(f"{subset} | {item['n_pairs']} | " + " | ".join(chunks))


def main() -> int:
    args = parse_args()
    if int(args.pairwise_pairs) <= 0:
        raise ValueError("--pairwise-pairs must be positive")
    if int(args.random_ndcg_trials) <= 0:
        raise ValueError("--random-ndcg-trials must be positive")

    subset_ids = load_subset_ids(args.rerank_path)
    references = load_reference_metrics(rpool_path=args.rpool_path, stage1a_path=args.stage1a_path)

    rng = np.random.default_rng(int(args.seed))
    pool_records = []
    for pair_id in range(N_EXPECTED_PAIRS):
        path = pool_path(args.pool_dir, pair_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing PushT pool: {path}")
        pool = load_torch(path)
        pool_records.append(
            analyze_pool(
                pool=pool,
                pair_id=pair_id,
                pairwise_pairs=int(args.pairwise_pairs),
                random_ndcg_trials=int(args.random_ndcg_trials),
                rng=rng,
            )
        )

    endpoint = load_torch(args.endpoint_latents)
    endpoint_costs = make_endpoint_projection_costs(endpoint)
    endpoint_records, endpoint_summary = endpoint_restricted_records(
        pool_records=pool_records,
        endpoint_costs=endpoint_costs,
    )
    summary = summarize_pool_records(
        records=pool_records,
        subset_ids=subset_ids,
        endpoint_summary=endpoint_summary,
        references=references,
    )

    output = {
        "metadata": {
            "format": "delta_cem_sanity_checks_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "inputs": {
                "pool_dir": display_path(args.pool_dir),
                "rerank_path": display_path(args.rerank_path),
                "endpoint_latents": display_path(args.endpoint_latents),
                "stage1a_path": display_path(args.stage1a_path),
                "rpool_path": display_path(args.rpool_path),
            },
            "pairwise_thresholds": list(PAIRWISE_THRESHOLDS),
            "pairwise_sampled_pairs_per_pool": int(args.pairwise_pairs),
            "ndcg_ks": list(NDCG_KS),
            "random_ndcg_trials_per_pool": int(args.random_ndcg_trials),
            "seed": int(args.seed),
            "endpoint_reference": {
                "control": "C2 Gaussian projection",
                "dimension": ENDPOINT_C2_DIM,
                "seeds": list(ENDPOINT_C2_SEEDS),
                "projection_scaling": "torch.randn(192, m) / sqrt(m)",
            },
            "notes": [
                "Pairwise accuracy counts learned-cost ties as incorrect when the physical gap exceeds threshold.",
                "NDCG uses per-pool linear relevance max(C_real_state) - C_real_state, so lower physical cost is more relevant.",
                "Endpoint pool-range restriction filters Track A endpoint rows to each pool's [min C_real_state, max C_real_state].",
            ],
        },
        "summary": summary,
        "per_pair": pool_records,
        "endpoint_range_match_per_pair": endpoint_records,
        "summary_table_markdown": make_summary_table(summary),
    }
    write_json(args.output, output)

    print("\nDelta_CEM matched-distribution sanity checks")
    print(make_summary_table(summary))
    print_subset_pairwise(summary)
    compression = summary["overall"]["compression"]
    print("\nPool compression diagnostic")
    print(
        "C_model range / C_real range: "
        f"{fmt(compression['C_model_to_C_real_range_ratio']['mean'])} "
        f"(C_model range {fmt(compression['C_model_dynamic_range']['mean'])}, "
        f"C_real range {fmt(compression['C_real_dynamic_range']['mean'])})"
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
