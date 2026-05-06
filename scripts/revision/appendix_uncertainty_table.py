#!/usr/bin/env python3
"""Build the appendix uncertainty table for second-round revision."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import beta, rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BOOTSTRAP = PROJECT_ROOT / "results" / "revision" / "bootstrap_ci.json"
DEFAULT_PUSHT_STAGE1A = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_full.json"
DEFAULT_PUSHT_LATENTS = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
DEFAULT_PUSHT_RPOOL = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_PUSHT_V3 = PROJECT_ROOT / "results" / "revision" / "v3_pool_analysis_pusht.json"
DEFAULT_CUBE_STAGE1A = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1a.json"
DEFAULT_CUBE_RPOOL = PROJECT_ROOT / "results" / "revision" / "rpool_v1_cube.json"
DEFAULT_MPPI = PROJECT_ROOT / "results" / "revision" / "mppi_pool_analysis.json"
DEFAULT_TRACK_B = PROJECT_ROOT / "results" / "phase2" / "track_b" / "ranking_comparison.json"
DEFAULT_DINO_FEATURES = PROJECT_ROOT / "results" / "phase2" / "track_b" / "dinov2_features.pt"
DEFAULT_SUBSPACE = PROJECT_ROOT / "results" / "phase2" / "subspace_cem" / "stage_a_pusht.json"
DEFAULT_COST_HEAD_SPLIT3 = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "split3_planning.json"
DEFAULT_COST_HEAD_CEM_AWARE = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "cem_aware_planning.json"
DEFAULT_HYBRID_K60 = (
    PROJECT_ROOT / "results" / "phase2" / "p2_0" / "oracle_budget_cem_corrected" / "split3_k60.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "appendix_uncertainty_table.json"
DEFAULT_TEX = PROJECT_ROOT / "docs" / "revision" / "appendix_uncertainty_table.tex"

LATENT_DIM = 192
PROJECTION_SEEDS = tuple(range(10))


@dataclass
class Row:
    metric_name: str
    environment: str
    protocol_experiment: str
    n_pairs: str
    seeds: str
    point_estimate: str
    ci_95: str
    method: str
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-path", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--pusht-stage1a-path", type=Path, default=DEFAULT_PUSHT_STAGE1A)
    parser.add_argument("--pusht-latents-path", type=Path, default=DEFAULT_PUSHT_LATENTS)
    parser.add_argument("--pusht-rpool-path", type=Path, default=DEFAULT_PUSHT_RPOOL)
    parser.add_argument("--pusht-v3-path", type=Path, default=DEFAULT_PUSHT_V3)
    parser.add_argument("--cube-stage1a-path", type=Path, default=DEFAULT_CUBE_STAGE1A)
    parser.add_argument("--cube-rpool-path", type=Path, default=DEFAULT_CUBE_RPOOL)
    parser.add_argument("--mppi-path", type=Path, default=DEFAULT_MPPI)
    parser.add_argument("--track-b-path", type=Path, default=DEFAULT_TRACK_B)
    parser.add_argument("--dino-features-path", type=Path, default=DEFAULT_DINO_FEATURES)
    parser.add_argument("--subspace-path", type=Path, default=DEFAULT_SUBSPACE)
    parser.add_argument("--cost-head-split3-path", type=Path, default=DEFAULT_COST_HEAD_SPLIT3)
    parser.add_argument("--cost-head-cem-aware-path", type=Path, default=DEFAULT_COST_HEAD_CEM_AWARE)
    parser.add_argument("--hybrid-k60-path", type=Path, default=DEFAULT_HYBRID_K60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--alpha", type=float, default=0.05)
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


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    return (
        float(np.quantile(samples, alpha / 2.0)),
        float(np.quantile(samples, 1.0 - alpha / 2.0)),
    )


def mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    method: str = "pair-level bootstrap",
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("mean_ci expects a non-empty 1D array")
    n = len(values)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        boot[idx] = float(np.mean(values[rng.integers(0, n, size=n)]))
    lo, hi = percentile_ci(boot, alpha)
    return {
        "estimate": float(np.mean(values)),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n),
        "n_bootstrap": int(n_bootstrap),
        "method": method,
    }


def paired_diff_ci(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    scale: float = 1.0,
    method: str = "paired bootstrap over pairs",
) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    if baseline.shape != comparison.shape or baseline.ndim != 1:
        raise ValueError(f"Expected paired arrays, got {baseline.shape} and {comparison.shape}")
    diffs = (comparison - baseline) * scale
    n = len(diffs)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        boot[idx] = float(np.mean(diffs[rng.integers(0, n, size=n)]))
    lo, hi = percentile_ci(boot, alpha)
    return {
        "baseline_estimate": float(np.mean(baseline) * scale),
        "comparison_estimate": float(np.mean(comparison) * scale),
        "estimate": float(np.mean(diffs)),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n),
        "n_bootstrap": int(n_bootstrap),
        "method": method,
    }


def clopper_pearson(k: int, n: int, *, alpha: float) -> dict[str, Any]:
    k = int(k)
    n = int(n)
    if not (0 <= k <= n) or n <= 0:
        raise ValueError(f"Invalid binomial count k={k}, n={n}")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return {
        "successes": k,
        "trials": n,
        "estimate": float(k / n),
        "ci_low": lower,
        "ci_high": upper,
        "method": "Clopper-Pearson exact binomial interval",
    }


def squared_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum((a - b) ** 2, dim=1)


def stage1a_endpoint_value(data: dict[str, Any], dim: int) -> float:
    for row in data["summary_table"]:
        if row.get("control") == "C2" and row.get("config") == f"gaussian_m={dim}":
            return float(row["global_spearman_mean"])
    raise KeyError(f"Missing C2 gaussian_m={dim}")


def rank_cluster_sufficient_stats(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> np.ndarray:
    x = rankdata(costs, method="average").astype(np.float64)
    y = rankdata(labels, method="average").astype(np.float64)
    unique_pairs = np.asarray(sorted(np.unique(pair_ids).tolist()), dtype=np.int64)
    stats = np.zeros((len(unique_pairs), 6), dtype=np.float64)
    for out_idx, pair_id in enumerate(unique_pairs):
        mask = pair_ids == pair_id
        xp = x[mask]
        yp = y[mask]
        stats[out_idx] = [
            float(mask.sum()),
            float(xp.sum()),
            float(yp.sum()),
            float(np.square(xp).sum()),
            float(np.square(yp).sum()),
            float((xp * yp).sum()),
        ]
    return stats


def pearson_from_sums(sums: np.ndarray) -> float:
    n, sx, sy, sx2, sy2, sxy = [float(item) for item in sums]
    numerator = n * sxy - sx * sy
    denom_x = n * sx2 - sx * sx
    denom_y = n * sy2 - sy * sy
    denom = math.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0))
    if denom == 0.0:
        return float("nan")
    return float(numerator / denom)


def rank_cluster_bootstrap(
    cost_arrays: list[np.ndarray],
    labels: np.ndarray,
    pair_ids: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    method: str = "pair-level bootstrap over rank-transformed endpoint records",
) -> dict[str, Any]:
    stats_by_seed = np.stack(
        [rank_cluster_sufficient_stats(costs, labels, pair_ids) for costs in cost_arrays],
        axis=0,
    )
    n_pairs = stats_by_seed.shape[1]
    full = np.asarray([pearson_from_sums(seed_stats.sum(axis=0)) for seed_stats in stats_by_seed])
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, n_pairs, size=n_pairs)
        sampled = stats_by_seed[:, sample, :].sum(axis=1)
        boot[idx] = float(np.nanmean([pearson_from_sums(row) for row in sampled]))
    lo, hi = percentile_ci(boot, alpha)
    return {
        "estimate": float(np.nanmean(full)),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n_pairs),
        "n_bootstrap": int(n_bootstrap),
        "method": method,
    }


def rank_cluster_diff_bootstrap(
    baseline_costs: np.ndarray,
    comparison_costs: np.ndarray,
    labels: np.ndarray,
    pair_ids: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    method: str = "pair-level bootstrap over rank-transformed endpoint records",
) -> dict[str, Any]:
    base_stats = rank_cluster_sufficient_stats(baseline_costs, labels, pair_ids)
    comp_stats = rank_cluster_sufficient_stats(comparison_costs, labels, pair_ids)
    n_pairs = base_stats.shape[0]
    full_base = pearson_from_sums(base_stats.sum(axis=0))
    full_comp = pearson_from_sums(comp_stats.sum(axis=0))
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, n_pairs, size=n_pairs)
        boot[idx] = pearson_from_sums(base_stats[sample].sum(axis=0)) - pearson_from_sums(
            comp_stats[sample].sum(axis=0)
        )
    lo, hi = percentile_ci(boot, alpha)
    return {
        "baseline_estimate": full_base,
        "comparison_estimate": full_comp,
        "estimate": float(full_base - full_comp),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n_pairs),
        "n_bootstrap": int(n_bootstrap),
        "method": method,
    }


def endpoint_projection_costs(latents: dict[str, Any], dim: int, seeds: tuple[int, ...]) -> list[np.ndarray]:
    z_terminal = latents["z_terminal"].detach().cpu().to(torch.float32)
    z_goal = latents["z_goal"].detach().cpu().to(torch.float32)
    out = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        projection = torch.randn((LATENT_DIM, dim), generator=generator, dtype=torch.float32) / math.sqrt(dim)
        out.append(squared_l2(z_terminal @ projection, z_goal @ projection).cpu().numpy().astype(np.float64))
    return out


def cube_pair_mean_values(cube_rpool: dict[str, Any], dim: int, key: str) -> np.ndarray:
    values_by_pair: dict[int, list[float]] = defaultdict(list)
    for row in cube_rpool["per_pool"]:
        if int(row["dimension"]) != int(dim):
            continue
        value = clean_float(row.get(key))
        if value is not None:
            values_by_pair[int(row["pair_id"])].append(value)
    return np.asarray(
        [float(np.mean(values_by_pair[pair_id])) for pair_id in sorted(values_by_pair)],
        dtype=np.float64,
    )


def mppi_pair_arrays(data: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    cem = {int(row["pair_id"]): row for row in data["cem_per_pair"]}
    mppi = {int(row["pair_id"]): row for row in data["mppi_per_pair"]}
    pair_ids = sorted(set(cem) & set(mppi))
    return (
        np.asarray([float(cem[pair_id][key]) for pair_id in pair_ids], dtype=np.float64),
        np.asarray([float(mppi[pair_id][key]) for pair_id in pair_ids], dtype=np.float64),
    )


def subspace_pair_arrays(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    default_by_pair = {int(row["pair_id"]): float(bool(row["rank1_success"])) for row in data["default_cem"]}
    subspace_by_pair: dict[int, list[float]] = defaultdict(list)
    for row in data["subspace_cem"]:
        subspace_by_pair[int(row["pair_id"])].append(float(bool(row["rank1_success"])))
    pair_ids = sorted(default_by_pair)
    return (
        np.asarray([default_by_pair[pair_id] for pair_id in pair_ids], dtype=np.float64),
        np.asarray([float(np.mean(subspace_by_pair[pair_id])) for pair_id in pair_ids], dtype=np.float64),
    )


def fmt_number(value: float, digits: int = 3) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.{digits}f}"


def fmt_ci(stat: dict[str, Any], digits: int = 3, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "\\%" if percent else ""
    return f"[{fmt_number(float(stat['ci_low']) * scale, digits)}, {fmt_number(float(stat['ci_high']) * scale, digits)}]{suffix}"


def fmt_est(value: float, digits: int = 3, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "\\%" if percent else ""
    return f"{fmt_number(float(value) * scale, digits)}{suffix}"


def fmt_pp(value: float, digits: int = 1) -> str:
    return f"{fmt_number(float(value), digits)} pp"


def fmt_pp_ci(stat: dict[str, Any], digits: int = 1) -> str:
    return f"[{fmt_number(float(stat['ci_low']), digits)}, {fmt_number(float(stat['ci_high']), digits)}] pp"


def fmt_percent_already_scaled(value: float, digits: int = 1) -> str:
    return f"{fmt_number(float(value), digits)}\\%"


def row(
    metric_name: str,
    environment: str,
    protocol_experiment: str,
    n_pairs: str | int,
    seeds: str | int,
    point_estimate: str,
    ci_95: str,
    method: str,
    raw: dict[str, Any],
) -> Row:
    return Row(
        metric_name=metric_name,
        environment=environment,
        protocol_experiment=protocol_experiment,
        n_pairs=str(n_pairs),
        seeds=str(seeds),
        point_estimate=point_estimate,
        ci_95=ci_95,
        method=method,
        raw=raw,
    )


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def latex_row(item: Row) -> str:
    cells = [
        item.metric_name,
        item.environment,
        item.protocol_experiment,
        item.n_pairs,
        item.seeds,
        item.point_estimate,
        item.ci_95,
        item.method,
    ]
    return " & ".join(cells) + r" \\"


def write_latex(path: Path, rows: list[Row]) -> None:
    lines = [
        r"% Generated by scripts/revision/appendix_uncertainty_table.py",
        r"\begin{table*}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}L{0.18\textwidth}L{0.07\textwidth}L{0.20\textwidth}L{0.05\textwidth}L{0.06\textwidth}L{0.15\textwidth}L{0.12\textwidth}L{0.15\textwidth}@{}}",
        r"\toprule",
        r"Metric & Env. & Protocol / experiment & $N$ & Seeds & Point estimate & 95\% CI & Method \\",
        r"\midrule",
    ]
    for idx, item in enumerate(rows):
        if idx in (8, 12):
            lines.append(r"\addlinespace")
        lines.append(latex_row(item))
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{Uncertainty summary for the main endpoint-pool, optimizer, and repair metrics. Bootstrap intervals use 10{,}000 resamples over the experimental unit shown in the Method column. Exact intervals are Clopper--Pearson binomial intervals. Endpoint global Spearman intervals bootstrap pair clusters after rank-transforming endpoint records, so the point estimates match the global Spearman values reported in the main tables.}",
            r"\label{tab:appendix-uncertainty}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def rows_to_json(rows: list[Row]) -> list[dict[str, Any]]:
    return [
        {
            "metric_name": item.metric_name,
            "environment": item.environment,
            "protocol_experiment": item.protocol_experiment,
            "n_pairs": item.n_pairs,
            "seeds": item.seeds,
            "point_estimate": item.point_estimate,
            "ci_95": item.ci_95,
            "method": item.method,
            "raw": item.raw,
        }
        for item in rows
    ]


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    bootstrap = load_json(args.bootstrap_path)
    pusht_stage1a = load_json(args.pusht_stage1a_path)
    cube_stage1a = load_json(args.cube_stage1a_path)
    cube_rpool = load_json(args.cube_rpool_path)
    mppi = load_json(args.mppi_path)
    track_b = load_json(args.track_b_path)
    subspace = load_json(args.subspace_path)
    split3 = load_json(args.cost_head_split3_path)
    cem_aware = load_json(args.cost_head_cem_aware_path)
    hybrid_k60 = load_json(args.hybrid_k60_path)

    latents = torch.load(args.pusht_latents_path, map_location="cpu", weights_only=False)
    labels = latents["C_real_state"].detach().cpu().numpy().astype(np.float64)
    v1_labels = latents["v1_cost"].detach().cpu().numpy().astype(np.float64)
    pair_ids = latents["pair_id"].detach().cpu().numpy().astype(np.int64)

    endpoint_stats = {}
    for dim in (64, 192):
        costs = endpoint_projection_costs(latents, dim, PROJECTION_SEEDS)
        endpoint_stats[dim] = rank_cluster_bootstrap(
            costs,
            labels,
            pair_ids,
            rng=rng,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
        )
        endpoint_stats[dim]["stage1a_point_estimate"] = stage1a_endpoint_value(pusht_stage1a, dim)

    dino_features = torch.load(args.dino_features_path, map_location="cpu", weights_only=False)
    lewm_cost = squared_l2(latents["z_terminal"], latents["z_goal"]).cpu().numpy().astype(np.float64)
    dino_cost = squared_l2(dino_features["d_terminal_mean"], dino_features["d_goal_mean"]).cpu().numpy().astype(np.float64)
    dino_gap = rank_cluster_diff_bootstrap(
        lewm_cost,
        dino_cost,
        v1_labels,
        pair_ids,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
    )
    track_rows = {row["encoder"]: row for row in track_b["rows"]}
    dino_gap["reported_lewm_global"] = float(track_rows["LeWM (SIGReg)"]["global_spearman"])
    dino_gap["reported_dino_mean_global"] = float(track_rows["DINOv2 mean-pool"]["global_spearman"])

    cube_endpoint_m64 = stage1a_endpoint_value(cube_stage1a, 64)
    cube_rpool_m64 = mean_ci(
        cube_pair_mean_values(cube_rpool, 64, "Rpool_Cmodel"),
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
    )
    cube_delta_values = cube_endpoint_m64 - cube_pair_mean_values(cube_rpool, 64, "Rpool_Cmodel")
    cube_delta = mean_ci(
        cube_delta_values,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
    )

    mppi_rpool_cem, mppi_rpool_mppi = mppi_pair_arrays(mppi, "Rpool_Cmodel")
    mppi_rpool_diff = paired_diff_ci(
        mppi_rpool_cem,
        mppi_rpool_mppi,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
    )
    existing_mppi_rpool_diff = dict(bootstrap["metrics"]["mppi_minus_cem_rpool_cmodel"])
    existing_mppi_rpool_diff["baseline_estimate"] = mppi_rpool_diff["baseline_estimate"]
    existing_mppi_rpool_diff["comparison_estimate"] = mppi_rpool_diff["comparison_estimate"]
    mppi_success_cem, mppi_success_mppi = mppi_pair_arrays(mppi, "planning_success")
    mppi_success_diff = paired_diff_ci(
        mppi_success_cem,
        mppi_success_mppi,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        scale=100.0,
    )
    mppi_creal_cem, mppi_creal_mppi = mppi_pair_arrays(mppi, "pool_Creal_std")
    mppi_creal_diff = paired_diff_ci(
        mppi_creal_cem,
        mppi_creal_mppi,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
    )

    sub_default, sub_space = subspace_pair_arrays(subspace)
    subspace_delta = paired_diff_ci(
        sub_default,
        sub_space,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        scale=100.0,
    )

    split3_successes = int(round(float(split3["aggregate"]["cpsi_success_rate"]) * int(split3["aggregate"]["n_pairs"])))
    split3_ci = clopper_pearson(split3_successes, int(split3["aggregate"]["n_pairs"]), alpha=args.alpha)
    cem_aware_successes = int(
        round(float(cem_aware["aggregate"]["cpsi_success_rate"]) * int(cem_aware["aggregate"]["n_pairs"]))
    )
    cem_aware_ci = clopper_pearson(cem_aware_successes, int(cem_aware["aggregate"]["n_pairs"]), alpha=args.alpha)
    hybrid_ci = clopper_pearson(
        int(hybrid_k60["aggregate"]["success_count"]),
        int(hybrid_k60["aggregate"]["n_pairs"]),
        alpha=args.alpha,
    )

    metrics = bootstrap["metrics"]
    repairs = bootstrap["repair_failures"]
    rows = [
        row(
            r"$R_\mathrm{endpoint}$ ($m=64$)",
            "PushT",
            "Stage 1A C2 endpoint projection",
            100,
            "10 proj.",
            fmt_est(endpoint_stats[64]["stage1a_point_estimate"]),
            fmt_ci(endpoint_stats[64]),
            "pair bootstrap",
            endpoint_stats[64],
        ),
        row(
            r"$R_\mathrm{endpoint}$ ($m=192$)",
            "PushT",
            "Stage 1A C2 endpoint projection",
            100,
            "10 proj.",
            fmt_est(endpoint_stats[192]["stage1a_point_estimate"]),
            fmt_ci(endpoint_stats[192]),
            "pair bootstrap",
            endpoint_stats[192],
        ),
        row(
            r"$R_\mathrm{pool}(C_\mathrm{model})$",
            "PushT",
            "Default CEM final pools",
            100,
            1,
            fmt_est(metrics["rpool_cmodel_overall"]["estimate"]),
            fmt_ci(metrics["rpool_cmodel_overall"]),
            "pair bootstrap",
            metrics["rpool_cmodel_overall"],
        ),
        row(
            r"$R_\mathrm{pool}(C_\mathrm{V3})$",
            "PushT",
            "Actual-terminal encoder ranker",
            100,
            1,
            fmt_est(metrics["rpool_v3_overall"]["estimate"]),
            fmt_ci(metrics["rpool_v3_overall"]),
            "pair bootstrap",
            metrics["rpool_v3_overall"],
        ),
        row(
            r"$R_\mathrm{pool}(C_\mathrm{V1})$",
            "PushT",
            "Physical hinge ranker",
            100,
            1,
            fmt_est(metrics["rpool_v1_effective_overall"]["estimate"]),
            fmt_ci(metrics["rpool_v1_effective_overall"]),
            "pair bootstrap",
            metrics["rpool_v1_effective_overall"],
        ),
        row(
            r"$\Delta_\mathrm{CEM}$",
            "PushT",
            "C0 endpoint ref. minus default pool",
            100,
            1,
            fmt_est(metrics["delta_cem_overall"]["estimate"]),
            fmt_ci(metrics["delta_cem_overall"]),
            "pair bootstrap",
            metrics["delta_cem_overall"],
        ),
        row(
            r"$R_\mathrm{pool}(C_\mathrm{model})$",
            "Cube",
            "Full projected CEM, $m=64$",
            50,
            "3 proj.",
            fmt_est(cube_rpool_m64["estimate"]),
            fmt_ci(cube_rpool_m64),
            "pair bootstrap",
            cube_rpool_m64,
        ),
        row(
            r"$\Delta_\mathrm{CEM}$",
            "Cube",
            "Endpoint $m=64$ minus full CEM pool",
            50,
            "3 proj.",
            fmt_est(cube_delta["estimate"]),
            fmt_ci(cube_delta),
            "pair bootstrap",
            cube_delta,
        ),
        row(
            r"$R_\mathrm{pool}(C_\mathrm{model})$ diff.",
            "PushT",
            "MPPI minus CEM",
            30,
            3,
            f"{fmt_est(existing_mppi_rpool_diff['estimate'])} ({fmt_est(existing_mppi_rpool_diff['baseline_estimate'])} to {fmt_est(existing_mppi_rpool_diff['comparison_estimate'])})",
            fmt_ci(existing_mppi_rpool_diff),
            "paired bootstrap",
            existing_mppi_rpool_diff,
        ),
        row(
            "Planning success diff.",
            "PushT",
            "MPPI minus CEM",
            30,
            3,
            f"{fmt_pp(mppi_success_diff['estimate'])} ({fmt_percent_already_scaled(mppi_success_diff['baseline_estimate'])} to {fmt_percent_already_scaled(mppi_success_diff['comparison_estimate'])})",
            fmt_pp_ci(mppi_success_diff),
            "paired bootstrap",
            mppi_success_diff,
        ),
        row(
            r"Pool $C_\mathrm{real}$ std diff.",
            "PushT",
            "MPPI minus CEM",
            30,
            3,
            f"{fmt_est(mppi_creal_diff['estimate'])} ({fmt_est(mppi_creal_diff['baseline_estimate'])} to {fmt_est(mppi_creal_diff['comparison_estimate'])})",
            fmt_ci(mppi_creal_diff),
            "paired bootstrap",
            mppi_creal_diff,
        ),
        row(
            "MPPI attribution fraction",
            "PushT",
            "CEM-to-MPPI gap fraction",
            30,
            3,
            fmt_est(metrics["cem_specific_attribution_fraction"]["estimate"], percent=True),
            fmt_ci(metrics["cem_specific_attribution_fraction"], percent=True),
            "paired bootstrap",
            metrics["cem_specific_attribution_fraction"],
        ),
        row(
            "Cost head Split 3 success",
            "PushT",
            "Learned cost repair",
            16,
            1,
            f"{split3_ci['successes']}/{split3_ci['trials']} ({fmt_est(split3_ci['estimate'], percent=True)})",
            fmt_ci(split3_ci, percent=True),
            "Clopper-Pearson",
            split3_ci,
        ),
        row(
            "Cost head CEM-aware success",
            "PushT",
            "Learned cost repair",
            16,
            1,
            f"{cem_aware_ci['successes']}/{cem_aware_ci['trials']} ({fmt_est(cem_aware_ci['estimate'], percent=True)})",
            fmt_ci(cem_aware_ci, percent=True),
            "Clopper-Pearson",
            cem_aware_ci,
        ),
        row(
            "DINOv2 gap vs. LeWM",
            "PushT",
            "Track B endpoint replacement",
            100,
            1,
            f"{fmt_est(dino_gap['estimate'])} ({fmt_est(dino_gap['comparison_estimate'])} vs. {fmt_est(dino_gap['baseline_estimate'])})",
            fmt_ci(dino_gap),
            "pair bootstrap",
            dino_gap,
        ),
        row(
            "Subspace-CEM success delta",
            "PushT",
            "Subspace minus default CEM",
            30,
            2,
            f"{fmt_pp(subspace_delta['estimate'])} ({fmt_percent_already_scaled(subspace_delta['baseline_estimate'])} to {fmt_percent_already_scaled(subspace_delta['comparison_estimate'])})",
            fmt_pp_ci(subspace_delta),
            "paired bootstrap",
            subspace_delta,
        ),
        row(
            "Hybrid CEM 20\\% threshold",
            "PushT",
            "V1 oracle re-rank, $K=60$",
            16,
            1,
            f"{hybrid_ci['successes']}/{hybrid_ci['trials']} ({fmt_est(hybrid_ci['estimate'], percent=True)})",
            fmt_ci(hybrid_ci, percent=True),
            "Clopper-Pearson",
            hybrid_ci,
        ),
    ]

    payload = {
        "metadata": {
            "format": "appendix_uncertainty_table_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": "scripts/revision/appendix_uncertainty_table.py",
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "alpha": float(args.alpha),
            "inputs": {
                "bootstrap": str(args.bootstrap_path.relative_to(PROJECT_ROOT)),
                "pusht_stage1a": str(args.pusht_stage1a_path.relative_to(PROJECT_ROOT)),
                "pusht_latents": str(args.pusht_latents_path.relative_to(PROJECT_ROOT)),
                "cube_stage1a": str(args.cube_stage1a_path.relative_to(PROJECT_ROOT)),
                "cube_rpool": str(args.cube_rpool_path.relative_to(PROJECT_ROOT)),
                "mppi": str(args.mppi_path.relative_to(PROJECT_ROOT)),
                "track_b": str(args.track_b_path.relative_to(PROJECT_ROOT)),
                "dino_features": str(args.dino_features_path.relative_to(PROJECT_ROOT)),
                "subspace": str(args.subspace_path.relative_to(PROJECT_ROOT)),
                "cost_head_split3": str(args.cost_head_split3_path.relative_to(PROJECT_ROOT)),
                "cost_head_cem_aware": str(args.cost_head_cem_aware_path.relative_to(PROJECT_ROOT)),
                "hybrid_k60": str(args.hybrid_k60_path.relative_to(PROJECT_ROOT)),
            },
        },
        "rows": rows_to_json(rows),
    }
    write_json(args.output, payload)
    write_latex(args.tex_output, rows)
    print(f"Wrote {args.output.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {args.tex_output.relative_to(PROJECT_ROOT)}")
    print("Key fresh intervals:")
    print(f"  PushT R_endpoint m=64: {fmt_est(endpoint_stats[64]['stage1a_point_estimate'])} {fmt_ci(endpoint_stats[64])}")
    print(f"  PushT R_endpoint m=192: {fmt_est(endpoint_stats[192]['stage1a_point_estimate'])} {fmt_ci(endpoint_stats[192])}")
    print(f"  Cube Rpool m=64: {fmt_est(cube_rpool_m64['estimate'])} {fmt_ci(cube_rpool_m64)}")
    print(f"  MPPI success diff: {fmt_pp(mppi_success_diff['estimate'])} {fmt_pp_ci(mppi_success_diff)}")
    print(f"  DINOv2 gap: {fmt_est(dino_gap['estimate'])} {fmt_ci(dino_gap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
