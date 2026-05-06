#!/usr/bin/env python3
"""Bootstrap confidence intervals for revision headline metrics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import beta, binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RPOOL_PATH = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_MPPI_PATH = PROJECT_ROOT / "results" / "revision" / "mppi_pool_analysis.json"
DEFAULT_V3_PATH = PROJECT_ROOT / "results" / "revision" / "v3_pool_analysis_pusht.json"
DEFAULT_STAGE1A_PATH = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_full.json"
DEFAULT_SUBSPACE_PATH = PROJECT_ROOT / "results" / "phase2" / "subspace_cem" / "stage_a_pusht.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "bootstrap_ci.json"
DEFAULT_MEMO = PROJECT_ROOT / "docs" / "revision" / "bootstrap_ci_memo.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpool-path", type=Path, default=DEFAULT_RPOOL_PATH)
    parser.add_argument("--mppi-path", type=Path, default=DEFAULT_MPPI_PATH)
    parser.add_argument("--v3-path", type=Path, default=DEFAULT_V3_PATH)
    parser.add_argument("--stage1a-path", type=Path, default=DEFAULT_STAGE1A_PATH)
    parser.add_argument("--subspace-path", type=Path, default=DEFAULT_SUBSPACE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memo", type=Path, default=DEFAULT_MEMO)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--endpoint-reference",
        type=float,
        default=None,
        help="Endpoint R reference for Delta_CEM. Defaults to Stage 1A C0 global Spearman.",
    )
    parser.add_argument(
        "--mppi-endpoint-reference",
        type=float,
        default=None,
        help="Endpoint R reference for MPPI attribution. Defaults to mppi_pool_analysis decision field.",
    )
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


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
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


def finite_array(values: list[Any]) -> np.ndarray:
    out = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            out.append(number)
    return np.asarray(out, dtype=np.float64)


def percentile_ci(samples: np.ndarray, *, alpha: float) -> tuple[float, float]:
    return (
        float(np.quantile(samples, alpha / 2.0)),
        float(np.quantile(samples, 1.0 - alpha / 2.0)),
    )


def bootstrap_stat(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    stat_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap_stat expects a non-empty 1D array")
    if stat_fn is None:
        stat_fn = lambda arr: float(np.mean(arr))
    n = int(len(values))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for idx in range(int(n_bootstrap)):
        sampled = values[rng.integers(0, n, size=n)]
        boot[idx] = stat_fn(sampled)
    ci_low, ci_high = percentile_ci(boot, alpha=alpha)
    return {
        "estimate": clean_float(stat_fn(values)),
        "ci_low": clean_float(ci_low),
        "ci_high": clean_float(ci_high),
        "n": n,
        "n_bootstrap": int(n_bootstrap),
        "method": "percentile bootstrap over pairs",
    }


def bootstrap_paired(
    cem: np.ndarray,
    mppi: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    method: str,
) -> dict[str, Any]:
    cem = np.asarray(cem, dtype=np.float64)
    mppi = np.asarray(mppi, dtype=np.float64)
    if cem.shape != mppi.shape or cem.ndim != 1:
        raise ValueError(f"Expected paired 1D arrays with same shape, got {cem.shape} and {mppi.shape}")
    n = int(len(cem))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for idx in range(int(n_bootstrap)):
        sample = rng.integers(0, n, size=n)
        boot[idx] = stat_fn(cem[sample], mppi[sample])
    ci_low, ci_high = percentile_ci(boot, alpha=alpha)
    return {
        "estimate": clean_float(stat_fn(cem, mppi)),
        "ci_low": clean_float(ci_low),
        "ci_high": clean_float(ci_high),
        "n": n,
        "n_bootstrap": int(n_bootstrap),
        "method": method,
    }


def clopper_pearson(k: int, n: int, *, alpha: float) -> dict[str, Any]:
    if not (0 <= int(k) <= int(n)) or int(n) <= 0:
        raise ValueError(f"Bad binomial count k={k}, n={n}")
    k = int(k)
    n = int(n)
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return {
        "successes": k,
        "trials": n,
        "estimate": clean_float(k / n),
        "ci_low": clean_float(lower),
        "ci_high": clean_float(upper),
        "method": f"Clopper-Pearson exact {100 * (1 - alpha):.1f}% binomial interval",
    }


def load_endpoint_reference(stage1a_path: Path) -> tuple[float, str]:
    data = load_json(stage1a_path)
    value = data["controls"]["C0"]["metrics"]["global_spearman"]
    return float(value), "results/phase2/stage1/stage1a_full.json controls.C0.metrics.global_spearman"


def rpool_values(data: dict[str, Any], key: str) -> np.ndarray:
    return finite_array([record.get(key) for record in data["per_pair"]])


def paired_mppi_arrays(data: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    cem_by_pair = {int(record["pair_id"]): record for record in data["cem_per_pair"]}
    mppi_by_pair = {int(record["pair_id"]): record for record in data["mppi_per_pair"]}
    pair_ids = sorted(set(cem_by_pair) & set(mppi_by_pair))
    missing_cem = sorted(set(mppi_by_pair) - set(cem_by_pair))
    missing_mppi = sorted(set(cem_by_pair) - set(mppi_by_pair))
    if missing_cem or missing_mppi:
        raise ValueError(f"Unmatched MPPI/CEM pairs: missing_cem={missing_cem}, missing_mppi={missing_mppi}")
    cem = finite_array([cem_by_pair[pair_id].get(key) for pair_id in pair_ids])
    mppi = finite_array([mppi_by_pair[pair_id].get(key) for pair_id in pair_ids])
    if len(cem) != len(pair_ids) or len(mppi) != len(pair_ids):
        raise ValueError(f"Metric {key!r} contains missing values in paired MPPI/CEM rows")
    return cem, mppi, pair_ids


def subspace_regression_stats(path: Path, *, alpha: float) -> dict[str, Any]:
    data = load_json(path)
    default_by_pair = {int(record["pair_id"]): bool(record["rank1_success"]) for record in data["default_cem"]}
    b_regress = 0
    c_improve = 0
    seed_pairs = 0
    for record in data["subspace_cem"]:
        pair_id = int(record["pair_id"])
        default_success = default_by_pair[pair_id]
        subspace_success = bool(record["rank1_success"])
        seed_pairs += 1
        if default_success and not subspace_success:
            b_regress += 1
        elif (not default_success) and subspace_success:
            c_improve += 1

    discordant = b_regress + c_improve
    exact = clopper_pearson(b_regress, discordant, alpha=alpha) if discordant else None
    mcnemar = binomtest(min(b_regress, c_improve), n=discordant, p=0.5, alternative="two-sided") if discordant else None
    all_seed_regression = clopper_pearson(b_regress, seed_pairs, alpha=alpha)
    return {
        "default_success": data["summary"]["overall"]["default"]["success"]["mean"],
        "subspace_success": data["summary"]["overall"]["subspace"]["success"]["mean"],
        "success_delta_subspace_minus_default": (
            float(data["summary"]["overall"]["subspace"]["success"]["mean"])
            - float(data["summary"]["overall"]["default"]["success"]["mean"])
        ),
        "discordant_regressions_default_success_subspace_failure": int(b_regress),
        "discordant_improvements_default_failure_subspace_success": int(c_improve),
        "discordant_total": int(discordant),
        "exact_conditional_regression_ci": exact,
        "exact_mcnemar_p_two_sided": clean_float(mcnemar.pvalue) if mcnemar is not None else None,
        "all_seed_pair_regression_rate_ci": all_seed_regression,
        "method_note": (
            "Seed-level paired comparison repeats each default CEM outcome for the two subspace projection seeds; "
            "b is default-success/subspace-failure and c is default-failure/subspace-success."
        ),
    }


def fmt(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.1f}%" if percent else f"{float(value):.3f}"


def metric_row(label: str, metric: dict[str, Any], *, percent: bool = False) -> str:
    return (
        f"| {label} | {fmt(metric['estimate'], percent=percent)} | "
        f"[{fmt(metric['ci_low'], percent=percent)}, {fmt(metric['ci_high'], percent=percent)}] | "
        f"{metric.get('n', metric.get('trials', ''))} | {metric['method']} |"
    )


def write_memo(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    repairs = payload["repair_failures"]
    lines = [
        "# Bootstrap Confidence Intervals",
        "",
        f"Generated: `{payload['metadata']['created_at']}`",
        f"Bootstrap resamples: {payload['metadata']['n_bootstrap']}",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Estimate | 95% CI | n | Method |",
        "|---|---:|---:|---:|---|",
        metric_row("Rpool(C_model), PushT same-pool", metrics["rpool_cmodel_overall"]),
    ]
    if "rpool_v3_overall" in metrics:
        lines.append(metric_row("Rpool(C_V3), PushT same-pool", metrics["rpool_v3_overall"]))
    lines.extend(
        [
            metric_row("Rpool(C_V1), effective", metrics["rpool_v1_effective_overall"]),
            metric_row("Delta_CEM = Rendpoint - Rpool(C_model)", metrics["delta_cem_overall"]),
            metric_row("MPPI - CEM Rpool(C_model)", metrics["mppi_minus_cem_rpool_cmodel"]),
            metric_row("CEM-specific attribution fraction", metrics["cem_specific_attribution_fraction"], percent=True),
            "",
            "Endpoint references:",
            f"- Delta_CEM reference: {fmt(payload['metadata']['endpoint_reference'])} ({payload['metadata']['endpoint_reference_source']})",
            f"- MPPI attribution reference: {fmt(payload['metadata']['mppi_endpoint_reference'])} ({payload['metadata']['mppi_endpoint_reference_source']})",
            "",
            "## Repair Failures",
            "",
            "| Repair | Estimate | 95% CI | n | Method |",
            "|---|---:|---:|---:|---|",
            metric_row("Cost head hard-pair success", repairs["cost_head_0_of_16"], percent=True),
        ]
    )
    sub = repairs["subspace_cem_regression"]
    all_seed = sub["all_seed_pair_regression_rate_ci"]
    lines.append(metric_row("Subspace-CEM regression among seed-pairs", all_seed, percent=True))
    if sub["exact_conditional_regression_ci"] is not None:
        cond = sub["exact_conditional_regression_ci"]
        lines.append(metric_row("Subspace-CEM regression among discordant pairs", cond, percent=True))
    lines.extend(
        [
            "",
            f"Subspace-CEM exact McNemar/binomial p-value: {fmt(sub['exact_mcnemar_p_two_sided'])}",
            f"Subspace-CEM success delta (subspace minus default): {fmt(sub['success_delta_subspace_minus_default'], percent=True)}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.rpool_path = args.rpool_path.expanduser().resolve()
    args.mppi_path = args.mppi_path.expanduser().resolve()
    args.v3_path = args.v3_path.expanduser().resolve()
    args.stage1a_path = args.stage1a_path.expanduser().resolve()
    args.subspace_path = args.subspace_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.memo = args.memo.expanduser().resolve()
    if int(args.n_bootstrap) <= 0:
        raise ValueError("--n-bootstrap must be positive")
    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError("--alpha must be in (0, 1)")

    rng = np.random.default_rng(int(args.seed))
    rpool = load_json(args.rpool_path)
    mppi = load_json(args.mppi_path)

    if args.endpoint_reference is None:
        endpoint_reference, endpoint_source = load_endpoint_reference(args.stage1a_path)
    else:
        endpoint_reference = float(args.endpoint_reference)
        endpoint_source = "command line --endpoint-reference"

    if args.mppi_endpoint_reference is None:
        mppi_endpoint_reference = float(mppi["decision"]["endpoint_reference_R"])
        mppi_endpoint_source = "results/revision/mppi_pool_analysis.json decision.endpoint_reference_R"
    else:
        mppi_endpoint_reference = float(args.mppi_endpoint_reference)
        mppi_endpoint_source = "command line --mppi-endpoint-reference"

    cmodel = rpool_values(rpool, "Rpool_Cmodel")
    v1_effective = rpool_values(rpool, "Rpool_V1_effective")
    metrics: dict[str, Any] = {
        "rpool_cmodel_overall": bootstrap_stat(
            cmodel,
            rng=rng,
            n_bootstrap=int(args.n_bootstrap),
            alpha=float(args.alpha),
        ),
        "rpool_v1_effective_overall": bootstrap_stat(
            v1_effective,
            rng=rng,
            n_bootstrap=int(args.n_bootstrap),
            alpha=float(args.alpha),
        ),
        "delta_cem_overall": bootstrap_stat(
            cmodel,
            rng=rng,
            n_bootstrap=int(args.n_bootstrap),
            alpha=float(args.alpha),
            stat_fn=lambda arr: float(endpoint_reference - np.mean(arr)),
        ),
    }
    metrics["delta_cem_overall"]["endpoint_reference"] = clean_float(endpoint_reference)

    if args.v3_path.exists():
        v3 = load_json(args.v3_path)
        v3_values = finite_array([record.get("Rpool_V3") for record in v3["per_pair"]])
        metrics["rpool_v3_overall"] = bootstrap_stat(
            v3_values,
            rng=rng,
            n_bootstrap=int(args.n_bootstrap),
            alpha=float(args.alpha),
        )

    cem_matched, mppi_matched, pair_ids = paired_mppi_arrays(mppi, "Rpool_Cmodel")
    metrics["mppi_minus_cem_rpool_cmodel"] = bootstrap_paired(
        cem_matched,
        mppi_matched,
        rng=rng,
        n_bootstrap=int(args.n_bootstrap),
        alpha=float(args.alpha),
        stat_fn=lambda cem, mppi_arr: float(np.mean(mppi_arr - cem)),
        method="paired percentile bootstrap over matched MPPI/CEM pairs",
    )
    metrics["cem_specific_attribution_fraction"] = bootstrap_paired(
        cem_matched,
        mppi_matched,
        rng=rng,
        n_bootstrap=int(args.n_bootstrap),
        alpha=float(args.alpha),
        stat_fn=lambda cem, mppi_arr: float(
            np.mean(mppi_arr - cem) / (mppi_endpoint_reference - np.mean(cem))
        ),
        method="paired percentile bootstrap of (Rpool_MPPI - Rpool_CEM) / (Rendpoint - Rpool_CEM)",
    )
    metrics["cem_specific_attribution_fraction"]["endpoint_reference"] = clean_float(mppi_endpoint_reference)

    repair_failures = {
        "cost_head_0_of_16": {
            **clopper_pearson(0, 16, alpha=float(args.alpha)),
            "description": "Cost-head repair successes on 16 hard pairs.",
        },
        "subspace_cem_regression": subspace_regression_stats(args.subspace_path, alpha=float(args.alpha)),
    }

    payload = {
        "metadata": {
            "format": "revision_bootstrap_ci",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "rpool_path": str(args.rpool_path),
            "mppi_path": str(args.mppi_path),
            "v3_path": str(args.v3_path) if args.v3_path.exists() else None,
            "stage1a_path": str(args.stage1a_path),
            "subspace_path": str(args.subspace_path),
            "output": str(args.output),
            "memo": str(args.memo),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "alpha": float(args.alpha),
            "confidence": clean_float(1.0 - float(args.alpha)),
            "endpoint_reference": clean_float(endpoint_reference),
            "endpoint_reference_source": endpoint_source,
            "mppi_endpoint_reference": clean_float(mppi_endpoint_reference),
            "mppi_endpoint_reference_source": mppi_endpoint_source,
            "mppi_pair_ids": pair_ids,
        },
        "metrics": metrics,
        "repair_failures": repair_failures,
    }
    write_json(args.output, payload)
    write_memo(args.memo, payload)

    print("== Bootstrap CI summary ==")
    for key, metric in metrics.items():
        print(f"{key}: {fmt(metric['estimate'])} [{fmt(metric['ci_low'])}, {fmt(metric['ci_high'])}]")
    print(
        "cost_head_0_of_16: "
        f"{fmt(repair_failures['cost_head_0_of_16']['estimate'], percent=True)} "
        f"[{fmt(repair_failures['cost_head_0_of_16']['ci_low'], percent=True)}, "
        f"{fmt(repair_failures['cost_head_0_of_16']['ci_high'], percent=True)}]"
    )
    print(f"saved: {args.output}")
    print(f"memo: {args.memo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
