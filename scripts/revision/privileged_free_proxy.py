#!/usr/bin/env python3
"""Privilege-free proxy validation for PushT R_pool revision artifacts."""

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
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RPOOL_PATH = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "revision" / "proxy_analysis_pusht.json"
DEFAULT_MEMO_PATH = PROJECT_ROOT / "docs" / "revision" / "proxy_analysis_memo.md"
PROXY_NAMES = (
    "pool_Cmodel_std",
    "top30_Cmodel_std",
    "C_model_dynamic_range",
    "elite_compression_ratio",
)
SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)
MONITORING_RHO_THRESHOLD = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpool-path", type=Path, default=DEFAULT_RPOOL_PATH)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--memo", type=Path, default=DEFAULT_MEMO_PATH)
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
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def spearman_corr(x: list[float | None], y: list[float | None]) -> dict[str, Any]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return {"rho": None, "p_value": None, "n": int(len(pairs))}
    x_arr = np.asarray([a for a, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([b for _, b in pairs], dtype=np.float64)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return {"rho": None, "p_value": None, "n": int(len(pairs))}
    result = spearmanr(x_arr, y_arr)
    return {
        "rho": clean_float(result.statistic),
        "p_value": clean_float(result.pvalue),
        "n": int(len(pairs)),
    }


def scalar_summary(values: list[float | None]) -> dict[str, Any]:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "mean": clean_float(float(arr.mean())) if len(arr) else None,
        "std": clean_float(float(arr.std(ddof=1))) if len(arr) > 1 else None,
        "min": clean_float(float(arr.min())) if len(arr) else None,
        "max": clean_float(float(arr.max())) if len(arr) else None,
        "n": int(len(arr)),
        "ddof": 1,
    }


def dynamic_range_from_record_or_pool(record: dict[str, Any], pool_dir: Path) -> tuple[float | None, str]:
    for key in ("C_model_dynamic_range", "default_cost_dynamic_range", "pool_Cmodel_range"):
        value = record.get(key)
        if value is not None:
            return clean_float(value), f"phase_b_json.{key}"

    pair_id = int(record["pair_id"])
    pool_path = pool_dir / f"pair_{pair_id}.pt"
    if not pool_path.exists():
        raise FileNotFoundError(pool_path)
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    default_costs = pool.get("default_costs")
    if not torch.is_tensor(default_costs):
        raise TypeError(f"{pool_path} is missing tensor key 'default_costs'")
    costs = default_costs.detach().cpu().numpy().astype(np.float64, copy=False)
    return clean_float(float(np.max(costs) - np.min(costs))), "pool_file.default_costs"


def build_pair_proxy_record(record: dict[str, Any], pool_dir: Path) -> dict[str, Any]:
    pool_cmodel_std = clean_float(record.get("pool_Cmodel_std"))
    top30_cmodel_std = clean_float(record.get("top30_Cmodel_std"))
    dynamic_range, dynamic_range_source = dynamic_range_from_record_or_pool(record, pool_dir)
    elite_ratio = None
    if pool_cmodel_std is not None and pool_cmodel_std > 0 and top30_cmodel_std is not None:
        elite_ratio = clean_float(float(top30_cmodel_std) / float(pool_cmodel_std))

    return {
        "pair_id": int(record["pair_id"]),
        "cell": str(record["cell"]),
        "subsets": list(record.get("subsets", [])),
        "pool_Cmodel_std": pool_cmodel_std,
        "top30_Cmodel_std": top30_cmodel_std,
        "C_model_dynamic_range": dynamic_range,
        "C_model_dynamic_range_source": dynamic_range_source,
        "elite_compression_ratio": elite_ratio,
        "selection_regret": clean_float(record.get("selection_regret")),
        "Rpool_Cmodel": clean_float(record.get("Rpool_Cmodel")),
        "Rpool_Cmodel_effective": clean_float(record.get("Rpool_Cmodel_effective")),
        "pool_Creal_std": clean_float(record.get("pool_Creal_std")),
        "pool_success_mass": clean_float(record.get("pool_success_mass")),
    }


def correlation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selection_regrets = [record.get("selection_regret") for record in records]
    rpool_cmodel = [record.get("Rpool_Cmodel") for record in records]
    rows: list[dict[str, Any]] = []
    for proxy in PROXY_NAMES:
        values = [record.get(proxy) for record in records]
        regret_corr = spearman_corr(values, selection_regrets)
        rpool_corr = spearman_corr(values, rpool_cmodel)
        rows.append(
            {
                "proxy_metric": proxy,
                "selection_regret": regret_corr,
                "Rpool_Cmodel": rpool_corr,
            }
        )
    return rows


def subset_records(records: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    return [record for record in records if subset in record.get("subsets", [])]


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "NA"
    if float(value) < 0.001:
        return "<0.001"
    return f"{float(value):.3f}"


def make_overall_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| proxy metric | corr with selection_regret | p(selection_regret) | corr with Rpool(C_model) | p(Rpool) | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        regret = row["selection_regret"]
        rpool = row["Rpool_Cmodel"]
        lines.append(
            f"| {row['proxy_metric']} | {fmt(regret['rho'])} | {fmt_p(regret['p_value'])} | "
            f"{fmt(rpool['rho'])} | {fmt_p(rpool['p_value'])} | {regret['n']} |"
        )
    return "\n".join(lines)


def make_subset_table(by_subset: dict[str, Any]) -> str:
    lines = [
        "| subset | proxy metric | corr with selection_regret | p(selection_regret) | corr with Rpool(C_model) | p(Rpool) | n |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subset in SUBSET_ORDER:
        for row in by_subset[subset]["correlations"]:
            regret = row["selection_regret"]
            rpool = row["Rpool_Cmodel"]
            lines.append(
                f"| {subset} | {row['proxy_metric']} | {fmt(regret['rho'])} | {fmt_p(regret['p_value'])} | "
                f"{fmt(rpool['rho'])} | {fmt_p(rpool['p_value'])} | {regret['n']} |"
            )
    return "\n".join(lines)


def build_memo(result: dict[str, Any]) -> str:
    decision = result["summary"]["decision"]
    cross = result["summary"]["cross_check_pool_Cmodel_std_vs_pool_Creal_std"]
    best = decision["best_overall_proxy_by_selection_regret"]
    best_subset = decision["best_subset_proxy_by_selection_regret"]
    if decision["global_usable_monitoring_signal_exists"]:
        conclusion = "A global privilege-free monitoring signal exists"
    elif decision["subset_specific_monitoring_signal_exists"]:
        conclusion = "A subset-specific privilege-free monitoring signal exists, but no global 100-pair proxy crossed the threshold"
    else:
        conclusion = "No privilege-free proxy crossed the pre-registered monitoring threshold"
    return "\n".join(
        [
            "# Privilege-Free Proxy Validation Memo",
            "",
            f"Generated: `{result['metadata']['created_at']}`",
            "",
            "## Overall Correlations",
            "",
            make_overall_table(result["summary"]["overall_correlations"]),
            "",
            "## Per-Subset Correlations",
            "",
            make_subset_table(result["summary"]["by_subset"]),
            "",
            "## Cross-Check",
            "",
            (
                f"`corr(pool_Cmodel_std, pool_Creal_std)` = `{fmt(cross['rho'])}` "
                f"with p-value `{fmt_p(cross['p_value'])}` over n=`{cross['n']}` pairs."
            ),
            "",
            "## Conclusion",
            "",
            (
                f"{conclusion}. The strongest proxy for `selection_regret` was "
                f"`{best['proxy_metric']}` with Spearman `{fmt(best['rho'])}` "
                f"(p=`{fmt_p(best['p_value'])}`, n=`{best['n']}`). "
                f"The strongest subset-specific proxy was `{best_subset['proxy_metric']}` in "
                f"`{best_subset['subset']}` with Spearman `{fmt(best_subset['rho'])}` "
                f"(p=`{fmt_p(best_subset['p_value'])}`, n=`{best_subset['n']}`). "
                f"The decision threshold was Spearman > `{MONITORING_RHO_THRESHOLD}`."
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.memo.parent.mkdir(parents=True, exist_ok=True)

    print("Privilege-free proxy analysis")
    print(f"rpool_path: {args.rpool_path}")
    print(f"pool_dir: {args.pool_dir}")

    rpool = load_json(args.rpool_path)
    phase_b_records = rpool.get("per_pair")
    if not isinstance(phase_b_records, list):
        raise ValueError(f"{args.rpool_path} is missing list key 'per_pair'")
    print(f"loaded_phase_b_pairs: {len(phase_b_records)}")

    records: list[dict[str, Any]] = []
    for idx, record in enumerate(phase_b_records, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"pair {idx}/{len(phase_b_records)}")
        records.append(build_pair_proxy_record(record, args.pool_dir))

    if len(records) != 100:
        raise RuntimeError(f"Expected 100 records, found {len(records)}")

    overall_correlations = correlation_rows(records)
    by_subset: dict[str, Any] = {}
    for subset in SUBSET_ORDER:
        items = subset_records(records, subset)
        by_subset[subset] = {
            "n_pairs": int(len(items)),
            "pair_ids": [int(record["pair_id"]) for record in items],
            "correlations": correlation_rows(items),
        }

    cross_check = spearman_corr(
        [record.get("pool_Cmodel_std") for record in records],
        [record.get("pool_Creal_std") for record in records],
    )

    best = max(
        overall_correlations,
        key=lambda row: float("-inf")
        if row["selection_regret"]["rho"] is None
        else float(row["selection_regret"]["rho"]),
    )
    best_regret = best["selection_regret"]
    global_usable_signal = best_regret["rho"] is not None and float(best_regret["rho"]) > MONITORING_RHO_THRESHOLD

    subset_candidates: list[dict[str, Any]] = []
    for subset, subset_summary in by_subset.items():
        for row in subset_summary["correlations"]:
            rho = row["selection_regret"]["rho"]
            subset_candidates.append(
                {
                    "subset": subset,
                    "proxy_metric": row["proxy_metric"],
                    "rho": rho,
                    "p_value": row["selection_regret"]["p_value"],
                    "n": row["selection_regret"]["n"],
                }
            )
    best_subset = max(
        subset_candidates,
        key=lambda row: float("-inf") if row["rho"] is None else float(row["rho"]),
    )
    subset_specific_signal = best_subset["rho"] is not None and float(best_subset["rho"]) > MONITORING_RHO_THRESHOLD

    result = {
        "metadata": {
            "format": "privilege_free_proxy_pusht_revision_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "rpool_path": str(args.rpool_path),
            "pool_dir": str(args.pool_dir),
            "output": str(args.output),
            "memo": str(args.memo),
            "n_pairs": int(len(records)),
            "proxy_metrics": list(PROXY_NAMES),
            "decision_rule": f"usable monitoring signal if any proxy has Spearman > {MONITORING_RHO_THRESHOLD} with selection_regret",
            "dynamic_range_fallback": "If absent from Phase B JSON, C_model_dynamic_range is computed from pool_file.default_costs only.",
        },
        "per_pair": records,
        "summary": {
            "proxy_summaries": {
                proxy: scalar_summary([record.get(proxy) for record in records])
                for proxy in PROXY_NAMES
            },
            "target_summaries": {
                "selection_regret": scalar_summary([record.get("selection_regret") for record in records]),
                "Rpool_Cmodel": scalar_summary([record.get("Rpool_Cmodel") for record in records]),
                "pool_Creal_std": scalar_summary([record.get("pool_Creal_std") for record in records]),
            },
            "overall_correlations": overall_correlations,
            "by_subset": by_subset,
            "cross_check_pool_Cmodel_std_vs_pool_Creal_std": cross_check,
            "decision": {
                "threshold": MONITORING_RHO_THRESHOLD,
                "global_usable_monitoring_signal_exists": bool(global_usable_signal),
                "subset_specific_monitoring_signal_exists": bool(subset_specific_signal),
                "usable_monitoring_signal_exists": bool(global_usable_signal or subset_specific_signal),
                "best_overall_proxy_by_selection_regret": {
                    "proxy_metric": best["proxy_metric"],
                    "rho": best_regret["rho"],
                    "p_value": best_regret["p_value"],
                    "n": best_regret["n"],
                },
                "best_subset_proxy_by_selection_regret": best_subset,
            },
        },
    }

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(result), handle, indent=2)
        handle.write("\n")
    args.memo.write_text(build_memo(result), encoding="utf-8")

    print()
    print(make_overall_table(overall_correlations))
    print()
    print(make_subset_table(by_subset))
    print()
    print(
        "cross_check corr(pool_Cmodel_std, pool_Creal_std): "
        f"rho={fmt(cross_check['rho'])}, p={fmt_p(cross_check['p_value'])}, n={cross_check['n']}"
    )
    print(
        "decision: "
        f"global_usable_monitoring_signal_exists={global_usable_signal}; "
        f"subset_specific_monitoring_signal_exists={subset_specific_signal}; "
        f"best_overall_proxy={best['proxy_metric']}; rho={fmt(best_regret['rho'])}; "
        f"p={fmt_p(best_regret['p_value'])}; "
        f"best_subset_proxy={best_subset['subset']}/{best_subset['proxy_metric']}; "
        f"rho={fmt(best_subset['rho'])}; p={fmt_p(best_subset['p_value'])}"
    )
    print(f"wrote_json: {args.output}")
    print(f"wrote_memo: {args.memo}")


if __name__ == "__main__":
    main()
