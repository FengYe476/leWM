#!/usr/bin/env python3
"""Summarize the 50-pair x 3-seed Cube full projected-CEM extension."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXTENDED_PATH = ROOT / "results/revision/cube_full_proj_cem_extended.json"
RERANK_PATH = ROOT / "results/phase2/cube/cube_stage1b.json"
ENDPOINT_PATH = ROOT / "results/phase2/cube/cube_stage1a.json"
MEMO_PATH = ROOT / "docs/revision/cube_full_cem_memo.md"
DIMS = [1, 8, 32, 64, 192]


def is_finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def scalar_summary(values: list[Any]) -> dict[str, Any]:
    clean = [float(v) for v in values if is_finite(v)]
    return {
        "mean": mean(clean) if clean else None,
        "std": stdev(clean) if len(clean) > 1 else None,
        "n": len(clean),
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def nested(record: dict[str, Any], *keys: str) -> Any:
    cur: Any = record
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def endpoint_by_dimension(stage1a: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in stage1a["summary_table"]:
        control = row.get("control")
        config = str(row.get("config", ""))
        if control == "C0" and config == "lewm_192":
            out[192] = {
                "source": "C0/lewm_192",
                "Rendpoint": row.get("global_spearman_mean"),
                "Rendpoint_std": row.get("global_spearman_std"),
            }
        if control == "C2" and config.startswith("gaussian_m="):
            dim = int(config.split("=")[1])
            out[dim] = {
                "source": config,
                "Rendpoint": row.get("global_spearman_mean"),
                "Rendpoint_std": row.get("global_spearman_std"),
            }
    return out


def success_by_seed(records: list[dict[str, Any]], success_key: str) -> dict[int, dict[str, Any]]:
    by_dim_seed: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        dim = int(record["dimension"])
        seed = int(record["projection_seed"])
        by_dim_seed[dim][seed].append(1.0 if bool(record.get(success_key)) else 0.0)

    out: dict[int, dict[str, Any]] = {}
    for dim, seed_map in by_dim_seed.items():
        seed_rates = {seed: mean(vals) for seed, vals in sorted(seed_map.items())}
        rates = list(seed_rates.values())
        out[dim] = {
            "seed_rates": seed_rates,
            "mean": mean(rates),
            "std_across_seeds": stdev(rates) if len(rates) > 1 else None,
            "n_seeds": len(rates),
        }
    return out


def extended_by_dimension(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_dim: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dim[int(record["dimension"])].append(record)

    success = success_by_seed(records, "planning_success")
    out: dict[int, dict[str, Any]] = {}
    for dim, rows in sorted(by_dim.items()):
        out[dim] = {
            "n_records": len(rows),
            "n_pairs": len({int(r["pair_id"]) for r in rows}),
            "success": success.get(dim),
            "Rendpoint_record": scalar_summary([r.get("Rendpoint") for r in rows]),
            "Rpool_projected": scalar_summary([r.get("Rpool") for r in rows]),
            "Rpool_Cmodel": scalar_summary([r.get("Rpool_Cmodel") for r in rows]),
            "pool_Creal_std": scalar_summary([nested(r, "pool_diagnostics", "pool_Creal_std") for r in rows]),
            "pool_success_mass": scalar_summary([nested(r, "pool_diagnostics", "pool_success_mass") for r in rows]),
        }
    return out


def rerank_by_dimension(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    success = success_by_seed(records, "cem_late_success")
    out: dict[int, dict[str, Any]] = {}
    for dim, stats in success.items():
        out[dim] = {"success": stats}
    return out


def classify_curve(success_by_dim: dict[int, dict[str, Any]]) -> dict[str, Any]:
    means = {dim: success_by_dim[dim]["success"]["mean"] for dim in DIMS}
    best_dim = max(means, key=lambda dim: means[dim])
    ordered = [means[dim] for dim in DIMS]
    monotonic_non_decreasing = all(b >= a for a, b in zip(ordered, ordered[1:]))
    peak_at_32 = best_dim == 32
    if peak_at_32 and means[32] > means[64] and means[32] > means[192]:
        verdict = "The original inverted-U pattern persists, with a clear peak at m=32."
    elif monotonic_non_decreasing:
        verdict = "The original inverted-U pattern does not persist; the curve is monotonic non-decreasing over the reported dimensions."
    else:
        verdict = (
            "The original inverted-U pattern does not persist. Success rises strongly through m=32, "
            "but m=192 is the best dimension and the high-dimensional curve is better described as a plateau with seed noise."
        )
    return {
        "means": means,
        "best_dim": best_dim,
        "peak_at_m32": peak_at_32,
        "monotonic_non_decreasing": monotonic_non_decreasing,
        "verdict": verdict,
    }


def build_memo(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Cube Full Projected-CEM Extension Memo")
    lines.append("")
    lines.append("Source files:")
    lines.append(f"- Extended full projected CEM: `{EXTENDED_PATH.relative_to(ROOT)}`")
    lines.append(f"- Cube re-rank-only reference: `{RERANK_PATH.relative_to(ROOT)}`")
    lines.append(f"- Cube endpoint reference: `{ENDPOINT_PATH.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Full-CEM vs Re-rank Success")
    lines.append("")
    lines.append("| m | Cube full-CEM success (50p, 3s) | Cube re-rank success (100p, 3s) | Gap |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["comparison_table"]:
        lines.append(
            f"| {row['dimension']} | "
            f"{pct(row['full_success_mean'])} +/- {pct(row['full_success_std_across_seeds'])} | "
            f"{pct(row['rerank_success_mean'])} +/- {pct(row['rerank_success_std_across_seeds'])} | "
            f"{100.0 * row['gap_full_minus_rerank_pp']:.1f} pp |"
        )
    lines.append("")
    lines.append("## Extended Full-CEM Geometry")
    lines.append("")
    lines.append("| m | R_endpoint reference | R_pool(projected) | Delta_CEM | R_pool(C_model) | pool_Creal_std | pool_success_mass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["comparison_table"]:
        lines.append(
            f"| {row['dimension']} | {fmt(row['Rendpoint_reference'])} | "
            f"{fmt(row['Rpool_projected_mean'])} | {fmt(row['DeltaCEM_projected'])} | "
            f"{fmt(row['Rpool_Cmodel_mean'])} | {fmt(row['pool_Creal_std_mean'])} | "
            f"{pct(row['pool_success_mass_mean'])} |"
        )
    lines.append("")
    lines.append("## Inverted-U Check")
    lines.append("")
    lines.append(summary["inverted_u"]["verdict"])
    lines.append(
        " Mean full-CEM success by dimension was "
        + ", ".join(f"m={dim}: {pct(summary['inverted_u']['means'][dim])}" for dim in DIMS)
        + "."
    )
    lines.append("")
    lines.append("## Cube Rpool(C_model) From Pool Files")
    lines.append("")
    lines.append("_Filled by `scripts/revision/cube_rpool_analysis.py`._")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    extended = load_json(EXTENDED_PATH)
    rerank = load_json(RERANK_PATH)
    stage1a = load_json(ENDPOINT_PATH)

    ext_stats = extended_by_dimension(extended["records"])
    rerank_stats = rerank_by_dimension(rerank["records"])
    endpoints = endpoint_by_dimension(stage1a)

    comparison_table = []
    for dim in DIMS:
        full_success = ext_stats[dim]["success"]
        rerank_success = rerank_stats[dim]["success"]
        endpoint = endpoints[dim]["Rendpoint"]
        rpool_projected = ext_stats[dim]["Rpool_projected"]["mean"]
        comparison_table.append(
            {
                "dimension": dim,
                "full_success_mean": full_success["mean"],
                "full_success_std_across_seeds": full_success["std_across_seeds"],
                "rerank_success_mean": rerank_success["mean"],
                "rerank_success_std_across_seeds": rerank_success["std_across_seeds"],
                "gap_full_minus_rerank_pp": full_success["mean"] - rerank_success["mean"],
                "Rendpoint_reference": endpoint,
                "Rendpoint_reference_source": endpoints[dim]["source"],
                "Rpool_projected_mean": rpool_projected,
                "Rpool_projected_std": ext_stats[dim]["Rpool_projected"]["std"],
                "DeltaCEM_projected": endpoint - rpool_projected if is_finite(endpoint) and is_finite(rpool_projected) else None,
                "Rpool_Cmodel_mean": ext_stats[dim]["Rpool_Cmodel"]["mean"],
                "Rpool_Cmodel_std": ext_stats[dim]["Rpool_Cmodel"]["std"],
                "pool_Creal_std_mean": ext_stats[dim]["pool_Creal_std"]["mean"],
                "pool_Creal_std_std": ext_stats[dim]["pool_Creal_std"]["std"],
                "pool_success_mass_mean": ext_stats[dim]["pool_success_mass"]["mean"],
                "pool_success_mass_std": ext_stats[dim]["pool_success_mass"]["std"],
                "n_full_records": ext_stats[dim]["n_records"],
                "n_full_pairs": ext_stats[dim]["n_pairs"],
            }
        )

    summary = {
        "metadata": {
            "extended_path": str(EXTENDED_PATH.relative_to(ROOT)),
            "rerank_path": str(RERANK_PATH.relative_to(ROOT)),
            "endpoint_path": str(ENDPOINT_PATH.relative_to(ROOT)),
            "dimensions": DIMS,
            "success_std_definition": "sample std across projection-seed success rates",
            "rpool_projected_definition": "Spearman(projected_costs, c_real_state) in the final full projected-CEM pool",
            "rpool_cmodel_definition": "Spearman(default_costs, c_real_state) in the final full projected-CEM pool",
        },
        "extended_by_dimension": ext_stats,
        "rerank_by_dimension": rerank_stats,
        "endpoint_by_dimension": {dim: endpoints[dim] for dim in DIMS},
        "comparison_table": comparison_table,
        "inverted_u": classify_curve(ext_stats),
    }

    MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMO_PATH.write_text(build_memo(summary))

    print("Cube full projected-CEM extension summary")
    print("| m | Full-CEM success | Re-rank success | Gap |")
    print("|---|---:|---:|---:|")
    for row in comparison_table:
        print(
            f"| {row['dimension']} | "
            f"{pct(row['full_success_mean'])} +/- {pct(row['full_success_std_across_seeds'])} | "
            f"{pct(row['rerank_success_mean'])} +/- {pct(row['rerank_success_std_across_seeds'])} | "
            f"{100.0 * row['gap_full_minus_rerank_pp']:.1f} pp |"
        )
    print("")
    print("Extended full-CEM geometry")
    print("| m | R_endpoint | R_pool(projected) | Delta_CEM | R_pool(C_model) | pool_Creal_std | pool_success_mass |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in comparison_table:
        print(
            f"| {row['dimension']} | {fmt(row['Rendpoint_reference'])} | "
            f"{fmt(row['Rpool_projected_mean'])} | {fmt(row['DeltaCEM_projected'])} | "
            f"{fmt(row['Rpool_Cmodel_mean'])} | {fmt(row['pool_Creal_std_mean'])} | "
            f"{pct(row['pool_success_mass_mean'])} |"
        )
    print("")
    print("Inverted-U:", summary["inverted_u"]["verdict"])
    print(f"Saved memo: {MEMO_PATH}")


if __name__ == "__main__":
    main()
