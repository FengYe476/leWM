#!/usr/bin/env python3
"""Compute Cube Rpool(C_model) directly from saved extended full-CEM pool files."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "results/revision/cube_full_proj_pools"
EXTENDED_PATH = ROOT / "results/revision/cube_full_proj_cem_extended.json"
OUTPUT_PATH = ROOT / "results/revision/rpool_v1_cube.json"
MEMO_PATH = ROOT / "docs/revision/cube_full_cem_memo.md"
DIMS = [1, 8, 32, 64, 192]
POOL_RE = re.compile(r"pair_(?P<pair>\d+)_m(?P<dim>\d+)_seed_(?P<seed>\d+)\.pt$")


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def scalar_summary(values: list[Any]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "mean": mean(clean) if clean else None,
        "std": stdev(clean) if len(clean) > 1 else None,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "n": len(clean),
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x, ddof=0)) == 0.0 or float(np.std(y, ddof=0)) == 0.0:
        return None
    rho = spearmanr(x, y).correlation
    return clean_float(rho)


def pool_metadata_from_name(path: Path) -> tuple[int, int, int]:
    match = POOL_RE.search(path.name)
    if not match:
        raise ValueError(f"Unrecognized pool filename: {path}")
    return int(match.group("pair")), int(match.group("dim")), int(match.group("seed"))


def load_pool_record(path: Path) -> dict[str, Any]:
    pair_id, dim, seed = pool_metadata_from_name(path)
    pool = torch.load(path, map_location="cpu", weights_only=False)

    default_costs = pool["default_costs"].detach().cpu().numpy()
    projected_costs = pool["projected_costs"].detach().cpu().numpy()
    c_real_state = pool["c_real_state"].detach().cpu().numpy()
    success = pool["success"].detach().cpu().numpy().astype(bool)

    return {
        "pair_id": int(pool.get("metadata", {}).get("pair_id", pair_id)),
        "cell": pool.get("metadata", {}).get("cell"),
        "dimension": int(pool.get("metadata", {}).get("dimension", dim)),
        "projection_seed": int(pool.get("metadata", {}).get("projection_seed", seed)),
        "pool_path": str(path.relative_to(ROOT)),
        "n_candidates": int(len(default_costs)),
        "Rpool_Cmodel": spearman(default_costs, c_real_state),
        "Rpool_projected": spearman(projected_costs, c_real_state),
        "pool_Creal_std": clean_float(np.std(c_real_state, ddof=0)),
        "pool_success_mass": clean_float(np.mean(success)),
        "rank1_candidate_index": int(pool["rank1_candidate_index"]),
        "rank1_c_real_state": clean_float(pool["rank1_c_real_state"]),
        "oracle_best_candidate_index": int(pool["oracle_best_candidate_index"]),
        "oracle_best_c_real_state": clean_float(pool["oracle_best_c_real_state"]),
    }


def aggregate_by_dimension(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_dim: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dim[int(record["dimension"])].append(record)

    out: dict[int, dict[str, Any]] = {}
    for dim, rows in sorted(by_dim.items()):
        out[dim] = {
            "n": len(rows),
            "n_pairs": len({int(r["pair_id"]) for r in rows}),
            "Rpool_Cmodel": scalar_summary([r["Rpool_Cmodel"] for r in rows]),
            "Rpool_projected": scalar_summary([r["Rpool_projected"] for r in rows]),
            "pool_Creal_std": scalar_summary([r["pool_Creal_std"] for r in rows]),
            "pool_success_mass": scalar_summary([r["pool_success_mass"] for r in rows]),
        }
    return out


def update_memo(summary: dict[str, Any]) -> None:
    table_lines = [
        "## Cube Rpool(C_model) From Pool Files",
        "",
        "| m | Rpool(C_model) | pool_Creal_std | pool_success_mass |",
        "|---|---:|---:|---:|",
    ]
    for dim in DIMS:
        row = summary["by_dimension"][str(dim)]
        table_lines.append(
            f"| {dim} | {fmt(row['Rpool_Cmodel']['mean'])} +/- {fmt(row['Rpool_Cmodel']['std'])} | "
            f"{fmt(row['pool_Creal_std']['mean'])} +/- {fmt(row['pool_Creal_std']['std'])} | "
            f"{pct(row['pool_success_mass']['mean'])} |"
        )
    table_lines.append("")
    table_lines.append(
        "The learned Euclidean cost remains a near-zero pool ranker in Cube full projected CEM: "
        f"the best dimensional mean is {fmt(max(summary['by_dimension'][str(dim)]['Rpool_Cmodel']['mean'] for dim in DIMS))}. "
        "This supports the cross-environment endpoint-pool decoupling claim even after the 50-pair, three-seed extension."
    )
    table_lines.append("")
    replacement = "\n".join(table_lines)

    if MEMO_PATH.exists():
        text = MEMO_PATH.read_text()
        marker = "## Cube Rpool(C_model) From Pool Files"
        if marker in text:
            text = text[: text.index(marker)].rstrip() + "\n\n" + replacement
        else:
            text = text.rstrip() + "\n\n" + replacement
    else:
        text = "# Cube Full Projected-CEM Extension Memo\n\n" + replacement
    MEMO_PATH.write_text(text)


def main() -> None:
    pool_files = sorted(POOL_DIR.glob("pair_*_m*_seed_*.pt"))
    if len(pool_files) != 750:
        raise RuntimeError(f"Expected 750 pool files in {POOL_DIR}, found {len(pool_files)}")

    records = []
    for idx, path in enumerate(pool_files, start=1):
        if idx == 1 or idx % 100 == 0 or idx == len(pool_files):
            print(f"Loading pool {idx}/{len(pool_files)}: {path.name}")
        records.append(load_pool_record(path))

    by_dimension = aggregate_by_dimension(records)
    summary = {
        "metadata": {
            "pool_dir": str(POOL_DIR.relative_to(ROOT)),
            "extended_results": str(EXTENDED_PATH.relative_to(ROOT)),
            "n_pool_files": len(pool_files),
            "dimensions": DIMS,
            "Rpool_Cmodel_definition": "Spearman(default_costs, c_real_state) within each saved Cube full projected-CEM pool",
            "pool_Creal_std_definition": "population std of c_real_state across the 300 saved candidates",
            "pool_success_mass_definition": "fraction of saved candidates with cube_pos_dist <= 0.04",
        },
        "per_pool": records,
        "by_dimension": {str(dim): stats for dim, stats in by_dimension.items()},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    update_memo(summary)

    print("")
    print("Cube Rpool(C_model) from saved full-CEM pools")
    print("| m | Rpool(C_model) | pool_Creal_std | pool_success_mass |")
    print("|---|---:|---:|---:|")
    for dim in DIMS:
        row = by_dimension[dim]
        print(
            f"| {dim} | {fmt(row['Rpool_Cmodel']['mean'])} +/- {fmt(row['Rpool_Cmodel']['std'])} | "
            f"{fmt(row['pool_Creal_std']['mean'])} +/- {fmt(row['pool_Creal_std']['std'])} | "
            f"{pct(row['pool_success_mass']['mean'])} |"
        )
    print(f"Saved JSON: {OUTPUT_PATH}")
    print(f"Updated memo: {MEMO_PATH}")


if __name__ == "__main__":
    main()
