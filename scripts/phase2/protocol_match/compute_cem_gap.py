#!/usr/bin/env python3
"""Compute Block 2.2 CEM Compatibility Gap from offline artifacts.

The script reads Stage 1A endpoint controls plus available Stage 1B/protocol
matching JSONs. It does not load simulators, policies, checkpoints, or GPUs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PUSHT_STAGE1A = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1a_full.json"
DEFAULT_CUBE_STAGE1A = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1a.json"
DEFAULT_PUSHT_FULL_CEM = PROJECT_ROOT / "results" / "phase2" / "stage1" / "stage1b_full.json"
DEFAULT_PUSHT_RERANK = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_CUBE_RERANK = PROJECT_ROOT / "results" / "phase2" / "cube" / "cube_stage1b.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "cem_gap_table.json"

DIMENSIONS = (1, 2, 4, 8, 16, 32, 64, 128, 192)
SUBSET_ORDER = ("invisible_quadrant", "sign_reversal", "latent_favorable", "v1_favorable", "ordinary")
EXPECTED_SUBSET_COUNTS = {
    "invisible_quadrant": 16,
    "sign_reversal": 21,
    "latent_favorable": 12,
    "v1_favorable": 13,
    "ordinary": 47,
}
RHO_RET_THRESHOLD = 0.1
FLATNESS_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pusht-stage1a", type=Path, default=DEFAULT_PUSHT_STAGE1A)
    parser.add_argument("--cube-stage1a", type=Path, default=DEFAULT_CUBE_STAGE1A)
    parser.add_argument("--pusht-full-cem", type=Path, default=DEFAULT_PUSHT_FULL_CEM)
    parser.add_argument("--pusht-rerank", type=Path, default=DEFAULT_PUSHT_RERANK)
    parser.add_argument("--cube-rerank", type=Path, default=DEFAULT_CUBE_RERANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=False) + "\n")


def stat_mean(value: Any) -> float | None:
    if isinstance(value, dict):
        if "mean" in value:
            return clean_float(value.get("mean"))
        if "value" in value:
            return clean_float(value.get("value"))
        return None
    return clean_float(value)


def nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def mean_or_none(values: list[float | None]) -> float | None:
    arr = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))])
    return clean_float(arr.mean()) if len(arr) else None


def std_or_none(values: list[float | None]) -> float | None:
    arr = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))])
    return clean_float(arr.std(ddof=1)) if len(arr) > 1 else None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= RHO_RET_THRESHOLD:
        return None
    return clean_float(float(numerator) / float(denominator))


def load_stage1a_endpoint_by_dim(data: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    c2_by_dim = data.get("controls", {}).get("C2", {}).get("by_dim", {})
    for dim in DIMENSIONS:
        value = stat_mean(nested_get(c2_by_dim, (str(dim), "aggregate", "global_spearman")))
        if value is not None:
            out[int(dim)] = value

    if len(out) == len(DIMENSIONS):
        return out

    dim_pattern = re.compile(r"gaussian_m=(\d+)$")
    for row in data.get("summary_table", []):
        if not isinstance(row, dict) or row.get("control") != "C2":
            continue
        match = dim_pattern.match(str(row.get("config", "")))
        if match:
            out[int(match.group(1))] = clean_float(row.get("global_spearman_mean"))

    missing = [dim for dim in DIMENSIONS if dim not in out or out[dim] is None]
    if missing:
        raise ValueError(f"Stage 1A input is missing C2 endpoint Spearman for dimensions: {missing}")
    return out


def validate_anchor_definitions(anchor_definitions: dict[str, Any]) -> dict[str, list[int]]:
    subset_ids: dict[str, list[int]] = {}
    for name, expected_count in EXPECTED_SUBSET_COUNTS.items():
        entry = anchor_definitions.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing anchor definition: {name}")
        pair_ids = [int(item) for item in entry.get("pair_ids", [])]
        if len(pair_ids) != expected_count:
            raise ValueError(f"Unexpected anchor count for {name}: {len(pair_ids)} != {expected_count}")
        subset_ids[name] = pair_ids
    return subset_ids


def stage1a_subset_endpoint(
    *,
    stage1a_data: dict[str, Any],
    dim: int,
    pair_ids: list[int],
) -> dict[str, Any]:
    per_seed_values: list[float] = []
    per_seed_pair_counts: list[int] = []
    per_seed = nested_get(stage1a_data, ("controls", "C2", "by_dim", str(int(dim)), "per_seed"))
    if not isinstance(per_seed, list):
        raise ValueError(f"Stage 1A input is missing C2 per_seed records for dim={dim}")

    for seed_record in per_seed:
        by_pair = nested_get(seed_record, ("metrics", "per_pair_spearman", "by_pair"))
        if not isinstance(by_pair, dict):
            continue
        values: list[float] = []
        for pair_id in pair_ids:
            rho = nested_get(by_pair, (str(int(pair_id)), "spearman"))
            rho = clean_float(rho)
            if rho is not None:
                values.append(float(rho))
        if values:
            per_seed_values.append(float(np.mean(values)))
            per_seed_pair_counts.append(len(values))

    return {
        "mean": mean_or_none(per_seed_values),
        "std": std_or_none(per_seed_values),
        "n_seeds": int(len(per_seed_values)),
        "n_pairs_requested": int(len(pair_ids)),
        "per_seed_pair_counts": per_seed_pair_counts,
    }


def records_for_dim(data: dict[str, Any], dim: int) -> list[dict[str, Any]]:
    return [record for record in data.get("records", []) if int(record.get("dimension")) == int(dim)]


def observed_pair_ids_for_dim(data: dict[str, Any], dim: int) -> list[int]:
    return sorted({int(record["pair_id"]) for record in records_for_dim(data, dim)})


def group_for_dim(data: dict[str, Any], dim: int) -> dict[str, Any]:
    group = data.get("aggregate", {}).get("by_dimension", {}).get(str(int(dim)))
    if not isinstance(group, dict):
        raise ValueError(f"Planning result is missing aggregate.by_dimension.{dim}")
    return group


def group_for_dim_subset(data: dict[str, Any], dim: int, subset: str) -> dict[str, Any]:
    aggregate = data.get("aggregate", {})
    container = aggregate.get("by_dimension_x_subset")
    if not isinstance(container, dict):
        container = aggregate.get("by_dimension_and_subset")
    group = nested_get(container or {}, (str(int(dim)), str(subset)))
    if not isinstance(group, dict):
        raise ValueError(f"Planning result is missing dimension/subset aggregate for dim={dim}, subset={subset}")
    return group


def r_pool_from_group(group: dict[str, Any]) -> float | None:
    value = stat_mean(group.get("pool_spearman"))
    if value is not None:
        return value
    return stat_mean(group.get("endpoint_spearman"))


def rank1_from_group(group: dict[str, Any]) -> float | None:
    value = stat_mean(group.get("rank1_success_rate"))
    if value is not None:
        return value
    return stat_mean(group.get("projected_success_rate"))


def rerank_pool_success(data: dict[str, Any]) -> float | None:
    return stat_mean(nested_get(data, ("aggregate", "default_baselines", "candidate_pool_success_rate")))


def subset_rerank_pool_success(data: dict[str, Any], pair_ids: list[int]) -> float | None:
    pair_id_set = set(int(item) for item in pair_ids)
    values: list[float | None] = []
    for record in data.get("default_baselines", []):
        if int(record.get("pair_id")) not in pair_id_set:
            continue
        values.append(stat_mean(nested_get(record, ("candidate_pool", "success_rate"))))
    return mean_or_none(values)


def pusht_rerank_regret_from_group(group: dict[str, Any]) -> float | None:
    return stat_mean(group.get("selection_regret"))


def cube_regret_for_records(data: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = {int(record["pair_id"]): record for record in data.get("default_baselines", [])}
    regrets: list[float] = []
    missing = 0
    for record in records:
        pair_id = int(record["pair_id"])
        baseline = baselines.get(pair_id)
        rank1 = nested_get(record, ("projected_cem_diagnostics", "rank1_candidate_index"))
        c_real_state = nested_get(baseline or {}, ("candidate_pool", "c_real_state"))
        if rank1 is None or not isinstance(c_real_state, list):
            missing += 1
            continue
        rank1 = int(rank1)
        arr = np.asarray(c_real_state, dtype=np.float64)
        if not 0 <= rank1 < len(arr) or len(arr) == 0:
            missing += 1
            continue
        regrets.append(float(arr[rank1] - np.nanmin(arr)))
    return {
        "mean": mean_or_none(regrets),
        "std": std_or_none(regrets),
        "n": int(len(regrets)),
        "missing": int(missing),
    }


def base_record(
    *,
    env: str,
    protocol: str,
    dim: int,
    n_records: int,
    n_pairs: int,
    r_endpoint: float | None,
    r_pool: float | None,
    m_rank1: float | None,
    m_pool_success: float | None,
    regret: float | None,
    metric_notes: list[str],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    delta = clean_float(float(r_endpoint) - float(r_pool)) if r_endpoint is not None and r_pool is not None else None
    rho = safe_ratio(r_pool, r_endpoint)
    if r_endpoint is not None and r_endpoint <= RHO_RET_THRESHOLD:
        metric_notes.append(f"rho_ret omitted because R_endpoint <= {RHO_RET_THRESHOLD}")
    return {
        "environment": env,
        "protocol": protocol,
        "dimension": int(dim),
        "n_records": int(n_records),
        "n_pairs": int(n_pairs),
        "R_endpoint": clean_float(r_endpoint),
        "R_pool": clean_float(r_pool),
        "Delta_CEM": delta,
        "rho_ret": rho,
        "M_rank1": clean_float(m_rank1),
        "M_pool_success": clean_float(m_pool_success),
        "regret": clean_float(regret),
        "metric_notes": metric_notes,
        "source_paths": source_paths,
    }


def overall_records(
    *,
    pusht_stage1a_endpoint: dict[int, float],
    cube_stage1a_endpoint: dict[int, float],
    pusht_full: dict[str, Any],
    pusht_rerank: dict[str, Any],
    cube_rerank: dict[str, Any],
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    protocol_specs = [
        {
            "env": "PushT",
            "protocol": "full_projected_cem",
            "data": pusht_full,
            "endpoint": pusht_stage1a_endpoint,
            "planning_path_key": "pusht_full_cem",
            "stage1a_path_key": "pusht_stage1a",
        },
        {
            "env": "PushT",
            "protocol": "rerank_only",
            "data": pusht_rerank,
            "endpoint": pusht_stage1a_endpoint,
            "planning_path_key": "pusht_rerank",
            "stage1a_path_key": "pusht_stage1a",
        },
        {
            "env": "Cube",
            "protocol": "rerank_only",
            "data": cube_rerank,
            "endpoint": cube_stage1a_endpoint,
            "planning_path_key": "cube_rerank",
            "stage1a_path_key": "cube_stage1a",
        },
    ]

    for spec in protocol_specs:
        data = spec["data"]
        for dim in DIMENSIONS:
            group = group_for_dim(data, dim)
            dim_records = records_for_dim(data, dim)
            metric_notes: list[str] = []
            m_pool_success: float | None
            regret: float | None

            if spec["protocol"] == "full_projected_cem":
                m_pool_success = None
                regret = None
                metric_notes.append("Exact projected final-pool success mass is unavailable in stage1b_full.json")
                metric_notes.append("Exact projected final-pool oracle regret is unavailable in stage1b_full.json")
            elif spec["env"] == "Cube":
                m_pool_success = rerank_pool_success(data)
                regret_summary = cube_regret_for_records(data, dim_records)
                regret = regret_summary["mean"]
                if regret_summary["missing"]:
                    metric_notes.append(f"Cube regret missing for {regret_summary['missing']} records")
            else:
                m_pool_success = rerank_pool_success(data)
                regret = pusht_rerank_regret_from_group(group)

            out.append(
                base_record(
                    env=spec["env"],
                    protocol=spec["protocol"],
                    dim=dim,
                    n_records=int(group.get("n_records", len(dim_records))),
                    n_pairs=len(observed_pair_ids_for_dim(data, dim)),
                    r_endpoint=spec["endpoint"][dim],
                    r_pool=r_pool_from_group(group),
                    m_rank1=rank1_from_group(group),
                    m_pool_success=m_pool_success,
                    regret=regret,
                    metric_notes=metric_notes,
                    source_paths={
                        "stage1a": str(paths[spec["stage1a_path_key"]].relative_to(PROJECT_ROOT)),
                        "planning": str(paths[spec["planning_path_key"]].relative_to(PROJECT_ROOT)),
                    },
                )
            )
    return out


def subset_records(
    *,
    pusht_stage1a: dict[str, Any],
    pusht_full: dict[str, Any],
    pusht_rerank: dict[str, Any],
    subset_pair_ids: dict[str, list[int]],
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    specs = [
        ("full_projected_cem", pusht_full, "pusht_full_cem"),
        ("rerank_only", pusht_rerank, "pusht_rerank"),
    ]
    for protocol, data, planning_path_key in specs:
        for dim in DIMENSIONS:
            dim_observed_pair_ids = set(observed_pair_ids_for_dim(data, dim))
            for subset in SUBSET_ORDER:
                anchor_pair_ids = subset_pair_ids[subset]
                observed_subset_pair_ids = sorted(dim_observed_pair_ids & set(anchor_pair_ids))
                group = group_for_dim_subset(data, dim, subset)
                endpoint_summary = stage1a_subset_endpoint(
                    stage1a_data=pusht_stage1a,
                    dim=dim,
                    pair_ids=observed_subset_pair_ids,
                )
                metric_notes: list[str] = []
                if len(observed_subset_pair_ids) != len(anchor_pair_ids):
                    metric_notes.append(
                        f"Endpoint subset computed on {len(observed_subset_pair_ids)} observed protocol pairs "
                        f"out of {len(anchor_pair_ids)} anchor pairs"
                    )

                if protocol == "full_projected_cem":
                    m_pool_success = None
                    regret = None
                    metric_notes.append("Exact projected final-pool success mass is unavailable in stage1b_full.json")
                    metric_notes.append("Exact projected final-pool oracle regret is unavailable in stage1b_full.json")
                else:
                    m_pool_success = subset_rerank_pool_success(data, observed_subset_pair_ids)
                    regret = pusht_rerank_regret_from_group(group)

                record = base_record(
                    env="PushT",
                    protocol=protocol,
                    dim=dim,
                    n_records=int(group.get("n_records", 0)),
                    n_pairs=len(observed_subset_pair_ids),
                    r_endpoint=endpoint_summary["mean"],
                    r_pool=r_pool_from_group(group),
                    m_rank1=rank1_from_group(group),
                    m_pool_success=m_pool_success,
                    regret=regret,
                    metric_notes=metric_notes,
                    source_paths={
                        "stage1a": str(paths["pusht_stage1a"].relative_to(PROJECT_ROOT)),
                        "planning": str(paths[planning_path_key].relative_to(PROJECT_ROOT)),
                    },
                )
                record["subset"] = subset
                record["anchor_pair_count"] = len(anchor_pair_ids)
                record["observed_pair_ids"] = observed_subset_pair_ids
                record["R_endpoint_summary"] = endpoint_summary
                out.append(record)
    return out


def print_delta_table(records: list[dict[str, Any]]) -> None:
    headers = ["Env", "Protocol", "Dim", "R_endpoint", "R_pool", "Delta_CEM", "M_rank1"]
    rows = []
    for record in records:
        rows.append(
            [
                record["environment"],
                record["protocol"],
                str(record["dimension"]),
                fmt(record["R_endpoint"]),
                fmt(record["R_pool"]),
                fmt(record["Delta_CEM"]),
                fmt(record["M_rank1"]),
            ]
        )
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print("\nDelta_CEM by environment/protocol/dimension")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def print_flatness_summary(records: list[dict[str, Any]]) -> None:
    rerank = [
        record
        for record in records
        if record["environment"] == "PushT" and record["protocol"] == "rerank_only" and record["Delta_CEM"] is not None
    ]
    by_dim = {int(record["dimension"]): float(record["Delta_CEM"]) for record in rerank}
    values = [by_dim[dim] for dim in DIMENSIONS if dim in by_dim]
    if not values:
        print("\nPushT re-rank-only flatness: unavailable")
        return
    delta_range = max(values) - min(values)
    low_mean = float(np.mean([by_dim[dim] for dim in (1, 2, 4, 8) if dim in by_dim]))
    high_mean = float(np.mean([by_dim[dim] for dim in (64, 128, 192) if dim in by_dim]))
    label = "approximately flat" if delta_range <= FLATNESS_THRESHOLD else "not flat"
    print("\nPushT re-rank-only flatness")
    print(f"Delta_CEM range across dimensions: {delta_range:.4f}")
    print(f"Low-dim mean (1,2,4,8): {low_mean:.4f}")
    print(f"High-dim mean (64,128,192): {high_mean:.4f}")
    print(f"Conclusion by <= {FLATNESS_THRESHOLD:.2f} range rule: {label}")


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.4f}"


def main() -> int:
    args = parse_args()
    paths = {
        "pusht_stage1a": args.pusht_stage1a,
        "cube_stage1a": args.cube_stage1a,
        "pusht_full_cem": args.pusht_full_cem,
        "pusht_rerank": args.pusht_rerank,
        "cube_rerank": args.cube_rerank,
    }

    pusht_stage1a = load_json(args.pusht_stage1a)
    cube_stage1a = load_json(args.cube_stage1a)
    pusht_full = load_json(args.pusht_full_cem)
    pusht_rerank = load_json(args.pusht_rerank)
    cube_rerank = load_json(args.cube_rerank)

    pusht_endpoint = load_stage1a_endpoint_by_dim(pusht_stage1a)
    cube_endpoint = load_stage1a_endpoint_by_dim(cube_stage1a)
    subset_pair_ids = validate_anchor_definitions(pusht_rerank["metadata"]["anchor_definitions"])

    overall = overall_records(
        pusht_stage1a_endpoint=pusht_endpoint,
        cube_stage1a_endpoint=cube_endpoint,
        pusht_full=pusht_full,
        pusht_rerank=pusht_rerank,
        cube_rerank=cube_rerank,
        paths=paths,
    )
    by_subset = subset_records(
        pusht_stage1a=pusht_stage1a,
        pusht_full=pusht_full,
        pusht_rerank=pusht_rerank,
        subset_pair_ids=subset_pair_ids,
        paths=paths,
    )

    output = {
        "metadata": {
            "format": "cem_gap_table_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "inputs": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in paths.items()},
            "dimensions": list(DIMENSIONS),
            "rho_ret_threshold": RHO_RET_THRESHOLD,
            "flatness_threshold": FLATNESS_THRESHOLD,
            "anchor_definitions": {
                name: {
                    "pair_ids": subset_pair_ids[name],
                    "n_pairs": len(subset_pair_ids[name]),
                }
                for name in SUBSET_ORDER
            },
            "notes": [
                "PushT full projected CEM uses observed artifact counts; local stage1b_full.json has 63 pairs / 1701 records.",
                "PushT full projected CEM lacks exact projected final-pool simulator scores for all candidates; unavailable pool success and regret metrics are null.",
                "Subset endpoint Spearman uses Stage 1A C2 per-pair Spearman means on the protocol-observed subset pairs.",
            ],
        },
        "overall": overall,
        "by_subset": by_subset,
    }
    write_json(args.output, output)
    print_delta_table(overall)
    print_flatness_summary(overall)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
