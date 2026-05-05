#!/usr/bin/env python3
"""R_pool(V1) attribution analysis for PushT final CEM pools."""

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


LATENT_DIM = 192
N_EXPECTED_PAIRS = 100
N_CANDIDATES = 300
TOPK = 30
PROJECTION_DIMS = (8, 32, 64, 192)
PROJECTION_SEEDS = (0, 1, 2)
SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)

DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
DEFAULT_RERANK_PATH = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_MEMO_PATH = PROJECT_ROOT / "docs" / "revision" / "rpool_v1_memo.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--rerank-path", type=Path, default=DEFAULT_RERANK_PATH)
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


def tensor_to_numpy(pool: dict[str, Any], key: str, *, dtype: np.dtype) -> np.ndarray:
    value = pool[key]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected pool['{key}'] to be a tensor, got {type(value).__name__}")
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rho = spearmanr(x, y).statistic
    return clean_float(rho)


def rho_effective(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def scalar_summary(values: list[float | int | bool | None], *, none_as_zero: bool = False) -> dict[str, Any]:
    arr_values: list[float] = []
    n_missing = 0
    for value in values:
        if value is None:
            n_missing += 1
            if none_as_zero:
                arr_values.append(0.0)
            continue
        number = float(value)
        if math.isfinite(number):
            arr_values.append(number)
        else:
            n_missing += 1
            if none_as_zero:
                arr_values.append(0.0)
    arr = np.asarray(arr_values, dtype=np.float64)
    return {
        "mean": clean_float(float(arr.mean())) if len(arr) else None,
        "std": clean_float(float(arr.std(ddof=1))) if len(arr) > 1 else None,
        "min": clean_float(float(arr.min())) if len(arr) else None,
        "max": clean_float(float(arr.max())) if len(arr) else None,
        "n": int(len(arr)),
        "n_missing": int(n_missing),
        "none_as_zero": bool(none_as_zero),
        "ddof": 1,
    }


def make_projection(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((LATENT_DIM, int(dim)), generator=generator, dtype=torch.float32) / math.sqrt(int(dim))


def projected_costs(z_pred: np.ndarray, z_goal: np.ndarray, projection: torch.Tensor) -> np.ndarray:
    projection_np = projection.detach().cpu().numpy().astype(np.float32, copy=False)
    pred_proj = np.asarray(z_pred, dtype=np.float32) @ projection_np
    goal_proj = np.asarray(z_goal, dtype=np.float32) @ projection_np
    return np.sum((pred_proj - goal_proj[None, :]) ** 2, axis=1).astype(np.float64)


def validate_pool_shapes(pool: dict[str, Any], *, pair_id: int) -> None:
    expected_shapes = {
        "z_pred": (N_CANDIDATES, LATENT_DIM),
        "z_goal": (LATENT_DIM,),
        "blocked_actions": (N_CANDIDATES, 5, 10),
        "raw_actions": (N_CANDIDATES, 25, 2),
        "default_costs": (N_CANDIDATES,),
        "v1_hinge_costs": (N_CANDIDATES,),
        "c_real_state": (N_CANDIDATES,),
        "block_pos_dist": (N_CANDIDATES,),
        "angle_dist": (N_CANDIDATES,),
        "success": (N_CANDIDATES,),
    }
    for key, shape in expected_shapes.items():
        value = pool.get(key)
        if not torch.is_tensor(value):
            raise TypeError(f"pair_{pair_id}.pt missing tensor key {key!r}")
        if tuple(value.shape) != shape:
            raise ValueError(f"pair_{pair_id}.pt key {key!r} has shape {tuple(value.shape)}, expected {shape}")


def load_pair_metadata(path: Path) -> dict[int, dict[str, Any]]:
    data = load_json(path)
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"{path} is missing a list-valued 'pairs' key")
    out = {int(pair["pair_id"]): pair for pair in pairs}
    if len(out) != N_EXPECTED_PAIRS:
        raise ValueError(f"Expected {N_EXPECTED_PAIRS} Track A pairs, found {len(out)}")
    return out


def load_subset_membership(path: Path) -> tuple[dict[str, list[int]], dict[int, list[str]], dict[str, Any]]:
    data = load_json(path)
    anchors = data.get("metadata", {}).get("anchor_definitions")
    if not isinstance(anchors, dict):
        raise ValueError(f"{path} is missing metadata.anchor_definitions")

    subset_ids: dict[str, list[int]] = {}
    for subset in SUBSET_ORDER:
        entry = anchors.get(subset)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing anchor definition for subset {subset!r}")
        pair_ids = [int(pair_id) for pair_id in entry.get("pair_ids", [])]
        if not pair_ids:
            raise ValueError(f"Subset {subset!r} has no pair_ids")
        subset_ids[subset] = pair_ids

    if len(subset_ids["invisible_quadrant"]) != 16:
        raise ValueError(
            f"Expected 16 invisible-quadrant pairs, got {len(subset_ids['invisible_quadrant'])}"
        )

    pair_to_subsets: dict[int, list[str]] = {pair_id: [] for pair_id in range(N_EXPECTED_PAIRS)}
    for subset in SUBSET_ORDER:
        for pair_id in subset_ids[subset]:
            pair_to_subsets.setdefault(int(pair_id), []).append(subset)

    missing = [pair_id for pair_id in range(N_EXPECTED_PAIRS) if not pair_to_subsets.get(pair_id)]
    if missing:
        raise ValueError(f"Subset anchor definitions leave pair IDs unclassified: {missing}")

    return subset_ids, pair_to_subsets, anchors


def analyze_pair(
    *,
    pool: dict[str, Any],
    pair_metadata: dict[str, Any],
    subsets: list[str],
    projections: dict[tuple[int, int], torch.Tensor],
) -> dict[str, Any]:
    pair_id = int(pair_metadata["pair_id"])
    validate_pool_shapes(pool, pair_id=pair_id)

    default_costs = tensor_to_numpy(pool, "default_costs", dtype=np.float64)
    v1_costs = tensor_to_numpy(pool, "v1_hinge_costs", dtype=np.float64)
    c_real_state = tensor_to_numpy(pool, "c_real_state", dtype=np.float64)
    success = tensor_to_numpy(pool, "success", dtype=np.bool_)
    z_pred = tensor_to_numpy(pool, "z_pred", dtype=np.float32)
    z_goal = tensor_to_numpy(pool, "z_goal", dtype=np.float32)

    rank1_index = int(np.argmin(default_costs))
    top30_indices = np.argsort(default_costs, kind="mergesort")[:TOPK]
    c_real_min = float(np.min(c_real_state))
    selection_regret = float(c_real_state[rank1_index] - c_real_min)

    projection_records: dict[str, dict[str, Any]] = {}
    projection_seed_records: list[dict[str, Any]] = []
    for dim in PROJECTION_DIMS:
        seed_values: dict[str, float | None] = {}
        seed_effective_values: dict[str, float] = {}
        for seed in PROJECTION_SEEDS:
            costs = projected_costs(z_pred, z_goal, projections[(int(dim), int(seed))])
            rho = spearman_corr(costs, c_real_state)
            seed_values[str(seed)] = rho
            seed_effective_values[str(seed)] = rho_effective(rho)
            projection_seed_records.append(
                {
                    "m": int(dim),
                    "projection_seed": int(seed),
                    "Rpool_Cproj": rho,
                    "Rpool_Cproj_effective": rho_effective(rho),
                }
            )
        projection_records[str(dim)] = {
            "by_seed": seed_values,
            "by_seed_effective": seed_effective_values,
            "mean": scalar_summary(list(seed_values.values())),
            "mean_effective": scalar_summary(list(seed_values.values()), none_as_zero=True),
        }

    rpool_cmodel = spearman_corr(default_costs, c_real_state)
    rpool_v1 = spearman_corr(v1_costs, c_real_state)

    return {
        "pair_id": pair_id,
        "cell": str(pair_metadata["cell"]),
        "episode_id": int(pair_metadata["episode_id"]),
        "start_row": int(pair_metadata["start_row"]),
        "goal_row": int(pair_metadata["goal_row"]),
        "block_displacement_px": clean_float(pair_metadata.get("block_displacement_px")),
        "required_rotation_rad": clean_float(pair_metadata.get("required_rotation_rad")),
        "physical_pose_distance": clean_float(pair_metadata.get("physical_pose_distance")),
        "subsets": list(subsets),
        "rank1_index_default_cost": rank1_index,
        "pool_oracle_best_index": int(np.argmin(c_real_state)),
        "selection_regret": clean_float(selection_regret),
        "Rpool_Cmodel": rpool_cmodel,
        "Rpool_Cmodel_effective": rho_effective(rpool_cmodel),
        "Rpool_V1": rpool_v1,
        "Rpool_V1_effective": rho_effective(rpool_v1),
        "Rpool_Cproj": projection_records,
        "Rpool_Cproj_seed_records": projection_seed_records,
        "pool_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
        "pool_Creal_range": clean_float(float(np.max(c_real_state) - np.min(c_real_state))),
        "pool_success_mass": clean_float(float(np.mean(success))),
        "pool_Cmodel_std": clean_float(float(np.std(default_costs, ddof=0))),
        "top30_Creal_std": clean_float(float(np.std(c_real_state[top30_indices], ddof=0))),
        "top30_Cmodel_std": clean_float(float(np.std(default_costs[top30_indices], ddof=0))),
        "pool_Creal_min": clean_float(c_real_min),
        "pool_Creal_rank1": clean_float(float(c_real_state[rank1_index])),
        "pool_Cmodel_rank1": clean_float(float(default_costs[rank1_index])),
        "pool_success_rank1": bool(success[rank1_index]),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    projection_by_dim: dict[str, Any] = {}
    for dim in PROJECTION_DIMS:
        dim_key = str(dim)
        seed_mean_values = [record["Rpool_Cproj"][dim_key]["mean"]["mean"] for record in records]
        seed_mean_effective_values = [
            record["Rpool_Cproj"][dim_key]["mean_effective"]["mean"] for record in records
        ]
        all_seed_values: list[float | None] = []
        for record in records:
            all_seed_values.extend(record["Rpool_Cproj"][dim_key]["by_seed"].values())
        projection_by_dim[dim_key] = {
            "per_pair_seed_mean": scalar_summary(seed_mean_values),
            "per_pair_seed_mean_effective": scalar_summary(seed_mean_effective_values),
            "all_pair_seed_records": scalar_summary(all_seed_values),
            "all_pair_seed_records_effective": scalar_summary(all_seed_values, none_as_zero=True),
        }

    metrics = (
        "Rpool_Cmodel",
        "Rpool_Cmodel_effective",
        "Rpool_V1",
        "Rpool_V1_effective",
        "pool_Creal_std",
        "pool_Creal_range",
        "pool_success_mass",
        "pool_Cmodel_std",
        "top30_Creal_std",
        "top30_Cmodel_std",
        "selection_regret",
    )
    out = {"n_pairs": int(len(records))}
    for metric in metrics:
        out[metric] = scalar_summary([record.get(metric) for record in records])
    out["Rpool_Cproj_by_dim"] = projection_by_dim
    return out


def summarize_by_subset(
    records: list[dict[str, Any]],
    subset_ids: dict[str, list[int]],
) -> dict[str, Any]:
    by_id = {int(record["pair_id"]): record for record in records}
    out: dict[str, Any] = {}
    for subset in SUBSET_ORDER:
        subset_records = [by_id[pair_id] for pair_id in subset_ids[subset] if pair_id in by_id]
        out[subset] = summarize_records(subset_records)
        out[subset]["pair_ids"] = [int(pair_id) for pair_id in subset_ids[subset]]
    return out


def classify_mechanism(*, rpool_v1_eff: float | None, rpool_cmodel_eff: float | None, pool_std: float | None, low_std_threshold: float) -> str:
    if rpool_v1_eff is None or rpool_cmodel_eff is None or pool_std is None:
        return "Insufficient finite metrics for pre-registered classification"
    if rpool_v1_eff < 0.05 and pool_std <= low_std_threshold:
        return "CEM convergence compression is the dominant mechanism"
    if rpool_v1_eff > 0.15 and rpool_cmodel_eff < 0.05:
        return "Local representation failure is the dominant mechanism"
    if 0.05 <= rpool_v1_eff <= 0.15:
        return "Mixed mechanism"
    return "Pre-registered rules do not select one dominant mechanism"


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_summary(summary: dict[str, Any], key: str, *, digits: int = 3) -> str:
    stat = summary.get(key, {})
    if not isinstance(stat, dict):
        return "NA"
    value = stat.get("mean")
    std = stat.get("std")
    n = stat.get("n")
    if value is None:
        return f"NA (n={n})"
    if std is None:
        return f"{fmt(value, digits)} (n={n})"
    return f"{fmt(value, digits)} +/- {fmt(std, digits)} (n={n})"


def make_main_table(summary: dict[str, Any]) -> str:
    rows = [
        "| row | Rpool(C_model) | Rpool(V1) | Rpool(C_proj) | pool_Creal_std | pool_success_mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = summary["overall"]
    rows.append(
        "| baseline | "
        f"{fmt_summary(overall, 'Rpool_Cmodel_effective')} | "
        f"{fmt_summary(overall, 'Rpool_V1_effective')} | "
        "NA | "
        f"{fmt_summary(overall, 'pool_Creal_std')} | "
        f"{fmt_summary(overall, 'pool_success_mass')} |"
    )
    for dim in PROJECTION_DIMS:
        dim_summary = overall["Rpool_Cproj_by_dim"][str(dim)]["per_pair_seed_mean_effective"]
        rows.append(
            f"| m={dim} | NA | NA | "
            f"{fmt(dim_summary.get('mean'))} +/- {fmt(dim_summary.get('std'))} (n={dim_summary.get('n')}) | "
            f"{fmt_summary(overall, 'pool_Creal_std')} | "
            f"{fmt_summary(overall, 'pool_success_mass')} |"
        )
    return "\n".join(rows)


def make_subset_table(summary: dict[str, Any]) -> str:
    rows = [
        "| subset | n | Rpool(C_model) | Rpool(V1) | Cproj m=8 | Cproj m=32 | Cproj m=64 | Cproj m=192 | pool_Creal_std | pool_success_mass | selection_regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subset in SUBSET_ORDER:
        item = summary["by_subset"][subset]
        proj = item["Rpool_Cproj_by_dim"]
        rows.append(
            f"| {subset} | {item['n_pairs']} | "
            f"{fmt(item['Rpool_Cmodel_effective']['mean'])} | "
            f"{fmt(item['Rpool_V1_effective']['mean'])} | "
            f"{fmt(proj['8']['per_pair_seed_mean_effective']['mean'])} | "
            f"{fmt(proj['32']['per_pair_seed_mean_effective']['mean'])} | "
            f"{fmt(proj['64']['per_pair_seed_mean_effective']['mean'])} | "
            f"{fmt(proj['192']['per_pair_seed_mean_effective']['mean'])} | "
            f"{fmt(item['pool_Creal_std']['mean'])} | "
            f"{fmt(item['pool_success_mass']['mean'])} | "
            f"{fmt(item['selection_regret']['mean'])} |"
        )
    return "\n".join(rows)


def make_invisible_table(summary: dict[str, Any]) -> str:
    rows = [
        "| pair_id | cell | Rpool(V1) | Rpool(C_model) | pool_Creal_std | pool_success_mass | selection_regret |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary["invisible_quadrant_focus"]["pairs"]:
        rows.append(
            f"| {record['pair_id']} | {record['cell']} | "
            f"{fmt(record['Rpool_V1_effective'])} | "
            f"{fmt(record['Rpool_Cmodel_effective'])} | "
            f"{fmt(record['pool_Creal_std'])} | "
            f"{fmt(record['pool_success_mass'])} | "
            f"{fmt(record['selection_regret'])} |"
        )
    return "\n".join(rows)


def build_memo(result: dict[str, Any]) -> str:
    summary = result["summary"]
    overall = summary["overall"]
    invisible = summary["invisible_quadrant_focus"]["summary"]
    overall_decision = summary["interpretation"]["overall_decision"]
    invisible_decision = summary["interpretation"]["invisible_quadrant_decision"]
    low_threshold = summary["low_pool_Creal_std_threshold"]

    return "\n".join(
        [
            "# Rpool(V1) PushT Final-Pool Attribution Memo",
            "",
            f"Generated: `{result['metadata']['created_at']}`",
            "",
            "## Main Table",
            "",
            "Values are mean +/- sample std. Rank correlations use the effective convention for this memo: undefined Spearman from constant costs counts as `0.0` ranking signal.",
            "",
            make_main_table(summary),
            "",
            "## Per-Subset Breakdown",
            "",
            make_subset_table(summary),
            "",
            "## Invisible-Quadrant Focus",
            "",
            f"Low `pool_Creal_std` threshold: empirical Q25 = `{fmt(low_threshold)}`.",
            "",
            make_invisible_table(summary),
            "",
            "## Interpretation",
            "",
            (
                f"Overall, effective `Rpool(V1)` is `{fmt(overall['Rpool_V1_effective']['mean'])}`, "
                f"effective `Rpool(C_model)` is `{fmt(overall['Rpool_Cmodel_effective']['mean'])}`, "
                f"and mean `pool_Creal_std` is `{fmt(overall['pool_Creal_std']['mean'])}`. "
                f"By the pre-registered rules, the overall classification is: **{overall_decision}**."
            ),
            "",
            (
                f"For the 16 invisible-quadrant pairs, effective `Rpool(V1)` is "
                f"`{fmt(invisible['Rpool_V1_effective']['mean'])}`, effective `Rpool(C_model)` is "
                f"`{fmt(invisible['Rpool_Cmodel_effective']['mean'])}`, and mean `pool_Creal_std` is "
                f"`{fmt(invisible['pool_Creal_std']['mean'])}`. The invisible-quadrant classification is: "
                f"**{invisible_decision}**."
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.memo.parent.mkdir(parents=True, exist_ok=True)

    print("Rpool(V1) PushT analysis")
    print(f"pool_dir: {args.pool_dir}")
    print(f"pairs_path: {args.pairs_path}")
    print(f"rerank_path: {args.rerank_path}")

    pair_metadata = load_pair_metadata(args.pairs_path)
    subset_ids, pair_to_subsets, anchor_definitions = load_subset_membership(args.rerank_path)
    projections = {
        (int(dim), int(seed)): make_projection(int(dim), int(seed))
        for dim in PROJECTION_DIMS
        for seed in PROJECTION_SEEDS
    }

    records: list[dict[str, Any]] = []
    for idx, pair_id in enumerate(range(N_EXPECTED_PAIRS), start=1):
        pool_path = args.pool_dir / f"pair_{pair_id}.pt"
        if not pool_path.exists():
            raise FileNotFoundError(pool_path)
        if pair_id % 10 == 0:
            print(f"pair {pair_id}/100")
        pool = torch.load(pool_path, map_location="cpu", weights_only=False)
        pair_spec = pool.get("pair_spec", {})
        if int(pair_spec.get("pair_id", -1)) != pair_id:
            raise ValueError(f"{pool_path} pair_spec pair_id mismatch: {pair_spec.get('pair_id')}")
        records.append(
            analyze_pair(
                pool=pool,
                pair_metadata=pair_metadata[pair_id],
                subsets=pair_to_subsets[pair_id],
                projections=projections,
            )
        )

    if len(records) != N_EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {N_EXPECTED_PAIRS} records, got {len(records)}")

    pool_stds = np.asarray([float(record["pool_Creal_std"]) for record in records], dtype=np.float64)
    low_std_threshold = clean_float(float(np.quantile(pool_stds, 0.25)))
    if low_std_threshold is None:
        raise RuntimeError("Could not compute empirical low pool_Creal_std threshold")

    overall = summarize_records(records)
    by_subset = summarize_by_subset(records, subset_ids)
    invisible_records = [record for record in records if "invisible_quadrant" in record["subsets"]]
    invisible_records = sorted(invisible_records, key=lambda item: int(item["pair_id"]))
    if len(invisible_records) != 16:
        raise RuntimeError(f"Expected 16 invisible-quadrant records, got {len(invisible_records)}")

    projection_records_count = len(records) * len(PROJECTION_DIMS) * len(PROJECTION_SEEDS)
    if projection_records_count != N_EXPECTED_PAIRS * len(PROJECTION_DIMS) * len(PROJECTION_SEEDS):
        raise RuntimeError(f"Unexpected projection record count: {projection_records_count}")

    invisible_summary = summarize_records(invisible_records)
    interpretation = {
        "low_pool_Creal_std_rule": "pool_Creal_std <= empirical Q25 across 100 pairs",
        "overall_decision": classify_mechanism(
            rpool_v1_eff=overall["Rpool_V1_effective"]["mean"],
            rpool_cmodel_eff=overall["Rpool_Cmodel_effective"]["mean"],
            pool_std=overall["pool_Creal_std"]["mean"],
            low_std_threshold=float(low_std_threshold),
        ),
        "invisible_quadrant_decision": classify_mechanism(
            rpool_v1_eff=invisible_summary["Rpool_V1_effective"]["mean"],
            rpool_cmodel_eff=invisible_summary["Rpool_Cmodel_effective"]["mean"],
            pool_std=invisible_summary["pool_Creal_std"]["mean"],
            low_std_threshold=float(low_std_threshold),
        ),
    }

    result = {
        "metadata": {
            "format": "rpool_v1_pusht_revision_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(),
            "pool_dir": str(args.pool_dir),
            "pairs_path": str(args.pairs_path),
            "subset_source": str(args.rerank_path),
            "output": str(args.output),
            "memo": str(args.memo),
            "n_pairs": N_EXPECTED_PAIRS,
            "n_candidates_per_pair": N_CANDIDATES,
            "projection_dims": list(PROJECTION_DIMS),
            "projection_seeds": list(PROJECTION_SEEDS),
            "projection_records_count": int(projection_records_count),
            "projection_rule": "torch.randn((192, m), generator=torch.Generator(device='cpu').manual_seed(projection_seed)) / sqrt(m)",
            "spearman": {
                "function": "scipy.stats.spearmanr",
                "undefined_storage": None,
                "effective_rank_signal_for_decision_rules": "undefined Spearman counts as 0.0",
            },
            "low_pool_Creal_std_threshold": low_std_threshold,
            "anchor_definitions": anchor_definitions,
        },
        "per_pair": records,
        "summary": {
            "projection_records_count": int(projection_records_count),
            "low_pool_Creal_std_threshold": low_std_threshold,
            "overall": overall,
            "by_subset": by_subset,
            "invisible_quadrant_focus": {
                "pair_ids": subset_ids["invisible_quadrant"],
                "summary": invisible_summary,
                "pairs": [
                    {
                        "pair_id": int(record["pair_id"]),
                        "cell": record["cell"],
                        "Rpool_V1": record["Rpool_V1"],
                        "Rpool_V1_effective": record["Rpool_V1_effective"],
                        "Rpool_Cmodel": record["Rpool_Cmodel"],
                        "Rpool_Cmodel_effective": record["Rpool_Cmodel_effective"],
                        "pool_Creal_std": record["pool_Creal_std"],
                        "pool_success_mass": record["pool_success_mass"],
                        "selection_regret": record["selection_regret"],
                    }
                    for record in invisible_records
                ],
            },
            "interpretation": interpretation,
        },
    }

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(result), handle, indent=2)
        handle.write("\n")

    memo = build_memo(result)
    args.memo.write_text(memo, encoding="utf-8")

    print(f"loaded_pairs: {len(records)}")
    print(f"invisible_quadrant_pairs: {len(invisible_records)}")
    print(f"projection_records_count: {projection_records_count}")
    print(f"low_pool_Creal_std_threshold_q25: {fmt(low_std_threshold)}")
    print()
    print(make_main_table(result["summary"]))
    print()
    print(make_subset_table(result["summary"]))
    print()
    print(f"overall_decision: {interpretation['overall_decision']}")
    print(f"invisible_quadrant_decision: {interpretation['invisible_quadrant_decision']}")
    print(f"wrote_json: {args.output}")
    print(f"wrote_memo: {args.memo}")


if __name__ == "__main__":
    main()
