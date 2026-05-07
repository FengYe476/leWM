#!/usr/bin/env python3
"""Full 30-pair, 3-seed in-loop CEM validation with the v3 warp ensemble."""

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
from method_local_warp import parse_int_list, set_seed, warped_costs  # noqa: E402
from method_warp_ensemble_v3 import LocalWarpV3  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    DEFAULT_PAIRS_PATH,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    TOPK,
    VAR_SCALE,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)
from scripts.phase2.stage1.projected_cem import (  # noqa: E402
    blocked_batch_to_raw_fast,
    score_raw_actions,
)
from scripts.phase2.train_cem_aware import rollout_candidate_latents  # noqa: E402


DEFAULT_MPPI_RUN = PROJECT_ROOT / "results" / "revision" / "mppi_pusht_30pair.json"
DEFAULT_MPPI_ANALYSIS = PROJECT_ROOT / "results" / "revision" / "mppi_pool_analysis.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "warp_v3_cem_full_30pair.json"
DEFAULT_POOL_ROOT = PROJECT_ROOT / "results" / "revision" / "warp_v3_cem_full_pools"
DEFAULT_ENSEMBLE_DIR = PROJECT_ROOT / "results" / "revision" / "warp_ensemble_v3"
DEFAULT_SEEDS = (0, 1, 2)
VARIANTS = ("default_cem", "v3_warp_cem")
SUBSET_ORDER = ("invisible_quadrant", "ordinary", "latent_favorable", "v1_favorable", "sign_reversal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--mppi-run", type=Path, default=DEFAULT_MPPI_RUN)
    parser.add_argument("--mppi-analysis", type=Path, default=DEFAULT_MPPI_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_selected_pairs(path: Path, *, pair_limit: int | None) -> list[dict[str, Any]]:
    data = load_json(path)
    selected = data.get("metadata", {}).get("selected_pairs")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"{path} is missing metadata.selected_pairs")
    selected_pairs = [dict(pair) for pair in selected]
    if pair_limit is not None:
        if int(pair_limit) < 1:
            raise ValueError("--pair-limit must be positive when provided")
        selected_pairs = selected_pairs[: int(pair_limit)]
    required = {"pair_id", "start_row", "goal_row", "primary_subset", "subset_memberships"}
    for pair in selected_pairs:
        missing = sorted(required - set(pair))
        if missing:
            raise ValueError(f"Selected pair {pair.get('pair_id')} missing keys: {missing}")
    return selected_pairs


def pool_path(pool_root: Path, *, variant: str, pair_id: int, seed: int) -> Path:
    return pool_root / variant / f"pair_{int(pair_id)}_seed_{int(seed)}.pt"


def run_key(record: dict[str, Any]) -> tuple[int, str, int]:
    return int(record["pair_id"]), str(record["variant"]), int(record["seed"])


def expected_keys(selected_pairs: list[dict[str, Any]], seeds: tuple[int, ...]) -> set[tuple[int, str, int]]:
    return {
        (int(pair["pair_id"]), variant, int(seed))
        for pair in selected_pairs
        for variant in VARIANTS
        for seed in seeds
    }


def load_existing_records(output_path: Path, *, resume: bool, expected: set[tuple[int, str, int]]) -> list[dict[str, Any]]:
    if not resume or not output_path.exists():
        return []
    data = load_json(output_path)
    records_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for record in data.get("records", []):
        try:
            key = run_key(record)
        except (KeyError, TypeError, ValueError):
            continue
        if key not in expected:
            continue
        pool = record.get("pool_path")
        if pool and Path(pool).exists():
            records_by_key[key] = record
    return [records_by_key[key] for key in sorted(records_by_key)]


def make_warp_from_payload(payload: dict[str, Any], *, device: torch.device | str) -> LocalWarpV3:
    config = payload.get("config", {})
    warp = LocalWarpV3(
        hidden=int(config.get("hidden", 32)),
        scale=float(config.get("scale", 0.05)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device=device)
    warp.load_state_dict(payload["model_state_dict"])
    warp.eval()
    return warp


def load_warps(ensemble_dir: Path, *, device: torch.device | str) -> list[nn.Module]:
    warps: list[nn.Module] = []
    for fold_idx in range(1, 11):
        path = ensemble_dir / f"warp_fold_{fold_idx}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing v3 warp checkpoint: {path}. Run method_warp_ensemble_v3.py first.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        warps.append(make_warp_from_payload(payload, device=device))
    return warps


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


@torch.no_grad()
def run_cem(
    *,
    model,
    prepared_info: dict[str, Any],
    warps: list[nn.Module],
    pair_id: int,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed) + int(pair_id) * 1009)
    mean = torch.zeros((1, PLANNING_HORIZON, ACTION_BLOCK * 2), dtype=torch.float32, device=device)
    sampling_std = VAR_SCALE * torch.ones_like(mean)
    final: dict[str, Any] | None = None
    diagnostics = []
    started = time.time()
    for warp in warps:
        warp.to(device=device)
        warp.eval()

    for iter_idx in range(1, CEM_ITERS + 1):
        candidates = torch.randn(
            1,
            NUM_SAMPLES,
            PLANNING_HORIZON,
            ACTION_BLOCK * 2,
            generator=generator,
            device=device,
        )
        candidates = candidates * sampling_std.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean
        z_pred, z_goal = rollout_candidate_latents(model, prepared_info, candidates)
        default_cost = torch.sum((z_pred - z_goal.unsqueeze(1).expand_as(z_pred)) ** 2, dim=-1)

        if variant == "default_cem":
            select_cost = default_cost[0]
            if iter_idx == CEM_ITERS:
                individual_costs = warp_cost_matrix(warps, z_pred[0], z_goal[0])
                warp_score = rank_average_score(individual_costs)
            else:
                individual_costs = torch.empty((0, NUM_SAMPLES), dtype=torch.float32, device=device)
                warp_score = torch.empty((0,), dtype=torch.float32, device=device)
        elif variant == "v3_warp_cem":
            individual_costs = warp_cost_matrix(warps, z_pred[0], z_goal[0])
            warp_score = rank_average_score(individual_costs)
            select_cost = warp_score
        else:
            raise ValueError(f"Unknown CEM variant: {variant}")

        top_vals, top_idx = torch.topk(select_cost, k=TOPK, largest=False)
        elite_candidates = candidates[:, top_idx]

        if iter_idx == CEM_ITERS:
            select_np = select_cost.detach().cpu().numpy().astype(np.float64)
            default_np = default_cost[0].detach().cpu().numpy().astype(np.float64)
            final = {
                "blocked_candidates": candidates[0].detach().cpu().numpy().astype(np.float32),
                "rank1_candidate_index": int(deterministic_argmin(select_np)),
                "select_costs": select_np,
                "default_costs": default_np,
                "warp_rank_avg_costs": warp_score.detach().cpu().numpy().astype(np.float64),
                "individual_warp_costs": individual_costs.detach().cpu().numpy().astype(np.float64),
                "z_pred": z_pred[0].detach().cpu().numpy().astype(np.float32),
                "z_goal": z_goal[0].detach().cpu().numpy().astype(np.float32),
                "elite_candidate_indices": top_idx.detach().cpu().numpy().astype(np.int64),
                "elite_costs": top_vals.detach().cpu().numpy().astype(np.float64),
                "elite_cost_std": clean_float(float(top_vals.detach().cpu().numpy().astype(np.float64).std(ddof=0))),
                "select_cost_dynamic_range": clean_float(float(np.max(select_np) - np.min(select_np))),
                "default_cost_dynamic_range": clean_float(float(np.max(default_np) - np.min(default_np))),
                "warp_cost_dynamic_range": clean_float(
                    float(np.max(final_warp := warp_score.detach().cpu().numpy().astype(np.float64)) - np.min(final_warp))
                )
                if warp_score.numel()
                else None,
            }

        mean = elite_candidates.mean(dim=1)
        sampling_std = elite_candidates.std(dim=1)
        diagnostics.append(
            {
                "iteration": int(iter_idx),
                "elite_cost_std": clean_float(float(top_vals.std(unbiased=False).detach().cpu().item())),
                "sampling_std_mean_post_update": clean_float(float(sampling_std.mean().detach().cpu().item())),
            }
        )

    if final is None:
        raise RuntimeError("CEM final iteration was not captured")
    final["iteration_diagnostics"] = diagnostics
    final["wallclock_seconds"] = clean_float(time.time() - started)
    return final


def build_pool_and_record(
    *,
    pair_spec: dict[str, Any],
    initial: dict[str, Any],
    goal: dict[str, Any],
    policy,
    env,
    cem_result: dict[str, Any],
    variant: str,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_id = int(pair_spec["pair_id"])
    raw_actions = blocked_batch_to_raw_fast(
        np.asarray(cem_result["blocked_candidates"], dtype=np.float32),
        action_processor=policy.process["action"],
    )
    v1_costs, c_real_state, success, metrics = score_raw_actions(
        env=env,
        initial_state=np.asarray(initial["state"], dtype=np.float32),
        goal_state=np.asarray(goal["state"], dtype=np.float32),
        raw_actions_batch=raw_actions,
        seed_base=int(seed) + pair_id * 100_000,
    )
    select_costs = np.asarray(cem_result["select_costs"], dtype=np.float64)
    default_costs = np.asarray(cem_result["default_costs"], dtype=np.float64)
    warp_costs = np.asarray(cem_result["warp_rank_avg_costs"], dtype=np.float64)
    rank1 = int(deterministic_argmin(select_costs))
    oracle = int(deterministic_argmin(c_real_state))
    pool = {
        "metadata": {
            "format": "pusht_warp_v3_cem_full_pool_v1",
            "created_at_unix": clean_float(time.time()),
            "pair_id": pair_id,
            "cell": str(pair_spec["cell"]),
            "start_row": int(pair_spec["start_row"]),
            "goal_row": int(pair_spec["goal_row"]),
            "seed": int(seed),
            "cem_sampling_seed": int(seed) + pair_id * 1009,
            "device": str(device),
            "variant": str(variant),
            "primary_subset": str(pair_spec["primary_subset"]),
            "subset_memberships": list(pair_spec["subset_memberships"]),
            "wallclock_seconds": cem_result["wallclock_seconds"],
        },
        "pair_spec": dict(pair_spec),
        "z_pred": torch.as_tensor(np.asarray(cem_result["z_pred"], dtype=np.float32)),
        "z_goal": torch.as_tensor(np.asarray(cem_result["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.as_tensor(np.asarray(cem_result["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.as_tensor(raw_actions.astype(np.float32)),
        "select_costs": torch.as_tensor(select_costs, dtype=torch.float64),
        "default_costs": torch.as_tensor(default_costs, dtype=torch.float64),
        "warp_rank_avg_costs": torch.as_tensor(warp_costs, dtype=torch.float64),
        "individual_warp_costs": torch.as_tensor(cem_result["individual_warp_costs"], dtype=torch.float64),
        "v1_hinge_costs": torch.as_tensor(v1_costs, dtype=torch.float64),
        "c_real_state": torch.as_tensor(c_real_state, dtype=torch.float64),
        "success": torch.as_tensor(success, dtype=torch.bool),
        "candidate_metrics": metrics,
        "variant": str(variant),
        "rank1_candidate_index": int(rank1),
        "oracle_best_candidate_index": int(oracle),
        "elite_candidate_indices": torch.as_tensor(cem_result["elite_candidate_indices"], dtype=torch.int64),
        "elite_costs": torch.as_tensor(cem_result["elite_costs"], dtype=torch.float64),
        "elite_cost_std_final": cem_result["elite_cost_std"],
        "iteration_diagnostics": cem_result["iteration_diagnostics"],
    }
    record = {
        "pair_id": pair_id,
        "cell": str(pair_spec["cell"]),
        "seed": int(seed),
        "variant": str(variant),
        "primary_subset": str(pair_spec["primary_subset"]),
        "subset_memberships": list(pair_spec["subset_memberships"]),
        "planning_success": bool(success[rank1]),
        "rank1_success": bool(success[rank1]),
        "rank1_candidate_index": int(rank1),
        "rank1_c_real": clean_float(float(c_real_state[rank1])),
        "oracle_best_candidate_index": int(oracle),
        "oracle_c_real": clean_float(float(c_real_state[oracle])),
        "selection_regret": clean_float(float(c_real_state[rank1] - c_real_state[oracle])),
        "Rpool_Cmodel": spearman_corr(default_costs, c_real_state),
        "Rpool_warp_rank_avg": spearman_corr(warp_costs, c_real_state),
        "Rpool_select": spearman_corr(select_costs, c_real_state),
        "Rpool_V1": spearman_corr(v1_costs, c_real_state),
        "pool_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
        "pool_Cmodel_std": clean_float(float(np.std(default_costs, ddof=0))),
        "pool_success_mass": clean_float(float(np.mean(success))),
        "elite_cost_std": cem_result["elite_cost_std"],
        "select_cost_dynamic_range": cem_result["select_cost_dynamic_range"],
        "default_cost_dynamic_range": cem_result["default_cost_dynamic_range"],
        "warp_cost_dynamic_range": cem_result["warp_cost_dynamic_range"],
        "pool_path": None,
    }
    return pool, record


def finite_mean(values: list[float | int | bool | None]) -> float | None:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return clean_float(float(arr.mean())) if len(arr) else None


def per_pair_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((int(record["pair_id"]), str(record["variant"])), []).append(record)
    out = []
    for (pair_id, variant), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        first = ordered[0]
        out.append(
            {
                "pair_id": int(pair_id),
                "cell": first["cell"],
                "variant": variant,
                "primary_subset": first["primary_subset"],
                "subset_memberships": first["subset_memberships"],
                "n_seeds": int(len(rows)),
                "seeds": [int(row["seed"]) for row in ordered],
                "planning_success": finite_mean([row["planning_success"] for row in rows]),
                "selection_regret": finite_mean([row["selection_regret"] for row in rows]),
                "Rpool_Cmodel": finite_mean([row["Rpool_Cmodel"] for row in rows]),
                "Rpool_warp_rank_avg": finite_mean([row["Rpool_warp_rank_avg"] for row in rows]),
                "Rpool_select": finite_mean([row["Rpool_select"] for row in rows]),
                "Rpool_V1": finite_mean([row["Rpool_V1"] for row in rows]),
                "pool_Creal_std": finite_mean([row["pool_Creal_std"] for row in rows]),
                "pool_Cmodel_std": finite_mean([row["pool_Cmodel_std"] for row in rows]),
                "pool_success_mass": finite_mean([row["pool_success_mass"] for row in rows]),
                "elite_cost_std": finite_mean([row["elite_cost_std"] for row in rows]),
            }
        )
    return out


def aggregate_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(rows)),
        "planning_success": scalar_summary([row.get("planning_success") for row in rows]),
        "selection_regret": scalar_summary([row.get("selection_regret") for row in rows]),
        "Rpool_Cmodel": scalar_summary([row.get("Rpool_Cmodel") for row in rows]),
        "Rpool_warp_rank_avg": scalar_summary([row.get("Rpool_warp_rank_avg") for row in rows]),
        "Rpool_select": scalar_summary([row.get("Rpool_select") for row in rows]),
        "Rpool_V1": scalar_summary([row.get("Rpool_V1") for row in rows]),
        "pool_Creal_std": scalar_summary([row.get("pool_Creal_std") for row in rows]),
        "pool_Cmodel_std": scalar_summary([row.get("pool_Cmodel_std") for row in rows]),
        "pool_success_mass": scalar_summary([row.get("pool_success_mass") for row in rows]),
        "elite_cost_std": scalar_summary([row.get("elite_cost_std") for row in rows]),
    }


def aggregate_by_variant(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        variant: aggregate_pairs([row for row in per_pair if row["variant"] == variant])
        for variant in VARIANTS
    }


def aggregate_by_subset(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in per_pair if row["variant"] == variant]
        out[variant] = {
            subset: aggregate_pairs([row for row in variant_rows if row.get("primary_subset") == subset])
            for subset in SUBSET_ORDER
        }
    return out


def load_mppi_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("summary", {})
    if "mppi_tau_1" not in summary:
        raise ValueError(f"{path} is missing summary.mppi_tau_1")
    return summary


def metric_mean(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return None


def fmt_metric(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.1f}%" if percent else f"{float(value):.3f}"


def make_comparison_table(summary_by_variant: dict[str, Any], mppi_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("Planning success (30p, 3s)", "planning_success", True),
        ("Rpool(C_model) mean", "Rpool_Cmodel", False),
        ("Rpool(warp rank avg) mean", "Rpool_warp_rank_avg", False),
        ("Mean selection regret", "selection_regret", False),
        ("Pool C_real_state std", "pool_Creal_std", False),
    ]
    mppi = mppi_summary["mppi_tau_1"]
    return [
        {
            "metric": label,
            "key": key,
            "default_cem": metric_mean(summary_by_variant["default_cem"], key),
            "v3_warp_cem": metric_mean(summary_by_variant["v3_warp_cem"], key),
            "mppi_tau_1": metric_mean(mppi, key),
            "format": "percent" if percent else "float",
        }
        for label, key, percent in rows
    ]


def make_subset_table(
    subset_by_variant: dict[str, Any],
    mppi_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    mppi_by_subset = mppi_summary.get("mppi_by_subset", {})
    rows = []
    for subset in SUBSET_ORDER:
        for label, key, percent in (
            ("Planning success", "planning_success", True),
            ("Rpool(C_model)", "Rpool_Cmodel", False),
            ("Rpool(warp rank avg)", "Rpool_warp_rank_avg", False),
            ("Mean selection regret", "selection_regret", False),
            ("Pool C_real_state std", "pool_Creal_std", False),
        ):
            rows.append(
                {
                    "subset": subset,
                    "metric": label,
                    "key": key,
                    "default_cem": metric_mean(subset_by_variant["default_cem"][subset], key),
                    "v3_warp_cem": metric_mean(subset_by_variant["v3_warp_cem"][subset], key),
                    "mppi_tau_1": metric_mean(mppi_by_subset.get(subset, {}), key),
                    "format": "percent" if percent else "float",
                    "n_pairs_default": subset_by_variant["default_cem"][subset]["n_pairs"],
                    "n_pairs_warp": subset_by_variant["v3_warp_cem"][subset]["n_pairs"],
                    "n_pairs_mppi": (mppi_by_subset.get(subset, {}) or {}).get("n_pairs", 0),
                }
            )
    return rows


def build_output(
    *,
    args: argparse.Namespace,
    selected_pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    mppi_summary: dict[str, Any],
    started: float,
    expected_count: int,
) -> dict[str, Any]:
    per_pair = per_pair_records(records)
    summary_by_variant = aggregate_by_variant(per_pair)
    subset_by_variant = aggregate_by_subset(per_pair)
    comparison = make_comparison_table(summary_by_variant, mppi_summary)
    subset_table = make_subset_table(subset_by_variant, mppi_summary)
    pair30 = [row for row in sorted(records, key=lambda item: (str(item["variant"]), int(item["seed"]))) if int(row["pair_id"]) == 30]
    return {
        "metadata": {
            "format": "pusht_warp_v3_cem_full_30pair_v1",
            "created_at_unix": clean_float(time.time()),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "pairs_path": str(args.pairs_path),
            "mppi_run": str(args.mppi_run),
            "mppi_analysis": str(args.mppi_analysis),
            "output": str(args.output),
            "pool_root": str(args.pool_root),
            "checkpoint_dir": str(args.checkpoint_dir),
            "ensemble_dir": str(args.ensemble_dir),
            "device": str(args.device),
            "seeds": [int(seed) for seed in args.seeds],
            "pair_ids": [int(pair["pair_id"]) for pair in selected_pairs],
            "n_pairs": int(len(selected_pairs)),
            "variants": list(VARIANTS),
            "expected_records": int(expected_count),
            "n_records": int(len(records)),
            "wallclock_seconds": clean_float(time.time() - started),
            "aggregation": "Seed runs are averaged within pair, then pair means are averaged across pairs.",
        },
        "records": sorted(records, key=lambda row: (int(row["pair_id"]), str(row["variant"]), int(row["seed"]))),
        "per_pair": per_pair,
        "summary_by_variant": summary_by_variant,
        "subset_by_variant": subset_by_variant,
        "mppi_summary": mppi_summary,
        "comparison_table": comparison,
        "subset_breakdown_table": subset_table,
        "pair30_spotlight": pair30,
    }


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    table = []
    for row in rows:
        percent = row["format"] == "percent"
        table.append(
            [
                row["metric"],
                fmt_metric(row["default_cem"], percent=percent),
                fmt_metric(row["v3_warp_cem"], percent=percent),
                fmt_metric(row["mppi_tau_1"], percent=percent),
            ]
        )
    print("\nAggregate comparison")
    print(
        tabulate(
            table,
            headers=["Metric", "Default CEM", "V3 Warp CEM", "MPPI (tau=1.0)"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def print_subset_table(rows: list[dict[str, Any]]) -> None:
    table = []
    for row in rows:
        percent = row["format"] == "percent"
        table.append(
            [
                row["subset"],
                row["metric"],
                fmt_metric(row["default_cem"], percent=percent),
                fmt_metric(row["v3_warp_cem"], percent=percent),
                fmt_metric(row["mppi_tau_1"], percent=percent),
            ]
        )
    print("\nPer-subset breakdown")
    print(
        tabulate(
            table,
            headers=["Subset", "Metric", "Default CEM", "V3 Warp CEM", "MPPI (tau=1.0)"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def print_pair30_table(records: list[dict[str, Any]]) -> None:
    rows = []
    for record in sorted(records, key=lambda row: (str(row["variant"]), int(row["seed"]))):
        rows.append(
            [
                record["seed"],
                record["variant"],
                fmt_float(record["Rpool_Cmodel"]),
                fmt_float(record["Rpool_warp_rank_avg"]),
                str(record["planning_success"]),
                fmt_float(record["selection_regret"]),
                fmt_float(record["pool_Creal_std"]),
            ]
        )
    print("\nPair 30 spotlight")
    print(
        tabulate(
            rows,
            headers=["Seed", "Variant", "Rpool(C_model)", "Rpool(warp)", "Success", "Regret", "Pool CReal Std"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def running_stats(records: list[dict[str, Any]]) -> str:
    if not records:
        return "no records yet"
    per_pair = per_pair_records(records)
    summary = aggregate_by_variant(per_pair)
    parts = []
    for variant in VARIANTS:
        stats = summary[variant]
        if stats["n_pairs"] == 0:
            continue
        parts.append(
            f"{variant}: success={fmt_metric(metric_mean(stats, 'planning_success'), percent=True)} "
            f"RpoolC={fmt_metric(metric_mean(stats, 'Rpool_Cmodel'))} "
            f"RpoolW={fmt_metric(metric_mean(stats, 'Rpool_warp_rank_avg'))} "
            f"regret={fmt_metric(metric_mean(stats, 'selection_regret'))}"
        )
    return " | ".join(parts) if parts else "no pair-complete records yet"


def print_assessment(summary_by_variant: dict[str, Any], mppi_summary: dict[str, Any]) -> None:
    default = summary_by_variant["default_cem"]
    warp = summary_by_variant["v3_warp_cem"]
    mppi = mppi_summary["mppi_tau_1"]
    print("\nAssessment helpers")
    for key, label, higher_better in (
        ("planning_success", "planning success", True),
        ("Rpool_Cmodel", "Rpool(C_model)", True),
        ("Rpool_warp_rank_avg", "Rpool(warp rank avg)", True),
        ("pool_Creal_std", "pool C_real_state std", True),
        ("selection_regret", "selection regret", False),
    ):
        d = metric_mean(default, key)
        w = metric_mean(warp, key)
        m = metric_mean(mppi, key)

        def outcome(candidate: float | None, baseline: float | None) -> str:
            if candidate is None or baseline is None:
                return "NA"
            if math.isclose(float(candidate), float(baseline), rel_tol=0.0, abs_tol=1e-12):
                return "tie"
            wins = candidate > baseline if higher_better else candidate < baseline
            return "win" if wins else "loss"

        print(
            f"- {label}: v3 vs default={outcome(w, d)} "
            f"({fmt_metric(w, percent=key == 'planning_success')} vs "
            f"{fmt_metric(d, percent=key == 'planning_success')}), "
            f"v3 vs MPPI={outcome(w, m)} "
            f"({fmt_metric(w, percent=key == 'planning_success')} vs "
            f"{fmt_metric(m, percent=key == 'planning_success')})"
        )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.mppi_run = args.mppi_run.expanduser().resolve()
    args.mppi_analysis = args.mppi_analysis.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pool_root = args.pool_root.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.ensemble_dir = args.ensemble_dir.expanduser().resolve()
    args.device = resolve_device(args.device)
    args.seeds = tuple(int(seed) for seed in args.seeds)

    selected_pairs = load_selected_pairs(args.mppi_run, pair_limit=args.pair_limit)
    pairs_data, _ = load_pairs(args.pairs_path, max_pairs=None, pair_ids=None)
    pair_metadata = pairs_data["metadata"]
    validate_requested_pair_offsets(selected_pairs, offset=int(pair_metadata["offset"]))
    dataset_path = Path(pair_metadata["dataset_path"])
    mppi_summary = load_mppi_summary(args.mppi_analysis)
    expected = expected_keys(selected_pairs, args.seeds)
    records = load_existing_records(args.output, resume=not args.no_resume, expected=expected)
    seen = {run_key(record) for record in records}

    print("== V3 Warp Ensemble CEM full 30-pair validation ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"mppi_run: {args.mppi_run}")
    print(f"mppi_analysis: {args.mppi_analysis}")
    print(f"output: {args.output}")
    print(f"pool_root: {args.pool_root}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"ensemble_dir: {args.ensemble_dir}")
    print(f"dataset_name: {dataset_path.stem}")
    print(f"cache_dir: {dataset_path.parent}")
    print(f"device: {args.device}")
    print(f"seeds: {[int(seed) for seed in args.seeds]}")
    print(f"pairs: {len(selected_pairs)}")
    print(f"variants: {list(VARIANTS)}")
    print(f"expected_runs: {len(expected)}")
    print(f"expected_simulator_rollouts: {len(expected) * NUM_SAMPLES}")
    print(f"resume_records: {len(records)}")

    set_seed(min(int(seed) for seed in args.seeds))
    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=min(int(seed) for seed in args.seeds),
        ),
        process,
    )
    model = policy.solver.model
    warps = load_warps(args.ensemble_dir, device=next(model.parameters()).device)

    total_started = time.time()
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair_spec in enumerate(selected_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            initial, goal = load_pair_rows_direct(dataset, pair_spec)
            prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
            pair_started = time.time()
            pair_completed_before = len(records)
            for seed_idx, seed in enumerate(args.seeds, start=1):
                seed = int(seed)
                for variant_idx, variant in enumerate(VARIANTS, start=1):
                    key = (pair_id, variant, seed)
                    if key in seen:
                        print(
                            f"[pair {pair_idx}/{len(selected_pairs)} seed {seed_idx}/{len(args.seeds)} "
                            f"variant {variant_idx}/{len(VARIANTS)}] pair_id={pair_id} seed={seed} "
                            f"{variant}: resume"
                        )
                        continue

                    run_started = time.time()
                    print(
                        f"[pair {pair_idx}/{len(selected_pairs)} seed {seed_idx}/{len(args.seeds)} "
                        f"variant {variant_idx}/{len(VARIANTS)}] pair_id={pair_id} seed={seed} "
                        f"{variant}: CEM + score 300"
                    )
                    cem_result = run_cem(
                        model=model,
                        prepared_info=prepared_info,
                        warps=warps,
                        pair_id=pair_id,
                        seed=seed,
                        variant=variant,
                    )
                    pool, record = build_pool_and_record(
                        pair_spec=pair_spec,
                        initial=initial,
                        goal=goal,
                        policy=policy,
                        env=env,
                        cem_result=cem_result,
                        variant=variant,
                        seed=seed,
                        device=str(args.device),
                    )
                    path = pool_path(args.pool_root, variant=variant, pair_id=pair_id, seed=seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(pool, path)

                    record["pool_path"] = str(path)
                    records.append(record)
                    seen.add(key)

                    output = build_output(
                        args=args,
                        selected_pairs=selected_pairs,
                        records=records,
                        mppi_summary=mppi_summary,
                        started=total_started,
                        expected_count=len(expected),
                    )
                    write_json_atomic(args.output, output)
                    print(
                        f"  saved {path}; success={record['planning_success']} "
                        f"RpoolC={fmt_float(record['Rpool_Cmodel'])} "
                        f"RpoolW={fmt_float(record['Rpool_warp_rank_avg'])} "
                        f"regret={fmt_float(record['selection_regret'])} "
                        f"pool_std={fmt_float(record['pool_Creal_std'])} "
                        f"elapsed={seconds_to_hms(time.time() - run_started)}"
                    )
                    print(
                        f"  progress {len(records)}/{len(expected)}; "
                        f"running {running_stats(records)}"
                    )
            if len(records) > pair_completed_before:
                print(
                    f"[pair {pair_idx}/{len(selected_pairs)}] pair_id={pair_id} complete/update; "
                    f"elapsed={seconds_to_hms(time.time() - pair_started)}"
                )
    finally:
        if hasattr(env, "close"):
            env.close()

    output = build_output(
        args=args,
        selected_pairs=selected_pairs,
        records=records,
        mppi_summary=mppi_summary,
        started=total_started,
        expected_count=len(expected),
    )
    write_json_atomic(args.output, output)
    print_comparison_table(output["comparison_table"])
    print_subset_table(output["subset_breakdown_table"])
    print_pair30_table(output["pair30_spotlight"])
    print_assessment(output["summary_by_variant"], mppi_summary)
    print(f"\nWrote summary: {args.output}")
    print(f"Wrote pools under: {args.pool_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
