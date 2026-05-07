#!/usr/bin/env python3
"""Run in-loop CEM with the v3 warp ensemble rank-averaged cost."""

from __future__ import annotations

import argparse
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
    load_pair_rows_direct,
    scalar_summary,
    seconds_to_hms,
    spearman_corr,
)
from method_local_warp import parse_int_list, set_seed, warped_costs, write_json_atomic  # noqa: E402
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


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "method_warp_ensemble_v3_cem.json"
DEFAULT_ENSEMBLE_DIR = PROJECT_ROOT / "results" / "revision" / "warp_ensemble_v3"
DEFAULT_POOL_ROOT = PROJECT_ROOT / "results" / "revision" / "warp_v3_cem_pools"
DEFAULT_PAIR_IDS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL_ROOT)
    parser.add_argument("--pair-ids", type=parse_int_list, default=DEFAULT_PAIR_IDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


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
            warp_score = torch.empty((0,), dtype=torch.float32, device=device)
            individual_costs = torch.empty((0, NUM_SAMPLES), dtype=torch.float32, device=device)
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


def build_pool_record(
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
    rank1 = int(deterministic_argmin(select_costs))
    oracle = int(deterministic_argmin(c_real_state))
    pool = {
        "metadata": {
            "format": "pusht_warp_v3_cem_pool_v1",
            "created_at_unix": clean_float(time.time()),
            "pair_id": pair_id,
            "cell": str(pair_spec["cell"]),
            "start_row": int(pair_spec["start_row"]),
            "goal_row": int(pair_spec["goal_row"]),
            "seed": int(seed),
            "cem_sampling_seed": int(seed) + pair_id * 1009,
            "device": str(device),
            "variant": str(variant),
            "wallclock_seconds": cem_result["wallclock_seconds"],
        },
        "pair_spec": dict(pair_spec),
        "z_pred": torch.as_tensor(np.asarray(cem_result["z_pred"], dtype=np.float32)),
        "z_goal": torch.as_tensor(np.asarray(cem_result["z_goal"], dtype=np.float32)),
        "blocked_actions": torch.as_tensor(np.asarray(cem_result["blocked_candidates"], dtype=np.float32)),
        "raw_actions": torch.as_tensor(raw_actions.astype(np.float32)),
        "select_costs": torch.as_tensor(select_costs, dtype=torch.float64),
        "default_costs": torch.as_tensor(default_costs, dtype=torch.float64),
        "warp_rank_avg_costs": torch.as_tensor(cem_result["warp_rank_avg_costs"], dtype=torch.float64),
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
        "Pair": pair_id,
        "Variant": str(variant),
        "Rpool": spearman_corr(select_costs, c_real_state),
        "Rpool_Cmodel": spearman_corr(default_costs, c_real_state),
        "Rpool_warped": spearman_corr(select_costs, c_real_state),
        "rank1_candidate_index": int(rank1),
        "rank1_success": bool(success[rank1]),
        "rank1_c_real": clean_float(float(c_real_state[rank1])),
        "oracle_best_candidate_index": int(oracle),
        "oracle_c_real": clean_float(float(c_real_state[oracle])),
        "selection_regret": clean_float(float(c_real_state[rank1] - c_real_state[oracle])),
        "pool_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
        "elite_cost_std": cem_result["elite_cost_std"],
        "pool_path": None,
    }
    return pool, record


def save_pool(root: Path, variant: str, pair_id: int, pool: dict[str, Any]) -> Path:
    path = root / variant / f"pair_{int(pair_id)}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pool, path)
    return path


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_variant.setdefault(str(record["Variant"]), []).append(record)
    return {
        variant: {
            "n": int(len(rows)),
            "Rpool": scalar_summary([row.get("Rpool") for row in rows]),
            "Rpool_Cmodel": scalar_summary([row.get("Rpool_Cmodel") for row in rows]),
            "rank1_success_rate": scalar_summary([row.get("rank1_success") for row in rows]),
            "selection_regret": scalar_summary([row.get("selection_regret") for row in rows]),
            "pool_Creal_std": scalar_summary([row.get("pool_Creal_std") for row in rows]),
            "elite_cost_std": scalar_summary([row.get("elite_cost_std") for row in rows]),
        }
        for variant, rows in by_variant.items()
    }


def print_table(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rows = [
        [
            record["Pair"],
            record["Variant"],
            fmt_float(record["Rpool"]),
            str(record["rank1_success"]),
            fmt_float(record["selection_regret"]),
            fmt_float(record["pool_Creal_std"]),
            fmt_float(record["elite_cost_std"]),
        ]
        for record in records
    ]
    print("\nV3 warp CEM comparison")
    print(
        tabulate(
            rows,
            headers=["Pair", "Variant", "Rpool", "Success", "Regret", "Pool CReal Std", "Elite Std"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )
    summary_rows = []
    for variant, stats in summary.items():
        summary_rows.append(
            [
                variant,
                stats["n"],
                fmt_float(stats["Rpool"]["mean"]),
                fmt_float(stats["rank1_success_rate"]["mean"]),
                fmt_float(stats["selection_regret"]["mean"]),
                fmt_float(stats["pool_Creal_std"]["mean"]),
                fmt_float(stats["elite_cost_std"]["mean"]),
            ]
        )
    print("\nMean summary")
    print(
        tabulate(
            summary_rows,
            headers=["Variant", "n", "mean_Rpool", "success_rate", "mean_regret", "pool_Creal_std", "elite_cost_std"],
            tablefmt="github",
            stralign="right",
            numalign="right",
        )
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.ensemble_dir = args.ensemble_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pool_root = args.pool_root.expanduser().resolve()
    args.device = resolve_device(str(args.device))
    set_seed(int(args.seed))
    total_started = time.time()

    print("== In-loop CEM with V3 warp ensemble rank averaging ==")
    print(f"pairs_path: {args.pairs_path}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"ensemble_dir: {args.ensemble_dir}")
    print(f"output: {args.output}")
    print(f"pool_root: {args.pool_root}")
    print(f"pair_ids: {list(args.pair_ids)}")
    print(f"device: {args.device}")

    pairs_data, requested_pairs = load_pairs(args.pairs_path, max_pairs=None, pair_ids=list(args.pair_ids))
    requested_pairs = sorted(requested_pairs, key=lambda pair: int(pair["pair_id"]))
    validate_requested_pair_offsets(requested_pairs, offset=int(pairs_data["metadata"]["offset"]))
    dataset_path = Path(pairs_data["metadata"]["dataset_path"])
    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(checkpoint_dir=args.checkpoint_dir, device=args.device, seed=int(args.seed)),
        process,
    )
    model = policy.solver.model
    warps = load_warps(args.ensemble_dir, device=next(model.parameters()).device)
    records: list[dict[str, Any]] = []
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair_spec in enumerate(requested_pairs, start=1):
            pair_id = int(pair_spec["pair_id"])
            initial, goal = load_pair_rows_direct(dataset, pair_spec)
            prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
            for variant in ("default_cem", "v3_warp_cem"):
                started = time.time()
                print(f"[CEM {pair_idx}/{len(requested_pairs)}] pair_id={pair_id} {variant}")
                cem_result = run_cem(
                    model=model,
                    prepared_info=prepared_info,
                    warps=warps,
                    pair_id=pair_id,
                    seed=int(args.seed),
                    variant=variant,
                )
                pool, record = build_pool_record(
                    pair_spec=pair_spec,
                    initial=initial,
                    goal=goal,
                    policy=policy,
                    env=env,
                    cem_result=cem_result,
                    variant=variant,
                    seed=int(args.seed),
                    device=str(args.device),
                )
                path = save_pool(args.pool_root, variant, pair_id, pool)
                record["pool_path"] = str(path)
                records.append(record)
                print(
                    f"  saved {path}; Rpool={fmt_float(record['Rpool'])} "
                    f"success={record['rank1_success']} regret={fmt_float(record['selection_regret'])} "
                    f"pool_std={fmt_float(record['pool_Creal_std'])} "
                    f"elapsed={seconds_to_hms(time.time() - started)}"
                )
    finally:
        if hasattr(env, "close"):
            env.close()

    summary = aggregate_records(records)
    print_table(records, summary)
    pair30 = [record for record in records if int(record["Pair"]) == 30]
    if pair30:
        print("\nPair 30 spotlight")
        print_table(pair30, aggregate_records(pair30))

    payload = {
        "metadata": {
            "format": "pusht_warp_v3_cem_v1",
            "created_at_unix": clean_float(time.time()),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "pairs_path": str(args.pairs_path),
            "checkpoint_dir": str(args.checkpoint_dir),
            "ensemble_dir": str(args.ensemble_dir),
            "output": str(args.output),
            "pool_root": str(args.pool_root),
            "device": str(args.device),
            "seed": int(args.seed),
            "pair_ids": [int(item) for item in args.pair_ids],
            "wallclock_seconds": clean_float(time.time() - total_started),
        },
        "records": records,
        "summary": summary,
        "pair30_spotlight": pair30,
    }
    write_json_atomic(args.output, payload)
    print(f"\nWrote summary: {args.output}")
    print(f"Total elapsed: {seconds_to_hms(time.time() - total_started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
