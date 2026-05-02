#!/usr/bin/env python3
"""Deep diagnostics for metric-to-planning failures in Phase 2.

The learned P2-0 costs can rank fixed Track A candidate sets better than the
Euclidean latent baseline, while still failing inside receding-horizon CEM. This
script instruments the CEM search itself: for selected pairs, it logs the model
cost and real V1 oracle cost of elites at every iteration, measures search
distribution collapse, probes local cost landscapes, and summarizes Split 1
planning regressions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch


warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")
warnings.filterwarnings("ignore", message=".*obs returned by the `step\\(\\)` method.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR as DEFAULT_MODEL_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import (  # noqa: E402
    prepare_pair_info,
)
from lewm_audit.eval.oracle_cem import (  # noqa: E402
    ANGLE_SUCCESS_THRESHOLD_RAD,
    BLOCK_SUCCESS_THRESHOLD_PX,
    block_pose_components,
    cost_v1_hinge,
    rollout_final_state,
)
from scripts.phase2.eval_planning import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    TOPK,
    VAR_SCALE,
    load_planning_cost_model,
)
from scripts.phase2.mahalanobis_baseline import spearman  # noqa: E402
from scripts.phase2.splits import DEFAULT_PAIRS_PATH, load_track_a_pairs  # noqa: E402
from scripts.phase2.train_cem_aware import (  # noqa: E402
    blocked_batch_to_raw,
    candidate_costs_from_latents,
    rollout_candidate_latents,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "deep_diagnosis.json"
DEFAULT_SPLIT3_PLAN = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "split3_planning.json"
DEFAULT_SPLIT1_PLAN = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "split1_planning.json"
DEFAULT_COST_CHECKPOINT_DIR = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "split3_small"
DEFAULT_V1_DIR = PROJECT_ROOT / "results" / "phase1" / "v1_oracle_ablation"
DEFAULT_DATASET_NAME = "pusht_expert_train"
LATENT_DIM = 192


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, choices=(1, 3), default=3)
    parser.add_argument("--variant", choices=("small", "large"), default="small")
    parser.add_argument("--model-type", choices=("mlp", "mahalanobis"), default="mlp")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_COST_CHECKPOINT_DIR)
    parser.add_argument("--planning-result", type=Path, default=DEFAULT_SPLIT3_PLAN)
    parser.add_argument("--split1-planning-result", type=Path, default=DEFAULT_SPLIT1_PLAN)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-checkpoint-dir", type=Path, default=DEFAULT_MODEL_CHECKPOINT_DIR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pair-ids",
        default=None,
        help="Optional comma-separated diagnostic pair IDs. Defaults to auto-selected V1-success/C_psi-fail pairs.",
    )
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument(
        "--elite-v1-count",
        type=int,
        default=30,
        help="Number of model elites per CEM iteration to score with real V1 rollouts.",
    )
    parser.add_argument("--landscape-samples", type=int, default=1000)
    parser.add_argument("--landscape-noise", type=float, default=0.15)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    """Convert nested values to JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: dict) -> None:
    """Save pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")


def parse_pair_ids(raw: str | None) -> list[int] | None:
    """Parse comma-separated pair IDs."""
    if raw is None:
        return None
    pair_ids = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            pair_ids.append(int(chunk))
    return list(dict.fromkeys(pair_ids))


def load_json(path: Path) -> dict:
    """Load JSON from a path."""
    return json.loads(path.read_text())


def load_pairs_by_id(path: Path) -> dict[int, dict]:
    """Load Track A pair metadata keyed by pair ID."""
    return {int(pair["pair_id"]): pair for pair in load_track_a_pairs(path)}


def load_pair_rows(dataset, pair: dict) -> tuple[dict, dict]:
    """Load initial and goal rows for one Track A pair."""
    rows = dataset.get_row_data([int(pair["start_row"]), int(pair["goal_row"])])
    return {key: value[0] for key, value in rows.items()}, {
        key: value[1] for key, value in rows.items()
    }


def load_v1_reference(v1_dir: Path) -> dict[int, dict]:
    """Load per-pair V1 oracle CEM summary from Phase 1 artifacts."""
    out: dict[int, dict] = {}
    for path in sorted(v1_dir.glob("v1_d*.json")):
        data = load_json(path)
        for pair in data.get("pairs", []):
            early = [a for a in pair.get("actions", []) if a.get("source") == "CEM_early_V1"]
            late = [a for a in pair.get("actions", []) if a.get("source") == "CEM_late_V1"]
            out[int(pair["pair_id"])] = {
                "cell": str(pair["cell"]),
                "early_success_count": int(sum(bool(a.get("success")) for a in early)),
                "late_success_count": int(sum(bool(a.get("success")) for a in late)),
                "early_n": len(early),
                "late_n": len(late),
                "late_success_any": bool(any(bool(a.get("success")) for a in late)),
                "late_min_v1": min((float(a["C_variant"]) for a in late), default=None),
                "late_min_c_real_state": min((float(a["C_real_state"]) for a in late), default=None),
            }
    return out


def select_diagnostic_pairs(
    *,
    explicit_pair_ids: list[int] | None,
    planning_result: dict,
    v1_reference: dict[int, dict],
    max_pairs: int,
) -> list[int]:
    """Choose pairs where V1 late CEM succeeded but learned C_psi planning failed."""
    if explicit_pair_ids is not None:
        return explicit_pair_ids[:max_pairs]
    selected = []
    seen_cells = set()
    records = planning_result.get("records", [])
    for record in records:
        pair_id = int(record["pair_id"])
        cell = str(record["cell"])
        v1 = v1_reference.get(pair_id, {})
        if bool(record.get("cpsi", {}).get("success")):
            continue
        if not bool(v1.get("late_success_any")):
            continue
        if cell in seen_cells:
            continue
        selected.append(pair_id)
        seen_cells.add(cell)
        if len(selected) >= max_pairs:
            return selected
    for record in records:
        pair_id = int(record["pair_id"])
        v1 = v1_reference.get(pair_id, {})
        if pair_id in selected:
            continue
        if bool(record.get("cpsi", {}).get("success")):
            continue
        if bool(v1.get("late_success_any")):
            selected.append(pair_id)
            if len(selected) >= max_pairs:
                break
    return selected


def make_policy_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return args for existing policy builder."""
    return argparse.Namespace(
        checkpoint_dir=args.model_checkpoint_dir,
        device=args.device,
        num_samples=NUM_SAMPLES,
        var_scale=VAR_SCALE,
        cem_iters=CEM_ITERS,
        topk=TOPK,
        seed=args.seed,
        horizon=PLANNING_HORIZON,
        receding_horizon=PLANNING_HORIZON,
        action_block=ACTION_BLOCK,
        img_size=IMG_SIZE,
    )


def euclidean_costs(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean terminal latent costs."""
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    return torch.sum((z_pred - z_goal) ** 2, dim=-1)


def entropy_diag(std: torch.Tensor) -> float:
    """Return diagonal-Gaussian entropy proxy from per-dimension std."""
    std = torch.clamp(std, min=1e-8)
    return float((0.5 * torch.sum(torch.log(2.0 * math.pi * math.e * std**2))).detach().cpu())


def score_raw_actions_v1(
    *,
    env,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions_batch: np.ndarray,
    seed_base: int,
) -> tuple[np.ndarray, list[dict]]:
    """Roll out raw action sequences and return V1 costs plus physical metrics."""
    costs = []
    metrics = []
    for idx, raw_actions in enumerate(raw_actions_batch):
        terminal_state = rollout_final_state(
            env,
            initial_state,
            goal_state,
            raw_actions,
            seed=seed_base + idx,
        )
        v1 = float(cost_v1_hinge(terminal_state, goal_state))
        pose = block_pose_components(terminal_state, goal_state)
        costs.append(v1)
        metrics.append(
            {
                "v1_cost": v1,
                "block_pos_dist": float(pose["block_pos_dist"]),
                "angle_dist": float(pose["angle_dist"]),
                "success": bool(pose["success"]),
            }
        )
    return np.asarray(costs, dtype=np.float64), metrics


def run_cem_trace(
    *,
    method: str,
    cost_head,
    world_model,
    policy,
    env,
    initial: dict,
    goal: dict,
    pair: dict,
    args: argparse.Namespace,
) -> dict:
    """Run one instrumented CEM solve and V1-score selected elites each iteration."""
    device = next(world_model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(args.seed + int(pair["pair_id"]) * 1009)
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    action_processor = policy.process["action"]
    initial_state = np.asarray(initial["state"], dtype=np.float32)
    goal_state = np.asarray(goal["state"], dtype=np.float32)
    mean = torch.zeros((1, PLANNING_HORIZON, ACTION_BLOCK * 2), device=device)
    std = VAR_SCALE * torch.ones((1, PLANNING_HORIZON, ACTION_BLOCK * 2), device=device)
    elite_v1_count = min(args.elite_v1_count, TOPK)

    trace = []
    started = time.time()
    for iter_idx in range(1, CEM_ITERS + 1):
        search_mean = mean.detach().cpu()
        search_std = std.detach().cpu()
        candidates = torch.randn(
            1,
            NUM_SAMPLES,
            PLANNING_HORIZON,
            ACTION_BLOCK * 2,
            generator=generator,
            device=device,
        )
        candidates = candidates * std.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        z_pred, z_goal = rollout_candidate_latents(world_model, prepared_info, candidates)
        if method == "euclidean":
            model_costs = euclidean_costs(z_pred, z_goal)
        elif method == "cpsi":
            model_costs = candidate_costs_from_latents(cost_head, z_pred, z_goal)
        else:
            raise ValueError(f"Unknown CEM trace method: {method}")

        top_vals, top_inds = torch.topk(model_costs, k=TOPK, dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]
        elite_blocked = elite_candidates[0, :elite_v1_count].detach().cpu().numpy().astype(np.float32)
        raw_elites = blocked_batch_to_raw(
            elite_blocked,
            action_processor=action_processor,
        )
        v1_costs, v1_metrics = score_raw_actions_v1(
            env=env,
            initial_state=initial_state,
            goal_state=goal_state,
            raw_actions_batch=raw_elites,
            seed_base=args.seed + int(pair["pair_id"]) * 100_000 + iter_idx * 1_000,
        )
        oracle_best_rank = int(np.argmin(v1_costs))
        topk_success_count = int(sum(bool(item["success"]) for item in v1_metrics))

        next_mean = elite_candidates.mean(dim=1)
        next_std = elite_candidates.std(dim=1)
        trace.append(
            {
                "iteration": iter_idx,
                "search_mean_l2": float(torch.linalg.norm(search_mean).item()),
                "search_std_mean": float(search_std.mean().item()),
                "search_std_min": float(search_std.min().item()),
                "search_std_max": float(search_std.max().item()),
                "search_entropy_diag": entropy_diag(search_std),
                "model_cost_best": float(top_vals[0, 0].detach().cpu()),
                "model_cost_top30_mean": float(top_vals[0].mean().detach().cpu()),
                "model_cost_top30_std": float(top_vals[0].std(unbiased=False).detach().cpu()),
                "model_cost_all_min": float(model_costs[0].min().detach().cpu()),
                "model_cost_all_max": float(model_costs[0].max().detach().cpu()),
                "selected_best_v1": float(v1_costs[0]),
                "selected_best_success": bool(v1_metrics[0]["success"]),
                "selected_best_block_pos_dist": float(v1_metrics[0]["block_pos_dist"]),
                "selected_best_angle_dist": float(v1_metrics[0]["angle_dist"]),
                "elite_oracle_best_v1": float(v1_costs[oracle_best_rank]),
                "elite_oracle_best_rank": oracle_best_rank,
                "elite_oracle_best_success": bool(v1_metrics[oracle_best_rank]["success"]),
                "elite_success_count": topk_success_count,
                "next_mean_l2": float(torch.linalg.norm(next_mean.detach().cpu()).item()),
                "next_std_mean": float(next_std.detach().cpu().mean().item()),
                "elite_mean_action": next_mean[0].detach().cpu().numpy().astype(np.float32),
            }
        )
        mean = next_mean
        std = next_std

    return {
        "method": method,
        "trace": trace,
        "final_mean": mean[0].detach().cpu().numpy().astype(np.float32),
        "final_std": std[0].detach().cpu().numpy().astype(np.float32),
        "elapsed_seconds": float(time.time() - started),
    }


def trace_summary(trace: dict) -> dict:
    """Compress an iteration trace into hypothesis-friendly numbers."""
    rows = trace["trace"]
    selected_v1 = np.asarray([row["selected_best_v1"] for row in rows], dtype=np.float64)
    elite_best_v1 = np.asarray([row["elite_oracle_best_v1"] for row in rows], dtype=np.float64)
    std_mean = np.asarray([row["search_std_mean"] for row in rows], dtype=np.float64)
    entropy = np.asarray([row["search_entropy_diag"] for row in rows], dtype=np.float64)
    success_iters = [row["iteration"] for row in rows if row["selected_best_success"]]
    elite_success_iters = [row["iteration"] for row in rows if row["elite_success_count"] > 0]
    return {
        "selected_best_v1_iter1": float(selected_v1[0]),
        "selected_best_v1_iter30": float(selected_v1[-1]),
        "selected_best_v1_min": float(np.min(selected_v1)),
        "selected_best_v1_argmin_iter": int(np.argmin(selected_v1) + 1),
        "elite_oracle_best_v1_iter1": float(elite_best_v1[0]),
        "elite_oracle_best_v1_iter30": float(elite_best_v1[-1]),
        "elite_oracle_best_v1_min": float(np.min(elite_best_v1)),
        "elite_oracle_best_v1_argmin_iter": int(np.argmin(elite_best_v1) + 1),
        "first_selected_success_iter": success_iters[0] if success_iters else None,
        "first_elite_contains_success_iter": elite_success_iters[0] if elite_success_iters else None,
        "iters_elite_contains_success": len(elite_success_iters),
        "search_std_mean_iter1": float(std_mean[0]),
        "search_std_mean_iter10": float(std_mean[9]),
        "search_std_mean_iter20": float(std_mean[19]),
        "search_std_mean_iter30": float(std_mean[29]),
        "entropy_iter1": float(entropy[0]),
        "entropy_iter30": float(entropy[-1]),
        "entropy_drop": float(entropy[0] - entropy[-1]),
    }


def divergence_summary(euclidean_trace: dict, cpsi_trace: dict) -> dict:
    """Compare CEM search-mean trajectories between Euclidean and C_psi."""
    distances = []
    for e_row, c_row in zip(euclidean_trace["trace"], cpsi_trace["trace"], strict=True):
        e_mean = np.asarray(e_row["elite_mean_action"], dtype=np.float64).reshape(-1)
        c_mean = np.asarray(c_row["elite_mean_action"], dtype=np.float64).reshape(-1)
        distances.append(float(np.linalg.norm(e_mean - c_mean)))
    first_gt_1 = next((idx + 1 for idx, value in enumerate(distances) if value > 1.0), None)
    first_gt_5 = next((idx + 1 for idx, value in enumerate(distances) if value > 5.0), None)
    return {
        "mean_action_l2_by_iter": distances,
        "final_mean_action_l2": distances[-1],
        "first_iter_l2_gt_1": first_gt_1,
        "first_iter_l2_gt_5": first_gt_5,
    }


def probe_local_landscape(
    *,
    reference_mean: np.ndarray,
    cost_head,
    world_model,
    policy,
    initial: dict,
    goal: dict,
    args: argparse.Namespace,
    pair_id: int,
) -> dict:
    """Perturb the Euclidean final mean action and compare local cost smoothness."""
    rng = np.random.default_rng(args.seed + pair_id * 7919 + 17)
    noise = rng.normal(
        loc=0.0,
        scale=args.landscape_noise,
        size=(args.landscape_samples, *reference_mean.shape),
    ).astype(np.float32)
    candidates_np = reference_mean[None, ...].astype(np.float32) + noise
    perturb_mag = np.linalg.norm(noise.reshape(noise.shape[0], -1), axis=1).astype(np.float64)
    device = next(world_model.parameters()).device
    candidates = torch.as_tensor(candidates_np[None, ...], dtype=torch.float32, device=device)
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    z_pred, z_goal = rollout_candidate_latents(world_model, prepared_info, candidates)
    euc_cost = euclidean_costs(z_pred, z_goal)[0].detach().cpu().numpy().astype(np.float64)
    cpsi_cost = candidate_costs_from_latents(cost_head, z_pred, z_goal)[0].detach().cpu().numpy().astype(
        np.float64
    )
    return {
        "pair_id": int(pair_id),
        "samples": int(args.landscape_samples),
        "noise_std": float(args.landscape_noise),
        "perturb_magnitude_min": float(np.min(perturb_mag)),
        "perturb_magnitude_max": float(np.max(perturb_mag)),
        "spearman_perturb_vs_euclidean": spearman(perturb_mag, euc_cost),
        "spearman_perturb_vs_cpsi": spearman(perturb_mag, cpsi_cost),
        "spearman_euclidean_vs_cpsi_cost": spearman(euc_cost, cpsi_cost),
        "euclidean_cost_min": float(np.min(euc_cost)),
        "euclidean_cost_max": float(np.max(euc_cost)),
        "euclidean_cost_std": float(np.std(euc_cost)),
        "cpsi_cost_min": float(np.min(cpsi_cost)),
        "cpsi_cost_max": float(np.max(cpsi_cost)),
        "cpsi_cost_std": float(np.std(cpsi_cost)),
    }


def failure_reason(block_pos_dist: float, angle_dist: float) -> str:
    """Name which PushT success condition failed."""
    pos_ok = block_pos_dist < BLOCK_SUCCESS_THRESHOLD_PX
    angle_ok = angle_dist < ANGLE_SUCCESS_THRESHOLD_RAD
    if pos_ok and angle_ok:
        return "success"
    if not pos_ok and not angle_ok:
        return "position+angle"
    if not pos_ok:
        return "position"
    return "angle"


def analyze_split1_regressions(path: Path) -> dict:
    """Summarize Split 1 pairs where Euclidean succeeds and C_psi fails."""
    if not path.exists():
        return {"path": str(path), "available": False, "regressions": []}
    data = load_json(path)
    regressions = []
    rescues = []
    for record in data.get("records", []):
        euc = record["euclidean"]
        cpsi = record["cpsi"]
        if bool(euc["success"]) and not bool(cpsi["success"]):
            regressions.append(
                {
                    "pair_id": int(record["pair_id"]),
                    "cell": str(record["cell"]),
                    "easy_cell_d0": str(record["cell"]).startswith("D0"),
                    "latent_favorable_cell": str(record["cell"]) in {"D0xR1", "D1xR0"},
                    "euclidean_block_pos_dist": float(euc["block_pos_dist"]),
                    "euclidean_angle_dist": float(euc["angle_dist"]),
                    "euclidean_failure_reason": failure_reason(
                        float(euc["block_pos_dist"]),
                        float(euc["angle_dist"]),
                    ),
                    "cpsi_block_pos_dist": float(cpsi["block_pos_dist"]),
                    "cpsi_angle_dist": float(cpsi["angle_dist"]),
                    "cpsi_failure_reason": failure_reason(
                        float(cpsi["block_pos_dist"]),
                        float(cpsi["angle_dist"]),
                    ),
                    "block_pos_delta_cpsi_minus_euclidean": float(
                        cpsi["block_pos_dist"] - euc["block_pos_dist"]
                    ),
                    "angle_delta_cpsi_minus_euclidean": float(
                        cpsi["angle_dist"] - euc["angle_dist"]
                    ),
                }
            )
        if (not bool(euc["success"])) and bool(cpsi["success"]):
            rescues.append({"pair_id": int(record["pair_id"]), "cell": str(record["cell"])})
    return {
        "path": str(path),
        "available": True,
        "n_pairs": int(len(data.get("records", []))),
        "euclidean_success_count": int(sum(bool(r["euclidean"]["success"]) for r in data.get("records", []))),
        "cpsi_success_count": int(sum(bool(r["cpsi"]["success"]) for r in data.get("records", []))),
        "regressions": regressions,
        "rescues": rescues,
    }


def build_summary(
    *,
    pair_results: list[dict],
    landscape: dict,
    split1_regressions: dict,
) -> dict:
    """Build high-level hypothesis summaries from raw diagnostics."""
    h1_rows = []
    h2_rows = []
    for result in pair_results:
        e_summary = result["euclidean"]["summary"]
        c_summary = result["cpsi"]["summary"]
        h1_rows.append(
            {
                "pair_id": result["pair_id"],
                "cell": result["cell"],
                "v1_late_success_count": result["v1_reference"].get("late_success_count"),
                "v1_late_n": result["v1_reference"].get("late_n"),
                "euclidean_selected_v1_iter1_to_30": [
                    e_summary["selected_best_v1_iter1"],
                    e_summary["selected_best_v1_iter30"],
                ],
                "cpsi_selected_v1_iter1_to_30": [
                    c_summary["selected_best_v1_iter1"],
                    c_summary["selected_best_v1_iter30"],
                ],
                "euclidean_selected_v1_min": e_summary["selected_best_v1_min"],
                "cpsi_selected_v1_min": c_summary["selected_best_v1_min"],
                "euclidean_elite_best_v1_min": e_summary["elite_oracle_best_v1_min"],
                "cpsi_elite_best_v1_min": c_summary["elite_oracle_best_v1_min"],
                "euclidean_iters_elite_contains_success": e_summary[
                    "iters_elite_contains_success"
                ],
                "cpsi_iters_elite_contains_success": c_summary["iters_elite_contains_success"],
            }
        )
        h2_rows.append(
            {
                "pair_id": result["pair_id"],
                "cell": result["cell"],
                "euclidean_std_iter1_10_20_30": [
                    e_summary["search_std_mean_iter1"],
                    e_summary["search_std_mean_iter10"],
                    e_summary["search_std_mean_iter20"],
                    e_summary["search_std_mean_iter30"],
                ],
                "cpsi_std_iter1_10_20_30": [
                    c_summary["search_std_mean_iter1"],
                    c_summary["search_std_mean_iter10"],
                    c_summary["search_std_mean_iter20"],
                    c_summary["search_std_mean_iter30"],
                ],
                "euclidean_entropy_drop": e_summary["entropy_drop"],
                "cpsi_entropy_drop": c_summary["entropy_drop"],
                "final_mean_action_l2_between_methods": result["divergence"][
                    "final_mean_action_l2"
                ],
                "first_divergence_iter_l2_gt_1": result["divergence"]["first_iter_l2_gt_1"],
            }
        )
    return {
        "hypothesis1_oracle_online_state_access": h1_rows,
        "hypothesis2_cem_dynamics": h2_rows,
        "hypothesis3_local_landscape": landscape,
        "hypothesis4_split1_easy_pair_regressions": split1_regressions,
    }


def print_report(payload: dict) -> None:
    """Print a compact structured diagnosis."""
    print("\n== Deep Diagnosis ==")
    print(f"device: {payload['metadata']['device']}")
    print(f"cost checkpoint: {payload['metadata']['cost_model_checkpoint']}")
    print(f"diagnostic pairs: {payload['metadata']['diagnostic_pair_ids']}")

    print("\nHypothesis 1: V1 oracle access vs learned cost elites")
    print("pair cell V1late  E_selV1(1->30,min)  C_selV1(1->30,min)  E_top30_succ_iters C_top30_succ_iters")
    for row in payload["summary"]["hypothesis1_oracle_online_state_access"]:
        e1, e30 = row["euclidean_selected_v1_iter1_to_30"]
        c1, c30 = row["cpsi_selected_v1_iter1_to_30"]
        print(
            f"{row['pair_id']:>4} {row['cell']:<5} "
            f"{row['v1_late_success_count']}/{row['v1_late_n']} "
            f"{e1:7.2f}->{e30:7.2f},min={row['euclidean_selected_v1_min']:7.2f} "
            f"{c1:7.2f}->{c30:7.2f},min={row['cpsi_selected_v1_min']:7.2f} "
            f"{row['euclidean_iters_elite_contains_success']:>2} "
            f"{row['cpsi_iters_elite_contains_success']:>2}"
        )

    print("\nHypothesis 2: CEM dynamics and collapse")
    print("pair cell  E_std[1,10,20,30]        C_std[1,10,20,30]        final_mean_L2 div_iter>1")
    for row in payload["summary"]["hypothesis2_cem_dynamics"]:
        e_std = ",".join(f"{value:.3f}" for value in row["euclidean_std_iter1_10_20_30"])
        c_std = ",".join(f"{value:.3f}" for value in row["cpsi_std_iter1_10_20_30"])
        print(
            f"{row['pair_id']:>4} {row['cell']:<5} "
            f"[{e_std:<23}] [{c_std:<23}] "
            f"{row['final_mean_action_l2_between_methods']:.2f} "
            f"{row['first_divergence_iter_l2_gt_1']}"
        )

    print("\nHypothesis 3: Local landscape around Euclidean CEM mean")
    landscape = payload["summary"]["hypothesis3_local_landscape"]
    print(
        f"pair={landscape['pair_id']} samples={landscape['samples']} "
        f"rho(|noise|, Euclidean)={landscape['spearman_perturb_vs_euclidean']:.4f} "
        f"rho(|noise|, C_psi)={landscape['spearman_perturb_vs_cpsi']:.4f} "
        f"rho(Euclidean,C_psi)={landscape['spearman_euclidean_vs_cpsi_cost']:.4f}"
    )
    print(
        f"cost std: Euclidean={landscape['euclidean_cost_std']:.4f} "
        f"C_psi={landscape['cpsi_cost_std']:.4f}; "
        f"ranges: E=[{landscape['euclidean_cost_min']:.2f},{landscape['euclidean_cost_max']:.2f}] "
        f"C=[{landscape['cpsi_cost_min']:.2f},{landscape['cpsi_cost_max']:.2f}]"
    )

    print("\nHypothesis 4: Split 1 regressions")
    split1 = payload["summary"]["hypothesis4_split1_easy_pair_regressions"]
    print(
        f"Split1 counts: Euclidean={split1.get('euclidean_success_count')} "
        f"C_psi={split1.get('cpsi_success_count')} "
        f"regressions={len(split1.get('regressions', []))} "
        f"rescues={len(split1.get('rescues', []))}"
    )
    for row in split1.get("regressions", []):
        print(
            f"pair={row['pair_id']} cell={row['cell']} "
            f"E=({row['euclidean_block_pos_dist']:.2f},{row['euclidean_angle_dist']:.3f}) "
            f"C=({row['cpsi_block_pos_dist']:.2f},{row['cpsi_angle_dist']:.3f}) "
            f"C_fail={row['cpsi_failure_reason']} "
            f"latent_favorable={row['latent_favorable_cell']}"
        )
    print(f"\nsaved_results: {payload['metadata']['output_path']}")


def run(args: argparse.Namespace) -> dict:
    """Run the deep diagnosis."""
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.planning_result = args.planning_result.expanduser().resolve()
    args.split1_planning_result = args.split1_planning_result.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.v1_dir = args.v1_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.model_checkpoint_dir = args.model_checkpoint_dir.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    if args.elite_v1_count < 1 or args.elite_v1_count > TOPK:
        raise ValueError(f"--elite-v1-count must be in [1, {TOPK}]")

    planning_result = load_json(args.planning_result)
    v1_reference = load_v1_reference(args.v1_dir)
    pair_ids = select_diagnostic_pairs(
        explicit_pair_ids=parse_pair_ids(args.pair_ids),
        planning_result=planning_result,
        v1_reference=v1_reference,
        max_pairs=args.max_pairs,
    )
    if not pair_ids:
        raise RuntimeError("No diagnostic pairs selected")

    checkpoint_path, cost_head, loaded_variant = load_planning_cost_model(
        checkpoint_dir=args.checkpoint_dir,
        split=args.split,
        variant=args.variant,
        model_type=args.model_type,
        device=args.device,
    )
    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(make_policy_args(args), process)
    world_model = policy.solver.model
    env = gym.make("swm/PushT-v1")
    pairs_by_id = load_pairs_by_id(args.pairs_path)

    print("== Deep diagnosis setup ==")
    print(f"device: {args.device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"pairs: {pair_ids}")
    print(f"elite_v1_count: {args.elite_v1_count}")

    started = time.time()
    pair_results = []
    try:
        for idx, pair_id in enumerate(pair_ids, start=1):
            pair = pairs_by_id[pair_id]
            initial, goal = load_pair_rows(dataset, pair)
            print(f"\n[{idx}/{len(pair_ids)}] pair={pair_id} cell={pair['cell']} Euclidean trace")
            euclidean_trace = run_cem_trace(
                method="euclidean",
                cost_head=None,
                world_model=world_model,
                policy=policy,
                env=env,
                initial=initial,
                goal=goal,
                pair=pair,
                args=args,
            )
            print(f"[{idx}/{len(pair_ids)}] pair={pair_id} cell={pair['cell']} C_psi trace")
            cpsi_trace = run_cem_trace(
                method="cpsi",
                cost_head=cost_head,
                world_model=world_model,
                policy=policy,
                env=env,
                initial=initial,
                goal=goal,
                pair=pair,
                args=args,
            )
            pair_results.append(
                {
                    "pair_id": pair_id,
                    "cell": str(pair["cell"]),
                    "v1_reference": v1_reference.get(pair_id, {}),
                    "euclidean": {
                        **euclidean_trace,
                        "summary": trace_summary(euclidean_trace),
                    },
                    "cpsi": {
                        **cpsi_trace,
                        "summary": trace_summary(cpsi_trace),
                    },
                    "divergence": divergence_summary(euclidean_trace, cpsi_trace),
                }
            )

        first_pair = pairs_by_id[pair_ids[0]]
        first_initial, first_goal = load_pair_rows(dataset, first_pair)
        print(f"\nLocal landscape probe around Euclidean final mean for pair={pair_ids[0]}")
        landscape = probe_local_landscape(
            reference_mean=pair_results[0]["euclidean"]["final_mean"],
            cost_head=cost_head,
            world_model=world_model,
            policy=policy,
            initial=first_initial,
            goal=first_goal,
            args=args,
            pair_id=pair_ids[0],
        )
    finally:
        env.close()

    split1_regressions = analyze_split1_regressions(args.split1_planning_result)
    payload = {
        "metadata": {
            "seed": args.seed,
            "device": args.device,
            "split": args.split,
            "model_type": args.model_type,
            "variant": loaded_variant,
            "cost_model_checkpoint": checkpoint_path,
            "planning_result": args.planning_result,
            "split1_planning_result": args.split1_planning_result,
            "pairs_path": args.pairs_path,
            "v1_dir": args.v1_dir,
            "diagnostic_pair_ids": pair_ids,
            "elite_v1_count": args.elite_v1_count,
            "landscape_samples": args.landscape_samples,
            "landscape_noise": args.landscape_noise,
            "output_path": args.output_path,
            "elapsed_seconds": float(time.time() - started),
            "cem_config": {
                "num_samples": NUM_SAMPLES,
                "iterations": CEM_ITERS,
                "topk": TOPK,
                "var_scale": VAR_SCALE,
                "planning_horizon": PLANNING_HORIZON,
                "action_block": ACTION_BLOCK,
            },
        },
        "summary": build_summary(
            pair_results=pair_results,
            landscape=landscape,
            split1_regressions=split1_regressions,
        ),
        "pair_results": pair_results,
    }
    save_json(args.output_path, payload)
    return payload


def main() -> int:
    """CLI entry point."""
    payload = run(parse_args())
    print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
