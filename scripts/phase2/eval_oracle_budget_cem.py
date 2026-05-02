#!/usr/bin/env python3
"""Evaluate oracle-budget CEM with a Top-K coarse prefilter.

This diagnostic planner keeps the LeWM CEM sampling/refit loop intact, but
changes how elites are selected. For K=0 it uses the usual Euclidean predicted
latent cost or a trained C_psi cost, depending on ``--prefilter``. For K>0 it
first takes the coarse top-K candidates, labels only those candidates with the
real PushT V1 hinge cost, and refits CEM from the 30 lowest-V1 candidates inside
that prefiltered pool.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass
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
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import (  # noqa: E402
    block_pose_metrics,
    configure_goal_render,
    prepare_pair_info,
)
from lewm_audit.eval.oracle_cem import cost_v1_hinge, rollout_final_state  # noqa: E402
from scripts.phase2.eval_planning import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    EVAL_BUDGET,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    RECEDING_HORIZON,
    TOPK,
    VAR_SCALE,
    find_cost_head_checkpoint,
    load_cost_head,
)
from scripts.phase2.splits import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    load_track_a_pairs,
    split1_random_holdout,
    split3_hard_pair_holdout,
)
from scripts.phase2.train_cem_aware import rollout_candidate_latents  # noqa: E402


DEFAULT_DATASET_NAME = "pusht_expert_train"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "oracle_budget_cem"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "oracle_budget_cem_summary.json"
DEFAULT_V1_DIR = PROJECT_ROOT / "results" / "phase1" / "v1_oracle_ablation"
DEFAULT_K_VALUES = (0, 30, 60, 90, 150, 300)
OFFSET = 50


@dataclass
class PairRows:
    """Initial and goal rows for one Track A pair."""

    initial: dict[str, Any]
    goal: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, choices=(1, 3), default=3)
    parser.add_argument("--k-values", default="0,30,60,90,150,300")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--pairs-subset", default=None)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--prefilter",
        choices=("euclidean", "cpsi"),
        default="euclidean",
        help="Coarse cost used for K=0 elite selection and K>0 Top-K prefiltering.",
    )
    parser.add_argument(
        "--cpsi-checkpoint",
        type=Path,
        default=None,
        help="Trained C_psi checkpoint file or directory. Required when --prefilter=cpsi.",
    )
    parser.add_argument(
        "--cpsi-variant",
        choices=("small", "large"),
        default="small",
        help="C_psi architecture variant to load when --prefilter=cpsi.",
    )
    parser.add_argument(
        "--skip-split1-followup",
        action="store_true",
        help="Do not automatically run Split 1 at the selected Split 3 knee K.",
    )
    parser.add_argument(
        "--oracle-iters",
        default="all",
        help=(
            "0-indexed CEM iterations that receive oracle re-ranking. "
            "Keywords: all, all-30, early-5, early-10, late-10, every-3rd, first-only. "
            "Non-oracle iterations use Euclidean elite selection."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the Split 3 open-loop K=300 validation against Phase 1 V1 artifacts.",
    )
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
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")


def parse_k_values(raw: str) -> list[int]:
    """Parse comma-separated K values."""
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 0 or value > NUM_SAMPLES:
            raise argparse.ArgumentTypeError(f"K must be in [0, {NUM_SAMPLES}], got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--k-values must include at least one K")
    for value in values:
        if value != 0 and value < TOPK:
            raise argparse.ArgumentTypeError(f"Positive K must be >= TOPK={TOPK}, got {value}")
    return values


def parse_pair_ids(raw: str | None) -> list[int] | None:
    """Parse an optional comma-separated pair ID list."""
    if raw is None:
        return None
    pair_ids = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            pair_ids.append(int(chunk))
    if not pair_ids:
        raise argparse.ArgumentTypeError("--pairs-subset must include at least one pair ID")
    return list(dict.fromkeys(pair_ids))


def parse_oracle_iters(raw: str) -> list[int]:
    """Parse oracle intervention iterations as 0-indexed CEM iteration IDs."""
    normalized = raw.strip().lower()
    presets = {
        "all": list(range(CEM_ITERS)),
        "all-30": list(range(CEM_ITERS)),
        "early-5": list(range(5)),
        "early-10": list(range(10)),
        "late-10": list(range(CEM_ITERS - 10, CEM_ITERS)),
        "every-3rd": list(range(0, CEM_ITERS, 3)),
        "first-only": [0],
    }
    if normalized in presets:
        return presets[normalized]
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 0 or value >= CEM_ITERS:
            raise argparse.ArgumentTypeError(
                f"oracle iteration must be in [0, {CEM_ITERS - 1}], got {value}"
            )
        if value not in out:
            out.append(value)
    if not out:
        raise argparse.ArgumentTypeError("--oracle-iters must include at least one iteration")
    return out


def oracle_iters_label(raw: str) -> str:
    """Return a filesystem-friendly label for an oracle-iteration spec."""
    return raw.strip().lower().replace(",", "-").replace(" ", "")


def load_pairs_by_id(path: Path) -> dict[int, dict]:
    """Load Track A pairs keyed by pair ID."""
    return {int(pair["pair_id"]): pair for pair in load_track_a_pairs(path)}


def split_pair_ids(split: int, *, pairs_path: Path, seed: int) -> list[int]:
    """Return test pair IDs for the requested split."""
    if split == 1:
        return split1_random_holdout(pairs_path, seed=seed)["test_pair_ids"]
    if split == 3:
        return split3_hard_pair_holdout(pairs_path, seed=seed)["test_pair_ids"]
    raise ValueError(f"Unsupported split: {split}")


def load_pair_rows(dataset, pair: dict) -> PairRows:
    """Load initial and goal rows for a Track A pair."""
    rows = dataset.get_row_data([int(pair["start_row"]), int(pair["goal_row"])])
    return PairRows(
        initial={key: value[0] for key, value in rows.items()},
        goal={key: value[1] for key, value in rows.items()},
    )


def make_policy_args(args: argparse.Namespace, *, horizon: int = PLANNING_HORIZON) -> argparse.Namespace:
    """Build args consumed by the existing Phase 1 policy helper."""
    return argparse.Namespace(
        checkpoint_dir=args.model_checkpoint_dir,
        device=args.device,
        num_samples=NUM_SAMPLES,
        var_scale=VAR_SCALE,
        cem_iters=CEM_ITERS,
        topk=TOPK,
        seed=args.seed,
        horizon=horizon,
        receding_horizon=horizon,
        action_block=ACTION_BLOCK,
        img_size=IMG_SIZE,
    )


def euclidean_costs(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean terminal latent costs."""
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    return torch.sum((z_pred - z_goal) ** 2, dim=-1)


def cpsi_costs(
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    cost_head: torch.nn.Module,
) -> torch.Tensor:
    """Compute trained C_psi costs for candidate terminal latents."""
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    flat_pred = z_pred.reshape(-1, z_pred.shape[-1])
    flat_goal = z_goal.reshape(-1, z_goal.shape[-1])
    costs = cost_head(flat_pred, flat_goal)
    return costs.reshape(z_pred.shape[:-1])


def prefilter_costs(
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    *,
    prefilter: str,
    cost_head: torch.nn.Module | None,
) -> torch.Tensor:
    """Compute the coarse prefilter costs used for CEM elite selection."""
    if prefilter == "euclidean":
        return euclidean_costs(z_pred, z_goal)
    if prefilter == "cpsi":
        if cost_head is None:
            raise ValueError("C_psi prefilter requested without a loaded cost head")
        return cpsi_costs(z_pred, z_goal, cost_head)
    raise ValueError(f"Unknown prefilter: {prefilter!r}")


def blocked_batch_to_raw_fast(blocked: np.ndarray, *, action_processor) -> np.ndarray:
    """Convert blocked normalized candidates to raw env actions in one transform call."""
    blocked = np.asarray(blocked, dtype=np.float32)
    if blocked.ndim != 3 or blocked.shape[-1] != ACTION_BLOCK * 2:
        raise ValueError(f"Expected blocked shape (N,H,{ACTION_BLOCK * 2}), got {blocked.shape}")
    n_cand, horizon, _ = blocked.shape
    normalized = blocked.reshape(n_cand * horizon * ACTION_BLOCK, 2)
    raw = action_processor.inverse_transform(normalized).astype(np.float32)
    return raw.reshape(n_cand, horizon * ACTION_BLOCK, 2)


def set_unwrapped_state(env_unwrapped, initial_state: np.ndarray, goal_state: np.ndarray) -> None:
    """Set PushT state and goal on an unwrapped environment."""
    configure_goal_render(env_unwrapped, goal_state)
    if hasattr(env_unwrapped, "_set_state"):
        env_unwrapped._set_state(np.asarray(initial_state, dtype=np.float32))
    elif hasattr(env_unwrapped, "set_state"):
        env_unwrapped.set_state(np.asarray(initial_state, dtype=np.float32))
    else:
        raise AttributeError("PushT environment does not expose a supported state setter")


def get_unwrapped_state(env_unwrapped) -> np.ndarray:
    """Get PushT physical state from an unwrapped environment."""
    if hasattr(env_unwrapped, "_get_obs"):
        return np.asarray(env_unwrapped._get_obs(), dtype=np.float32)
    if hasattr(env_unwrapped, "get_state"):
        return np.asarray(env_unwrapped.get_state(), dtype=np.float32)
    raise AttributeError("PushT environment does not expose a supported state getter")


def score_candidates_v1(
    *,
    oracle_env,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions_batch: np.ndarray,
    seed_base: int,
    seed_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Score raw action candidates using the exact Phase 1 reset/rollout path."""
    raw_actions_batch = np.asarray(raw_actions_batch, dtype=np.float32)
    if seed_indices is None:
        seed_indices = np.arange(raw_actions_batch.shape[0], dtype=np.int64)
    else:
        seed_indices = np.asarray(seed_indices, dtype=np.int64)
        if seed_indices.shape != (raw_actions_batch.shape[0],):
            raise ValueError(
                "seed_indices must have one value per candidate: "
                f"{seed_indices.shape} vs {raw_actions_batch.shape[0]}"
            )
    costs = np.empty((raw_actions_batch.shape[0],), dtype=np.float64)
    metrics = []
    for idx, raw_actions in enumerate(raw_actions_batch):
        terminal_state = rollout_final_state(
            oracle_env,
            initial_state,
            goal_state,
            raw_actions,
            seed=seed_base + int(seed_indices[idx]),
        )
        cost = float(cost_v1_hinge(terminal_state, goal_state))
        pose = block_pose_metrics(terminal_state, goal_state)
        costs[idx] = cost
        metrics.append(
            {
                "v1_cost": cost,
                "block_pos_dist": float(pose["block_pos_dist"]),
                "angle_dist": float(pose["angle_dist"]),
                "success": bool(pose["success"]),
            }
        )
    return costs, metrics


@torch.inference_mode()
def solve_oracle_budget_cem(
    *,
    world_model,
    policy,
    oracle_env,
    action_processor,
    prefilter: str,
    cost_head: torch.nn.Module | None,
    current_pixels: np.ndarray,
    goal_pixels: np.ndarray,
    current_state: np.ndarray,
    goal_state: np.ndarray,
    k_value: int,
    oracle_iterations: set[int],
    generator: torch.Generator,
    init_action: torch.Tensor | None,
    horizon: int,
) -> dict:
    """Solve one CEM planning problem with partial oracle elite selection."""
    device = next(world_model.parameters()).device
    action_dim = ACTION_BLOCK * 2
    if init_action is None:
        mean = torch.zeros((1, 0, action_dim), dtype=torch.float32, device=device)
    else:
        mean = init_action.to(device=device, dtype=torch.float32)
    remaining = horizon - mean.shape[1]
    if remaining > 0:
        mean = torch.cat(
            [mean, torch.zeros((1, remaining, action_dim), dtype=torch.float32, device=device)],
            dim=1,
        )
    elif remaining < 0:
        mean = mean[:, :horizon]
    std = VAR_SCALE * torch.ones((1, horizon, action_dim), dtype=torch.float32, device=device)

    prepared_info = prepare_pair_info(policy, current_pixels, goal_pixels)
    oracle_rollouts = 0
    oracle_env_steps = 0
    iter_stats = []

    for iter_idx in range(1, CEM_ITERS + 1):
        iter_zero = iter_idx - 1
        candidates = torch.randn(
            1,
            NUM_SAMPLES,
            horizon,
            action_dim,
            generator=generator,
            device=device,
        )
        candidates = candidates * std.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        z_pred, z_goal = rollout_candidate_latents(world_model, prepared_info, candidates)
        euc_cost = euclidean_costs(z_pred, z_goal)
        coarse_cost = prefilter_costs(
            z_pred,
            z_goal,
            prefilter=prefilter,
            cost_head=cost_head,
        )

        if k_value == 0:
            top_vals, top_inds = torch.topk(coarse_cost, k=TOPK, dim=1, largest=False)
            elite_inds = top_inds[0]
            selected_costs = top_vals[0].detach().cpu().numpy().astype(np.float64)
            oracle_best = None
            used_oracle = False
        elif iter_zero not in oracle_iterations:
            top_vals, top_inds = torch.topk(euc_cost, k=TOPK, dim=1, largest=False)
            elite_inds = top_inds[0]
            selected_costs = top_vals[0].detach().cpu().numpy().astype(np.float64)
            oracle_best = None
            used_oracle = False
        else:
            pre_vals, pre_inds = torch.topk(coarse_cost, k=k_value, dim=1, largest=False)
            # Membership is determined by the coarse prefilter. Simulator scoring
            # order and V1 tie-breaking are then restored to original candidate
            # order so K=300 is exactly full-oracle and independent of prefilter.
            pre_inds_sorted = torch.sort(pre_inds[0]).values
            pre_original_indices = pre_inds_sorted.detach().cpu().numpy().astype(np.int64)
            blocked = candidates[0, pre_inds_sorted].detach().cpu().numpy().astype(np.float32)
            raw_candidates = blocked_batch_to_raw_fast(
                blocked,
                action_processor=action_processor,
            )
            oracle_costs, oracle_metrics = score_candidates_v1(
                oracle_env=oracle_env,
                initial_state=current_state,
                goal_state=goal_state,
                raw_actions_batch=raw_candidates,
                seed_base=iter_idx * 100_000,
                seed_indices=pre_original_indices,
            )
            local_order = np.lexsort((pre_original_indices, oracle_costs))[:TOPK]
            elite_inds = pre_inds_sorted[torch.as_tensor(local_order, dtype=torch.long, device=device)]
            selected_costs = oracle_costs[local_order]
            oracle_rollouts += int(k_value)
            oracle_env_steps += int(k_value * horizon * ACTION_BLOCK)
            best_local = int(local_order[0])
            best_original_index = int(pre_original_indices[best_local])
            prefilter_rank = int(
                (pre_inds[0].detach().cpu().numpy().astype(np.int64) == best_original_index)
                .nonzero()[0][0]
            )
            oracle_best = {
                **oracle_metrics[best_local],
                "prefilter_rank": prefilter_rank,
                "original_candidate_index": best_original_index,
                "prefilter_cost": float(
                    coarse_cost[0, best_original_index].detach().cpu()
                ),
            }
            used_oracle = True

        elite_candidates = candidates[:, elite_inds]
        mean = elite_candidates.mean(dim=1)
        std = elite_candidates.std(dim=1)
        iter_stats.append(
            {
                "iteration": iter_idx,
                "iteration_zero_indexed": iter_zero,
                "k": int(k_value),
                "prefilter": prefilter,
                "used_oracle": bool(used_oracle),
                "selected_cost_min": float(np.min(selected_costs)),
                "selected_cost_mean": float(np.mean(selected_costs)),
                "selected_cost_std": float(np.std(selected_costs)),
                "std_mean": float(std.detach().cpu().mean()),
                "oracle_best": oracle_best,
            }
        )

    return {
        "actions": mean.detach().cpu(),
        "var": std.detach().cpu(),
        "oracle_rollouts": oracle_rollouts,
        "oracle_env_steps": oracle_env_steps,
        "iterations": iter_stats,
    }


class OracleBudgetPlanner:
    """Small receding-horizon policy wrapper for oracle-budget CEM."""

    def __init__(
        self,
        *,
        policy,
        k_value: int,
        seed: int,
        device: str,
        oracle_env,
        prefilter: str,
        cost_head: torch.nn.Module | None,
        oracle_iterations: set[int],
    ) -> None:
        self.policy = policy
        self.world_model = policy.solver.model
        self.action_processor = policy.process["action"]
        self.k_value = int(k_value)
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.oracle_env = oracle_env
        self.prefilter = prefilter
        self.cost_head = cost_head
        self.oracle_iterations = set(oracle_iterations)
        self.action_buffer: deque[np.ndarray] = deque(maxlen=RECEDING_HORIZON * ACTION_BLOCK)
        self.next_init: torch.Tensor | None = None
        self.oracle_rollouts = 0
        self.oracle_env_steps = 0
        self.solve_count = 0
        self.solve_logs = []

    def reset_pair(self) -> None:
        """Clear receding-horizon warm-start state for a new pair."""
        self.action_buffer.clear()
        self.next_init = None

    def get_action(
        self,
        *,
        pixels: np.ndarray,
        goal_pixels: np.ndarray,
        current_state: np.ndarray,
        goal_state: np.ndarray,
    ) -> np.ndarray:
        """Return one raw environment action, replanning when the buffer is empty."""
        if not self.action_buffer:
            solve = solve_oracle_budget_cem(
                world_model=self.world_model,
                policy=self.policy,
                oracle_env=self.oracle_env,
                action_processor=self.action_processor,
                prefilter=self.prefilter,
                cost_head=self.cost_head,
                current_pixels=pixels,
                goal_pixels=goal_pixels,
                current_state=current_state,
                goal_state=goal_state,
                k_value=self.k_value,
                oracle_iterations=self.oracle_iterations,
                generator=self.generator,
                init_action=self.next_init,
                horizon=PLANNING_HORIZON,
            )
            actions = solve["actions"]
            keep_horizon = RECEDING_HORIZON
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self.next_init = rest
            blocked = plan.numpy().reshape(1, keep_horizon, ACTION_BLOCK * 2)
            raw_plan = blocked_batch_to_raw_fast(blocked, action_processor=self.action_processor)
            self.action_buffer.extend(raw_plan[0])
            self.oracle_rollouts += int(solve["oracle_rollouts"])
            self.oracle_env_steps += int(solve["oracle_env_steps"])
            self.solve_count += 1
            self.solve_logs.append(
                {
                    "solve_index": self.solve_count,
                    "oracle_rollouts": int(solve["oracle_rollouts"]),
                    "oracle_env_steps": int(solve["oracle_env_steps"]),
                    "final_iteration": solve["iterations"][-1],
                }
            )
        return np.asarray(self.action_buffer.popleft(), dtype=np.float32)


def rollout_pair_planning(
    *,
    env,
    planner: OracleBudgetPlanner,
    pair_rows: PairRows,
    pair_id: int,
    seed: int,
) -> dict:
    """Run one 100-step receding-horizon rollout for one Track A pair."""
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    initial_state = np.asarray(pair_rows.initial["state"], dtype=np.float32)
    goal_state = np.asarray(pair_rows.goal["state"], dtype=np.float32)
    goal_pixels = np.asarray(pair_rows.goal["pixels"], dtype=np.uint8)
    set_unwrapped_state(env_unwrapped, initial_state, goal_state)
    planner.reset_pair()

    before_rollouts = planner.oracle_rollouts
    before_steps = planner.oracle_env_steps
    before_solves = planner.solve_count
    env_success = False
    started = time.time()
    for _ in range(EVAL_BUDGET):
        pixels = np.asarray(env_unwrapped.render(), dtype=np.uint8)
        current_state = get_unwrapped_state(env_unwrapped)
        action = planner.get_action(
            pixels=pixels,
            goal_pixels=goal_pixels,
            current_state=current_state,
            goal_state=goal_state,
        )
        _, _, terminated, _, _ = env.step(np.asarray(action, dtype=np.float32))
        env_success = env_success or bool(terminated)

    terminal_state = get_unwrapped_state(env_unwrapped)
    metrics = block_pose_metrics(terminal_state, goal_state)
    return {
        "success": bool(metrics["success"]),
        "env_success": bool(env_success),
        "block_pos_dist": float(metrics["block_pos_dist"]),
        "angle_dist": float(metrics["angle_dist"]),
        "c_real_state": float(metrics["c_real_state"]),
        "oracle_rollouts": int(planner.oracle_rollouts - before_rollouts),
        "oracle_env_steps": int(planner.oracle_env_steps - before_steps),
        "cem_solves": int(planner.solve_count - before_solves),
        "elapsed_seconds": float(time.time() - started),
    }


def evaluate_k_on_pairs(
    *,
    k_value: int,
    selected_pairs: list[dict],
    dataset,
    process: dict,
    args: argparse.Namespace,
    output_path: Path,
    prefilter: str,
    cost_head: torch.nn.Module | None,
    oracle_iterations: set[int],
) -> dict:
    """Evaluate one K value on the selected planning pairs."""
    policy = build_policy(make_policy_args(args), process)
    env = gym.make("swm/PushT-v1")
    oracle_env = gym.make("swm/PushT-v1")
    oracle_env.reset(seed=args.seed + 12345)
    planner = OracleBudgetPlanner(
        policy=policy,
        k_value=k_value,
        seed=args.seed,
        device=args.device,
        oracle_env=oracle_env,
        prefilter=prefilter,
        cost_head=cost_head,
        oracle_iterations=oracle_iterations,
    )
    records = []
    started = time.time()
    try:
        for idx, pair in enumerate(selected_pairs, start=1):
            pair_id = int(pair["pair_id"])
            pair_rows = load_pair_rows(dataset, pair)
            print(f"[K={k_value:>3} {idx:02d}/{len(selected_pairs):02d}] pair={pair_id} cell={pair['cell']}")
            result = rollout_pair_planning(
                env=env,
                planner=planner,
                pair_rows=pair_rows,
                pair_id=pair_id,
                seed=args.seed + pair_id * 1009,
            )
            print(
                f"  success={result['success']} dist={result['block_pos_dist']:.2f} "
                f"angle={result['angle_dist']:.3f} "
                f"oracle_rollouts={result['oracle_rollouts']} elapsed={result['elapsed_seconds']:.1f}s"
            )
            records.append(
                {
                    "pair_id": pair_id,
                    "cell": str(pair["cell"]),
                    "episode_id": int(pair["episode_id"]),
                    "start_row": int(pair["start_row"]),
                    "goal_row": int(pair["goal_row"]),
                    "result": result,
                }
            )
    finally:
        env.close()
        oracle_env.close()

    elapsed = time.time() - started
    success_count = int(sum(bool(record["result"]["success"]) for record in records))
    aggregate = {
        "k": int(k_value),
        "n_pairs": int(len(records)),
        "success_count": success_count,
        "success_rate": float(success_count / len(records)) if records else None,
        "oracle_rollouts": int(sum(record["result"]["oracle_rollouts"] for record in records)),
        "oracle_env_steps": int(sum(record["result"]["oracle_env_steps"] for record in records)),
        "elapsed_seconds": float(elapsed),
    }
    payload = {
        "metadata": {
            "split": int(args.split),
            "k": int(k_value),
            "prefilter": prefilter,
            "oracle_iters": args.oracle_iters,
            "oracle_iterations": sorted(int(item) for item in oracle_iterations),
            "seed": int(args.seed),
            "device": args.device,
            "pairs": [int(pair["pair_id"]) for pair in selected_pairs],
            "output_path": output_path,
            "cem_config": cem_config(planning=True),
        },
        "records": records,
        "aggregate": aggregate,
    }
    save_json(output_path, payload)
    return payload


def cem_config(*, planning: bool) -> dict:
    """Return the CEM configuration as JSON metadata."""
    return {
        "num_samples": NUM_SAMPLES,
        "iterations": CEM_ITERS,
        "topk": TOPK,
        "var_scale": VAR_SCALE,
        "planning_horizon": PLANNING_HORIZON if planning else OFFSET // ACTION_BLOCK,
        "receding_horizon": RECEDING_HORIZON if planning else None,
        "action_block": ACTION_BLOCK,
        "eval_budget": EVAL_BUDGET if planning else None,
        "offset": OFFSET,
    }


def per_k_output_path(
    output_dir: Path,
    *,
    split: int,
    k_value: int,
    prefilter: str,
    oracle_label: str,
) -> Path:
    """Return the per-K artifact path without overwriting Euclidean baselines."""
    suffix = "" if oracle_label in {"all", "all-30"} else f"_oracle_{oracle_label}"
    if prefilter == "euclidean":
        return output_dir / f"split{split}_k{k_value}{suffix}.json"
    return output_dir / f"split{split}_{prefilter}_k{k_value}{suffix}.json"


def load_prefilter_model(args: argparse.Namespace) -> tuple[Path | None, torch.nn.Module | None]:
    """Load the optional trained C_psi prefilter model."""
    if args.prefilter == "euclidean":
        return None, None
    if args.cpsi_checkpoint is None:
        raise ValueError("--cpsi-checkpoint is required when --prefilter=cpsi")
    checkpoint_root = args.cpsi_checkpoint.expanduser().resolve()
    checkpoint_path = find_cost_head_checkpoint(
        checkpoint_root,
        split=args.split,
        variant=args.cpsi_variant,
    )
    model = load_cost_head(
        checkpoint_path,
        variant=args.cpsi_variant,
        device=args.device,
    )
    return checkpoint_path, model


def load_v1_reference(v1_dir: Path) -> dict[int, dict]:
    """Load Phase 1 V1 late CEM records keyed by pair ID."""
    out: dict[int, dict] = {}
    for path in sorted(v1_dir.glob("v1_d*.json")):
        data = json.loads(path.read_text())
        for pair in data.get("pairs", []):
            late = [action for action in pair.get("actions", []) if action.get("source") == "CEM_late_V1"]
            out[int(pair["pair_id"])] = {
                "cell": str(pair["cell"]),
                "late_n": len(late),
                "late_success_count": int(sum(bool(action["success"]) for action in late)),
                "late_min_v1": min((float(action["C_variant"]) for action in late), default=None),
            }
    return out


def run_open_loop_full_oracle_validation_pair(
    *,
    pair: dict,
    dataset,
    action_processor,
    oracle_env,
    seed: int,
) -> dict:
    """Run Track A style open-loop full-oracle CEM for one pair."""
    rows = load_pair_rows(dataset, pair)
    initial_state = np.asarray(rows.initial["state"], dtype=np.float32)
    goal_state = np.asarray(rows.goal["state"], dtype=np.float32)
    pair_id = int(pair["pair_id"])
    horizon = OFFSET // ACTION_BLOCK
    action_dim = ACTION_BLOCK * 2
    rng = np.random.default_rng(seed + pair_id * 1009)
    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = VAR_SCALE * np.ones((horizon, action_dim), dtype=np.float32)
    final_elite_costs = None
    final_elite_metrics = None
    final_elite_raw = None

    for iter_idx in range(CEM_ITERS):
        blocked_candidates = rng.normal(
            loc=mean[None, :, :],
            scale=std[None, :, :],
            size=(NUM_SAMPLES, horizon, action_dim),
        ).astype(np.float32)
        blocked_candidates[0] = mean
        raw_candidates = blocked_batch_to_raw_fast(
            blocked_candidates,
            action_processor=action_processor,
        )
        costs, metrics = score_candidates_v1(
            oracle_env=oracle_env,
            initial_state=initial_state,
            goal_state=goal_state,
            raw_actions_batch=raw_candidates,
            seed_base=0,
        )
        elite_indices = np.argsort(costs, kind="stable")[:TOPK]
        elite_blocked = blocked_candidates[elite_indices]
        final_elite_costs = costs[elite_indices].astype(np.float64)
        final_elite_metrics = [metrics[int(idx)] for idx in elite_indices]
        final_elite_raw = raw_candidates[elite_indices].astype(np.float32)
        mean = elite_blocked.mean(axis=0).astype(np.float32)
        std = elite_blocked.std(axis=0).astype(np.float32)

    late20_costs = final_elite_costs[:20]
    late20_planner_metrics = final_elite_metrics[:20]
    # Phase 1 stores CEM's oracle costs, but reports success/C_variant after
    # re-executing selected early then late actions. Match that artifact
    # semantics exactly: late actions have action_id 20..39 in CEM-only output.
    late20_reexec_costs = []
    late20_reexec_metrics = []
    for source_index, raw_actions in enumerate(final_elite_raw[:20]):
        terminal_state = rollout_final_state(
            oracle_env,
            initial_state,
            goal_state,
            raw_actions,
            seed=seed + pair_id * 10_000 + 20 + source_index,
        )
        cost = float(cost_v1_hinge(terminal_state, goal_state))
        pose = block_pose_metrics(terminal_state, goal_state)
        late20_reexec_costs.append(cost)
        late20_reexec_metrics.append(
            {
                "v1_cost": cost,
                "block_pos_dist": float(pose["block_pos_dist"]),
                "angle_dist": float(pose["angle_dist"]),
                "success": bool(pose["success"]),
            }
        )
    return {
        "pair_id": pair_id,
        "cell": str(pair["cell"]),
        "late20_success_count": int(
            sum(bool(item["success"]) for item in late20_reexec_metrics)
        ),
        "late20_min_v1": float(np.min(late20_reexec_costs)),
        "late20_reexec_costs": [float(value) for value in late20_reexec_costs],
        "late20_planner_success_count": int(
            sum(bool(item["success"]) for item in late20_planner_metrics)
        ),
        "late20_planner_min_v1": float(np.min(late20_costs)),
        "late20_planner_costs": late20_costs.tolist(),
    }


def validate_full_oracle_against_phase1(
    *,
    selected_pairs: list[dict],
    dataset,
    process: dict,
    args: argparse.Namespace,
) -> dict:
    """Validate open-loop K=300 against Phase 1 V1 oracle artifacts."""
    policy = build_policy(make_policy_args(args, horizon=OFFSET // ACTION_BLOCK), process)
    action_processor = policy.process["action"]
    v1_reference = load_v1_reference(args.v1_dir)
    oracle_env = gym.make("swm/PushT-v1")
    oracle_env.reset(seed=args.seed + 54321)
    records = []
    started = time.time()
    try:
        for idx, pair in enumerate(selected_pairs, start=1):
            print(f"[validation {idx:02d}/{len(selected_pairs):02d}] pair={pair['pair_id']} cell={pair['cell']}")
            observed = run_open_loop_full_oracle_validation_pair(
                pair=pair,
                dataset=dataset,
                action_processor=action_processor,
                oracle_env=oracle_env,
                seed=args.seed,
            )
            expected = v1_reference.get(int(pair["pair_id"]))
            if expected is None:
                raise KeyError(f"Missing Phase 1 V1 reference for pair {pair['pair_id']}")
            min_diff = abs(float(observed["late20_min_v1"]) - float(expected["late_min_v1"]))
            success_diff = int(observed["late20_success_count"]) - int(expected["late_success_count"])
            matches = success_diff == 0 and min_diff <= 1e-4
            records.append(
                {
                    **observed,
                    "phase1_late20_success_count": int(expected["late_success_count"]),
                    "phase1_late20_min_v1": float(expected["late_min_v1"]),
                    "success_count_diff": success_diff,
                    "min_v1_abs_diff": float(min_diff),
                    "matches_phase1": bool(matches),
                }
            )
            print(
                f"  observed={observed['late20_success_count']}/20 min={observed['late20_min_v1']:.6f} "
                f"phase1={expected['late_success_count']}/20 min={expected['late_min_v1']:.6f} "
                f"match={matches}"
            )
    finally:
        oracle_env.close()

    mismatches = [record for record in records if not record["matches_phase1"]]
    payload = {
        "n_pairs": len(records),
        "n_mismatches": len(mismatches),
        "matches_phase1": len(mismatches) == 0,
        "records": records,
        "elapsed_seconds": float(time.time() - started),
    }
    return payload


def choose_knee(sweep_rows: list[dict]) -> int:
    """Choose the smallest K within one success of the best Split 3 count."""
    best = max(int(row["success_count"]) for row in sweep_rows)
    threshold = best - 1
    for row in sorted(sweep_rows, key=lambda item: int(item["k"])):
        if int(row["success_count"]) >= threshold:
            return int(row["k"])
    return int(sorted(sweep_rows, key=lambda item: int(item["k"]))[0]["k"])


def summarize_split1_regressions(split1_payload: dict) -> dict:
    """Report whether known Split 1 learned-cost regressions remain solved."""
    by_pair = {int(record["pair_id"]): record for record in split1_payload.get("records", [])}
    out = {}
    for pair_id in (29, 41):
        record = by_pair.get(pair_id)
        if record is None:
            out[pair_id] = {"evaluated": False}
            continue
        result = record["result"]
        out[pair_id] = {
            "evaluated": True,
            "cell": record["cell"],
            "success": bool(result["success"]),
            "block_pos_dist": float(result["block_pos_dist"]),
            "angle_dist": float(result["angle_dist"]),
        }
    return out


def print_sweep_table(title: str, rows: list[dict], *, baseline_success: int | None = None) -> None:
    """Print K sweep aggregate rows."""
    print(f"\n== {title} ==")
    print("K | success | rate | delta_vs_K0 | oracle_rollouts | oracle_env_steps | wallclock_s")
    for row in rows:
        base = baseline_success if baseline_success is not None else rows[0]["success_count"]
        delta = int(row["success_count"]) - int(base)
        print(
            f"{int(row['k']):>3} | "
            f"{int(row['success_count']):>2}/{int(row['n_pairs']):<2} | "
            f"{float(row['success_rate']):.4f} | "
            f"{delta:+d} | "
            f"{int(row['oracle_rollouts']):>8} | "
            f"{int(row['oracle_env_steps']):>10} | "
            f"{float(row['elapsed_seconds']):.1f}"
        )


def run(args: argparse.Namespace) -> dict:
    """Run oracle-budget CEM evaluation."""
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    args.output_path = args.output_path.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.v1_dir = args.v1_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.model_checkpoint_dir = args.model_checkpoint_dir.expanduser().resolve()
    if args.cpsi_checkpoint is not None:
        args.cpsi_checkpoint = args.cpsi_checkpoint.expanduser().resolve()
    k_values = parse_k_values(args.k_values)
    explicit_pair_ids = parse_pair_ids(args.pairs_subset)
    oracle_iterations = set(parse_oracle_iters(args.oracle_iters))
    oracle_label = oracle_iters_label(args.oracle_iters)
    cpsi_checkpoint_path, cost_head = load_prefilter_model(args)

    pairs_by_id = load_pairs_by_id(args.pairs_path)
    selected_pair_ids = (
        explicit_pair_ids
        if explicit_pair_ids is not None
        else split_pair_ids(args.split, pairs_path=args.pairs_path, seed=args.seed)
    )
    missing = sorted(set(selected_pair_ids) - set(pairs_by_id))
    if missing:
        raise ValueError(f"Missing Track A pair IDs: {missing}")
    selected_pairs = [pairs_by_id[pair_id] for pair_id in selected_pair_ids]

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    output_dir = args.output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("== Oracle-budget CEM setup ==")
    print(f"device: {args.device}")
    print(f"split: {args.split}")
    print(f"k_values: {k_values}")
    print(f"prefilter: {args.prefilter}")
    print(f"oracle_iters: {args.oracle_iters} -> {sorted(oracle_iterations)}")
    if cpsi_checkpoint_path is not None:
        print(f"cpsi_checkpoint: {cpsi_checkpoint_path}")
    print(f"pairs: {selected_pair_ids}")
    print(f"output_path: {args.output_path}")

    validation = None
    if (
        args.split == 3
        and not args.skip_validation
        and args.prefilter == "euclidean"
        and 300 in k_values
        and oracle_label in {"all", "all-30"}
    ):
        print("\n== K=300 open-loop Phase 1 validation ==")
        validation = validate_full_oracle_against_phase1(
            selected_pairs=selected_pairs,
            dataset=dataset,
            process=process,
            args=args,
        )
        validation_path = output_dir / "split3_k300_phase1_validation.json"
        save_json(validation_path, validation)
        if not validation["matches_phase1"]:
            raise RuntimeError(
                "K=300 open-loop validation did not match Phase 1 V1 artifacts; "
                f"details saved to {validation_path}"
            )

    sweep_payloads = []
    for k_value in k_values:
        per_k_path = per_k_output_path(
            output_dir,
            split=args.split,
            k_value=k_value,
            prefilter=args.prefilter,
            oracle_label=oracle_label,
        )
        print(f"\n== Planning evaluation K={k_value} ==")
        payload = evaluate_k_on_pairs(
            k_value=k_value,
            selected_pairs=selected_pairs,
            dataset=dataset,
            process=process,
            args=args,
            output_path=per_k_path,
            prefilter=args.prefilter,
            cost_head=cost_head,
            oracle_iterations=oracle_iterations,
        )
        sweep_payloads.append(payload)

    sweep_rows = [payload["aggregate"] for payload in sweep_payloads]
    baseline_success = next(
        (int(row["success_count"]) for row in sweep_rows if int(row["k"]) == 0),
        int(sweep_rows[0]["success_count"]),
    )
    print_sweep_table(f"Split {args.split} K sweep", sweep_rows, baseline_success=baseline_success)

    knee_k = choose_knee(sweep_rows)
    split1_payload = None
    if args.split == 3 and explicit_pair_ids is None and not args.skip_split1_followup:
        split1_ids = split_pair_ids(1, pairs_path=args.pairs_path, seed=args.seed)
        split1_pairs = [pairs_by_id[pair_id] for pair_id in split1_ids]
        split1_path = per_k_output_path(
            output_dir,
            split=1,
            k_value=knee_k,
            prefilter=args.prefilter,
            oracle_label=oracle_label,
        )
        split1_args = argparse.Namespace(**vars(args))
        split1_args.split = 1
        print(f"\n== Split 1 follow-up at knee K={knee_k} ==")
        split1_payload = evaluate_k_on_pairs(
            k_value=knee_k,
            selected_pairs=split1_pairs,
            dataset=dataset,
            process=process,
            args=split1_args,
            output_path=split1_path,
            prefilter=args.prefilter,
            cost_head=cost_head,
            oracle_iterations=oracle_iterations,
        )
        print_sweep_table(
            f"Split 1 knee K={knee_k}",
            [split1_payload["aggregate"]],
            baseline_success=4,
        )

    summary = {
        "metadata": {
            "seed": int(args.seed),
            "device": args.device,
            "split": int(args.split),
            "k_values": k_values,
            "prefilter": args.prefilter,
            "oracle_iters": args.oracle_iters,
            "oracle_iterations": sorted(int(item) for item in oracle_iterations),
            "cpsi_checkpoint": cpsi_checkpoint_path,
            "pairs": selected_pair_ids,
            "pairs_subset_override": explicit_pair_ids is not None,
            "output_path": args.output_path,
            "cem_config": cem_config(planning=True),
        },
        "phase1_validation": validation,
        "sweep": {
            "rows": sweep_rows,
            "baseline_k0_success_count": baseline_success,
            "knee_k": int(knee_k),
        },
        "split1_followup": (
            {
                "k": int(knee_k),
                "aggregate": split1_payload["aggregate"],
                "known_regression_pairs": summarize_split1_regressions(split1_payload),
                "path": split1_payload["metadata"]["output_path"],
            }
            if split1_payload is not None
            else None
        ),
        "per_k_paths": [payload["metadata"]["output_path"] for payload in sweep_payloads],
    }
    save_json(args.output_path, summary)
    return summary


def main() -> int:
    """CLI entry point."""
    summary = run(parse_args())
    print(f"\nselected_knee_k: {summary['sweep']['knee_k']}")
    if summary.get("split1_followup") is not None:
        follow = summary["split1_followup"]
        agg = follow["aggregate"]
        print(
            "split1_followup: "
            f"K={follow['k']} success={agg['success_count']}/{agg['n_pairs']} "
            f"known_regressions={follow['known_regression_pairs']}"
        )
    print(f"saved_summary: {summary['metadata']['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
