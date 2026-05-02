"""Oracle CEM planner with pluggable real-state terminal costs.

This module intentionally evaluates CEM candidates by rolling them out in the
real PushT environment and scoring the final state. It is an oracle planner for
diagnosis only; it does not use the LeWM predictor cost during CEM selection.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


BLOCK_SUCCESS_THRESHOLD_PX = 20.0
ANGLE_SUCCESS_THRESHOLD_RAD = math.pi / 9.0
HINGE_ALPHA = BLOCK_SUCCESS_THRESHOLD_PX / ANGLE_SUCCESS_THRESHOLD_RAD


def _angular_distance(a: float, b: float) -> float:
    diff = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
    return abs(diff)


def block_pose_components(state: np.ndarray, goal_state: np.ndarray) -> dict:
    """Return the same block pose components used by Track A C_real_state."""
    state = np.asarray(state, dtype=np.float64)
    goal_state = np.asarray(goal_state, dtype=np.float64)
    block_pos_dist = float(np.linalg.norm(state[2:4] - goal_state[2:4]))
    angle_dist = _angular_distance(float(state[4]), float(goal_state[4]))
    return {
        "block_pos_dist": block_pos_dist,
        "angle_dist": angle_dist,
        "success": bool(
            block_pos_dist < BLOCK_SUCCESS_THRESHOLD_PX
            and angle_dist < ANGLE_SUCCESS_THRESHOLD_RAD
        ),
    }


def cost_v1_hinge(state, goal_state) -> float:
    metrics = block_pose_components(state, goal_state)
    return float(
        max(metrics["block_pos_dist"] - BLOCK_SUCCESS_THRESHOLD_PX, 0.0)
        + HINGE_ALPHA * max(metrics["angle_dist"] - ANGLE_SUCCESS_THRESHOLD_RAD, 0.0)
    )


def cost_v2_indicator(state, goal_state) -> float:
    return 0.0 if block_pose_components(state, goal_state)["success"] else 1.0


def cost_v3_baseline(state, goal_state) -> float:
    metrics = block_pose_components(state, goal_state)
    return float(metrics["block_pos_dist"] + metrics["angle_dist"])


def _configure_goal_render(env_unwrapped, goal_state: np.ndarray) -> None:
    if hasattr(env_unwrapped, "_set_goal_state"):
        env_unwrapped._set_goal_state(goal_state)
    if hasattr(env_unwrapped, "goal_pose"):
        env_unwrapped.goal_pose = np.asarray(goal_state[2:5], dtype=np.float64)
    elif hasattr(env_unwrapped, "goal_state"):
        env_unwrapped.goal_state = np.asarray(goal_state, dtype=np.float64)


def _set_env_state(env, init_state: np.ndarray, goal_state: np.ndarray, seed: int) -> None:
    if hasattr(env, "reset"):
        env.reset(seed=seed)
    env_unwrapped = getattr(env, "unwrapped", env)
    _configure_goal_render(env_unwrapped, goal_state)
    if hasattr(env_unwrapped, "_set_state"):
        env_unwrapped._set_state(np.asarray(init_state, dtype=np.float32))
    elif hasattr(env_unwrapped, "set_state"):
        env_unwrapped.set_state(np.asarray(init_state, dtype=np.float32))
    elif hasattr(env_unwrapped, "state"):
        env_unwrapped.state = np.asarray(init_state, dtype=np.float32).copy()
    else:
        raise AttributeError("Environment does not expose a supported state-setting method")


def _get_env_state(env) -> np.ndarray:
    env_unwrapped = getattr(env, "unwrapped", env)
    if hasattr(env_unwrapped, "_get_obs"):
        return np.asarray(env_unwrapped._get_obs(), dtype=np.float32)
    if hasattr(env_unwrapped, "get_state"):
        return np.asarray(env_unwrapped.get_state(), dtype=np.float32)
    if hasattr(env_unwrapped, "state"):
        return np.asarray(env_unwrapped.state, dtype=np.float32)
    raise AttributeError("Environment does not expose a supported state getter")


def rollout_final_state(
    env,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Roll out raw actions from a fixed state using the Track A reset pattern."""
    _set_env_state(env, init_state, goal_state, seed=seed)
    for action in np.asarray(raw_actions, dtype=np.float32):
        env.step(np.asarray(action, dtype=np.float32))
    return _get_env_state(env)


def evaluate_candidates_oracle(
    env_factory,
    init_state,
    candidate_actions,
    cost_fn,
    goal_state,
) -> np.ndarray:
    """Evaluate candidates by real-env terminal state cost.

    `candidate_actions` are raw env actions with shape
    `(n_cand, n_steps, action_dim)`. Each candidate is reset to the same
    initial and goal state before rollout, matching the Track A C_real_state
    execution path.
    """
    init_state = np.asarray(init_state, dtype=np.float32)
    goal_state = np.asarray(goal_state, dtype=np.float32)
    candidate_actions = np.asarray(candidate_actions, dtype=np.float32)
    if candidate_actions.ndim != 3:
        raise ValueError("candidate_actions must have shape (n_cand, n_steps, action_dim)")

    env = env_factory()
    try:
        costs = np.empty(candidate_actions.shape[0], dtype=np.float64)
        for idx, raw_actions in enumerate(candidate_actions):
            terminal_state = rollout_final_state(
                env,
                init_state,
                goal_state,
                raw_actions,
                seed=idx,
            )
            costs[idx] = float(cost_fn(terminal_state, goal_state))
    finally:
        if hasattr(env, "close"):
            env.close()
    return costs


def _default_action_transform(
    blocked_actions: np.ndarray,
    *,
    action_block: int,
) -> np.ndarray:
    blocked_actions = np.asarray(blocked_actions, dtype=np.float32)
    if blocked_actions.shape[-1] % action_block != 0:
        raise ValueError("action_dim must be divisible by action_block")
    raw_action_dim = blocked_actions.shape[-1] // action_block
    return blocked_actions.reshape(blocked_actions.shape[0] * action_block, raw_action_dim)


def cem_with_oracle_cost(
    env_factory,
    init_state,
    goal_state,
    cost_fn,
    n_samples: int = 300,
    n_iters: int = 30,
    n_elites: int = 30,
    horizon: int = 5,
    receding_horizon: int = 5,
    action_block: int = 5,
    rng=None,
    *,
    action_dim: int = 2,
    var_scale: float = 1.0,
    action_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Run CEM where candidate scores come from real-env oracle rollouts.

    The search distribution is over blocked actions with shape
    `(horizon, action_dim)`. `action_transform` converts each blocked candidate
    to raw env actions before rollout; when omitted, candidates are reshaped
    using `action_block`.
    """
    if n_elites > n_samples:
        raise ValueError("n_elites must be <= n_samples")
    if n_samples < 1 or n_iters < 1 or n_elites < 1:
        raise ValueError("n_samples, n_iters, and n_elites must be positive")
    if rng is None:
        rng = np.random.default_rng()
    if action_transform is None:
        action_transform = lambda blocked: _default_action_transform(  # noqa: E731
            blocked,
            action_block=action_block,
        )

    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    var = var_scale * np.ones((horizon, action_dim), dtype=np.float32)
    candidates_per_iter = []
    blocked_candidates_per_iter = []
    elite_indices_per_iter = []
    elite_costs_per_iter = []
    best_action_seq = None
    best_cost = math.inf

    for _ in range(n_iters):
        blocked_candidates = rng.normal(
            loc=mean[None, :, :],
            scale=var[None, :, :],
            size=(n_samples, horizon, action_dim),
        ).astype(np.float32)
        blocked_candidates[0] = mean
        raw_candidates = np.stack(
            [action_transform(candidate) for candidate in blocked_candidates],
            axis=0,
        ).astype(np.float32)
        costs = evaluate_candidates_oracle(
            env_factory,
            init_state,
            raw_candidates,
            cost_fn,
            goal_state,
        )
        elite_indices = np.argsort(costs, kind="stable")[:n_elites]
        elite_blocked = blocked_candidates[elite_indices]

        candidates_per_iter.append(raw_candidates)
        blocked_candidates_per_iter.append(blocked_candidates)
        elite_indices_per_iter.append(elite_indices.astype(np.int64))
        elite_costs_per_iter.append(costs[elite_indices].astype(np.float64))

        if float(costs[elite_indices[0]]) < best_cost:
            best_cost = float(costs[elite_indices[0]])
            best_action_seq = raw_candidates[elite_indices[0]].copy()

        mean = elite_blocked.mean(axis=0).astype(np.float32)
        var = elite_blocked.std(axis=0).astype(np.float32)

    return {
        "candidates_per_iter": candidates_per_iter,
        "blocked_candidates_per_iter": blocked_candidates_per_iter,
        "elite_indices_per_iter": elite_indices_per_iter,
        "elite_costs_per_iter": elite_costs_per_iter,
        "best_action_seq": best_action_seq,
        "best_cost": float(best_cost),
        "early_iter_idx": 3,
        "late_iter_idx": n_iters,
        "receding_horizon": int(receding_horizon),
    }
