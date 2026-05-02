"""State-capture rollout helpers for PushT diagnostic visualizations."""

from __future__ import annotations

import numpy as np

from lewm_audit.eval.oracle_cem import (
    _get_env_state,
    _set_env_state,
    block_pose_components,
)


def rollout_with_state_capture(
    env_factory,
    init_state,
    goal_state,
    action_seq,
    *,
    seed: int = 0,
) -> dict:
    """Roll out raw actions while recording every simulator state.

    The reset and state-setting path intentionally reuses the oracle-CEM
    helpers, matching the Track A and oracle ablation execution semantics.
    """
    init_state = np.asarray(init_state, dtype=np.float32)
    goal_state = np.asarray(goal_state, dtype=np.float32)
    action_seq = np.asarray(action_seq, dtype=np.float32)
    if action_seq.ndim != 2:
        raise ValueError("action_seq must have shape (n_steps, action_dim)")

    env = env_factory()
    try:
        _set_env_state(env, init_state, goal_state, seed=seed)
        states = [_get_env_state(env).astype(np.float32)]
        for action in action_seq:
            env.step(np.asarray(action, dtype=np.float32))
            states.append(_get_env_state(env).astype(np.float32))
    finally:
        if hasattr(env, "close"):
            env.close()

    states_arr = np.asarray(states, dtype=np.float32)
    metrics = [block_pose_components(state, goal_state) for state in states_arr]
    step_success = np.asarray([metric["success"] for metric in metrics], dtype=bool)
    return {
        "states": states_arr,
        "block_xy": states_arr[:, 2:4].astype(np.float32),
        "agent_xy": states_arr[:, 0:2].astype(np.float32),
        "block_angle": states_arr[:, 4].astype(np.float32),
        "step_success": step_success,
        "final_success": bool(step_success[-1]),
    }
