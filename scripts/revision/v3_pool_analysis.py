#!/usr/bin/env python3
"""V3 same-pool ranker decomposition for PushT final CEM pools.

V3 scores each saved final-pool action by replaying the action sequence in the
PushT simulator, encoding the actual terminal observation, and measuring
Euclidean distance to the encoded goal observation.  This removes predictor
rollout error while keeping the learned encoder and latent cost shape.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pusht_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from lewm_audit.diagnostics.three_cost import (  # noqa: E402
    block_pose_metrics,
    execute_raw_actions,
)
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    CEM_ITERS,
    DEFAULT_PAIRS_PATH,
    IMG_SIZE,
    NUM_SAMPLES,
    PLANNING_HORIZON,
    RECEDING_HORIZON,
    TOPK,
    VAR_SCALE,
    load_pairs,
    make_policy_namespace,
    validate_requested_pair_offsets,
)


LATENT_DIM = 192
N_EXPECTED_PAIRS = 100
N_CANDIDATES = 300
SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)

DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
DEFAULT_RERANK_PATH = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "v3_pool_analysis_pusht.json"
DEFAULT_MEMO = PROJECT_ROOT / "docs" / "revision" / "v3_pool_memo.md"


def parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--rerank-path", type=Path, default=DEFAULT_RERANK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memo", type=Path, default=DEFAULT_MEMO)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", type=parse_int_list, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute pairs already present in the output.")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def tensor_to_numpy(pool: dict[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    value = pool[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    rho = spearmanr(x, y).statistic
    return clean_float(rho)


def effective_rho(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def scalar_summary(
    values: list[float | int | bool | None],
    *,
    none_as_zero: bool = False,
) -> dict[str, Any]:
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
        "n_missing": int(0 if none_as_zero else n_missing),
        "none_as_zero": bool(none_as_zero),
        "ddof": 1,
    }


def load_subset_membership(path: Path) -> tuple[dict[str, list[int]], dict[int, list[str]], dict[str, Any]]:
    data = load_json(path)
    anchors = data.get("metadata", {}).get("anchor_definitions")
    if not isinstance(anchors, dict):
        raise ValueError(f"{path} is missing metadata.anchor_definitions")

    subset_ids: dict[str, list[int]] = {}
    pair_to_subsets: dict[int, list[str]] = {pair_id: [] for pair_id in range(N_EXPECTED_PAIRS)}
    for subset in SUBSET_ORDER:
        entry = anchors.get(subset)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing anchor definition for subset {subset!r}")
        pair_ids = [int(pair_id) for pair_id in entry.get("pair_ids", [])]
        subset_ids[subset] = pair_ids
        for pair_id in pair_ids:
            pair_to_subsets.setdefault(int(pair_id), []).append(subset)

    missing = [pair_id for pair_id in range(N_EXPECTED_PAIRS) if not pair_to_subsets.get(pair_id)]
    if missing:
        raise ValueError(f"Subset anchor definitions leave pair IDs unclassified: {missing}")
    return subset_ids, pair_to_subsets, anchors


def pool_path(pool_dir: Path, pair_id: int) -> Path:
    return pool_dir / f"pair_{int(pair_id)}.pt"


def validate_pool(pool: dict[str, Any], *, pair: dict[str, Any]) -> None:
    pair_id = int(pair["pair_id"])
    metadata = pool.get("metadata", {})
    if metadata.get("format") != "pusht_rerank_only_pool_v1":
        raise ValueError(f"pair_{pair_id}.pt has unexpected format {metadata.get('format')!r}")
    for key in ("pair_id", "start_row", "goal_row"):
        if int(metadata.get(key, -1)) != int(pair[key]):
            raise ValueError(f"pair_{pair_id}.pt metadata mismatch for {key}")

    expected_shapes = {
        "z_pred": (N_CANDIDATES, LATENT_DIM),
        "z_goal": (LATENT_DIM,),
        "raw_actions": (N_CANDIDATES, PLANNING_HORIZON * ACTION_BLOCK, 2),
        "default_costs": (N_CANDIDATES,),
        "v1_hinge_costs": (N_CANDIDATES,),
        "c_real_state": (N_CANDIDATES,),
        "success": (N_CANDIDATES,),
    }
    for key, shape in expected_shapes.items():
        value = pool.get(key)
        if value is None:
            raise ValueError(f"pair_{pair_id}.pt missing key {key!r}")
        observed = tuple(value.shape) if torch.is_tensor(value) else tuple(np.asarray(value).shape)
        if observed != shape:
            raise ValueError(f"pair_{pair_id}.pt key {key!r} has shape {observed}, expected {shape}")


def load_pair_rows_direct(dataset, pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = dataset.get_row_data([int(pair["start_row"]), int(pair["goal_row"])])
    return {key: value[0] for key, value in rows.items()}, {key: value[1] for key, value in rows.items()}


def batch_encode_pixels(policy, model, pixels: np.ndarray, *, batch_size: int) -> np.ndarray:
    pixels = np.asarray(pixels)
    if pixels.ndim != 4:
        raise ValueError(f"Expected pixel batch with shape (B,H,W,C), got {pixels.shape}")
    device = next(model.parameters()).device
    latents: list[np.ndarray] = []
    for start in range(0, int(pixels.shape[0]), int(batch_size)):
        batch = pixels[start : start + int(batch_size)]
        info = policy._prepare_info({"pixels": batch[:, None, ...]})
        pixels_t = info["pixels"].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            encoded = model.encode({"pixels": pixels_t})
        latents.append(encoded["emb"][:, -1, :].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def replay_terminal_pixels(
    *,
    env,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    seeds: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    terminal_pixels: list[np.ndarray] = []
    replay_metrics: list[dict[str, Any]] = []
    for candidate_index, actions in enumerate(raw_actions):
        rollout = execute_raw_actions(
            env,
            initial_state=initial_state,
            goal_state=goal_state,
            raw_actions=np.asarray(actions, dtype=np.float32),
            seed=int(seeds[candidate_index]),
        )
        pose = block_pose_metrics(rollout["terminal_state"], goal_state)
        terminal_pixels.append(np.asarray(rollout["terminal_pixels"], dtype=np.uint8))
        replay_metrics.append(
            {
                "candidate_index": int(candidate_index),
                "seed": int(seeds[candidate_index]),
                "c_real_state": clean_float(float(pose["c_real_state"])),
                "block_pos_dist": clean_float(float(pose["block_pos_dist"])),
                "angle_dist": clean_float(float(pose["angle_dist"])),
                "success": bool(pose["success"]),
                "env_success": bool(rollout["env_success"]),
            }
        )
    return np.stack(terminal_pixels, axis=0), replay_metrics


def candidate_seeds(pool: dict[str, Any], *, pair_id: int, seed: int) -> np.ndarray:
    metrics = pool.get("candidate_metrics")
    if isinstance(metrics, list) and len(metrics) == N_CANDIDATES:
        return np.asarray([int(item["seed"]) for item in metrics], dtype=np.int64)
    return int(seed) + int(pair_id) * 100_000 + np.arange(N_CANDIDATES, dtype=np.int64)


def analyze_pair(
    *,
    pair: dict[str, Any],
    subsets: list[str],
    pool_file: Path,
    pool: dict[str, Any],
    dataset,
    policy,
    model,
    env,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    pair_id = int(pair["pair_id"])
    started = time.time()
    validate_pool(pool, pair=pair)
    initial, goal = load_pair_rows_direct(dataset, pair)

    default_costs = tensor_to_numpy(pool, "default_costs", dtype=np.float64)
    v1_costs = tensor_to_numpy(pool, "v1_hinge_costs", dtype=np.float64)
    c_real_state = tensor_to_numpy(pool, "c_real_state", dtype=np.float64)
    success = tensor_to_numpy(pool, "success", dtype=np.bool_)
    raw_actions = tensor_to_numpy(pool, "raw_actions", dtype=np.float32)
    stored_z_goal = tensor_to_numpy(pool, "z_goal", dtype=np.float32)

    seeds = candidate_seeds(pool, pair_id=pair_id, seed=seed)
    terminal_pixels, replay_metrics = replay_terminal_pixels(
        env=env,
        initial_state=np.asarray(initial["state"], dtype=np.float32),
        goal_state=np.asarray(goal["state"], dtype=np.float32),
        raw_actions=raw_actions,
        seeds=seeds,
    )
    z_terminal = batch_encode_pixels(policy, model, terminal_pixels, batch_size=batch_size)
    z_goal_real = batch_encode_pixels(
        policy,
        model,
        np.asarray(goal["pixels"], dtype=np.uint8)[None, ...],
        batch_size=1,
    )[0]

    v3_costs = np.sum((z_terminal - z_goal_real[None, :]) ** 2, axis=1).astype(np.float64)
    v3_stored_goal_costs = np.sum((z_terminal - stored_z_goal[None, :]) ** 2, axis=1).astype(np.float64)
    replay_c_real = np.asarray([float(item["c_real_state"]) for item in replay_metrics], dtype=np.float64)
    replay_success = np.asarray([bool(item["success"]) for item in replay_metrics], dtype=bool)

    rank1_model = int(np.argmin(default_costs))
    rank1_v3 = int(np.argmin(v3_costs))
    rank1_v1 = int(np.argmin(v1_costs))
    oracle_best = int(np.argmin(c_real_state))

    rpool_cmodel = spearman_corr(default_costs, c_real_state)
    rpool_v3 = spearman_corr(v3_costs, c_real_state)
    rpool_v3_stored_goal = spearman_corr(v3_stored_goal_costs, c_real_state)
    rpool_v1 = spearman_corr(v1_costs, c_real_state)

    return {
        "pair_id": pair_id,
        "cell": str(pair["cell"]),
        "episode_id": int(pair["episode_id"]),
        "start_row": int(pair["start_row"]),
        "goal_row": int(pair["goal_row"]),
        "subsets": subsets,
        "pool_path": str(pool_file),
        "Rpool_Cmodel": rpool_cmodel,
        "Rpool_Cmodel_effective": effective_rho(rpool_cmodel),
        "Rpool_V3": rpool_v3,
        "Rpool_V3_effective": effective_rho(rpool_v3),
        "Rpool_V3_stored_goal": rpool_v3_stored_goal,
        "Rpool_V3_stored_goal_effective": effective_rho(rpool_v3_stored_goal),
        "Rpool_V1": rpool_v1,
        "Rpool_V1_effective": effective_rho(rpool_v1),
        "z_goal_stored_vs_real_l2": clean_float(float(np.sum((stored_z_goal - z_goal_real) ** 2))),
        "z_goal_stored_vs_real_max_abs": clean_float(float(np.max(np.abs(stored_z_goal - z_goal_real)))),
        "replay_c_real_max_abs_diff": clean_float(float(np.max(np.abs(replay_c_real - c_real_state)))),
        "replay_c_real_mean_abs_diff": clean_float(float(np.mean(np.abs(replay_c_real - c_real_state)))),
        "replay_success_disagreement_rate": clean_float(float(np.mean(replay_success != success))),
        "rank1_index_Cmodel": rank1_model,
        "rank1_index_V3": rank1_v3,
        "rank1_index_V1": rank1_v1,
        "oracle_best_index": oracle_best,
        "rank1_success_Cmodel": bool(success[rank1_model]),
        "rank1_success_V3": bool(success[rank1_v3]),
        "rank1_success_V1": bool(success[rank1_v1]),
        "rank1_c_real_Cmodel": clean_float(float(c_real_state[rank1_model])),
        "rank1_c_real_V3": clean_float(float(c_real_state[rank1_v3])),
        "rank1_c_real_V1": clean_float(float(c_real_state[rank1_v1])),
        "oracle_best_c_real": clean_float(float(c_real_state[oracle_best])),
        "selection_regret_Cmodel": clean_float(float(c_real_state[rank1_model] - c_real_state[oracle_best])),
        "selection_regret_V3": clean_float(float(c_real_state[rank1_v3] - c_real_state[oracle_best])),
        "selection_regret_V1": clean_float(float(c_real_state[rank1_v1] - c_real_state[oracle_best])),
        "pool_success_mass": clean_float(float(np.mean(success))),
        "pool_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
        "pool_Cmodel_std": clean_float(float(np.std(default_costs, ddof=0))),
        "pool_V3_cost_std": clean_float(float(np.std(v3_costs, ddof=0))),
        "pool_V1_cost_std": clean_float(float(np.std(v1_costs, ddof=0))),
        "C_V3_costs": v3_costs.astype(np.float32),
        "C_V3_stored_goal_costs": v3_stored_goal_costs.astype(np.float32),
        "wallclock_seconds": clean_float(time.time() - started),
    }


def aggregate_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": int(len(records)),
        "Rpool_Cmodel": scalar_summary([record.get("Rpool_Cmodel") for record in records]),
        "Rpool_Cmodel_effective": scalar_summary(
            [record.get("Rpool_Cmodel") for record in records], none_as_zero=True
        ),
        "Rpool_V3": scalar_summary([record.get("Rpool_V3") for record in records]),
        "Rpool_V3_effective": scalar_summary(
            [record.get("Rpool_V3") for record in records], none_as_zero=True
        ),
        "Rpool_V3_stored_goal": scalar_summary([record.get("Rpool_V3_stored_goal") for record in records]),
        "Rpool_V3_stored_goal_effective": scalar_summary(
            [record.get("Rpool_V3_stored_goal") for record in records], none_as_zero=True
        ),
        "Rpool_V1": scalar_summary([record.get("Rpool_V1") for record in records]),
        "Rpool_V1_effective": scalar_summary(
            [record.get("Rpool_V1") for record in records], none_as_zero=True
        ),
        "pool_success_mass": scalar_summary([record.get("pool_success_mass") for record in records]),
        "pool_Creal_std": scalar_summary([record.get("pool_Creal_std") for record in records]),
        "pool_Cmodel_std": scalar_summary([record.get("pool_Cmodel_std") for record in records]),
        "pool_V3_cost_std": scalar_summary([record.get("pool_V3_cost_std") for record in records]),
        "selection_regret_Cmodel": scalar_summary([record.get("selection_regret_Cmodel") for record in records]),
        "selection_regret_V3": scalar_summary([record.get("selection_regret_V3") for record in records]),
        "selection_regret_V1": scalar_summary([record.get("selection_regret_V1") for record in records]),
        "rank1_success_Cmodel": scalar_summary([record.get("rank1_success_Cmodel") for record in records]),
        "rank1_success_V3": scalar_summary([record.get("rank1_success_V3") for record in records]),
        "rank1_success_V1": scalar_summary([record.get("rank1_success_V1") for record in records]),
        "z_goal_stored_vs_real_l2": scalar_summary([record.get("z_goal_stored_vs_real_l2") for record in records]),
        "replay_c_real_max_abs_diff": scalar_summary([record.get("replay_c_real_max_abs_diff") for record in records]),
        "replay_success_disagreement_rate": scalar_summary(
            [record.get("replay_success_disagreement_rate") for record in records]
        ),
    }


def aggregate_by_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        subset: aggregate_pairs([record for record in records if subset in record.get("subsets", [])])
        for subset in SUBSET_ORDER
    }


def make_decomposition_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    overall = summary["overall"]
    return [
        {
            "cost_ranker": "C_model (predicted latent)",
            "key": "Rpool_Cmodel",
            "mean": overall["Rpool_Cmodel"]["mean"],
            "std": overall["Rpool_Cmodel"]["std"],
            "what_it_tests": "End-to-end: predictor + encoder + latent cost shape",
        },
        {
            "cost_ranker": "C_V3 (actual terminal latent)",
            "key": "Rpool_V3",
            "mean": overall["Rpool_V3"]["mean"],
            "std": overall["Rpool_V3"]["std"],
            "what_it_tests": "Encoder + latent cost shape after removing predictor rollout",
        },
        {
            "cost_ranker": "C_V1 (physical hinge)",
            "key": "Rpool_V1_effective",
            "mean": overall["Rpool_V1_effective"]["mean"],
            "std": overall["Rpool_V1_effective"]["std"],
            "what_it_tests": "Physical ground truth ranker on the same pool",
        },
    ]


def interpretation(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary["overall"]
    c_model = overall["Rpool_Cmodel"]["mean"]
    v3 = overall["Rpool_V3"]["mean"]
    v1 = overall["Rpool_V1_effective"]["mean"]
    if c_model is None or v3 is None or v1 is None:
        verdict = "incomplete"
        rationale = "At least one ranker lacks finite Rpool values."
    elif v3 > c_model + 0.10 and v3 < v1 - 0.10:
        verdict = "mixed_predictor_plus_encoder_geometry"
        rationale = (
            "V3 improves over predicted-latent C_model but remains well below V1, "
            "so predictor rollout contributes while learned terminal-latent geometry remains limiting."
        )
    elif abs(v3 - c_model) <= 0.10 and v1 > v3 + 0.20:
        verdict = "encoder_latent_geometry_bottleneck"
        rationale = (
            "V3 is close to C_model and both are far below V1, so removing predictor rollout "
            "does not restore useful same-pool ranking."
        )
    elif abs(v3 - v1) <= 0.10 and v3 > c_model + 0.10:
        verdict = "predictor_rollout_bottleneck"
        rationale = (
            "V3 nearly matches V1 and both exceed C_model, so the encoder geometry works "
            "when actual terminal observations replace predicted latents."
        )
    else:
        verdict = "ambiguous"
        rationale = "The three rankers do not match a single clean attribution pattern."
    return {
        "verdict": verdict,
        "rationale": rationale,
        "thresholds": {
            "meaningful_gap": 0.10,
            "large_v1_gap": 0.20,
        },
    }


def build_payload(
    *,
    args: argparse.Namespace,
    pairs_path: Path,
    rerank_path: Path,
    records: list[dict[str, Any]],
    anchor_definitions: dict[str, Any],
    started_at: str,
    completed: bool,
) -> dict[str, Any]:
    summary = {
        "overall": aggregate_pairs(records),
        "by_subset": aggregate_by_subset(records),
    }
    summary["decomposition_table"] = make_decomposition_table(summary)
    summary["interpretation"] = interpretation(summary)
    return {
        "metadata": {
            "format": "v3_pool_analysis_pusht",
            "created_at": started_at,
            "updated_at": iso_now(),
            "git_commit": get_git_commit(),
            "script_path": str(Path(__file__).resolve()),
            "pool_dir": str(args.pool_dir),
            "pairs_path": str(pairs_path),
            "rerank_path": str(rerank_path),
            "output": str(args.output),
            "memo": str(args.memo),
            "checkpoint_dir": str(args.checkpoint_dir),
            "device": str(args.device),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "n_pairs_completed": int(len(records)),
            "n_candidates_per_pair": int(N_CANDIDATES),
            "completed": bool(completed),
            "v3_definition": "C_V3 = ||encoder(actual terminal observation) - encoder(goal observation)||_2^2",
            "spearman": "scipy.stats.spearmanr(cost_ranker, c_real_state) within each final pool",
            "anchor_definitions": anchor_definitions,
        },
        "per_pair": sorted(records, key=lambda item: int(item["pair_id"])),
        "summary": summary,
    }


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def fmt_pm(summary: dict[str, Any], key: str) -> str:
    stats = summary[key]
    return f"{fmt(stats['mean'])} ± {fmt(stats['std'])}"


def write_memo(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = summary["decomposition_table"]
    lines = [
        "# V3 Same-Pool Ranker Decomposition (PushT)",
        "",
        f"Generated: `{payload['metadata']['updated_at']}`",
        f"Pairs completed: {payload['metadata']['n_pairs_completed']}",
        "",
        "## Decomposition Table",
        "",
        "| Cost ranker | Rpool (mean ± std) | What it tests |",
        "|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cost_ranker']} | {fmt(row['mean'])} ± {fmt(row['std'])} | {row['what_it_tests']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Subset Breakdown",
            "",
            "| Subset | n | C_model | C_V3 | C_V1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for subset in SUBSET_ORDER:
        stats = summary["by_subset"][subset]
        lines.append(
            f"| {subset} | {stats['n_pairs']} | "
            f"{fmt_pm(stats, 'Rpool_Cmodel')} | {fmt_pm(stats, 'Rpool_V3')} | "
            f"{fmt_pm(stats, 'Rpool_V1_effective')} |"
        )

    interp = summary["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Verdict: `{interp['verdict']}`",
            "",
            interp["rationale"],
            "",
            "## Replay Sanity Checks",
            "",
            f"- Stored-vs-real goal latent L2: {fmt_pm(summary['overall'], 'z_goal_stored_vs_real_l2')}",
            f"- Max replay C_real absolute difference: {fmt_pm(summary['overall'], 'replay_c_real_max_abs_diff')}",
            f"- Replay success disagreement rate: {fmt_pm(summary['overall'], 'replay_success_disagreement_rate')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_records(path: Path) -> tuple[str, dict[int, dict[str, Any]]]:
    if not path.exists():
        return iso_now(), {}
    data = load_json(path)
    started_at = str(data.get("metadata", {}).get("created_at") or iso_now())
    records = {
        int(record["pair_id"]): record
        for record in data.get("per_pair", [])
        if record.get("Rpool_V3") is not None
    }
    return started_at, records


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    args.pool_dir = args.pool_dir.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.rerank_path = args.rerank_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.memo = args.memo.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")

    pairs_data, all_pairs = load_pairs(args.pairs_path, max_pairs=None, pair_ids=args.pair_ids)
    all_pairs = sorted(all_pairs, key=lambda pair: int(pair["pair_id"]))
    if args.max_pairs is not None:
        all_pairs = all_pairs[: int(args.max_pairs)]
    offset = int(pairs_data["metadata"]["offset"])
    validate_requested_pair_offsets(all_pairs, offset=offset)

    _, pair_to_subsets, anchor_definitions = load_subset_membership(args.rerank_path)
    started_at, existing_by_pair = load_existing_records(args.output)
    if args.force:
        existing_by_pair = {}

    dataset_path = Path(pairs_data["metadata"]["dataset_path"])
    dataset = get_dataset(dataset_path.parent, dataset_path.stem)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=int(args.seed),
        ),
        process,
    )
    model = policy.solver.model
    model.eval()

    print("== PushT V3 pool analysis setup ==")
    print(f"pairs: {len(all_pairs)}")
    print(f"already_complete: {len(existing_by_pair)}")
    print(f"pool_dir: {args.pool_dir}")
    print(f"output: {args.output}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"dataset_name: {dataset_path.stem}")
    print(f"device: {args.device}")
    print(f"batch_size: {args.batch_size}")
    print(f"expected_new_rollouts: {(len(all_pairs) - len(set(existing_by_pair) & {int(p['pair_id']) for p in all_pairs})) * N_CANDIDATES}")

    records_by_pair = dict(existing_by_pair)
    total_started = time.time()
    env = gym.make("swm/PushT-v1")
    try:
        for pair_idx, pair in enumerate(all_pairs, start=1):
            pair_id = int(pair["pair_id"])
            if pair_id in records_by_pair and not args.force:
                print(f"[{pair_idx}/{len(all_pairs)}] pair_id={pair_id}: loaded existing V3 result")
                continue

            pair_started = time.time()
            print(f"[{pair_idx}/{len(all_pairs)}] pair_id={pair_id} cell={pair['cell']}: replay 300 + encode terminals")
            path = pool_path(args.pool_dir, pair_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing pool file: {path}")
            pool = torch.load(path, map_location="cpu", weights_only=False)
            record = analyze_pair(
                pair=pair,
                subsets=pair_to_subsets[int(pair_id)],
                pool_file=path,
                pool=pool,
                dataset=dataset,
                policy=policy,
                model=model,
                env=env,
                seed=int(args.seed),
                batch_size=int(args.batch_size),
            )
            records_by_pair[pair_id] = record

            completed = len(records_by_pair) >= len(all_pairs)
            payload = build_payload(
                args=args,
                pairs_path=args.pairs_path,
                rerank_path=args.rerank_path,
                records=list(records_by_pair.values()),
                anchor_definitions=anchor_definitions,
                started_at=started_at,
                completed=completed,
            )
            write_json(args.output, payload)
            write_memo(args.memo, payload)

            print(
                f"  Rpool: C_model={fmt(record['Rpool_Cmodel'])} "
                f"V3={fmt(record['Rpool_V3'])} V1={fmt(record['Rpool_V1'])}; "
                f"elapsed={time.time() - pair_started:.1f}s"
            )
            if pair_idx % 10 == 0 or pair_idx == len(all_pairs):
                overall = payload["summary"]["overall"]
                print(
                    f"  progress {len(records_by_pair)}/{len(all_pairs)}: "
                    f"mean C_model={fmt(overall['Rpool_Cmodel']['mean'])}, "
                    f"V3={fmt(overall['Rpool_V3']['mean'])}, "
                    f"V1={fmt(overall['Rpool_V1']['mean'])}"
                )
    finally:
        env.close()

    final_payload = build_payload(
        args=args,
        pairs_path=args.pairs_path,
        rerank_path=args.rerank_path,
        records=list(records_by_pair.values()),
        anchor_definitions=anchor_definitions,
        started_at=started_at,
        completed=len(records_by_pair) >= len(all_pairs),
    )
    final_payload["metadata"]["wallclock_seconds"] = clean_float(time.time() - total_started)
    write_json(args.output, final_payload)
    write_memo(args.memo, final_payload)

    print("== V3 decomposition summary ==")
    for row in final_payload["summary"]["decomposition_table"]:
        print(f"{row['cost_ranker']}: {fmt(row['mean'])} ± {fmt(row['std'])}")
    print(final_payload["summary"]["interpretation"]["rationale"])
    print(f"saved: {args.output}")
    print(f"memo: {args.memo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
