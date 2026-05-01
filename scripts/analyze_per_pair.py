#!/usr/bin/env python3
"""Per-pair failure characterization for LeWM three-cost attribution results."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch

from analyze_three_cost import (
    fmt_float,
    fmt_mean_std,
    fmt_pct,
    pearson_corr,
    spearman_corr,
    summarize_values,
)
from eval_pusht_baseline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    PROJECT_ROOT,
    build_policy,
    build_processors,
    get_dataset,
    resolve_device,
)
from three_cost_eval import (
    SOURCE_CEM_LATE,
    configure_goal_render,
    encode_pixels,
    expand_info_for_candidates,
    generate_cem_action_sequences,
    load_pair_rows,
    make_policy_args,
    prepare_pair_info,
    tensor_clone_info,
)


DEFAULT_INPUT = PROJECT_ROOT / "results" / "three_cost_offset50.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "per_pair_analysis.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
STATE_FORMAT = {
    "shape": [7],
    "fields": [
        "agent_x",
        "agent_y",
        "block_x",
        "block_y",
        "block_angle",
        "agent_vx",
        "agent_vy",
    ],
    "source": "stable_worldmodel.envs.pusht.PushT._get_obs",
    "note": (
        "The 7-D PushT state does not include goal_x/goal_y. A goal is represented "
        "by a separate future state row, and the goal block pose is goal_state[2:5]."
    ),
}
CATEGORY_ORDER = ("Easy", "Hard", "Impossible")
CATEGORY_COLORS = {
    "Easy": "#2f6f4e",
    "Hard": "#d08c2c",
    "Impossible": "#b23a48",
}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize pair-dependent three-cost failures for PushT."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", DEFAULT_CACHE_DIR)),
    )
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--var-scale", type=float, default=None)
    parser.add_argument("--cem-early-iters", type=int, default=3)
    parser.add_argument("--cem-late-iters", type=int, default=None)
    parser.add_argument(
        "--num-per-source",
        type=int,
        default=None,
        help="Number of CEM_late candidates to regenerate for representative rollouts.",
    )
    parser.add_argument(
        "--representatives-per-category",
        type=int,
        default=2,
        help="Number of Easy and Impossible pairs used for per-step analysis.",
    )
    parser.add_argument(
        "--skip-step-analysis",
        action="store_true",
        help="Skip model loading, CEM regeneration, latent plot, and per-step rollout.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args()


def angular_distance(a: float, b: float) -> float:
    diff = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
    return abs(diff)


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def categorize_pair(success_count: int, total_actions: int) -> str:
    if success_count == 0:
        return "Impossible"
    if success_count > 25:
        return "Easy"
    return "Hard"


def load_results(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not data.get("pairs"):
        raise ValueError(f"No pairs found in {path}")
    return data


def pair_action_corr(pair: dict) -> dict:
    c_real_z = np.asarray([action["c_real_z"] for action in pair["actions"]], dtype=np.float64)
    c_real_state = np.asarray(
        [action["c_real_state"] for action in pair["actions"]], dtype=np.float64
    )
    return {
        "pearson_c_real_z_vs_c_real_state": pearson_corr(c_real_z, c_real_state),
        "spearman_c_real_z_vs_c_real_state": spearman_corr(c_real_z, c_real_state),
    }


def get_best_cem_late_action(pair: dict) -> dict | None:
    cem_late = [action for action in pair["actions"] if action["source"] == SOURCE_CEM_LATE]
    if not cem_late:
        return None
    # This is the planner-best action, not the oracle-best action: CEM optimizes C_model.
    return min(cem_late, key=lambda action: action["c_model"])


def extract_physical_features(initial_state: np.ndarray, goal_state: np.ndarray) -> dict:
    block_displacement = l2(initial_state[2:4], goal_state[2:4])
    rotation_required = angular_distance(initial_state[4], goal_state[4])
    agent_to_block_start = l2(initial_state[:2], initial_state[2:4])
    agent_to_goal_block_start = l2(initial_state[:2], goal_state[2:4])
    agent_displacement = l2(initial_state[:2], goal_state[:2])
    physical_pose_distance = block_displacement + rotation_required
    return {
        "block_displacement": block_displacement,
        "rotation_required": rotation_required,
        "agent_to_block_start": agent_to_block_start,
        "agent_to_goal_block_start": agent_to_goal_block_start,
        "agent_displacement": agent_displacement,
        "physical_pose_distance": physical_pose_distance,
    }


def build_pair_records(data: dict, dataset) -> list[dict]:
    offset = int(data["offset"])
    pair_records = []
    for pair_index, pair in enumerate(data["pairs"]):
        pair_rows = load_pair_rows(dataset, int(pair["dataset_row"]), offset)
        initial_state = np.asarray(pair_rows["initial"]["state"], dtype=np.float64)
        goal_state = np.asarray(pair_rows["goal"]["state"], dtype=np.float64)
        success_count = int(sum(action["success"] for action in pair["actions"]))
        total_actions = len(pair["actions"])
        env_success_count = int(sum(action["env_success"] for action in pair["actions"]))
        category = categorize_pair(success_count, total_actions)
        corr = pair_action_corr(pair)
        best_cem_late = get_best_cem_late_action(pair)

        pair_records.append(
            {
                "pair_index": pair_index,
                "episode_id": int(pair["episode_id"]),
                "start_step": int(pair["start_step"]),
                "goal_step": int(pair["goal_step"]),
                "dataset_row": int(pair["dataset_row"]),
                "category": category,
                "success_count": success_count,
                "total_actions": total_actions,
                "success_rate": float(success_count / total_actions),
                "env_success_count": env_success_count,
                "env_success_rate": float(env_success_count / total_actions),
                "correlations": corr,
                "physical": extract_physical_features(initial_state, goal_state),
                "initial_state": initial_state.tolist(),
                "goal_state": goal_state.tolist(),
                "best_cem_late_action": simplify_action_record(best_cem_late),
                "latent": {},
                "per_step": None,
            }
        )
    return pair_records


def simplify_action_record(action: dict | None) -> dict | None:
    if action is None:
        return None
    keys = (
        "source",
        "source_index",
        "c_model",
        "c_real_z",
        "c_real_state",
        "block_pos_dist",
        "angle_dist",
        "success",
        "env_success",
    )
    return {key: action[key] for key in keys if key in action}


def group_by_category(pair_records: list[dict]) -> dict[str, list[dict]]:
    grouped = {category: [] for category in CATEGORY_ORDER}
    for record in pair_records:
        grouped.setdefault(record["category"], []).append(record)
    return grouped


def category_summary(pair_records: list[dict]) -> dict:
    grouped = group_by_category(pair_records)
    summary = {}
    for category in CATEGORY_ORDER:
        records = grouped.get(category, [])
        summary[category] = {
            "count": len(records),
            "success_count": summarize_values(record["success_count"] for record in records),
            "success_rate": summarize_values(record["success_rate"] for record in records),
            "spearman_c_real_z_vs_c_real_state": summarize_values(
                record["correlations"]["spearman_c_real_z_vs_c_real_state"]
                for record in records
                if record["correlations"]["spearman_c_real_z_vs_c_real_state"] is not None
            ),
            "pearson_c_real_z_vs_c_real_state": summarize_values(
                record["correlations"]["pearson_c_real_z_vs_c_real_state"]
                for record in records
                if record["correlations"]["pearson_c_real_z_vs_c_real_state"] is not None
            ),
            "block_displacement": summarize_values(
                record["physical"]["block_displacement"] for record in records
            ),
            "rotation_required": summarize_values(
                record["physical"]["rotation_required"] for record in records
            ),
            "agent_to_block_start": summarize_values(
                record["physical"]["agent_to_block_start"] for record in records
            ),
            "physical_pose_distance": summarize_values(
                record["physical"]["physical_pose_distance"] for record in records
            ),
            "latent_distance": summarize_values(
                record["latent"].get("latent_distance")
                for record in records
                if record["latent"].get("latent_distance") is not None
            ),
        }
    return summary


def physical_feature_correlations(pair_records: list[dict]) -> dict:
    features = (
        "block_displacement",
        "rotation_required",
        "agent_to_block_start",
        "agent_to_goal_block_start",
        "agent_displacement",
        "physical_pose_distance",
    )
    success_count = np.asarray([record["success_count"] for record in pair_records], dtype=np.float64)
    success_rate = np.asarray([record["success_rate"] for record in pair_records], dtype=np.float64)
    impossible = np.asarray(
        [1.0 if record["category"] == "Impossible" else 0.0 for record in pair_records],
        dtype=np.float64,
    )
    out = {}
    for feature in features:
        values = np.asarray([record["physical"][feature] for record in pair_records], dtype=np.float64)
        out[feature] = {
            "vs_success_count": {
                "pearson": pearson_corr(values, success_count),
                "spearman": spearman_corr(values, success_count),
            },
            "vs_success_rate": {
                "pearson": pearson_corr(values, success_rate),
                "spearman": spearman_corr(values, success_rate),
            },
            "vs_impossible_indicator": {
                "pearson": pearson_corr(values, impossible),
                "spearman": spearman_corr(values, impossible),
            },
        }
    return out


def make_analysis_args(args: argparse.Namespace, data: dict, horizon_blocks: int) -> argparse.Namespace:
    metadata = data.get("metadata", {})
    cem = metadata.get("cem", {})
    seed = args.seed if args.seed is not None else int(metadata.get("seed", 42))
    num_samples = args.num_samples if args.num_samples is not None else int(cem.get("num_samples", 300))
    topk = args.topk if args.topk is not None else int(cem.get("topk", 30))
    var_scale = args.var_scale if args.var_scale is not None else float(cem.get("var_scale", 1.0))
    cem_late_iters = (
        args.cem_late_iters
        if args.cem_late_iters is not None
        else int(cem.get("late_iters", 30))
    )
    num_per_source = (
        args.num_per_source
        if args.num_per_source is not None
        else int(metadata.get("num_per_source", 10))
    )
    return argparse.Namespace(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        num_samples=num_samples,
        var_scale=var_scale,
        cem_iters=cem_late_iters,
        cem_early_iters=args.cem_early_iters,
        cem_late_iters=cem_late_iters,
        topk=topk,
        seed=seed,
        horizon=horizon_blocks,
        receding_horizon=horizon_blocks,
        action_block=args.action_block,
        img_size=args.img_size,
        num_per_source=num_per_source,
    )


def compute_latent_distances(pair_records: list[dict], *, dataset, policy, model, offset: int) -> None:
    for record in pair_records:
        rows = load_pair_rows(dataset, record["dataset_row"], offset)
        z_init = encode_pixels(policy, model, rows["initial"]["pixels"])
        z_goal = encode_pixels(policy, model, rows["goal"]["pixels"])
        latent_distance = float(torch.linalg.norm(z_init - z_goal).detach().cpu().item())
        latent_sq_distance = float(torch.sum((z_init - z_goal) ** 2).detach().cpu().item())
        record["latent"] = {
            "latent_distance": latent_distance,
            "latent_squared_distance": latent_sq_distance,
            "physical_pose_distance": record["physical"]["physical_pose_distance"],
            "block_displacement": record["physical"]["block_displacement"],
            "rotation_required": record["physical"]["rotation_required"],
        }


def select_representatives(pair_records: list[dict], per_category: int) -> list[dict]:
    easy = [
        record
        for record in pair_records
        if record["category"] == "Easy" and record["best_cem_late_action"] is not None
    ]
    impossible = [
        record
        for record in pair_records
        if record["category"] == "Impossible" and record["best_cem_late_action"] is not None
    ]
    easy = sorted(easy, key=lambda record: (-record["success_count"], record["pair_index"]))
    impossible = sorted(
        impossible,
        key=lambda record: (
            record["correlations"]["spearman_c_real_z_vs_c_real_state"]
            if record["correlations"]["spearman_c_real_z_vs_c_real_state"] is not None
            else 999.0,
            record["pair_index"],
        ),
    )
    return easy[:per_category] + impossible[:per_category]


def move_tensors_to_device(info: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            if device.type == "mps" and value.is_floating_point():
                moved[key] = value.to(device=device, dtype=torch.float32)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def predictor_step_embeddings(
    *,
    model,
    prepared_info: dict,
    blocked_normalized_actions: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    action_tensor = torch.as_tensor(
        blocked_normalized_actions[None, None, ...], dtype=torch.float32, device=device
    )
    expanded_info = expand_info_for_candidates(prepared_info, 1)
    expanded_info = move_tensors_to_device(expanded_info, device)
    with torch.inference_mode():
        rollout = model.rollout(tensor_clone_info(expanded_info), action_tensor)
    return rollout["predicted_emb"][0, 0].detach().cpu().numpy()


def execute_real_step_embeddings(
    *,
    env,
    policy,
    model,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    action_block: int,
    seed: int,
) -> tuple[np.ndarray, list[list[float]]]:
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    configure_goal_render(env_unwrapped, goal_state)
    env_unwrapped._set_state(initial_state)

    embeddings = [encode_pixels(policy, model, env_unwrapped.render()).detach().cpu().numpy()[0]]
    states = [np.asarray(env_unwrapped._get_obs(), dtype=np.float64).tolist()]

    for block_start in range(0, len(raw_actions), action_block):
        for action in raw_actions[block_start : block_start + action_block]:
            env.step(np.asarray(action, dtype=np.float32))
        embeddings.append(encode_pixels(policy, model, env_unwrapped.render()).detach().cpu().numpy()[0])
        states.append(np.asarray(env_unwrapped._get_obs(), dtype=np.float64).tolist())

    return np.stack(embeddings), states


def regenerate_selected_cem_late(
    *,
    record: dict,
    dataset,
    policy,
    model,
    analysis_args: argparse.Namespace,
    offset: int,
    horizon_blocks: int,
) -> dict:
    selected = record["best_cem_late_action"]
    if selected is None:
        raise ValueError(f"Pair {record['pair_index']} has no CEM_late action")
    source_index = int(selected["source_index"])
    if source_index >= analysis_args.num_per_source:
        analysis_args.num_per_source = source_index + 1
    if analysis_args.num_per_source > analysis_args.topk:
        raise ValueError("--num-per-source must be <= --topk for CEM regeneration")

    rows = load_pair_rows(dataset, record["dataset_row"], offset)
    prepared_info = prepare_pair_info(policy, rows["initial"]["pixels"], rows["goal"]["pixels"])
    sequences = generate_cem_action_sequences(
        model=model,
        prepared_info=prepared_info,
        args=analysis_args,
        horizon_blocks=horizon_blocks,
        action_dim=analysis_args.action_block * 2,
        action_processor=policy.process["action"],
        pair_index=record["pair_index"],
    )
    late_sequences = [sequence for sequence in sequences if sequence["source"] == SOURCE_CEM_LATE]
    for sequence in late_sequences:
        if int(sequence["source_index"]) == source_index:
            return {
                "sequence": sequence,
                "prepared_info": prepared_info,
                "initial_state": np.asarray(rows["initial"]["state"], dtype=np.float64),
                "goal_state": np.asarray(rows["goal"]["state"], dtype=np.float64),
            }
    raise ValueError(
        f"Could not regenerate CEM_late source_index={source_index} for pair {record['pair_index']}"
    )


def compute_per_step_analysis(
    *,
    representatives: list[dict],
    dataset,
    policy,
    model,
    env,
    analysis_args: argparse.Namespace,
    offset: int,
    horizon_blocks: int,
) -> list[dict]:
    per_step = []
    for record in representatives:
        regenerated = regenerate_selected_cem_late(
            record=record,
            dataset=dataset,
            policy=policy,
            model=model,
            analysis_args=analysis_args,
            offset=offset,
            horizon_blocks=horizon_blocks,
        )
        sequence = regenerated["sequence"]
        pred_embeddings = predictor_step_embeddings(
            model=model,
            prepared_info=regenerated["prepared_info"],
            blocked_normalized_actions=sequence["blocked_normalized"],
        )
        real_embeddings, real_states = execute_real_step_embeddings(
            env=env,
            policy=policy,
            model=model,
            initial_state=regenerated["initial_state"],
            goal_state=regenerated["goal_state"],
            raw_actions=sequence["raw"],
            action_block=analysis_args.action_block,
            seed=analysis_args.seed + record["pair_index"] * 10_000 + int(sequence["source_index"]),
        )

        n_steps = min(len(pred_embeddings), len(real_embeddings))
        errors = np.linalg.norm(pred_embeddings[:n_steps] - real_embeddings[:n_steps], axis=1)
        real_deltas = np.zeros(n_steps, dtype=np.float64)
        real_deltas[1:] = np.linalg.norm(
            real_embeddings[1:n_steps] - real_embeddings[: n_steps - 1], axis=1
        )
        ratios = errors / (real_deltas + EPS)
        # Step 0 compares initial embeddings; the blow-up diagnostic is meaningful after motion starts.
        blowup_step = int(np.argmax(ratios[1:]) + 1) if n_steps > 1 else 0

        entry = {
            "pair_index": record["pair_index"],
            "category": record["category"],
            "success_count": record["success_count"],
            "selected_cem_late_source_index": int(sequence["source_index"]),
            "selected_from_json": record["best_cem_late_action"],
            "regenerated_cem_model_cost": float(sequence.get("cem_model_cost", math.nan)),
            "steps": list(range(n_steps)),
            "prediction_error_l2": errors.tolist(),
            "real_latent_delta_l2": real_deltas.tolist(),
            "error_to_real_delta_ratio": ratios.tolist(),
            "mean_error": float(np.mean(errors[1:])) if n_steps > 1 else float(errors[0]),
            "max_error": float(np.max(errors[1:])) if n_steps > 1 else float(errors[0]),
            "blowup_step": blowup_step,
            "blowup_ratio": float(ratios[blowup_step]),
            "real_states": real_states[:n_steps],
            "note": (
                "CEM_late actions are regenerated from the saved seed and pair index because "
                "the three-cost JSON stores costs, not raw action tensors."
            ),
        }
        record["per_step"] = {
            "mean_error": entry["mean_error"],
            "max_error": entry["max_error"],
            "blowup_step": entry["blowup_step"],
            "blowup_ratio": entry["blowup_ratio"],
        }
        per_step.append(entry)
    return per_step


def latent_correlations(pair_records: list[dict]) -> dict:
    latent = np.asarray(
        [record["latent"].get("latent_distance", np.nan) for record in pair_records],
        dtype=np.float64,
    )
    physical = np.asarray(
        [record["physical"]["physical_pose_distance"] for record in pair_records],
        dtype=np.float64,
    )
    block = np.asarray(
        [record["physical"]["block_displacement"] for record in pair_records],
        dtype=np.float64,
    )
    success = np.asarray([record["success_count"] for record in pair_records], dtype=np.float64)
    mask = np.isfinite(latent)
    if not np.any(mask):
        return {}
    return {
        "latent_vs_physical_pose_distance": {
            "pearson": pearson_corr(physical[mask], latent[mask]),
            "spearman": spearman_corr(physical[mask], latent[mask]),
        },
        "latent_vs_block_displacement": {
            "pearson": pearson_corr(block[mask], latent[mask]),
            "spearman": spearman_corr(block[mask], latent[mask]),
        },
        "latent_vs_success_count": {
            "pearson": pearson_corr(latent[mask], success[mask]),
            "spearman": spearman_corr(latent[mask], success[mask]),
        },
    }


def make_plots(
    *,
    pair_records: list[dict],
    per_step: list[dict],
    figures_dir: Path,
    no_plots: bool,
) -> dict:
    if no_plots:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    if any(record["latent"].get("latent_distance") is not None for record in pair_records):
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        grouped = group_by_category(pair_records)
        for category in CATEGORY_ORDER:
            records = [
                record
                for record in grouped.get(category, [])
                if record["latent"].get("latent_distance") is not None
            ]
            if not records:
                continue
            ax.scatter(
                [record["physical"]["physical_pose_distance"] for record in records],
                [record["latent"]["latent_distance"] for record in records],
                s=60,
                alpha=0.85,
                color=CATEGORY_COLORS[category],
                label=category,
                edgecolors="white",
                linewidths=0.8,
            )
            for record in records:
                ax.annotate(
                    str(record["pair_index"]),
                    (
                        record["physical"]["physical_pose_distance"],
                        record["latent"]["latent_distance"],
                    ),
                    fontsize=7,
                    alpha=0.65,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        ax.set_title("Encoder latent distance vs physical block-pose distance")
        ax.set_xlabel("physical_pose_distance = block displacement + rotation")
        ax.set_ylabel("||z_init - z_goal||")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figures_dir / "latent_vs_physical_distance.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["latent_vs_physical_distance"] = str(path)

    if per_step:
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        for entry in per_step:
            steps = entry["steps"][1:]
            errors = entry["prediction_error_l2"][1:]
            category = entry["category"]
            ax.plot(
                steps,
                errors,
                marker="o",
                linewidth=2.0,
                color=CATEGORY_COLORS.get(category, "#555555"),
                alpha=0.9,
                label=(
                    f"pair {entry['pair_index']} {category} "
                    f"({entry['success_count']}/40)"
                ),
            )
            ax.axvline(
                entry["blowup_step"],
                color=CATEGORY_COLORS.get(category, "#555555"),
                linestyle=":",
                alpha=0.25,
                linewidth=1.0,
            )
        ax.set_title("Per-step predictor error on planner-best CEM_late actions")
        ax.set_xlabel("latent step (5 raw env actions per step)")
        ax.set_ylabel("||predicted z_t - encoded real z_t||")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = figures_dir / "per_step_error.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["per_step_error"] = str(path)

    return paths


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_report(results: dict) -> str:
    lines = []
    lines.append("Per-Pair Failure Characterization")
    lines.append("=" * 35)
    lines.append(
        f"Input: offset={results['offset']} pairs={results['counts']['pairs']} "
        f"actions={results['counts']['actions']} device={results['metadata']['device']}"
    )
    lines.append("State format verified: [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]")
    lines.append("")

    lines.append("1. Category Summary")
    lines.append(
        "Category    Count  Mean corr(Creal_z, Creal_state)  "
        "Avg block_displacement  Avg rotation_required  Avg latent_dist"
    )
    for category in CATEGORY_ORDER:
        entry = results["category_summary"][category]
        lines.append(
            f"{category:<11} {entry['count']:>5}  "
            f"{fmt_mean_std(entry['spearman_c_real_z_vs_c_real_state']):>31}  "
            f"{fmt_float(entry['block_displacement']['mean']):>22}  "
            f"{fmt_float(entry['rotation_required']['mean']):>21}  "
            f"{fmt_float(entry['latent_distance']['mean']):>15}"
        )
    lines.append("")

    lines.append("2. Physical Feature Correlations")
    lines.append("Feature                         Spearman vs successes  Spearman vs impossible")
    for feature, entry in results["physical_feature_correlations"].items():
        lines.append(
            f"{feature:<31} "
            f"{fmt_float(entry['vs_success_count']['spearman']):>20}  "
            f"{fmt_float(entry['vs_impossible_indicator']['spearman']):>22}"
        )
    strongest = results["physical_feature_summary"]["strongest_abs_spearman_vs_success"]
    if strongest:
        direction = "fewer" if strongest["spearman"] < 0 else "more"
        lines.append(
            f"Strongest feature association: {strongest['feature']} "
            f"(Spearman {strongest['spearman']:.3f}; larger values imply {direction} successes)."
        )
    lines.append("")

    lines.append("3. Encoder Geometry By Pair")
    easy = results["category_summary"]["Easy"]["spearman_c_real_z_vs_c_real_state"]
    impossible = results["category_summary"]["Impossible"]["spearman_c_real_z_vs_c_real_state"]
    lines.append(
        "Easy mean encoder Spearman: "
        f"{fmt_mean_std(easy)}; Impossible mean encoder Spearman: {fmt_mean_std(impossible)}"
    )
    latent_corr = results["latent_correlations"].get("latent_vs_physical_pose_distance")
    if latent_corr:
        lines.append(
            "Latent-vs-physical pose distance: "
            f"Pearson={fmt_float(latent_corr['pearson'])}, "
            f"Spearman={fmt_float(latent_corr['spearman'])}"
        )
    lines.append("")

    lines.append("4. Representative Per-Step Predictor Error")
    if results["per_step_analysis"]:
        lines.append(
            "Pair  Category     Succ  CEM_idx  Mean_err  Max_err  Blowup_step  Blowup_ratio"
        )
        for entry in results["per_step_analysis"]:
            lines.append(
                f"{entry['pair_index']:>4}  {entry['category']:<11} "
                f"{entry['success_count']:>2}/40  "
                f"{entry['selected_cem_late_source_index']:>7}  "
                f"{entry['mean_error']:>8.3f}  "
                f"{entry['max_error']:>7.3f}  "
                f"{entry['blowup_step']:>11}  "
                f"{entry['blowup_ratio']:>12.3f}"
            )
    else:
        lines.append("Per-step analysis skipped or no representatives were available.")
    lines.append("")

    lines.append("5. Outputs")
    lines.append(f"Detailed JSON: {results['output_path']}")
    if results["figure_paths"]:
        lines.append("Figures:")
        for path in results["figure_paths"].values():
            lines.append(f"  {path}")
    else:
        lines.append("Figures: not generated")
    return "\n".join(lines)


def summarize_physical_feature_correlations(correlations: dict) -> dict:
    candidates = []
    for feature, entry in correlations.items():
        value = entry["vs_success_count"]["spearman"]
        if value is not None:
            candidates.append({"feature": feature, "spearman": value})
    if not candidates:
        return {"strongest_abs_spearman_vs_success": None}
    strongest = max(candidates, key=lambda item: abs(item["spearman"]))
    return {"strongest_abs_spearman_vs_success": strongest}


def main() -> int:
    args = parse_args()
    started = time.time()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.figures_dir = args.figures_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.device = resolve_device(args.device)

    data = load_results(args.input)
    offset = int(data["offset"])
    if offset % args.action_block != 0:
        raise ValueError("--action-block must divide the saved offset")
    horizon_blocks = offset // args.action_block

    metadata = data.get("metadata", {})
    if args.seed is None:
        args.seed = int(metadata.get("seed", 42))

    print("== Per-pair analysis setup ==")
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"figures_dir: {args.figures_dir}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"device: {args.device}")
    print(f"offset: {offset}")
    print(f"horizon_blocks: {horizon_blocks}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    pair_records = build_pair_records(data, dataset)

    policy = None
    model = None
    per_step = []
    representatives = []
    analysis_args = make_analysis_args(args, data, horizon_blocks)

    if not args.skip_step_analysis:
        process = build_processors(dataset, ["action", "proprio", "state"])
        policy = build_policy(make_policy_args(analysis_args, horizon_blocks), process)
        model = policy.solver.model
        print("Encoding pair endpoints for latent geometry...")
        compute_latent_distances(
            pair_records,
            dataset=dataset,
            policy=policy,
            model=model,
            offset=offset,
        )
        representatives = select_representatives(
            pair_records,
            per_category=args.representatives_per_category,
        )
        print(
            "Running per-step predictor diagnostics for representatives: "
            + ", ".join(
                f"pair {record['pair_index']} ({record['category']})"
                for record in representatives
            )
        )
        env = gym.make("swm/PushT-v1")
        try:
            per_step = compute_per_step_analysis(
                representatives=representatives,
                dataset=dataset,
                policy=policy,
                model=model,
                env=env,
                analysis_args=analysis_args,
                offset=offset,
                horizon_blocks=horizon_blocks,
            )
        finally:
            env.close()

    cat_summary = category_summary(pair_records)
    feature_corr = physical_feature_correlations(pair_records)
    feature_corr_summary = summarize_physical_feature_correlations(feature_corr)
    lat_corr = latent_correlations(pair_records)
    figure_paths = make_plots(
        pair_records=pair_records,
        per_step=per_step,
        figures_dir=args.figures_dir,
        no_plots=args.no_plots,
    )

    results = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "offset": offset,
        "metadata": {
            "seed": args.seed,
            "device": args.device,
            "checkpoint_dir": str(args.checkpoint_dir),
            "cache_dir": str(args.cache_dir),
            "dataset_name": args.dataset_name,
            "action_block": args.action_block,
            "horizon_blocks": horizon_blocks,
            "cem_regeneration": {
                "selection_rule": "min_c_model among CEM_late actions",
                "num_samples": analysis_args.num_samples,
                "topk": analysis_args.topk,
                "var_scale": analysis_args.var_scale,
                "cem_late_iters": analysis_args.cem_late_iters,
                "num_per_source": analysis_args.num_per_source,
                "note": (
                    "The source JSON stores costs, not action tensors, so representative "
                    "CEM_late actions are regenerated from seed and pair index."
                ),
            },
            "elapsed_seconds": time.time() - started,
        },
        "state_format": STATE_FORMAT,
        "category_definitions": {
            "Easy": ">25/40 successful actions",
            "Hard": "1-25/40 successful actions",
            "Impossible": "0/40 successful actions",
        },
        "counts": {
            "pairs": len(pair_records),
            "actions": sum(record["total_actions"] for record in pair_records),
            "by_category": {
                category: cat_summary[category]["count"] for category in CATEGORY_ORDER
            },
        },
        "category_summary": cat_summary,
        "physical_feature_correlations": feature_corr,
        "physical_feature_summary": feature_corr_summary,
        "latent_correlations": lat_corr,
        "representative_pairs": [
            {
                "pair_index": record["pair_index"],
                "category": record["category"],
                "success_count": record["success_count"],
                "best_cem_late_action": record["best_cem_late_action"],
            }
            for record in representatives
        ],
        "per_step_analysis": per_step,
        "pairs": pair_records,
        "figure_paths": figure_paths,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(results), indent=2, allow_nan=False) + "\n")

    print()
    print(build_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
