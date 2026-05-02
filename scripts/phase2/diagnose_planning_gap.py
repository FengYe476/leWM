#!/usr/bin/env python3
"""Diagnose why P2-0 ranking gains do not transfer to CEM planning.

The main test compares the distribution and label-ranking behavior of real
terminal latents, produced by encoding simulator observations, against imagined
terminal latents, produced by LeWM's predictor rollout. CEM only ever sees the
imagined latents, while the P2-0 cost head was trained on the real ones.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    expand_info_for_candidates,
    prepare_pair_info,
    tensor_clone_info,
)
from lewm_audit.eval.pusht import analyze_offset, prepare_dataset_index  # noqa: E402
from scripts.phase1.eval_track_a_three_cost import (  # noqa: E402
    ACTION_BLOCK,
    DEFAULT_PAIRS_PATH,
    IMG_SIZE,
    RANDOM_WAYPOINTS,
    SOURCE_LABELS,
    TOPK,
    VAR_SCALE,
    load_pairs,
    make_policy_namespace,
    make_three_cost_namespace,
    parse_pair_ids,
    select_action_sequences,
    validate_requested_pair_offsets,
)
from scripts.phase2.cost_head_model import LATENT_DIM, make_cost_head  # noqa: E402


DEFAULT_ARTIFACT = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "planning_gap_diagnosis.json"
DEFAULT_CHECKPOINT_DIR_SMALL_SPLIT3 = (
    PROJECT_ROOT / "results" / "phase2" / "p2_0" / "split3_small"
)
DEFAULT_TARGET_CELLS = (
    "D0xR0",
    "D0xR2",
    "D1xR0",
    "D1xR2",
    "D2xR0",
    "D2xR1",
    "D2xR2",
    "D2xR3",
    "D3xR0",
    "D3xR3",
)
ACTION_COUNTS = {"data": 20, "smooth_random": 20, "CEM_early": 20, "CEM_late": 20}
NUM_SAMPLES = 300
CEM_ITERS = 30
CEM_TRACE_ITERS = (1, 2, 3, 10, 30)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--model-checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR_SMALL_SPLIT3)
    parser.add_argument("--variant", choices=("small", "large"), default="small")
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-ids", type=parse_pair_ids, default=None)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-pair-id", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    """Convert nested values to strict JSON-safe objects."""
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


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks with tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman correlation without requiring scipy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def pairwise_accuracy(costs: np.ndarray, labels: np.ndarray) -> float | None:
    """Return pairwise ranking accuracy, skipping tied labels."""
    costs = np.asarray(costs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    correct = 0
    total = 0
    for i in range(len(costs)):
        for j in range(i + 1, len(costs)):
            if labels[i] == labels[j]:
                continue
            total += 1
            if (costs[i] < costs[j]) == (labels[i] < labels[j]):
                correct += 1
    return correct / total if total else None


def summarize_array(values: np.ndarray) -> dict[str, float | int | None]:
    """Return compact descriptive statistics."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def load_cost_head_checkpoint(path: Path, *, split: int, variant: str, device: str):
    """Load the requested trained cost head checkpoint."""
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob(f"split{split}_*_{variant}_seed*_best.pt"))
        if not candidates:
            candidates = sorted(path.glob(f"*_{variant}_seed*_best.pt"))
        if not candidates:
            raise FileNotFoundError(f"No {variant} *_best.pt checkpoint found in {path}")
        path = candidates[0]
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    checkpoint_args = checkpoint.get("args", {})
    use_temperature = bool(
        checkpoint.get("temperature")
        or checkpoint_args.get("temperature")
        or "log_temperature" in state_dict
    )
    temperature_init = float(checkpoint_args.get("temperature_init", 10.0))
    model = make_cost_head(
        variant,
        temperature=use_temperature,
        temperature_init=temperature_init,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return path, model


def select_diverse_pairs(pairs: list[dict], *, pair_ids: list[int] | None, max_pairs: int) -> list[dict]:
    """Select a deterministic 10-pair diagnostic subset across cells."""
    if pair_ids is not None:
        by_id = {int(pair["pair_id"]): pair for pair in pairs}
        missing = sorted(set(pair_ids) - set(by_id))
        if missing:
            raise ValueError(f"Requested pair IDs not found: {missing}")
        return [by_id[pair_id] for pair_id in pair_ids]

    selected = []
    used_ids = set()
    for cell in DEFAULT_TARGET_CELLS:
        matches = [pair for pair in pairs if str(pair["cell"]) == cell]
        if not matches:
            continue
        pair = sorted(matches, key=lambda item: int(item["pair_id"]))[0]
        selected.append(pair)
        used_ids.add(int(pair["pair_id"]))
        if len(selected) >= max_pairs:
            return selected

    for pair in sorted(pairs, key=lambda item: int(item["pair_id"])):
        if int(pair["pair_id"]) in used_ids:
            continue
        selected.append(pair)
        if len(selected) >= max_pairs:
            break
    return selected


def artifact_records_for_pairs(artifact: dict, pair_ids: set[int]) -> dict[tuple[int, int], dict]:
    """Return artifact records keyed by ``(pair_id, action_id)``."""
    records = {}
    n_records = int(artifact["pair_id"].numel())
    for idx in range(n_records):
        pair_id = int(artifact["pair_id"][idx])
        if pair_id not in pair_ids:
            continue
        action_id = int(artifact["action_id"][idx])
        records[(pair_id, action_id)] = {
            "pair_id": pair_id,
            "action_id": action_id,
            "source": artifact["source"][idx],
            "source_index": int(artifact["source_index"][idx]),
            "action_key": artifact["action_key"][idx],
            "cell": artifact["cell"][idx],
            "z_terminal": artifact["z_terminal"][idx].detach().cpu().numpy(),
            "z_goal": artifact["z_goal"][idx].detach().cpu().numpy(),
            "v1_cost": float(artifact["v1_cost"][idx]),
            "C_model": float(artifact["C_model"][idx]),
            "C_real_z": float(artifact["C_real_z"][idx]),
            "success": bool(artifact["success"][idx]),
        }
    return records


def source_index_for_action(source_counts: dict[str, int], source: str) -> int:
    """Return and increment the inferred per-source action index."""
    source_index = source_counts.get(source, 0)
    source_counts[source] = source_index + 1
    return source_index


def move_info_to_model_device(info: dict, model) -> dict:
    """Move tensor values in an info dict to the model device."""
    device = next(model.parameters()).device
    out = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            if device.type == "mps" and value.is_floating_point():
                out[key] = value.to(device=device, dtype=torch.float32)
            else:
                out[key] = value.to(device)
        else:
            out[key] = value
    return out


def rollout_predicted_latents(model, prepared_info: dict, blocked_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the predictor and return terminal predicted latents and goal latents."""
    device = next(model.parameters()).device
    candidates = torch.as_tensor(
        blocked_actions[None, ...],
        dtype=torch.float32,
        device=device,
    )
    info = expand_info_for_candidates(prepared_info, candidates.shape[1])
    info = tensor_clone_info(info)
    info = move_info_to_model_device(info, model)

    goal = {key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)}
    goal["pixels"] = goal["goal"]
    for key in list(info.keys()):
        if key.startswith("goal_"):
            goal[key[len("goal_") :]] = goal.pop(key)
    goal.pop("action")

    with torch.inference_mode():
        goal = model.encode(goal)
        info["goal_emb"] = goal["emb"]
        info = model.rollout(info, candidates)
    z_pred = info["predicted_emb"][0, :, -1, :].detach().cpu().numpy()
    z_goal = info["goal_emb"][0, -1, :].detach().cpu().numpy()
    return z_pred.astype(np.float32), z_goal.astype(np.float32)


@torch.inference_mode()
def predict_cost_head(cost_head, z: np.ndarray, z_goal: np.ndarray, *, device: str) -> np.ndarray:
    """Predict C_psi costs for a batch of terminal and goal latents."""
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    zg_t = torch.as_tensor(z_goal, dtype=torch.float32, device=device)
    return cost_head(z_t, zg_t).detach().cpu().numpy().astype(np.float64)


def cpsi_candidate_costs(model, cost_head, prepared_info: dict, candidates: torch.Tensor) -> torch.Tensor:
    """Compute C_psi costs for CEM candidates from imagined latents."""
    info = expand_info_for_candidates(prepared_info, candidates.shape[1])
    info = tensor_clone_info(info)
    info = move_info_to_model_device(info, model)

    goal = {key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)}
    goal["pixels"] = goal["goal"]
    for key in list(info.keys()):
        if key.startswith("goal_"):
            goal[key[len("goal_") :]] = goal.pop(key)
    goal.pop("action")

    with torch.inference_mode():
        goal = model.encode(goal)
        info["goal_emb"] = goal["emb"]
        info = model.rollout(info, candidates)
        pred = info["predicted_emb"][..., -1, :]
        goal_emb = info["goal_emb"][:, -1, :]
        while goal_emb.ndim < pred.ndim:
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb.expand_as(pred)
        flat_pred = pred.reshape(-1, pred.shape[-1])
        flat_goal = goal_emb.reshape(-1, goal_emb.shape[-1])
        costs = cost_head(flat_pred, flat_goal).reshape(pred.shape[:-1])
    return costs


def run_cem_trace(
    *,
    model,
    cost_head,
    prepared_info: dict,
    seed: int,
    use_cpsi: bool,
) -> dict:
    """Run one manual CEM solve and record cost ranges and final action mean."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    mean = torch.zeros((1, 5, ACTION_BLOCK * 2), device=device)
    var = VAR_SCALE * torch.ones((1, 5, ACTION_BLOCK * 2), device=device)
    iterations = []

    for iter_idx in range(1, CEM_ITERS + 1):
        candidates = torch.randn(
            1,
            NUM_SAMPLES,
            5,
            ACTION_BLOCK * 2,
            generator=generator,
            device=device,
        )
        candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean
        if use_cpsi:
            costs = cpsi_candidate_costs(model, cost_head, prepared_info, candidates)
        else:
            expanded = expand_info_for_candidates(prepared_info, NUM_SAMPLES)
            costs = model.get_cost(tensor_clone_info(expanded), candidates)
        top_vals, top_inds = torch.topk(costs, k=TOPK, dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]
        if iter_idx in CEM_TRACE_ITERS:
            cost_np = costs[0].detach().cpu().numpy()
            top_np = top_vals[0].detach().cpu().numpy()
            iterations.append(
                {
                    "iteration": iter_idx,
                    "cost_min": float(np.min(cost_np)),
                    "cost_max": float(np.max(cost_np)),
                    "cost_mean": float(np.mean(cost_np)),
                    "cost_std": float(np.std(cost_np)),
                    "top30_min": float(np.min(top_np)),
                    "top30_max": float(np.max(top_np)),
                    "top30_std": float(np.std(top_np)),
                }
            )
        mean = elite_candidates.mean(dim=1)
        var = elite_candidates.std(dim=1)
    return {
        "iterations": iterations,
        "final_action_mean": mean.detach().cpu().numpy()[0],
    }


def metric_block(costs: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> dict:
    """Return overall and per-pair ranking metrics."""
    per_pair = {}
    rhos = []
    accuracies = []
    for pair_id in sorted(set(int(pid) for pid in pair_ids)):
        mask = pair_ids == pair_id
        rho = spearman(costs[mask], labels[mask])
        pa = pairwise_accuracy(costs[mask], labels[mask])
        per_pair[pair_id] = {
            "spearman": rho,
            "pairwise_accuracy": pa,
            "n_records": int(np.count_nonzero(mask)),
        }
        if rho is not None:
            rhos.append(rho)
        if pa is not None:
            accuracies.append(pa)
    return {
        "global_spearman": spearman(costs, labels),
        "global_pairwise_accuracy": pairwise_accuracy(costs, labels),
        "per_pair_spearman_mean": float(np.mean(rhos)) if rhos else None,
        "per_pair_spearman_std": float(np.std(rhos)) if rhos else None,
        "per_pair_pairwise_accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
        "per_pair": per_pair,
    }


def diagnose(args: argparse.Namespace) -> dict:
    """Run the planning-gap diagnostic."""
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    args.artifact = args.artifact.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.model_checkpoint_dir = args.model_checkpoint_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    _, all_pairs = load_pairs(args.pairs_path, max_pairs=None, pair_ids=None)
    selected_pairs = select_diverse_pairs(all_pairs, pair_ids=args.pair_ids, max_pairs=args.max_pairs)
    selected_pair_ids = [int(pair["pair_id"]) for pair in selected_pairs]
    artifact_lookup = artifact_records_for_pairs(artifact, set(selected_pair_ids))

    pair_metadata = json.loads(args.pairs_path.read_text())["metadata"]
    offset = int(pair_metadata["offset"])
    validate_requested_pair_offsets(selected_pairs, offset=offset)

    checkpoint_path, cost_head = load_cost_head_checkpoint(
        args.checkpoint_dir,
        split=args.split,
        variant=args.variant,
        device=args.device,
    )

    print("== P2-0 planning-gap diagnostic ==")
    print(f"device: {args.device}")
    print(f"pairs: {selected_pair_ids}")
    print(f"cost_head_checkpoint: {checkpoint_path}")

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    index = prepare_dataset_index(dataset)
    analysis = analyze_offset(index, offset)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy = build_policy(
        make_policy_namespace(
            checkpoint_dir=args.model_checkpoint_dir,
            device=args.device,
            seed=args.seed,
        ),
        process,
    )
    model = policy.solver.model
    cost_args = make_three_cost_namespace(
        checkpoint_dir=args.model_checkpoint_dir,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        device=args.device,
        seed=args.seed,
        offset=offset,
        max_cem_count=max(ACTION_COUNTS["CEM_early"], ACTION_COUNTS["CEM_late"]),
    )

    rows = []
    mismatches = []
    started = time.time()
    for idx, pair_spec in enumerate(selected_pairs, start=1):
        pair_id = int(pair_spec["pair_id"])
        rows_data = dataset.get_row_data([int(pair_spec["start_row"]), int(pair_spec["goal_row"])])
        initial = {key: value[0] for key, value in rows_data.items()}
        goal = {key: value[1] for key, value in rows_data.items()}
        prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
        raw_steps = int(pair_spec["goal_row"]) - int(pair_spec["start_row"])
        print(f"[{idx:02d}/{len(selected_pairs):02d}] pair={pair_id} cell={pair_spec['cell']}")
        sequences = select_action_sequences(
            dataset=dataset,
            valid_action_indices=analysis["valid_indices"],
            policy=policy,
            model=model,
            prepared_info=prepared_info,
            args=cost_args,
            pair_id=pair_id,
            raw_steps=raw_steps,
            action_counts=ACTION_COUNTS,
        )
        blocked = np.stack([sequence["blocked_normalized"] for sequence in sequences])
        z_pred, z_goal_pred = rollout_predicted_latents(model, prepared_info, blocked)
        source_counts: dict[str, int] = {}

        for action_id, sequence in enumerate(sequences):
            key = (pair_id, action_id)
            if key not in artifact_lookup:
                raise KeyError(f"Missing artifact record for {key}")
            artifact_record = artifact_lookup[key]
            source = SOURCE_LABELS[sequence["source"]]
            source_index = source_index_for_action(source_counts, source)
            if (
                source != artifact_record["source"]
                or source_index != artifact_record["source_index"]
            ):
                mismatches.append(
                    {
                        "pair_id": pair_id,
                        "action_id": action_id,
                        "expected": [artifact_record["source"], artifact_record["source_index"]],
                        "replayed": [source, source_index],
                    }
                )
            z_terminal = artifact_record["z_terminal"]
            z_goal = artifact_record["z_goal"]
            pred_euclidean = float(np.sum((z_pred[action_id] - z_goal) ** 2))
            rows.append(
                {
                    "pair_id": pair_id,
                    "cell": str(pair_spec["cell"]),
                    "action_id": action_id,
                    "source": source,
                    "source_index": source_index,
                    "v1_cost": artifact_record["v1_cost"],
                    "C_model_artifact": artifact_record["C_model"],
                    "C_model_from_z_pred": pred_euclidean,
                    "z_terminal": z_terminal,
                    "z_predicted": z_pred[action_id],
                    "z_goal": z_goal,
                    "z_goal_pred_check_l2": float(np.linalg.norm(z_goal_pred - z_goal)),
                }
            )

    pair_ids = np.asarray([row["pair_id"] for row in rows], dtype=np.int64)
    labels = np.asarray([row["v1_cost"] for row in rows], dtype=np.float64)
    z_terminal = np.stack([row["z_terminal"] for row in rows]).astype(np.float32)
    z_predicted = np.stack([row["z_predicted"] for row in rows]).astype(np.float32)
    z_goal = np.stack([row["z_goal"] for row in rows]).astype(np.float32)

    cpsi_terminal = predict_cost_head(cost_head, z_terminal, z_goal, device=args.device)
    cpsi_predicted = predict_cost_head(cost_head, z_predicted, z_goal, device=args.device)
    euclidean_terminal = np.sum((z_terminal - z_goal) ** 2, axis=1)
    euclidean_predicted = np.sum((z_predicted - z_goal) ** 2, axis=1)

    l2_terminal_pred = np.linalg.norm(z_terminal - z_predicted, axis=1)
    l2_terminal_goal = np.linalg.norm(z_terminal - z_goal, axis=1)
    l2_pred_goal = np.linalg.norm(z_predicted - z_goal, axis=1)
    cmodel_abs_error = np.abs(
        np.asarray([row["C_model_artifact"] for row in rows]) - euclidean_predicted
    )
    z_terminal_dim_mean = z_terminal.mean(axis=0)
    z_pred_dim_mean = z_predicted.mean(axis=0)
    z_terminal_dim_std = z_terminal.std(axis=0)
    z_pred_dim_std = z_predicted.std(axis=0)

    trace_pair_id = args.trace_pair_id if args.trace_pair_id is not None else selected_pair_ids[0]
    trace_pair = next(pair for pair in selected_pairs if int(pair["pair_id"]) == trace_pair_id)
    rows_data = dataset.get_row_data([int(trace_pair["start_row"]), int(trace_pair["goal_row"])])
    trace_info = prepare_pair_info(
        policy,
        rows_data["pixels"][0],
        rows_data["pixels"][1],
    )
    euc_trace = run_cem_trace(
        model=model,
        cost_head=cost_head,
        prepared_info=trace_info,
        seed=args.seed + trace_pair_id * 1009,
        use_cpsi=False,
    )
    cpsi_trace = run_cem_trace(
        model=model,
        cost_head=cost_head,
        prepared_info=trace_info,
        seed=args.seed + trace_pair_id * 1009,
        use_cpsi=True,
    )
    action_delta = np.asarray(euc_trace["final_action_mean"]) - np.asarray(
        cpsi_trace["final_action_mean"]
    )

    summary = {
        "metadata": {
            "seed": args.seed,
            "device": args.device,
            "artifact": args.artifact,
            "pairs_path": args.pairs_path,
            "model_checkpoint_dir": args.model_checkpoint_dir,
            "cost_head_checkpoint": checkpoint_path,
            "variant": args.variant,
            "split": args.split,
            "selected_pair_ids": selected_pair_ids,
            "selected_cells": [str(pair["cell"]) for pair in selected_pairs],
            "n_records": len(rows),
            "elapsed_seconds": time.time() - started,
        },
        "latent_gap": {
            "l2_terminal_vs_predicted": summarize_array(l2_terminal_pred),
            "l2_terminal_vs_goal": summarize_array(l2_terminal_goal),
            "l2_predicted_vs_goal": summarize_array(l2_pred_goal),
            "mean_gap_to_goal_ratio": float(np.mean(l2_terminal_pred) / np.mean(l2_terminal_goal)),
            "z_goal_reencode_l2_check": summarize_array(
                np.asarray([row["z_goal_pred_check_l2"] for row in rows])
            ),
            "cmodel_reconstruction_abs_error": summarize_array(cmodel_abs_error),
            "per_dim": {
                "terminal_mean_mean": float(np.mean(z_terminal_dim_mean)),
                "predicted_mean_mean": float(np.mean(z_pred_dim_mean)),
                "mean_abs_mean_shift": float(np.mean(np.abs(z_terminal_dim_mean - z_pred_dim_mean))),
                "terminal_std_mean": float(np.mean(z_terminal_dim_std)),
                "predicted_std_mean": float(np.mean(z_pred_dim_std)),
                "mean_std_ratio_predicted_over_terminal": float(
                    np.mean(z_pred_dim_std / np.maximum(z_terminal_dim_std, 1e-8))
                ),
            },
        },
        "ranking": {
            "cpsi_on_terminal": metric_block(cpsi_terminal, labels, pair_ids),
            "cpsi_on_predicted": metric_block(cpsi_predicted, labels, pair_ids),
            "euclidean_on_terminal": metric_block(euclidean_terminal, labels, pair_ids),
            "euclidean_on_predicted": metric_block(euclidean_predicted, labels, pair_ids),
            "cpsi_terminal_vs_predicted_cost_spearman": spearman(
                cpsi_terminal,
                cpsi_predicted,
            ),
            "cpsi_terminal_vs_predicted_cost_l2": summarize_array(
                np.abs(cpsi_terminal - cpsi_predicted)
            ),
        },
        "cost_scale": {
            "track_a_cpsi_terminal": summarize_array(cpsi_terminal),
            "track_a_cpsi_predicted": summarize_array(cpsi_predicted),
            "track_a_euclidean_predicted": summarize_array(euclidean_predicted),
            "cem_trace_pair_id": trace_pair_id,
            "cem_euclidean": euc_trace["iterations"],
            "cem_cpsi": cpsi_trace["iterations"],
        },
        "patching_verification": {
            "trace_pair_id": trace_pair_id,
            "euclidean_and_cpsi_select_different_actions": bool(np.max(np.abs(action_delta)) > 1e-6),
            "final_action_mean_abs_delta": float(np.mean(np.abs(action_delta))),
            "final_action_max_abs_delta": float(np.max(np.abs(action_delta))),
        },
        "replay_validation": {
            "source_mismatches": mismatches[:20],
            "n_source_mismatches": len(mismatches),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False) + "\n")
    return summary


def print_summary(summary: dict) -> None:
    """Print the requested diagnostic summary."""
    gap = summary["latent_gap"]
    ranking = summary["ranking"]
    scale = summary["cost_scale"]
    patch = summary["patching_verification"]
    term = ranking["cpsi_on_terminal"]
    pred = ranking["cpsi_on_predicted"]
    term_rho = term["global_spearman"]
    pred_rho = pred["global_spearman"]
    term_pa = term["global_pairwise_accuracy"]
    pred_pa = pred["global_pairwise_accuracy"]
    rho_gap = None if term_rho is None or pred_rho is None else term_rho - pred_rho
    pa_gap = None if term_pa is None or pred_pa is None else term_pa - pred_pa
    cpsi_last = scale["cem_cpsi"][-1]
    euc_last = scale["cem_euclidean"][-1]

    print("\nPlanning Gap Diagnosis")
    print("Encoder vs Predictor latent gap:")
    print(
        "Mean L2(z_terminal, z_predicted): "
        f"{gap['l2_terminal_vs_predicted']['mean']:.4f}"
    )
    print(
        "Mean L2(z_terminal, z_goal): "
        f"{gap['l2_terminal_vs_goal']['mean']:.4f}"
    )
    print(f"Ratio: {gap['mean_gap_to_goal_ratio']:.4f}")
    print("C_psi ranking quality on imagined vs real latents:")
    print(f"Spearman on z_terminal: {term_rho:.4f}")
    print(f"Spearman on z_predicted: {pred_rho:.4f}")
    print(f"Spearman gap: {rho_gap:.4f}")
    print(f"Pairwise acc on z_terminal: {term_pa:.4f}")
    print(f"Pairwise acc on z_predicted: {pred_pa:.4f}")
    print(f"Pairwise acc gap: {pa_gap:.4f}")
    print("Cost scale:")
    print(
        "C_psi output range during final CEM trace iter: "
        f"[{cpsi_last['cost_min']:.4f}, {cpsi_last['cost_max']:.4f}]"
    )
    print(
        "Euclidean cost range during final CEM trace iter: "
        f"[{euc_last['cost_min']:.4f}, {euc_last['cost_max']:.4f}]"
    )
    print(f"C_psi elite spread top-30 std: {cpsi_last['top30_std']:.4f}")
    print("Patching verification:")
    print(
        "Euclidean and C_psi select different actions: "
        f"{'yes' if patch['euclidean_and_cpsi_select_different_actions'] else 'no'}"
    )
    if pred_rho is not None and term_rho is not None and pred_rho < term_rho - 0.10:
        fix = "distribution shift"
    elif cpsi_last["top30_std"] < 1e-3:
        fix = "cost scale"
    elif not patch["euclidean_and_cpsi_select_different_actions"]:
        fix = "patching bug"
    else:
        fix = "mixed: imagined-latent ranking and CEM objective mismatch"
    print(f"Recommended fix: {fix}")


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    summary = diagnose(args)
    print_summary(summary)
    print(f"\nsaved_results: {args.output}")
    print(f"elapsed_seconds: {summary['metadata']['elapsed_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
