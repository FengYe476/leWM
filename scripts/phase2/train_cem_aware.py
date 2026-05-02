#!/usr/bin/env python3
"""Train CEM-aware cost heads from online PushT oracle labels.

Offline P2-0 training ranks fixed Track A candidates. CEM planning instead
queries the cost function on candidates produced by its own iterative sampling
distribution. This script distills V1 oracle labels onto those CEM-generated
predictor latents: for each training pair, run CEM under the current cost head,
collect candidate predictor latents from the final iterations, label selected
candidates by real PushT rollout with the V1 hinge cost, and update the head
with scalar Huber regression.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch
from torch.nn import functional as F


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
    blocked_normalized_to_raw,
    expand_info_for_candidates,
    prepare_pair_info,
    tensor_clone_info,
)
from lewm_audit.eval.oracle_cem import cost_v1_hinge, rollout_final_state  # noqa: E402
from scripts.phase2.cost_head_model import LATENT_DIM, make_cost_head  # noqa: E402
from scripts.phase2.dataloader import (  # noqa: E402
    DEFAULT_LATENT_ARTIFACT,
    DEFAULT_PREDICTED_LATENT_ARTIFACT,
    latent_examples_from_artifact,
    load_latent_artifact,
    load_predicted_latent_artifact,
    validate_predicted_artifact_alignment,
)
from scripts.phase2.mahalanobis_baseline import spearman  # noqa: E402
from scripts.phase2.splits import (  # noqa: E402
    DEFAULT_PAIRS_PATH,
    load_track_a_pairs,
    split1_random_holdout,
    split3_hard_pair_holdout,
)


NUM_SAMPLES = 300
CEM_ITERS = 30
TOPK = 30
VAR_SCALE = 1.0
PLANNING_HORIZON = 5
ACTION_BLOCK = 5
IMG_SIZE = 224
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "cem_aware"
DEFAULT_DATASET_NAME = "pusht_expert_train"


@dataclass
class PairCollectionStats:
    """Diagnostics from one online pair collection/update."""

    epoch: int
    pair_id: int
    cell: str
    collected: int
    label_min: float | None
    label_max: float | None
    label_mean: float | None
    loss_first: float | None
    loss_last: float | None
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, choices=(1, 3), default=3)
    parser.add_argument("--variant", choices=("small", "large"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--huber-beta", type=float, default=10.0)
    parser.add_argument("--cem-iters-to-collect", type=int, default=5)
    parser.add_argument("--candidates-per-iter", type=int, default=100)
    parser.add_argument("--train-steps-per-pair", type=int, default=10)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument(
        "--predicted-artifact",
        type=Path,
        default=DEFAULT_PREDICTED_LATENT_ARTIFACT,
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
    """Convert nested objects to JSON-safe values."""
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


def save_json(path: Path, data: dict) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n")


def load_pairs_by_id(path: Path) -> dict[int, dict]:
    """Load Track A pair metadata keyed by pair ID."""
    return {int(pair["pair_id"]): pair for pair in load_track_a_pairs(path)}


def split_definition(args: argparse.Namespace) -> dict[str, list[int]]:
    """Return the requested Phase 2 train/val/test split."""
    if args.split == 1:
        return split1_random_holdout(args.pairs_path, seed=args.seed)
    if args.split == 3:
        return split3_hard_pair_holdout(args.pairs_path, seed=args.seed)
    raise ValueError(f"Unsupported CEM-aware split: {args.split}")


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


def rollout_candidate_latents(
    model,
    prepared_info: dict,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predictor terminal latents and goal latents for CEM candidates."""
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
        z_pred = info["predicted_emb"][..., -1, :]
        z_goal = info["goal_emb"][:, -1, :]
    return z_pred, z_goal


@torch.inference_mode()
def candidate_costs_from_latents(
    cost_head,
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
) -> torch.Tensor:
    """Score candidate predictor latents with the current cost head."""
    while z_goal.ndim < z_pred.ndim:
        z_goal = z_goal.unsqueeze(1)
    z_goal = z_goal.expand_as(z_pred)
    flat_pred = z_pred.reshape(-1, z_pred.shape[-1])
    flat_goal = z_goal.reshape(-1, z_goal.shape[-1])
    return cost_head(flat_pred, flat_goal).reshape(z_pred.shape[:-1])


def load_pair_rows(dataset, pair: dict) -> tuple[dict, dict]:
    """Load initial and goal rows for a Track A pair."""
    rows = dataset.get_row_data([int(pair["start_row"]), int(pair["goal_row"])])
    initial = {key: value[0] for key, value in rows.items()}
    goal = {key: value[1] for key, value in rows.items()}
    return initial, goal


def blocked_batch_to_raw(
    blocked: np.ndarray,
    *,
    action_processor,
) -> np.ndarray:
    """Convert a batch of blocked normalized candidates to raw env actions."""
    return np.stack(
        [
            blocked_normalized_to_raw(
                candidate,
                action_processor=action_processor,
                action_block=ACTION_BLOCK,
            )
            for candidate in np.asarray(blocked, dtype=np.float32)
        ],
        axis=0,
    ).astype(np.float32)


def collect_pair_cem_samples(
    *,
    cost_head,
    world_model,
    policy,
    env,
    pair: dict,
    initial: dict,
    goal: dict,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    """Run CEM for one pair and collect oracle-labelled candidate latents."""
    device = next(world_model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(args.seed + int(pair["pair_id"]) * 1009)
    prepared_info = prepare_pair_info(policy, initial["pixels"], goal["pixels"])
    mean = torch.zeros((1, PLANNING_HORIZON, ACTION_BLOCK * 2), device=device)
    var = VAR_SCALE * torch.ones((1, PLANNING_HORIZON, ACTION_BLOCK * 2), device=device)
    collect_start_iter = CEM_ITERS - args.cem_iters_to_collect + 1
    action_processor = policy.process["action"]
    initial_state = np.asarray(initial["state"], dtype=np.float32)
    goal_state = np.asarray(goal["state"], dtype=np.float32)

    z_values = []
    z_goal_values = []
    labels = []
    iter_stats = []

    for iter_idx in range(1, CEM_ITERS + 1):
        candidates = torch.randn(
            1,
            NUM_SAMPLES,
            PLANNING_HORIZON,
            ACTION_BLOCK * 2,
            generator=generator,
            device=device,
        )
        candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        z_pred, z_goal = rollout_candidate_latents(world_model, prepared_info, candidates)
        costs = candidate_costs_from_latents(cost_head, z_pred, z_goal)
        top_vals, top_inds = torch.topk(costs, k=TOPK, dim=1, largest=False)
        elite_candidates = candidates[:, top_inds[0]]

        if iter_idx >= collect_start_iter:
            n_collect = min(args.candidates_per_iter, NUM_SAMPLES)
            selected = rng.choice(NUM_SAMPLES, size=n_collect, replace=False)
            blocked_np = candidates[0, selected].detach().cpu().numpy().astype(np.float32)
            raw_candidates = blocked_batch_to_raw(
                blocked_np,
                action_processor=action_processor,
            )
            z_np = z_pred[0, selected].detach().cpu().numpy().astype(np.float32)
            z_goal_np = z_goal[0].detach().cpu().numpy().astype(np.float32)
            iter_labels = []
            for local_idx, raw_actions in enumerate(raw_candidates):
                terminal_state = rollout_final_state(
                    env,
                    initial_state,
                    goal_state,
                    raw_actions,
                    seed=args.seed
                    + int(pair["pair_id"]) * 100_000
                    + iter_idx * 1_000
                    + int(local_idx),
                )
                label = float(cost_v1_hinge(terminal_state, goal_state))
                labels.append(label)
                iter_labels.append(label)
            z_values.append(z_np)
            z_goal_values.append(np.repeat(z_goal_np[None, :], n_collect, axis=0))
            iter_stats.append(
                {
                    "iteration": iter_idx,
                    "collected": int(n_collect),
                    "model_cost_min": float(costs[0].min().detach().cpu()),
                    "model_cost_max": float(costs[0].max().detach().cpu()),
                    "model_top30_std": float(top_vals[0].std(unbiased=False).detach().cpu()),
                    "v1_min": float(np.min(iter_labels)) if iter_labels else None,
                    "v1_max": float(np.max(iter_labels)) if iter_labels else None,
                    "v1_mean": float(np.mean(iter_labels)) if iter_labels else None,
                }
            )

        mean = elite_candidates.mean(dim=1)
        var = elite_candidates.std(dim=1)

    if z_values:
        z_arr = np.concatenate(z_values, axis=0)
        z_goal_arr = np.concatenate(z_goal_values, axis=0)
        labels_arr = np.asarray(labels, dtype=np.float32)
    else:
        z_arr = np.empty((0, LATENT_DIM), dtype=np.float32)
        z_goal_arr = np.empty((0, LATENT_DIM), dtype=np.float32)
        labels_arr = np.empty((0,), dtype=np.float32)
    return {
        "z": z_arr,
        "z_goal": z_goal_arr,
        "labels": labels_arr,
        "iteration_stats": iter_stats,
    }


def train_on_collected_samples(
    *,
    cost_head,
    optimizer: torch.optim.Optimizer,
    samples: dict,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[float]:
    """Run scalar Huber distillation steps on one pair's collected samples."""
    z = torch.as_tensor(samples["z"], dtype=torch.float32, device=args.device)
    z_goal = torch.as_tensor(samples["z_goal"], dtype=torch.float32, device=args.device)
    labels = torch.as_tensor(samples["labels"], dtype=torch.float32, device=args.device)
    if z.shape[0] == 0:
        return []

    losses = []
    cost_head.train()
    n_samples = int(z.shape[0])
    batch_size = min(args.batch_size, n_samples)
    for _ in range(args.train_steps_per_pair):
        if batch_size < n_samples:
            idx_np = rng.choice(n_samples, size=batch_size, replace=False)
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=args.device)
        else:
            idx = torch.arange(n_samples, dtype=torch.long, device=args.device)
        optimizer.zero_grad(set_to_none=True)
        pred = cost_head(z[idx], z_goal[idx])
        loss = F.smooth_l1_loss(pred, labels[idx], beta=args.huber_beta)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.inference_mode()
def evaluate_on_predicted_artifact(
    cost_head,
    *,
    real_artifact: dict,
    pred_artifact: dict,
    pair_ids: list[int],
    device: str,
    batch_size: int,
) -> dict[str, float | None]:
    """Evaluate Spearman on held-out predictor-latent artifact records."""
    validate_predicted_artifact_alignment(real_artifact, pred_artifact)
    joined = dict(real_artifact)
    joined["z_predicted"] = pred_artifact["z_predicted"]
    examples = latent_examples_from_artifact(
        joined,
        pair_ids=set(pair_ids),
        latent_key="z_predicted",
        latent_type="predictor",
    )
    if not examples:
        return {"spearman": None, "n_examples": 0}
    preds = []
    labels = []
    cost_head.eval()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        z = torch.as_tensor(
            np.stack([example.z for example in chunk]),
            dtype=torch.float32,
            device=device,
        )
        z_goal = torch.as_tensor(
            np.stack([example.z_g for example in chunk]),
            dtype=torch.float32,
            device=device,
        )
        preds.append(cost_head(z, z_goal).detach().cpu().numpy())
        labels.extend(example.v1_cost for example in chunk)
    pred_arr = np.concatenate(preds).astype(np.float64)
    label_arr = np.asarray(labels, dtype=np.float64)
    return {
        "spearman": spearman(pred_arr, label_arr),
        "n_examples": len(examples),
        "prediction_min": float(np.min(pred_arr)),
        "prediction_max": float(np.max(pred_arr)),
        "label_min": float(np.min(label_arr)),
        "label_max": float(np.max(label_arr)),
    }


def save_checkpoint(
    path: Path,
    *,
    cost_head,
    args: argparse.Namespace,
    epoch: int,
    val_metrics: dict,
) -> None:
    """Save a CEM-aware MLP checkpoint compatible with eval_planning.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": cost_head.state_dict(),
            "model_type": "mlp",
            "variant": args.variant,
            "split": args.split,
            "epoch": epoch,
            "val_spearman_predicted": val_metrics.get("spearman"),
            "args": vars(args),
            "training_objective": "cem_aware_v1_smooth_l1",
            "huber_beta": args.huber_beta,
        },
        path,
    )


def run(args: argparse.Namespace) -> dict:
    """Run CEM-aware online distillation."""
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.model_checkpoint_dir = args.model_checkpoint_dir.expanduser().resolve()
    args.artifact = args.artifact.expanduser().resolve()
    args.predicted_artifact = args.predicted_artifact.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.cem_iters_to_collect < 1 or args.cem_iters_to_collect > CEM_ITERS:
        raise ValueError(f"--cem-iters-to-collect must be in [1, {CEM_ITERS}]")
    if args.candidates_per_iter < 1 or args.candidates_per_iter > NUM_SAMPLES:
        raise ValueError(f"--candidates-per-iter must be in [1, {NUM_SAMPLES}]")

    split = split_definition(args)
    train_pair_ids = split["train_pair_ids"]
    if args.max_train_pairs is not None:
        train_pair_ids = train_pair_ids[: args.max_train_pairs]
    pairs_by_id = load_pairs_by_id(args.pairs_path)
    train_pairs = [pairs_by_id[pair_id] for pair_id in train_pair_ids]
    real_artifact = load_latent_artifact(args.artifact)
    pred_artifact = load_predicted_latent_artifact(args.predicted_artifact)

    dataset = get_dataset(args.cache_dir, args.dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy_args = argparse.Namespace(
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
    policy = build_policy(policy_args, process)
    world_model = policy.solver.model
    cost_head = make_cost_head(args.variant).to(args.device)
    optimizer = torch.optim.Adam(cost_head.parameters(), lr=args.lr)
    env = gym.make("swm/PushT-v1")
    rng = np.random.default_rng(args.seed)

    prefix = args.output_dir / f"split{args.split}_cem_aware_{args.variant}_seed{args.seed}"
    best_checkpoint = prefix.with_name(prefix.name + "_best.pt")
    log_path = prefix.with_name(prefix.name + "_train_log.json")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    epoch_checkpoint_template = str(prefix.with_name(prefix.name + "_epoch{epoch}.pt"))

    print("== CEM-aware online distillation ==")
    print(f"device: {args.device}")
    print(f"split: {args.split}")
    print(f"variant: {args.variant}")
    print(f"train_pairs: {len(train_pairs)}")
    print(f"output_dir: {args.output_dir}")
    print(
        "collection: "
        f"last_iters={args.cem_iters_to_collect} "
        f"candidates_per_iter={args.candidates_per_iter} "
        f"train_steps_per_pair={args.train_steps_per_pair}"
    )

    all_pair_logs = []
    epoch_logs = []
    best_val_spearman = -math.inf
    started = time.time()
    try:
        for epoch in range(1, args.outer_epochs + 1):
            epoch_start = time.time()
            epoch_losses = []
            epoch_collected = 0
            for pair_index, pair in enumerate(train_pairs, start=1):
                pair_start = time.time()
                pair_id = int(pair["pair_id"])
                initial, goal = load_pair_rows(dataset, pair)
                samples = collect_pair_cem_samples(
                    cost_head=cost_head,
                    world_model=world_model,
                    policy=policy,
                    env=env,
                    pair=pair,
                    initial=initial,
                    goal=goal,
                    args=args,
                    rng=rng,
                )
                losses = train_on_collected_samples(
                    cost_head=cost_head,
                    optimizer=optimizer,
                    samples=samples,
                    args=args,
                    rng=rng,
                )
                labels = samples["labels"]
                epoch_losses.extend(losses)
                epoch_collected += int(len(labels))
                pair_stats = PairCollectionStats(
                    epoch=epoch,
                    pair_id=pair_id,
                    cell=str(pair["cell"]),
                    collected=int(len(labels)),
                    label_min=float(np.min(labels)) if len(labels) else None,
                    label_max=float(np.max(labels)) if len(labels) else None,
                    label_mean=float(np.mean(labels)) if len(labels) else None,
                    loss_first=losses[0] if losses else None,
                    loss_last=losses[-1] if losses else None,
                    elapsed_seconds=float(time.time() - pair_start),
                )
                all_pair_logs.append({**asdict(pair_stats), "iterations": samples["iteration_stats"]})
                print(
                    f"[epoch={epoch} pair={pair_index:03d}/{len(train_pairs):03d} "
                    f"id={pair_id} cell={pair['cell']}] "
                    f"collected={pair_stats.collected} "
                    f"v1=[{pair_stats.label_min},{pair_stats.label_max}] "
                    f"loss={pair_stats.loss_first}->{pair_stats.loss_last} "
                    f"elapsed={pair_stats.elapsed_seconds:.1f}s"
                )

            val_metrics = evaluate_on_predicted_artifact(
                cost_head,
                real_artifact=real_artifact,
                pred_artifact=pred_artifact,
                pair_ids=split["val_pair_ids"],
                device=args.device,
                batch_size=args.batch_size,
            )
            test_probe_metrics = evaluate_on_predicted_artifact(
                cost_head,
                real_artifact=real_artifact,
                pred_artifact=pred_artifact,
                pair_ids=split["test_pair_ids"],
                device=args.device,
                batch_size=args.batch_size,
            )
            epoch_checkpoint = Path(epoch_checkpoint_template.format(epoch=epoch))
            save_checkpoint(
                epoch_checkpoint,
                cost_head=cost_head,
                args=args,
                epoch=epoch,
                val_metrics=val_metrics,
            )
            val_spearman = val_metrics.get("spearman")
            if val_spearman is not None and val_spearman > best_val_spearman + 1e-12:
                best_val_spearman = float(val_spearman)
                save_checkpoint(
                    best_checkpoint,
                    cost_head=cost_head,
                    args=args,
                    epoch=epoch,
                    val_metrics=val_metrics,
                )
            elif not best_checkpoint.exists():
                save_checkpoint(
                    best_checkpoint,
                    cost_head=cost_head,
                    args=args,
                    epoch=epoch,
                    val_metrics=val_metrics,
                )

            epoch_record = {
                "epoch": epoch,
                "collected": epoch_collected,
                "loss_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
                "loss_first": epoch_losses[0] if epoch_losses else None,
                "loss_last": epoch_losses[-1] if epoch_losses else None,
                "val_predicted": val_metrics,
                "test_predicted_probe": test_probe_metrics,
                "checkpoint_path": str(epoch_checkpoint),
                "elapsed_seconds": float(time.time() - epoch_start),
            }
            epoch_logs.append(epoch_record)
            print(
                f"[epoch={epoch}] collected={epoch_collected} "
                f"loss_mean={epoch_record['loss_mean']} "
                f"val_spearman={val_metrics.get('spearman')} "
                f"test_probe_spearman={test_probe_metrics.get('spearman')} "
                f"elapsed={epoch_record['elapsed_seconds']:.1f}s"
            )
    finally:
        env.close()

    total_elapsed = time.time() - started
    payload = {
        "args": vars(args),
        "split": split,
        "train_pair_ids_used": train_pair_ids,
        "epoch_logs": epoch_logs,
        "pair_logs": all_pair_logs,
        "best_checkpoint_path": str(best_checkpoint),
        "elapsed_seconds": float(total_elapsed),
    }
    save_json(log_path, payload)
    save_json(
        summary_path,
        {
            "best_checkpoint_path": str(best_checkpoint),
            "log_path": str(log_path),
            "epochs": epoch_logs,
            "elapsed_seconds": float(total_elapsed),
            "samples_collected": int(sum(item["collected"] for item in all_pair_logs)),
        },
    )
    print("== CEM-aware training complete ==")
    print(f"best_checkpoint: {best_checkpoint}")
    print(f"log: {log_path}")
    print(f"elapsed_seconds: {total_elapsed:.1f}")
    return payload


def main() -> int:
    """CLI entry point."""
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
