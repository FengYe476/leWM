#!/usr/bin/env python3
"""Generate revised paper figures from real PushT audit artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import stable_worldmodel as swm  # noqa: F401 - registers swm/... envs.
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lewm_audit.eval.oracle_cem import (  # noqa: E402
    block_pose_components,
    rollout_final_state,
)

DEFAULT_RPOOL_PATH = PROJECT_ROOT / "results" / "revision" / "rpool_v1_pusht.json"
DEFAULT_TRACK_A_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
DEFAULT_TRACK_A_LATENTS_PATH = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_predicted_latents.pt"
DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
DEFAULT_FIG1 = PROJECT_ROOT / "paper" / "figures" / "fig1_real_data.pdf"
DEFAULT_ATTRIBUTION = PROJECT_ROOT / "paper" / "figures" / "fig_new_a_attribution.pdf"

SUBSET_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)
SUBSET_LABELS = {
    "invisible_quadrant": "Invisible quadrant",
    "sign_reversal": "Sign reversal",
    "latent_favorable": "Latent-favorable",
    "v1_favorable": "V1-favorable",
    "ordinary": "Ordinary",
}
SUBSET_COLORS = {
    "invisible_quadrant": "#d62728",
    "sign_reversal": "#ff7f0e",
    "latent_favorable": "#1f77b4",
    "v1_favorable": "#9467bd",
    "ordinary": "#2ca02c",
}
SUBSET_MARKERS = {
    "invisible_quadrant": "o",
    "sign_reversal": "s",
    "latent_favorable": "^",
    "v1_favorable": "D",
    "ordinary": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpool-path", type=Path, default=DEFAULT_RPOOL_PATH)
    parser.add_argument("--track-a-path", type=Path, default=DEFAULT_TRACK_A_PATH)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--track-a-latents-path", type=Path, default=DEFAULT_TRACK_A_LATENTS_PATH)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--fig1-output", type=Path, default=DEFAULT_FIG1)
    parser.add_argument("--attribution-output", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument(
        "--pair-id",
        type=int,
        default=None,
        help="Representative pair to use for Fig. 1. Defaults to the top eligible ordinary pair.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_to_numpy(value: Any, *, dtype: np.dtype = np.float64) -> np.ndarray:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor, got {type(value).__name__}")
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return clean_float(spearmanr(x, y).statistic)


def pca_project_with_goal(candidates: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2D PCA on candidate latents plus the goal and return projected coordinates."""

    candidates = np.asarray(candidates, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64).reshape(1, -1)
    if candidates.ndim != 2 or goal.shape[1] != candidates.shape[1]:
        raise ValueError(f"Bad PCA shapes: candidates={candidates.shape}, goal={goal.shape}")

    points = np.vstack([candidates, goal])
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    return projected[:-1], projected[-1]


def values_for_pair(track_pair: dict[str, Any]) -> dict[str, np.ndarray]:
    actions = track_pair.get("actions", [])
    return {
        "action_id": np.asarray([int(action["action_id"]) for action in actions], dtype=np.int64),
        "C_model": np.asarray([float(action["C_model"]) for action in actions], dtype=np.float64),
        "C_real_state": np.asarray(
            [float(action["C_real_state"]) for action in actions], dtype=np.float64
        ),
        "success": np.asarray([bool(action["success"]) for action in actions], dtype=bool),
    }


def endpoint_metrics_by_pair(track_pairs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for pair in track_pairs:
        pair_id = int(pair["pair_id"])
        values = values_for_pair(pair)
        out[pair_id] = {
            "Rendpoint": spearman_corr(values["C_model"], values["C_real_state"]),
            "n_endpoint_records": int(len(values["C_model"])),
            "has_physical_terminal_xy": any(
                key in pair or any(key in action for action in pair.get("actions", []))
                for key in ("block_x", "block_y", "terminal_block_x", "terminal_block_y", "x", "y")
            ),
        }
    return out


def primary_subset(subsets: list[str]) -> str:
    for subset in SUBSET_ORDER:
        if subset in subsets:
            return subset
    raise ValueError(f"Pair has no recognized subset: {subsets}")


def pair_sort_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["Rendpoint"]),
        float(row["Rpool_V1_effective"]),
        float(row["pool_success_mass"]),
        -abs(float(row["Rpool_Cmodel"])),
    )


def find_representative_candidates(
    rpool_rows: list[dict[str, Any]],
    endpoint_metrics: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rpool_rows:
        pair_id = int(row["pair_id"])
        rendpoint = endpoint_metrics[pair_id]["Rendpoint"]
        rpool_v1 = row.get("Rpool_V1")
        if (
            rendpoint is not None
            and rendpoint > 0.4
            and "ordinary" in row.get("subsets", [])
            and clean_float(row.get("Rpool_Cmodel")) is not None
            and float(row["Rpool_Cmodel"]) < 0.05
            and rpool_v1 is not None
            and float(rpool_v1) > 0.5
            and float(row.get("pool_success_mass", 0.0)) > 0.05
        ):
            candidates.append(
                {
                    **row,
                    "Rendpoint": float(rendpoint),
                    "primary_subset": primary_subset(row["subsets"]),
                }
            )
    return sorted(candidates, key=pair_sort_score, reverse=True)


def print_candidates(candidates: list[dict[str, Any]]) -> None:
    print("Top eligible ordinary representatives:")
    if not candidates:
        print("  none")
        return
    for rank, row in enumerate(candidates[:3], start=1):
        print(
            "  "
            f"{rank}. pair_id={int(row['pair_id'])} "
            f"cell={row['cell']} "
            f"R_endpoint={row['Rendpoint']:.3f} "
            f"R_pool(C_model)={float(row['Rpool_Cmodel']):.3f} "
            f"R_pool(V1)={float(row['Rpool_V1']):.3f} "
            f"pool_success_mass={float(row['pool_success_mass']):.3f}"
        )


def load_endpoint_latents(path: Path, pair_id: int) -> tuple[np.ndarray, np.ndarray]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    pair_ids = tensor_to_numpy(artifact["pair_id"], dtype=np.int64)
    action_ids = tensor_to_numpy(artifact["action_id"], dtype=np.int64)
    mask = pair_ids == int(pair_id)
    if int(mask.sum()) != 80:
        raise ValueError(f"Expected 80 endpoint latents for pair {pair_id}, found {int(mask.sum())}")
    order = np.argsort(action_ids[mask])
    z_pred = tensor_to_numpy(artifact["z_predicted"], dtype=np.float64)[mask][order]
    z_goal = tensor_to_numpy(artifact["z_goal_pred"], dtype=np.float64)[mask][order][0]
    return z_pred, z_goal


def load_pool(pool_dir: Path, pair_id: int) -> dict[str, np.ndarray]:
    pool = load_pool_artifact(pool_dir, pair_id)
    return {
        "z_pred": tensor_to_numpy(pool["z_pred"], dtype=np.float64),
        "z_goal": tensor_to_numpy(pool["z_goal"], dtype=np.float64),
        "C_model": tensor_to_numpy(pool["default_costs"], dtype=np.float64),
        "V1": tensor_to_numpy(pool["v1_hinge_costs"], dtype=np.float64),
        "C_real_state": tensor_to_numpy(pool["c_real_state"], dtype=np.float64),
        "block_pos_dist": tensor_to_numpy(pool["block_pos_dist"], dtype=np.float64),
        "success": tensor_to_numpy(pool["success"], dtype=np.float64).astype(bool),
        "raw_actions": tensor_to_numpy(pool["raw_actions"], dtype=np.float32),
        "metadata": pool.get("metadata", {}),
        "pair_spec": pool.get("pair_spec", {}),
    }


def load_pool_artifact(pool_dir: Path, pair_id: int) -> dict[str, Any]:
    return torch.load(pool_dir / f"pair_{pair_id}.pt", map_location="cpu", weights_only=False)


def load_pair_states(pairs_path: Path, pair_spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pairs_data = load_json(pairs_path)
    dataset_path = Path(pairs_data["metadata"]["dataset_path"]).expanduser()
    rows = [int(pair_spec["start_row"]), int(pair_spec["goal_row"])]
    with h5py.File(dataset_path, "r") as handle:
        states = np.asarray(handle["state"][rows], dtype=np.float32)
    return states[0], states[1]


def replay_pool_block_xy(
    *,
    raw_actions_batch: np.ndarray,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_actions_batch = np.asarray(raw_actions_batch, dtype=np.float32)
    block_xy = np.empty((raw_actions_batch.shape[0], 2), dtype=np.float64)
    block_pos_dist = np.empty((raw_actions_batch.shape[0],), dtype=np.float64)

    env = gym.make("swm/PushT-v1")
    try:
        for idx, raw_actions in enumerate(raw_actions_batch):
            terminal_state = rollout_final_state(
                env,
                initial_state,
                goal_state,
                raw_actions,
                seed=int(seed_base) + int(idx),
            )
            block_xy[idx] = np.asarray(terminal_state[2:4], dtype=np.float64)
            block_pos_dist[idx] = float(block_pose_components(terminal_state, goal_state)["block_pos_dist"])
    finally:
        if hasattr(env, "close"):
            env.close()
    return block_xy, block_pos_dist


def make_norm(values: np.ndarray) -> Normalize:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Cannot build color norm for all-nonfinite values")
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def style_latent_axis(ax, title: str, annotation: str) -> None:
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.92},
    )
    ax.grid(True, color="0.90", linewidth=0.6)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def style_physical_axis(ax, title: str, annotation: str) -> None:
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Terminal block x (px)")
    ax.set_ylabel("Terminal block y (px)")
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.92},
    )
    ax.grid(True, color="0.90", linewidth=0.6)
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_latent_scatter(
    ax,
    coords: np.ndarray,
    goal: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    annotation: str,
    cmap: str,
) -> Any:
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=cmap,
        norm=make_norm(values),
        s=34,
        linewidth=0.25,
        edgecolor="black",
        alpha=0.92,
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        marker="*",
        s=230,
        color="gold",
        edgecolor="black",
        linewidth=0.8,
        zorder=10,
        label="Goal",
    )
    style_latent_axis(ax, title, annotation)
    return scatter


def plot_physical_scatter(
    ax,
    coords: np.ndarray,
    goal: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    annotation: str,
    cmap: str,
) -> Any:
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=cmap,
        norm=make_norm(values),
        s=34,
        linewidth=0.25,
        edgecolor="black",
        alpha=0.92,
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        marker="*",
        s=230,
        color="gold",
        edgecolor="black",
        linewidth=0.8,
        zorder=10,
        label="Goal",
    )
    style_physical_axis(ax, title, annotation)
    return scatter


def generate_fig1(
    *,
    track_pair: dict[str, Any],
    selected: dict[str, Any],
    endpoint_latents_path: Path,
    pool_dir: Path,
    pairs_path: Path,
    output: Path,
) -> dict[str, Any]:
    pair_id = int(selected["pair_id"])
    endpoint_values = values_for_pair(track_pair)
    endpoint_z, endpoint_goal = load_endpoint_latents(endpoint_latents_path, pair_id)
    endpoint_coords, endpoint_goal_coord = pca_project_with_goal(endpoint_z, endpoint_goal)

    pool = load_pool(pool_dir, pair_id)
    initial_state, goal_state = load_pair_states(pairs_path, pool["pair_spec"])
    seed_base = int(pool["metadata"].get("seed", 0)) + pair_id * 100_000
    pool_coords, replayed_block_pos_dist = replay_pool_block_xy(
        raw_actions_batch=pool["raw_actions"],
        initial_state=initial_state,
        goal_state=goal_state,
        seed_base=seed_base,
    )
    stored_block_pos_dist = pool["block_pos_dist"]
    max_block_pos_dist_error = float(np.max(np.abs(replayed_block_pos_dist - stored_block_pos_dist)))
    if max_block_pos_dist_error > 1e-3:
        raise RuntimeError(
            "Replayed pool terminal positions do not match stored block_pos_dist: "
            f"max_abs_error={max_block_pos_dist_error:.6g}"
        )
    pool_goal_coord = np.asarray(goal_state[2:4], dtype=np.float64)

    cmap = "RdYlGn_r"
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=False)

    scatter_a = plot_latent_scatter(
        axes[0],
        endpoint_coords,
        endpoint_goal_coord,
        endpoint_values["C_model"],
        title="Endpoint Evaluation",
        annotation=rf"$R_{{endpoint}}={float(selected['Rendpoint']):.3f}$",
        cmap=cmap,
    )
    scatter_b = plot_physical_scatter(
        axes[1],
        pool_coords,
        pool_goal_coord,
        pool["C_model"],
        title="CEM Pool (learned cost)",
        annotation=rf"$R_{{pool}}(C_{{model}})={float(selected['Rpool_Cmodel']):.3f}$",
        cmap=cmap,
    )
    scatter_c = plot_physical_scatter(
        axes[2],
        pool_coords,
        pool_goal_coord,
        pool["V1"],
        title="CEM Pool (oracle cost)",
        annotation=rf"$R_{{pool}}(V1)={float(selected['Rpool_V1']):.3f}$",
        cmap=cmap,
    )

    for ax, scatter, label in zip(
        axes,
        (scatter_a, scatter_b, scatter_c),
        ("Learned cost", "Learned cost", "V1 hinge cost"),
        strict=True,
    ):
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(label, fontsize=10)
        colorbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Endpoint-planning decoupling on PushT pair {pair_id}", fontsize=14, y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "pool_xy_span_x": float(np.ptp(pool_coords[:, 0])),
        "pool_xy_span_y": float(np.ptp(pool_coords[:, 1])),
        "pool_xy_goal_x": float(pool_goal_coord[0]),
        "pool_xy_goal_y": float(pool_goal_coord[1]),
        "replay_max_block_pos_dist_error": max_block_pos_dist_error,
        "pool_physical_x_spearman_Cmodel": spearman_corr(pool_coords[:, 0], pool["C_model"]),
        "pool_physical_y_spearman_Cmodel": spearman_corr(pool_coords[:, 1], pool["C_model"]),
        "pool_physical_goal_dist_spearman_Cmodel": spearman_corr(
            np.linalg.norm(pool_coords - pool_goal_coord[None, :], axis=1),
            pool["C_model"],
        ),
        "pool_physical_goal_dist_spearman_V1": spearman_corr(
            np.linalg.norm(pool_coords - pool_goal_coord[None, :], axis=1),
            pool["V1"],
        ),
    }


def generate_attribution(rpool_rows: list[dict[str, Any]], output: Path) -> None:
    rows = [
        {
            **row,
            "primary_subset": primary_subset(row["subsets"]),
            "Rpool_V1_plot": float(row.get("Rpool_V1_effective", 0.0)),
        }
        for row in rpool_rows
    ]

    fig, ax = plt.subplots(figsize=(6.0, 5.5), constrained_layout=False)
    all_values = np.asarray(
        [float(row["Rpool_Cmodel"]) for row in rows] + [float(row["Rpool_V1_plot"]) for row in rows],
        dtype=np.float64,
    )
    pad = 0.06
    lower = float(np.floor((np.nanmin(all_values) - pad) * 10.0) / 10.0)
    upper = float(np.ceil((np.nanmax(all_values) + pad) * 10.0) / 10.0)
    lower = min(lower, -0.7)
    upper = max(upper, 1.05)

    ax.plot([lower, upper], [lower, upper], linestyle="--", color="0.45", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="0.82", linewidth=0.9, zorder=0)
    ax.axvline(0.0, color="0.82", linewidth=0.9, zorder=0)

    handles: list[Line2D] = []
    for subset in SUBSET_ORDER:
        subset_rows = [row for row in rows if row["primary_subset"] == subset]
        if not subset_rows:
            continue
        ax.scatter(
            [float(row["Rpool_Cmodel"]) for row in subset_rows],
            [float(row["Rpool_V1_plot"]) for row in subset_rows],
            s=46,
            marker=SUBSET_MARKERS[subset],
            color=SUBSET_COLORS[subset],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.90,
            zorder=3,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker=SUBSET_MARKERS[subset],
                linestyle="none",
                markersize=7,
                markerfacecolor=SUBSET_COLORS[subset],
                markeredgecolor="black",
                markeredgewidth=0.45,
                label=f"{SUBSET_LABELS[subset]} (n={len(subset_rows)})",
            )
        )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"$R_{pool}(C_{model})$")
    ax.set_ylabel(r"$R_{pool}(V1)$")
    ax.set_title("V1 Attribution of CEM-Pool Ranking", fontsize=13, pad=8)
    ax.grid(True, color="0.92", linewidth=0.6)
    ax.tick_params(labelsize=9)
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def main() -> int:
    args = parse_args()
    args.rpool_path = args.rpool_path.expanduser().resolve()
    args.track_a_path = args.track_a_path.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.track_a_latents_path = args.track_a_latents_path.expanduser().resolve()
    args.pool_dir = args.pool_dir.expanduser().resolve()
    args.fig1_output = args.fig1_output.expanduser().resolve()
    args.attribution_output = args.attribution_output.expanduser().resolve()

    set_plot_style()

    rpool_data = load_json(args.rpool_path)
    rpool_rows = rpool_data["per_pair"]
    track_data = load_json(args.track_a_path)
    track_pairs = {int(pair["pair_id"]): pair for pair in track_data["pairs"]}
    endpoint_metrics = endpoint_metrics_by_pair(track_data["pairs"])
    candidates = find_representative_candidates(rpool_rows, endpoint_metrics)
    print_candidates(candidates)

    if args.pair_id is None:
        if not candidates:
            raise RuntimeError("No eligible representative pair found.")
        selected = candidates[0]
    else:
        selected_rows = [row for row in rpool_rows if int(row["pair_id"]) == int(args.pair_id)]
        if not selected_rows:
            raise RuntimeError(f"Unknown pair_id={args.pair_id}")
        selected = {
            **selected_rows[0],
            "Rendpoint": endpoint_metrics[int(args.pair_id)]["Rendpoint"],
            "primary_subset": primary_subset(selected_rows[0]["subsets"]),
        }

    pair_id = int(selected["pair_id"])
    endpoint_has_xy = endpoint_metrics[pair_id]["has_physical_terminal_xy"]
    print(
        f"Selected pair_id={pair_id}; endpoint physical terminal xy present={endpoint_has_xy}. "
        "Using PCA of Track A predicted endpoint latents for Panel A and replayed physical "
        "terminal block coordinates for Panels B/C."
    )

    fig1_diagnostics = generate_fig1(
        track_pair=track_pairs[pair_id],
        selected=selected,
        endpoint_latents_path=args.track_a_latents_path,
        pool_dir=args.pool_dir,
        pairs_path=args.pairs_path,
        output=args.fig1_output,
    )
    print(f"Wrote {args.fig1_output}")
    print(
        "Fig. 1 pool replay diagnostics: "
        f"max_block_pos_dist_error={fig1_diagnostics['replay_max_block_pos_dist_error']:.3g}; "
        f"x_span={fig1_diagnostics['pool_xy_span_x']:.2f}px; "
        f"y_span={fig1_diagnostics['pool_xy_span_y']:.2f}px; "
        f"rho(goal_dist,C_model)={fig1_diagnostics['pool_physical_goal_dist_spearman_Cmodel']:.3f}; "
        f"rho(goal_dist,V1)={fig1_diagnostics['pool_physical_goal_dist_spearman_V1']:.3f}"
    )

    generate_attribution(rpool_rows, args.attribution_output)
    print(f"Wrote {args.attribution_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
