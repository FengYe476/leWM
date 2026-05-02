#!/usr/bin/env python3
"""Render per-trajectory figures for the Track A visualization pass."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from lewm_audit.diagnostics.three_cost import spearman_corr
from lewm_audit.eval.oracle_cem import (
    ANGLE_SUCCESS_THRESHOLD_RAD,
    BLOCK_SUCCESS_THRESHOLD_PX,
    block_pose_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "results" / "phase1" / "per_trajectory"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "phase1" / "figures" / "per_trajectory"
DEFAULT_TRACK_A_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
PAIR_ORDER = [
    (80, "D3xR1", "F4"),
    (74, "D3xR0", "F5"),
    (6, "D0xR1", "F6"),
    (93, "D3xR3", "F7"),
    (20, "D0xR3", "sign-reversal"),
]
VARIANT_ORDER = ("latent", "V3", "V1", "V2")
VARIANT_COLORS = {
    "latent": "#666666",
    "V3": "#1f77b4",
    "V1": "#2ca02c",
    "V2": "#ff7f0e",
}
SOURCE_ORDER = ("data", "smooth_random", "CEM_early", "CEM_late")
SOURCE_COLORS = {
    "data": "#1f77b4",
    "smooth_random": "#ff7f0e",
    "CEM_early": "#2ca02c",
    "CEM_late": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--track-a-path", type=Path, default=DEFAULT_TRACK_A_PATH)
    return parser.parse_args()


def load_trajectory(trajectory_dir: Path, pair_id: int, cell: str, variant: str) -> dict:
    path = trajectory_dir / f"{pair_id}_{cell}_{variant}.json"
    return json.loads(path.read_text())


def load_all_trajectories(trajectory_dir: Path) -> dict[tuple[int, str], dict[str, dict]]:
    data = {}
    for pair_id, cell, _ in PAIR_ORDER:
        data[(pair_id, cell)] = {
            variant: load_trajectory(trajectory_dir, pair_id, cell, variant)
            for variant in VARIANT_ORDER
        }
    return data


def pair_records(track_a_data: dict, pair_id: int) -> list[dict]:
    pair = next(pair for pair in track_a_data["pairs"] if int(pair["pair_id"]) == pair_id)
    return pair["actions"]


def infer_workspace(trajectories: dict[str, dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = []
    for traj in trajectories.values():
        points.append(np.asarray(traj["block_xy"], dtype=np.float64))
        points.append(np.asarray(traj["agent_xy"], dtype=np.float64))
        points.append(np.asarray(traj["goal_state"], dtype=np.float64)[2:4][None, :])
    stacked = np.concatenate(points, axis=0)
    lo = np.nanmin(stacked, axis=0)
    hi = np.nanmax(stacked, axis=0)
    if np.all(lo >= -5.0) and np.all(hi <= 517.0):
        return (0.0, 512.0), (0.0, 512.0)
    span = np.maximum(hi - lo, 1.0)
    pad = np.maximum(30.0, 0.08 * span)
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (
        float(lo[1] - pad[1]),
        float(hi[1] + pad[1]),
    )


def add_time_graded_line(ax, xy: np.ndarray, color: str, label: str) -> None:
    xy = np.asarray(xy, dtype=np.float64)
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    rgba = np.tile(np.asarray(to_rgba(color)), (len(segments), 1))
    rgba[:, 3] = np.linspace(0.25, 1.0, len(segments))
    collection = LineCollection(
        segments,
        colors=rgba,
        linewidths=2.5,
        label=label,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)
    ax.scatter(xy[-1, 0], xy[-1, 1], s=28, color=color, marker="x", linewidths=1.6)


def state_metric_series(traj: dict) -> dict[str, np.ndarray]:
    goal_state = np.asarray(traj["goal_state"], dtype=np.float64)
    metrics = [
        block_pose_components(np.asarray(state, dtype=np.float64), goal_state)
        for state in traj["states"]
    ]
    return {
        "block_pos_dist": np.asarray([metric["block_pos_dist"] for metric in metrics]),
        "angle_dist": np.asarray([metric["angle_dist"] for metric in metrics]),
        "success": np.asarray([metric["success"] for metric in metrics], dtype=bool),
    }


def render_block_trajectory(
    *,
    pair_id: int,
    cell: str,
    trajectories: dict[str, dict],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    init_xy = np.asarray(next(iter(trajectories.values()))["init_state"], dtype=np.float64)[2:4]
    goal_xy = np.asarray(next(iter(trajectories.values()))["goal_state"], dtype=np.float64)[2:4]

    ax.add_patch(
        Circle(
            goal_xy,
            radius=BLOCK_SUCCESS_THRESHOLD_PX,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.35,
        )
    )
    for variant in VARIANT_ORDER:
        traj = trajectories[variant]
        label = (
            f"{variant} success={traj['final_success']} "
            f"C={float(traj['C_real_state_final']):.1f}"
        )
        add_time_graded_line(
            ax,
            np.asarray(traj["block_xy"], dtype=np.float64),
            VARIANT_COLORS[variant],
            label,
        )

    ax.scatter(init_xy[0], init_xy[1], marker="o", s=70, color="black", label="init")
    ax.scatter(goal_xy[0], goal_xy[1], marker="*", s=130, color="black", label="goal")
    xlim, ylim = infer_workspace(trajectories)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("block x")
    ax.set_ylabel("block y")
    ax.set_title(f"Pair {pair_id} - {cell} - block trajectories")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_cost_panel(
    *,
    pair_id: int,
    cell: str,
    trajectories: dict[str, dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    steps = np.arange(len(next(iter(trajectories.values()))["states"]))

    for variant in VARIANT_ORDER:
        metrics = state_metric_series(trajectories[variant])
        axes[0].plot(
            steps,
            metrics["block_pos_dist"],
            color=VARIANT_COLORS[variant],
            linewidth=2.0,
            label=variant,
        )
        axes[1].plot(
            steps,
            metrics["angle_dist"],
            color=VARIANT_COLORS[variant],
            linewidth=2.0,
            label=variant,
        )

    axes[0].axhline(BLOCK_SUCCESS_THRESHOLD_PX, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("block_pos_dist over time")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("pixels")
    axes[0].grid(True, linewidth=0.4, alpha=0.25)

    axes[1].axhline(ANGLE_SUCCESS_THRESHOLD_RAD, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("angle_dist over time")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("radians")
    axes[1].grid(True, linewidth=0.4, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    x = np.arange(len(VARIANT_ORDER))
    block_values = [
        float(trajectories[variant]["block_pos_dist_final"]) for variant in VARIANT_ORDER
    ]
    angle_values = [
        float(trajectories[variant]["angle_dist_final"]) for variant in VARIANT_ORDER
    ]
    width = 0.35
    axes[2].bar(
        x - width / 2,
        block_values,
        width=width,
        color=[VARIANT_COLORS[variant] for variant in VARIANT_ORDER],
        alpha=0.75,
        label="block_pos_dist",
    )
    axes2 = axes[2].twinx()
    axes2.bar(
        x + width / 2,
        angle_values,
        width=width,
        color=[VARIANT_COLORS[variant] for variant in VARIANT_ORDER],
        alpha=0.35,
        hatch="//",
        label="angle_dist",
    )
    axes[2].axhline(BLOCK_SUCCESS_THRESHOLD_PX, color="black", linestyle="--", linewidth=1.0)
    axes2.axhline(ANGLE_SUCCESS_THRESHOLD_RAD, color="black", linestyle=":", linewidth=1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(VARIANT_ORDER)
    axes[2].set_ylabel("block_pos_dist final (px)")
    axes2.set_ylabel("angle_dist final (rad)")
    axes[2].set_title("final-state metrics")
    axes[2].grid(True, axis="y", linewidth=0.4, alpha=0.25)
    handles = [
        Line2D([0], [0], color="black", linewidth=6, alpha=0.75, label="block_pos_dist"),
        Line2D([0], [0], color="black", linewidth=6, alpha=0.35, label="angle_dist"),
    ]
    axes[2].legend(handles=handles, loc="upper right", fontsize=8)

    fig.suptitle(f"Pair {pair_id} - {cell} - cost and success summary", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def grouped_by_source(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)
    return grouped


def render_sign_reversal_scatter(
    *,
    pair_id: int,
    cell: str,
    records: list[dict],
    output_path: Path,
) -> None:
    x = np.asarray([float(record["C_real_z"]) for record in records], dtype=np.float64)
    y = np.asarray([float(record["C_real_state"]) for record in records], dtype=np.float64)
    rho = spearman_corr(x, y)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for source in SOURCE_ORDER:
        group = [record for record in records if record["source"] == source]
        if not group:
            continue
        for success, marker, size, label_suffix in (
            (False, "o", 36, "failure"),
            (True, "*", 90, "success"),
        ):
            sub = [record for record in group if bool(record["success"]) is success]
            if not sub:
                continue
            ax.scatter(
                [float(record["C_real_z"]) for record in sub],
                [float(record["C_real_state"]) for record in sub],
                color=SOURCE_COLORS[source],
                marker=marker,
                s=size,
                alpha=0.82,
                edgecolors="black" if success else "none",
                linewidths=0.45,
                label=f"{source} {label_suffix}",
            )

    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.2, alpha=0.75)

    ax.text(
        0.03,
        0.96,
        f"Spearman rho = {rho:.3f}" if rho is not None else "Spearman rho = NA",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_title(f"Pair {pair_id} (cell {cell}) - encoder cost vs physical cost")
    ax.set_xlabel("C_real_z")
    ax.set_ylabel("C_real_state")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncols=2)
    success_count = sum(bool(record["success"]) for record in records)
    extra = "\nNo success stars appear: all 80 actions fail." if success_count == 0 else ""
    fig.text(
        0.5,
        0.02,
        "Negative slope: encoder ranks states inversely to physical proximity." + extra,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.08, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def source_means(records: list[dict], key: str) -> list[float]:
    grouped = grouped_by_source(records)
    return [
        float(np.mean([float(record[key]) for record in grouped[source]]))
        if grouped[source]
        else math.nan
        for source in SOURCE_ORDER
    ]


def source_success_rates(records: list[dict]) -> list[float]:
    grouped = grouped_by_source(records)
    rates = []
    for source in SOURCE_ORDER:
        group = grouped[source]
        rates.append(100.0 * sum(bool(record["success"]) for record in group) / len(group))
    return rates


def render_cost_cascade(
    *,
    pair_id: int,
    cell: str,
    records: list[dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    x = np.arange(len(SOURCE_ORDER))
    colors = [SOURCE_COLORS[source] for source in SOURCE_ORDER]
    panels = [
        ("mean C_model", source_means(records, "C_model"), "C_model"),
        ("mean C_real_z", source_means(records, "C_real_z"), "C_real_z"),
        ("success rate", source_success_rates(records), "success rate (%)"),
    ]
    for ax, (title, values, ylabel) in zip(axes, panels, strict=True):
        ax.bar(x, values, color=colors, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(SOURCE_ORDER, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.25)
    axes[2].set_ylim(0, 100)
    fig.suptitle(f"Pair {pair_id} - {cell} - latent cost cascade", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.trajectory_dir = args.trajectory_dir.expanduser().resolve()
    args.figure_dir = args.figure_dir.expanduser().resolve()
    args.track_a_path = args.track_a_path.expanduser().resolve()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_all_trajectories(args.trajectory_dir)
    track_a_data = json.loads(args.track_a_path.read_text())
    outputs = []

    for pair_id, cell, anchor in PAIR_ORDER:
        pair_traj = trajectories[(pair_id, cell)]
        block_path = args.figure_dir / f"{pair_id}_{cell}_block_trajectories.png"
        panel_path = args.figure_dir / f"{pair_id}_{cell}_cost_panel.png"
        render_block_trajectory(
            pair_id=pair_id,
            cell=cell,
            trajectories=pair_traj,
            output_path=block_path,
        )
        render_cost_panel(
            pair_id=pair_id,
            cell=cell,
            trajectories=pair_traj,
            output_path=panel_path,
        )
        outputs.extend([block_path, panel_path])

        records = pair_records(track_a_data, pair_id)
        if anchor == "sign-reversal":
            scatter_path = args.figure_dir / f"{pair_id}_signreversal_scatter.png"
            render_sign_reversal_scatter(
                pair_id=pair_id,
                cell=cell,
                records=records,
                output_path=scatter_path,
            )
            outputs.append(scatter_path)
        else:
            cascade_path = args.figure_dir / f"{pair_id}_{cell}_cost_cascade.png"
            render_cost_cascade(
                pair_id=pair_id,
                cell=cell,
                records=records,
                output_path=cascade_path,
            )
            outputs.append(cascade_path)

    print("== Rendered per-trajectory figures ==")
    for path in outputs:
        print(path.relative_to(PROJECT_ROOT))
    print(f"figure_count: {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
