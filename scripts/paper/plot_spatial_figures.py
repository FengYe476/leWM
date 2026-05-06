#!/usr/bin/env python3
"""Build spatial latent-space figures for the revised paper."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"

PUSHT_RPOOL_PATH = ROOT / "results" / "revision" / "rpool_v1_pusht.json"
PUSHT_POOL_DIR = ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
CUBE_RPOOL_PATH = ROOT / "results" / "revision" / "rpool_v1_cube.json"
CUBE_POOL_DIR = ROOT / "results" / "revision" / "cube_full_proj_pools"
MPPI_ANALYSIS_PATH = ROOT / "results" / "revision" / "mppi_pool_analysis.json"

FIG5_PATH = FIGURE_DIR / "fig5_enhanced.pdf"
MPPI_PATH = FIGURE_DIR / "fig_mppi_comparison.pdf"
CROSS_ENV_PATH = FIGURE_DIR / "fig_cross_env.pdf"

SUBSET_COLORS = {
    "invisible_quadrant": "#d62728",
    "sign_reversal": "#ff7f0e",
    "latent_favorable": "#1f77b4",
    "v1_favorable": "#9467bd",
    "ordinary": "#2ca02c",
}
SUBSET_LABELS = {
    "invisible_quadrant": "Invisible quadrant",
    "sign_reversal": "Sign reversal",
    "latent_favorable": "Latent-favorable",
    "v1_favorable": "V1-favorable",
    "ordinary": "Ordinary",
}
SUBSET_PRIORITY = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)
SUBSET_MARKERS = {
    "invisible_quadrant": "o",
    "sign_reversal": "s",
    "latent_favorable": "^",
    "v1_favorable": "D",
    "ordinary": "P",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pool(path: Path) -> dict[str, Any]:
    return torch.load(path, weights_only=False, map_location="cpu")


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def primary_subset(subsets: list[str]) -> str:
    subset_set = set(subsets)
    for subset in SUBSET_PRIORITY:
        if subset in subset_set:
            return subset
    return "ordinary"


def pair_pool_path(pair_id: int) -> Path:
    return PUSHT_POOL_DIR / f"pair_{pair_id}.pt"


def cube_pair_id_from_path(path: str) -> int:
    match = re.search(r"pair_(\d+)_", Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse Cube pair id from {path}")
    return int(match.group(1))


def valid_float_values(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            output.append(numeric)
    return output


def plot_attribution_scatter(ax: plt.Axes, pusht_rpool: dict[str, Any]) -> None:
    rows = pusht_rpool["per_pair"]
    by_subset = {subset: [] for subset in SUBSET_PRIORITY}
    for row in rows:
        subset = primary_subset(row.get("subsets", []))
        by_subset[subset].append(row)

    plot_order = (
        "ordinary",
        "sign_reversal",
        "invisible_quadrant",
        "latent_favorable",
        "v1_favorable",
    )
    marker_sizes = {
        "ordinary": 34,
        "sign_reversal": 42,
        "invisible_quadrant": 46,
        "latent_favorable": 48,
        "v1_favorable": 54,
    }
    for zorder, subset in enumerate(plot_order, start=2):
        subset_rows = by_subset[subset]
        if not subset_rows:
            continue
        x = [float(row["Rpool_Cmodel_effective"]) for row in subset_rows]
        y = [float(row["Rpool_V1_effective"]) for row in subset_rows]
        ax.scatter(
            x,
            y,
            s=marker_sizes[subset],
            marker=SUBSET_MARKERS[subset],
            color=SUBSET_COLORS[subset],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            label=f"{SUBSET_LABELS[subset]} ({len(subset_rows)})",
            zorder=zorder,
        )

    ax.plot([-0.35, 1.05], [-0.35, 1.05], color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xlim(-0.35, 1.05)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"$R_{\mathrm{pool}}(C_{\mathrm{model}})$")
    ax.set_ylabel(r"$R_{\mathrm{pool}}(C_{\mathrm{V1}})$")
    ax.set_title("(a) V1 attribution across PushT pairs", loc="left", fontsize=10)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handletextpad=0.2,
        borderaxespad=0.0,
    )
    ax.grid(color="0.92", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    summary = pusht_rpool["summary"]["overall"]
    ax.text(
        0.95,
        0.05,
        "\n".join(
            [
                rf"mean $R_{{pool}}(C_{{V1}})$ = {summary['Rpool_V1_effective']['mean']:.3f}",
                rf"mean $R_{{pool}}(C_{{model}})$ = {summary['Rpool_Cmodel_effective']['mean']:.3f}",
                f"n = {summary['n_pairs']} pairs",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.2",
        zorder=20,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "wheat",
            "edgecolor": "0.75",
            "alpha": 0.5,
        },
    )


def plot_pool_3d(
    ax: plt.Axes,
    pool: dict[str, Any],
    title: str,
    norm: mcolors.Normalize | None = None,
    pca: PCA | None = None,
    show_special_markers: bool = True,
    view: tuple[float, float] = (28, -55),
    pc3_label_mode: str = "text2d",
    title_fontsize: float = 10,
    title_fontweight: str | None = None,
    axis_label_fontsize: float = 9,
    tick_labelsize: float = 7,
) -> tuple[Any, PCA, mcolors.Normalize]:
    z_pred = as_numpy(pool["z_pred"])
    if pca is None:
        pca = PCA(n_components=3, random_state=0)
        z_3d = pca.fit_transform(z_pred)
    else:
        z_3d = pca.transform(z_pred)

    c_real = as_numpy(pool["c_real_state"]).astype(float)
    success = as_numpy(pool["success"]).astype(bool)
    if norm is None:
        norm = mcolors.Normalize(vmin=float(np.nanmin(c_real)), vmax=float(np.nanmax(c_real)))

    failed = ~success
    scatter = ax.scatter(
        z_3d[failed, 0],
        z_3d[failed, 1],
        z_3d[failed, 2],
        c=c_real[failed],
        cmap="RdYlGn_r",
        norm=norm,
        s=18,
        marker="o",
        alpha=0.6,
        linewidth=0.0,
        depthshade=False,
    )
    if success.any():
        ax.scatter(
            z_3d[success, 0],
            z_3d[success, 1],
            z_3d[success, 2],
            c=c_real[success],
            cmap="RdYlGn_r",
            norm=norm,
            s=46,
            marker="*",
            alpha=0.78,
            edgecolor="0.2",
            linewidth=0.35,
            depthshade=False,
        )

    if show_special_markers:
        rank1_idx = int(
            pool.get("default_rank1_candidate_index", pool.get("rank1_candidate_index", 0))
        )
        oracle_idx = int(
            pool.get(
                "oracle_best_candidate_index",
                int(np.nanargmin(c_real)),
            )
        )
        ax.scatter(
            z_3d[rank1_idx, 0],
            z_3d[rank1_idx, 1],
            z_3d[rank1_idx, 2],
            marker="D",
            s=110,
            color="none",
            edgecolor="black",
            linewidth=1.6,
            depthshade=False,
            label="Rank-1 selected",
        )
        ax.scatter(
            z_3d[oracle_idx, 0],
            z_3d[oracle_idx, 1],
            z_3d[oracle_idx, 2],
            marker="*",
            s=170,
            color="none",
            edgecolor="#f2c94c",
            linewidth=1.8,
            depthshade=False,
            label="Oracle best",
        )

    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("PC1", labelpad=4, fontsize=axis_label_fontsize)
    ax.set_ylabel("PC2", labelpad=4, fontsize=axis_label_fontsize)
    if pc3_label_mode == "axis":
        ax.set_zlabel("PC3", labelpad=15, fontsize=axis_label_fontsize)
    else:
        ax.set_zlabel("")
        ax.text2D(
            0.93,
            0.55,
            "PC3",
            transform=ax.transAxes,
            fontsize=axis_label_fontsize,
            ha="center",
            va="center",
        )
    ax.set_title(title, loc="left", fontsize=title_fontsize, fontweight=title_fontweight)
    ax.tick_params(axis="x", labelsize=tick_labelsize, pad=3)
    ax.tick_params(axis="y", labelsize=tick_labelsize, pad=3)
    ax.tick_params(axis="z", labelsize=tick_labelsize, pad=8 if pc3_label_mode == "axis" else 5)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)
    return scatter, pca, norm


def build_fig5_enhanced() -> None:
    pusht_rpool = load_json(PUSHT_RPOOL_PATH)
    pool = load_pool(pair_pool_path(51))
    c_real = as_numpy(pool["c_real_state"]).astype(float)
    rank1_idx = int(pool["default_rank1_candidate_index"])
    oracle_idx = int(pool.get("oracle_best_candidate_index", np.nanargmin(c_real)))
    regret = c_real[rank1_idx] - c_real[oracle_idx]

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.32)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_pool = fig.add_subplot(gs[0, 1], projection="3d")

    plot_attribution_scatter(ax_scatter, pusht_rpool)
    scatter_3d, _pca, color_norm = plot_pool_3d(
        ax_pool,
        pool,
        "(b) CEM pool in latent space (pair 51)",
        show_special_markers=True,
    )
    ax_pool.text2D(
        0.5,
        -0.17,
        rf"rank-1 regret = {regret:.1f}",
        transform=ax_pool.transAxes,
        fontsize=9,
        color="0.2",
        ha="center",
        va="top",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    ax_pool.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        markerscale=0.8,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    scalar_mappable = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=color_norm)
    scalar_mappable.set_array(c_real)
    cbar = fig.colorbar(scalar_mappable, ax=ax_pool, shrink=0.6, pad=0.1)
    cbar.set_label(r"$C_{\mathrm{real\_state}}$", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=7)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG5_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {FIG5_PATH.relative_to(ROOT)}")


def choose_mppi_example() -> dict[str, Any]:
    analysis = load_json(MPPI_ANALYSIS_PATH)
    cem_by_pair = {int(row["pair_id"]): row for row in analysis["cem_per_pair"]}
    candidates = []
    for row in analysis["mppi_per_seed"]:
        pair_id = int(row["pair_id"])
        cem_path = pair_pool_path(pair_id)
        mppi_path = Path(row["pool_path"])
        if not mppi_path.exists():
            mppi_path = ROOT / "results" / "revision" / "mppi_pusht_pools" / mppi_path.name
        if pair_id not in cem_by_pair or not cem_path.exists() or not mppi_path.exists():
            continue
        contrast = float(row["pool_Creal_std"]) - float(cem_by_pair[pair_id]["pool_Creal_std"])
        candidates.append((contrast, row, cem_by_pair[pair_id], cem_path, mppi_path))
    if not candidates:
        raise RuntimeError("Could not find a matching CEM/MPPI pair")
    _contrast, mppi_row, cem_row, cem_path, mppi_path = max(candidates, key=lambda item: item[0])
    return {
        "pair_id": int(mppi_row["pair_id"]),
        "seed": int(mppi_row["seed"]),
        "cem_row": cem_row,
        "mppi_row": mppi_row,
        "cem_path": cem_path,
        "mppi_path": mppi_path,
        "summary": analysis["summary"],
    }


def build_mppi_comparison() -> None:
    example = choose_mppi_example()
    cem_pool = load_pool(example["cem_path"])
    mppi_pool = load_pool(example["mppi_path"])
    cem_c = as_numpy(cem_pool["c_real_state"]).astype(float)
    mppi_c = as_numpy(mppi_pool["c_real_state"]).astype(float)
    norm = mcolors.Normalize(
        vmin=float(min(np.nanmin(cem_c), np.nanmin(mppi_c))),
        vmax=float(max(np.nanmax(cem_c), np.nanmax(mppi_c))),
    )

    cem_agg_std = example["summary"]["cem_default"]["pool_Creal_std"]["mean"]
    mppi_agg_std = example["summary"]["mppi_tau_1"]["pool_Creal_std"]["mean"]
    cem_pair_std = float(example["cem_row"]["pool_Creal_std"])
    mppi_pair_std = float(example["mppi_row"]["pool_Creal_std"])

    fig, (ax_cem, ax_mppi) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        subplot_kw={"projection": "3d"},
    )
    scatter, pca, _norm = plot_pool_3d(
        ax_cem,
        cem_pool,
        f"(a) CEM pool (std = {cem_agg_std:.1f})\npair {example['pair_id']} CEM std: {cem_pair_std:.1f}",
        norm=norm,
        show_special_markers=False,
        view=(25, -60),
        pc3_label_mode="axis",
    )
    _scatter_right, _pca, _norm = plot_pool_3d(
        ax_mppi,
        mppi_pool,
        f"(b) MPPI pool (std = {mppi_agg_std:.1f})\npair {example['pair_id']}, seed {example['seed']} MPPI std: {mppi_pair_std:.1f}",
        norm=norm,
        pca=pca,
        show_special_markers=False,
        view=(25, -60),
        pc3_label_mode="axis",
    )

    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.15, top=0.85, wspace=0.05)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.012, 0.7])
    scalar_mappable = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    scalar_mappable.set_array(np.concatenate([cem_c, mppi_c]))
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
    cbar.set_label(r"$C_{\mathrm{real\_state}}$", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=7)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MPPI_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {MPPI_PATH.relative_to(ROOT)}")
    print(
        "MPPI comparison example: "
        f"pair={example['pair_id']}, seed={example['seed']}, "
        f"CEM pair std={cem_pair_std:.3f}, MPPI pair std={mppi_pair_std:.3f}"
    )


def tsne_embedding(values: np.ndarray, seed: int) -> np.ndarray:
    perplexity = min(18, max(5, (len(values) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=seed,
    ).fit_transform(values)


def collect_pusht_goal_latents() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rpool = load_json(PUSHT_RPOOL_PATH)
    rows = sorted(rpool["per_pair"], key=lambda row: int(row["pair_id"]))
    z_goals = []
    r_values = []
    success_masses = []
    for row in rows:
        pair_id = int(row["pair_id"])
        pool = load_pool(pair_pool_path(pair_id))
        z_goals.append(as_numpy(pool["z_goal"]))
        r_values.append(float(row["Rpool_Cmodel_effective"]))
        success_masses.append(float(row["pool_success_mass"]))
    return np.vstack(z_goals), np.array(r_values), np.array(success_masses)


def collect_cube_goal_latents() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rpool = load_json(CUBE_RPOOL_PATH)
    rows_by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in rpool["per_pool"]:
        rows_by_pair.setdefault(int(row["pair_id"]), []).append(row)

    z_goals = []
    r_values = []
    success_masses = []
    for pair_id in sorted(rows_by_pair):
        rows = rows_by_pair[pair_id]
        preferred = [
            row
            for row in rows
            if int(row["dimension"]) == 192 and int(row["projection_seed"]) == 0
        ]
        row_for_goal = preferred[0] if preferred else rows[0]
        pool_path = ROOT / row_for_goal["pool_path"]
        if not pool_path.exists():
            pool_path = CUBE_POOL_DIR / Path(row_for_goal["pool_path"]).name
        pool = load_pool(pool_path)
        z_goals.append(as_numpy(pool["z_goal"]))
        r_vals = valid_float_values([row["Rpool_Cmodel"] for row in rows])
        success_vals = valid_float_values([row["pool_success_mass"] for row in rows])
        r_values.append(float(np.mean(r_vals)) if r_vals else 0.0)
        success_masses.append(float(np.mean(success_vals)) if success_vals else 0.0)
    return np.vstack(z_goals), np.array(r_values), np.array(success_masses)


def plot_goal_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    rpool_values: np.ndarray,
    success_masses: np.ndarray,
    title: str,
    norm: mcolors.Normalize,
    size_scale: float = 1.0,
    title_fontsize: float = 10,
    title_fontweight: str | None = None,
    axis_label_fontsize: float = 10,
    tick_labelsize: float = 8,
    annotation_fontsize: float = 8,
    show_ticks: bool = False,
) -> Any:
    min_size = 30.0
    sizes = size_scale * (min_size + 180 * np.clip(success_masses, 0.0, 1.0))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=rpool_values,
        cmap="RdYlGn",
        norm=norm,
        s=sizes,
        alpha=0.9,
        edgecolors="gray",
        linewidths=0.3,
    )
    ax.axhline(0, color="0.92", linewidth=0.7, zorder=0)
    ax.axvline(0, color="0.92", linewidth=0.7, zorder=0)
    ax.set_title(title, loc="left", fontsize=title_fontsize, fontweight=title_fontweight)
    ax.set_xlabel("t-SNE 1", fontsize=axis_label_fontsize)
    ax.set_ylabel("t-SNE 2", fontsize=axis_label_fontsize)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(labelsize=tick_labelsize)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    mean_r = float(np.nanmean(rpool_values))
    ax.text(
        0.05,
        0.05,
        rf"mean $R_{{pool}}$ = {mean_r:.3f}",
        transform=ax.transAxes,
        fontsize=annotation_fontsize,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    return scatter


def build_cross_env() -> None:
    pusht_z, pusht_r, pusht_success = collect_pusht_goal_latents()
    cube_z, cube_r, cube_success = collect_cube_goal_latents()
    pusht_emb = tsne_embedding(pusht_z, seed=4)
    cube_emb = tsne_embedding(cube_z, seed=8)
    norm = mcolors.Normalize(
        vmin=float(min(np.nanmin(pusht_r), np.nanmin(cube_r), -0.25)),
        vmax=float(max(np.nanmax(pusht_r), np.nanmax(cube_r), 0.55)),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    scatter = plot_goal_panel(
        axes[0],
        pusht_emb,
        pusht_r,
        pusht_success,
        r"(a) PushT: goal latents colored by $R_{\mathrm{pool}}$",
        norm,
    )
    plot_goal_panel(
        axes[1],
        cube_emb,
        cube_r,
        cube_success,
        r"(b) Cube: goal latents colored by $R_{\mathrm{pool}}$",
        norm,
        size_scale=1.5,
    )
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.16, top=0.9, wspace=0.12)
    cbar = fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.035)
    cbar.set_label("R_pool(C_model)", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=8)
    fig.text(
        0.5,
        0.015,
        "Marker area is proportional to final-pool success mass.",
        ha="center",
        fontsize=8,
        color="0.25",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CROSS_ENV_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {CROSS_ENV_PATH.relative_to(ROOT)}")
    print(
        f"Cross-env Rpool means: PushT={np.nanmean(pusht_r):.3f}, "
        f"Cube={np.nanmean(cube_r):.3f}"
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    build_fig5_enhanced()
    build_mppi_comparison()
    build_cross_env()


if __name__ == "__main__":
    main()
