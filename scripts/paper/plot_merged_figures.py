#!/usr/bin/env python3
"""Build merged main-paper figures from existing audit artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.revision import generate_figures as revision_figs  # noqa: E402
from scripts.paper import plot_spatial_figures as spatial_figs  # noqa: E402


FIGURE_DIR = ROOT / "paper" / "figures"
FIG2_COMBINED = FIGURE_DIR / "fig2_combined.pdf"
FIG6_SPATIAL = FIGURE_DIR / "fig6_spatial.pdf"


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def load_track_pair_and_selected(pair_id: int = 51) -> tuple[dict, dict]:
    rpool_data = revision_figs.load_json(revision_figs.DEFAULT_RPOOL_PATH)
    track_data = revision_figs.load_json(revision_figs.DEFAULT_TRACK_A_PATH)
    endpoint_metrics = revision_figs.endpoint_metrics_by_pair(track_data["pairs"])
    track_pairs = {int(pair["pair_id"]): pair for pair in track_data["pairs"]}
    row = next(row for row in rpool_data["per_pair"] if int(row["pair_id"]) == pair_id)
    selected = {
        **row,
        "Rendpoint": endpoint_metrics[pair_id]["Rendpoint"],
        "primary_subset": revision_figs.primary_subset(row["subsets"]),
    }
    return track_pairs[pair_id], selected


def plot_symbolic_cost_colorbar(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    mappable = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=mcolors.Normalize(0, 1))
    mappable.set_array([0, 1])
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.018, pad=0.018)
    cbar.set_label("Within-panel cost", fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=8)


def build_fig2_combined() -> None:
    pair_id = 51
    track_pair, selected = load_track_pair_and_selected(pair_id)
    endpoint_values = revision_figs.values_for_pair(track_pair)
    endpoint_z, endpoint_goal = revision_figs.load_endpoint_latents(
        revision_figs.DEFAULT_TRACK_A_LATENTS_PATH,
        pair_id,
    )
    endpoint_coords, endpoint_goal_coord = revision_figs.pca_project_with_goal(
        endpoint_z,
        endpoint_goal,
    )

    pool = revision_figs.load_pool(revision_figs.DEFAULT_POOL_DIR, pair_id)
    initial_state, goal_state = revision_figs.load_pair_states(
        revision_figs.DEFAULT_PAIRS_PATH,
        pool["pair_spec"],
    )
    seed_base = int(pool["metadata"].get("seed", 0)) + pair_id * 100_000
    pool_coords, replayed_block_pos_dist = revision_figs.replay_pool_block_xy(
        raw_actions_batch=pool["raw_actions"],
        initial_state=initial_state,
        goal_state=goal_state,
        seed_base=seed_base,
    )
    max_error = float(np.max(np.abs(replayed_block_pos_dist - pool["block_pos_dist"])))
    if max_error > 1e-3:
        raise RuntimeError(f"Pair {pair_id} replay mismatch: max error={max_error:.6g}")
    pool_goal_coord = np.asarray(goal_state[2:4], dtype=np.float64)

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.1], hspace=0.3, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])

    cmap = "RdYlGn_r"
    scatter_a = revision_figs.plot_latent_scatter(
        ax_a,
        endpoint_coords,
        endpoint_goal_coord,
        endpoint_values["C_model"],
        title="Endpoint Evaluation",
        annotation=rf"$R_{{endpoint}}={float(selected['Rendpoint']):.3f}$",
        cmap=cmap,
    )
    revision_figs.plot_physical_scatter(
        ax_b,
        pool_coords,
        pool_goal_coord,
        pool["C_model"],
        title="CEM Pool (learned cost)",
        annotation=rf"$R_{{pool}}(C_{{model}})={float(selected['Rpool_Cmodel']):.3f}$",
        cmap=cmap,
    )
    revision_figs.plot_physical_scatter(
        ax_c,
        pool_coords,
        pool_goal_coord,
        pool["V1"],
        title="CEM Pool (oracle cost)",
        annotation=rf"$R_{{pool}}(C_{{V1}})={float(selected['Rpool_V1']):.3f}$",
        cmap=cmap,
    )
    for label, ax in zip(("(a)", "(b)", "(c)"), (ax_a, ax_b, ax_c), strict=True):
        panel_label(ax, label)
    plot_symbolic_cost_colorbar(fig, [ax_a, ax_b, ax_c])

    spatial_figs.plot_attribution_scatter(ax_d, spatial_figs.load_json(spatial_figs.PUSHT_RPOOL_PATH))
    ax_d.set_title("Pool-level V1 attribution across PushT pairs", loc="left", fontsize=12)
    panel_label(ax_d, "(d)", x=-0.045, y=1.04)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG2_COMBINED, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {FIG2_COMBINED.relative_to(ROOT)}")


def build_fig6_spatial() -> None:
    example = spatial_figs.choose_mppi_example()
    cem_pool = spatial_figs.load_pool(example["cem_path"])
    mppi_pool = spatial_figs.load_pool(example["mppi_path"])
    cem_c = spatial_figs.as_numpy(cem_pool["c_real_state"]).astype(float)
    mppi_c = spatial_figs.as_numpy(mppi_pool["c_real_state"]).astype(float)
    cost_norm = mcolors.Normalize(
        vmin=float(min(np.nanmin(cem_c), np.nanmin(mppi_c))),
        vmax=float(max(np.nanmax(cem_c), np.nanmax(mppi_c))),
    )
    cem_std = example["summary"]["cem_default"]["pool_Creal_std"]["mean"]
    mppi_std = example["summary"]["mppi_tau_1"]["pool_Creal_std"]["mean"]

    pusht_z, pusht_r, pusht_success = spatial_figs.collect_pusht_goal_latents()
    cube_z, cube_r, cube_success = spatial_figs.collect_cube_goal_latents()
    pusht_emb = spatial_figs.tsne_embedding(pusht_z, seed=4)
    cube_emb = spatial_figs.tsne_embedding(cube_z, seed=8)
    rpool_norm = mcolors.Normalize(
        vmin=float(min(np.nanmin(pusht_r), np.nanmin(cube_r), -0.25)),
        vmax=float(max(np.nanmax(pusht_r), np.nanmax(cube_r), 0.55)),
    )

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    scatter_cost, pca, _ = spatial_figs.plot_pool_3d(
        ax_a,
        cem_pool,
        f"(a) CEM pool (std = {cem_std:.1f})",
        norm=cost_norm,
        show_special_markers=False,
        view=(25, -60),
        pc3_label_mode="axis",
        title_fontsize=14,
        title_fontweight="bold",
        axis_label_fontsize=11,
        tick_labelsize=9,
    )
    spatial_figs.plot_pool_3d(
        ax_b,
        mppi_pool,
        f"(b) MPPI pool (std = {mppi_std:.1f})",
        norm=cost_norm,
        pca=pca,
        show_special_markers=False,
        view=(25, -60),
        pc3_label_mode="axis",
        title_fontsize=14,
        title_fontweight="bold",
        axis_label_fontsize=11,
        tick_labelsize=9,
    )

    scatter_rpool = spatial_figs.plot_goal_panel(
        ax_c,
        pusht_emb,
        pusht_r,
        pusht_success,
        r"(c) PushT: goal latents colored by $R_{\mathrm{pool}}$",
        rpool_norm,
        title_fontsize=14,
        title_fontweight="bold",
        axis_label_fontsize=12,
        tick_labelsize=10,
        annotation_fontsize=12,
        show_ticks=True,
    )
    spatial_figs.plot_goal_panel(
        ax_d,
        cube_emb,
        cube_r,
        cube_success,
        r"(d) Cube: goal latents colored by $R_{\mathrm{pool}}$",
        rpool_norm,
        size_scale=1.5,
        title_fontsize=14,
        title_fontweight="bold",
        axis_label_fontsize=12,
        tick_labelsize=10,
        annotation_fontsize=12,
        show_ticks=True,
    )

    cost_sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=cost_norm)
    cost_sm.set_array(np.concatenate([cem_c, mppi_c]))
    cost_cax = fig.add_axes([0.94, 0.55, 0.015, 0.35])
    cost_cbar = fig.colorbar(cost_sm, cax=cost_cax)
    cost_cbar.set_label("C_real_state", fontsize=11, labelpad=10)
    cost_cbar.ax.tick_params(labelsize=10)

    rpool_cax = fig.add_axes([0.94, 0.08, 0.015, 0.35])
    rpool_cbar = fig.colorbar(scatter_rpool, cax=rpool_cax)
    rpool_cbar.set_label("R_pool(C_model)", fontsize=11, labelpad=10)
    rpool_cbar.ax.tick_params(labelsize=10)
    fig.text(
        0.5,
        0.035,
        "Bottom-row marker area is proportional to final-pool success mass.",
        ha="center",
        fontsize=11,
        color="0.25",
    )
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.08, top=0.95)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG6_SPATIAL, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {FIG6_SPATIAL.relative_to(ROOT)}")


def main() -> None:
    revision_figs.set_plot_style()
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    build_fig2_combined()
    build_fig6_spatial()


if __name__ == "__main__":
    main()
