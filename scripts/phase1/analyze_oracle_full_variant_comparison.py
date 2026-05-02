#!/usr/bin/env python3
"""Build the full latent/V3/V1/V2 oracle variant comparison report."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lewm_audit.diagnostics.heatmap import render_heatmap  # noqa: E402


ROWS = ("D0", "D1", "D2", "D3")
RBINS = ("R0", "R1", "R2", "R3")
CELLS = [f"{row}x{rbin}" for row in ROWS for rbin in RBINS]
DISPLACEMENT_EDGES = [0.0, 10.0, 50.0, 120.0, math.inf]
ROTATION_EDGES = [0.0, 0.25, 0.75, 1.25, math.inf]

TRACK_A_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
V3_PATHS = {
    row: PROJECT_ROOT / "results" / "phase1" / f"{row.lower()}_oracle_ablation" / f"{row.lower()}_oracle_V3.json"
    for row in ROWS
}
V1_PATHS = {
    row: PROJECT_ROOT / "results" / "phase1" / "v1_oracle_ablation" / f"v1_{row.lower()}.json"
    for row in ROWS
}
V2_PATHS = {
    row: PROJECT_ROOT / "results" / "phase1" / "v2_oracle_ablation" / f"v2_{row.lower()}.json"
    for row in ROWS
}
V3_REPORT_PATH = PROJECT_ROOT / "docs" / "phase1" / "oracle_v3_row_comparison.md"
REPORT_PATH = PROJECT_ROOT / "docs" / "phase1" / "oracle_full_variant_comparison.md"
FIG_DIR = PROJECT_ROOT / "results" / "phase1" / "figures" / "track_a"

GROUP_1 = ["D3xR0", "D3xR1", "D3xR2", "D3xR3", "D1xR1", "D1xR2", "D2xR1", "D2xR2", "D0xR2"]
GROUP_2 = ["D0xR0", "D0xR1", "D1xR0"]
GROUP_3 = ["D0xR3", "D1xR3", "D2xR0", "D2xR3"]


def display_path(path: Path) -> str:
    path = path.expanduser().resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def pp(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100.0 * value:.2f}"


def rate(successes: int, total: int) -> float | None:
    return None if total == 0 else float(successes / total)


def row_for_cell(cell: str) -> str:
    return cell.split("x", maxsplit=1)[0]


def collect_rates(paths: dict[str, Path] | Path, source: str) -> tuple[dict[str, float], dict[str, int], dict[str, dict]]:
    if isinstance(paths, Path):
        data_by_label = {"all": load_json(paths)}
    else:
        data_by_label = {label: load_json(path) for label, path in paths.items()}
    counts = {cell: 0 for cell in CELLS}
    successes = {cell: 0 for cell in CELLS}
    for data in data_by_label.values():
        for pair in data["pairs"]:
            cell = pair["cell"]
            for action in pair["actions"]:
                if action["source"] != source:
                    continue
                counts[cell] += 1
                successes[cell] += int(bool(action["success"]))
    rates = {cell: rate(successes[cell], counts[cell]) for cell in CELLS}
    missing = [cell for cell, value in rates.items() if value is None]
    if missing:
        raise ValueError(f"Missing {source} records for cells: {missing}")
    return rates, counts, data_by_label


def matrix_from_rates(rates: dict[str, float]) -> np.ndarray:
    matrix = np.zeros((len(ROWS), len(RBINS)), dtype=np.float64)
    for d_idx, row in enumerate(ROWS):
        for r_idx, rbin in enumerate(RBINS):
            matrix[d_idx, r_idx] = rates[f"{row}x{rbin}"]
    return matrix


def label_delta(delta: float, a_name: str, b_name: str) -> str:
    delta_pp = 100.0 * delta
    if delta_pp >= 20.0:
        label = f"{b_name}++"
    elif delta_pp >= 5.0:
        label = f"{b_name}+"
    elif delta_pp <= -20.0:
        label = f"{a_name}++"
    elif delta_pp <= -5.0:
        label = f"{a_name}+"
    else:
        label = "tie"
    return f"{label} ({delta_pp:+.2f} pp)"


def render_who_wins_grid(title: str, a_rates: dict[str, float], b_rates: dict[str, float], a_name: str, b_name: str) -> list[str]:
    lines = [f"### {title}", "", "| D row \\ R bin | R0 | R1 | R2 | R3 |", "|---|---:|---:|---:|---:|"]
    for row in ROWS:
        cells = []
        for rbin in RBINS:
            cell = f"{row}x{rbin}"
            cells.append(label_delta(b_rates[cell] - a_rates[cell], a_name, b_name))
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def extract_v3_grid() -> list[str]:
    text = V3_REPORT_PATH.read_text()
    match = re.search(r"## 4\. Cell-Level Who Wins Grid\n\n(?P<table>\| D row.*?\n\n)", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not extract Grid A from {V3_REPORT_PATH}")
    return ["### Grid A. latent vs V3", "", *match.group("table").strip().splitlines(), ""]


def table_row(cell: str, rates: dict[str, dict[str, float]]) -> list[str]:
    latent = rates["latent"][cell]
    v3 = rates["V3"][cell]
    v1 = rates["V1"][cell]
    v2 = rates["V2"][cell]
    return [
        cell,
        pct(latent),
        pct(v3),
        pct(v1),
        pct(v2),
        pp(v3 - latent),
        pp(v1 - v3),
        pp(v2 - v1),
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---", *["---:" for _ in headers[1:]]]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def group_table(cells: list[str], headers: list[str], row_fn) -> list[str]:
    return markdown_table(headers, [row_fn(cell) for cell in cells])


def render_delta_heatmap(delta_rates: dict[str, float], title: str, output_path: Path) -> Path:
    matrix = matrix_from_rates(delta_rates) * 100.0
    return render_heatmap(
        matrix,
        DISPLACEMENT_EDGES,
        ROTATION_EDGES,
        title,
        "RdBu",
        output_path,
        annotate=True,
        value_fmt="{:+.1f}",
    )


def render_full_variant_success_grid(rates: dict[str, dict[str, float]], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.5), sharey=True, constrained_layout=True)
    names = ["latent", "V3", "V1", "V2"]
    colors = ["#555555", "#2f6fbd", "#2d9c68", "#b45f06"]
    for d_idx, row in enumerate(ROWS):
        for r_idx, rbin in enumerate(RBINS):
            cell = f"{row}x{rbin}"
            ax = axes[d_idx, r_idx]
            values = [100.0 * rates[name][cell] for name in names]
            ax.bar(np.arange(len(names)), values, color=colors, width=0.68)
            ax.set_title(cell, fontsize=10)
            ax.set_ylim(0, 105)
            ax.set_xticks(np.arange(len(names)), labels=names, rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            for idx, value in enumerate(values):
                ax.text(idx, min(value + 3.0, 102.0), f"{value:.0f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("CEM_late success by cell and variant", fontsize=14)
    fig.supylabel("Success rate (%)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def variant_sanity(rates: dict[str, dict[str, float]], variant: str) -> dict[str, dict]:
    out = {}
    for row in ROWS:
        cells = [f"{row}x{rbin}" for rbin in RBINS]
        lower = [
            (cell, 100.0 * (rates[variant][cell] - rates["V3"][cell]))
            for cell in cells
            if rates[variant][cell] < rates["V3"][cell]
        ]
        out[row] = {
            "ge_v3_count": len(cells) - len(lower),
            "lt_v3_count": len(lower),
            "lt_v3_cells": lower,
        }
    return out


def v2_degeneracy_proxy(v2_data: dict[str, dict]) -> dict[str, list[int]]:
    flagged: dict[str, list[int]] = {}
    for data in v2_data.values():
        for pair in data["pairs"]:
            late = [action for action in pair["actions"] if action["source"] == "CEM_late_V2"]
            early = [action for action in pair["actions"] if action["source"] == "CEM_early_V2"]
            if late and early and all(float(action.get("cem_oracle_cost", -1.0)) == 1.0 for action in [*early, *late]):
                flagged.setdefault(pair["cell"], []).append(int(pair["pair_id"]))
    return flagged


def summarize_group_1(rates: dict[str, dict[str, float]]) -> str:
    v1_deltas = [100.0 * (rates["V1"][cell] - rates["V3"][cell]) for cell in GROUP_1]
    v2_deltas = [100.0 * (rates["V2"][cell] - rates["V3"][cell]) for cell in GROUP_1]
    best_deltas = [max(v1, v2) for v1, v2 in zip(v1_deltas, v2_deltas, strict=True)]
    return (
        f"Group 1: V1 exceeded V3 in {sum(delta > 0 for delta in v1_deltas)}/{len(GROUP_1)} cells "
        f"(mean delta {np.mean(v1_deltas):+.2f} pp); V2 exceeded V3 in "
        f"{sum(delta > 0 for delta in v2_deltas)}/{len(GROUP_1)} cells "
        f"(mean delta {np.mean(v2_deltas):+.2f} pp). The best of V1/V2 exceeded V3 in "
        f"{sum(delta > 0 for delta in best_deltas)}/{len(GROUP_1)} cells."
    )


def summarize_group_2(rates: dict[str, dict[str, float]]) -> str:
    closures = []
    beats = []
    for cell in GROUP_2:
        latent = rates["latent"][cell]
        v3 = rates["V3"][cell]
        for variant in ("V1", "V2"):
            closures.append((cell, variant, 100.0 * (rates[variant][cell] - v3)))
            if rates[variant][cell] > latent:
                beats.append(f"{variant} {cell}")
    close_gt_30 = [f"{variant} {cell} ({delta:+.2f} pp)" for cell, variant, delta in closures if delta > 30.0]
    return (
        f"Group 2: V1/V2 closure versus V3 ranged from {min(delta for _, _, delta in closures):+.2f} pp "
        f"to {max(delta for _, _, delta in closures):+.2f} pp; >30 pp closures: "
        f"{', '.join(close_gt_30) if close_gt_30 else 'none'}. Variants beating latent: "
        f"{', '.join(beats) if beats else 'none'}."
    )


def summarize_group_3(rates: dict[str, dict[str, float]]) -> str:
    v1_deltas = [100.0 * (rates["V1"][cell] - rates["V3"][cell]) for cell in GROUP_3]
    v2_deltas = [100.0 * (rates["V2"][cell] - rates["V3"][cell]) for cell in GROUP_3]
    return (
        f"Group 3: V1 moved relative to V3 by {min(v1_deltas):+.2f} to {max(v1_deltas):+.2f} pp; "
        f"V2 moved relative to V3 by {min(v2_deltas):+.2f} to {max(v2_deltas):+.2f} pp."
    )


def cost_shape_summary(rates: dict[str, dict[str, float]]) -> tuple[str, list[tuple[str, float]], tuple[str, float], tuple[str, float]]:
    deltas = [(cell, 100.0 * (rates["V1"][cell] - rates["V2"][cell])) for cell in CELLS]
    spread = [(cell, delta) for cell, delta in deltas if abs(delta) > 5.0]
    most_positive = max(deltas, key=lambda item: item[1])
    most_negative = min(deltas, key=lambda item: item[1])
    v1_gt = sum(delta > 0 for _, delta in spread)
    v2_gt = sum(delta < 0 for _, delta in spread)
    summary = (
        f"V1 and V2 differ by >5 pp in {len(spread)}/16 cells: "
        f"V1 > V2 in {v1_gt}, V2 > V1 in {v2_gt}. "
        f"Most positive V1 - V2: {most_positive[0]} ({most_positive[1]:+.2f} pp). "
        f"Most negative V1 - V2: {most_negative[0]} ({most_negative[1]:+.2f} pp)."
    )
    return summary, spread, most_positive, most_negative


def build_report() -> dict:
    latent_rates, latent_counts, latent_data = collect_rates(TRACK_A_PATH, "CEM_late")
    v3_rates, v3_counts, v3_data = collect_rates(V3_PATHS, "CEM_late_V3")
    v1_rates, v1_counts, v1_data = collect_rates(V1_PATHS, "CEM_late_V1")
    v2_rates, v2_counts, v2_data = collect_rates(V2_PATHS, "CEM_late_V2")
    rates = {"latent": latent_rates, "V3": v3_rates, "V1": v1_rates, "V2": v2_rates}
    counts = {"latent": latent_counts, "V3": v3_counts, "V1": v1_counts, "V2": v2_counts}

    v3_v1_heatmap = render_delta_heatmap(
        {cell: v1_rates[cell] - v3_rates[cell] for cell in CELLS},
        "V1 - V3 CEM_late success delta (pp)",
        FIG_DIR / "v3_vs_v1_delta_heatmap.png",
    )
    v3_v2_heatmap = render_delta_heatmap(
        {cell: v2_rates[cell] - v3_rates[cell] for cell in CELLS},
        "V2 - V3 CEM_late success delta (pp)",
        FIG_DIR / "v3_vs_v2_delta_heatmap.png",
    )
    v1_v2_heatmap = render_delta_heatmap(
        {cell: v2_rates[cell] - v1_rates[cell] for cell in CELLS},
        "V2 - V1 CEM_late success delta (pp)",
        FIG_DIR / "v1_vs_v2_delta_heatmap.png",
    )
    success_grid = render_full_variant_success_grid(rates, FIG_DIR / "full_variant_success_grid.png")

    sanity_v1 = variant_sanity(rates, "V1")
    sanity_v2 = variant_sanity(rates, "V2")
    cost_summary, spread, most_positive, most_negative = cost_shape_summary(rates)
    group_1_summary = summarize_group_1(rates)
    group_2_summary = summarize_group_2(rates)
    group_3_summary = summarize_group_3(rates)
    degeneracy = v2_degeneracy_proxy(v2_data)

    v1_wall = {row: v1_data[row]["metadata"].get("wallclock_seconds") for row in ROWS}
    v2_wall = {row: v2_data[row]["metadata"].get("wallclock_seconds") for row in ROWS}

    lines: list[str] = []
    lines.append("# Oracle Full Variant Comparison")
    lines.append("")
    lines.append("## 1. Provenance")
    lines.append("")
    lines.append(f"- Report generation git HEAD: `{git_head()}`")
    lines.append(f"- Latent reference: `{display_path(TRACK_A_PATH)}`")
    for row in ROWS:
        lines.append(f"- {row} V3 data: `{display_path(V3_PATHS[row])}`")
    for row in ROWS:
        lines.append(f"- {row} V1 data: `{display_path(V1_PATHS[row])}`")
    for row in ROWS:
        lines.append(f"- {row} V2 data: `{display_path(V2_PATHS[row])}`")
    lines.append(f"- Latent reference metadata git commit: `{latent_data['all']['metadata'].get('git_commit')}`")
    for row in ROWS:
        lines.append(f"- {row} V3 metadata git commit: `{v3_data[row]['metadata'].get('git_commit')}`")
    for row in ROWS:
        lines.append(f"- {row} V1 metadata git commit: `{v1_data[row]['metadata'].get('git_commit')}`")
    for row in ROWS:
        lines.append(f"- {row} V2 metadata git commit: `{v2_data[row]['metadata'].get('git_commit')}`")
    lines.append("- Seed: `0` for latent, V3, V1, and V2 artifacts.")
    lines.append(
        "- V1/V2 data and smooth_random records were not recomputed; V1/V2 JSONs store only "
        "`CEM_early` and `CEM_late` records, with baselines sourced from the matching V3 row JSON."
    )
    lines.append(
        "- Execution pattern: one invocation per row x variant with "
        "`scripts/phase1/eval_oracle_cem_only_variant.py`; V1 rows D0-D3 were run first, "
        "then V2 rows D0-D3, each with `--cell-filter`, `--variant`, `--output-path`, "
        "`--data-smooth-random-source`, and `--resume`."
    )
    for row in ROWS:
        lines.append(
            f"- {row} V1 wall-clock: `{v1_wall[row]:.2f}` seconds; "
            f"{row} V2 wall-clock: `{v2_wall[row]:.2f}` seconds."
        )
    lines.append(f"- V1 total wall-clock: `{sum(v1_wall.values()):.2f}` seconds.")
    lines.append(f"- V2 total wall-clock: `{sum(v2_wall.values()):.2f}` seconds.")
    lines.append("")

    lines.append("## 2. Headline 4x4 Grids")
    lines.append("")
    lines.extend(extract_v3_grid())
    lines.extend(render_who_wins_grid("Grid B. V3 vs V1", v3_rates, v1_rates, "V3", "V1"))
    lines.extend(render_who_wins_grid("Grid C. V3 vs V2", v3_rates, v2_rates, "V3", "V2"))
    lines.extend(render_who_wins_grid("Grid D. V1 vs V2", v1_rates, v2_rates, "V1", "V2"))

    lines.append("## 3. Full 16-Cell Variant Matrix")
    lines.append("")
    headers = [
        "cell",
        "latent_CEM_late",
        "V3_CEM_late",
        "V1_CEM_late",
        "V2_CEM_late",
        "delta_V3_vs_latent_pp",
        "delta_V1_vs_V3_pp",
        "delta_V2_vs_V1_pp",
    ]
    lines.extend(markdown_table(headers, [table_row(cell, rates) for cell in CELLS]))
    lines.append("")

    lines.append("## 4. Critical Cell Groups")
    lines.append("")
    lines.append("### Group 1. Oracle-Favorable Cells")
    lines.append("")
    lines.append("Question: does V1 or V2 add further gain over V3?")
    lines.append("")
    lines.extend(
        group_table(
            GROUP_1,
            ["cell", "V3", "V1", "V2", "V1-V3 pp", "V2-V3 pp", "V2-V1 pp"],
            lambda cell: [
                cell,
                pct(v3_rates[cell]),
                pct(v1_rates[cell]),
                pct(v2_rates[cell]),
                pp(v1_rates[cell] - v3_rates[cell]),
                pp(v2_rates[cell] - v3_rates[cell]),
                pp(v2_rates[cell] - v1_rates[cell]),
            ],
        )
    )
    lines.append("")
    lines.append("### Group 2. Latent-Favorable Cells")
    lines.append("")
    lines.append("Question: does V1 or V2 close the gap to latent or beat it?")
    lines.append("")
    lines.extend(
        group_table(
            GROUP_2,
            ["cell", "latent", "V3", "V1", "V2", "V1-V3 pp", "V2-V3 pp", "V1-latent pp", "V2-latent pp"],
            lambda cell: [
                cell,
                pct(latent_rates[cell]),
                pct(v3_rates[cell]),
                pct(v1_rates[cell]),
                pct(v2_rates[cell]),
                pp(v1_rates[cell] - v3_rates[cell]),
                pp(v2_rates[cell] - v3_rates[cell]),
                pp(v1_rates[cell] - latent_rates[cell]),
                pp(v2_rates[cell] - latent_rates[cell]),
            ],
        )
    )
    lines.append("")
    lines.append("### Group 3. Tie Cells")
    lines.append("")
    lines.append("Question: does V1 or V2 break the tie either way?")
    lines.append("")
    lines.extend(
        group_table(
            GROUP_3,
            ["cell", "latent", "V3", "V1", "V2", "V1-V3 pp", "V2-V3 pp", "V2-V1 pp"],
            lambda cell: [
                cell,
                pct(latent_rates[cell]),
                pct(v3_rates[cell]),
                pct(v1_rates[cell]),
                pct(v2_rates[cell]),
                pp(v1_rates[cell] - v3_rates[cell]),
                pp(v2_rates[cell] - v3_rates[cell]),
                pp(v2_rates[cell] - v1_rates[cell]),
            ],
        )
    )
    lines.append("")

    lines.append("## 5. Cost-Shape Sensitivity Check")
    lines.append("")
    lines.append(cost_summary)
    if spread:
        lines.append("")
        lines.extend(
            markdown_table(
                ["cell", "V1-V2 pp", "direction"],
                [
                    [cell, f"{delta:+.2f}", "V1 > V2" if delta > 0 else "V2 > V1"]
                    for cell, delta in sorted(spread)
                ],
            )
        )
    lines.append("")

    lines.append("## 6. Pattern Summary")
    lines.append("")
    lines.append(f"- {group_1_summary}")
    lines.append(f"- {group_2_summary}")
    lines.append(f"- {group_3_summary}")
    lines.append(f"- V1 vs V2: {cost_summary}")
    lines.append("- V1 per-row sanity versus V3:")
    for row in ROWS:
        row_sanity = sanity_v1[row]
        lower = ", ".join(f"{cell} ({delta:+.2f} pp)" for cell, delta in row_sanity["lt_v3_cells"]) or "none"
        lines.append(
            f"  - {row}: V1 >= V3 in {row_sanity['ge_v3_count']}/4 cells; "
            f"V1 < V3 in {row_sanity['lt_v3_count']}/4 cells: {lower}."
        )
    lines.append("- V2 per-row sanity versus V3:")
    for row in ROWS:
        row_sanity = sanity_v2[row]
        lower = ", ".join(f"{cell} ({delta:+.2f} pp)" for cell, delta in row_sanity["lt_v3_cells"]) or "none"
        lines.append(
            f"  - {row}: V2 >= V3 in {row_sanity['ge_v3_count']}/4 cells; "
            f"V2 < V3 in {row_sanity['lt_v3_count']}/4 cells: {lower}."
        )
    lines.append("")

    lines.append("## 7. Limitations")
    lines.append("")
    lines.append("- Ceiling effects remain important: D0 latent success is already high in some cells, so absolute pp gaps can understate relative improvement.")
    lines.append("- Oracle CEM uses real-env state; these variants are diagnostic upper-bound planners, not deployable latent planners.")
    lines.append("- The V3 scalar cost is not equivalent to PushT's conjunctive success criterion; negative-delta cells should be read against that mismatch.")
    lines.append("- Per-cell sample sizes are 6 or 7 pairs, so cell-level rates can move substantially if a small number of pairs change outcome.")
    lines.append("- V1 and V2 share the same alpha=20/(pi/9) hinge boundary; a different alpha might yield different results.")
    lines.append("- V2 indicator has zero gradient inside the success region; CEM may behave qualitatively differently from gradient-based assumptions.")
    lines.append("- All variants share the same 80 actions sampling design; V1/V2 store only the 40 recomputed CEM records and rely on V3 for the 40 shared data/smooth_random records.")
    if degeneracy:
        flagged_pairs = sum(len(pair_ids) for pair_ids in degeneracy.values())
        lines.append(
            f"- V2 selected-elite cost degeneracy proxy: {flagged_pairs} pairs had all stored early and late V2 elite costs equal to 1.0; "
            "the artifacts do not store every candidate cost for all 30 iterations."
        )
    else:
        lines.append("- V2 selected-elite cost degeneracy proxy did not flag any pair with all stored early and late elite costs equal to 1.0.")
    lines.append("")

    lines.append("## 8. Recommended Next Step")
    lines.append("")
    lines.append(
        "Recommendation, user decides: run per-trajectory visualization on a curated set of telling cells. "
        "The full-grid table now identifies where success-aligned oracle costs helped, hurt, or split by cost shape; "
        "visualizing a small set from Groups 1, 2, and 3 is the most direct way to inspect whether the numerical changes "
        "come from qualitatively different pushes, rotations, or terminal near-misses before Track A consolidation freezes the narrative."
    )
    lines.append("")

    lines.append("## 9. Heatmap PNGs")
    lines.append("")
    for label, path in (
        ("V3 vs V1 delta heatmap", v3_v1_heatmap),
        ("V3 vs V2 delta heatmap", v3_v2_heatmap),
        ("V1 vs V2 delta heatmap", v1_v2_heatmap),
        ("Full variant success grid", success_grid),
    ):
        rel = Path("../../") / Path(display_path(path))
        lines.append(f"![{label}]({rel.as_posix()})")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    return {
        "report_path": display_path(REPORT_PATH),
        "heatmaps": [display_path(path) for path in (v3_v1_heatmap, v3_v2_heatmap, v1_v2_heatmap, success_grid)],
        "rates": rates,
        "counts": counts,
        "v1_wall": v1_wall,
        "v2_wall": v2_wall,
        "group_summaries": [group_1_summary, group_2_summary, group_3_summary],
        "cost_shape": {
            "summary": cost_summary,
            "spread_cells": spread,
            "most_positive": most_positive,
            "most_negative": most_negative,
        },
        "sanity": {"V1": sanity_v1, "V2": sanity_v2},
        "degeneracy_proxy": degeneracy,
    }


def main() -> int:
    result = build_report()
    serializable = {
        "report_path": result["report_path"],
        "heatmaps": result["heatmaps"],
        "v1_wall": result["v1_wall"],
        "v2_wall": result["v2_wall"],
        "group_summaries": result["group_summaries"],
        "cost_shape": result["cost_shape"],
        "sanity": result["sanity"],
        "degeneracy_proxy": result["degeneracy_proxy"],
    }
    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
