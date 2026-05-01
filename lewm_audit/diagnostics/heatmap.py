"""Small matplotlib heatmap helpers for Track A grid diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def _validate_edges(edges: list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError(f"{name} must contain at least two edges")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def cell_label(d_idx: int, r_idx: int) -> str:
    return f"D{d_idx}xR{r_idx}"


def axis_bin(value: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges, float(value), side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        raise ValueError(f"Value {value} is outside edges {edges.tolist()}")
    return idx


def cell_grid_from_records(
    records: Iterable[dict],
    displacement_edges: list[float],
    rotation_edges: list[float],
) -> dict:
    """Group records into a D x R grid using physical displacement/rotation."""
    d_edges = _validate_edges(displacement_edges, name="displacement_edges")
    r_edges = _validate_edges(rotation_edges, name="rotation_edges")
    grid = {
        cell_label(d_idx, r_idx): []
        for d_idx in range(len(d_edges) - 1)
        for r_idx in range(len(r_edges) - 1)
    }
    for record in records:
        d_idx = axis_bin(record["block_displacement_px"], d_edges)
        r_idx = axis_bin(record["required_rotation_rad"], r_edges)
        cell = cell_label(d_idx, r_idx)
        enriched = dict(record)
        enriched["cell_d"] = d_idx
        enriched["cell_r"] = r_idx
        enriched.setdefault("cell", cell)
        grid[cell].append(enriched)
    return grid


def per_cell_metric(
    records: Iterable[dict],
    metric_fn: Callable[[list[dict]], float | None],
    agg: str = "mean",
) -> dict[str, float | None]:
    """Compute a metric per existing `cell` label.

    `metric_fn` is called once for each cell with that cell's records. The `agg`
    argument is accepted for the public API and for scalar metric values is
    equivalent to returning the metric directly.
    """
    if agg not in {"mean", "std", "median", "sum", "count"}:
        raise ValueError(f"Unsupported aggregation: {agg}")
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["cell"]), []).append(record)
    return {cell: metric_fn(cell_records) for cell, cell_records in sorted(grouped.items())}


def matrix_from_cell_values(
    cell_values: dict[str, float | None],
    *,
    n_displacement_bins: int,
    n_rotation_bins: int,
) -> np.ndarray:
    matrix = np.full((n_displacement_bins, n_rotation_bins), np.nan, dtype=np.float64)
    for cell, value in cell_values.items():
        left, right = cell.split("x")
        d_idx = int(left[1:])
        r_idx = int(right[1:])
        matrix[d_idx, r_idx] = np.nan if value is None else float(value)
    return matrix


def edge_labels(edges: list[float]) -> list[str]:
    labels = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        right_text = "inf" if math.isinf(float(right)) else f"{float(right):g}"
        labels.append(f"{float(left):g}-{right_text}")
    return labels


def _norm_for_matrix(matrix: np.ndarray, cmap: str):
    finite = matrix[np.isfinite(matrix)]
    if len(finite) == 0:
        return None
    if cmap in {"RdBu", "RdBu_r", "coolwarm", "bwr", "seismic", "PiYG", "PRGn"}:
        vmax = float(np.max(np.abs(finite)))
        if vmax == 0:
            return None
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return None


def render_heatmap(
    matrix,
    displacement_edges,
    rotation_edges,
    title,
    cmap,
    output_path,
    annotate=True,
    value_fmt="{:.2f}",
    mask=None,
    annotation_counts=None,
):
    """Render a D x R matrix as a PNG heatmap."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    d_edges = _validate_edges(displacement_edges, name="displacement_edges")
    r_edges = _validate_edges(rotation_edges, name="rotation_edges")
    expected_shape = (len(d_edges) - 1, len(r_edges) - 1)
    if matrix.shape != expected_shape:
        raise ValueError(f"matrix shape {matrix.shape} does not match expected {expected_shape}")

    plot_matrix = np.ma.array(matrix, mask=np.zeros_like(matrix, dtype=bool))
    if mask is not None:
        plot_matrix.mask = np.asarray(mask, dtype=bool)
    plot_matrix.mask = np.asarray(plot_matrix.mask) | ~np.isfinite(matrix)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    image = ax.imshow(plot_matrix, cmap=cmap, norm=_norm_for_matrix(matrix, cmap), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Required rotation bin (rad)")
    ax.set_ylabel("Block displacement bin (px)")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=edge_labels(list(r_edges)))
    ax.set_yticks(np.arange(matrix.shape[0]), labels=edge_labels(list(d_edges)))
    ax.tick_params(axis="x", labelrotation=30)
    fig.colorbar(image, ax=ax, shrink=0.82)

    if annotate:
        counts = None if annotation_counts is None else np.asarray(annotation_counts)
        finite = matrix[np.isfinite(matrix)]
        midpoint = float(np.nanmean(finite)) if len(finite) else 0.0
        for d_idx in range(matrix.shape[0]):
            for r_idx in range(matrix.shape[1]):
                if plot_matrix.mask[d_idx, r_idx]:
                    text = "NA"
                else:
                    text = value_fmt.format(matrix[d_idx, r_idx])
                if counts is not None:
                    text = f"{text}\n(n={int(counts[d_idx, r_idx])})"
                color = "white" if np.isfinite(matrix[d_idx, r_idx]) and matrix[d_idx, r_idx] > midpoint else "black"
                ax.text(r_idx, d_idx, text, ha="center", va="center", color=color, fontsize=9)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
