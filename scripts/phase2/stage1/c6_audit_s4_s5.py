#!/usr/bin/env python3
"""C6 Audit S4/S5: raw-pixel and hand-crafted pixel-only baselines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.stage1.c6_audit import (  # noqa: E402
    DEFAULT_PIXEL_ARTIFACT,
    add_replay_args_if_available,
    audit_anchor_definitions,
    build_common,
    collect_cached_pixels_once,
    collect_replay_pixels_once,
    import_replay_helpers,
    load_cached_pixels,
    pixels_to_policy_array,
)
from scripts.phase2.stage1.stage1a_controls import (  # noqa: E402
    EXPECTED_RECORDS,
    FALSE_ELITE_K,
    LATENT_DIM,
    TOPK_VALUES,
    clean_float,
    compute_metrics,
    iso_now,
    jsonable,
    load_latent_artifact,
    make_anchor_masks,
    run_single_metrics,
    squared_l2_torch,
    summary_row,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "stage1" / "c6_audit" / "s4_s5_results.json"
FOREGROUND_THRESHOLD = 25.0
EDGE_THRESHOLD = 25.0
MIN_COMPONENT_AREA = 20
MAX_EXAMPLE_FAILURES = 20


@dataclass(frozen=True)
class RawPixelBundle:
    terminal: np.ndarray
    goal: np.ndarray
    goal_pair_ids: list[int]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args_if_available(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixel-artifact", type=Path, default=DEFAULT_PIXEL_ARTIFACT)
    parser.add_argument("--foreground-threshold", type=float, default=FOREGROUND_THRESHOLD)
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD)
    parser.add_argument("--min-component-area", type=int, default=MIN_COMPONENT_AREA)
    args = parser.parse_args()
    if float(args.foreground_threshold) <= 0:
        parser.error("--foreground-threshold must be positive")
    if float(args.edge_threshold) <= 0:
        parser.error("--edge-threshold must be positive")
    if int(args.min_component_area) < 1:
        parser.error("--min-component-area must be positive")
    return args


def fmt(value: float | None) -> str:
    return "nan" if value is None else f"{float(value):.4f}"


def print_summary_table(rows: list[dict]) -> None:
    headers = [
        "Control",
        "Config",
        "Seeds",
        "Spearman",
        "Pairwise",
        "PerPairRho",
        "FalseElite",
    ]
    table = []
    for row in rows:
        table.append(
            [
                str(row["control"]),
                str(row["config"]),
                str(row["n_seeds"]),
                f"{fmt(row['global_spearman_mean'])}/{fmt(row['global_spearman_std'])}",
                f"{fmt(row['pairwise_accuracy_mean'])}/{fmt(row['pairwise_accuracy_std'])}",
                f"{fmt(row['per_pair_rho_mean'])}/{fmt(row['per_pair_rho_mean_std'])}",
                f"{fmt(row['false_elite_rate_mean'])}/{fmt(row['false_elite_rate_std'])}",
            ]
        )
    widths = [max(len(headers[i]), *(len(record[i]) for record in table)) for i in range(len(headers))]
    print("C6 Audit S4/S5 pixel-baseline summary")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for record in table:
        print(" | ".join(record[i].ljust(widths[i]) for i in range(len(headers))))


@lru_cache(maxsize=None)
def optional_module(name: str) -> Any | None:
    if importlib.util.find_spec(name) is None:
        return None
    return __import__(name)


def cv2_module() -> Any | None:
    return optional_module("cv2")


def scipy_ndimage_module() -> Any | None:
    if importlib.util.find_spec("scipy") is None:
        return None
    if importlib.util.find_spec("scipy.ndimage") is None:
        return None
    from scipy import ndimage  # noqa: PLC0415

    return ndimage


def stage1a_helper_names() -> list[str]:
    return [
        compute_metrics.__name__,
        summary_row.__name__,
        load_latent_artifact.__name__,
        make_anchor_masks.__name__,
        run_single_metrics.__name__,
        squared_l2_torch.__name__,
    ]


def prepare_raw_pixels_once(
    *,
    args: argparse.Namespace,
    latent_artifact: dict[str, Any],
) -> RawPixelBundle:
    started = time.time()
    cached = load_cached_pixels(
        pixel_artifact_path=args.pixel_artifact,
        latent_artifact=latent_artifact,
    )
    if cached is not None:
        cached_pixels, cached_metadata = cached
        terminal_pixels, goal_map, pixel_metadata = collect_cached_pixels_once(
            args=args,
            cached_pixels=cached_pixels,
        )
        pixel_metadata.update(
            {
                "artifact_metadata": cached_metadata.get("artifact_metadata", {}),
                "terminal_pixel_shape": cached_metadata.get("terminal_pixel_shape"),
                "goal_pixel_shape": cached_metadata.get("goal_pixel_shape"),
            }
        )
    else:
        helpers = import_replay_helpers()
        ctx = helpers["build_replay_context"](args)
        terminal_pixels, goal_map, pixel_metadata = collect_replay_pixels_once(
            args=args,
            latent_artifact=latent_artifact,
            helpers=helpers,
            ctx=ctx,
        )

    pair_ids = latent_artifact["pair_id"].detach().cpu().numpy().astype(np.int64)
    unique_pair_ids = sorted(set(int(item) for item in pair_ids.tolist()))
    terminal = np.stack([pixels_to_policy_array(item) for item in terminal_pixels]).astype(np.uint8, copy=False)
    goal_by_pair = {
        int(pair_id): pixels_to_policy_array(goal_map[int(pair_id)])
        for pair_id in unique_pair_ids
    }
    goal = np.stack([goal_by_pair[int(pair_id)] for pair_id in pair_ids]).astype(np.uint8, copy=False)
    if terminal.shape != goal.shape:
        raise ValueError(f"Terminal/goal pixel shape mismatch: {terminal.shape} vs {goal.shape}")
    if int(terminal.shape[0]) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} terminal pixels, found {int(terminal.shape[0])}")
    metadata = {
        **pixel_metadata,
        "raw_pixels_prepared_once": True,
        "terminal_shape": list(terminal.shape),
        "goal_shape": list(goal.shape),
        "goal_pairs": int(len(unique_pair_ids)),
        "dtype": str(terminal.dtype),
        "wallclock_seconds": clean_float(time.time() - started),
    }
    return RawPixelBundle(
        terminal=terminal,
        goal=goal,
        goal_pair_ids=unique_pair_ids,
        metadata=metadata,
    )


def grayscale(images: np.ndarray) -> np.ndarray:
    arr = images.astype(np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def raw_pixel_l2(terminal: np.ndarray, goal: np.ndarray) -> np.ndarray:
    diff = terminal.astype(np.float32) - goal.astype(np.float32)
    return np.sum(diff * diff, axis=(1, 2, 3), dtype=np.float64)


def mean_rgb_diff(terminal: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(terminal.astype(np.float32) - goal.astype(np.float32)), axis=(1, 2, 3), dtype=np.float64)


def edge_density(images: np.ndarray, *, threshold: float) -> tuple[np.ndarray, str]:
    gray = grayscale(images)
    cv2 = cv2_module()
    if cv2 is not None:
        values = []
        for image in gray:
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            values.append(float(np.mean(mag > float(threshold))))
        return np.asarray(values, dtype=np.float64), "cv2.Sobel"

    ndimage = scipy_ndimage_module()
    if ndimage is not None:
        gx = ndimage.sobel(gray, axis=2, mode="nearest")
        gy = ndimage.sobel(gray, axis=1, mode="nearest")
        mag = np.sqrt(gx * gx + gy * gy)
        return np.mean(mag > float(threshold), axis=(1, 2)).astype(np.float64), "scipy.ndimage.sobel"

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    return np.mean(mag > float(threshold), axis=(1, 2)).astype(np.float64), "numpy_central_difference"


def border_background_rgb(image: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            image[0, :, :],
            image[-1, :, :],
            image[:, 0, :],
            image[:, -1, :],
        ],
        axis=0,
    )
    return np.median(border.astype(np.float32), axis=0)


def foreground_mask(image: np.ndarray, *, threshold: float) -> np.ndarray:
    background = border_background_rgb(image)
    dist = np.linalg.norm(image.astype(np.float32) - background[None, None, :], axis=2)
    return dist > float(threshold)


def foreground_mass(images: np.ndarray, *, threshold: float) -> np.ndarray:
    return np.asarray(
        [float(np.count_nonzero(foreground_mask(image, threshold=threshold))) for image in images],
        dtype=np.float64,
    )


def clean_mask(mask: np.ndarray) -> tuple[np.ndarray, str]:
    cv2 = cv2_module()
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned.astype(bool), "cv2_morphology"
    return mask.astype(bool), "none"


def component_centroid_cv2(mask: np.ndarray, *, min_area: int) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    cv2 = cv2_module()
    if cv2 is None:
        return None, {"method": "cv2_unavailable"}
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    candidates = []
    height, width = mask.shape
    for label in range(1, int(n_labels)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        comp_width = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = left <= 0 or top <= 0 or left + comp_width >= width or top + comp_height >= height
        if touches_border:
            continue
        candidates.append((area, label))
    if not candidates:
        return None, {"method": "cv2.connectedComponentsWithStats", "n_labels": int(n_labels), "n_candidates": 0}
    _, label = max(candidates)
    x, y = centroids[label]
    return (float(x), float(y)), {
        "method": "cv2.connectedComponentsWithStats",
        "n_labels": int(n_labels),
        "n_candidates": int(len(candidates)),
        "selected_area": int(stats[label, cv2.CC_STAT_AREA]),
    }


def component_centroid_scipy(mask: np.ndarray, *, min_area: int) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    ndimage = scipy_ndimage_module()
    if ndimage is None:
        return None, {"method": "scipy_unavailable"}
    labels, n_labels = ndimage.label(mask)
    height, width = mask.shape
    candidates = []
    for label in range(1, int(n_labels) + 1):
        ys, xs = np.nonzero(labels == label)
        area = int(len(xs))
        if area < int(min_area):
            continue
        touches_border = xs.min() <= 0 or ys.min() <= 0 or xs.max() >= width - 1 or ys.max() >= height - 1
        if touches_border:
            continue
        candidates.append((area, label, xs, ys))
    if not candidates:
        return None, {"method": "scipy.ndimage.label", "n_labels": int(n_labels), "n_candidates": 0}
    area, _label, xs, ys = max(candidates, key=lambda item: item[0])
    return (float(xs.mean()), float(ys.mean())), {
        "method": "scipy.ndimage.label",
        "n_labels": int(n_labels),
        "n_candidates": int(len(candidates)),
        "selected_area": int(area),
    }


def component_centroid_numpy(mask: np.ndarray, *, min_area: int) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    best: tuple[int, list[tuple[int, int]]] | None = None
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy = y + dy
                    xx = x + dx
                    if yy < 0 or yy >= height or xx < 0 or xx >= width:
                        continue
                    if visited[yy, xx] or not mask[yy, xx]:
                        continue
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        area = len(component)
        if area < int(min_area):
            continue
        ys = np.asarray([item[0] for item in component], dtype=np.int64)
        xs = np.asarray([item[1] for item in component], dtype=np.int64)
        touches_border = xs.min() <= 0 or ys.min() <= 0 or xs.max() >= width - 1 or ys.max() >= height - 1
        if touches_border:
            continue
        if best is None or area > best[0]:
            best = (area, component)
    if best is None:
        return None, {"method": "numpy_flood_fill", "n_candidates": 0}
    area, component = best
    ys = np.asarray([item[0] for item in component], dtype=np.float64)
    xs = np.asarray([item[1] for item in component], dtype=np.float64)
    return (float(xs.mean()), float(ys.mean())), {
        "method": "numpy_flood_fill",
        "n_candidates": 1,
        "selected_area": int(area),
    }


def estimate_block_center(
    image: np.ndarray,
    *,
    threshold: float,
    min_area: int,
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    mask = foreground_mask(image, threshold=threshold)
    mask, cleanup_method = clean_mask(mask)
    if not mask.any():
        return None, {"status": "failed", "reason": "empty_foreground_mask", "cleanup_method": cleanup_method}
    for finder in (component_centroid_cv2, component_centroid_scipy, component_centroid_numpy):
        center, metadata = finder(mask, min_area=min_area)
        if center is not None:
            metadata.update(
                {
                    "status": "ok",
                    "cleanup_method": cleanup_method,
                    "foreground_pixels": int(np.count_nonzero(mask)),
                }
            )
            return center, metadata
    return None, {
        "status": "failed",
        "reason": "no_component_after_filters",
        "cleanup_method": cleanup_method,
        "foreground_pixels": int(np.count_nonzero(mask)),
    }


def block_center_distance(
    terminal: np.ndarray,
    goal: np.ndarray,
    *,
    threshold: float,
    min_area: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    costs = np.full((terminal.shape[0],), np.nan, dtype=np.float64)
    terminal_failures = 0
    goal_failures = 0
    pair_failures = 0
    examples = []
    method_counts: dict[str, int] = {}
    for idx, (terminal_image, goal_image) in enumerate(zip(terminal, goal, strict=True)):
        terminal_center, terminal_meta = estimate_block_center(
            terminal_image,
            threshold=threshold,
            min_area=min_area,
        )
        goal_center, goal_meta = estimate_block_center(
            goal_image,
            threshold=threshold,
            min_area=min_area,
        )
        for meta in (terminal_meta, goal_meta):
            method = str(meta.get("method", meta.get("reason", "unknown")))
            method_counts[method] = method_counts.get(method, 0) + 1
        if terminal_center is None:
            terminal_failures += 1
        if goal_center is None:
            goal_failures += 1
        if terminal_center is None or goal_center is None:
            pair_failures += 1
            if len(examples) < MAX_EXAMPLE_FAILURES:
                examples.append(
                    {
                        "row": int(idx),
                        "terminal_status": terminal_meta,
                        "goal_status": goal_meta,
                    }
                )
            continue
        costs[idx] = float(np.linalg.norm(np.asarray(terminal_center) - np.asarray(goal_center)))
    metadata = {
        "feature": "pixel_threshold_connected_component_center",
        "cost": "euclidean_distance_between_terminal_and_goal_centers",
        "angle_estimation": "dropped_center_only",
        "uses_simulator_state": False,
        "foreground_threshold": float(threshold),
        "min_component_area": int(min_area),
        "n_records": int(len(costs)),
        "n_finite": int(np.isfinite(costs).sum()),
        "n_nan": int(np.isnan(costs).sum()),
        "terminal_center_failures": int(terminal_failures),
        "goal_center_failures": int(goal_failures),
        "pair_failures": int(pair_failures),
        "method_counts": method_counts,
        "example_failures": examples,
    }
    return costs, metadata


def finite_stats(costs: np.ndarray) -> dict[str, Any]:
    finite = costs[np.isfinite(costs)]
    return {
        "shape": list(costs.shape),
        "finite": bool(np.isfinite(costs).all()),
        "n_finite": int(len(finite)),
        "n_nan": int(np.isnan(costs).sum()),
        "min": clean_float(finite.min()) if len(finite) else None,
        "max": clean_float(finite.max()) if len(finite) else None,
        "mean": clean_float(finite.mean()) if len(finite) else None,
    }


def metrics_for_costs(costs: np.ndarray, common: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    costs = np.asarray(costs, dtype=np.float64)
    finite_mask = np.isfinite(costs)
    if not finite_mask.any():
        raise RuntimeError("No finite costs available for metric computation")
    filtered_anchor_masks = {
        name: mask[finite_mask]
        for name, mask in common["anchor_masks"].items()
    }
    filtered_pair_ids = common["pair_ids"][finite_mask]
    skipped_pairs = sorted(set(int(item) for item in common["pair_ids"][~finite_mask].tolist()))
    metrics = compute_metrics(
        costs=costs[finite_mask],
        labels=common["labels"][finite_mask],
        v1_cost=common["v1_cost"][finite_mask],
        c0_cost=common["c0_cost"][finite_mask],
        success=common["success"][finite_mask],
        pair_ids=filtered_pair_ids,
        action_ids=common["action_ids"][finite_mask],
        cells=common["cells"][finite_mask],
        anchor_masks=filtered_anchor_masks,
    )
    filter_metadata = {
        "n_original": int(len(costs)),
        "n_finite": int(finite_mask.sum()),
        "n_nan": int((~finite_mask).sum()),
        "n_pairs_with_finite_costs": int(len(np.unique(filtered_pair_ids))),
        "skipped_pair_ids": skipped_pairs,
    }
    return metrics, filter_metadata


def result_block(
    *,
    name: str,
    costs: np.ndarray,
    common: dict[str, Any],
    cost_metadata: dict[str, Any] | None = None,
    extraction_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics, filter_metadata = metrics_for_costs(costs, common)
    block = {
        "cost_metadata": {
            "name": name,
            **(cost_metadata or {}),
            "finite_filter": filter_metadata,
        },
        "validation": finite_stats(costs),
        "metrics": metrics,
    }
    if extraction_metadata is not None:
        block["extraction_metadata"] = extraction_metadata
    return block


def edge_density_diff(terminal: np.ndarray, goal: np.ndarray, *, threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    terminal_density, method = edge_density(terminal, threshold=threshold)
    goal_density, goal_method = edge_density(goal, threshold=threshold)
    if method != goal_method:
        raise RuntimeError(f"Edge method mismatch: terminal={method}, goal={goal_method}")
    return np.abs(terminal_density - goal_density).astype(np.float64), {
        "edge_density_method": method,
        "edge_threshold": float(threshold),
    }


def foreground_mass_diff(terminal: np.ndarray, goal: np.ndarray, *, threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    terminal_mass = foreground_mass(terminal, threshold=threshold)
    goal_mass = foreground_mass(goal, threshold=threshold)
    return np.abs(terminal_mass - goal_mass).astype(np.float64), {
        "foreground_method": "border_median_rgb_distance",
        "foreground_threshold": float(threshold),
    }


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pixel_artifact = args.pixel_artifact.expanduser().resolve()

    latent_artifact = load_latent_artifact(args.latent_artifact)
    common = build_common(latent_artifact)

    print("Preparing raw C6 S4/S5 pixels once...")
    raw_pixels = prepare_raw_pixels_once(args=args, latent_artifact=latent_artifact)

    print("Computing S4 raw-pixel cost signals...")
    edge_cost, edge_metadata = edge_density_diff(
        raw_pixels.terminal,
        raw_pixels.goal,
        threshold=float(args.edge_threshold),
    )
    foreground_cost, foreground_metadata = foreground_mass_diff(
        raw_pixels.terminal,
        raw_pixels.goal,
        threshold=float(args.foreground_threshold),
    )
    s4 = {
        "raw_pixel_l2": result_block(
            name="raw_pixel_l2",
            costs=raw_pixel_l2(raw_pixels.terminal, raw_pixels.goal),
            common=common,
            cost_metadata={"cost": "sum((terminal_rgb - goal_rgb)^2)"},
        ),
        "mean_rgb_diff": result_block(
            name="mean_rgb_diff",
            costs=mean_rgb_diff(raw_pixels.terminal, raw_pixels.goal),
            common=common,
            cost_metadata={"cost": "mean(abs(terminal_rgb - goal_rgb))"},
        ),
        "edge_density_diff": result_block(
            name="edge_density_diff",
            costs=edge_cost,
            common=common,
            cost_metadata={"cost": "abs(edge_density_terminal - edge_density_goal)", **edge_metadata},
        ),
        "foreground_mass_diff": result_block(
            name="foreground_mass_diff",
            costs=foreground_cost,
            common=common,
            cost_metadata={"cost": "abs(foreground_pixel_count_terminal - foreground_pixel_count_goal)", **foreground_metadata},
        ),
    }

    print("Computing S5 pixel-derived block-center cost signal...")
    center_cost, extraction_metadata = block_center_distance(
        raw_pixels.terminal,
        raw_pixels.goal,
        threshold=float(args.foreground_threshold),
        min_area=int(args.min_component_area),
    )
    s5 = {
        "block_center_distance": result_block(
            name="block_center_distance",
            costs=center_cost,
            common=common,
            cost_metadata={
                "cost": "euclidean_distance_between_pixel_derived_terminal_and_goal_block_centers",
                "nan_policy": "records with failed center extraction keep NaN cost and are filtered before metric computation",
            },
            extraction_metadata=extraction_metadata,
        )
    }

    summary_rows = [
        summary_row(control="C6_S4", config="raw_pixel_l2", n_seeds=1, metrics=s4["raw_pixel_l2"]["metrics"]),
        summary_row(control="C6_S4", config="mean_rgb_diff", n_seeds=1, metrics=s4["mean_rgb_diff"]["metrics"]),
        summary_row(control="C6_S4", config="edge_density_diff", n_seeds=1, metrics=s4["edge_density_diff"]["metrics"]),
        summary_row(control="C6_S4", config="foreground_mass_diff", n_seeds=1, metrics=s4["foreground_mass_diff"]["metrics"]),
        summary_row(control="C6_S5", config="block_center_distance", n_seeds=1, metrics=s5["block_center_distance"]["metrics"]),
    ]
    output = {
        "metadata": {
            "format": "c6_audit_s4_s5_results",
            "created_at": iso_now(),
            "sub_experiments": ["S4", "S5"],
            "description": "Raw-pixel and pixel-only hand-crafted baselines for C6 audit.",
            "latent_artifact": str(args.latent_artifact),
            "output": str(args.output),
            "pixel_artifact": str(args.pixel_artifact),
            "n_records": EXPECTED_RECORDS,
            "latent_dim": LATENT_DIM,
            "topk_values": list(TOPK_VALUES),
            "false_elite_k": FALSE_ELITE_K,
            "anchor_definitions": audit_anchor_definitions(common["pair_ids"], common["cells"]),
            "pixel_preparation": raw_pixels.metadata,
            "image_processing": {
                "cv2_available": cv2_module() is not None,
                "scipy_ndimage_available": scipy_ndimage_module() is not None,
                "foreground_threshold": float(args.foreground_threshold),
                "edge_threshold": float(args.edge_threshold),
                "min_component_area": int(args.min_component_area),
            },
            "metric_helpers_reused": stage1a_helper_names(),
            "uses_neural_network": False,
            "uses_simulator_state_for_features": False,
        },
        "S4": s4,
        "S5": s5,
        "summary_table": summary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2, allow_nan=False) + "\n")
    print_summary_table(summary_rows)
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
