#!/usr/bin/env python3
"""Offline orthogonal projection sanity check for Block 2.1.

This script reads saved PushT re-rank-only pool tensors, verifies that a
same-dimensional orthogonal transform preserves LeWM rank-1 selection, and
writes a compact markdown report. It does not load checkpoints, models, or
simulators.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_pools"
DEFAULT_RERANK_JSON = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "phase2" / "orthogonal_sanity_check.md"
LATENT_DIM = 192
POOL_SIZE = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--rerank-json", type=Path, default=DEFAULT_RERANK_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-pools", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-atol", type=float, default=1e-3)
    parser.add_argument("--cost-rtol", type=float, default=1e-6)
    parser.add_argument("--gaussian-tolerance-pp", type=float, default=2.0)
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pair_id_from_pool_path(path: Path) -> int:
    match = re.fullmatch(r"pair_(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"Pool filename does not match pair_<id>.pt: {path}")
    return int(match.group(1))


def sorted_pool_paths(pool_dir: Path, n_pools: int) -> list[Path]:
    paths = sorted(pool_dir.glob("pair_*.pt"), key=pair_id_from_pool_path)
    if len(paths) < n_pools:
        raise FileNotFoundError(f"Expected at least {n_pools} pool files in {pool_dir}, found {len(paths)}")
    return paths[:n_pools]


def require_tensor_shape(pool: dict[str, Any], key: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = pool.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"Pool key {key!r} is missing or is not a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"Pool key {key!r} has shape {tuple(value.shape)}, expected {shape}")
    return value


def validate_pool(pool: dict[str, Any], path: Path) -> dict[str, torch.Tensor | int]:
    z_pred = require_tensor_shape(pool, "z_pred", (POOL_SIZE, LATENT_DIM))
    z_goal = require_tensor_shape(pool, "z_goal", (LATENT_DIM,))
    default_costs = require_tensor_shape(pool, "default_costs", (POOL_SIZE,))
    default_rank1 = pool.get("default_rank1_candidate_index")
    if not isinstance(default_rank1, int):
        raise TypeError(f"{path} is missing integer default_rank1_candidate_index")
    if not 0 <= int(default_rank1) < POOL_SIZE:
        raise ValueError(f"{path} default_rank1_candidate_index is out of range: {default_rank1}")
    return {
        "z_pred": z_pred,
        "z_goal": z_goal,
        "default_costs": default_costs,
        "default_rank1": int(default_rank1),
    }


def orthogonal_matrix(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn((LATENT_DIM, LATENT_DIM), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(gaussian)
    return q


def check_pool(path: Path, q: torch.Tensor, cost_atol: float, cost_rtol: float) -> dict[str, Any]:
    pool = validate_pool(torch_load(path), path)
    z_pred = pool["z_pred"].to(dtype=torch.float64)
    z_goal = pool["z_goal"].to(dtype=torch.float64)
    default_costs = pool["default_costs"].to(dtype=torch.float64)
    saved_rank1 = int(pool["default_rank1"])

    diff = z_pred - z_goal.unsqueeze(0)
    identity_costs = torch.sum(diff**2, dim=1)
    orthogonal_costs = torch.sum((diff @ q) ** 2, dim=1)

    default_argmin = int(torch.argmin(default_costs).item())
    orthogonal_argmin = int(torch.argmin(orthogonal_costs).item())

    identity_diff = torch.max(torch.abs(identity_costs - default_costs)).item()
    orthogonal_diff = torch.max(torch.abs(orthogonal_costs - default_costs)).item()
    scale = max(1.0, float(torch.max(torch.abs(default_costs)).item()))
    cost_tolerance = max(float(cost_atol), float(cost_rtol) * scale)
    saved_matches_default = saved_rank1 == default_argmin
    rank1_match = default_argmin == orthogonal_argmin
    cost_match = orthogonal_diff <= cost_tolerance

    return {
        "pair_id": pair_id_from_pool_path(path),
        "path": str(path.relative_to(PROJECT_ROOT)),
        "saved_rank1": saved_rank1,
        "default_argmin": default_argmin,
        "orthogonal_argmin": orthogonal_argmin,
        "max_identity_default_abs_diff": clean_float(identity_diff),
        "max_orthogonal_default_abs_diff": clean_float(orthogonal_diff),
        "cost_tolerance": clean_float(cost_tolerance),
        "saved_matches_default": bool(saved_matches_default),
        "rank1_match": bool(rank1_match),
        "cost_match": bool(cost_match),
        "pass": bool(saved_matches_default and rank1_match and cost_match),
    }


def gaussian_m192_smoke(path: Path, tolerance_pp: float) -> dict[str, Any]:
    data = load_json(path)
    default_success = data["aggregate"]["default_baselines"]["rank1_success_rate"]["mean"]
    projected_success = data["aggregate"]["by_dimension"]["192"]["projected_success_rate"]["mean"]
    gap_pp = 100.0 * (float(projected_success) - float(default_success))
    return {
        "default_success": clean_float(default_success),
        "gaussian_m192_success": clean_float(projected_success),
        "gap_pp": clean_float(gap_pp),
        "tolerance_pp": clean_float(tolerance_pp),
        "pass": bool(abs(gap_pp) <= float(tolerance_pp)),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def write_report(
    *,
    output: Path,
    pool_rows: list[dict[str, Any]],
    gaussian_smoke: dict[str, Any],
    q_error: float,
    seed: int,
    pool_dir: Path,
    rerank_json: Path,
) -> None:
    match_rate = sum(1 for row in pool_rows if row["rank1_match"]) / float(len(pool_rows))
    pass_rate = sum(1 for row in pool_rows if row["pass"]) / float(len(pool_rows))
    overall_pass = pass_rate == 1.0 and bool(gaussian_smoke["pass"])

    orth_rows = [
        [
            row["pair_id"],
            row["saved_rank1"],
            row["default_argmin"],
            row["orthogonal_argmin"],
            fmt(row["max_identity_default_abs_diff"], 4),
            fmt(row["max_orthogonal_default_abs_diff"], 4),
            fmt(row["rank1_match"]),
            fmt(row["pass"]),
        ]
        for row in pool_rows
    ]
    gaussian_rows = [
        [
            "PushT re-rank-only",
            f"{100.0 * gaussian_smoke['default_success']:.2f}%",
            f"{100.0 * gaussian_smoke['gaussian_m192_success']:.2f}%",
            f"{gaussian_smoke['gap_pp']:+.2f}",
            f"{gaussian_smoke['tolerance_pp']:.2f}",
            fmt(gaussian_smoke["pass"]),
        ]
    ]

    text = "\n".join(
        [
            "# Orthogonal Projection Sanity Check",
            "",
            f"Generated: `{iso_now()}`",
            f"Git commit: `{get_git_commit()}`",
            f"Overall status: **{fmt(overall_pass)}**",
            "",
            "This is an offline Block 2.1 check. It reads saved PushT pool tensors and does not load a simulator, GPU, checkpoint, or policy.",
            "",
            "## Orthogonal Identity Check",
            "",
            f"Pool dir: `{pool_dir.relative_to(PROJECT_ROOT)}`",
            f"Seed: `{seed}`",
            f"Max `|Q^T Q - I|`: `{q_error:.3e}`",
            f"Rank-1 match rate: `{match_rate:.3f}`",
            "",
            markdown_table(
                [
                    "Pair",
                    "Saved Rank-1",
                    "Default Argmin",
                    "Orthogonal Argmin",
                    "Max Identity-Default",
                    "Max Orthogonal-Default",
                    "Rank Match",
                    "Pass",
                ],
                orth_rows,
            ),
            "",
            "## Gaussian m=192 Smoke",
            "",
            f"Source: `{rerank_json.relative_to(PROJECT_ROOT)}`",
            "",
            markdown_table(
                ["Protocol", "Default Success", "Gaussian m=192 Success", "Gap pp", "Tolerance pp", "Pass"],
                gaussian_rows,
            ),
            "",
            "## Cube Note",
            "",
            "Cube orthogonal tensor sanity is skipped because no Cube pool `.pt` artifacts equivalent to PushT `pusht_pools/pair_*.pt` are currently available. The Cube m=192 Gaussian gap remains a JSON-level smoke comparison, not an orthogonal identity check.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> int:
    args = parse_args()
    if args.n_pools < 1:
        raise ValueError("--n-pools must be positive")

    q = orthogonal_matrix(args.seed)
    q_error = float(torch.max(torch.abs(q.T @ q - torch.eye(LATENT_DIM, dtype=torch.float64))).item())
    pool_rows = [
        check_pool(path, q=q, cost_atol=args.cost_atol, cost_rtol=args.cost_rtol)
        for path in sorted_pool_paths(args.pool_dir, args.n_pools)
    ]
    gaussian_smoke = gaussian_m192_smoke(args.rerank_json, args.gaussian_tolerance_pp)
    write_report(
        output=args.output,
        pool_rows=pool_rows,
        gaussian_smoke=gaussian_smoke,
        q_error=q_error,
        seed=args.seed,
        pool_dir=args.pool_dir,
        rerank_json=args.rerank_json,
    )

    match_rate = sum(1 for row in pool_rows if row["rank1_match"]) / float(len(pool_rows))
    print(f"Wrote {args.output}")
    print(f"Orthogonal rank-1 match rate: {match_rate:.3f}")
    print(f"Gaussian m=192 gap: {gaussian_smoke['gap_pp']:+.2f} pp")
    return 0 if match_rate == 1.0 and bool(gaussian_smoke["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
