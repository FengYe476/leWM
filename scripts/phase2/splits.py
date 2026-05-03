#!/usr/bin/env python3
"""Pair-level train/validation/test splits for Phase 2 P2-0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
SEED = 0
SPLIT3_TEST_PAIR_IDS = [25, 46, 60, 61, 67, 70, 71, 73, 78, 86, 87, 93, 94, 96, 97, 99]


def load_track_a_pairs(path: Path = DEFAULT_PAIRS_PATH) -> list[dict]:
    """Load Track A pair metadata sorted by ``pair_id``."""
    data = json.loads(path.read_text())
    pairs = sorted(data["pairs"], key=lambda pair: int(pair["pair_id"]))
    return pairs


def pair_ids_from_metadata(pairs: list[dict]) -> list[int]:
    """Return sorted integer pair IDs from Track A metadata."""
    return sorted(int(pair["pair_id"]) for pair in pairs)


def cell_by_pair_id(pairs: list[dict]) -> dict[int, str]:
    """Return ``pair_id -> D/R cell`` mapping."""
    return {int(pair["pair_id"]): str(pair["cell"]) for pair in pairs}


def _seeded_val_split(
    train_pool: list[int],
    *,
    seed: int = SEED,
    val_fraction: float = 0.10,
) -> tuple[list[int], list[int]]:
    """Split a validation subset from a training pool using a fixed seed."""
    if not train_pool:
        raise ValueError("train_pool must not be empty")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(train_pool)).tolist()
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(int(pair_id) for pair_id in shuffled[:n_val])
    train_ids = sorted(int(pair_id) for pair_id in shuffled[n_val:])
    return train_ids, val_ids


def split1_random_holdout(
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    *,
    seed: int = SEED,
) -> dict[str, list[int]]:
    """Return the random 70/15/15 pair holdout split."""
    pair_ids = pair_ids_from_metadata(load_track_a_pairs(pairs_path))
    if len(pair_ids) != 100:
        raise ValueError(f"Expected 100 Track A pair IDs, got {len(pair_ids)}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pair_ids).tolist()
    return {
        "train_pair_ids": sorted(int(pair_id) for pair_id in shuffled[:70]),
        "val_pair_ids": sorted(int(pair_id) for pair_id in shuffled[70:85]),
        "test_pair_ids": sorted(int(pair_id) for pair_id in shuffled[85:]),
    }


def split2_leave_one_cell_out(
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    *,
    seed: int = SEED,
) -> dict[str, dict[str, list[int]]]:
    """Return 16 leave-one-cell-out folds keyed by held-out cell."""
    pairs = load_track_a_pairs(pairs_path)
    by_cell: dict[str, list[int]] = {}
    for pair in pairs:
        by_cell.setdefault(str(pair["cell"]), []).append(int(pair["pair_id"]))

    folds = {}
    all_ids = pair_ids_from_metadata(pairs)
    for cell in sorted(by_cell):
        test_ids = sorted(by_cell[cell])
        train_pool = sorted(pair_id for pair_id in all_ids if pair_id not in set(test_ids))
        train_ids, val_ids = _seeded_val_split(
            train_pool,
            seed=seed + sum(ord(ch) for ch in cell),
        )
        folds[cell] = {
            "train_pair_ids": train_ids,
            "val_pair_ids": val_ids,
            "test_pair_ids": test_ids,
        }
    if len(folds) != 16:
        raise ValueError(f"Expected 16 leave-one-cell-out folds, got {len(folds)}")
    return folds


def split3_hard_pair_holdout(
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    *,
    seed: int = SEED,
) -> dict[str, list[int]]:
    """Return the all_fail + strong_rho hard-pair holdout split."""
    pair_ids = pair_ids_from_metadata(load_track_a_pairs(pairs_path))
    test_set = set(SPLIT3_TEST_PAIR_IDS)
    missing = sorted(test_set - set(pair_ids))
    if missing:
        raise ValueError(f"Split 3 test pair IDs missing from Track A pairs: {missing}")
    train_pool = sorted(pair_id for pair_id in pair_ids if pair_id not in test_set)
    train_ids, val_ids = _seeded_val_split(train_pool, seed=seed)
    return {
        "train_pair_ids": train_ids,
        "val_pair_ids": val_ids,
        "test_pair_ids": sorted(test_set),
    }


def make_all_splits(pairs_path: Path = DEFAULT_PAIRS_PATH, *, seed: int = SEED) -> dict:
    """Return all Phase 2 split definitions."""
    return {
        "split1_random_70_15_15": split1_random_holdout(pairs_path, seed=seed),
        "split2_leave_one_cell_out": split2_leave_one_cell_out(pairs_path, seed=seed),
        "split3_all_fail_strong_rho": split3_hard_pair_holdout(pairs_path, seed=seed),
    }


def _counts(split: dict[str, list[int]]) -> dict[str, int]:
    return {name: len(ids) for name, ids in split.items()}


def main() -> int:
    """Print split sanity-check counts."""
    splits = make_all_splits()
    print("split1:", _counts(splits["split1_random_70_15_15"]))
    print("split2 folds:", len(splits["split2_leave_one_cell_out"]))
    for cell, fold in splits["split2_leave_one_cell_out"].items():
        print(cell, _counts(fold))
    print("split3:", _counts(splits["split3_all_fail_strong_rho"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
