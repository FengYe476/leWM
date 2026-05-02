#!/usr/bin/env python3
"""Data loading and pairwise-ranking triples for Phase 2 P2-0.

The current Phase 1 JSON files store scalar costs and success flags, but they
do not store terminal latent vectors, terminal pixels/states, or raw action
sequences. This module detects that explicitly. If future JSONs include
terminal latents, it can build training triples directly; if they include
terminal observations, use ``load_lewm_encoder`` and
``materialize_latent_examples`` to re-encode them with the frozen LeWM encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_THREE_COST_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_three_cost.json"
DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
DEFAULT_LATENT_ARTIFACT = PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_latents.pt"
DEFAULT_PREDICTED_LATENT_ARTIFACT = (
    PROJECT_ROOT / "results" / "phase2" / "p2_0" / "track_a_predicted_latents.pt"
)
DEFAULT_V1_PATHS = tuple(
    PROJECT_ROOT / "results" / "phase1" / "v1_oracle_ablation" / f"v1_d{idx}.json"
    for idx in range(4)
)

SEED = 0
LATENT_DIM = 192
LATENT_KEYS = ("z", "latent", "emb", "terminal_z", "terminal_latent", "terminal_emb")
GOAL_LATENT_KEYS = ("z_g", "goal_z", "goal_latent", "goal_emb", "goal_embedding")
PIXEL_KEYS = ("pixels", "terminal_pixels", "terminal_image", "image")
GOAL_PIXEL_KEYS = ("goal_pixels", "goal_image", "goal")
STATE_KEYS = ("state", "terminal_state")
GOAL_STATE_KEYS = ("goal_state",)


class LatentUnavailableError(RuntimeError):
    """Raised when Phase 2 examples cannot be built from the available JSON."""


@dataclass(frozen=True)
class EncoderBundle:
    """Frozen LeWM encoder objects needed to encode stored observations."""

    policy: Any
    model: Any
    device: str
    dataset: Any | None = None


@dataclass(frozen=True)
class LatentExample:
    """One scored terminal latent for a single pair/action candidate."""

    pair_id: int
    source: str
    source_index: int
    action_id: int
    z: np.ndarray
    z_g: np.ndarray
    v1_cost: float
    latent_type: str = "encoder"


def load_json(path: Path) -> dict:
    """Load a JSON object from ``path``."""
    return json.loads(path.read_text())


def load_latent_artifact(path: Path = DEFAULT_LATENT_ARTIFACT) -> dict:
    """Load a Phase 2 latent artifact produced by ``extract_latents.py``."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "pair_id",
        "action_id",
        "source",
        "source_index",
        "z_terminal",
        "z_goal",
        "v1_cost",
        "success",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Latent artifact {path} missing keys: {missing}")
    if artifact["z_terminal"].shape[-1] != LATENT_DIM:
        raise ValueError(
            f"Expected terminal latent dim {LATENT_DIM}, got {artifact['z_terminal'].shape}"
        )
    if artifact["z_goal"].shape[-1] != LATENT_DIM:
        raise ValueError(f"Expected goal latent dim {LATENT_DIM}, got {artifact['z_goal'].shape}")
    return artifact


def load_predicted_latent_artifact(
    path: Path = DEFAULT_PREDICTED_LATENT_ARTIFACT,
) -> dict:
    """Load a predictor-latent artifact produced by ``extract_predicted_latents.py``."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {"pair_id", "action_id", "source", "source_index", "z_predicted"}
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Predicted latent artifact {path} missing keys: {missing}")
    if artifact["z_predicted"].shape[-1] != LATENT_DIM:
        raise ValueError(
            f"Expected predicted latent dim {LATENT_DIM}, got {artifact['z_predicted'].shape}"
        )
    return artifact


def latent_examples_from_artifact(
    artifact: dict,
    *,
    pair_ids: set[int] | None = None,
    latent_key: str = "z_terminal",
    latent_type: str = "encoder",
) -> list[LatentExample]:
    """Convert a saved latent artifact into scored latent examples."""
    if latent_key not in artifact:
        raise ValueError(f"Latent artifact missing key {latent_key!r}")
    examples = []
    n_records = int(artifact["pair_id"].numel())
    for idx in range(n_records):
        pair_id = int(artifact["pair_id"][idx])
        if pair_ids is not None and pair_id not in pair_ids:
            continue
        examples.append(
            LatentExample(
                pair_id=pair_id,
                source=str(artifact["source"][idx]),
                source_index=int(artifact["source_index"][idx]),
                action_id=int(artifact["action_id"][idx]),
                z=artifact[latent_key][idx].cpu().numpy().astype(np.float32),
                z_g=artifact["z_goal"][idx].cpu().numpy().astype(np.float32),
                v1_cost=float(artifact["v1_cost"][idx]),
                latent_type=latent_type,
            )
        )
    return examples


def validate_predicted_artifact_alignment(real_artifact: dict, pred_artifact: dict) -> None:
    """Assert predicted latents align record-for-record with the real artifact."""
    keys = ("pair_id", "action_id", "source_index")
    for key in keys:
        if not torch.equal(real_artifact[key].cpu(), pred_artifact[key].cpu()):
            raise ValueError(f"Predicted artifact {key} does not match real artifact")
    for idx, (real_source, pred_source) in enumerate(
        zip(real_artifact["source"], pred_artifact["source"], strict=True)
    ):
        if str(real_source) != str(pred_source):
            raise ValueError(
                f"Predicted artifact source mismatch at idx={idx}: "
                f"{real_source!r} != {pred_source!r}"
            )


def mixed_latent_examples_from_artifacts(
    real_artifact: dict,
    pred_artifact: dict,
    *,
    pair_ids: set[int] | None = None,
) -> list[LatentExample]:
    """Return encoder and predictor latent examples for mixed-latent training."""
    validate_predicted_artifact_alignment(real_artifact, pred_artifact)
    examples = latent_examples_from_artifact(
        real_artifact,
        pair_ids=pair_ids,
        latent_key="z_terminal",
        latent_type="encoder",
    )
    pred_joined = dict(real_artifact)
    pred_joined["z_predicted"] = pred_artifact["z_predicted"]
    examples.extend(
        latent_examples_from_artifact(
            pred_joined,
            pair_ids=pair_ids,
            latent_key="z_predicted",
            latent_type="predictor",
        )
    )
    return examples


def artifact_record_count_by_pair(artifact: dict) -> dict[int, int]:
    """Return number of saved action records per pair ID."""
    counts: dict[int, int] = {}
    for pair_id_t in artifact["pair_id"]:
        pair_id = int(pair_id_t)
        counts[pair_id] = counts.get(pair_id, 0) + 1
    return counts


def _pair_metadata(pair: dict) -> dict:
    """Return pair metadata without the bulky action list."""
    return {key: value for key, value in pair.items() if key != "actions"}


def flatten_action_records(data: dict, *, cost_key: str | None = None) -> list[dict]:
    """Flatten ``{"pairs": [{"actions": ...}]}`` JSON into action records.

    Missing ``source_index`` values are inferred by action order within each
    ``(pair_id, source)`` bucket, which is the Track A three-cost schema.
    """
    records = []
    for pair in data.get("pairs", []):
        pair_meta = _pair_metadata(pair)
        source_counts: dict[str, int] = {}
        for action in pair.get("actions", []):
            source = str(action["source"])
            inferred_source_index = source_counts.get(source, 0)
            source_counts[source] = inferred_source_index + 1
            source_index = int(action.get("source_index", inferred_source_index))
            record = {
                "pair_id": int(pair["pair_id"]),
                "cell": str(pair["cell"]),
                "source": source,
                "source_index": source_index,
                "action_id": int(action.get("action_id", len(records))),
                "pair": pair_meta,
                "action": action,
            }
            if cost_key is not None and cost_key in action:
                record["cost"] = float(action[cost_key])
            records.append(record)
    return records


def load_track_a_three_cost(path: Path = DEFAULT_THREE_COST_PATH) -> list[dict]:
    """Load and flatten Track A three-cost records."""
    return flatten_action_records(load_json(path))


def load_v1_oracle_records(paths: tuple[Path, ...] = DEFAULT_V1_PATHS) -> list[dict]:
    """Load and flatten V1 oracle-CEM records with ``C_variant`` labels."""
    records = []
    seen_pair_ids: set[int] = set()
    for path in paths:
        data = load_json(path)
        variant = data.get("metadata", {}).get("variant")
        if variant != "V1":
            raise ValueError(f"Expected V1 metadata in {path}, got {variant!r}")
        for record in flatten_action_records(data, cost_key="C_variant"):
            pair_id = int(record["pair_id"])
            records.append(record)
            seen_pair_ids.add(pair_id)
    if len(seen_pair_ids) != 100:
        raise ValueError(f"Expected V1 records for 100 pairs, got {len(seen_pair_ids)}")
    return sorted(
        records,
        key=lambda item: (
            int(item["pair_id"]),
            str(item["source"]),
            int(item["source_index"]),
            int(item["action_id"]),
        ),
    )


def _strip_variant_suffix(source: str) -> str:
    """Return source without a trailing oracle variant suffix."""
    for suffix in ("_V1", "_V2", "_V3"):
        if source.endswith(suffix):
            return source[: -len(suffix)]
    return source


def action_join_key(record: dict, *, strip_variant_suffix: bool = False) -> tuple[int, str, int]:
    """Return a schema-level action key for diagnostics and joins."""
    source = str(record["source"])
    if strip_variant_suffix:
        source = _strip_variant_suffix(source)
    return int(record["pair_id"]), source, int(record["source_index"])


def join_records_by_action_key(
    left: list[dict],
    right: list[dict],
    *,
    strip_variant_suffix: bool = False,
) -> list[tuple[dict, dict]]:
    """Join two flattened record lists by ``(pair_id, source, source_index)``.

    Stripping oracle suffixes is useful for schema diagnostics only: V1 oracle
    CEM candidates are generated by a different planner from Track A latent CEM
    candidates, so matching keys do not prove action identity.
    """
    right_by_key = {
        action_join_key(record, strip_variant_suffix=strip_variant_suffix): record
        for record in right
    }
    joined = []
    for record in left:
        match = right_by_key.get(
            action_join_key(record, strip_variant_suffix=strip_variant_suffix)
        )
        if match is not None:
            joined.append((record, match))
    return joined


def _vector_from_mapping(mapping: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    """Extract a 192-d vector from one of ``keys`` if present."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list) and len(value) == LATENT_DIM:
            return np.asarray(value, dtype=np.float32)
        if isinstance(value, np.ndarray) and value.shape[-1] == LATENT_DIM:
            return np.asarray(value, dtype=np.float32)
    return None


def _array_from_mapping(mapping: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    """Extract a numeric array from one of ``keys`` if present."""
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if isinstance(value, list | tuple | np.ndarray):
                return np.asarray(value)
    return None


def _first_not_none(*values):
    """Return the first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def terminal_latent(record: dict) -> np.ndarray | None:
    """Return a stored terminal latent from a flattened record, if available."""
    return _vector_from_mapping(record["action"], LATENT_KEYS)


def goal_latent(record: dict) -> np.ndarray | None:
    """Return a stored goal latent from a flattened record, if available."""
    return _first_not_none(
        _vector_from_mapping(record["action"], GOAL_LATENT_KEYS),
        _vector_from_mapping(record["pair"], GOAL_LATENT_KEYS),
    )


def terminal_observation(record: dict) -> np.ndarray | None:
    """Return stored terminal pixels or state from an action record, if present."""
    return _first_not_none(
        _array_from_mapping(record["action"], PIXEL_KEYS),
        _array_from_mapping(record["action"], STATE_KEYS),
    )


def goal_observation(record: dict) -> np.ndarray | None:
    """Return stored goal pixels or state from a pair/action record, if present."""
    return _first_not_none(
        _array_from_mapping(record["action"], GOAL_PIXEL_KEYS),
        _array_from_mapping(record["pair"], GOAL_PIXEL_KEYS),
        _array_from_mapping(record["pair"], GOAL_STATE_KEYS),
    )


def latent_storage_report(records: list[dict]) -> dict[str, int]:
    """Count records that already contain latents or re-encodable observations."""
    return {
        "records": len(records),
        "terminal_latents": sum(terminal_latent(record) is not None for record in records),
        "goal_latents": sum(goal_latent(record) is not None for record in records),
        "terminal_observations": sum(terminal_observation(record) is not None for record in records),
        "goal_observations": sum(goal_observation(record) is not None for record in records),
    }


def load_lewm_encoder(
    *,
    checkpoint_dir: Path | None = None,
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    cache_dir: Path | None = None,
    dataset_name: str | None = None,
    device: str = "auto",
    seed: int = SEED,
) -> EncoderBundle:
    """Load the frozen LeWM encoder using the existing Phase 1 policy path.

    The policy object is required because it owns the image preprocessing used
    before ``model.encode``. Dataset loading is included so goal rows can be
    encoded later when records identify ``goal_row`` but do not store pixels.
    """
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))

    from eval_pusht_baseline import (  # noqa: PLC0415
        DEFAULT_CHECKPOINT_DIR,
        build_policy,
        build_processors,
        get_dataset,
        resolve_device,
    )

    pairs_data = load_json(pairs_path)
    if cache_dir is None or dataset_name is None:
        dataset_path = Path(pairs_data["metadata"]["dataset_path"])
        cache_dir = cache_dir or dataset_path.parent
        dataset_name = dataset_name or dataset_path.stem

    resolved_device = resolve_device(device)
    dataset = get_dataset(cache_dir.expanduser().resolve(), dataset_name)
    process = build_processors(dataset, ["action", "proprio", "state"])
    policy_args = argparse.Namespace(
        checkpoint_dir=(checkpoint_dir or DEFAULT_CHECKPOINT_DIR).expanduser().resolve(),
        device=resolved_device,
        num_samples=300,
        var_scale=1.0,
        cem_iters=30,
        topk=30,
        seed=seed,
        horizon=5,
        receding_horizon=5,
        action_block=5,
        img_size=224,
    )
    policy = build_policy(policy_args, process)
    return EncoderBundle(
        policy=policy,
        model=policy.solver.model,
        device=resolved_device,
        dataset=dataset,
    )


def encode_pixels_with_bundle(bundle: EncoderBundle, pixels: np.ndarray) -> np.ndarray:
    """Encode RGB pixels with the frozen LeWM encoder."""
    from lewm_audit.diagnostics.three_cost import encode_pixels  # noqa: PLC0415

    emb = encode_pixels(bundle.policy, bundle.model, np.asarray(pixels))
    return emb.detach().cpu().numpy()[0].astype(np.float32)


def render_pusht_state_pixels(
    state: np.ndarray,
    goal_state: np.ndarray,
    *,
    seed: int = SEED,
) -> np.ndarray:
    """Render PushT pixels for a stored simulator state and goal state."""
    import gymnasium as gym  # noqa: PLC0415
    import stable_worldmodel as swm  # noqa: F401, PLC0415
    from lewm_audit.diagnostics.three_cost import configure_goal_render  # noqa: PLC0415

    env = gym.make("swm/PushT-v1")
    try:
        env.reset(seed=seed)
        env_unwrapped = env.unwrapped
        configure_goal_render(env_unwrapped, np.asarray(goal_state, dtype=np.float32))
        env_unwrapped._set_state(np.asarray(state, dtype=np.float32))
        return np.asarray(env_unwrapped.render(), dtype=np.uint8)
    finally:
        env.close()


def encode_goal_from_dataset(bundle: EncoderBundle, record: dict) -> np.ndarray | None:
    """Encode a pair goal row from the cached PushT dataset, if possible."""
    if bundle.dataset is None:
        return None
    pair = record["pair"]
    goal_row = pair.get("goal_row")
    if goal_row is None:
        return None
    row = bundle.dataset.get_row_data([int(goal_row)])
    pixels = row.get("pixels")
    if pixels is None:
        return None
    return encode_pixels_with_bundle(bundle, pixels[0])


def _encode_observation(
    observation: np.ndarray | None,
    *,
    bundle: EncoderBundle,
    goal_state: np.ndarray | None = None,
) -> np.ndarray | None:
    """Encode stored pixels, or render and encode a stored simulator state."""
    if observation is None:
        return None
    arr = np.asarray(observation)
    if arr.ndim >= 3:
        return encode_pixels_with_bundle(bundle, arr)
    if arr.ndim == 1 and goal_state is not None:
        pixels = render_pusht_state_pixels(arr, goal_state)
        return encode_pixels_with_bundle(bundle, pixels)
    return None


def materialize_latent_examples(
    records: list[dict],
    *,
    encoder_bundle: EncoderBundle | None = None,
) -> list[LatentExample]:
    """Convert V1 action records into scored latent examples.

    Args:
        records: Flattened V1 oracle records.
        encoder_bundle: Optional frozen LeWM encoder for records that contain
            terminal pixels/states instead of terminal latents.

    Raises:
        LatentUnavailableError: if terminal or goal latents cannot be obtained.
    """
    examples = []
    missing = []
    for record in records:
        z = terminal_latent(record)
        z_g = goal_latent(record)
        if encoder_bundle is not None:
            goal_state = _array_from_mapping(record["pair"], GOAL_STATE_KEYS)
            if z_g is None:
                z_g = _encode_observation(
                    goal_observation(record),
                    bundle=encoder_bundle,
                    goal_state=goal_state,
                )
            if z_g is None:
                z_g = encode_goal_from_dataset(encoder_bundle, record)
            if z is None:
                z = _encode_observation(
                    terminal_observation(record),
                    bundle=encoder_bundle,
                    goal_state=goal_state,
                )
        if z is None or z_g is None:
            missing.append(
                (
                    int(record["pair_id"]),
                    str(record["source"]),
                    int(record["source_index"]),
                    z is not None,
                    z_g is not None,
                )
            )
            continue
        examples.append(
            LatentExample(
                pair_id=int(record["pair_id"]),
                source=str(record["source"]),
                source_index=int(record["source_index"]),
                action_id=int(record["action_id"]),
                z=np.asarray(z, dtype=np.float32),
                z_g=np.asarray(z_g, dtype=np.float32),
                v1_cost=float(record["cost"]),
            )
        )

    if missing:
        preview = missing[:5]
        raise LatentUnavailableError(
            "Cannot build Phase 2 latent examples from the current records. "
            "The Phase 1 Track A/V1 JSONs store scalar costs only; they do not "
            "store terminal z vectors, terminal pixels/states, or raw actions "
            "needed to replay rollouts. First produce an augmented artifact with "
            "terminal_emb/goal_emb, or terminal_pixels plus goal pixels/states. "
            f"Missing examples preview: {preview}"
        )
    return examples


def make_ranking_triples(
    examples: list[LatentExample],
    *,
    pair_ids: set[int] | None = None,
    max_triples_per_pair: int | None = None,
    seed: int = SEED,
    min_cost_gap: float = 0.0,
) -> list[dict]:
    """Sample pairwise rankings where ``V1_cost(a_i) < V1_cost(a_j)``."""
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[int, str], list[LatentExample]] = {}
    for example in examples:
        if pair_ids is not None and example.pair_id not in pair_ids:
            continue
        grouped.setdefault((example.pair_id, example.latent_type), []).append(example)

    triples = []
    for pair_id, latent_type in sorted(grouped):
        group = grouped[(pair_id, latent_type)]
        candidates = [
            (better, worse)
            for better in group
            for worse in group
            if better.v1_cost + min_cost_gap < worse.v1_cost
        ]
        if max_triples_per_pair is not None and len(candidates) > max_triples_per_pair:
            selected = rng.choice(len(candidates), size=max_triples_per_pair, replace=False)
            candidates = [candidates[int(idx)] for idx in selected]
        for better, worse in candidates:
            triples.append(
                {
                    "pair_id": pair_id,
                    "z_pos": better.z,
                    "z_neg": worse.z,
                    "z_g": better.z_g,
                    "cost_pos": better.v1_cost,
                    "cost_neg": worse.v1_cost,
                    "source_pos": better.source,
                    "source_neg": worse.source,
                    "latent_type": latent_type,
                }
            )
    return triples


class PairwiseRankingDataset(Dataset):
    """PyTorch dataset of ``(z_+, z_-, z_g)`` ranking triples."""

    def __init__(
        self,
        examples: list[LatentExample],
        *,
        pair_ids: set[int] | None = None,
        max_triples_per_pair: int | None = None,
        seed: int = SEED,
        min_cost_gap: float = 0.0,
    ) -> None:
        self.triples = make_ranking_triples(
            examples,
            pair_ids=pair_ids,
            max_triples_per_pair=max_triples_per_pair,
            seed=seed,
            min_cost_gap=min_cost_gap,
        )

    def __len__(self) -> int:
        """Return number of ranking triples."""
        return len(self.triples)

    def __getitem__(self, index: int) -> dict:
        """Return one ranking triple as tensors plus metadata."""
        item = self.triples[index]
        return {
            "pair_id": int(item["pair_id"]),
            "z_pos": torch.as_tensor(item["z_pos"], dtype=torch.float32),
            "z_neg": torch.as_tensor(item["z_neg"], dtype=torch.float32),
            "z_g": torch.as_tensor(item["z_g"], dtype=torch.float32),
            "cost_pos": torch.tensor(float(item["cost_pos"]), dtype=torch.float32),
            "cost_neg": torch.tensor(float(item["cost_neg"]), dtype=torch.float32),
        }


def collate_ranking_triples(batch: list[dict]) -> dict:
    """Collate ``PairwiseRankingDataset`` items into batched tensors."""
    return {
        "pair_id": torch.as_tensor([item["pair_id"] for item in batch], dtype=torch.long),
        "z_pos": torch.stack([item["z_pos"] for item in batch], dim=0),
        "z_neg": torch.stack([item["z_neg"] for item in batch], dim=0),
        "z_g": torch.stack([item["z_g"] for item in batch], dim=0),
        "cost_pos": torch.stack([item["cost_pos"] for item in batch], dim=0),
        "cost_neg": torch.stack([item["cost_neg"] for item in batch], dim=0),
    }


def main() -> int:
    """Print schema and latent-availability sanity checks."""
    if DEFAULT_LATENT_ARTIFACT.exists():
        artifact = load_latent_artifact(DEFAULT_LATENT_ARTIFACT)
        examples = latent_examples_from_artifact(artifact)
        dataset = PairwiseRankingDataset(examples, max_triples_per_pair=128)
        counts = artifact_record_count_by_pair(artifact)
        print(f"Latent artifact: {DEFAULT_LATENT_ARTIFACT}")
        print(f"Records: {len(examples)}")
        print(f"Pairs: {len(counts)}")
        print(f"Records per pair: min={min(counts.values())} max={max(counts.values())}")
        print(f"Ranking triples with cap=128/pair: {len(dataset)}")
        if len(dataset):
            batch = collate_ranking_triples([dataset[0]])
            print("First batch shapes:", {key: tuple(value.shape) for key, value in batch.items()})
        return 0

    track_records = load_track_a_three_cost()
    v1_records = load_v1_oracle_records()
    exact_join = join_records_by_action_key(track_records, v1_records)
    suffix_join = join_records_by_action_key(
        track_records,
        v1_records,
        strip_variant_suffix=True,
    )
    print(f"Track A records: {len(track_records)}")
    print(f"V1 oracle records: {len(v1_records)}")
    print(f"Exact action-key joins: {len(exact_join)}")
    print(f"Suffix-stripped schema joins: {len(suffix_join)}")
    print("Track A latent storage:", latent_storage_report(track_records))
    print("V1 latent storage:", latent_storage_report(v1_records))
    try:
        examples = materialize_latent_examples(v1_records)
        dataset = PairwiseRankingDataset(examples, max_triples_per_pair=128)
        print(f"Ranking triples: {len(dataset)}")
        if len(dataset):
            batch = collate_ranking_triples([dataset[0]])
            print("First batch shapes:", {key: tuple(value.shape) for key, value in batch.items()})
    except LatentUnavailableError as exc:
        print(f"Ranking dataset unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
