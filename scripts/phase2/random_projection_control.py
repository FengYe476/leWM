#!/usr/bin/env python3
"""Create a fixed random-projection control for Track B.

This is the fallback/control path used when DINOv2 is unavailable. It applies a
seed-0 Gaussian linear map to LeWM latents and preserves the Track A ordering.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT_LOCAL = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_LOCAL))

from scripts.phase2.track_b_common import DEFAULT_LATENT_ARTIFACT, PROJECT_ROOT


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "phase2" / "track_b" / "random_projection_features.pt"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-artifact", type=Path, default=DEFAULT_LATENT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.latent_artifact = args.latent_artifact.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    torch.manual_seed(int(args.seed))

    artifact = torch.load(args.latent_artifact, map_location="cpu", weights_only=False)
    z_terminal = artifact["z_terminal"].to(dtype=torch.float32)
    z_goal = artifact["z_goal"].to(dtype=torch.float32)
    dim = int(z_terminal.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    projection = torch.randn((dim, dim), generator=generator, dtype=torch.float32) / (dim**0.5)
    r_terminal = z_terminal @ projection
    r_goal = z_goal @ projection

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "format": "phase2_track_b_random_projection_features",
                "created_at": iso_now(),
                "seed": int(args.seed),
                "latent_artifact": str(args.latent_artifact),
                "projection_shape": list(projection.shape),
                "note": "Gaussian R ~ N(0, 1/dim) applied as z @ R.",
            },
            "pair_id": artifact["pair_id"].clone(),
            "action_id": artifact["action_id"].clone(),
            "source": list(artifact["source"]),
            "source_index": artifact["source_index"].clone(),
            "cell": list(artifact["cell"]),
            "r_terminal": r_terminal,
            "r_goal": r_goal,
            "projection": projection,
        },
        args.output,
    )
    print(f"Saved: {args.output}")
    print(f"records: {int(r_terminal.shape[0])}")
    print(f"dim: {dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
