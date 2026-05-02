#!/usr/bin/env python3
"""Neural cost heads for Phase 2 LeWM planning experiments."""

from __future__ import annotations

import torch
from torch import nn


LATENT_DIM = 192
FEATURE_DIM = LATENT_DIM * 4


def cost_head_features(z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
    """Return concat([z, z_g, z - z_g, |z - z_g|]) features.

    Args:
        z: Terminal latent tensor with trailing dimension 192.
        z_g: Goal latent tensor broadcastable to ``z`` with trailing dimension 192.

    Returns:
        A tensor with trailing dimension 768.
    """
    if z.shape[-1] != LATENT_DIM:
        raise ValueError(f"Expected z trailing dim {LATENT_DIM}, got {z.shape[-1]}")
    if z_g.shape[-1] != LATENT_DIM:
        raise ValueError(f"Expected z_g trailing dim {LATENT_DIM}, got {z_g.shape[-1]}")
    return torch.cat([z, z_g, z - z_g, torch.abs(z - z_g)], dim=-1)


class SmallCostHead(nn.Module):
    """Two-layer MLP cost head with width 128."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Compute scalar costs for terminal and goal latents."""
        return self.net(cost_head_features(z, z_g)).squeeze(-1)


class LargeCostHead(nn.Module):
    """Diagnostic three-layer MLP cost head with width 512."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Compute scalar costs for terminal and goal latents."""
        return self.net(cost_head_features(z, z_g)).squeeze(-1)


class TemperatureScaledCostHead(nn.Module):
    """Wrap a cost head with a learnable positive scalar temperature."""

    def __init__(self, base: nn.Module, temperature_init: float = 10.0) -> None:
        super().__init__()
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive")
        self.base = base
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature_init)).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive output scale."""
        return self.log_temperature.exp()

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Compute temperature-scaled scalar costs."""
        return self.temperature * self.base(z, z_g)


def make_cost_head(
    variant: str = "small",
    *,
    temperature: bool = False,
    temperature_init: float = 10.0,
) -> nn.Module:
    """Instantiate a Phase 2 cost head.

    Args:
        variant: Either ``"small"`` or ``"large"``.
        temperature: If true, wrap the head with a learnable positive scalar.
        temperature_init: Initial scale for the optional temperature wrapper.

    Returns:
        The requested cost-head module.
    """
    normalized = variant.lower()
    if normalized == "small":
        model = SmallCostHead()
    elif normalized == "large":
        model = LargeCostHead()
    else:
        raise ValueError(f"Unknown cost-head variant: {variant!r}")
    if temperature:
        return TemperatureScaledCostHead(model, temperature_init=temperature_init)
    return model


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def main() -> int:
    """Run a small shape and parameter-count sanity check."""
    torch.manual_seed(0)
    z = torch.randn(4, LATENT_DIM)
    z_g = torch.randn(4, LATENT_DIM)
    for variant in ("small", "large"):
        model = make_cost_head(variant)
        out = model(z, z_g)
        print(
            f"{variant}: output_shape={tuple(out.shape)} "
            f"parameters={count_parameters(model)}"
        )
    model = make_cost_head("small", temperature=True)
    out = model(z, z_g)
    print(
        f"small+temperature: output_shape={tuple(out.shape)} "
        f"parameters={count_parameters(model)} "
        f"temperature={float(model.temperature.detach()):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
