"""Numerical helpers shared by the simulation layers."""

from typing import Dict, Iterable, Union

import numpy as np
import torch


def clip_probability(value: Union[float, int, torch.Tensor]) -> float:
    return float(np.clip(float(value.detach().item()) if isinstance(value, torch.Tensor) else value, 0.0, 1.0))


def convex_blend(base, alt, weight):
    """(1-weight)*base + weight*alt.

    The shared form behind every "biasing, not dictating" control blend in
    the model (descending-prediction mixture in training_loop.py, execution-
    fidelity mixture in substrate.py) -- works for floats, numpy arrays, and
    torch tensors alike.
    """
    return (1.0 - weight) * base + weight * alt


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def policy_posterior(log_prior: np.ndarray, costs: np.ndarray, precision: float) -> np.ndarray:
    return softmax(log_prior - precision * costs)


def clamp_activation(values: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.clamp(values, lower, upper)


def ou_step_scalar(
    value: torch.Tensor,
    target: torch.Tensor,
    dt: float,
    tau: float,
    noise_level: float,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    drift = -(value.detach() - target.detach()) / max(tau, dt)
    noise = torch.randn_like(value) * np.sqrt(noise_level * dt)
    return clamp_activation(value.detach() + drift * dt + noise, clip_min, clip_max)


def forward_error(prediction: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - observation) ** 2)


def networks_to_tensor(network_values: Dict[str, Union[float, torch.Tensor]], networks: Iterable[str]) -> torch.Tensor:
    return torch.stack([
        value if isinstance(value := network_values[network], torch.Tensor) else torch.tensor(value, dtype=torch.float32)
        for network in networks
    ])
