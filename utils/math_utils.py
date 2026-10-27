"""Numerical helpers shared by the simulation layers."""

from typing import Dict, Iterable, Union

import numpy as np
import torch


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    return np.log(p / (1.0 - p))


def networks_to_tensor(network_values: Dict[str, Union[float, torch.Tensor]], networks: Iterable[str]) -> torch.Tensor:
    return torch.stack([
        value if isinstance(value := network_values[network], torch.Tensor) else torch.tensor(value, dtype=torch.float32)
        for network in networks
    ])
