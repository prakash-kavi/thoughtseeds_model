"""Discrete Gamma duration model and broadcast stabilization, both evaluated by the L1 process."""

from functools import lru_cache
from math import ceil, lgamma, log
from typing import Optional
import numpy as np

from config.defaults import DEFAULT_DT, FIRST_ORDER_CONTENTS, THOUGHTSEED_DIAGNOSTICITY
from config.profiles import DWELL_CV, DWELL_MEAN_SECONDS, get_exit_transition_probs


@lru_cache(maxsize=None)
def _pmf(level: str, state: str) -> np.ndarray:
    """Discrete approximation to the configured Gamma dwell-time distribution."""
    mean_steps = DWELL_MEAN_SECONDS[level][state] / DEFAULT_DT
    shape = 1.0 / (DWELL_CV ** 2)
    scale = mean_steps / shape
    horizon = max(2, int(ceil(mean_steps + 8.0 * mean_steps * DWELL_CV)))
    ages = np.arange(1, horizon + 1, dtype=float)
    log_density = (shape - 1.0) * np.log(ages) - ages / scale - shape * log(scale) - lgamma(shape)
    masses = np.exp(log_density)
    masses /= masses.sum()
    return masses


@lru_cache(maxsize=None)
def _exit_probabilities(level: str, state: str) -> np.ndarray:
    """f_k / S_k at each dwell age; the last age exits with certainty."""
    masses = _pmf(level, state)
    table = np.array([masses[k] / masses[k:].sum() for k in range(len(masses))])
    table[-1] = 1.0
    return table


def exit_probability(level: str, state: str, age_steps: float) -> float:
    """Probability of leaving the regime at this age, given that it has lasted so far."""
    table = _exit_probabilities(level, state)
    return float(table[min(max(0, int(np.ceil(age_steps)) - 1), len(table) - 1)])


def stabilization(accessed: Optional[str], state: str, detection_completed: bool) -> float:
    """eq:act: S = d_o(s) for accessed first-order content; zero for an empty slot, the cue, or while D_t holds."""
    if accessed not in FIRST_ORDER_CONTENTS or detection_completed:
        return 0.0
    return float(THOUGHTSEED_DIAGNOSTICITY[accessed][state])


def exit_law(level: str, state: str, age_steps: float,
             stabilization: float = 0.0, odds_factor: float = 1.0) -> tuple:
    """h~ = h (1 - S) (eq:act) with its odds multiplied by Lambda, h~Lambda / (1 + h~(Lambda - 1)), and the destination table."""
    if not 0.0 <= stabilization <= 1.0:
        raise ValueError('Broadcast stabilization must lie in [0, 1].')
    if not 0.0 < odds_factor < float('inf'):
        raise ValueError('The exit odds factor must be positive and finite.')
    stabilized = exit_probability(level, state, age_steps) * (1.0 - stabilization)
    weighted = stabilized * odds_factor / (1.0 + stabilized * (odds_factor - 1.0))
    return weighted, get_exit_transition_probs(level, state)
