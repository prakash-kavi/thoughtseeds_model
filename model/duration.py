"""Shared discrete Gamma duration model for the L1 process and L2 prior."""

from functools import lru_cache
from math import ceil, lgamma, log
import numpy as np

from config.defaults import DEFAULT_DT
from config.profiles import DWELL_CV, DWELL_MEAN_SECONDS, get_exit_transition_probs


@lru_cache(maxsize=None)
def _pmf(level: str, state: str) -> np.ndarray:
    """Finite discrete approximation to the configured Gamma duration law."""
    mean_steps = DWELL_MEAN_SECONDS[level][state] / DEFAULT_DT
    shape = 1.0 / (DWELL_CV ** 2)
    scale = mean_steps / shape
    horizon = max(2, int(ceil(mean_steps + 8.0 * mean_steps * DWELL_CV)))
    ages = np.arange(1, horizon + 1, dtype=float)
    log_density = (shape - 1.0) * np.log(ages) - ages / scale - shape * log(scale) - lgamma(shape)
    masses = np.exp(log_density)
    masses /= masses.sum()
    return masses


def hazard(level: str, state: str, age_steps: float) -> float:
    """Probability of exit at this age conditional on survival so far."""
    masses = _pmf(level, state)
    index = max(0, int(np.ceil(age_steps)) - 1)
    if index >= len(masses) - 1:
        return 1.0
    return float(masses[index] / masses[index:].sum())


def exit_law(level: str, state: str, age_steps: float,
             hazard_modulation: float = 0.0) -> tuple:
    """Shared Gamma exit probability and unchanged phenotype destinations.

    L2 encodes any release of broadcast stabilization in hazard_modulation.
    No additional exit clock or destination-specific probability mass is added.
    """
    if not np.isfinite(hazard_modulation):
        raise ValueError('Exit control must be finite.')
    effective_hazard = float(np.clip(hazard(level, state, age_steps) * (1.0 + hazard_modulation), 0.0, 1.0))
    return effective_hazard, get_exit_transition_probs(level, state)


def median_steps(level: str, state: str) -> int:
    """Smallest dwell length (in steps) whose cumulative probability reaches 0.5."""
    cdf = np.cumsum(_pmf(level, state))
    return int(np.searchsorted(cdf, 0.5)) + 1


def survival_probability(level: str, state: str, dwell_steps: float) -> float:
    """Probability that a fresh dwell in this regime lasts at least dwell_steps."""
    masses = _pmf(level, state)
    index = max(0, int(np.ceil(dwell_steps)) - 1)
    if index >= len(masses):
        return 0.0
    return float(masses[index:].sum())
