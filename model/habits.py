"""System 1 learned priors held by L2: transition habit and workspace-access prior."""

from typing import Dict, List, Optional

import numpy as np

from config.defaults import (
    ACCESS_CREDIT_HOLD_STEPS, ACCESS_PRIOR_CONCENTRATION, META_THRESHOLD, PRACTICE_OBJECT, STATES, THOUGHTSEEDS,
)
from utils.math_utils import clip_probability


class HabitPrior:
    """Dirichlet E-vector over destination states, one per source state.

    A habit is the learned, efficient context-to-response default (System 1).
    It is updated from the belief-weighted switch commitments that L3 returns,
    so repeated deliberate choices gradually train the automatic default.
    """

    def __init__(self) -> None:
        self._alpha: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._alpha.clear()

    @staticmethod
    def _state_weights(state_belief: Dict[str, float]) -> np.ndarray:
        values = np.array([float(state_belief[s]) for s in STATES], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or values.sum() <= 0.0:
            raise ValueError("state_belief must be a finite, non-negative distribution.")
        return values / values.sum()

    def _log_prior_for_source(self, source: str, candidates: List[str]) -> np.ndarray:
        alpha = self._alpha.get(source)
        if alpha is None:
            alpha = np.ones(len(STATES), dtype=np.float64)
        probabilities = alpha / alpha.sum()
        index = {state: i for i, state in enumerate(STATES)}
        return np.log(probabilities[[index[candidate] for candidate in candidates]])

    def log_prior(self, state_belief: Dict[str, float], candidates: List[str]) -> np.ndarray:
        """Belief-weighted habit log-prior sum_s q(s) ln p_h(pi | s)."""
        log_prior = np.zeros(len(candidates), dtype=np.float64)
        for weight, source in zip(self._state_weights(state_belief), STATES):
            log_prior += weight * self._log_prior_for_source(source, candidates)
        return log_prior

    def update(
        self,
        state_belief: Dict[str, float],
        intended_by_source: Dict[str, Dict[str, float]],
    ) -> None:
        for source_weight, source in zip(self._state_weights(state_belief), STATES):
            for destination, probability in intended_by_source.get(source, {}).items():
                if destination == source:
                    continue
                if destination not in STATES or not np.isfinite(probability) or probability < 0.0:
                    raise ValueError("Intended policies must be finite probabilities over known states.")
                alpha = self._alpha.setdefault(source, np.ones(len(STATES), dtype=np.float64))
                alpha[STATES.index(destination)] += source_weight * float(probability)


class AccessPrior:
    """Learned access prior beta over thoughtseeds (System 1): which content wins access.

    Distinct from the habit prior, which concerns which state transition is
    expected. beta is the mean of a Dirichlet over thoughtseeds with uniform
    initial pseudo-counts. Credit goes to the practice object only after
    successful regulation: it re-ignites while meta-awareness is present and
    then holds the workspace for one latent time constant. Credit is the
    meta-awareness excess above META_THRESHOLD at re-ignition, so routine
    returns at the monitor's resting level earn none. The Dirichlet mean keeps
    beta bounded in [0, 1].
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.alpha = np.full(len(THOUGHTSEEDS), float(ACCESS_PRIOR_CONCENTRATION))
        self.clear_pending()

    def clear_pending(self) -> None:
        self._onset_credit: Optional[float] = None
        self._held = 0

    def mean(self) -> np.ndarray:
        return self.alpha / self.alpha.sum()

    def observe(self, previous: Optional[str], current: Optional[str], meta_awareness: float) -> bool:
        """Track practice-object re-ignitions; return True when credit is committed."""
        if current != PRACTICE_OBJECT:
            self.clear_pending()
            return False
        if previous != PRACTICE_OBJECT:
            self._onset_credit = max(0.0, clip_probability(meta_awareness) - float(META_THRESHOLD))
            self._held = 0
        if self._onset_credit is None:
            return False
        self._held += 1
        if self._held < ACCESS_CREDIT_HOLD_STEPS:
            return False
        self.alpha[THOUGHTSEEDS.index(PRACTICE_OBJECT)] += self._onset_credit
        self.clear_pending()
        return True
