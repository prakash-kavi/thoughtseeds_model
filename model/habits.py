"""Learned workspace-access prior, held by L3."""

from typing import Optional

import numpy as np

from config.defaults import (
    ACCESS_CREDIT_HOLD_STEPS, ACCESS_PRIOR_CONCENTRATION, ACCESSIBLE_MASK, PRACTICE_OBJECT, THOUGHTSEEDS,
)

_PRACTICE = THOUGHTSEEDS.index(PRACTICE_OBJECT)


class AccessPrior:
    """Access prior beta: Dirichlet mean over the accessible contents (zero for the cue). Only the practice object
    earns credit, [m - m_rest]_+ taken at re-entry and committed after ACCESS_CREDIT_HOLD_STEPS in the slot."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.practice_count = float(ACCESS_PRIOR_CONCENTRATION)
        self.clear_pending()

    def clear_pending(self) -> None:
        self._onset_credit: Optional[float] = None
        self._held = 0

    def mean(self) -> np.ndarray:
        alpha = float(ACCESS_PRIOR_CONCENTRATION) * np.array(ACCESSIBLE_MASK, dtype=np.float64)
        alpha[_PRACTICE] = self.practice_count
        return alpha / alpha.sum()

    def observe(self, previous: Optional[str], current: Optional[str], clarity: float, resting_clarity: float) -> bool:
        """Return True when credit is committed."""
        if current != PRACTICE_OBJECT:
            self.clear_pending()
            return False
        if previous != PRACTICE_OBJECT:
            self._onset_credit = max(0.0, float(clarity) - float(resting_clarity))
            self._held = 0
        if self._onset_credit is None:
            return False
        self._held += 1
        if self._held < ACCESS_CREDIT_HOLD_STEPS:
            return False
        self.practice_count += self._onset_credit
        self.clear_pending()
        return True
