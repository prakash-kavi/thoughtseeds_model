"""L3 return policy over MA's successor (eq:policy) and its learned habit."""

from typing import Mapping, Optional

import numpy as np

from config.defaults import (
    ACCESS_PRIOR_CONCENTRATION, ACCESSIBLE_CONTENTS, MA_STATE, OFF_TASK_STATE, PRACTICE_STATE, RETURN_STATE,
    THOUGHTSEED_DIAGNOSTICITY, THOUGHTSEED_STATE_PRIORS,
)
from config.profiles import get_exit_transition_probs


class ReturnPolicy:
    """eq:policy: P(pi) = softmax(ln E - gamma_t G) over {RA, MW} and the exit-odds factor Lambda; E is the learned
    habit, G risk plus ambiguity, staying scored by G(MA) (one-step approximation). L1 applies both only in MA."""

    def __init__(self, level: str, habit: Optional[Mapping[str, float]] = None) -> None:
        self.configured_habit = get_exit_transition_probs(level, MA_STATE)
        assert set(self.configured_habit) == {RETURN_STATE, OFF_TASK_STATE}
        goal = np.array([THOUGHTSEED_DIAGNOSTICITY[t][PRACTICE_STATE] for t in ACCESSIBLE_CONTENTS])
        log_goal = np.log(goal / goal.sum())
        self.expected_free_energy = {}
        for state in (*self.configured_habit, MA_STATE):
            likelihood = np.array([THOUGHTSEED_STATE_PRIORS[state][t] for t in ACCESSIBLE_CONTENTS])
            likelihood /= likelihood.sum()
            risk = float(likelihood @ (np.log(likelihood) - log_goal))
            ambiguity = float(-(likelihood * np.log(likelihood)).sum())
            self.expected_free_energy[state] = risk + ambiguity
        assert self.expected_free_energy[RETURN_STATE] < self.expected_free_energy[OFF_TASK_STATE]
        self.reset()
        if habit is not None:
            # A saved habit, for readouts of a finished run.
            assert set(habit) == set(self.configured_habit)
            self.habit_counts = {s: float(ACCESS_PRIOR_CONCENTRATION) * float(habit[s]) for s in self.configured_habit}

    @staticmethod
    def precision(clarity: float, lapse_belief: float, recognized: bool) -> float:
        """gamma_t = D_t m_t q_t."""
        return float(bool(recognized)) * float(clarity) * float(lapse_belief)

    def reset(self) -> None:
        """Prior counts: the configured MA exit row at the shared Dirichlet concentration."""
        self.habit_counts = {s: float(ACCESS_PRIOR_CONCENTRATION) * p for s, p in self.configured_habit.items()}

    @property
    def habit(self) -> dict:
        """E: the Dirichlet mean of the habit counts."""
        total = sum(self.habit_counts.values())
        return {s: c / total for s, c in self.habit_counts.items()}

    def learn(self, precision: float) -> dict:
        """Add the act formed at gamma_t to the habit counts and return it."""
        act = self.probabilities(precision)
        for s, p in act.items():
            self.habit_counts[s] += p
        return act

    def probabilities(self, precision: float) -> dict:
        habit = self.habit
        states = list(habit)
        logits = np.array([np.log(habit[s]) - float(precision) * self.expected_free_energy[s] for s in states])
        weights = np.exp(logits - logits.max())
        return dict(zip(states, (weights / weights.sum()).tolist()))

    def exit_odds_factor(self, precision: float) -> float:
        """Lambda = sum_j E_j exp[-gamma_t (G(j) - G(MA))]: the factor on MA's stabilized exit odds."""
        stay = self.expected_free_energy[MA_STATE]
        return float(sum(
            p * np.exp(-float(precision) * (self.expected_free_energy[s] - stay)) for s, p in self.habit.items()))
