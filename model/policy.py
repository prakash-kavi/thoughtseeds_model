"""L3 policy arbitration (System 2)."""

from typing import List

import numpy as np

from config.defaults import EPS
from utils.math_utils import clip_probability, policy_posterior


class PolicyEngine:
    """Arbitrate between System 1 defaults and L2 evidence.

    The dwell and habit priors are the automatic defaults; meta-awareness sets
    the precision of L2's policy evidence. L2's dwell prior incorporates
    workspace stabilization and its release after MA detection completes.
    """

    def select_policy(
        self,
        g_vals: List[float],
        dwell_priors: List[float],
        candidates: List[str],
        habit_log_prior: np.ndarray,
        meta_precision: float,
    ) -> np.ndarray:
        priors = np.asarray(dwell_priors, dtype=np.float64)
        costs = np.asarray(g_vals, dtype=np.float64)
        habit = np.asarray(habit_log_prior, dtype=np.float64)
        if not (len(candidates) == len(priors) == len(costs) == len(habit)):
            raise ValueError("Policy candidates, priors, habits, and costs must have the same length.")
        if not np.all(np.isfinite(priors)) or np.any(priors <= 0.0) or not np.all(np.isfinite(costs)):
            raise ValueError("Policy priors must be positive and all policy costs finite.")
        gamma = max(EPS, clip_probability(meta_precision))
        log_prior = np.log(priors) + habit
        return policy_posterior(log_prior, costs, gamma)
