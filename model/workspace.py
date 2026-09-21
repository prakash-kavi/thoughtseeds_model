"""L3 global workspace: fast content access (GNW) and slow meta-awareness (System 2)."""

from typing import List, Optional, Sequence

import numpy as np

from config.defaults import (
    ACCESS_ADAPTATION_TAU, ACCESS_ADAPTATION_WEIGHT, ACCESS_REVERBERATION, ACCESS_THRESHOLD,
    DEFAULT_DT, EPS, META_RESTING, META_TAU, META_THRESHOLD, ON_TASK_CONTRAST,
    ON_TASK_LOG_EVIDENCE,
    THOUGHTSEEDS,
)
from utils.math_utils import clip_probability, softmax

_INDEX = {name: i for i, name in enumerate(THOUGHTSEEDS)}
_ON_TASK = np.array([ON_TASK_CONTRAST[name] for name in THOUGHTSEEDS], dtype=np.float64)


class GlobalWorkspace:
    """Two coupled L3 processes on different timescales.

    Access (GNW): at each step thoughtseeds compete for a single workspace slot.
    The winner is broadcast all-or-none. The incumbent receives a reverberation
    bonus, so access is self-sustaining with hysteresis, while content-specific
    adaptation progressively removes that advantage and permits release. A challenger can break in only if
    the correction step revised it upward (prediction error).

    Meta-awareness (System 2): m_t is a slow, graded measure of monitoring
    clarity. It integrates the diagnosticity-derived signal of accessed content
    with policy--habit discrepancy, biases subsequent access, and sets policy
    precision. The discrepancy input is phenotype-graded but modest: it
    modulates monitoring rather than triggering detection.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.accessed_content: Optional[str] = None
        self.meta_awareness = float(META_RESTING)
        self.adaptation = np.zeros(len(THOUGHTSEEDS), dtype=np.float64)

    @staticmethod
    def _vector(values: Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(THOUGHTSEEDS),) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} needs one finite value per configured thoughtseed.")
        return array

    def access_drive(self, z_star: Sequence[float], access_prior: Sequence[float]) -> np.ndarray:
        """u_i = z*_i + beta_i + kappa*[i incumbent] + (m_{t-1} - m_rest) w_i - w_a*r_i.

        Monitoring clarity biases access towards on-task content, graded by that
        content's diagnosticity-derived contrast w_i = tanh(c(i)/2), and
        referenced to the monitor's resting fixed point so that resting clarity
        applies no bias at all. Above resting, on-task content is favoured;
        below it, on-task content fades and distractor content is favoured.
        The slot can be empty when no candidate clears the access threshold.
        """
        drive = self._vector(z_star, "z_star") + self._vector(access_prior, "access_prior")
        drive -= ACCESS_ADAPTATION_WEIGHT * self.adaptation
        drive += (self.meta_awareness - float(META_RESTING)) * _ON_TASK
        if self.accessed_content is not None:
            drive[_INDEX[self.accessed_content]] += ACCESS_REVERBERATION
        return drive

    def select_access(
        self,
        z_star: Sequence[float],
        z_prior: Sequence[float],
        access_prior: Sequence[float],
    ) -> Optional[str]:
        """Resolve the single workspace slot for this step and update adaptation."""
        drive = self.access_drive(z_star, access_prior)
        candidates = self._vector(z_star, "z_star") > self._vector(z_prior, "z_prior")
        if self.accessed_content is not None:
            candidates[_INDEX[self.accessed_content]] = True
        candidates &= drive >= ACCESS_THRESHOLD
        winner = None
        if candidates.any():
            winner = THOUGHTSEEDS[int(np.argmax(np.where(candidates, drive, -np.inf)))]

        occupied = np.zeros(len(THOUGHTSEEDS), dtype=np.float64)
        if winner is not None:
            occupied[_INDEX[winner]] = 1.0
        rate = DEFAULT_DT / max(float(ACCESS_ADAPTATION_TAU), DEFAULT_DT)
        self.adaptation += rate * (occupied - self.adaptation)
        self.accessed_content = winner
        return winner

    @staticmethod
    def on_task_evidence(content: Optional[str]) -> float:
        """Diagnosticity-derived log-odds contrast; zero for an empty slot."""
        if content is None:
            return 0.0
        return float(ON_TASK_LOG_EVIDENCE[content])

    def update_meta_awareness(self, g_vals: List[float], habit_log_prior: Sequence[float]) -> float:
        """tau_m dm/dt = -m + sigmoid(c(o_t) + KL(q_evid || q_habit) - b_m).

        c(o_t), the diagnosticity-derived contrast carried by the broadcast
        content, is the primary drive. Policy--habit discrepancy is an
        additional input on the same (nat) scale: its contribution was larger
        in experts in the evaluated runs and influences monitoring-dependent
        control, but in those runs it never produced a new monitoring-threshold
        crossing during distractor broadcast, so it is not an independent
        detection trigger.
        """
        costs = np.asarray(g_vals, dtype=np.float64)
        habit = np.asarray(habit_log_prior, dtype=np.float64)
        if costs.shape != habit.shape or not np.all(np.isfinite(costs)) or not np.all(np.isfinite(habit)):
            raise ValueError("Policy costs and habit log-prior must be finite and aligned.")
        q_evid = np.clip(softmax(-costs), EPS, 1.0)
        q_habit = np.clip(softmax(habit), EPS, 1.0)
        q_evid /= q_evid.sum()
        q_habit /= q_habit.sum()
        discrepancy = float(np.sum(q_evid * np.log(q_evid / q_habit)))

        drive = self.on_task_evidence(self.accessed_content) + discrepancy - float(META_THRESHOLD)
        target = 1.0 / (1.0 + np.exp(-drive))
        rate = DEFAULT_DT / max(float(META_TAU), DEFAULT_DT)
        self.meta_awareness = clip_probability(self.meta_awareness + rate * (target - self.meta_awareness))
        return self.meta_awareness
