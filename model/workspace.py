"""L3 global workspace: fast content access (GNW), slow monitoring clarity and detection."""

from typing import Optional, Sequence

import numpy as np

from config.defaults import (
    ACCESS_ADAPTATION_TAU, ACCESS_ADAPTATION_WEIGHT, ACCESS_PRIOR_CONCENTRATION,
    ACCESS_REVERBERATION,
    ACCESS_THRESHOLD, ACCESSIBLE_MASK, CLARITY_BIAS, DEFAULT_DT, CLARITY_TAU, LAPSE_EVIDENCE, LAPSE_STATES,
    CUE_MONITOR_CENTRE, CUE_MONITOR_GAIN, RECOGNITION_CUE,
    LATENT_TAU, ON_OBJECT_STATES, ON_TASK_POSTERIOR,
    ON_TASK_WEIGHT,
    ON_TASK_CONTRAST,
    THOUGHTSEEDS,
    on_task_log_odds_contrast,
)
from config.profiles import DWELL_MEAN_SECONDS
from utils.math_utils import logit, sigmoid
from .habits import AccessPrior

_INDEX = {name: i for i, name in enumerate(THOUGHTSEEDS)}
_ON_TASK_WEIGHT = np.array([ON_TASK_WEIGHT[name] for name in THOUGHTSEEDS], dtype=np.float64)
_ON_TASK_POSTERIOR = np.array([ON_TASK_POSTERIOR[name] for name in THOUGHTSEEDS], dtype=np.float64)
_ON_TASK_CONTRAST = np.array([ON_TASK_CONTRAST[name] for name in THOUGHTSEEDS], dtype=np.float64)
_ACCESSIBLE = np.array(ACCESSIBLE_MASK, dtype=bool)
_ACCESSIBLE_CONTRAST = np.where(_ACCESSIBLE, _ON_TASK_CONTRAST, 0.0)
_CUE = _INDEX[RECOGNITION_CUE]
# Contents whose access is evidence of a departure: positive lapse evidence (the distractors, by the import assert).
_DEPARTURE_EVIDENCE = frozenset(content for content, value in LAPSE_EVIDENCE.items() if value > 0)


class GlobalWorkspace:
    """L3: single-slot access (eq:access), monitoring clarity (eq:gate), lapse belief (eq:belief), detection
    (eq:detect). Arrays are indexed by THOUGHTSEEDS."""

    def __init__(self, level: str) -> None:
        # eps_on, eps_off: per-step lapse onset and offset from the configured dwell means.
        dwell = DWELL_MEAN_SECONDS[level]
        self._lapse_onset = DEFAULT_DT / sum(dwell[s] for s in ON_OBJECT_STATES)
        self._lapse_offset = DEFAULT_DT / sum(dwell[s] for s in LAPSE_STATES)
        self.access_prior = AccessPrior()
        self.reset_tonic()
        self.reset()

    def reset(self) -> None:
        """Reset transient state; the learned access prior and content counts are kept."""
        self.access_prior.clear_pending()
        self.accessed_content: Optional[str] = None
        self.clarity = self.resting_clarity
        self.adaptation = np.zeros(len(THOUGHTSEEDS), dtype=np.float64)
        self.detection_completed = False
        self.departure_evidenced = False
        self.lapse_belief = self._lapse_onset / (self._lapse_onset + self._lapse_offset)

    def reset_tonic(self) -> None:
        """Uniform Dirichlet counts over the accessible contents; none for the recognition cue."""
        self.content_counts = float(ACCESS_PRIOR_CONCENTRATION) * _ACCESSIBLE.astype(np.float64)

    def learn_tonic(self) -> None:
        if self.accessed_content is not None:
            self.content_counts[_INDEX[self.accessed_content]] += 1.0

    @property
    def tonic_contrast(self) -> float:
        """g: on-task contrast of sum_o p(o) P(on task|o), p the Dirichlet mean of the counts."""
        on_task = float(self.content_counts @ _ON_TASK_POSTERIOR / self.content_counts.sum())
        return on_task_log_odds_contrast(on_task)

    @property
    def resting_clarity(self) -> float:
        """m_rest = sigmoid(g - b_m), the monitor's fixed point at zero background contrast; one half when untrained."""
        return float(sigmoid(self.tonic_contrast - float(CLARITY_BIAS)))

    @staticmethod
    def _vector(values: Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(THOUGHTSEEDS),) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} needs one finite value per configured thoughtseed.")
        return array

    def access_drive(self, z_star: Sequence[float]) -> np.ndarray:
        """u_i = z*_i + beta_i + kappa*[incumbent] + (m_{t-1} - m_rest) w_i - w_a r_i, with w_i = tanh(c(i)/2);
        beta is the learned access prior held here."""
        drive = self._vector(z_star, "z_star") + self.access_prior.mean()
        drive -= ACCESS_ADAPTATION_WEIGHT * self.adaptation
        drive += (self.clarity - self.resting_clarity) * _ON_TASK_WEIGHT
        if self.accessed_content is not None:
            drive[_INDEX[self.accessed_content]] += ACCESS_REVERBERATION
        return drive

    def select_access(self, z_star: Sequence[float]) -> Optional[str]:
        """The accessible content with the highest drive at or above threshold, else an empty slot."""
        drive = self.access_drive(z_star)
        candidates = _ACCESSIBLE & (drive >= ACCESS_THRESHOLD)
        winner = None
        if candidates.any():
            winner = THOUGHTSEEDS[int(np.argmax(np.where(candidates, drive, -np.inf)))]

        occupied = np.zeros(len(THOUGHTSEEDS), dtype=np.float64)
        if winner is not None:
            occupied[_INDEX[winner]] = 1.0
        rate = DEFAULT_DT / float(ACCESS_ADAPTATION_TAU)
        self.adaptation += rate * (occupied - self.adaptation)
        self.accessed_content = winner
        return winner

    def update_lapse_belief(self, previous_clarity: float) -> float:
        """eq:belief: predict with eps_on and eps_off, then add lambda(o) at precision m_{t-1}, once per latent time constant."""
        predicted = self.lapse_belief * (1.0 - self._lapse_offset) + (1.0 - self.lapse_belief) * self._lapse_onset
        log_odds = logit(predicted)
        if self.accessed_content in LAPSE_EVIDENCE:
            log_odds += float(previous_clarity) * (DEFAULT_DT / LATENT_TAU) * LAPSE_EVIDENCE[self.accessed_content]
        self.lapse_belief = float(sigmoid(log_odds))
        return self.lapse_belief

    def holds(self, clarity: float, lapse_belief: float) -> bool:
        """psi = [m > m_rest and q >= 1/2]: the one predicate for detection onset and persistence."""
        return bool(float(clarity) > self.resting_clarity and float(lapse_belief) >= 0.5)

    def update_detection(self, previous_clarity: float) -> bool:
        """eq:detect, after m_t: q_t at precision m_{t-1}; R_t = [o_t is departure evidence] or (R_{t-1} and
        q_t >= 1/2); D_t = psi(m_t, q_t) and (D_{t-1} or R_t); returns the onset D_t and not D_{t-1}. R_t keeps the
        prior alone from starting a detection."""
        self.update_lapse_belief(previous_clarity)
        self.departure_evidenced = bool(self.accessed_content in _DEPARTURE_EVIDENCE
                                        or (self.departure_evidenced and self.lapse_belief >= 0.5))
        previous = self.detection_completed
        self.detection_completed = self.holds(self.clarity, self.lapse_belief) and (previous or self.departure_evidenced)
        return self.detection_completed and not previous

    @staticmethod
    def background_contrast(z_star: Sequence[float]) -> float:
        """Monitor drive (eq:gate): the activation-weighted mean on-task contrast of the accessible contents plus the
        centred cue term c(cue) (z_cue - mu_cue(MW))."""
        z = GlobalWorkspace._vector(z_star, "z_star")
        accessible = z[_ACCESSIBLE].sum()
        if accessible <= 0.0:
            raise ValueError("z_star must have positive accessible activation.")
        return float(z @ _ACCESSIBLE_CONTRAST / accessible) + CUE_MONITOR_GAIN * (z[_CUE] - CUE_MONITOR_CENTRE)

    def update_clarity(self, z_star: Sequence[float]) -> float:
        """tau_m dm/dt = -m + sigmoid(c_bar(z*_t) + g - b_m), c_bar the monitor drive of `background_contrast` (eq:gate)."""
        drive = self.background_contrast(z_star) + self.tonic_contrast - float(CLARITY_BIAS)
        target = sigmoid(drive)
        rate = DEFAULT_DT / float(CLARITY_TAU)
        self.clarity = self.clarity + rate * (target - self.clarity)
        return self.clarity
