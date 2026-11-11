"""Layer 1 switching multivariate Ornstein--Uhlenbeck process."""

import numpy as np
import torch
from typing import Dict, Tuple

from config.defaults import (
    NETWORKS, DEFAULT_DT, CLIP_MIN, CLIP_MAX, NOISE_LEVEL,
    THETA_BASE, MA_STATE, PRACTICE_STATE,
)
from config.profiles import (
    L1_SUBSTEPS, NETWORK_PROFILES, THETA_SETTLING_MARGIN, THETA_STABILITY_LIMIT,
)
from model.phenotype import PhenotypeConfig, EXPERT_PHENOTYPE
from model import duration
from utils.math_utils import networks_to_tensor

class Layer1Process:
    """MVOU generative process for brain network dynamics."""

    N_SUBSTEPS = L1_SUBSTEPS
    INIT_ACTIVATION = 0.5

    def __init__(self, phenotype: PhenotypeConfig = None, seed: int = None):
        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level
        self.dt = DEFAULT_DT
        self.rng = np.random.RandomState(seed)

        self.net_idx = {n: i for i, n in enumerate(NETWORKS)}
        self.theta_by_state = {
            state: self._clamp_theta(self._get_coupling(state)) for state in THETA_BASE
        }
        self.attractor_by_state = {
            state: networks_to_tensor(NETWORK_PROFILES[state][self.level], NETWORKS) for state in THETA_BASE
        }
        self.reset()

    def _check_transition(self, stabilization: float = 0.0, act: Dict = None, odds_factor: float = 1.0) -> str:
        """Advance the dwell age and test for exit; in MA the act gives the successor and odds_factor scales the odds."""
        self.current_dwell += 1
        p_exit, exit_probs = duration.exit_law(
            self.level, self.current_state, self.current_dwell,
            stabilization, odds_factor if self.current_state == MA_STATE else 1.0,
        )
        if self.rng.random_sample() >= p_exit:
            return self.current_state
        if self.current_state == MA_STATE and act:
            exit_probs = act

        states = list(exit_probs)
        p_array = np.array([exit_probs[s] for s in states], dtype=float)
        p_array /= p_array.sum()

        next_state = self.rng.choice(states, p=p_array)
        self.current_state = next_state
        self.current_dwell = 0

        return self.current_state
    
    def _get_coupling(self, state: str) -> torch.Tensor:
        """Off-diagonal Theta(s); _clamp_theta sets the diagonal."""
        theta_np = np.zeros((len(NETWORKS), len(NETWORKS)))

        coupling_map = THETA_BASE[state]
        for (row_net, col_net), value in coupling_map.items():
            r_idx = self.net_idx[row_net]
            c_idx = self.net_idx[col_net]
            theta_np[r_idx, c_idx] = value

        theta = torch.tensor(theta_np, dtype=torch.float32)
        return theta
    
    def _clamp_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Diagonal = row sum of |off-diagonal| + settling margin (Gershgorin); Euler-stable while dt_sub * row bound < 2."""
        n = len(NETWORKS)
        off_diag_mask = 1.0 - torch.eye(n)
        theta_off = theta * off_diag_mask
        off_diag_sum = torch.sum(torch.abs(theta_off), dim=1)
        final_diag = off_diag_sum + THETA_SETTLING_MARGIN

        if not torch.all(final_diag > off_diag_sum):
            raise RuntimeError(
                "Theta(s) lost diagonal dominance. Diagonal dominance with a "
                "positive diagonal is the Gershgorin sufficient condition for "
                "the OU drift to be mean-reverting (supplementary S1.2); this "
                "must hold for every call."
            )
        row_bound = float((final_diag + off_diag_sum).max())
        if row_bound >= THETA_STABILITY_LIMIT:
            raise RuntimeError(
                f"Theta(s) Gershgorin row bound {row_bound:.2f} reaches the "
                f"explicit-Euler stability bound {THETA_STABILITY_LIMIT:.1f} for "
                f"the configured substep; reduce the couplings or the settling "
                f"margin, or add substeps."
            )
        return theta_off + torch.diag(final_diag)
    
    def update(self, active_states: Dict) -> Tuple[Dict[str, torch.Tensor], str]:
        """eq:l1: one step under the control forwarded on the previous step; returns (network activations, regime)."""
        stabilization = duration.stabilization(
            active_states.get('accessed_content'), self.current_state,
            bool(active_states.get('detection_completed', False)))

        self.current_state = self._check_transition(
            stabilization, active_states.get('act'), float(active_states.get('exit_odds_factor', 1.0)))
        theta = self.theta_by_state[self.current_state]
        mu = self.attractor_by_state[self.current_state]

        sigma = np.sqrt(NOISE_LEVEL)
        dt_sub = self.dt / self.N_SUBSTEPS
        curr_x = self.x

        for _ in range(self.N_SUBSTEPS):
            drift = -torch.matmul(theta, (curr_x - mu)) * dt_sub
            diffusion = torch.randn_like(curr_x) * sigma * np.sqrt(dt_sub)
            curr_x = curr_x + drift + diffusion
            curr_x = torch.clamp(curr_x, CLIP_MIN, CLIP_MAX)

        self.x = curr_x
        network_acts = {net: self.x[i] for i, net in enumerate(NETWORKS)}
        return network_acts, self.current_state

    def reset(self, state: str = PRACTICE_STATE) -> None:
        self.current_state = state
        self.current_dwell = 0
        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
