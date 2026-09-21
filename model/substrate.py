"""Layer 1 switching multivariate Ornstein--Uhlenbeck process."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from config.defaults import (
    NETWORKS, DEFAULT_DT, CLIP_MIN, CLIP_MAX, EPS, NOISE_LEVEL,
    THETA_BASE,
    EXECUTION_FIDELITY,
)
from config.profiles import (
    L1_SUBSTEPS, STATE_TRANSITION_PROBS, THETA_SETTLING_MARGIN, THETA_STABILITY_LIMIT,
)
from model.phenotype import PhenotypeConfig, EXPERT_PHENOTYPE
from model import duration
from utils.math_utils import clip_probability, convex_blend

class Layer1Process(nn.Module):
    """MVOU generative process for brain network dynamics."""

    # MVOU integration constants
    N_SUBSTEPS = L1_SUBSTEPS
    INIT_ACTIVATION = 0.5

    def __init__(self, phenotype: PhenotypeConfig = None, seed: int = None):
        super().__init__()

        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level
        self.dt = DEFAULT_DT
        self.rng = np.random.RandomState(seed)

        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
        self.current_state = 'breath_focus'
        self.current_dwell = 0
        self.net_idx = {n: i for i, n in enumerate(NETWORKS)}
        # Theta(s) depends only on the state and the fixed phenotype, so the
        # four matrices are built once here rather than on every timestep.
        self.theta_by_state = {
            state: self._clamp_theta(self._get_coupling(state)) for state in THETA_BASE
        }
        
    def _exit_destination_kernel(
        self,
        policy_state_probs: Dict[str, float],
        exit_prior: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """Blend controller and phenotype priors after an exit has occurred."""
        if set(policy_state_probs) != set(STATE_TRANSITION_PROBS[self.level][self.current_state]) | {self.current_state}:
            raise ValueError("policy_state_probs must contain exactly the model states.")
        destinations = STATE_TRANSITION_PROBS[self.level][self.current_state] if exit_prior is None else exit_prior
        policy = np.array([float(policy_state_probs[state]) for state in destinations], dtype=float)
        full_policy = np.array(list(policy_state_probs.values()), dtype=float)
        if not np.all(np.isfinite(full_policy)) or np.any(full_policy < 0.0) or not np.isclose(full_policy.sum(), 1.0):
            raise ValueError("policy_state_probs must be a finite probability distribution.")
        if policy.sum() <= 0.0:
            raise ValueError("Policy must assign positive mass to at least one exit destination.")
        policy /= policy.sum()
        eta = clip_probability(EXECUTION_FIDELITY)
        return {
            state: convex_blend(float(destinations[state]), float(policy[index]), eta)
            for index, state in enumerate(destinations)
        }

    def _check_transition(
        self,
        policy_state_probs: Dict[str, float],
        hazard_modulation: float = 0.0,
    ) -> str:
        """Advance the shared Gamma hazard, then select an exit destination."""
        self.current_dwell += 1
        hazard, exit_prior = duration.exit_law(
            self.level, self.current_state, self.current_dwell,
            hazard_modulation,
        )
        if self.rng.random_sample() >= hazard:
            return self.current_state

        kernel = self._exit_destination_kernel(policy_state_probs, exit_prior)
        states = list(kernel.keys())
        p_array = np.array([kernel[s] for s in states], dtype=float)
        p_array /= p_array.sum()

        next_state = self.rng.choice(states, p=p_array)
        self.current_state = next_state
        self.current_dwell = 0

        return self.current_state
    
    def _get_coupling(self, state: str) -> torch.Tensor:
        """Build the off-diagonal Theta(s) template; _clamp_theta sets the diagonal."""
        theta_np = np.zeros((len(NETWORKS), len(NETWORKS)))

        coupling_map = THETA_BASE[state]
        for (row_net, col_net), value in coupling_map.items():
            r_idx = self.net_idx[row_net]
            c_idx = self.net_idx[col_net]
            theta_np[r_idx, c_idx] = value

        theta = torch.tensor(theta_np, dtype=torch.float32)
        return theta * float(self.phenotype.theta_scale)
    
    def _clamp_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Set each diagonal to its row's off-diagonal sum plus the settling margin.

        The margin is the slowest relaxation rate the construction leaves in any
        regime: for a row whose couplings are wholly cross-excitatory, the global
        co-activation mode relaxes at exactly that rate. It is therefore derived
        (THETA_SETTLING_MARGIN) so that every regime expresses its configured
        attractor within a typical dwell and L1 runs faster than the L2 latent it
        drives. No upper clip is applied: the diagonal is whatever the margin
        requires, and the integrator's stability bound is asserted instead, with
        roughly six-fold headroom at the configured couplings.
        """
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
        if float(final_diag.max()) >= THETA_STABILITY_LIMIT:
            raise RuntimeError(
                f"Theta(s) stiffness {float(final_diag.max()):.2f} reaches the "
                f"explicit-Euler stability bound {THETA_STABILITY_LIMIT:.1f} for "
                f"the configured substep; reduce the couplings or the settling "
                f"margin, or add substeps."
            )
        return theta_off + torch.diag(final_diag)
    
    def update(self, active_states: Dict) -> Tuple[Dict[str, torch.Tensor], str]:
        """Advance one timestep and return network activations + current state.
        
        Args:
            active_states: Control from L2 via Markov blanket
                - mu_x: Optional[torch.Tensor] L2 descending prediction in network space
                - policy_state_probs: Optional[Dict[str, float]] policy posterior over candidate states
                - hazard_modulation: Optional[float] broadcast-driven exit-timing gain

        Returns:
            network_acts: {network_name: activation}
            current_state: str
        """
        policy_state_probs = active_states['policy_state_probs']
        # Scalar broadcast timing signal; L1 never receives m_t itself.
        hazard_modulation = float(active_states.get('hazard_modulation', 0.0))

        self.current_state = self._check_transition(policy_state_probs, hazard_modulation)
        theta = self.theta_by_state[self.current_state]

        # L2->L1 control signals are mu_x and policy_state_probs; the blend is
        # formed upstream in L2/L3.
        mu = active_states['mu_x']

        # Process noise variance is fixed, not modulated by L2.
        sigma = np.sqrt(NOISE_LEVEL)
        dt_sub = self.dt / self.N_SUBSTEPS
        curr_x = self.x

        for _ in range(self.N_SUBSTEPS):
            # dX = -Theta(s) . (X - mu) dt + sigma dW
            drift = -torch.matmul(theta, (curr_x - mu)) * dt_sub
            diffusion = torch.randn_like(curr_x) * sigma * np.sqrt(dt_sub)
            curr_x = curr_x + drift + diffusion
            curr_x = torch.clamp(curr_x, CLIP_MIN, CLIP_MAX)

        self.x = curr_x
        network_acts = {net: self.x[i] for i, net in enumerate(NETWORKS)}
        return network_acts, self.current_state

    def reset(self, state: str = 'breath_focus') -> None:
        self.current_state = state
        self.current_dwell = 0
        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
