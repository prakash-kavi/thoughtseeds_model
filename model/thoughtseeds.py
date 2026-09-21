"""Layer 2 continuous thoughtseed inference and one-step policy scoring."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from config.defaults import (
    STATES, NETWORKS, THOUGHTSEEDS, CLIP_MIN, CLIP_MAX, EPS,
    THOUGHTSEED_STATE_PRIORS,
    get_policy_candidate_order,
    STATE_BELIEF_VAR,
    DEFAULT_DT, LATENT_TAU,
    NOISE_LEVEL,
    P_INV,
    DESCENDING_PREDICTION_GAIN,
    THOUGHTSEED_DIAGNOSTICITY,
    DETECTION_CONTENT,
    DETECTION_STATE,
    META_THRESHOLD,
)
from config.profiles import NETWORK_PROFILES
from .habits import AccessPrior, HabitPrior
from .markov_blankets import MarkovBlanketL1L2
from . import duration
from .phenotype import PhenotypeConfig, EXPERT_PHENOTYPE
from utils.math_utils import (
    clamp_activation,
    clip_probability,
    convex_blend,
    networks_to_tensor,
    ou_step_scalar,
)


def descending_prediction_weight(meta_awareness: float) -> float:
    """g * m_t, defensively clipped to [0, 1] (see Eq. mu_blend in the supplement).

    DESCENDING_PREDICTION_GAIN is itself asserted < 1 at config load, so this
    clip is a numerical safety net, not the source of the bound: the decoder
    contribution to mu_x_effective can never reach 1.0.
    """
    return clip_probability(DESCENDING_PREDICTION_GAIN * meta_awareness)

class ThoughtseedModel(nn.Module):
    """Linear decoder + linear forward model.

    There is no separate encoder network: the bottom-up "recognition" step is
    the closed-form correction implemented in `correct()`.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.forward_net = nn.Linear(input_dim + latent_dim, input_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Linear decode: generative map p(x|z) mean, W z + b."""
        return self.decoder(z)

    def predict_next(self, x_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Linear forward dynamics: predict next networks given (x_t, z_t)."""
        squeeze = x_t.dim() == 1
        if squeeze:
            x_t = x_t.unsqueeze(0)
            z_t = z_t.unsqueeze(0)
        combined = torch.cat([x_t, z_t], dim=-1)
        out = self.forward_net(combined)
        return out.squeeze(0) if squeeze else out

    def correct(
        self,
        x_obs: torch.Tensor,
        prior_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form correction under an assumed unit observation covariance:

            z* = (W^T W + p_inv * I)^-1 [W^T (x - b) + p_inv * prior_mean]

        Unit observation covariance is a modelling assumption, not a consequence
        of the normalized L1 activation scale; p_inv = P_INV is the latent prior
        precision derived from the OU process. Supports both
        unbatched ([D_x]/[D_z]) and batched ([B, D_x]/[B, D_z]) inputs, the latter
        used for evaluating counterfactual policy predictions in parallel.
        """
        squeeze = x_obs.dim() == 1
        if squeeze:
            x_obs = x_obs.unsqueeze(0)
            prior_mean = prior_mean.unsqueeze(0)

        W = self.decoder.weight   # [D_x, D_z]
        b = self.decoder.bias     # [D_x]
        dz = W.shape[1]
        eye = torch.eye(dz, dtype=W.dtype, device=W.device)
        A = W.T @ W + P_INV * eye                                # [D_z, D_z]
        resid = x_obs - b                                        # [B, D_x]
        rhs = resid @ W + P_INV * prior_mean                      # [B, D_z]

        z_star = torch.linalg.solve(A, rhs.T).T                   # [B, D_z]
        return z_star.squeeze(0) if squeeze else z_star


class Layer2Agent(nn.Module):
    """Continuous thoughtseed inference, policy scoring, and action construction."""
    
    def __init__(self, phenotype: PhenotypeConfig = None,
                 blanket_l1l2: Optional[MarkovBlanketL1L2] = None):
        super().__init__()

        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level
        
        self.blanket_l1l2 = blanket_l1l2 or MarkovBlanketL1L2()
        
        self.thoughtseed_model = ThoughtseedModel(
            input_dim=len(NETWORKS),
            latent_dim=len(THOUGHTSEEDS),
        )

        self.mu_params = nn.ParameterDict()
        for state in STATES:
            priors = THOUGHTSEED_STATE_PRIORS[state].copy()
            mu_vec = [priors[ts] for ts in THOUGHTSEEDS]
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32), requires_grad=False)
        self.register_buffer(
            "mu_stack",
            torch.stack([self.mu_params[s].detach() for s in STATES]),
        )
        # System 1 learned defaults: which transition is expected (habit) and
        # which content wins workspace access (access prior beta).
        self.habits = HabitPrior()
        self.access_prior = AccessPrior()
        self.reset_detection()

    def reset_detection(self) -> None:
        """Clear transient detection completion, including at frozen-rollout reset."""
        self.detection_completed = False

    def observe_detection(self, current_state: str, accessed: Optional[str],
                          meta_awareness: float) -> bool:
        """Complete detection when clarity crosses b_m under lapse-detection content.

        Recognition is graded, not instantaneous: the detection content being
        broadcast is the inkling, and detection completes only once monitoring
        clarity reaches the same meaningful-monitoring threshold that gates
        access learning. Because MA still carries mind-wandering content, the
        detection content must first win the workspace against it, so the
        latency reflects access competition as well as the clarity crossing.
        """
        if current_state != DETECTION_STATE:
            self.reset_detection()
            return False
        event = (
            not self.detection_completed
            and accessed == DETECTION_CONTENT
            and float(meta_awareness) >= float(META_THRESHOLD)
        )
        self.detection_completed = self.detection_completed or event
        return event

    def _ou_step_z(
        self,
        current_state: str,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """OU-like latent dynamics for thoughtseeds (slow latent causes)."""
        dt = float(DEFAULT_DT)
        tau = max(float(LATENT_TAU), dt)
        mu = self.mu_params[current_state].detach()
        return ou_step_scalar(
            value=z_prev,
            target=mu,
            dt=dt,
            tau=tau,
            noise_level=float(NOISE_LEVEL),
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX,
        )

    def decode_with_state(self, z: torch.Tensor) -> torch.Tensor:
        """Top-down: decode thoughtseeds -> networks."""
        if z.dim() == 1:
            z_in = z.unsqueeze(0)
            decoded = self.thoughtseed_model.decode(z_in).squeeze(0)
        else:
            decoded = self.thoughtseed_model.decode(z)
        
        return torch.clamp(decoded, 0.0, 1.0)

    def reconstruction_error(
        self,
        z: torch.Tensor,
        observed_x: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder reconstruction error sum((x - (Wz+b))^2).

        This is the local learning loss's decoder term, and the only part of
        `compute_vfe` that carries gradients into (W, b).
        """
        recon_x = self.thoughtseed_model.decode(z)
        observed_x = observed_x.detach()

        return torch.sum((recon_x - observed_x) ** 2)

    def compute_vfe(
        self,
        z: torch.Tensor,
        observed_x: torch.Tensor,
        prior_target: torch.Tensor,
    ) -> torch.Tensor:
        """MAP inference objective solved by `correct()`, reported as a diagnostic:

            F(z) = sum((x - (Wz+b))^2) + P_INV * sum((z - prior)^2)

        Uses the raw linear decoder and the OU-propagated prior from the same
        correction step. At every call site both z and prior_target are already
        detached, so the prior-deviation term carries no parameter gradient; the
        learning step therefore uses `reconstruction_error` directly.
        """
        recon_loss = self.reconstruction_error(z, observed_x)
        prior_match = P_INV * torch.sum((z - prior_target) ** 2)

        return recon_loss + prior_match
    
    def update_posterior_z(
        self,
        z_prior: torch.Tensor,
        observed_x: torch.Tensor,
    ) -> torch.Tensor:
        """Posterior update q(z): closed-form, fixed-prior-covariance correction.

        z_prior is the OU-propagated predict-step mean z_t^-; the correction
        objective's prior covariance is the fixed P = P_INV^-1, not the
        state-conditioned attractor mu_z(s_t) -- that attractor only shapes
        z_prior itself, via the OU drift. One `torch.linalg.solve` call, no
        gradient steps, no learned recognition net.
        """
        with torch.no_grad():
            z_star = self.thoughtseed_model.correct(observed_x, z_prior)
            z_star = clamp_activation(z_star, CLIP_MIN, CLIP_MAX)
        return z_star

    def infer_z_step(
        self,
        current_state: str,
        activations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer corrected thoughtseeds for the current step: predict (OU) then correct (closed form)."""
        z_prior = self._ou_step_z(
            current_state,
            activations,
        )

        network_acts = self.blanket_l1l2.sensory_states
        x = networks_to_tensor(network_acts, NETWORKS)

        z_posterior = self.update_posterior_z(
            z_prior=z_prior,
            observed_x=x,
        )
        return z_posterior, z_prior

    def infer_state_belief(self, z: torch.Tensor) -> Dict[str, float]:
        """Map thoughtseed activations to a normalized regime belief."""
        with torch.no_grad():
            mu_all = self.mu_stack
            if mu_all.device != z.device or mu_all.dtype != z.dtype:
                mu_all = mu_all.to(device=z.device, dtype=z.dtype)
            dist_vec = torch.mean((mu_all - z) ** 2, dim=-1)
            logits = -0.5 * dist_vec / float(STATE_BELIEF_VAR)
            probs = torch.softmax(logits, dim=0).tolist()
            return {state: float(probs[i]) for i, state in enumerate(STATES)}

    def _compute_dwell_prior(self, current_state: str, candidates: list,
                             hazard: float, exit_probs: dict) -> list:
        """Construct the stay/switch prior from the configured duration hazard."""
        return [
            max(EPS, 1.0 - hazard) if state == current_state else max(EPS, hazard * exit_probs[state])
            for state in candidates
        ]

    def _evaluate_policy_score(
        self,
        x_current: torch.Tensor,
        candidates: list,
    ):
        """Compute pragmatic-control costs for the current policy candidates."""
        with torch.no_grad():
            mu_candidates = [self.mu_params[s] for s in candidates]
            mu_c_stack = torch.stack(mu_candidates)
            x_pred = self.thoughtseed_model.predict_next(x_current.expand(len(candidates), -1), mu_c_stack)
            x_pref = self.decode_with_state(mu_c_stack)

            g_vals = [float(g) for g in torch.mean((x_pred - x_pref) ** 2, dim=-1).tolist()]

        return g_vals, mu_candidates

    def _select_attractor(self, q_pi, mu_candidates: list):
        """Posterior-weighted latent prediction.

        mu = sum_pi q(pi) mu_z(s_pi)
        Returns (selected_mu, mu_x).
        """
        weights = torch.tensor(q_pi, dtype=mu_candidates[0].dtype)
        mu = torch.sum(weights.unsqueeze(-1) * torch.stack(mu_candidates, 0), dim=0)
        return mu, self.decode_with_state(mu)

    def evaluate_policy_evidence(
        self,
        current_state: str,
        x_current: torch.Tensor,
    ) -> Dict:
        """Evaluate policy evidence independent of the next hazard control."""
        candidates = get_policy_candidate_order(current_state)
        g_vals, mu_candidates = self._evaluate_policy_score(x_current, candidates)
        return {
            'candidates':     candidates,
            'g_vals':         g_vals,
            'mu_candidates':  mu_candidates,
        }

    def dwell_priors_for_next_l1_update(
        self,
        current_state: str,
        candidates: list,
        hazard_modulation: float,
    ) -> list:
        """Form the dwell prior for the next L1 exit draw under new control.

        L1 has already completed its update for t when L2 runs. The control
        formed here applies at t+1, after L1 increments the current dwell age;
        this uses that same prospective age and broadcast modulation.
        """
        next_dwell_age = float(self.blanket_l1l2.sensory_states['dwell_age']) + 1.0
        hazard, exit_probs = duration.exit_law(
            self.level, current_state, next_dwell_age, hazard_modulation,
        )
        return self._compute_dwell_prior(current_state, candidates, hazard, exit_probs)

    def broadcast_hazard_modulation(self, accessed: Optional[str], current_state: str) -> float:
        """Stabilize broadcast content, except after detection completes in MA."""
        if accessed is None or (current_state == DETECTION_STATE and self.detection_completed):
            return 0.0
        return -float(THOUGHTSEED_DIAGNOSTICITY[accessed][current_state])

    def descending_control(self, l3_active_states: Dict, current_state: str,
                           candidates: list, mu_candidates: list) -> Dict:
        """Transform L3's returned active states into the L2->L1 control signals.

        L1 never reads L3 directly: meta-awareness m_t and the accessed content
        arrive here across the L2-L3 interface and are converted into the
        descending prediction (gain g*m_t), the policy distribution, and the
        hazard modulation $-d_o(s_t)$ of the accessed content. It reduces the
        exit hazard in proportion to how well broadcast content fits the
        ongoing regime. Completed detection releases this stabilization in MA,
        restoring its baseline Gamma hazard without changing destination priors.
        """
        posterior = l3_active_states['policy_posterior']
        meta_awareness = float(l3_active_states['meta_awareness'])
        accessed = l3_active_states.get('accessed_content')
        q_pi = [posterior[candidate] for candidate in candidates]
        selected_mu, mu_x_policy = self._select_attractor(q_pi, mu_candidates)
        mu_x_state = networks_to_tensor(NETWORK_PROFILES[current_state][self.level], NETWORKS).to(mu_x_policy.device)
        mu_x_effective = convex_blend(mu_x_state, mu_x_policy, descending_prediction_weight(meta_awareness))
        hazard_modulation = self.broadcast_hazard_modulation(accessed, current_state)
        return {
            'selected_action_mu': selected_mu,
            'mu_x':               mu_x_effective,
            'policy_state_probs': {state: float(q_pi[i]) for i, state in enumerate(candidates)},
            'hazard_modulation':  float(hazard_modulation),
        }
