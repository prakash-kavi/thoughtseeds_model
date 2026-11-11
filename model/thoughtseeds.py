"""Layer 2 continuous thoughtseed inference."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from config.defaults import (
    STATES, NETWORKS, THOUGHTSEEDS, CLIP_MIN, CLIP_MAX,
    THOUGHTSEED_STATE_PRIORS, PRACTICE_STATE,
    DEFAULT_DT, LATENT_TAU,
    NOISE_LEVEL,
    LEARNING_RATE,
    P_INV,
)
from config.profiles import DWELL_MEAN_SECONDS, NETWORK_PROFILES, get_exit_transition_probs
from .markov_blankets import MarkovBlanketL1L2
from utils.math_utils import networks_to_tensor

_PHI = 1.0 - DEFAULT_DT / LATENT_TAU   # OU predict factor per step
_QDT = NOISE_LEVEL * DEFAULT_DT        # process-noise variance per step


class ThoughtseedModel(nn.Module):
    """Linear decoder W z + b with the closed-form correction."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.decoder = nn.Linear(latent_dim, input_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def correct(
        self,
        x_obs: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_precision: torch.Tensor,
        observation_precision: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kalman measurement update (eq:l2); accepts batched inputs.

            A  = Pi_x W^T W + Lambda^-
            z* = A^-1 [Pi_x W^T (x - b) + Lambda^- prior_mean],   posterior covariance A^-1
        """
        squeeze = x_obs.dim() == 1
        if squeeze:
            x_obs = x_obs.unsqueeze(0)
            prior_mean = prior_mean.unsqueeze(0)

        W = self.decoder.weight   # [D_x, D_z]
        b = self.decoder.bias     # [D_x]
        A = observation_precision * W.T @ W + prior_precision            # [D_z, D_z]
        rhs = observation_precision * (x_obs - b) @ W + prior_mean @ prior_precision  # [B, D_z]

        z_star = torch.linalg.solve(A, rhs.T).T                           # [B, D_z]
        posterior_cov = torch.linalg.inv(A)
        return (z_star.squeeze(0) if squeeze else z_star), posterior_cov


def regime_transition_matrix(level: str) -> np.ndarray:
    """B[i, j] = P(s_t = j | s_{t-1} = i): stay 1 - dt / dwell mean, exits split by the configured exit row."""
    B = np.zeros((len(STATES), len(STATES)))
    for i, state in enumerate(STATES):
        exit_p = DEFAULT_DT / DWELL_MEAN_SECONDS[level][state]
        B[i, i] = 1.0 - exit_p
        for successor, p in get_exit_transition_probs(level, state).items():
            B[i, STATES.index(successor)] += exit_p * p
    return B


def configured_decoder(level: str) -> tuple[np.ndarray, np.ndarray]:
    """Centred minimum-norm affine map W0, b0 from the regime thoughtseed priors to the configured network
    attractors, so that W0 mu_s + b0 is the attractor of regime s."""
    M = np.array([[THOUGHTSEED_STATE_PRIORS[s][ts] for ts in THOUGHTSEEDS] for s in STATES])
    X = np.array([[NETWORK_PROFILES[s][level][n] for n in NETWORKS] for s in STATES])
    W = (X - X.mean(axis=0)).T @ np.linalg.pinv((M - M.mean(axis=0)).T)
    return W, X.mean(axis=0) - W @ M.mean(axis=0)


def _log_normal(x: np.ndarray, means: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """ln N(x; mean_k, cov) for each row of means."""
    chol = np.linalg.cholesky(cov)
    diff = np.linalg.solve(chol, (x[None, :] - means).T)
    return -0.5 * (diff ** 2).sum(axis=0) - np.log(np.diag(chol)).sum() - 0.5 * x.size * np.log(2.0 * np.pi)


class Layer2Agent(nn.Module):
    """L2 (eq:l2): switching Kalman filter with a one-step collapse (GPB1); the regime belief is inferred from the
    networks."""

    def __init__(self, level: str, blanket_l1l2: Optional[MarkovBlanketL1L2] = None):
        super().__init__()

        self.blanket_l1l2 = blanket_l1l2 or MarkovBlanketL1L2()

        self.thoughtseed_model = ThoughtseedModel(
            input_dim=len(NETWORKS),
            latent_dim=len(THOUGHTSEEDS),
        )
        W0, b0 = configured_decoder(level)
        with torch.no_grad():
            self.thoughtseed_model.decoder.weight.copy_(torch.tensor(W0, dtype=torch.float32))
            self.thoughtseed_model.decoder.bias.copy_(torch.tensor(b0, dtype=torch.float32))
        self.transition = regime_transition_matrix(level)

        self.register_buffer(
            "mu_stack",
            torch.stack([
                torch.tensor(
                    [THOUGHTSEED_STATE_PRIORS[state][ts] for ts in THOUGHTSEEDS],
                    dtype=torch.float32,
                )
                for state in STATES
            ]),
        )
        # 1/Pi_x, initialized from the first prediction error.
        self.residual_variance: Optional[float] = None
        self.reset_inference()

    def reset_inference(self) -> None:
        """Start the carried covariance at its no-evidence fixed point P_inf = 1/P_INV, and the belief in BF."""
        self.posterior_cov = torch.eye(len(THOUGHTSEEDS)) / P_INV
        self.prior_precision = torch.eye(len(THOUGHTSEEDS)) * P_INV
        self.regime_belief = np.eye(len(STATES))[STATES.index(PRACTICE_STATE)]
        self.surprisal: Optional[float] = None

    @property
    def observation_precision(self) -> float:
        return 1.0 / self.residual_variance

    def learn_observation_precision(self, squared_error: float) -> None:
        """v <- v + eta ((|x - (Wz*+b)|^2 + tr(W P W^T)) / D_x - v); approximate when clipping moves z*."""
        W = self.thoughtseed_model.decoder.weight.detach()
        expected_error = squared_error + float(torch.trace(W @ self.posterior_cov @ W.T))
        self.residual_variance += LEARNING_RATE * (expected_error / len(NETWORKS) - self.residual_variance)

    def _predict_covariance(self) -> torch.Tensor:
        """OU predict of the covariance, phi^2 P + Q dt I, shared by every regime; sets the prior precision."""
        prior_cov = _PHI ** 2 * self.posterior_cov + _QDT * torch.eye(len(THOUGHTSEEDS))
        self.prior_precision = torch.linalg.inv(prior_cov)
        return prior_cov

    def reconstruction_error(
        self,
        z: torch.Tensor,
        observed_x: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder term of the learning loss: sum((x - (Wz+b))^2)."""
        recon_x = self.thoughtseed_model.decode(z)
        observed_x = observed_x.detach()

        return torch.sum((recon_x - observed_x) ** 2)

    def infer_z_step(self, activations: torch.Tensor) -> torch.Tensor:
        """Switching predict and correct; returns z*. Sets `surprisal` = -ln sum_s q^-(s) N(x_t; W z^-_s + b, S),
        None on the step that initializes the observation precision."""
        prior_cov = self._predict_covariance()
        x = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS)
        with torch.no_grad():
            model = self.thoughtseed_model
            z_modes_prior = _PHI * activations.unsqueeze(0) + (1.0 - _PHI) * self.mu_stack
            q_pred = self.regime_belief @ self.transition
            initializes_precision = self.residual_variance is None
            if initializes_precision:
                z_prior = torch.tensor(q_pred, dtype=torch.float32) @ z_modes_prior
                self.residual_variance = float(torch.mean((x - model.decode(z_prior)) ** 2))
            W = model.decoder.weight.double().numpy()
            b = model.decoder.bias.double().numpy()
            S = W @ prior_cov.double().numpy() @ W.T + np.eye(len(NETWORKS)) / self.observation_precision
            log_q = np.log(np.maximum(q_pred, 1e-300)) + _log_normal(
                x.double().numpy(), z_modes_prior.double().numpy() @ W.T + b, S)
            q = np.exp(log_q - log_q.max())
            self.surprisal = None if initializes_precision else -float(log_q.max() + np.log(q.sum()))
            q /= q.sum()
            z_modes, P_post = model.correct(x.expand(len(STATES), -1), z_modes_prior, self.prior_precision,
                                            self.observation_precision)
            weights = torch.tensor(q, dtype=torch.float32)
            z_mix = weights @ z_modes
            spread = z_modes - z_mix
            self.posterior_cov = P_post + (spread.T * weights) @ spread
            z_star = torch.clamp(z_mix, CLIP_MIN, CLIP_MAX)
        self.regime_belief = q
        return z_star
