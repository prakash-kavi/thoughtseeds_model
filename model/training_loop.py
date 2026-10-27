"""Online training loop for the three-layer simulation; `_predict_correct_act_step` defines the per-step order
(Algorithm 1)."""

from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from config.defaults import (
    CLIP_MAX, CLIP_MIN, GRAD_CLIP, LEARNING_RATE, MODEL_VERSION,
    NETWORKS, THOUGHTSEEDS, THOUGHTSEED_STATE_PRIORS,
)
from utils.math_utils import networks_to_tensor
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import EXPERT_PHENOTYPE, PhenotypeConfig
from .policy import ReturnPolicy
from .substrate import Layer1Process
from .thoughtseeds import Layer2Agent
from .workspace import GlobalWorkspace


_STEP_KEYS = (
    'surprisal', 'clarity', 'lapse_belief', 'accessed_content',
    'detection_event', 'detection_completed', 'network_activations',
    'thoughtseed_activations', 'regime_belief',
)


class MeditationTrainer:
    """Run predict, correct, access and select, act, and learn at each simulation step."""

    def __init__(self, phenotype: PhenotypeConfig = None, seed: int = 42):
        self.phenotype = phenotype or EXPERT_PHENOTYPE
        self.level = self.phenotype.level
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.blanket_l1l2 = MarkovBlanketL1L2()
        self.blanket_l2l3 = MarkovBlanketL2L3()
        self.process = Layer1Process(phenotype=self.phenotype, seed=seed)
        # Built before the optimizer, which binds to its decoder parameters.
        self.agent = Layer2Agent(self.level, blanket_l1l2=self.blanket_l1l2)
        self.workspace = GlobalWorkspace(self.level)
        self.policy = ReturnPolicy(self.level)
        self.optimizer = optim.SGD(self.agent.parameters(), lr=LEARNING_RATE)

        self.history: Dict[str, list] = {}

    def _predict_correct_act_step(self, activations: torch.Tensor) -> tuple[Dict, str, torch.Tensor, Dict]:
        with torch.no_grad():
            network_acts, current_state = self.process.update(self.blanket_l1l2.active_states)
            self.blanket_l1l2.update_sensory_states(network_acts)
            x_current = networks_to_tensor(network_acts, NETWORKS)

            z_star = self.agent.infer_z_step(activations)
            self.blanket_l2l3.update_sensory_states({
                'thoughtseed_activations': z_star.cpu().numpy().tolist(),
            })
            sensory = self.blanket_l2l3.sensory_states

            # Access and the lapse evidence read m_{t-1}; the return act uses D_t m_t q_t.
            previous_access = self.blanket_l2l3.active_states.get('accessed_content')
            previous_clarity = self.workspace.clarity
            accessed_content = self.workspace.select_access(z_star=sensory['thoughtseed_activations'])
            clarity = self.workspace.update_clarity(sensory['thoughtseed_activations'])
            detection_event = self.workspace.update_detection(previous_clarity)
            precision = ReturnPolicy.precision(clarity, self.workspace.lapse_belief, self.workspace.detection_completed)
            self.blanket_l2l3.update_active_states({
                'accessed_content': accessed_content,
                'detection_completed': self.workspace.detection_completed,
                'act': self.policy.probabilities(precision),
                'exit_odds_factor': self.policy.exit_odds_factor(precision),
            })
            returned_content = self.blanket_l2l3.active_states['accessed_content']
            detection_completed = self.blanket_l2l3.active_states['detection_completed']

            # L1 uses these at its next update.
            self.blanket_l1l2.update_active_states({
                'accessed_content': returned_content,
                'detection_completed': detection_completed,
                'act': self.blanket_l2l3.active_states['act'],
                'exit_odds_factor': self.blanket_l2l3.active_states['exit_odds_factor'],
            })

            metrics = {
                'surprisal': self.agent.surprisal,
                'clarity': clarity,
                'lapse_belief': self.workspace.lapse_belief,
                'accessed_content': accessed_content,
                'detection_event': detection_event,
                'detection_completed': detection_completed,
                'network_activations': {
                    network: float(value.detach().item()) for network, value in network_acts.items()
                },
                'thoughtseed_activations': z_star.cpu().numpy().tolist(),
                'regime_belief': self.agent.regime_belief.tolist(),
            }
            buffer = {
                'x_curr': x_current,
                'z_star': z_star,
                'previous_access': previous_access,
                'accessed_content': accessed_content,
                'clarity': clarity,
                'precision': precision,
                'detection_event': detection_event,
            }
        return buffer, current_state, z_star, metrics

    def _local_learning_step(self, buffer: Dict) -> None:
        recon_loss = self.agent.reconstruction_error(
            z=buffer['z_star'],
            observed_x=buffer['x_curr'],
        )
        self.agent.learn_observation_precision(float(recon_loss.item()))
        self.optimizer.zero_grad()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=GRAD_CLIP)
        self.optimizer.step()
        self.workspace.access_prior.observe(
            buffer['previous_access'], buffer['accessed_content'], buffer['clarity'], self.workspace.resting_clarity,
        )
        self.workspace.learn_tonic()
        if buffer['detection_event']:
            # One count per recognized lapse, at its onset.
            self.policy.learn(buffer['precision'])

    def _run(self, timesteps: int, learn: bool, run_seed: int, preserve_learned: bool) -> Dict:
        self._reset_run_state(run_seed=run_seed, preserve_learned=preserve_learned)
        self.process.reset()
        current_state = self.process.current_state
        activations = torch.tensor(
            [THOUGHTSEED_STATE_PRIORS[current_state][thoughtseed] for thoughtseed in THOUGHTSEEDS],
            dtype=torch.float32,
        ).clamp(CLIP_MIN, CLIP_MAX)
        self.blanket_l1l2.update_active_states({'accessed_content': None, 'detection_completed': False})

        for t in range(timesteps):
            buffer, new_state, activations, metrics = self._predict_correct_act_step(activations)
            if new_state != current_state:
                self.history['transitions'].append({
                    'timestamp': t,
                    'from': current_state,
                    'to': new_state,
                })
                current_state = new_state
            self.history['states'].append(current_state)
            for key in _STEP_KEYS:
                self.history[key].append(metrics[key])
            if learn:
                self._local_learning_step(buffer)
        return self._package_results()

    def fit(self, timesteps: int) -> Dict:
        return self._run(timesteps, learn=True, run_seed=self.seed, preserve_learned=False)

    def rollout(self, timesteps: int, seed: int) -> Dict:
        return self._run(timesteps, learn=False, run_seed=seed, preserve_learned=True)

    def _reset_run_state(self, run_seed: int, preserve_learned: bool) -> None:
        """Reset transient state; learned quantities are kept unless a fit starts."""
        self.history = {'states': [], 'transitions': [], **{key: [] for key in _STEP_KEYS}}
        self.blanket_l1l2.reset()
        self.blanket_l2l3.reset()
        if not preserve_learned:
            self.workspace.access_prior.reset()
            self.workspace.reset_tonic()
            self.policy.reset()
        self.workspace.reset()
        self.agent.reset_inference()
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        self.process.rng = np.random.RandomState(run_seed)

    def _package_results(self) -> Dict:
        return {
            'experience_level': self.level,
            'model_version': MODEL_VERSION,
            'seed': self.seed,
            'timesteps': len(self.history['states']),
            'surprisal_history': self.history['surprisal'],
            'clarity_history': self.history['clarity'],
            'lapse_belief_history': self.history['lapse_belief'],
            'accessed_content_history': self.history['accessed_content'],
            'access_prior': self.workspace.access_prior.mean().tolist(),
            'return_habit': self.policy.habit,
            'observation_precision': self.agent.observation_precision,
            'tonic_contrast': self.workspace.tonic_contrast,
            'resting_clarity': self.workspace.resting_clarity,
            'decoder': {
                'weight': self.agent.thoughtseed_model.decoder.weight.detach().tolist(),
                'bias': self.agent.thoughtseed_model.decoder.bias.detach().tolist(),
            },
            'detection_event_history': self.history['detection_event'],
            'detection_completed_history': self.history['detection_completed'],
            'state_history': self.history['states'],
            'transitions': self.history['transitions'],
            'network_activations_history': self.history['network_activations'],
            'thoughtseed_activations_history': self.history['thoughtseed_activations'],
            'regime_belief_history': self.history['regime_belief'],
        }
