"""Online training loop for the state-conditioned three-layer simulation."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim

from config.defaults import (
    CLIP_MAX, CLIP_MIN, GRAD_CLIP, MODEL_VERSION,
    NETWORKS, THOUGHTSEEDS, THOUGHTSEED_STATE_PRIORS,
)
from config.profiles import NETWORK_PROFILES
from utils.math_utils import forward_error, networks_to_tensor
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import EXPERT_PHENOTYPE, PhenotypeConfig
from .policy import PolicyEngine
from .substrate import Layer1Process
from .thoughtseeds import Layer2Agent, descending_prediction_weight  # noqa: F401 (re-export)
from .workspace import GlobalWorkspace


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
        self.agent = Layer2Agent(phenotype=self.phenotype, blanket_l1l2=self.blanket_l1l2)
        self.policy = PolicyEngine()
        self.workspace = GlobalWorkspace()
        self.optimizer = optim.SGD(self.agent.parameters(), lr=self.phenotype.learning_rate)

        self.history: Dict[str, list] = {}
        self._last_observation: Optional[Dict[str, torch.Tensor]] = None

    def _predict_correct_act_step(self, t: int, activations: torch.Tensor) -> tuple[Dict, str, torch.Tensor, Dict]:
        with torch.no_grad():
            previous = self._last_observation
            network_acts, current_state = self.process.update(self.blanket_l1l2.active_states)
            self.blanket_l1l2.update_sensory_states({
                **network_acts,
                'dwell_age': float(self.process.current_dwell),
            })
            x_current = networks_to_tensor(network_acts, NETWORKS)

            forward_error_value = 0.0
            if previous is not None:
                x_prediction = self.agent.thoughtseed_model.predict_next(previous['x'], previous['action'])
                forward_error_value = float(forward_error(x_prediction, x_current).item())

            z_star, z_prior = self.agent.infer_z_step(current_state, activations)
            z_star = z_star.detach()
            free_energy = float(self.agent.compute_vfe(
                z=z_star,
                observed_x=x_current,
                prior_target=z_prior,
            ).item())
            state_belief = self.agent.infer_state_belief(z_star)
            policy_eval = self.agent.evaluate_policy_evidence(current_state, x_current)
            candidates = policy_eval['candidates']
            # L2 -> L3 sensory states, including the System 1 learned defaults.
            self.blanket_l2l3.update_sensory_states({
                'state_belief': state_belief,
                'policy_candidates': candidates,
                'policy_costs': policy_eval['g_vals'],
                'habit_log_prior': self.agent.habits.log_prior(state_belief, candidates),
                'access_prior': self.agent.access_prior.mean().tolist(),
                'thoughtseed_activations': z_star.cpu().numpy().tolist(),
                'thoughtseed_prior_activations': z_prior.detach().cpu().numpy().tolist(),
            })
            sensory = self.blanket_l2l3.sensory_states

            # L3 reads only its sensory states and returns its active states:
            # fast access first (using m_{t-1} for top-down amplification), then
            # slow meta-awareness, then policy arbitration.
            previous_access = self.blanket_l2l3.active_states.get('accessed_content')
            accessed_content = self.workspace.select_access(
                z_star=sensory['thoughtseed_activations'],
                z_prior=sensory['thoughtseed_prior_activations'],
                access_prior=sensory['access_prior'],
            )
            meta_awareness = self.workspace.update_meta_awareness(
                g_vals=sensory['policy_costs'],
                habit_log_prior=sensory['habit_log_prior'],
            )
            self.blanket_l2l3.update_active_states({
                'accessed_content': accessed_content,
                'meta_awareness': meta_awareness,
            })
            returned_content = self.blanket_l2l3.active_states['accessed_content']
            detection_event = self.agent.observe_detection(current_state, returned_content, meta_awareness)
            hazard_modulation = self.agent.broadcast_hazard_modulation(returned_content, current_state)
            dwell_priors = self.agent.dwell_priors_for_next_l1_update(
                current_state, candidates, hazard_modulation,
            )
            self.blanket_l2l3.update_sensory_states({'policy_priors': dwell_priors})
            q_pi = self.policy.select_policy(
                g_vals=sensory['policy_costs'],
                dwell_priors=dwell_priors,
                candidates=candidates,
                habit_log_prior=sensory['habit_log_prior'],
                meta_precision=meta_awareness,
            )
            self.blanket_l2l3.update_active_states({
                'policy_posterior': {
                    candidate: float(q_pi[index])
                    for index, candidate in enumerate(candidates)
                },
                'accessed_content': accessed_content,
                'meta_awareness': meta_awareness,
            })

            # L2 converts L3's active states into the L2 -> L1 control signals.
            control = self.agent.descending_control(
                self.blanket_l2l3.active_states, current_state, candidates, policy_eval['mu_candidates'],
            )
            self.blanket_l1l2.update_active_states({
                'mu_x': control['mu_x'],
                'policy_state_probs': control['policy_state_probs'],
                'hazard_modulation': control['hazard_modulation'],
            })
            self._last_observation = {
                'x': x_current.detach(),
                'action': control['selected_action_mu'].detach(),
            }

            metrics = {
                'timestamp': t,
                'free_energy': free_energy,
                'forward_error': forward_error_value,
                'meta_awareness': meta_awareness,
                'accessed_content': accessed_content,
                'hazard_modulation': control['hazard_modulation'],
                'detection_event': detection_event,
                'detection_completed': self.agent.detection_completed,
                'network_activations': {
                    network: float(value.detach().item()) for network, value in network_acts.items()
                },
                'thoughtseed_activations': z_star.cpu().numpy().tolist(),
                'thoughtseed_prior_activations': z_prior.cpu().numpy().tolist(),
            }
            buffer = {
                'x_prev': previous['x'] if previous is not None else None,
                'action_prev': previous['action'] if previous is not None else None,
                'x_curr': x_current.detach(),
                'z_star': z_star,
                'z_prior': z_prior.detach(),
                'state_belief': state_belief,
                'previous_access': previous_access,
                'accessed_content': accessed_content,
                'meta_awareness': meta_awareness,
                'intended_by_source': {
                    current_state: {
                        candidate: float(q_pi[index])
                        for index, candidate in enumerate(candidates)
                    }
                },
            }
        return buffer, current_state, z_star, metrics

    def _local_learning_step(self, buffer: Dict) -> None:
        recon_loss = self.agent.reconstruction_error(
            z=buffer['z_star'],
            observed_x=buffer['x_curr'],
        )
        if buffer['x_prev'] is None:
            forward_loss = torch.zeros((), dtype=recon_loss.dtype, device=recon_loss.device)
        else:
            prediction = self.agent.thoughtseed_model.predict_next(buffer['x_prev'], buffer['action_prev'])
            forward_loss = forward_error(prediction, buffer['x_curr'])
        self.optimizer.zero_grad()
        (recon_loss + forward_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=GRAD_CLIP)
        self.optimizer.step()
        self.agent.habits.update(buffer['state_belief'], buffer['intended_by_source'])
        self.agent.access_prior.observe(
            buffer['previous_access'], buffer['accessed_content'], buffer['meta_awareness'],
        )

    def _run(self, timesteps: int, learn: bool, run_seed: int, preserve_habits: bool) -> Dict:
        self._reset_run_state(run_seed=run_seed, preserve_habits=preserve_habits)
        self.process.reset(state='breath_focus')
        current_state = self.process.current_state
        activations = torch.tensor(
            [THOUGHTSEED_STATE_PRIORS[current_state][thoughtseed] for thoughtseed in THOUGHTSEEDS],
            dtype=torch.float32,
        ).clamp(CLIP_MIN, CLIP_MAX)
        initial_profile = NETWORK_PROFILES[current_state][self.level]
        self.blanket_l1l2.update_active_states({
            'mu_x': networks_to_tensor(initial_profile, NETWORKS),
            'policy_state_probs': {current_state: 1.0},
            'hazard_modulation': 0.0,
        })

        for t in range(timesteps):
            buffer, new_state, activations, metrics = self._predict_correct_act_step(t, activations)
            if new_state != current_state:
                self.history['transitions'].append({
                    'timestamp': t,
                    'from': current_state,
                    'to': new_state,
                    'free_energy': metrics['free_energy'],
                })
                current_state = new_state
            self.history['states'].append(current_state)
            for key in ('free_energy', 'forward_error', 'meta_awareness', 'accessed_content', 'hazard_modulation', 'detection_event', 'detection_completed', 'network_activations', 'thoughtseed_activations', 'thoughtseed_prior_activations'):
                self.history[key].append(metrics[key])
            if learn:
                self._local_learning_step(buffer)
            self.process.x = self.process.x.detach()
            activations = activations.detach()
        return self._package_results()

    def fit(self, timesteps: int) -> Dict:
        return self._run(timesteps, learn=True, run_seed=self.seed, preserve_habits=False)

    def rollout(self, timesteps: int, seed: int) -> Dict:
        return self._run(timesteps, learn=False, run_seed=seed, preserve_habits=True)

    def _reset_run_state(self, run_seed: int, preserve_habits: bool) -> None:
        self.history = {
            'states': [],
            'free_energy': [],
            'forward_error': [],
            'meta_awareness': [],
            'accessed_content': [],
            'hazard_modulation': [],
            'detection_event': [],
            'detection_completed': [],
            'transitions': [],
            'network_activations': [],
            'thoughtseed_activations': [],
            'thoughtseed_prior_activations': [],
        }
        self._last_observation = None
        self.blanket_l1l2.reset()
        self.blanket_l2l3.reset()
        self.workspace.reset()
        self.agent.reset_detection()
        if not preserve_habits:
            self.agent.habits.reset()
            self.agent.access_prior.reset()
        self.agent.access_prior.clear_pending()
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        self.process.rng = np.random.RandomState(run_seed)

    def _package_results(self) -> Dict:
        return {
            'experience_level': self.level,
            'model_version': MODEL_VERSION,
            'seed': self.seed,
            'timesteps': len(self.history['states']),
            'free_energy_history': self.history['free_energy'],
            'forward_error_history': self.history['forward_error'],
            'meta_awareness_history': self.history['meta_awareness'],
            'accessed_content_history': self.history['accessed_content'],
            'access_prior': self.agent.access_prior.mean().tolist(),
            'hazard_modulation_history': self.history['hazard_modulation'],
            'detection_event_history': self.history['detection_event'],
            'detection_completed_history': self.history['detection_completed'],
            'state_history': self.history['states'],
            'transitions': self.history['transitions'],
            'network_activations_history': self.history['network_activations'],
            'thoughtseed_activations_history': self.history['thoughtseed_activations'],
            'thoughtseed_prior_activations_history': self.history['thoughtseed_prior_activations'],
        }
