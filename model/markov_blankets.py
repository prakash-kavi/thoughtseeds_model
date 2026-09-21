"""Typed sensory and active interfaces between adjacent model layers."""

from __future__ import annotations
import torch
from typing import Dict, Any

from config.defaults import NETWORKS

class MarkovBlanket:
    """Base class for Markov blanket interfaces with strict key contracts."""

    allowed_sensory: set[str] = set()
    allowed_active: set[str] = set()

    def __init__(self, strict: bool = True):
        self.strict = bool(strict)
        self.sensory_states: Dict[str, Any] = {}
        self.active_states: Dict[str, Any] = {}

    def _detach_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {k: self._detach_value(v) for k, v in value.items()}
        return value

    def _validate_keys(self, new_states: Dict[str, Any], allowed: set[str], kind: str) -> None:
        if not self.strict:
            return
        extra = set(new_states.keys()) - allowed
        if extra:
            raise KeyError(f"Unknown {kind} keys for {self.__class__.__name__}: {sorted(extra)}")
        
    def update_sensory_states(self, new_states: Dict[str, Any]) -> None:
        self._validate_keys(new_states, self.allowed_sensory, "sensory")
        for key, new_val in new_states.items():
            self.sensory_states[key] = self._detach_value(new_val)

    def update_active_states(self, new_states: Dict[str, Any]) -> None:
        self._validate_keys(new_states, self.allowed_active, "active")
        detached = {k: self._detach_value(v) for k, v in new_states.items()}
        self.active_states.update(detached)

    def reset(self) -> None:
        self.sensory_states.clear()
        self.active_states.clear()

class MarkovBlanketL1L2(MarkovBlanket):
    """L1 sensory observations and L2 active control signals."""

    def __init__(self):
        self.allowed_sensory = set(NETWORKS) | {"dwell_age"}
        self.allowed_active = {"mu_x", "policy_state_probs", "hazard_modulation"}
        super().__init__(strict=True)
        

class MarkovBlanketL2L3(MarkovBlanket):
    """L2 sensory input to L3, and L3's selection returned as active states."""

    def __init__(self):
        self.allowed_sensory = {
            "state_belief", "policy_candidates", "policy_priors", "policy_costs",
            "habit_log_prior", "access_prior", "thoughtseed_activations", "thoughtseed_prior_activations",
        }
        self.allowed_active = {"policy_posterior", "accessed_content", "meta_awareness"}
        super().__init__(strict=True)
