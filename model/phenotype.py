"""Phenotype configuration for expert/novice meditation agents.

All phenotype-specific constants live here. The rest of the model code
is phenotype-agnostic - it reads fields from PhenotypeConfig rather than
branching on a string label.
"""

from dataclasses import dataclass
from config.profiles import LEARNING_RATES, PRACTICE_STEPS

@dataclass(frozen=True)
class PhenotypeConfig:
    """Single source of truth for all per-phenotype parameters.

    level is the key into DWELL_MEAN_SECONDS/NETWORK_PROFILES/STATE_TRANSITION_PROBS;
    theta_scale scales the Layer 1 coupling template (see substrate.py);
    practice_steps is the training exposure.
    """
    level:         str
    learning_rate: float
    theta_scale:   float
    practice_steps: int
    label:         str

EXPERT_PHENOTYPE = PhenotypeConfig(
    level='expert',
    learning_rate=LEARNING_RATES['expert'],
    theta_scale=1.1,
    practice_steps=PRACTICE_STEPS['expert'],
    label='EXPERT',
)

NOVICE_PHENOTYPE = PhenotypeConfig(
    level='novice',
    learning_rate=LEARNING_RATES['novice'],
    theta_scale=1.0,
    practice_steps=PRACTICE_STEPS['novice'],
    label='NOVICE',
)
