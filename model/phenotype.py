"""Per-phenotype parameters; model code reads PhenotypeConfig and never branches on a label."""

from dataclasses import dataclass
from config.profiles import PRACTICE_STEPS

@dataclass(frozen=True)
class PhenotypeConfig:
    """level keys the config.profiles tables."""
    level:         str
    practice_steps: int

EXPERT_PHENOTYPE = PhenotypeConfig(
    level='expert',
    practice_steps=PRACTICE_STEPS['expert'],
)

NOVICE_PHENOTYPE = PhenotypeConfig(
    level='novice',
    practice_steps=PRACTICE_STEPS['novice'],
)
