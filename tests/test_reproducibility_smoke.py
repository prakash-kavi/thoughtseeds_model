import numpy as np

from model.phenotype import EXPERT_PHENOTYPE
from model.training_loop import MeditationTrainer


def test_fixed_seed_produces_identical_core_histories():
    first = MeditationTrainer(EXPERT_PHENOTYPE, seed=42).fit(timesteps=40)
    second = MeditationTrainer(EXPERT_PHENOTYPE, seed=42).fit(timesteps=40)
    for key in ('forward_error_history', 'free_energy_history', 'meta_awareness_history', 'state_history',
                'detection_event_history', 'detection_completed_history'):
        assert np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
