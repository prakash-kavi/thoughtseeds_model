"""Fit each phenotype once, then evaluate it in a fresh frozen rollout."""

import json
from pathlib import Path

from model.training_loop import MeditationTrainer
from model.phenotype import EXPERT_PHENOTYPE, NOVICE_PHENOTYPE
from config.defaults import EVALUATION_STEPS, PLOT_STEPS

SEED = 103
ROBUSTNESS_SEEDS = (100, 101, 102, 103, 104)

CURRENT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = CURRENT_DIR / "data"


def result_path(level: str, seed: int) -> Path:
    return OUTPUT_DIR / f"training_results_{level}_seed{seed}.json"


def _combine_fit_and_rollout(fit_results, rollout_results, evaluation_seed):
    """Join fit and held-out rollout histories for full-run diagnostics."""
    keys = (
        'surprisal_history', 'clarity_history', 'lapse_belief_history',
        'accessed_content_history', 'state_history', 'network_activations_history', 'thoughtseed_activations_history',
        'regime_belief_history',
        'detection_event_history', 'detection_completed_history',
    )
    combined = {key: fit_results[key] + rollout_results[key] for key in keys}
    combined['transitions'] = list(fit_results['transitions']) + [
        {**transition, 'timestamp': transition['timestamp'] + fit_results['timesteps']}
        for transition in rollout_results['transitions']
    ]
    combined.update({
        'experience_level': fit_results['experience_level'],
        'model_version': fit_results['model_version'],
        'access_prior': fit_results['access_prior'],
        'return_habit': fit_results['return_habit'],
        'observation_precision': fit_results['observation_precision'],
        'resting_clarity': fit_results['resting_clarity'],
        'decoder': fit_results['decoder'],
        'seed': fit_results['seed'],
        'evaluation_seed': evaluation_seed,
        'timesteps': len(combined['state_history']),
        'training_steps': fit_results['timesteps'],
        'phase_history': ['fit'] * fit_results['timesteps'] + ['frozen_rollout'] * rollout_results['timesteps'],
    })
    return combined


def _fit_and_evaluate(phenotype, seed):
    trainer = MeditationTrainer(phenotype=phenotype, seed=seed)
    fit_results = trainer.fit(phenotype.practice_steps)
    evaluation_seed = seed + 1
    rollout_results = trainer.rollout(EVALUATION_STEPS, seed=evaluation_seed)
    return _combine_fit_and_rollout(fit_results, rollout_results, evaluation_seed)


def _save_results(results, seed):
    path = result_path(results['experience_level'], seed)
    with open(path, 'w') as handle:
        json.dump(results, handle, indent=2)
    return path


def _run_phase(number, phenotype, seed):
    print("\n" + "=" * 70)
    print(f"PHASE {number}: {phenotype.level.upper()} (fit -> frozen rollout; seed {seed})")
    print("=" * 70)
    results = _fit_and_evaluate(phenotype, seed)
    path = _save_results(results, seed)
    n = len(results["state_history"])
    print(f"    Done: {n} steps ({n - EVALUATION_STEPS} fit, {EVALUATION_STEPS} frozen; final {PLOT_STEPS} plotted)")
    return results, path


def run_training_and_simulation(seed: int = SEED):
    """Fit each phenotype, then run one independent frozen evaluation rollout."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expert_results, expert_path = _run_phase(1, EXPERT_PHENOTYPE, seed)
    novice_results, novice_path = _run_phase(2, NOVICE_PHENOTYPE, seed)

    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"  - {expert_path.name}")
    print(f"  - {novice_path.name}")
    print()

    return expert_results, novice_results


def load_results(seed: int = SEED):
    """Load one full fit-plus-frozen-rollout result per phenotype."""
    expert_path = result_path("expert", seed)
    novice_path = result_path("novice", seed)
    if not expert_path.exists() or not novice_path.exists():
        print("ERROR: Could not find saved results:")
        print(f"  Looking for: {expert_path}")
        print(f"  Looking for: {novice_path}")
        print("\nRun 'python run_model.py run' first to generate results.")
        return None, None

    print(f"\nLoading results from {OUTPUT_DIR}/...")
    with open(expert_path, "r") as f:
        expert_results = json.load(f)
    with open(novice_path, "r") as f:
        novice_results = json.load(f)
    return expert_results, novice_results
