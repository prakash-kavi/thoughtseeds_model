import numpy as np
import pytest
import torch

from config.defaults import DEFAULT_DT, NETWORKS, STATES, THOUGHTSEEDS, get_policy_candidate_order
from config.profiles import get_exit_transition_probs
from model import duration
from model.phenotype import EXPERT_PHENOTYPE, NOVICE_PHENOTYPE
from model.policy import PolicyEngine
from model.substrate import Layer1Process
from model.thoughtseeds import Layer2Agent
from model.training_loop import MeditationTrainer
from viz.analysis_utils import validate_network_schema
from viz.ignition import access_summary, breath_follow_through, detection_summary, meta_awareness_episodes


def test_detection_needs_clarity_to_cross_under_lapse_content_once_per_episode():
    from config.defaults import DETECTION_CONTENT, META_THRESHOLD
    agent = Layer2Agent(EXPERT_PHENOTYPE)
    clear, dim = float(META_THRESHOLD), float(META_THRESHOLD) - 0.05
    # Outside MA nothing is detected, whatever the content or the clarity.
    assert not agent.observe_detection('mind_wandering', DETECTION_CONTENT, 1.0)
    assert not agent.detection_completed
    # Inside MA the detection content alone is an inkling, not a detection:
    # clarity must reach the meaningful-monitoring threshold.
    assert not agent.observe_detection('meta_awareness', DETECTION_CONTENT, dim)
    assert not agent.detection_completed
    # Clarity alone is not enough either; the lapse content must hold the slot.
    assert not agent.observe_detection('meta_awareness', 'equanimity', 1.0)
    assert not agent.detection_completed
    assert agent.observe_detection('meta_awareness', DETECTION_CONTENT, clear)
    assert agent.detection_completed
    # One event per episode; completion is retained until MA ends.
    assert not agent.observe_detection('meta_awareness', DETECTION_CONTENT, 1.0)
    assert agent.detection_completed
    assert not agent.observe_detection('redirect_attention', DETECTION_CONTENT, 1.0)
    assert not agent.detection_completed
    assert agent.observe_detection('meta_awareness', DETECTION_CONTENT, clear)


@pytest.mark.parametrize('phenotype', [EXPERT_PHENOTYPE, NOVICE_PHENOTYPE])
def test_detection_restores_gamma_timing_without_changing_conditional_destinations(phenotype):
    agent = Layer2Agent(phenotype)
    state = 'meta_awareness'
    candidates = get_policy_candidate_order(state)
    agent.blanket_l1l2.update_sensory_states({'dwell_age': 11.0})
    modulation = agent.broadcast_hazard_modulation('aha_moment', state)
    before = agent.dwell_priors_for_next_l1_update(state, candidates, modulation)
    agent.observe_detection(state, 'aha_moment', 1.0)
    released_modulation = agent.broadcast_hazard_modulation('aha_moment', state)
    after = agent.dwell_priors_for_next_l1_update(state, candidates, released_modulation)
    h0, p0 = duration.exit_law(phenotype.level, state, 12, modulation)
    h1, p1 = duration.exit_law(phenotype.level, state, 12, released_modulation)
    assert 0 < h0 < h1 < 1
    assert released_modulation == 0.0
    assert h1 == duration.hazard(phenotype.level, state, 12)
    assert p0 == p1 == get_exit_transition_probs(phenotype.level, state)
    assert np.isclose(sum(after), 1.0)
    assert after[0] == pytest.approx(1 - h1)
    assert after[candidates.index('redirect_attention')] == pytest.approx(h1 * p1['redirect_attention'])
    engine = PolicyEngine()
    q0 = engine.select_policy([0.0] * 4, before, candidates, np.zeros(4), 0.4)
    q1 = engine.select_policy([0.0] * 4, after, candidates, np.zeros(4), 0.4)
    assert q1[candidates.index('redirect_attention')] > q0[candidates.index('redirect_attention')]
    # Detection increases exit-versus-stay mass, not RA's conditional exit share.
    process = Layer1Process(phenotype, seed=0)
    process.reset(state)
    k0 = process._exit_destination_kernel(dict(zip(candidates, q0)), p0)
    k1 = process._exit_destination_kernel(dict(zip(candidates, q1)), p1)
    assert k0 == pytest.approx(k1)
    assert all(0 < probability < 1 for probability in k1.values())


def test_l1_and_l2_apply_the_same_next_step_exit_probability():
    state = 'meta_awareness'
    h0, _ = duration.exit_law('expert', state, 3, -0.5)
    h1, _ = duration.exit_law('expert', state, 3, 0.0)

    class ControlledRandom:
        def random_sample(self):
            return (h0 + h1) / 2

        def choice(self, states, p):
            assert np.isclose(sum(p), 1.0)
            return 'mind_wandering'

    process = Layer1Process(EXPERT_PHENOTYPE)
    process.rng = ControlledRandom()
    policy = dict.fromkeys(STATES, 0.25)
    process.reset(state)
    process.current_dwell = 2
    assert process._check_transition(policy, -0.5) == state
    process.reset(state)
    process.current_dwell = 2
    assert process._check_transition(policy, 0.0) == 'mind_wandering'


def test_detection_control_is_confined_to_ma_and_absent_control_preserves_baseline():
    agent = Layer2Agent(EXPERT_PHENOTYPE)
    for state in STATES:
        for content in [None, *THOUGHTSEEDS]:
            agent.reset_detection()
            modulation = agent.broadcast_hazard_modulation(content, state)
            h, p = duration.exit_law('expert', state, 5, modulation)
            assert h == pytest.approx(duration.hazard('expert', state, 5) * (1 + modulation))
            assert p == get_exit_transition_probs('expert', state)
            agent.observe_detection('meta_awareness', 'aha_moment', 1.0)
            after = agent.broadcast_hazard_modulation(content, state)
            assert after == (0.0 if state == 'meta_awareness' else modulation)


@pytest.mark.parametrize('phenotype', [EXPERT_PHENOTYPE, NOVICE_PHENOTYPE])
def test_completed_detection_never_exceeds_baseline_gamma_hazard(phenotype):
    agent = Layer2Agent(phenotype)
    agent.observe_detection('meta_awareness', 'aha_moment', 1.0)
    for content in [None, *THOUGHTSEEDS]:
        modulation = agent.broadcast_hazard_modulation(content, 'meta_awareness')
        for age in range(1, len(duration._pmf(phenotype.level, 'meta_awareness')) + 2):
            h, destinations = duration.exit_law(phenotype.level, 'meta_awareness', age, modulation)
            assert h == duration.hazard(phenotype.level, 'meta_awareness', age)
            assert destinations == get_exit_transition_probs(phenotype.level, 'meta_awareness')


def test_invalid_hazard_control_is_rejected():
    for modulation in [float('nan'), float('inf'), -float('inf')]:
        with pytest.raises(ValueError, match='finite'):
            duration.exit_law('novice', 'meta_awareness', 3, modulation)


def test_detection_flows_through_l2_and_resets_for_frozen_rollout(monkeypatch):
    trainer = MeditationTrainer(EXPERT_PHENOTYPE, seed=42)
    trainer._reset_run_state(42, preserve_habits=False)
    trainer.process.reset('meta_awareness')
    trainer.blanket_l1l2.update_active_states({
        'mu_x': torch.full((len(NETWORKS),), 0.5),
        'policy_state_probs': dict.fromkeys(STATES, 0.25),
        'hazard_modulation': 0.0,
    })

    def access(**kwargs):
        trainer.workspace.accessed_content = 'aha_moment'
        return 'aha_moment'

    monkeypatch.setattr(trainer.workspace, 'select_access', access)
    # Detection needs clarity at the meaningful-monitoring threshold; this test
    # covers the plumbing, so start the monitor above it rather than waiting for
    # the crossing.
    trainer.workspace.meta_awareness = 1.0
    z = trainer.agent.state_attractor('meta_awareness')
    _, state, _, metrics = trainer._predict_correct_act_step(0, z)
    assert state == 'meta_awareness'
    assert metrics['detection_event'] and metrics['detection_completed']
    assert trainer.blanket_l1l2.active_states['hazard_modulation'] == 0.0
    assert trainer.blanket_l1l2.allowed_active == {'mu_x', 'policy_state_probs', 'hazard_modulation'}
    assert trainer.blanket_l2l3.allowed_active == {'policy_posterior', 'accessed_content', 'meta_awareness'}
    before = trainer.agent.access_prior.mean().copy()
    trainer._reset_run_state(43, preserve_habits=True)
    assert not trainer.agent.detection_completed
    assert np.array_equal(before, trainer.agent.access_prior.mean())


def test_window_boundary_does_not_create_an_access_or_monitor_onset():
    results = {
        'accessed_content_history': ['attend_breath'] * 4 + [None, 'attend_breath'],
        'state_history': ['breath_focus'] * 6,
        'meta_awareness_history': [0.6, 0.6, 0.6, 0.4, 0.6, 0.6],
    }
    summary = access_summary(results, window=4)
    assert summary['access_onsets'] == 1
    assert summary['meta_episodes'] == 1
    assert meta_awareness_episodes([0.6, 0.6, 0.4, 0.6], previous_value=0.6) == [(3, 4)]


def test_detection_latencies_separate_non_ra_and_censored_exits():
    ma, ra, mw = 'meta_awareness', 'redirect_attention', 'mind_wandering'
    results = {
        'state_history': [ma, ma, ra, ma, ma, mw, ma, ma],
        'detection_event_history': [False, True, False, True, False, False, False, True],
    }
    summary = detection_summary(results, window=8)
    assert summary['detected_ma_episodes'] == 3
    assert summary['direct_ra_exits'] == 1
    assert summary['other_exits'] == 1
    assert summary['unresolved_detected_episodes'] == 1
    assert summary['detection_latency_seconds'] == pytest.approx(2 * DEFAULT_DT / 3)
    assert summary['redirection_latency_seconds'] == pytest.approx(DEFAULT_DT)
    assert detection_summary(results, window=7)['detected_ma_episodes'] == 2


def test_follow_through_reports_the_residual_bf_run_after_an_in_bf_event():
    length = duration.median_steps('expert', 'breath_focus')
    states = ['breath_focus'] * length + ['mind_wandering']
    # An event during BF is measured over the remainder of that run, and the
    # run is reported as an interval rather than thresholded, so the same run
    # cannot count as sustained for one phenotype and brief for another.
    assert breath_follow_through([0, 1], states) == [(0, length), (1, length)]
    # No BF run before the window ends is reported as missing, not as zero.
    assert breath_follow_through([0], ['mind_wandering', 'meta_awareness']) == [None]


@pytest.mark.parametrize('old_version', [None, 'clarity-v4'])
def test_old_results_cannot_be_presented_as_the_revised_model(old_version):
    with pytest.raises(ValueError, match='Rerun'):
        validate_network_schema({'seed': 104, 'model_version': old_version}, NETWORKS)


def test_exports_resolve_manuscript_macros_and_record_model_version(tmp_path, monkeypatch):
    import re
    from pathlib import Path
    from config.defaults import MODEL_VERSION
    from scripts import aggregate_robustness, export_stats

    def synthetic_run(level, seed):
        rng = np.random.RandomState(2 if level == 'expert' else 3)
        states = [state for _ in range(3) for state in STATES for _ in range(6)]
        count = len(states)
        events = [state == 'meta_awareness' and i % 6 == 1 for i, state in enumerate(states)]
        contents = ['aha_moment' if state == 'meta_awareness' else 'attend_breath' for state in states]
        return {
            'model_version': MODEL_VERSION, 'seed': seed, 'evaluation_seed': seed + 1,
            'experience_level': level, 'training_steps': 12000, 'state_history': states,
            'accessed_content_history': contents, 'meta_awareness_history': [0.4] * count,
            'detection_event_history': events,
            'detection_completed_history': [s == 'meta_awareness' and i % 6 >= 1 for i, s in enumerate(states)],
            'free_energy_history': [0.1] * count, 'forward_error_history': [0.1] * count,
            'network_activations_history': [dict(zip(NETWORKS, row)) for row in rng.uniform(0.1, 0.8, (count, 7))],
            'thoughtseed_activations_history': rng.uniform(0.1, 0.8, (count, 5)).tolist(),
            'thoughtseed_prior_activations_history': rng.uniform(0.1, 0.8, (count, 5)).tolist(),
            'access_prior': [0.2] * 5,
            'transitions': [
                {'timestamp': i, 'from': states[i - 1], 'to': states[i]}
                for i in range(1, count) if states[i] != states[i - 1]
            ],
        }

    monkeypatch.setattr(export_stats, '_load', synthetic_run)
    monkeypatch.setattr(aggregate_robustness, '_load', synthetic_run)
    monkeypatch.setattr(export_stats, 'OUT', tmp_path / 'stats.tex')
    monkeypatch.setattr(aggregate_robustness, 'OUT', tmp_path / 'robustness.tex')
    export_stats.main(104)
    aggregate_robustness.main()
    stats = export_stats.OUT.read_text()
    table = aggregate_robustness.OUT.read_text()
    assert rf'\newcommand{{\StatsModelVersion}}{{{MODEL_VERSION}}}' in stats
    assert rf'\def\RobustnessModelVersion{{{MODEL_VERSION}}}' in table
    assert 'same sign as the median' in table
    defined = set(re.findall(r'\\newcommand\{\\(\w+)\}', stats))
    root = Path(__file__).resolve().parents[1]
    # Manuscripts are local, ignored files; export checks above also run in clones.
    if not (root / 'noc-oup').is_dir():
        return
    for filename in ('main.tex', 'supplementary.tex'):
        manuscript = (root / 'noc-oup' / filename).read_text(encoding='utf-8')
        assert rf'\def\ExpectedModelVersion{{{MODEL_VERSION}}}' in manuscript
        used = set(re.findall(r'\\((?:Expert|Novice|Stats|PCA)\w+)', manuscript))
        assert used <= defined, sorted(used - defined)
