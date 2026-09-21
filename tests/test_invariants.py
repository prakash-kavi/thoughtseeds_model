import numpy as np
import pytest
import torch

from config.defaults import (
    ACCESS_CREDIT_HOLD_STEPS, ACCESS_THRESHOLD, CLIP_MAX, CLIP_MIN, DESCENDING_PREDICTION_GAIN,
    DEFAULT_DT, DISTRACTORS, EPS, LATENT_TAU, META_RESTING, META_TAU, META_THRESHOLD, NETWORKS, ON_TASK_CONTRAST,
    PRACTICE_OBJECT, STATES, THOUGHTSEEDS, get_policy_candidate_order,
)
from config.profiles import (
    ATTRACTOR_EXPRESSING_STATES, ATTRACTOR_SETTLING_HORIZONS, DWELL_MEAN_SECONDS, L1_SUBSTEPS,
    STATE_TRANSITION_PROBS, THETA_SETTLING_MARGIN, THETA_STABILITY_LIMIT, TIMESCALE_SEPARATION,
)
from model import duration
from model.markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from model.phenotype import EXPERT_PHENOTYPE, NOVICE_PHENOTYPE
from model.habits import AccessPrior, HabitPrior
from model.policy import PolicyEngine
from model.substrate import Layer1Process
from model.thoughtseeds import Layer2Agent
from model.training_loop import MeditationTrainer, descending_prediction_weight
from model.workspace import GlobalWorkspace
from utils.math_utils import clip_probability, policy_posterior, softmax
from viz.ignition import access_onsets, access_runs, access_summary


def test_softmax_and_policy_posterior_are_normalized():
    assert np.isclose(softmax(np.array([1e10, -1e10, 0.0])).sum(), 1.0)
    posterior = policy_posterior(np.log(np.array([0.4, 0.3, 0.2, 0.1])), np.array([0.5, 1.2, 0.3, 0.8]), 0.7)
    assert np.isclose(posterior.sum(), 1.0)
    assert np.all(posterior >= 0.0)


def test_closed_form_correction_has_finite_vfe():
    agent = Layer2Agent(EXPERT_PHENOTYPE)
    x = torch.full((len(NETWORKS),), 0.5)
    z_prior = torch.full((len(THOUGHTSEEDS),), 0.5)
    z_star = agent.thoughtseed_model.correct(x, z_prior)
    vfe = agent.compute_vfe(z_star, x, z_prior)
    assert torch.isfinite(vfe)
    assert vfe.item() >= 0.0


def test_state_belief_is_a_distribution():
    agent = Layer2Agent(EXPERT_PHENOTYPE)
    belief = agent.infer_state_belief(torch.full((len(THOUGHTSEEDS),), 0.5))
    assert set(belief) == set(STATES)
    assert np.isclose(sum(belief.values()), 1.0)


def test_duration_hazard_is_a_valid_increasing_exit_probability():
    hazards = [duration.hazard('expert', 'breath_focus', age) for age in range(1, 250)]
    assert all(0.0 <= value <= 1.0 for value in hazards)
    assert hazards[0] < hazards[-1]


def test_median_steps_is_the_50th_percentile_of_the_dwell_law():
    median = duration.median_steps('expert', 'breath_focus')
    assert median > 0
    # median is the smallest age at which cumulative probability reaches 0.5, so
    # P(dwell >= median) must clear 0.5 while P(dwell >= median + 1) must not.
    assert duration.survival_probability('expert', 'breath_focus', median) > 0.5
    assert duration.survival_probability('expert', 'breath_focus', median + 1) <= 0.5


def test_survival_probability_is_monotonically_decreasing_and_bounded():
    probs = [duration.survival_probability('novice', 'mind_wandering', d) for d in range(1, 200, 10)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))


def test_access_runs_and_onsets_ignore_the_empty_slot():
    history = [None, 'attend_breath', 'attend_breath', None, 'pending_tasks', 'attend_breath']
    assert access_runs(history) == [('attend_breath', 1, 3), ('pending_tasks', 4, 5), ('attend_breath', 5, 6)]
    assert access_onsets(history) == [1, 4, 5]


def test_exit_kernel_requires_complete_policy_distribution():
    process = Layer1Process(EXPERT_PHENOTYPE, seed=0)
    policy = {state: 1.0 / len(STATES) for state in STATES}
    kernel = process._exit_destination_kernel(policy)
    assert np.isclose(sum(kernel.values()), 1.0)
    assert process.current_state not in kernel
    with pytest.raises(ValueError):
        process._exit_destination_kernel({'breath_focus': 1.0})


def test_gamma_hazard_can_hold_an_early_state():
    process = Layer1Process(EXPERT_PHENOTYPE, seed=0)
    process.current_state = 'breath_focus'
    process.current_dwell = 0
    policy = {state: float(state == 'mind_wandering') for state in STATES}
    assert process._check_transition(policy) in STATES


def test_blankets_enforce_declared_interfaces():
    l1l2 = MarkovBlanketL1L2()
    l1l2.update_sensory_states({**{network: 0.5 for network in NETWORKS}, 'dwell_age': 2.0})
    with pytest.raises(KeyError):
        l1l2.update_sensory_states({'unknown': 0.0})
    l2l3 = MarkovBlanketL2L3()
    # L3 returns its selection through the blanket: the policy posterior, the
    # accessed content, and meta-awareness are its only active states. L1 never
    # receives any of them directly; L2 transforms them into control signals.
    assert l2l3.allowed_active == {'policy_posterior', 'accessed_content', 'meta_awareness'}
    assert {'habit_log_prior', 'access_prior'} <= l2l3.allowed_sensory
    assert 'meta_awareness' not in l1l2.allowed_active
    with pytest.raises(KeyError):
        l2l3.update_active_states({'hazard_modulation': 0.5})


def test_diagnosticity_rows_normalize_and_match_the_canonical_cycle():
    from config.defaults import THOUGHTSEED_DIAGNOSTICITY
    expected = {
        'attend_breath': 'breath_focus',
        'pain_discomfort': 'mind_wandering',
        'pending_tasks': 'mind_wandering',
        'aha_moment': 'meta_awareness',
        'equanimity': 'redirect_attention',
    }
    for thoughtseed, row in THOUGHTSEED_DIAGNOSTICITY.items():
        assert np.isclose(sum(row.values()), 1.0)
        assert max(row, key=row.get) == expected[thoughtseed]


def _uniform_prior():
    return np.full(len(THOUGHTSEEDS), 1.0 / len(THOUGHTSEEDS))


def test_access_requires_prediction_error_to_break_in():
    workspace = GlobalWorkspace()
    z_prior = np.array([0.10, 0.55, 0.30, 0.40, 0.20])
    # attend_breath is revised up the most, but pain_discomfort has the highest
    # drive among the contents revised upward at all.
    z_star = np.array([0.35, 0.60, 0.25, 0.35, 0.22])
    assert workspace.select_access(z_star, z_prior, _uniform_prior()) == 'pain_discomfort'
    # With no incumbent and no upward revision, the slot stays empty.
    fresh = GlobalWorkspace()
    assert fresh.select_access(z_prior - 0.01, z_prior, _uniform_prior()) is None


def test_incumbent_access_is_self_sustaining_with_hysteresis():
    workspace = GlobalWorkspace()
    prior = _uniform_prior()
    z_prior = np.full(len(THOUGHTSEEDS), 0.5)
    first = np.array([0.80, 0.70, 0.05, 0.05, 0.05])
    assert workspace.select_access(first, z_prior, prior) == 'attend_breath'
    # A slightly stronger, surprising challenger does not displace the incumbent,
    # and the incumbent needs no prediction error of its own to stay.
    challenger = np.array([0.55, 0.75, 0.05, 0.05, 0.05])
    assert workspace.select_access(challenger, z_prior, prior) == 'attend_breath'
    # A fresh workspace with the same inputs gives the slot to the challenger.
    assert GlobalWorkspace().select_access(challenger, z_prior, prior) == 'pain_discomfort'


def test_adaptation_eventually_releases_the_incumbent():
    workspace = GlobalWorkspace()
    prior = _uniform_prior()
    z_prior = np.full(len(THOUGHTSEEDS), 0.5)
    z_star = np.array([0.45, 0.60, 0.05, 0.05, 0.05])
    workspace.accessed_content = 'attend_breath'
    winners = [workspace.select_access(z_star, z_prior, prior) for _ in range(200)]
    assert winners[0] == 'attend_breath'
    assert 'pain_discomfort' in winners


def test_adaptation_turns_over_a_typical_sustained_breath_broadcast():
    workspace = GlobalWorkspace()
    # The calibrated adaptation weight must overcome the ordinary steady
    # Breath-Focus drive (corrected breath evidence, access prior, resting m,
    # and reverberation), rather than merely allowing a challenger to win.
    workspace.accessed_content = PRACTICE_OBJECT
    workspace.meta_awareness = 0.75
    z_prior = np.full(len(THOUGHTSEEDS), 0.5)
    z_star = np.array([0.85, 0.05, 0.05, 0.05, 0.05])
    access_prior = np.full(len(THOUGHTSEEDS), 0.20)
    winners = [workspace.select_access(z_star, z_prior, access_prior) for _ in range(150)]
    assert winners[0] == PRACTICE_OBJECT
    assert None in winners


def test_monitoring_clarity_amplifies_on_task_content_in_proportion_to_its_evidence():
    prior = _uniform_prior()
    z_star = np.array([0.30, 0.40, 0.05, 0.05, 0.05])
    resting, aware, lapsed = GlobalWorkspace(), GlobalWorkspace(), GlobalWorkspace()
    aware.meta_awareness, lapsed.meta_awareness = 1.0, 0.05
    breath, aha, pending = (THOUGHTSEEDS.index(n) for n in (PRACTICE_OBJECT, 'aha_moment', 'pending_tasks'))
    # Resting clarity is the neutral reference: it applies no bias at all.
    assert np.isclose(resting.meta_awareness, META_RESTING)
    assert np.allclose(resting.access_drive(z_star, prior),
                       np.asarray(z_star) + np.asarray(prior))
    gain = aware.access_drive(z_star, prior) - resting.access_drive(z_star, prior)
    # Clarity favours on-task content, graded by its own on-task contrast, so
    # the detection content gains with the practice object rather than losing to
    # it. The old rule amplified only the practice object, which made a clearer
    # monitor slower to notice a lapse.
    assert gain[breath] > 0.0 and gain[aha] > 0.0 > gain[pending]
    assert np.isclose(gain[breath] / gain[aha], ON_TASK_CONTRAST[PRACTICE_OBJECT] / ON_TASK_CONTRAST['aha_moment'])
    # A lapse reverses the bias without suppressing access as such: distractor
    # content is favoured while on-task content fades.
    fade = lapsed.access_drive(z_star, prior) - resting.access_drive(z_star, prior)
    assert fade[breath] < 0.0 < fade[pending]


def test_hazard_modulation_is_formed_by_l2_from_accessed_content():
    agent = Layer2Agent(EXPERT_PHENOTYPE)
    candidates = get_policy_candidate_order('mind_wandering')
    mu_candidates = [agent.state_attractor(s) for s in candidates]
    posterior = {state: 1.0 / len(candidates) for state in candidates}

    def modulation(m, content, state='mind_wandering'):
        active = {'policy_posterior': posterior, 'meta_awareness': m, 'accessed_content': content}
        return agent.descending_control(active, state, candidates, mu_candidates)['hazard_modulation']

    # Access broadcasts all-or-none, independently of meta-awareness. Content
    # that is more diagnostic of the ongoing regime stabilizes it more strongly.
    assert modulation(1.0, None) == 0.0
    assert modulation(0.0, 'aha_moment') == modulation(1.0, 'aha_moment') < 0.0
    assert modulation(0.0, 'pending_tasks') < modulation(0.0, 'attend_breath') < 0.0
    assert modulation(0.0, 'attend_breath', 'breath_focus') < modulation(0.0, 'attend_breath')


def test_dwell_prior_uses_the_next_l1_age_and_current_broadcast_control():
    agent = Layer2Agent(EXPERT_PHENOTYPE)
    state = 'breath_focus'
    candidates = get_policy_candidate_order(state)
    agent.blanket_l1l2.update_sensory_states({
        **{network: 0.5 for network in NETWORKS},
        'dwell_age': 10.0,
    })
    modulation = agent.broadcast_hazard_modulation('attend_breath', state)
    priors = agent.dwell_priors_for_next_l1_update(state, candidates, modulation)
    expected_hazard = clip_probability(
        duration.hazard('expert', state, 11.0) * (1.0 + modulation)
    )
    assert np.isclose(priors[0], max(EPS, 1.0 - expected_hazard))
    for index, destination in enumerate(candidates[1:], start=1):
        assert np.isclose(
            priors[index],
            max(EPS, expected_hazard * STATE_TRANSITION_PROBS['expert'][state][destination]),
        )


def test_policy_posterior_uses_only_dwell_habit_and_evidence_terms():
    engine = PolicyEngine()
    candidates = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_attention']
    dwell_priors = [0.4, 0.3, 0.2, 0.1]
    g_vals = [0.1, 0.2, 0.3, 0.4]
    habit = np.zeros(len(candidates))
    for precision in (0.0, 1.0):
        posterior = engine.select_policy(g_vals, dwell_priors, candidates, habit, precision)
        assert np.isclose(posterior.sum(), 1.0)
        assert np.all(posterior > 0.0)


def test_policy_habit_update_increases_intended_switch_prior():
    habits = HabitPrior()
    source = 'mind_wandering'
    destination = 'redirect_attention'
    belief = {state: float(state == source) for state in STATES}
    habits.update(belief, {source: {source: 0.75, destination: 0.25}})
    candidates = get_policy_candidate_order(source)
    log_prior = habits.log_prior(belief, candidates)
    assert candidates[np.argmax(log_prior)] == destination


def test_workspace_returns_unit_interval_meta_awareness():
    workspace = GlobalWorkspace()
    values = [workspace.update_meta_awareness([0.1, 0.5, 0.3, 0.2], np.log(np.full(4, 0.25)))
              for _ in range(50)]
    assert all(0.0 <= value <= 1.0 for value in values)


def test_meta_awareness_is_slow_graded_and_driven_by_accessed_content():
    habit = np.log(np.full(4, 0.25))
    costs = [0.1, 0.1, 0.1, 0.1]
    noticing, absorbed = GlobalWorkspace(), GlobalWorkspace()
    noticing.accessed_content, absorbed.accessed_content = 'aha_moment', 'pending_tasks'
    first = noticing.update_meta_awareness(costs, habit)
    # Graded: one step moves m only part of the way toward its target.
    assert 0.0 < first < 0.5
    for _ in range(30):
        noticing.update_meta_awareness(costs, habit)
        absorbed.update_meta_awareness(costs, habit)
    assert noticing.meta_awareness > absorbed.meta_awareness
    assert META_TAU > LATENT_TAU


def test_on_task_evidence_is_derived_from_the_diagnosticity_table():
    from config.defaults import OFF_TASK_STATE, THOUGHTSEED_DIAGNOSTICITY
    # The contrast follows configured diagnosticity against uniform regime
    # weights; an empty slot contributes zero.
    assert GlobalWorkspace.on_task_evidence(None) == 0.0
    for thoughtseed, row in THOUGHTSEED_DIAGNOSTICITY.items():
        evidence = GlobalWorkspace.on_task_evidence(thoughtseed)
        assert (evidence > 0.0) == (row[OFF_TASK_STATE] < 1.0 / len(STATES))
    assert GlobalWorkspace.on_task_evidence(PRACTICE_OBJECT) > 0.0
    for distractor in DISTRACTORS:
        assert GlobalWorkspace.on_task_evidence(distractor) < 0.0


def test_the_practice_object_drives_the_monitor_it_no_longer_sits_at_zero():
    # The hand-assigned content signal left attend_breath neutral, pinning BF
    # meta-awareness at the same value for both phenotypes.
    habit = np.log(np.full(4, 0.25))
    costs = [0.1, 0.1, 0.1, 0.1]
    breathing, empty = GlobalWorkspace(), GlobalWorkspace()
    breathing.accessed_content = PRACTICE_OBJECT
    for _ in range(60):
        breathing.update_meta_awareness(costs, habit)
        empty.update_meta_awareness(costs, habit)
    assert breathing.meta_awareness > empty.meta_awareness
    assert breathing.meta_awareness > 0.5 > empty.meta_awareness


def test_every_regime_can_express_its_attractor_within_a_typical_dwell():
    # The clamp's diagonal margin is the slowest relaxation rate L1 may keep, so
    # 3*tau_slow must fit inside the dwell of every attractor-expressing regime.
    for phenotype in (NOVICE_PHENOTYPE, EXPERT_PHENOTYPE):
        process = Layer1Process(phenotype, seed=0)
        for state in STATES:
            theta = process._clamp_theta(process._get_coupling(state)).detach().numpy()
            tau_slow = 1.0 / np.real(np.linalg.eigvals(theta)).min()
            settling = ATTRACTOR_SETTLING_HORIZONS * tau_slow
            assert settling <= min(
                DWELL_MEAN_SECONDS[level][attractor_state]
                for level in DWELL_MEAN_SECONDS
                for attractor_state in ATTRACTOR_EXPRESSING_STATES
            ) + 1e-9


def test_layer_timescales_are_ordered_from_fast_access_to_slow_learning():
    from config.defaults import (
        ACCESS_ADAPTATION_TAU, DEFAULT_DT, LATENT_TAU, META_TAU,
    )
    # L1 must run faster than the latent it drives, not straddle it.
    slowest = 0.0
    for phenotype in (NOVICE_PHENOTYPE, EXPERT_PHENOTYPE):
        process = Layer1Process(phenotype, seed=0)
        for state in STATES:
            theta = process._clamp_theta(process._get_coupling(state)).detach().numpy()
            rates = np.real(np.linalg.eigvals(theta))
            slowest = max(slowest, 1.0 / rates.min())
            assert np.abs(theta).sum(axis=1).max() < THETA_STABILITY_LIMIT
            update = np.eye(len(NETWORKS)) - DEFAULT_DT / L1_SUBSTEPS * theta
            assert max(abs(np.linalg.eigvals(update))) < 1.0
    assert slowest <= LATENT_TAU / TIMESCALE_SEPARATION + 1e-9
    # The layer relaxation timescales are strictly ordered, fastest first.
    layers = [
        DEFAULT_DT,     # L3 workspace access resolves in one step
        slowest,        # L1 slowest network mode
        LATENT_TAU,     # L2 thoughtseed latent
        META_TAU,       # L3 monitor
    ]
    assert layers == sorted(layers) and len(set(layers)) == len(layers), layers
    # Everything that is not a layer relaxation -- content adaptation, the
    # exogenous regime dwells, and learning -- is slower than all of them. These
    # are not ordered against each other: adaptation (5 s) outlasts the shortest
    # dwell (expert RA, 3 s), and neither is above the other in the hierarchy.
    slower_processes = [
        ACCESS_ADAPTATION_TAU,
        min(DWELL_MEAN_SECONDS[level][state]
            for level in DWELL_MEAN_SECONDS for state in STATES),
    ]
    assert min(slower_processes) > max(layers)


def test_configured_regulatory_dwell_ordering():
    # MA is the interval in which detection develops and RA executes a committed
    # policy, so MA is the longer regime, and both phenotypes agree on that.
    for level, dwells in DWELL_MEAN_SECONDS.items():
        assert dwells['meta_awareness'] > dwells['redirect_attention']
        assert dwells['breath_focus'] > dwells['meta_awareness']
        assert dwells['mind_wandering'] > dwells['meta_awareness']
    assert (DWELL_MEAN_SECONDS['novice']['redirect_attention']
            > DWELL_MEAN_SECONDS['expert']['redirect_attention'])


@pytest.mark.parametrize('content', [None] + THOUGHTSEEDS)
def test_monitor_matches_the_stated_update(content):
    habit = np.log(np.full(4, 0.25))
    costs = [0.0, 2.0, 2.0, 2.0]
    q_evid = np.exp(-np.array(costs))
    q_evid /= q_evid.sum()
    discrepancy = float(np.sum(q_evid * np.log(q_evid / np.full(4, 0.25))))
    workspace = GlobalWorkspace()
    workspace.accessed_content = content
    workspace.meta_awareness = 0.7
    drive = workspace.on_task_evidence(content) + discrepancy - META_THRESHOLD
    target = 1.0 / (1.0 + np.exp(-drive))
    expected = 0.7 + DEFAULT_DT / META_TAU * (target - 0.7)
    assert workspace.update_meta_awareness(costs, habit) == pytest.approx(expected)


def test_empty_workspace_relaxes_to_shared_resting_monitor():
    # Matched evidence and habit leave no discrepancy, so an empty slot relaxes
    # to sigmoid(-b_m) from either side.
    habit = np.log(np.full(4, 0.25))
    costs = [0.1, 0.1, 0.1, 0.1]
    for initial in (0.0, META_RESTING, 1.0):
        workspace = GlobalWorkspace()
        workspace.meta_awareness = initial
        for _ in range(200):
            workspace.update_meta_awareness(costs, habit)
        assert workspace.meta_awareness == pytest.approx(META_RESTING, abs=1e-8)


def test_policy_habit_discrepancy_is_non_negative_and_raises_the_monitor():
    habit = np.log(np.full(4, 0.25))
    matched, conflicted = GlobalWorkspace(), GlobalWorkspace()
    for workspace in (matched, conflicted):
        workspace.accessed_content = PRACTICE_OBJECT
    matched.update_meta_awareness([0.25, 0.25, 0.25, 0.25], habit)
    conflicted.update_meta_awareness([0.0, 2.0, 2.0, 2.0], habit)
    assert conflicted.meta_awareness > matched.meta_awareness


def test_threshold_crossing_under_distraction_would_need_discrepancy_far_above_observed():
    # A configuration-level bound, not an enforced runtime property: given the
    # c table and b_m, a discrepancy of 0.30 -- above the largest observed in
    # seeds 100-104 (0.269) -- still leaves the distractor target below
    # threshold. Crossing would need roughly 1.39 (pain) or 1.52 (pending
    # tasks). This does not constrain what discrepancies a simulation reaches.
    observed_max_discrepancy = 0.30
    for content in DISTRACTORS:
        drive = (GlobalWorkspace.on_task_evidence(content)
                 + observed_max_discrepancy - META_THRESHOLD)
        assert 1.0 / (1.0 + np.exp(-drive)) < META_THRESHOLD


def test_theta_guard_rejects_unstable_coupling_that_passed_diagonal_check():
    process = Layer1Process(EXPERT_PHENOTYPE)
    theta = torch.zeros((len(NETWORKS), len(NETWORKS)))
    theta[0, 1] = theta[1, 0] = 10.0
    # Constructed diagonals are 12 (<20), but an eigenvalue is 22: Euler is unstable.
    with pytest.raises(RuntimeError, match='Gershgorin row bound'):
        process._clamp_theta(theta)


def test_access_prior_credits_only_held_practice_reignition_after_meta_awareness():
    prior = AccessPrior()
    start = prior.mean()[THOUGHTSEEDS.index(PRACTICE_OBJECT)]
    # A brief return that does not hold earns nothing.
    prior.observe('pending_tasks', PRACTICE_OBJECT, 0.9)
    prior.observe(PRACTICE_OBJECT, 'pending_tasks', 0.9)
    assert prior.mean()[THOUGHTSEEDS.index(PRACTICE_OBJECT)] == start
    # Resting meta-awareness is below the meaningful-monitoring threshold and
    # therefore earns no access credit even when the return is sustained.
    resting = AccessPrior()
    resting_start = resting.mean()[THOUGHTSEEDS.index(PRACTICE_OBJECT)]
    resting.observe('pending_tasks', PRACTICE_OBJECT, 0.38)
    for _ in range(ACCESS_CREDIT_HOLD_STEPS - 1):
        resting.observe(PRACTICE_OBJECT, PRACTICE_OBJECT, 0.38)
    assert resting.mean()[THOUGHTSEEDS.index(PRACTICE_OBJECT)] == resting_start
    # A re-ignition that holds for the hold window is credited by m minus the
    # meaningful-monitoring threshold at onset.
    committed = [prior.observe('pending_tasks', PRACTICE_OBJECT, 0.9)]
    committed += [prior.observe(PRACTICE_OBJECT, PRACTICE_OBJECT, 0.0) for _ in range(ACCESS_CREDIT_HOLD_STEPS - 1)]
    assert committed[-1] and not any(committed[:-1])
    credited = prior.mean()[THOUGHTSEEDS.index(PRACTICE_OBJECT)]
    assert credited > start
    # Holding longer adds no further credit, and orchestrator access earns none.
    prior.observe(PRACTICE_OBJECT, PRACTICE_OBJECT, 1.0)
    prior.observe(None, 'aha_moment', 1.0)
    assert prior.mean()[THOUGHTSEEDS.index(PRACTICE_OBJECT)] == credited
    assert np.isclose(prior.mean().sum(), 1.0)


def test_descending_prediction_gain_is_bounded_below_one():
    # config/defaults.py asserts this at import time; re-check it here so the
    # invariant is verified by the test suite, not only by a load-time assert.
    assert 0.0 <= DESCENDING_PREDICTION_GAIN < 1.0


def test_descending_prediction_weight_never_lets_the_decoder_fully_replace_l1():
    # Full ignition (m_t=1) still leaves the decoder-derived prediction capped
    # at DESCENDING_PREDICTION_GAIN: L1's own attractor always keeps a majority
    # weight, the code-level form of "biasing, not dictating" (Eq. mu_blend).
    assert descending_prediction_weight(0.0) == 0.0
    assert descending_prediction_weight(1.0) == DESCENDING_PREDICTION_GAIN
    assert descending_prediction_weight(1.0) < 1.0
    # Defensive clip holds even if an out-of-range meta_awareness ever reached it.
    assert descending_prediction_weight(5.0) == pytest.approx(min(DESCENDING_PREDICTION_GAIN * 5.0, 1.0))


def test_theta_coupling_is_diagonally_dominant_and_mean_reverting():
    # supplementary S1.2 invokes Gershgorin: diagonal dominance with a positive
    # diagonal is the sufficient condition for -Theta(s) to be mean-reverting.
    # _clamp_theta now raises if this ever breaks; re-verify it here across
    # every phenotype x state combination, plus the eigenvalue claim itself.
    for phenotype in (EXPERT_PHENOTYPE, NOVICE_PHENOTYPE):
        process = Layer1Process(phenotype=phenotype, seed=0)
        for state in STATES:
            theta = process._clamp_theta(process._get_coupling(state)).numpy()
            diag = np.diag(theta)
            off_diag_sum = np.sum(np.abs(theta), axis=1) - np.abs(diag)
            assert np.all(diag > off_diag_sum)
            assert np.all(diag > 0.0)
            assert np.all(np.linalg.eigvals(theta).real > 0.0)


def test_state_transition_probs_rows_are_distributions():
    # config/profiles.py asserts this at import time; re-check here so a
    # future edit to the table is caught by the test suite too.
    for level, rows in STATE_TRANSITION_PROBS.items():
        for state, row in rows.items():
            assert np.isclose(sum(row.values()), 1.0), (level, state)


def test_training_runs_and_preserves_visualization_contract():
    result = MeditationTrainer(EXPERT_PHENOTYPE, seed=42).fit(timesteps=80)
    assert result['timesteps'] == 80
    for key in (
        'free_energy_history', 'forward_error_history', 'meta_awareness_history', 'accessed_content_history',
        'state_history',
        'network_activations_history', 'thoughtseed_activations_history', 'thoughtseed_prior_activations_history',
    ):
        assert len(result[key]) == 80
    assert np.all(np.isfinite(np.asarray(result['free_energy_history'])))
    assert np.all(np.isfinite(np.asarray(result['meta_awareness_history'])))
    assert all(CLIP_MIN <= value <= CLIP_MAX for step in result['thoughtseed_activations_history'] for value in step)
    assert set(result['accessed_content_history']) <= set(THOUGHTSEEDS) | {None}
    assert np.isclose(sum(result['access_prior']), 1.0)
    summary = access_summary(result, window=80)
    assert 0.0 <= summary['breath_access_percent'] <= 100.0


def test_theta_diagonal_is_exactly_the_off_diagonal_sum_plus_the_margin():
    # The diagonal is constructed from the couplings and the settling margin, not
    # configured. A configured base diagonal or per-phenotype diagonal boost would
    # be silently discarded, so assert the construction leaves no room for one.
    for phenotype in (EXPERT_PHENOTYPE, NOVICE_PHENOTYPE):
        process = Layer1Process(phenotype=phenotype, seed=0)
        for state in STATES:
            theta = process.theta_by_state[state]
            assert torch.equal(theta, process._clamp_theta(process._get_coupling(state)))
            off_diag = theta * (1.0 - torch.eye(len(NETWORKS)))
            expected = torch.sum(torch.abs(off_diag), dim=1) + THETA_SETTLING_MARGIN
            assert torch.equal(torch.diag(theta), expected)
            # the pre-diagonal template must carry no diagonal of its own
            assert torch.equal(
                torch.diag(process._get_coupling(state)), torch.zeros(len(NETWORKS))
            )


def test_learning_loss_and_map_objective_give_the_same_gradients():
    # The learning step uses reconstruction_error; compute_vfe adds the prior
    # deviation, which is constant in the parameters because z and the prior are
    # both detached. The two must therefore produce identical gradients.
    for phenotype in (EXPERT_PHENOTYPE, NOVICE_PHENOTYPE):
        agent = Layer2Agent(phenotype)
        x = torch.full((len(NETWORKS),), 0.5)
        z_prior = torch.full((len(THOUGHTSEEDS),), 0.4)
        z_star = agent.thoughtseed_model.correct(x, z_prior).detach()

        agent.zero_grad()
        agent.compute_vfe(z=z_star, observed_x=x, prior_target=z_prior).backward()
        from_vfe = {name: p.grad.clone() for name, p in agent.named_parameters() if p.grad is not None}

        agent.zero_grad()
        agent.reconstruction_error(z=z_star, observed_x=x).backward()
        from_recon = {name: p.grad.clone() for name, p in agent.named_parameters() if p.grad is not None}

        assert from_vfe and set(from_vfe) == set(from_recon)
        for name in from_vfe:
            assert torch.equal(from_vfe[name], from_recon[name])
