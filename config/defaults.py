"""Shared model vocabulary and numerical constants."""

from math import log, tanh

MODEL_VERSION = 'clarity-v5'

STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_attention']
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN', 'VIS', 'SOM', 'LIM']
THOUGHTSEEDS = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']

THETA_BASE = {
    'breath_focus': {('DMN', 'DAN'): 0.40, ('DAN', 'DMN'): 0.60, ('DAN', 'FPN'): 0.15, ('FPN', 'DAN'): 0.15, ('SOM', 'DAN'): 0.20, ('DAN', 'SOM'): 0.20, ('SOM', 'FPN'): 0.15, ('FPN', 'SOM'): 0.15},
    'mind_wandering': {('DMN', 'VAN'): -0.30, ('VAN', 'DMN'): -0.30, ('DMN', 'FPN'): -0.15, ('FPN', 'DMN'): -0.15, ('LIM', 'DMN'): 0.30, ('DMN', 'LIM'): 0.30, ('LIM', 'VAN'): -0.20, ('VAN', 'LIM'): -0.20},
    'meta_awareness': {('VAN', 'FPN'): 0.50, ('FPN', 'VAN'): 0.50, ('DMN', 'DAN'): -0.25, ('DAN', 'DMN'): -0.25, ('DMN', 'FPN'): -0.25, ('FPN', 'DMN'): -0.25, ('LIM', 'VAN'): 0.20, ('VAN', 'LIM'): 0.20},
    'redirect_attention': {('DMN', 'DAN'): -0.40, ('DAN', 'DMN'): -0.40, ('DMN', 'FPN'): -0.20, ('FPN', 'DMN'): -0.40, ('DAN', 'FPN'): 0.40, ('FPN', 'DAN'): 0.40, ('SOM', 'DAN'): 0.25, ('DAN', 'SOM'): 0.25, ('SOM', 'FPN'): 0.15, ('FPN', 'SOM'): 0.15},
}

THOUGHTSEED_STATE_PRIORS = {
    'breath_focus': {'attend_breath': 0.85, 'equanimity': 0.45, 'pain_discomfort': 0.20, 'pending_tasks': 0.05, 'aha_moment': 0.15},
    'mind_wandering': {'attend_breath': 0.15, 'equanimity': 0.10, 'pain_discomfort': 0.65, 'pending_tasks': 0.60, 'aha_moment': 0.15},
    'meta_awareness': {'attend_breath': 0.25, 'equanimity': 0.35, 'pain_discomfort': 0.45, 'pending_tasks': 0.45, 'aha_moment': 0.85},
    'redirect_attention': {'attend_breath': 0.70, 'equanimity': 0.85, 'pain_discomfort': 0.15, 'pending_tasks': 0.15, 'aha_moment': 0.25},
}

DEFAULT_DT = 0.2
EVALUATION_STEPS = 4000
PLOT_STEPS = 2000
CLIP_MIN = 0.05
CLIP_MAX = 0.9
EPS = 1e-6
NOISE_LEVEL = 0.002
GRAD_CLIP = 5.0
STATE_BELIEF_VAR = 0.1
LATENT_TAU = 1.0

_rho = 1.0 - DEFAULT_DT / max(LATENT_TAU, DEFAULT_DT)
P_INV = 1.0 / ((NOISE_LEVEL * DEFAULT_DT) / max(1.0 - _rho ** 2, EPS))

# L3 meta-awareness m_t (System 2): a slow, graded signal for monitoring
# clarity -- how clearly the agent can tell, from what is currently broadcast,
# that it is on task. It integrates the accessed content's diagnosticity-derived
# signal (below) with policy--habit discrepancy; it is not itself a workspace
# ignition.
META_TAU = 2.0        # slower than the L2 latent (LATENT_TAU)
META_THRESHOLD = 0.5  # control-engagement cost: prior evidence alone leaves m < 0.5
# Fixed point of the monitor when the slot is empty and evidence matches habit:
# no content is no evidence, so m relaxes to sigmoid(-b_m). Resting clarity, and
# therefore the neutral reference for top-down bias on workspace access.
META_RESTING = 1.0 / (1.0 + 2.718281828459045 ** META_THRESHOLD)
assert META_TAU > LATENT_TAU, (
    "META_TAU must exceed LATENT_TAU: System 2 monitoring is slower than "
    "System 1 thoughtseed dynamics."
)

# Global workspace access (GNW): a thresholded single-slot competition among
# thoughtseeds. One step (DEFAULT_DT = 0.2 s) resolves workspace access;
# an access event is an entry into the slot, while sustained access is repeated
# re-selection of the incumbent (reverberation).
ACCESS_THRESHOLD = 0.6
ACCESS_REVERBERATION = 0.25
# At the Breath-Focus baseline, this exceeds the steady incumbent support
# (thoughtseed evidence + access prior + resting m_t + reverberation) above
# threshold, so a continuously occupied slot can turn over after adaptation.
ACCESS_ADAPTATION_WEIGHT = 1.15
ACCESS_ADAPTATION_TAU = 5.0
ACCESS_PRIOR_CONCENTRATION = 20.0   # uniform Dirichlet pseudo-counts per thoughtseed
ACCESS_CREDIT_HOLD_STEPS = int(round(LATENT_TAU / DEFAULT_DT))  # one latent time constant
PRACTICE_OBJECT = 'attend_breath'
OFF_TASK_STATE = 'mind_wandering'
assert ACCESS_ADAPTATION_TAU > DEFAULT_DT and ACCESS_CREDIT_HOLD_STEPS >= 1

EXECUTION_FIDELITY = 0.7
DESCENDING_PREDICTION_GAIN = 0.25
assert 0.0 <= DESCENDING_PREDICTION_GAIN < 1.0, (
    "DESCENDING_PREDICTION_GAIN must stay in [0, 1): the decoder-derived "
    "descending prediction is structurally bounded to bias, not replace, "
    "L1's own state-conditioned attractor (see supplementary Eq. mu_blend)."
)

# Accessed content is broadcast all-or-none: while a thoughtseed holds the
# workspace, its diagnosticity d_i(s) stabilizes the current regime through the
# L2-formed exit hazard. Completed MA detection releases that stabilization in MA.


def _thoughtseed_diagnosticity() -> dict:
    """d_i(s) = mu_z,i(s) / sum_s' mu_z,i(s').

    How diagnostic thoughtseed i is of regime s, obtained by column-normalizing
    THOUGHTSEED_STATE_PRIORS. The broadcast readout is therefore derived from a
    table the model already commits to, and adds no free parameters of its own.
    """
    return {
        thoughtseed: {
            state: THOUGHTSEED_STATE_PRIORS[state][thoughtseed] / total
            for state in STATES
        }
        for thoughtseed in THOUGHTSEEDS
        if (total := sum(THOUGHTSEED_STATE_PRIORS[s][thoughtseed] for s in STATES))
    }


THOUGHTSEED_DIAGNOSTICITY = _thoughtseed_diagnosticity()

# The regime in which a lapse is recognized, and the content whose broadcast
# constitutes that recognition: the thoughtseed most diagnostic of that regime.
# Derived from the same table, so the detection content is not named by hand.
DETECTION_STATE = 'meta_awareness'
DETECTION_CONTENT = max(
    THOUGHTSEED_DIAGNOSTICITY,
    key=lambda thoughtseed: THOUGHTSEED_DIAGNOSTICITY[thoughtseed][DETECTION_STATE],
)


def _on_task_posterior() -> dict:
    """P(on task | o) = 1 - d_o(MW).

    Mind Wandering is the model's only off-task regime, so the diagnosticity
    table already answers how much the broadcast content testifies that the
    agent is still on task. No content is listed by hand and no parameter is
    added.
    """
    return {
        thoughtseed: 1.0 - row[OFF_TASK_STATE]
        for thoughtseed, row in THOUGHTSEED_DIAGNOSTICITY.items()
    }


def _on_task_log_evidence() -> dict:
    """c(o) = ln [P(on task | o)/P(off task | o)] - ln [P(on task)/P(off task)].

    A diagnosticity-derived log-odds contrast relative to uniform regime weights,
    not a calibrated likelihood of workspace observations. Positive for content
    diagnostic of regulated regimes and negative for Mind Wandering content.
    """
    prior_log_odds = log(len(STATES) - 1)
    return {
        thoughtseed: log(on_task / (1.0 - on_task)) - prior_log_odds
        for thoughtseed, on_task in ON_TASK_POSTERIOR.items()
    }


ON_TASK_POSTERIOR = _on_task_posterior()
ON_TASK_LOG_EVIDENCE = _on_task_log_evidence()

# The same evidence expressed as a contrast in [-1, 1], for biasing workspace
# access: tanh(c/2) is the odds-to-contrast map, so this is c(o) referenced to
# the same prior rather than a second quantity with its own reference point.
# Referencing it to one half instead would make a content's sign depend on
# whether it happens to exceed 50% posterior, which breaks as soon as MW and MA
# share content.
ON_TASK_CONTRAST = {
    thoughtseed: tanh(0.5 * evidence)
    for thoughtseed, evidence in ON_TASK_LOG_EVIDENCE.items()
}

# Reporting grouping only: the contents whose most diagnostic regime is the
# off-task one. Derived from the same table rather than listed.
DISTRACTORS = tuple(
    thoughtseed for thoughtseed, row in THOUGHTSEED_DIAGNOSTICITY.items()
    if max(row, key=row.get) == OFF_TASK_STATE
)
assert DISTRACTORS and PRACTICE_OBJECT not in DISTRACTORS


def get_policy_candidate_order(current_state: str) -> list[str]:
    return [current_state, *[state for state in STATES if state != current_state]]
