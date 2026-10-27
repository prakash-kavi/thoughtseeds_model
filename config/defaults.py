"""Shared model vocabulary and numerical constants."""

from math import log, tanh

MODEL_VERSION = 'clarity-v47'

STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_attention']
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN', 'VIS', 'SOM', 'LIM']
THOUGHTSEEDS = ['breath_sensation', 'pain_discomfort', 'pending_tasks', 'recognition_cue', 'intention_to_return']

THETA_BASE = {
    'breath_focus': {('DMN', 'DAN'): 0.40, ('DAN', 'DMN'): 0.60, ('DAN', 'FPN'): 0.15, ('FPN', 'DAN'): 0.15, ('SOM', 'DAN'): 0.20, ('DAN', 'SOM'): 0.20, ('SOM', 'FPN'): 0.15, ('FPN', 'SOM'): 0.15},
    'mind_wandering': {('DMN', 'VAN'): -0.30, ('VAN', 'DMN'): -0.30, ('DMN', 'FPN'): -0.15, ('FPN', 'DMN'): -0.15, ('LIM', 'DMN'): 0.30, ('DMN', 'LIM'): 0.30, ('LIM', 'VAN'): -0.20, ('VAN', 'LIM'): -0.20},
    'meta_awareness': {('VAN', 'FPN'): 0.50, ('FPN', 'VAN'): 0.50, ('DMN', 'DAN'): -0.25, ('DAN', 'DMN'): -0.25, ('DMN', 'FPN'): -0.25, ('FPN', 'DMN'): -0.25, ('LIM', 'VAN'): 0.20, ('VAN', 'LIM'): 0.20},
    'redirect_attention': {('DMN', 'DAN'): -0.40, ('DAN', 'DMN'): -0.40, ('DMN', 'FPN'): -0.20, ('FPN', 'DMN'): -0.40, ('DAN', 'FPN'): 0.40, ('FPN', 'DAN'): 0.40, ('SOM', 'DAN'): 0.25, ('DAN', 'SOM'): 0.25, ('SOM', 'FPN'): 0.15, ('FPN', 'SOM'): 0.15},
}

THOUGHTSEED_STATE_PRIORS = {
    'breath_focus': {'breath_sensation': 0.85, 'pain_discomfort': 0.20, 'pending_tasks': 0.05, 'recognition_cue': 0.15, 'intention_to_return': 0.45},
    'mind_wandering': {'breath_sensation': 0.15, 'pain_discomfort': 0.65, 'pending_tasks': 0.60, 'recognition_cue': 0.15, 'intention_to_return': 0.10},
    # MA shares MW's priors for the accessible contents, so distracting content can stay active; only the cue differs.
    'meta_awareness': {'breath_sensation': 0.15, 'pain_discomfort': 0.65, 'pending_tasks': 0.60, 'recognition_cue': 0.85, 'intention_to_return': 0.10},
    'redirect_attention': {'breath_sensation': 0.70, 'pain_discomfort': 0.15, 'pending_tasks': 0.15, 'recognition_cue': 0.25, 'intention_to_return': 0.85},
}

DEFAULT_DT = 0.2
EVALUATION_STEPS = 4000
PLOT_STEPS = 2000
CLIP_MIN = 0.05
CLIP_MAX = 0.9
EPS = 1e-6
NOISE_LEVEL = 0.002
GRAD_CLIP = 5.0
LEARNING_RATE = 0.01
LATENT_TAU = 1.0
assert LATENT_TAU > DEFAULT_DT, (
    "LATENT_TAU must exceed DEFAULT_DT: the latent OU update requires a positive time constant above one step."
)

_rho = 1.0 - DEFAULT_DT / LATENT_TAU
# 1/P_inf: no-evidence fixed point of the latent prior covariance, also its initial value.
P_INV = 1.0 / ((NOISE_LEVEL * DEFAULT_DT) / max(1.0 - _rho ** 2, EPS))

# L1's slowest mode is at least this factor faster than the L2 latent, and the L3 monitor this factor slower.
TIMESCALE_SEPARATION = 2.0

CLARITY_TAU = TIMESCALE_SEPARATION * LATENT_TAU
assert CLARITY_TAU > LATENT_TAU, (
    "CLARITY_TAU must exceed LATENT_TAU: the monitor is slower than the thoughtseed latent."
)

# Configured, not derived: low enough that distracting content can stay in the workspace during recognition.
ACCESS_THRESHOLD = 0.5
ACCESS_REVERBERATION = 0.25
# Saturated adaptation cancels the largest activation and the reverberation bonus.
ACCESS_ADAPTATION_WEIGHT = CLIP_MAX + ACCESS_REVERBERATION
ACCESS_ADAPTATION_TAU = 5.0
ACCESS_PRIOR_CONCENTRATION = 20.0  # initial Dirichlet count per thoughtseed
ACCESS_CREDIT_HOLD_STEPS = int(round(LATENT_TAU / DEFAULT_DT))
PRACTICE_OBJECT = 'breath_sensation'
OFF_TASK_STATE = 'mind_wandering'
assert ACCESS_ADAPTATION_TAU > DEFAULT_DT and ACCESS_CREDIT_HOLD_STEPS >= 1


def _thoughtseed_diagnosticity() -> dict:
    """d_i(s): each thoughtseed's priors normalized over regimes."""
    totals = {t: sum(THOUGHTSEED_STATE_PRIORS[s][t] for s in STATES) for t in THOUGHTSEEDS}
    return {t: {s: THOUGHTSEED_STATE_PRIORS[s][t] / totals[t] for s in STATES} for t in THOUGHTSEEDS}


THOUGHTSEED_DIAGNOSTICITY = _thoughtseed_diagnosticity()

MA_STATE = 'meta_awareness'
RECOGNITION_CUE = max(
    THOUGHTSEED_DIAGNOSTICITY,
    key=lambda thoughtseed: THOUGHTSEED_DIAGNOSTICITY[thoughtseed][MA_STATE],
)
PRACTICE_STATE = max(THOUGHTSEED_DIAGNOSTICITY[PRACTICE_OBJECT], key=THOUGHTSEED_DIAGNOSTICITY[PRACTICE_OBJECT].get)
RETURN_STATE = 'redirect_attention'


# P(on task | o) = 1 - d_o(MW).
ON_TASK_POSTERIOR = {t: 1.0 - row[OFF_TASK_STATE] for t, row in THOUGHTSEED_DIAGNOSTICITY.items()}


def on_task_log_odds_contrast(on_task: float) -> float:
    """ln[P(on task)/P(off task)] - ln(|S|-1)."""
    return log(on_task / (1.0 - on_task)) - log(len(STATES) - 1)


ON_TASK_CONTRAST = {t: on_task_log_odds_contrast(p) for t, p in ON_TASK_POSTERIOR.items()}

ON_TASK_WEIGHT = {
    thoughtseed: tanh(0.5 * contrast)
    for thoughtseed, contrast in ON_TASK_CONTRAST.items()
}

# Reporting only.
DISTRACTORS = tuple(
    thoughtseed for thoughtseed, row in THOUGHTSEED_DIAGNOSTICITY.items()
    if max(row, key=row.get) == OFF_TASK_STATE
)
assert DISTRACTORS and PRACTICE_OBJECT not in DISTRACTORS

# The recognition cue supplies no lapse evidence and no stabilization, and never holds the workspace.
FIRST_ORDER_CONTENTS = tuple(thoughtseed for thoughtseed in THOUGHTSEEDS if thoughtseed != RECOGNITION_CUE)
ACCESSIBLE_CONTENTS = FIRST_ORDER_CONTENTS
ACCESSIBLE_MASK = tuple(thoughtseed in ACCESSIBLE_CONTENTS for thoughtseed in THOUGHTSEEDS)
assert PRACTICE_OBJECT in ACCESSIBLE_CONTENTS and set(DISTRACTORS) <= set(ACCESSIBLE_CONTENTS)

# b_m: tonic contrast of uniform access counts, so the untrained monitor rests at one half.
CLARITY_BIAS = on_task_log_odds_contrast(
    sum(ON_TASK_POSTERIOR[content] for content in ACCESSIBLE_CONTENTS) / len(ACCESSIBLE_CONTENTS)
)

# q_t's target: sustained breath focus established (BF) or not (MW, MA, RA).
ON_OBJECT_STATES = (PRACTICE_STATE,)
LAPSE_STATES = tuple(state for state in STATES if state not in ON_OBJECT_STATES)


def _lapse_evidence() -> dict:
    """lambda(o) = ln[mean prior over lapse regimes / mean prior over on-object regimes], first-order contents only."""
    def mean_prior(states, thoughtseed):
        return sum(THOUGHTSEED_STATE_PRIORS[s][thoughtseed] for s in states) / len(states)
    return {
        thoughtseed: log(mean_prior(LAPSE_STATES, thoughtseed) / mean_prior(ON_OBJECT_STATES, thoughtseed))
        for thoughtseed in FIRST_ORDER_CONTENTS
    }


LAPSE_EVIDENCE = _lapse_evidence()
assert LAPSE_EVIDENCE[PRACTICE_OBJECT] < 0 and all(LAPSE_EVIDENCE[d] > 0 for d in DISTRACTORS)


# Cue term of the monitor drive: gain c(cue), zero at the MW (and BF) cue prior.
CUE_MONITOR_GAIN = ON_TASK_CONTRAST[RECOGNITION_CUE]
CUE_MONITOR_CENTRE = THOUGHTSEED_STATE_PRIORS[OFF_TASK_STATE][RECOGNITION_CUE]
# MA and MW have equal diagnosticity for every accessible content; the arg-max resolves to MW (first in STATES).
assert all(max(row, key=row.get) == OFF_TASK_STATE for t, row in THOUGHTSEED_DIAGNOSTICITY.items() if t in DISTRACTORS)
