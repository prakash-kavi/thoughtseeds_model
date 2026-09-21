"""Phenotype-indexed attractors, dwell means, transitions, and learning rates."""

from config.defaults import DEFAULT_DT, LATENT_TAU

NETWORK_PROFILES = {
    "breath_focus": {
        "novice": {"DMN": 0.50, "VAN": 0.45, "DAN": 0.58, "FPN": 0.60, "VIS": 0.30, "SOM": 0.55, "LIM": 0.35},
        "expert": {"DMN": 0.35, "VAN": 0.45, "DAN": 0.65, "FPN": 0.70, "VIS": 0.25, "SOM": 0.65, "LIM": 0.25},
    },
    "mind_wandering": {
        "novice": {"DMN": 0.82, "VAN": 0.35, "DAN": 0.30, "FPN": 0.33, "VIS": 0.42, "SOM": 0.30, "LIM": 0.75},
        "expert": {"DMN": 0.70, "VAN": 0.40, "DAN": 0.28, "FPN": 0.38, "VIS": 0.35, "SOM": 0.28, "LIM": 0.55},
    },
    # MA is the interval in which mind wandering becomes known: DMN and LIM
    # stay elevated between their MW and BF values while
    # VAN (salience) comes online. Disengagement belongs to RA.
    "meta_awareness": {
        "novice": {"DMN": 0.70, "VAN": 0.85, "DAN": 0.42, "FPN": 0.56, "VIS": 0.35, "SOM": 0.38, "LIM": 0.62},
        "expert": {"DMN": 0.58, "VAN": 0.85, "DAN": 0.42, "FPN": 0.60, "VIS": 0.30, "SOM": 0.40, "LIM": 0.48},
    },
    "redirect_attention": {
        "novice": {"DMN": 0.40, "VAN": 0.45, "DAN": 0.78, "FPN": 0.72, "VIS": 0.33, "SOM": 0.52, "LIM": 0.30},
        "expert": {"DMN": 0.30, "VAN": 0.40, "DAN": 0.82, "FPN": 0.72, "VIS": 0.28, "SOM": 0.60, "LIM": 0.22},
    },
}

# Phenomenologically motivated baseline dwell means, not inferred durations or
# relaxation-time derivations. MA includes time before and after detection;
# its dwell is not detection latency. Novice noticing and redirection are slower
# by configuration, while control can modify the realized durations.
DWELL_MEAN_SECONDS = {
    'expert': {'breath_focus': 20.0, 'mind_wandering': 14.0, 'meta_awareness': 4.5, 'redirect_attention': 3.0},
    'novice': {'breath_focus': 14.0, 'mind_wandering': 20.0, 'meta_awareness': 7.5, 'redirect_attention': 5.0},
}

DWELL_CV = 0.35

# States whose configured attractor mu_x(s) must be expressible within a typical
# dwell. RA is excluded: it is transitional and hands an incomplete relaxation on
# to BF, which continues it.
ATTRACTOR_EXPRESSING_STATES = ('breath_focus', 'mind_wandering', 'meta_awareness')
ATTRACTOR_SETTLING_HORIZONS = 3.0   # e^-3, ~5% residual, is "settled"

# How much faster a layer must run than the layer it feeds. Hierarchical
# predictive processing places the slower timescales above the faster ones, so
# L1's slowest relaxation mode must sit below the L2 latent it drives.
TIMESCALE_SEPARATION = 2.0

# Lower bound on L1 drift eigenvalue real parts, computed from chosen settling
# and timescale-separation requirements. Uncoupled coordinates attain the bound;
# phenotype scaling still changes the coupled relaxation modes.
THETA_SETTLING_MARGIN = max(
    ATTRACTOR_SETTLING_HORIZONS / min(
        DWELL_MEAN_SECONDS[level][state]
        for level in DWELL_MEAN_SECONDS
        for state in ATTRACTOR_EXPRESSING_STATES
    ),
    TIMESCALE_SEPARATION / LATENT_TAU,
)

# The Gershgorin row bound (diagonal + off-diagonal absolute sum) must stay below
# 2/dt_sub: a sufficient Euler-stability condition for the constructed matrices.
L1_SUBSTEPS = 2
THETA_STABILITY_LIMIT = 2.0 / (DEFAULT_DT / L1_SUBSTEPS)

# Noticing is the canonical way out of a lapse: most MW exits pass through MA.
# The residual direct MW->BF and MW->RA mass represents a return to the breath
# that no detection event accompanied.
STATE_TRANSITION_PROBS = {
    'expert': {
        'breath_focus': {'mind_wandering': 0.60, 'meta_awareness': 0.20, 'redirect_attention': 0.20},
        'mind_wandering': {'meta_awareness': 0.85, 'redirect_attention': 0.10, 'breath_focus': 0.05},
        'meta_awareness': {'redirect_attention': 0.85, 'breath_focus': 0.10, 'mind_wandering': 0.05},
        'redirect_attention': {'breath_focus': 0.80, 'meta_awareness': 0.15, 'mind_wandering': 0.05},
    },
    'novice': {
        'breath_focus': {'mind_wandering': 0.80, 'meta_awareness': 0.10, 'redirect_attention': 0.10},
        'mind_wandering': {'meta_awareness': 0.75, 'redirect_attention': 0.15, 'breath_focus': 0.10},
        'meta_awareness': {'redirect_attention': 0.60, 'breath_focus': 0.10, 'mind_wandering': 0.30},
        'redirect_attention': {'breath_focus': 0.60, 'mind_wandering': 0.25, 'meta_awareness': 0.15},
    },
}

for _level, _rows in STATE_TRANSITION_PROBS.items():
    for _state, _row in _rows.items():
        _total = sum(_row.values())
        assert abs(_total - 1.0) < 1e-9, (
            f"STATE_TRANSITION_PROBS[{_level!r}][{_state!r}] sums to {_total}, "
            "not 1.0 -- exit destinations must be a proper distribution."
        )

LEARNING_RATES = {'novice': 0.01, 'expert': 0.02}

# Hold training exposure constant. Phenotype differences arise from the
# configured dynamics and learning rates, not unequal training duration.
PRACTICE_STEPS = {'novice': 12000, 'expert': 12000}


def get_exit_transition_probs(experience_level: str, current_state: str) -> dict:
    return dict(STATE_TRANSITION_PROBS[experience_level][current_state])
