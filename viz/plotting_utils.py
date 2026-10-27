"""Shared plotting style, labels, colors and output handling."""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from config.defaults import STATES, THOUGHTSEEDS

STATE_DISPLAY_NAMES = {
    "breath_focus": "Breath Focus",
    "mind_wandering": "Mind Wandering",
    "meta_awareness": "Meta Awareness",
    "redirect_attention": "Redirect Attention"
}

STATE_SHORT_NAMES = {
    "breath_focus": "BF",
    "mind_wandering": "MW",
    "meta_awareness": "MA",
    "redirect_attention": "RA",
}

THOUGHTSEED_DISPLAY_NAMES = {name: r"$\mathit{" + name.replace("_", r"\_") + "}$" for name in THOUGHTSEEDS}

# Regime bands are the regime hue tinted toward white.
STATE_COLORS = {
    "breath_focus": "#2ca02c",
    "mind_wandering": "#1f77b4",
    "meta_awareness": "#d62728",
    "redirect_attention": "#ff7f0e",
}
REGIME_BAND_TINT = 0.32
REGIME_BAND_ALPHA = 0.55
STATE_BAND_COLORS = {
    state: mcolors.to_hex([1.0 - REGIME_BAND_TINT * (1.0 - c) for c in mcolors.to_rgb(color)])
    for state, color in STATE_COLORS.items()
}

NETWORK_COLORS = {
    'DMN': '#CA3542',
    'VAN': '#B77FB4',
    'DAN': '#2C8B4B',
    'FPN': '#E58429',
    'VIS': '#8B5FBF',
    'SOM': '#4A90D9',
    'LIM': '#C9A227',
}

THOUGHTSEED_COLORS = {
    'breath_sensation': '#08519c',
    'pain_discomfort': '#a50f15',
    'pending_tasks': '#6a51a3',
    'recognition_cue': '#d94801',
    'intention_to_return': '#238b45',
}
# Outside the regime and content palettes, so a known lapse is not read as Redirect Attention.
DETECTION_COLOR = '0.15'

assert list(STATE_DISPLAY_NAMES) == list(STATE_SHORT_NAMES) == list(STATE_COLORS) == STATES
assert list(THOUGHTSEED_COLORS) == THOUGHTSEEDS


def state_bands(ax, states, origin=0.0, step_width=1.0, zorder=1):
    """Shade the concurrent regime behind the traces; step k spans origin + [k, k+1) * step_width."""
    states = list(states)
    run_start = 0
    for t in range(1, len(states) + 1):
        if t == len(states) or states[t] != states[run_start]:
            ax.axvspan(origin + run_start * step_width, origin + t * step_width,
                       color=STATE_BAND_COLORS[states[run_start]], alpha=REGIME_BAND_ALPHA,
                       linewidth=0, zorder=zorder)
            run_start = t


def content_legend_handles(names=THOUGHTSEEDS):
    return [Patch(facecolor=THOUGHTSEED_COLORS[name], label=THOUGHTSEED_DISPLAY_NAMES[name]) for name in names]


def regime_legend_handles(short: bool = False):
    return [Patch(facecolor=STATE_BAND_COLORS[state], alpha=REGIME_BAND_ALPHA,
                  label=STATE_SHORT_NAMES[state] if short
                  else f"regime {STATE_SHORT_NAMES[state]}: {STATE_DISPLAY_NAMES[state]}") for state in STATES]


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

def save_figure(fig, save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
