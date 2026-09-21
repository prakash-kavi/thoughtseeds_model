"""Shared plotting style, labels, colors, and output handling."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

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

# Regimes and workspace contents appear together in Figs. 4 and 5, so they are
# separated by visual channel as well as by hue: regimes are pale background
# bands, contents are saturated foreground marks. STATE_BAND_COLORS is the tint
# used for those bands and for the matching legend patches, so a band and its
# legend entry cannot be mistaken for a content mark.
STATE_COLORS = {
    "breath_focus": "#2ca02c",
    "mind_wandering": "#1f77b4",
    "meta_awareness": "#d62728",
    "redirect_attention": "#ff7f0e",
}

STATE_BAND_COLORS = {
    "breath_focus": "#bcdfbc",
    "mind_wandering": "#bad3e8",
    "meta_awareness": "#f2c0c0",
    "redirect_attention": "#fbd9b5",
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
    'attend_breath': '#08519c',
    'equanimity': '#238b45',
    'aha_moment': '#d94801',
    'pain_discomfort': '#a50f15',
    'pending_tasks': '#6a51a3',
}


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

