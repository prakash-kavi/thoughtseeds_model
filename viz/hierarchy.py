"""Hierarchical dynamics visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from viz.plotting_utils import (
    STATE_BAND_COLORS,
    STATE_SHORT_NAMES,
    STATE_DISPLAY_NAMES,
    NETWORK_COLORS,
    THOUGHTSEED_COLORS,
    save_figure,
    set_plot_style,
)
from matplotlib.patches import Patch

from config.defaults import NETWORKS, THOUGHTSEEDS, STATES
from viz.ignition import access_runs


def _regime_bands(ax, states, start_step):
    """Shade the concurrent regime behind the traces, as in Fig. 4."""
    run_start = 0
    for t in range(1, len(states) + 1):
        if t == len(states) or states[t] != states[run_start]:
            ax.axvspan(start_step + run_start, start_step + t,
                       color=STATE_BAND_COLORS[states[run_start]], alpha=0.55,
                       linewidth=0, zorder=0)
            run_start = t


def plot_hierarchy_continuous(data, save_path: str, level_name: str):
    n_steps = len(data['state_history'])
    # Plot on the run's own timeline so this figure and Fig. 4 share an axis.
    start_step = int(data.get('tail_start', 0))
    time_steps = np.arange(start_step, start_step + n_steps)
    states = list(data['state_history'])

    set_plot_style()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1.2, 1.5], figure=fig)

    ax1 = fig.add_subplot(gs[0])
    meta_awareness = data['meta_awareness_history']
    _regime_bands(ax1, states, start_step)
    ax1.plot(time_steps, meta_awareness, color='0.15', linewidth=1.6, zorder=3)

    # Accessed workspace content as a colored strip above the m_t trace. The
    # strip is keyed to thoughtseeds, not regimes; both legends are labeled.
    for name, start, end in access_runs(data['accessed_content_history']):
        ax1.hlines(1.1, start_step + start, start_step + end,
                   color=THOUGHTSEED_COLORS[name], linewidth=6.0, zorder=4)

    ax1.set_ylabel('Meta-awareness $m_t$', fontsize=12, fontweight='bold')
    ax1.set_title('Layer 3: workspace content (strip, upper) and monitoring clarity $m_t$ (trace)',
                  fontsize=14, fontweight='bold', loc='left')
    ax1.set_ylim(0, 1.18)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.patch.set_visible(False)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _regime_bands(ax2, states, start_step)
    ts_history = data['thoughtseed_activations_history']
    ts_alpha = 0.2
    for i, ts in enumerate(THOUGHTSEEDS):
        ts_vals = [row[i] for row in ts_history]
        smoothed = pd.Series(ts_vals, dtype=float).ewm(alpha=ts_alpha, adjust=False).mean().to_numpy()
        ax2.plot(
            time_steps,
            smoothed,
            label=ts,
            color=THOUGHTSEED_COLORS[ts],
            linewidth=1.8,
        )
    ax2.set_ylabel('Thoughtseed Activation', fontsize=12, fontweight='bold')
    ax2.set_title('Layer 2: Thoughtseed Trajectories', fontsize=14, fontweight='bold', pad=8, loc='left')
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.patch.set_visible(False)
    ax2.legend(loc='upper right', fontsize=9, ncol=2, frameon=False)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _regime_bands(ax3, states, start_step)
    for net in NETWORKS:
        net_acts = [n[net] for n in data['network_activations_history']]
        alpha = 0.3
        smoothed_acts = pd.Series(net_acts, dtype=float).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        ax3.plot(time_steps, smoothed_acts, label=net, color=NETWORK_COLORS[net], linewidth=2)

    legend_elements = [
        plt.Line2D([0], [0], color=THOUGHTSEED_COLORS[ts], lw=5, label=f"content: {ts}")
        for ts in THOUGHTSEEDS
    ] + [
        Patch(facecolor=STATE_BAND_COLORS[state], alpha=0.55,
              label=f"regime {STATE_SHORT_NAMES[state]}: {STATE_DISPLAY_NAMES[state]}")
        for state in STATES
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               fontsize=9, frameon=False, ncol=5, bbox_to_anchor=(0.5, 0.005))

    ax3.set_xlabel('Time (timesteps)', fontsize=12, fontweight='bold', labelpad=12)
    ax3.set_ylabel('Network Activation', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title('Layer 1: Network Dynamics', fontsize=14, fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.patch.set_visible(False)

    fig.suptitle(f'Hierarchical Dynamics ({level_name})',
                 fontsize=16, fontweight='bold', x=0.5, ha='center')

    plt.tight_layout(rect=[0, 0.08, 1, 0.99])
    save_figure(fig, Path(save_path))
    plt.close(fig)
