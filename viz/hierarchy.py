"""Fig. 5 hierarchical traces."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

from viz.plotting_utils import (
    DETECTION_COLOR,
    NETWORK_COLORS,
    THOUGHTSEED_COLORS,
    THOUGHTSEED_DISPLAY_NAMES,
    content_legend_handles,
    regime_legend_handles,
    save_figure,
    set_plot_style,
    state_bands,
)

from config.defaults import DEFAULT_DT, NETWORKS, THOUGHTSEEDS
from viz.ignition import access_runs


def _frameless(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)


def plot_hierarchy_continuous(data, save_path: str, level_name: str):
    n_steps = len(data['state_history'])
    start_step = int(data['tail_start'])
    # One conversion for every plotted element: seconds from the start of the evaluation window.
    x = lambda step: (step - start_step) * DEFAULT_DT  # noqa: E731
    seconds = x(np.arange(start_step, start_step + n_steps))
    states = list(data['state_history'])

    set_plot_style()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1.2, 1.5], figure=fig)

    ax1 = fig.add_subplot(gs[0])
    clarity = data['clarity_history']
    state_bands(ax1, states, x(start_step), DEFAULT_DT, zorder=0)
    ax1.plot(seconds, clarity, color='0.15', linewidth=1.6, zorder=3)

    # Band above the clarity axis: accessed content, detection events above it, the recognition interval ($D_t = 1$) below it.
    ax1.axhline(1.0, color='0.75', linewidth=0.8, zorder=2)
    for name, start, end in access_runs(data['accessed_content_history']):
        ax1.hlines(1.1, x(start_step + start), x(start_step + end),
                   color=THOUGHTSEED_COLORS[name], linewidth=6.0, zorder=4)
    held = np.asarray(data['detection_completed_history'], dtype=int)
    bounds = np.flatnonzero(np.diff(np.concatenate(([0], held, [0]))))
    for start, end in zip(bounds[::2], bounds[1::2]):
        ax1.hlines(1.035, x(start_step + start), x(start_step + end), color=DETECTION_COLOR, linewidth=4.0, zorder=4)
    events = [x(start_step + t) for t, event in enumerate(data['detection_event_history']) if event]
    ax1.plot(events, [1.175] * len(events), linestyle='none', marker='v', markersize=10, color=DETECTION_COLOR,
             zorder=5)

    ax1.set_ylabel('Monitoring clarity $m_t$', fontsize=12, fontweight='bold')
    ax1.set_title('Layer 3: workspace content and recognition (band above the clarity axis), '
                  'monitoring clarity $m_t$ (trace)', fontsize=14, fontweight='bold', loc='left')
    ax1.set_ylim(0, 1.22)
    ax1.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    _frameless(ax1)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    state_bands(ax2, states, x(start_step), DEFAULT_DT, zorder=0)
    ts_history = data['thoughtseed_activations_history']
    ts_alpha = 0.2
    for i, ts in enumerate(THOUGHTSEEDS):
        ts_vals = [row[i] for row in ts_history]
        smoothed = pd.Series(ts_vals, dtype=float).ewm(alpha=ts_alpha, adjust=False).mean().to_numpy()
        ax2.plot(
            seconds,
            smoothed,
            label=THOUGHTSEED_DISPLAY_NAMES[ts],
            color=THOUGHTSEED_COLORS[ts],
            linewidth=1.8,
        )
    ax2.set_ylabel('Thoughtseed Activation', fontsize=12, fontweight='bold')
    ax2.set_title('Layer 2: Thoughtseed Trajectories', fontsize=14, fontweight='bold', pad=8, loc='left')
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    _frameless(ax2)
    ax2.legend(loc='upper right', fontsize=9, ncol=2, frameon=True, framealpha=0.85, edgecolor='none')

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    state_bands(ax3, states, x(start_step), DEFAULT_DT, zorder=0)
    for net in NETWORKS:
        net_acts = [n[net] for n in data['network_activations_history']]
        alpha = 0.3
        smoothed_acts = pd.Series(net_acts, dtype=float).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        ax3.plot(seconds, smoothed_acts, label=net, color=NETWORK_COLORS[net], linewidth=2)

    recognition_handles = [
        Line2D([], [], color=DETECTION_COLOR, linestyle='none', marker='v', markersize=8,
               label=r'Detection event $\Delta_t$'),
        Patch(facecolor=DETECTION_COLOR, label='Lapse known, $D_t = 1$'),
    ]
    fig.legend(handles=content_legend_handles() + recognition_handles + regime_legend_handles(), loc='lower center',
               fontsize=9, frameon=False, ncol=5, bbox_to_anchor=(0.5, 0.005))

    ax3.set_xlabel('Time from evaluation-window start (s)', fontsize=12, fontweight='bold', labelpad=12)
    ax3.set_ylabel('Network Activation', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title('Layer 1: Network Dynamics', fontsize=14, fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    _frameless(ax3)

    fig.suptitle(f'Hierarchical Dynamics ({level_name})',
                 fontsize=16, fontweight='bold', x=0.5, ha='center')

    plt.tight_layout(rect=[0, 0.08, 1, 0.99])
    save_figure(fig, Path(save_path))
    plt.close(fig)
