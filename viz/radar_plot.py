"""State-conditional network activation profiles."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from viz.plotting_utils import (
    STATE_COLORS,
    STATE_DISPLAY_NAMES,
    save_figure,
    set_plot_style,
)
from config.defaults import STATES, NETWORKS


def plot_comparison(novice_data: dict, expert_data: dict, save_path: str):
    set_plot_style()
    nov_data = novice_data['network_profiles_mean']
    exp_data = expert_data['network_profiles_mean']

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('State-Conditional Network Activation Profiles', fontsize=18, fontweight='bold')

    angles = np.linspace(0, 2*np.pi, len(NETWORKS), endpoint=False).tolist()
    angles += angles[:1]

    for i, state in enumerate(STATES):
        ax = fig.add_subplot(2, 2, i+1, polar=True)

        nov_vals = [float(nov_data[state][net]) for net in NETWORKS]
        exp_vals = [float(exp_data[state][net]) for net in NETWORKS]
        nov_vals += nov_vals[:1]
        exp_vals += exp_vals[:1]

        ax.plot(angles, nov_vals, color=STATE_COLORS[state], linewidth=2.6, linestyle='--', label="Novice", zorder=3)
        ax.fill(angles, nov_vals, color=STATE_COLORS[state], alpha=0.22, zorder=2)
        
        ax.plot(angles, exp_vals, color=STATE_COLORS[state], linewidth=2.8, linestyle='-', label="Expert", zorder=3)
        ax.fill(angles, exp_vals, color=STATE_COLORS[state], alpha=0.16, zorder=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(NETWORKS, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(STATE_DISPLAY_NAMES[state], fontsize=15, fontweight='bold', pad=18)
        ax.grid(True, linestyle='--', alpha=0.7)
        for lbl in ax.get_xticklabels():
            lbl.set_y(0.02)

    labels = ["Expert", "Novice"]
    handles = [
        plt.Line2D([0], [0], color='black', linewidth=2.6, label=labels[0]),
        plt.Line2D([0], [0], color='black', linewidth=2.2, linestyle='--', label=labels[1])
    ]
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.08), ncol=2, fontsize=13)

    plt.tight_layout()
    save_figure(fig, Path(save_path))
    plt.close(fig)
