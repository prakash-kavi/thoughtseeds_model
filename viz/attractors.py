"""PCA projections of the L1 and L2 plot-window trajectories."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from config.defaults import STATES, NETWORKS
from viz.plotting_utils import (
    STATE_COLORS,
    STATE_SHORT_NAMES,
    save_figure,
    set_plot_style,
)

def _prepare_data(tail_data: dict) -> Dict:
    return {
        "activations": np.asarray(tail_data['thoughtseed_activations_history'], dtype=float),
        "network_activations": np.asarray(
            [[row[net] for net in NETWORKS] for row in tail_data['network_activations_history']],
            dtype=float,
        ),
        "states": tail_data['state_history'],
    }


def _fit_pca(X: np.ndarray) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca, pca.explained_variance_ratio_


def _state_centroids_pca(
    projected: np.ndarray,
    states: List[str],
) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    for state in STATES:
        idx = [i for i, st in enumerate(states) if st == state]
        if idx:
            centroids[state] = projected[idx].mean(axis=0)
    return centroids


def _plot_pca_pair(
    axes: tuple[plt.Axes, plt.Axes],
    novice: Dict[str, np.ndarray],
    expert: Dict[str, np.ndarray],
    feature_key: str,
    row_title: str,
) -> None:
    all_acts = np.concatenate([novice[feature_key], expert[feature_key]], axis=0)
    pca, var_ratio = _fit_pca(all_acts)
    nov_proj = pca.transform(novice[feature_key])
    exp_proj = pca.transform(expert[feature_key])

    axis_titles = (
        f"PC1 ({var_ratio[0]*100:.1f}%)",
        f"PC2 ({var_ratio[1]*100:.1f}%)",
    )

    for ax, cohort_data, proj, title in zip(
        axes, (novice, expert), (nov_proj, exp_proj), ("Novice", "Expert")
    ):
        states = cohort_data["states"]

        x = proj[:, 0]
        y = proj[:, 1]

        if len(x) > 1:
            max_points = 1200
            step = max(1, len(x) // max_points)
            x_s = x[::step]
            y_s = y[::step]

            points = np.column_stack([x_s, y_s])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors="#1B4F72", linewidths=1.0, alpha=0.4)
            ax.add_collection(lc)


        centroids = _state_centroids_pca(proj, states)
        for state, centre in centroids.items():
            ax.text(
                centre[0],
                centre[1],
                STATE_SHORT_NAMES[state],
                color=STATE_COLORS[state],
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(axis_titles[0], fontweight="bold")
        ax.grid(False)
        ax.margins(0.05)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel(axis_titles[1], fontweight="bold")
    axes[0].text(
        -0.18,
        0.5,
        row_title,
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )


def plot_attractor_pca(
    novice_data: dict,
    expert_data: dict,
    save_path: str,
) -> None:
    novice = _prepare_data(novice_data)
    expert = _prepare_data(expert_data)

    set_plot_style()

    fig = plt.figure(figsize=(11.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.12)

    ax_ts_left = fig.add_subplot(gs[0, 0])
    ax_ts_right = fig.add_subplot(gs[0, 1], sharex=ax_ts_left, sharey=ax_ts_left)
    ax_net_left = fig.add_subplot(gs[1, 0])
    ax_net_right = fig.add_subplot(gs[1, 1], sharex=ax_net_left, sharey=ax_net_left)

    _plot_pca_pair(
        (ax_ts_left, ax_ts_right),
        novice,
        expert,
        "activations",
        "L2 Thoughtseeds",
    )
    _plot_pca_pair(
        (ax_net_left, ax_net_right),
        novice,
        expert,
        "network_activations",
        "L1 Networks",
    )

    ax_net_left.tick_params(axis="x", labelbottom=True)
    ax_net_right.tick_params(axis="x", labelbottom=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.12)

    fig.suptitle("PCA Trajectories Across the Hierarchy", fontsize=16, fontweight="bold")

    save_figure(fig, Path(save_path))
    plt.close(fig)
