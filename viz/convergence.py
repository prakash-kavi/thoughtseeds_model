"""Fig. S1 convergence diagnostics for a full simulation run."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from viz.plotting_utils import (
    set_plot_style,
    save_figure,
    STATE_COLORS,
    STATE_SHORT_NAMES
)
from config.defaults import STATES


def _rolling_summary(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if arr.size == 0:
        return np.array([]), np.array([])
    if window <= 1 or arr.size < window:
        missing = np.full(arr.shape, np.nan, dtype=float)
        return missing, missing
    series = pd.Series(arr, dtype=float)
    rolling = series.rolling(window=window, min_periods=window)
    return rolling.mean().to_numpy(), rolling.std(ddof=0).to_numpy()


def cumulative_state_fraction(states: List[str]) -> Dict[str, np.ndarray]:
    """Compute cumulative state fractions over time."""
    n = len(states)
    fractions = {state: np.zeros(n, dtype=float) for state in STATES}
    counts = {state: 0 for state in STATES}
    for idx, state in enumerate(states):
        counts[state] += 1
        denom = idx + 1
        for st in STATES:
            fractions[st][idx] = counts[st] / denom
    return fractions


def _deduplicate_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", frameon=True)


def _plot_single_convergence_pair(
    ax_loss,
    ax_occ,
    results: Dict,
    window: int,
    panel_title: str,
    highlight_spans: List[Dict],
):
    primary = np.asarray(results["free_energy_history"], dtype=float)
    phases = results.get("phase_history")
    states = results["state_history"]
    steps = np.arange(primary.size)
    plot_primary = primary.copy()
    if phases:
        boundaries = np.flatnonzero(np.asarray(phases[1:]) != np.asarray(phases[:-1])) + 1
        plot_primary[boundaries] = np.nan

    ax_loss.plot(
        steps,
        plot_primary,
        color="#cccccc",
        linewidth=1.0,
        label="MAP objective (raw)",
    )

    series_mean, series_std = _rolling_summary(plot_primary, window)
    ax_loss.plot(
        steps,
        series_mean,
        color="#E74C3C",
        linewidth=2.0,
        label=f"Rolling mean (w={window})",
    )

    valid = ~np.isnan(series_mean)
    if np.any(valid):
        lower = (series_mean - series_std)[valid]
        upper = (series_mean + series_std)[valid]
        ax_loss.fill_between(steps[valid], lower, upper, color="#E74C3C", alpha=0.18)

    ax_loss.set_ylabel("MAP inference objective", fontweight="bold")
    ax_loss.set_title(f"{panel_title}: MAP inference objective", fontsize=12, fontweight="bold")
    ax_loss.legend(loc="upper right", frameon=True)
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(max(float(np.nanmin(primary)) * 0.8, 1e-4), float(np.nanmax(primary)) * 1.2)

    fractions = cumulative_state_fraction(states)
    for state in STATES:
        ax_occ.plot(
            steps,
            fractions[state],
            color=STATE_COLORS[state],
            linewidth=1.6,
            label=STATE_SHORT_NAMES[state],
        )

    ax_occ.set_ylabel("Cumulative fraction", fontweight="bold")
    ax_occ.set_xlabel("Timestep", fontweight="bold")
    ax_occ.set_ylim(0.0, 1.0)
    ax_occ.set_title(f"{panel_title}: Cumulative state occupancy", fontsize=12, fontweight="bold")
    ax_occ.legend(loc="lower right", frameon=True)

    for ax in (ax_loss, ax_occ):
        for span in highlight_spans:
            start = max(0, int(span["start"]))
            end = min(int(span["end"]), primary.size)
            if end > start:
                ax.axvspan(
                    start,
                    end,
                    color=span["color"],
                    alpha=span["alpha"],
                    label=span["label"],
                )
        _deduplicate_legend(ax)


def plot_convergence(
    results: Dict,
    save_path: str,
    highlight_spans: List[Dict],
    window: int = 25,
    panel_title: str = "Convergence",
) -> None:
    set_plot_style()

    level = results["experience_level"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    _plot_single_convergence_pair(
        axes[0],
        axes[1],
        results,
        window,
        panel_title=panel_title,
        highlight_spans=highlight_spans,
    )

    fig.suptitle(f"Convergence diagnostics ({level.title()})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, Path(save_path))
    plt.close(fig)
