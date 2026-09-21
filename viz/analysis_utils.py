"""Tail-window aggregation and figures for the manuscript results."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from config.defaults import DEFAULT_DT, MODEL_VERSION, PLOT_STEPS, STATES
from viz.plotting_utils import STATE_COLORS, STATE_DISPLAY_NAMES, save_figure, set_plot_style


_TIME_SERIES_KEYS = (
    'state_history',
    'free_energy_history',
    'forward_error_history',
    'meta_awareness_history',
    'accessed_content_history',
    'network_activations_history',
    'thoughtseed_activations_history',
    'thoughtseed_prior_activations_history',
    'detection_event_history',
    'detection_completed_history',
)


def validate_network_schema(results: Dict, networks: List[str]) -> None:
    """Validate the current simulation-result contract used by all figures."""
    if results.get('model_version') != MODEL_VERSION:
        raise ValueError(
            f"Results predate model {MODEL_VERSION}. Rerun python run_model.py run --seed "
            f"{results.get('seed', 104)} before plotting or exporting statistics."
        )
    try:
        lengths = {key: len(results[key]) for key in _TIME_SERIES_KEYS}
    except KeyError as exc:
        raise ValueError(f"Results are missing required history: {exc.args[0]}.") from exc
    if not lengths['state_history'] or len(set(lengths.values())) != 1:
        raise ValueError("Result histories must be non-empty and aligned by timestep.")

    expected = set(networks)
    for index, row in enumerate(results['network_activations_history']):
        if not isinstance(row, dict) or set(row) != expected:
            actual = set(row) if isinstance(row, dict) else set()
            raise ValueError(
                f"Network schema mismatch at timestep {index}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
            )


def get_tail_window(results: Dict, tail_steps: int = PLOT_STEPS) -> Dict:
    """Return the final analysis window from an already validated result."""
    tail = dict(results)
    for key in _TIME_SERIES_KEYS:
        tail[key] = results[key][-tail_steps:]
    return tail


def _state_conditional_means(
    value_history: list,
    state_history: list,
    states: List[str],
    extractor,
) -> Dict[str, list]:
    means = {}
    for state in states:
        values = [extractor(value_history[index]) for index, observed in enumerate(state_history) if observed == state]
        means[state] = np.mean(values, axis=0).tolist() if values else None
    return means


def compute_network_profiles(
    results: Dict,
    states: List[str],
    networks: List[str],
    tail_steps: int = PLOT_STEPS,
) -> Dict[str, Dict[str, float]]:
    validate_network_schema(results, networks)
    return _network_profiles(get_tail_window(results, tail_steps), states, networks)


def _network_profiles(tail: Dict, states: List[str], networks: List[str]) -> Dict[str, Dict[str, float]]:
    means = _state_conditional_means(
        tail['network_activations_history'],
        tail['state_history'],
        states,
        lambda row: [row[network] for network in networks],
    )
    return {
        state: dict(zip(networks, means[state])) if means[state] is not None else {network: 0.0 for network in networks}
        for state in states
    }


def _fully_observed_dwell_lengths(
    state_history: List[str],
    states: List[str],
    tail_start: int,
) -> Dict[str, List[int]]:
    dwell_lengths = {state: [] for state in states}
    run_start = 0
    for index in range(1, len(state_history) + 1):
        if index < len(state_history) and state_history[index] == state_history[run_start]:
            continue
        if run_start >= tail_start and index < len(state_history) and state_history[run_start] in dwell_lengths:
            dwell_lengths[state_history[run_start]].append(index - run_start)
        run_start = index
    return dwell_lengths


def _transition_matrix(transitions: List[Dict], states: List[str], tail_start: int) -> tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    counts = {source: {destination: 0.0 for destination in states} for source in states}
    for transition in transitions:
        source = transition['from']
        destination = transition['to']
        if transition['timestamp'] >= tail_start and source in counts and destination in counts[source]:
            counts[source][destination] += 1.0
    exit_counts = {}
    for source, row in counts.items():
        total = sum(row.values())
        exit_counts[source] = int(total)
        if total:
            counts[source] = {destination: value / total for destination, value in row.items()}
    return counts, exit_counts


def compute_tail_statistics(
    results: Dict,
    states: List[str],
    tail_steps: int = PLOT_STEPS,
) -> Dict:
    state_history = results['state_history']
    tail_start = max(0, len(state_history) - tail_steps)
    dwell_lengths = _fully_observed_dwell_lengths(state_history, states, tail_start)
    transition_matrix, transition_exit_counts = _transition_matrix(results['transitions'], states, tail_start)
    return {
        'dwell_times': {
            state: float(np.mean(lengths)) if lengths else 0.0
            for state, lengths in dwell_lengths.items()
        },
        'dwell_run_lengths': dwell_lengths,
        'transition_matrix': transition_matrix,
        'transition_exit_counts': transition_exit_counts,
    }


def prepare_tail_data(
    results: Dict,
    states: List[str],
    networks: List[str],
    tail_steps: int = PLOT_STEPS,
) -> Dict:
    tail = get_tail_window(results, tail_steps)
    stats = compute_tail_statistics(results, states, tail_steps)
    tail['network_profiles_mean'] = _network_profiles(tail, states, networks)
    tail['tail_start'] = max(0, len(results['state_history']) - tail_steps)
    tail.update(stats)
    return tail


def plot_fe_and_dwell(novice_data: dict, expert_data: dict, save_path: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(STATES))
    width = 0.28

    novice_lengths = novice_data['dwell_run_lengths']
    expert_lengths = expert_data['dwell_run_lengths']

    def _summary(lengths):
        """Bar heights with asymmetric whiskers reaching the shortest and longest episode."""
        means, lower, upper = [], [], []
        for state in STATES:
            durations = DEFAULT_DT * np.asarray(lengths[state], dtype=float)
            if durations.size == 0:
                means.append(0.0)
                lower.append(0.0)
                upper.append(0.0)
                continue
            mean = float(durations.mean())
            means.append(mean)
            lower.append(mean - float(durations.min()))
            upper.append(float(durations.max()) - mean)
        return means, np.asarray([lower, upper])

    # Whiskers span the shortest and longest episode observed, not a standard
    # deviation. A standard deviation over three or four episodes is not an
    # estimate of the duration law: its spread is a fraction of the c_v the law
    # specifies, which would read as an unusually regular practitioner rather
    # than as a small sample. The range makes no such claim -- it reports only
    # what the window contained. Across-seed variability is reported in the
    # five-run robustness table.
    novice_means, novice_range = _summary(novice_lengths)
    expert_means, expert_range = _summary(expert_lengths)
    colors = [STATE_COLORS[state] for state in STATES]
    error_style = dict(ecolor='black', capsize=3, elinewidth=1, capthick=1)
    ax.bar(x - width / 2, novice_means, width, color=colors, alpha=0.4, hatch='//',
           edgecolor='black', linewidth=1, yerr=novice_range, error_kw=error_style)
    ax.bar(x + width / 2, expert_means, width, color=colors, alpha=0.7,
           edgecolor='black', linewidth=1, yerr=expert_range, error_kw=error_style)

    whisker_tops = np.concatenate([np.asarray(novice_means) + novice_range[1],
                                   np.asarray(expert_means) + expert_range[1]])
    upper = max(1.0, float(np.max(whisker_tops)))
    ax.set_ylim(0, upper * 1.15)
    ax.set_ylabel('Average Dwell Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Dwell times, with the range of episode durations observed',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_DISPLAY_NAMES[state] for state in STATES], fontsize=11)
    ax.legend(handles=[
        Patch(facecolor='0.7', alpha=0.4, hatch='//', edgecolor='black', label='Novice'),
        Patch(facecolor='0.7', alpha=0.7, edgecolor='black', label='Expert'),
    ], fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    save_figure(fig, Path(save_path))
    plt.close(fig)


def _transition_array(matrix: Dict[str, Dict[str, float]]) -> np.ndarray:
    return np.asarray([[matrix[source][destination] for destination in STATES] for source in STATES])


def _plot_transition_panel(ax, matrix: np.ndarray, title: str, exit_counts: Dict[str, int]):
    image = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    labels = [STATE_DISPLAY_NAMES[state] for state in STATES]
    ax.set_xticks(range(len(STATES)), [label.replace(' ', '\n') for label in labels], fontsize=8)
    ax.set_yticks(range(len(STATES)), labels, fontsize=8)
    ax.set_xlabel('To State', fontweight='bold')
    ax.set_ylabel('From State', fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    for row, column in np.ndindex(matrix.shape):
        value = matrix[row, column]
        if value > 0.01:
            ax.text(column, row, f'{value:.2f}', ha='center', va='center', color='white' if value > 0.5 else 'black', fontsize=8)
    ax.text(0.5, -0.25, 'Observed exits: ' + ', '.join(
        f'{STATE_DISPLAY_NAMES[state]}={exit_counts[state]}' for state in STATES
    ), transform=ax.transAxes, ha='center', fontsize=8)
    return image


def plot_transitions(novice_data: dict, expert_data: dict, save_path: str) -> None:
    set_plot_style()
    fig, (novice_ax, expert_ax) = plt.subplots(1, 2, figsize=(14, 6))
    novice_image = _plot_transition_panel(
        novice_ax, _transition_array(novice_data['transition_matrix']),
        'Novice Transition Dynamics', novice_data['transition_exit_counts'],
    )
    expert_image = _plot_transition_panel(
        expert_ax, _transition_array(expert_data['transition_matrix']),
        'Expert Transition Dynamics', expert_data['transition_exit_counts'],
    )
    fig.colorbar(novice_image, ax=novice_ax, label='Probability')
    fig.colorbar(expert_image, ax=expert_ax, label='Probability')
    fig.suptitle('State Transition Dynamics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, Path(save_path))
    plt.close(fig)
