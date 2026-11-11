"""Result validation, tail-window statistics and Fig. 4B."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from config.defaults import MODEL_VERSION, PLOT_STEPS, STATES
from experiments.training_pipeline import SEED
from viz.plotting_utils import STATE_DISPLAY_NAMES, save_figure, set_plot_style


_TIME_SERIES_KEYS = (
    'state_history',
    'surprisal_history',
    'clarity_history',
    'lapse_belief_history',
    'accessed_content_history',
    'network_activations_history',
    'thoughtseed_activations_history',
    'detection_event_history',
    'detection_completed_history',
    'regime_belief_history',
)


def validate_network_schema(results: Dict, networks: List[str]) -> None:
    if results.get('model_version') != MODEL_VERSION:
        raise ValueError(
            f"Results predate model {MODEL_VERSION}. Rerun python run_model.py run --seed "
            f"{results.get('seed', SEED)} before plotting or exporting statistics."
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


def regime_belief_accuracy(results: Dict, tail_steps: int = PLOT_STEPS) -> float:
    """Percentage of window steps at which the most probable regime of L2's belief is the regime L1 drew."""
    belief = np.asarray(results['regime_belief_history'][-tail_steps:])
    drawn = np.array([STATES.index(s) for s in results['state_history'][-tail_steps:]])
    return 100.0 * float(np.mean(belief.argmax(axis=1) == drawn))


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
            state: float(np.mean(lengths)) if lengths else None
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


def _transition_array(matrix: Dict[str, Dict[str, float]]) -> np.ndarray:
    return np.asarray([[matrix[source][destination] for destination in STATES] for source in STATES])


def _plot_transition_panel(ax, matrix: np.ndarray, title: str, exit_counts: Dict[str, int]):
    image = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    labels = [STATE_DISPLAY_NAMES[state] for state in STATES]
    ax.set_xticks(range(len(STATES)), [label.replace(' ', '\n') for label in labels], fontsize=8)
    ax.set_yticks(range(len(STATES)), labels, fontsize=8)
    ax.set_xlabel('To Regime', fontweight='bold')
    ax.set_ylabel('From Regime', fontweight='bold')
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
    fig.suptitle('Regime Transition Dynamics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, Path(save_path))
    plt.close(fig)
