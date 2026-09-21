"""Workspace access (GNW) and meta-awareness measures, and the Fig. 4 raster."""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config.defaults import DEFAULT_DT, DISTRACTORS, PLOT_STEPS, PRACTICE_OBJECT
from viz.plotting_utils import STATE_BAND_COLORS, STATE_SHORT_NAMES, save_figure, set_plot_style

META_EPISODE_THRESHOLD = 0.5
FOLLOW_THROUGH_COLOR = '#08519c'
DETECTION_COLOR = '#d94801'


def access_runs(content_history: Sequence[Optional[str]]) -> List[Tuple[str, int, int]]:
    """Contiguous [start, end) runs during which one thoughtseed holds the workspace."""
    runs, start = [], 0
    history = list(content_history)
    for t in range(1, len(history) + 1):
        if t == len(history) or history[t] != history[start]:
            if history[start] is not None:
                runs.append((history[start], start, t))
            start = t
    return runs


def access_onsets(content_history: Sequence[Optional[str]]) -> List[int]:
    """Steps at which a new content enters the workspace (one access event each)."""
    return [start for _, start, _ in access_runs(content_history)]


def meta_awareness_episodes(meta_awareness_history, threshold: float = META_EPISODE_THRESHOLD,
                            previous_value: Optional[float] = None):
    """[start, end) spans with m_t >= threshold; a span open at the end is closed there."""
    above = np.asarray(meta_awareness_history, dtype=float) >= threshold
    spans, start = [], None
    continuing = previous_value is not None and previous_value >= threshold
    for t, value in enumerate(above):
        if continuing:
            if not value:
                continuing = False
            continue
        if value and start is None:
            start = t
        elif not value and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, len(above)))
    return spans


def breath_follow_through(events, state_history):
    """The [start, end) of the breath-focus run that follows each event.

    Returns None for an event with no observed BF run before the window ends. An
    event occurring during BF is measured over the remainder of that run. The
    run length is reported directly rather than thresholded: a phenotype-scaled
    criterion made a 13.4 s novice run "sustained" while a longer 16.0 s expert
    run was not, which inverted the contrast it was meant to show.
    """
    states = list(state_history)
    breath_runs = []
    for event in events:
        step = event
        while step < len(states) and states[step] != 'breath_focus':
            step += 1
        if step == len(states):
            breath_runs.append(None)
            continue
        start = step
        while step < len(states) and states[step] == 'breath_focus':
            step += 1
        breath_runs.append((start, step))
    return breath_runs


def access_summary(results: Dict, window: int = PLOT_STEPS) -> Dict[str, float]:
    """Window-level access and meta-awareness measures used in the manuscript."""
    full_content = list(results['accessed_content_history'])
    content = full_content[-window:]
    window_start = max(0, len(full_content) - window)
    states = list(results['state_history'])[-window:]
    m = np.asarray(results['meta_awareness_history'], dtype=float)[-window:]
    practice_runs = [end - start for name, start, end in access_runs(content) if name == PRACTICE_OBJECT]

    def fraction(mask) -> float:
        mask = np.asarray(mask, dtype=bool)
        return float(mask.mean()) if mask.size else 0.0

    in_bf = np.asarray([s == 'breath_focus' for s in states])
    in_mw = np.asarray([s == 'mind_wandering' for s in states])
    distractor = np.asarray([c in DISTRACTORS for c in content])
    return {
        'access_onsets': float(sum(start >= window_start for start in access_onsets(full_content))),
        'breath_access_percent': 100.0 * fraction([c == PRACTICE_OBJECT for c in content]),
        'breath_run_seconds': DEFAULT_DT * float(np.mean(practice_runs)) if practice_runs else 0.0,
        'distractor_access_mw_percent': 100.0 * fraction(distractor[in_mw]) if in_mw.any() else 0.0,
        'meta_awareness_bf': float(m[in_bf].mean()) if in_bf.any() else 0.0,
        # Conditional on being in BF the phenotypes hold near-identical workspace
        # content, so a BF-conditional mean cannot show the clarity contrast. The
        # session mean can: it carries how much of the window each phenotype
        # spends with on-task content in the workspace.
        'meta_awareness_session': float(m.mean()) if m.size else 0.0,
        'meta_episodes': float(len(meta_awareness_episodes(
            m, previous_value=results['meta_awareness_history'][window_start - 1] if window_start else None,
        ))),
        'empty_slot_percent': 100.0 * fraction([c is None for c in content]),
    }


def detection_summary(results: Dict, window: int = PLOT_STEPS) -> Dict:
    """Detection within MA and subsequent direct RA exits, excluding left-cut MA.

    Detection latency uses every eligible detected MA, even if its exit is not
    observed. Redirection latency uses only observed direct RA exits; other
    destinations and right-censored episodes are counted separately. Because MA
    is the interval in which detection develops, fully observed detected
    episodes are also split at the detection event into a pre-detection segment,
    in which the lapse is not yet noticed, and a post-detection segment.
    """
    states = results['state_history']
    events = results['detection_event_history']
    start_window = max(0, len(states) - window)
    detected, redirected, other, unresolved = 0, 0, 0, 0
    detection_latencies, redirection_latencies = [], []
    pre_detection, post_detection = [], []
    i = start_window
    while i < len(states):
        if states[i] != 'meta_awareness':
            i += 1
            continue
        start = i
        while i < len(states) and states[i] == 'meta_awareness':
            i += 1
        if start == start_window and start > 0 and states[start - 1] == 'meta_awareness':
            continue
        onset = next((t for t in range(start, i) if events[t]), None)
        if onset is None:
            continue
        detected += 1
        detection_latencies.append((onset - start) * DEFAULT_DT)
        if i == len(states):
            unresolved += 1
        elif states[i] == 'redirect_attention':
            redirected += 1
            redirection_latencies.append((i - onset) * DEFAULT_DT)
        else:
            other += 1
        if i < len(states):
            pre_detection.append((onset - start) * DEFAULT_DT)
            post_detection.append((i - onset) * DEFAULT_DT)
    return {
        'detected_ma_episodes': detected,
        'direct_ra_exits': redirected,
        'other_exits': other,
        'unresolved_detected_episodes': unresolved,
        'detection_latency_seconds': float(np.mean(detection_latencies)) if detection_latencies else None,
        'redirection_latency_seconds': float(np.mean(redirection_latencies)) if redirection_latencies else None,
        'ma_pre_detection_seconds': float(np.mean(pre_detection)) if pre_detection else None,
        'ma_post_detection_seconds': float(np.mean(post_detection)) if post_detection else None,
    }


def _state_bands(ax, states, start_step):
    states = list(states)
    run_start = 0
    for t in range(1, len(states) + 1):
        if t == len(states) or states[t] != states[run_start]:
            ax.axvspan(start_step + run_start, start_step + t,
                       color=STATE_BAND_COLORS[states[run_start]], alpha=0.55, linewidth=0)
            run_start = t


def _raster_panel(ax, results, label, display_steps):
    start_step = len(results['state_history']) - display_steps
    m = np.asarray(results['meta_awareness_history'], dtype=float)[start_step:]
    states = results['state_history'][start_step:]
    detections = [t for t, event in enumerate(results['detection_event_history'][start_step:]) if event]
    # Same left-edge rule as detection_summary: an MA episode already under way
    # at the window's edge is not a detection observed within it, so the panel
    # count and the reported statistics cannot disagree.
    if start_step and states and states[0] == 'meta_awareness' \
            and results['state_history'][start_step - 1] == 'meta_awareness':
        first_complete = next((i for i, s in enumerate(states) if s != 'meta_awareness'), len(states))
        detections = [t for t in detections if t >= first_complete]
    breath_runs = breath_follow_through(detections, states)
    steps = np.arange(start_step, start_step + len(m))

    _state_bands(ax, states, start_step)
    ax.axhline(META_EPISODE_THRESHOLD, color='0.55', linestyle='--', linewidth=0.7, zorder=2)
    ax.plot(steps, m, color='0.15', linewidth=0.9, zorder=3)
    observed = []
    for detection, breath_run in zip(detections, breath_runs):
        ax.vlines(start_step + detection, 1.02, 1.14, color=DETECTION_COLOR, linewidth=0.8, zorder=4)
        ax.plot([start_step + detection], [1.14], linestyle='none', marker='v', markersize=4.0,
                color=DETECTION_COLOR, zorder=5)
        if breath_run is None:
            continue
        run_start, run_end = breath_run
        observed.append((run_end - run_start) * DEFAULT_DT)
        ax.hlines(1.08, start_step + run_start, start_step + run_end,
                  color=FOLLOW_THROUGH_COLOR, linewidth=3.0, zorder=4)
        ax.plot([start_step + detection, start_step + run_start], [1.12, 1.08],
                color=FOLLOW_THROUGH_COLOR, linewidth=0.6, alpha=0.7, zorder=3.6)

    ax.set_ylim(0.0, 1.18)
    ax.set_xlim(start_step, start_step + len(m))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('Meta-awareness $m_t$')
    above = float((m >= META_EPISODE_THRESHOLD).mean()) * 100.0
    breath = access_summary(results, display_steps)['breath_access_percent']
    run_text = (f"each followed by {np.mean(observed):.1f} s of breath focus on average"
                if observed else "no breath-focus run observed")
    ax.set_title(
        f"{label}: {len(detections)} detections of mind wandering, {run_text}\n"
        f"breath content occupies the global workspace {breath:.0f}% of the window, "
        f"and clarity stays at or above {META_EPISODE_THRESHOLD} for {above:.0f}% of it",
        loc='left', fontsize=9,
    )


def plot_meta_awareness_raster(novice_results, expert_results, save_path: str, display_steps: int = PLOT_STEPS) -> None:
    """Plot each MA detection event and the breath-focus run that follows it."""
    if display_steps <= 0 or display_steps > PLOT_STEPS:
        raise ValueError(f"display_steps must be between 1 and {PLOT_STEPS}.")
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    _raster_panel(axes[0], novice_results, 'Novice', display_steps)
    _raster_panel(axes[1], expert_results, 'Expert', display_steps)
    axes[1].set_xlabel('Timestep')
    handles = [
        Line2D([], [], color='0.15', linewidth=0.9, label='$m_t$'),
        Line2D([], [], color=DETECTION_COLOR, linestyle='none', marker='v', markersize=4.0,
               label='Detection event'),
        Line2D([], [], color=FOLLOW_THROUGH_COLOR, linewidth=3.0, label='Breath-focus run that follows'),
        Line2D([], [], color='0.55', linestyle='--', linewidth=0.7, label=f'$m_t$ = {META_EPISODE_THRESHOLD}'),
    ] + [Patch(facecolor=STATE_BAND_COLORS[s], alpha=0.55, label=f'regime: {STATE_SHORT_NAMES[s]}')
         for s in STATE_BAND_COLORS]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
