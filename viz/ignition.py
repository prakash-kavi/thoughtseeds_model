"""Access, recognition and practice measures; Figs. 6 and S2."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config.defaults import (ACCESSIBLE_CONTENTS, DEFAULT_DT, MA_STATE, DISTRACTORS, OFF_TASK_STATE,
                             PLOT_STEPS, PRACTICE_OBJECT, PRACTICE_STATE, RETURN_STATE, STATES)
from model.policy import ReturnPolicy
from viz.plotting_utils import (DETECTION_COLOR, STATE_BAND_COLORS, STATE_SHORT_NAMES, THOUGHTSEED_COLORS,
                                 content_legend_handles, save_figure, set_plot_style)


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
    return [start for _, start, _ in access_runs(content_history)]


def access_summary(results: Dict, window: int = PLOT_STEPS) -> Dict[str, float]:
    full_content = list(results['accessed_content_history'])
    content = full_content[-window:]
    window_start = max(0, len(full_content) - window)
    states = list(results['state_history'])[-window:]
    m = np.asarray(results['clarity_history'], dtype=float)[-window:]
    practice_runs = [end - start for name, start, end in access_runs(content) if name == PRACTICE_OBJECT]

    def fraction(mask) -> float:
        mask = np.asarray(mask, dtype=bool)
        return float(mask.mean()) if mask.size else 0.0

    in_bf = np.asarray([s == PRACTICE_STATE for s in states])
    in_mw = np.asarray([s == OFF_TASK_STATE for s in states])
    distractor = np.asarray([c in DISTRACTORS for c in content])
    return {
        'access_onsets': float(sum(start >= window_start for start in access_onsets(full_content))),
        'breath_access_percent': 100.0 * fraction([c == PRACTICE_OBJECT for c in content]),
        'breath_run_seconds': DEFAULT_DT * float(np.mean(practice_runs)) if practice_runs else 0.0,
        'distractor_access_mw_percent': 100.0 * fraction(distractor[in_mw]) if in_mw.any() else 0.0,
        'clarity_bf': float(m[in_bf].mean()) if in_bf.any() else 0.0,
        'clarity_session': float(m.mean()) if m.size else 0.0,
    }


def detected_ma_episodes(results: Dict, window: int = PLOT_STEPS) -> List[Tuple[int, int, int]]:
    """(start, first detection event, end) of each detected MA episode in the window, excluding left-cut MA."""
    states = results['state_history']
    events = results['detection_event_history']
    start_window = max(0, len(states) - window)
    episodes = []
    i = start_window
    while i < len(states):
        if states[i] != MA_STATE:
            i += 1
            continue
        start = i
        while i < len(states) and states[i] == MA_STATE:
            i += 1
        if start == start_window and start > 0 and states[start - 1] == MA_STATE:
            continue
        onset = next((t for t in range(start, i) if events[t]), None)
        if onset is not None:
            episodes.append((start, onset, i))
    return episodes


def detection_summary(results: Dict, window: int = PLOT_STEPS) -> Dict:
    """Median detection-to-RA latency over detected MA episodes with an observed direct RA exit."""
    states = results['state_history']
    latencies = [(i - onset) * DEFAULT_DT for _, onset, i in detected_ma_episodes(results, window)
                 if i < len(states) and states[i] == RETURN_STATE]
    return {'redirection_latency_seconds': float(np.median(latencies)) if latencies else None}


def _exit_effort(policy: ReturnPolicy, results: Dict, t: int) -> float:
    """KL[P(j | exit) || E] of the act an MA exit at t drew from, formed at t-1."""
    act = policy.probabilities(ReturnPolicy.precision(
        results['clarity_history'][t - 1], results['lapse_belief_history'][t - 1],
        results['detection_completed_history'][t - 1]))
    return sum(p * np.log(p / policy.habit[s]) for s, p in act.items())


def return_effort_summary(results: Dict, window: int = PLOT_STEPS) -> Dict:
    """Mean KL[P(j | exit) || E] (nats) over the window's MA exits, E the learned habit."""
    policy = ReturnPolicy(results['experience_level'], habit=results.get('return_habit'))
    window_start = max(0, len(results['state_history']) - window)
    efforts = []
    for transition in results['transitions']:
        t = transition['timestamp']
        if t >= window_start and t > 0 and transition['from'] == MA_STATE:
            efforts.append(_exit_effort(policy, results, t))
    return {
        'ma_exits': len(efforts),
        'return_effort_nats': float(np.mean(efforts)) if efforts else None,
    }


def practice_blocks(results: Dict, block: int = PLOT_STEPS) -> List[Dict]:
    """Fig. S2 measures per training block, then the final rollout block. The habit is replayed from the saved
    detection events; an MA exit at t uses the act formed at t-1 under the counts learned through t-2."""
    policy = ReturnPolicy(results['experience_level'])
    content = list(results['accessed_content_history'])
    clarity, belief = results['clarity_history'], results['lapse_belief_history']
    events = results['detection_event_history']
    training = int(results['training_steps'])
    total = len(results['state_history'])
    spans = [('fit', start, min(start + block, training)) for start in range(0, training, block)]
    if total > training:
        spans.append(('rollout', max(training, total - block), total))
    exits = {t['timestamp'] for t in results['transitions'] if t['from'] == MA_STATE and t['timestamp'] > 0}
    effort_at, habit_at = {}, {}
    for t in range(1, total + 1):
        if t in exits and t < total:
            effort_at[t] = _exit_effort(policy, results, t)
        if t - 1 < training and events[t - 1]:
            policy.learn(ReturnPolicy.precision(clarity[t - 1], belief[t - 1], True))
        habit_at[t] = policy.habit  # counts learned through step t-1
    blocks = []
    for phase, start, end in spans:
        breath = [c == PRACTICE_OBJECT for c in content[start:end]]
        runs = [e - s for name, s, e in access_runs(content[start:end]) if name == PRACTICE_OBJECT]
        efforts = [effort_at[t] for t in sorted(effort_at) if start <= t < end]
        blocks.append({
            'phase': phase, 'start': start, 'end': end,
            'breath_access_percent': 100.0 * float(np.mean(breath)) if breath else 0.0,
            'breath_run_seconds': DEFAULT_DT * float(np.mean(runs)) if runs else 0.0,
            'ma_exits': len(efforts),
            'return_effort_nats': float(np.mean(efforts)) if efforts else None,
            'habit': habit_at[end],
            'habit_return': habit_at[end][RETURN_STATE],
        })
    return blocks


def return_effort_training(results: Dict, block: int = PLOT_STEPS) -> Dict:
    """Return effort of the first and last training blocks of practice_blocks, and the final habit."""
    fit = [b for b in practice_blocks(results, block) if b['phase'] == 'fit']
    first, last = fit[0], fit[-1]
    return {
        'first_block_exits': first['ma_exits'], 'last_block_exits': last['ma_exits'],
        'first_block_nats': first['return_effort_nats'], 'last_block_nats': last['return_effort_nats'],
        'final_habit': last['habit'],
    }


PRACTICE_PANELS = (
    ('breath_run_seconds', 'Mean breath-access run (s)'),
    ('breath_access_percent', 'Breath-access occupancy (%)'),
    ('habit_return', 'Return habit $E(\\mathrm{RA})$'),
    ('return_effort_nats', 'Return effort at MA exits (nats)'),
)
PRACTICE_ROWS = ('Workspace access', 'Return policy')


def plot_practice_curves(blocks_by_phenotype: Dict[str, List[List[Dict]]], save_path: str) -> None:
    """Fig. S2: median (line) and range (band) over runs; the slot left of block 1 holds the configured habit."""
    set_plot_style()
    style = {'expert': dict(color='black', linestyle='-', marker='o', label='Expert'),
             'novice': dict(color='0.45', linestyle='--', marker='s', label='Novice')}
    any_run = next(iter(blocks_by_phenotype.values()))[0]
    n_fit = sum(b['phase'] == 'fit' for b in any_run)
    has_rollout = any(b['phase'] == 'rollout' for b in any_run)
    x = np.arange(1, n_fit + 1, dtype=float)
    x_roll = n_fit + 1.5
    with plt.rc_context({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5}):
        fig, axes = plt.subplots(2, 2, figsize=(5.95, 4.4), gridspec_kw={'hspace': 0.55, 'wspace': 0.32})
        for ax, (key, label), panel in zip(axes.ravel(), PRACTICE_PANELS, 'ABCD'):
            for phenotype, runs in blocks_by_phenotype.items():
                values = np.array([[np.nan if b[key] is None else b[key] for b in run] for run in runs],
                                  dtype=float)
                fit, roll = values[:, :n_fit], values[:, n_fit:]
                median = np.nanmedian(fit, axis=0)
                ax.fill_between(x, np.nanmin(fit, axis=0), np.nanmax(fit, axis=0),
                                color=style[phenotype]['color'], alpha=0.15, linewidth=0)
                ax.plot(x, median, markersize=3.5, linewidth=1.2, **style[phenotype])
                if has_rollout:
                    ax.errorbar([x_roll], [np.nanmedian(roll)], yerr=[[np.nanmedian(roll) - np.nanmin(roll)],
                                [np.nanmax(roll) - np.nanmedian(roll)]], color=style[phenotype]['color'],
                                marker=style[phenotype]['marker'], markersize=3.5, capsize=2, linestyle='none')
                if key == 'habit_return':
                    prior = ReturnPolicy(phenotype).configured_habit[RETURN_STATE]
                    ax.plot([0.0, 1.0], [prior, median[0]], color=style[phenotype]['color'], linewidth=0.8,
                            linestyle=':')
                    ax.plot([0.0], [prior], marker=style[phenotype]['marker'], markersize=3.5,
                            color=style[phenotype]['color'], markerfacecolor='white', linestyle='none')
            ticks, labels = [*x], [str(int(v)) for v in x]
            if has_rollout:
                ax.axvline(n_fit + 0.75, color='0.6', linestyle=':', linewidth=0.8)
                ticks, labels = [*ticks, x_roll], [*labels, 'Rollout\nwindow']
            if key == 'habit_return':
                ticks, labels = [0.0, *ticks], ['Prior', *labels]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xlim(-0.5, (x_roll if has_rollout else n_fit) + 0.5)
            ax.set_ylabel(label)
            ax.set_title(panel, loc='left', fontsize=8, fontweight='bold', pad=3)
            ax.grid(True, linestyle='--', alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel(f'Training block ({PLOT_STEPS * DEFAULT_DT:.0f} s each)')
        axes[0, 0].legend(fontsize=6.5, loc='upper left', frameon=False)
        for row, title in enumerate(PRACTICE_ROWS):
            fig.text(0.5, axes[row, 0].get_position().y1 + 0.045, title, ha='center', va='bottom', fontsize=8,
                     fontweight='bold')
        save_figure(fig, Path(save_path))
    plt.close(fig)


def _flag_end(held: Sequence[bool], t: int) -> int:
    """First step from t with D = 0, or len(held) when the flag is still set at the final step."""
    end = t
    while end < len(held) and held[end]:
        end += 1
    return end


def _run_end(content: Sequence[Optional[str]], t: int) -> int:
    """First step from t at which the content holding the slot at t no longer holds it."""
    end = t
    while end < len(content) and content[end] == content[t]:
        end += 1
    return end


def _flag_end_cause(results: Dict, end: int) -> str:
    """Why D cleared at step end: the belief (q < 1/2), clarity (m <= m_rest) or both."""
    by_belief = results['lapse_belief_history'][end] < 0.5
    by_clarity = results['clarity_history'][end] <= float(results['resting_clarity'])
    if not (by_belief or by_clarity):
        raise ValueError(f"Flag cleared at step {end} although psi(m_t, q_t) holds.")
    return 'both' if by_belief and by_clarity else 'belief' if by_belief else 'clarity'


def _detection_onsets(events: Sequence[bool], held: Sequence[bool], start: int) -> List[int]:
    """Detection event steps from start; each must begin a recognition interval (D_t = 1 at its own step)."""
    onsets = [t for t in range(start, len(events)) if events[t]]
    for t in onsets:
        if not held[t]:
            raise ValueError(f"Detection event at step {t} does not begin a recognition interval.")
    return onsets


def meta_awareness_detection_access(results: Dict, window: int = PLOT_STEPS) -> Dict[str, int]:
    """MA detection events in the window, and those with a distractor in the slot at the event step."""
    states, events = results['state_history'], results['detection_event_history']
    content = results['accessed_content_history']
    n = len(states)
    steps = [t for t in range(max(0, n - window), n) if events[t] and states[t] == MA_STATE]
    return {'detections': len(steps), 'distractor': sum(content[t] in DISTRACTORS for t in steps)}


def same_distractor_overlaps(results: Dict, lo: int, hi: int) -> List[Tuple[int, int]]:
    """(t, end) per detection event t in [lo, hi) with a distractor in the slot; over [t, end) it holds while D = 1."""
    content, events = results['accessed_content_history'], results['detection_event_history']
    held = [bool(x) for x in results['detection_completed_history']]
    return [(t, min(_flag_end(held, t), _run_end(content, t))) for t in range(lo, hi)
            if events[t] and held[t] and content[t] in DISTRACTORS]


# Event measures (Supplementary Section S3). The recognition offset f is the first step at or after an event
# with D = 0, or the recording end.

def recognition_to_content(results: Dict, target: str, window: int = PLOT_STEPS) -> Dict:
    """M1 (target = breath). Each detection event t in the window is 'present' (target in the slot at t), 'returned'
    (target first enters at u, t < u <= f), 'ended' (the flag ends first; by cause) or 'censored' (the recording
    ends with the flag set). `returns` keeps (t, u, f); `onsets` keeps (t, outcome), an ended outcome by its cause."""
    events, held = results['detection_event_history'], [bool(x) for x in results['detection_completed_history']]
    content = results['accessed_content_history']
    n = len(events)
    out = {'present': 0, 'returned': 0, 'ended': {'belief': 0, 'clarity': 0, 'both': 0}, 'censored': 0,
           'latency_seconds': [], 'returns': [], 'onsets': []}
    for t in _detection_onsets(events, held, max(0, n - window)):
        if content[t] == target:
            out['present'] += 1
            out['onsets'].append((t, 'present'))
            continue
        f = _flag_end(held, t)
        u = next((u for u in range(t + 1, min(f, n - 1) + 1) if content[u] == target), None)
        if u is not None:
            out['returned'] += 1
            out['latency_seconds'].append((u - t) * DEFAULT_DT)
            out['returns'].append((t, u, f))
            out['onsets'].append((t, 'returned'))
        elif f < n:
            cause = _flag_end_cause(results, f)
            out['ended'][cause] += 1
            out['onsets'].append((t, cause))
        else:
            out['censored'] += 1
            out['onsets'].append((t, 'censored'))
    return out


ONSET_OUTCOMES = ('present', 'returned', 'belief', 'clarity', 'both', 'censored')


def onset_outcomes(results: Dict, window: int = PLOT_STEPS) -> Dict:
    """Table S5: recognition onsets in the window by regime and M1 outcome, with each regime's exposure (episodes
    entered within the window, seconds)."""
    states = results['state_history']
    n = len(states)
    start = max(0, n - window)
    counts = {state: {outcome: 0 for outcome in ONSET_OUTCOMES} for state in STATES}
    for t, outcome in recognition_to_content(results, PRACTICE_OBJECT, window)['onsets']:
        counts[states[t]][outcome] += 1
    entries, seconds = {state: 0 for state in STATES}, {state: 0.0 for state in STATES}
    for t in range(start, n):
        seconds[states[t]] += DEFAULT_DT
        entries[states[t]] += t > 0 and states[t] != states[t - 1]
    return {'counts': counts, 'entries': entries, 'seconds': seconds}


def breath_reentry_to_offset(results: Dict, window: int = PLOT_STEPS) -> List[Dict]:
    """Per M1 breath re-entry u, the time from u to the offset f (censored if the flag is set at the final step),
    the offset cause and whether a distractor held the slot in [u, f)."""
    content = results['accessed_content_history']
    n = len(content)
    records = []
    for t, u, f in recognition_to_content(results, PRACTICE_OBJECT, window)['returns']:
        censored = f == n
        records.append({'onset': t, 'reentry': u, 'offset': f,
                        'seconds': None if censored else (f - u) * DEFAULT_DT,
                        'cause': 'censored' if censored else _flag_end_cause(results, f),
                        'recurred': any(content[j] in DISTRACTORS for j in range(u, f))})
    return records


def recognized_distraction_overlap(results: Dict, window: int = PLOT_STEPS) -> List[Optional[float]]:
    """M2: per event in the window with a distractor in the slot, the overlap (s) from t to the first of the offset f
    and the distractor's departure; None when both persist to the recording end."""
    events, held = results['detection_event_history'], [bool(x) for x in results['detection_completed_history']]
    content = results['accessed_content_history']
    n = len(events)
    overlaps = []
    for t in _detection_onsets(events, held, max(0, n - window)):
        if content[t] not in DISTRACTORS:
            continue
        end = min(_flag_end(held, t), _run_end(content, t))
        overlaps.append(None if end == n else (end - t) * DEFAULT_DT)
    return overlaps


def event_measure_summary(results: Dict, window: int = PLOT_STEPS) -> Dict:
    """Within-run M1-M3 (None without an eligible case). The M1 proportion has all onsets in the denominator and
    is not censoring-adjusted."""
    m1 = recognition_to_content(results, PRACTICE_OBJECT, window)
    onsets = m1['present'] + m1['returned'] + m1['censored'] + sum(m1['ended'].values())
    overlaps = recognized_distraction_overlap(results, window)
    observed = [s for s in overlaps if s is not None]
    share = recognized_distraction_share(results, window)
    median = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
    return {
        'recognition_onsets': onsets,
        'breath_returns': m1['returned'],
        'breath_return_proportion': m1['returned'] / onsets if onsets else None,
        'breath_return_latency_seconds': median(m1['latency_seconds']),
        'distractor_onsets': len(overlaps),
        'overlap_censored': len(overlaps) - len(observed),
        'overlap_seconds': median(observed),
        **{f'recognized_distraction_{key}': (k / total if total else None) for key, (k, total) in share.items()},
        'distraction_steps': share['all'][1],
    }


def recognized_distraction_share(results: Dict, window: int = PLOT_STEPS) -> Dict[str, Tuple[int, int]]:
    """M3: (distractor steps with D_t = 1, distractor steps) in the window, overall and by regime."""
    content, states = results['accessed_content_history'], results['state_history']
    held = results['detection_completed_history']
    n = len(content)
    counts = {'all': [0, 0], OFF_TASK_STATE: [0, 0], MA_STATE: [0, 0]}
    for t in range(max(0, n - window), n):
        if content[t] not in DISTRACTORS:
            continue
        for key in ('all', states[t]):
            if key in counts:
                counts[key][0] += bool(held[t])
                counts[key][1] += 1
    return {key: (k, total) for key, (k, total) in counts.items()}


def focal_distractor_detection(results: Dict, window: int = PLOT_STEPS) -> Optional[int]:
    """Fig. 6 event: the first MA detection event in the window with the same distractor at t-1, t and t+1."""
    content, states = results['accessed_content_history'], results['state_history']
    events, held = results['detection_event_history'], results['detection_completed_history']
    for t in range(max(1, len(content) - window), len(content) - 1):
        if (events[t] and held[t] and states[t] == MA_STATE and content[t] in DISTRACTORS
                and content[t - 1] == content[t] == content[t + 1]):
            return t
    return None


def anatomy_seed(pinned_seed: int, seeds: Sequence[int], load_pair, window: int = PLOT_STEPS) -> int:
    """Fig. 6 run: the pinned seed if both phenotypes have a focal-distractor detection, else the first such seed."""
    for seed in (pinned_seed, *sorted(s for s in seeds if s != pinned_seed)):
        if all(focal_distractor_detection(run, window) is not None for run in load_pair(seed)):
            return seed
    raise ValueError(f"No seed has a focal-distractor detection for both phenotypes in the final {window} steps.")


# Fig. 6 guides: (label, colour, line style, width).
ANATOMY_GUIDES = (('Recognition onset', '0.1', '-', 0.9),
                  ('First breath re-entry', THOUGHTSEED_COLORS[PRACTICE_OBJECT], '--', 1.0),
                  ('Recognition offset', '0.1', ':', 1.1))
ANATOMY_TRACKS = ('Regime (L1)', 'Workspace (L3)', 'Lapse known (L3)', 'Monitoring clarity\n$m_t$ (L3)',
                  'Lapse belief\n$q_t$ (L3)')


def _anatomy_column(axes, results, label, before_steps, after_steps):
    states = results['state_history']
    entry = focal_distractor_detection(results)
    if entry is None:
        raise ValueError(f"{label}: no focal-distractor detection in the final {PLOT_STEPS} steps.")
    lo, hi = entry - before_steps, entry + after_steps
    if lo < int(results['training_steps']):
        raise ValueError(f"{label}: the window would reach back before the frozen rollout.")
    held = [bool(x) for x in results['detection_completed_history']]
    offset = _flag_end(held, entry)
    reentry = next((u for t, u, _ in recognition_to_content(results, PRACTICE_OBJECT)['returns'] if t == entry), None)
    if reentry is None or offset > hi or hi > len(states):
        raise ValueError(f"{label}: the breath re-entry and recognition offset must fall inside the window.")
    (_, overlap_end), = same_distractor_overlaps(results, entry, entry + 1)
    seconds = (np.arange(lo, hi + 1) - entry) * DEFAULT_DT
    x = lambda step: (step - entry) * DEFAULT_DT  # noqa: E731
    regime_ax, content_ax, known_ax, clarity_ax, belief_ax = axes
    labelled = dict(fontsize=6, ha='left', va='center', zorder=7,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.6, alpha=0.9))

    for state, start, end in access_runs(states[lo:hi]):
        regime_ax.axvspan(x(lo + start), x(lo + end), ymin=0.15, ymax=0.85, color=STATE_BAND_COLORS[state],
                          linewidth=0, zorder=3)
        if x(lo + end) - x(lo + start) >= 1.2:
            regime_ax.text((x(lo + start) + x(lo + end)) / 2, 0.5, STATE_SHORT_NAMES[state], fontsize=6.5,
                           ha='center', va='center', zorder=7,
                           bbox=dict(facecolor=STATE_BAND_COLORS[state], edgecolor='none', pad=0.3))
    for name, start, end in access_runs(results['accessed_content_history'][lo:hi]):
        content_ax.axvspan(x(lo + start), x(lo + end), ymin=0.30, ymax=0.75, color=THOUGHTSEED_COLORS[name],
                           linewidth=0, zorder=3)
    content_ax.plot([0.0, x(overlap_end)], [0.88, 0.88], color='black', linewidth=2.0, solid_capstyle='butt',
                    zorder=6)
    content_ax.text(x(overlap_end) + 0.3, 0.88, f'Same-distractor overlap {x(overlap_end):.1f} s', **labelled)
    known_ax.axvspan(0.0, x(offset), ymin=0.30, ymax=0.70, color=DETECTION_COLOR, linewidth=0, zorder=3)
    known_ax.text(x(offset) + 0.3, 0.5, f'{x(offset):.1f} s', **labelled)
    for ax in axes[:3]:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.grid(False)
    regime_ax.set_title(label, loc='left', fontsize=8, pad=3)

    def trace(ax, values, reference, reference_label):
        values = np.asarray(values, dtype=float)[lo:hi]
        ax.step(seconds, np.append(values, values[-1]), where='post', color='0.1', linewidth=0.8, zorder=3)
        ax.axhline(reference, color='0.45', linestyle='--', linewidth=0.6, zorder=2)
        ax.text(seconds[-1] - 0.3, reference + 0.04, reference_label, fontsize=6.5, ha='right', va='bottom',
                color='0.3', zorder=4)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(axis='y', alpha=0.3)

    trace(clarity_ax, results['clarity_history'], float(results['resting_clarity']), r'$m_{\mathrm{rest}}$')
    trace(belief_ax, results['lapse_belief_history'], 0.5, '1/2')
    for (_, color, style, width), step in zip(ANATOMY_GUIDES, (entry, reentry, offset)):
        for ax in axes:
            ax.axvline(x(step), color=color, linestyle=style, linewidth=width, zorder=5)
    for ax in axes:
        ax.set_xlim(seconds[0], seconds[-1])


def plot_lapse_anatomy(novice_results, expert_results, save_path: str,
                       before_seconds: float = 8.0, after_seconds: float = 16.0) -> None:
    """Fig. 6: one focal-distractor detection per phenotype, aligned on recognition onset."""
    set_plot_style()
    before, after = int(round(before_seconds / DEFAULT_DT)), int(round(after_seconds / DEFAULT_DT))
    with plt.rc_context({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5}):
        fig, axes = plt.subplots(len(ANATOMY_TRACKS), 2, figsize=(5.95, 4.0), sharex=True, sharey='row',
                                 gridspec_kw={'height_ratios': [0.35, 0.6, 0.35, 1.0, 1.0], 'hspace': 0.12,
                                              'wspace': 0.08})
        _anatomy_column(axes[:, 0], novice_results, 'Novice', before, after)
        _anatomy_column(axes[:, 1], expert_results, 'Expert', before, after)
        for ax, name in zip(axes[:, 0], ANATOMY_TRACKS):
            if '$' in name:
                ax.set_ylabel(name)
            else:
                ax.set_ylabel(name, rotation=0, ha='right', va='center')
        for ax in axes[-1]:
            ax.set_xticks(np.arange(-before_seconds, after_seconds + 1, 4.0))
            ax.set_xlabel('Time from recognition onset (s)')
        handles = content_legend_handles(ACCESSIBLE_CONTENTS) + [
            Line2D([], [], color=color, linestyle=style, linewidth=width + 0.3, label=name)
            for name, color, style, width in ANATOMY_GUIDES]
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=6.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.92), columnspacing=1.0, handlelength=2.0)
        save_figure(fig, save_path)
    plt.close(fig)
