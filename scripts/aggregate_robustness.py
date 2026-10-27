"""Write the five-run outputs (seeds 100-104): robustness.tex, event_measures.tex, robustness_macros.tex and Fig. S2.

Usage:
    python scripts/aggregate_robustness.py
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.defaults import DEFAULT_DT, MODEL_VERSION, NETWORKS, PLOT_STEPS, STATES
from viz.analysis_utils import compute_tail_statistics, validate_network_schema
from experiments.training_pipeline import ROBUSTNESS_SEEDS as SEEDS, result_path
from viz.ignition import (access_summary, breath_reentry_to_offset, detection_summary, event_measure_summary,
                          meta_awareness_detection_access, onset_outcomes, plot_practice_curves, practice_blocks,
                          return_effort_summary)
from viz.plotting_utils import STATE_SHORT_NAMES

OUT = ROOT / "figures" / "robustness.tex"
EVENTS = ROOT / "figures" / "event_measures.tex"
MACROS = ROOT / "figures" / "robustness_macros.tex"
FIGURE = ROOT / "figures" / "figS2_practice.pdf"

MEASURES = [
    ("Breath Focus dwell (s)", "dwell", "breath_focus", 1),
    ("Mind Wandering dwell (s)", "dwell", "mind_wandering", 1),
    ("RA$\\rightarrow$BF transition", "transition", ("redirect_attention", "breath_focus"), 3),
    ("MW$\\rightarrow$MA transition", "transition", ("mind_wandering", "meta_awareness"), 3),
    ("Breath access run (s)", "access", "breath_run_seconds", 2),
    ("Mean $m_t$ over the window", "access", "clarity_session", 3),
    ("Mean $m_t$ in Breath Focus", "access", "clarity_bf", 3),
    ("Detection-to-RA latency, within-run median (s)", "detection", "redirection_latency_seconds", 2),
    ("Return effort at MA exit (nats)", "effort", "return_effort_nats", 3),
    (r"Onsets with breath re-entry, $\hat p_{\mathrm b}$", "event", "breath_return_proportion", 2),
    (r"Recognition-to-breath latency, within-run median of $\ell_k$ (s)", "event", "breath_return_latency_seconds", 1),
    ("Same-distractor overlap, within-run median (s)", "event", "overlap_seconds", 1),
    (r"Distraction time with recognition, $\hat p_{\mathrm d}$", "event", "recognized_distraction_all", 2),
    (r"\quad in Mind Wandering", "event", "recognized_distraction_mind_wandering", 2),
    (r"\quad in Meta Awareness", "event", "recognized_distraction_meta_awareness", 2),
]

# Table S4 group headings, each placed before the measure it names.
MEASURE_GROUPS = {
    "Breath Focus dwell (s)": "Access, monitoring and regime dynamics",
    "Detection-to-RA latency, within-run median (s)": "Recognition and return",
}

# Main-text event table: (label, within-run summary, source, key, digits). Sources: "event" for
# event_measure_summary, "detection" for detection_summary.
EVENT_ROWS = [
    (r"Recognition onsets with breath re-entry before or at recognition offset, $\hat p_{\mathrm b}$",
     "proportion of all onsets", "event", "breath_return_proportion", 2),
    (r"Recognition-to-breath latency (s), $\ell_k$, among those re-entries", "median", "event",
     "breath_return_latency_seconds", 1),
    ("Detection-to-RA latency (s), qualifying MA episodes with a direct RA exit", "median", "detection",
     "redirection_latency_seconds", 1),
    ("Same-distractor recognition overlap (s), onsets with a distractor in the workspace", "median", "event",
     "overlap_seconds", 1),
]


def _load(phenotype: str, seed: int) -> dict:
    with result_path(phenotype, seed).open() as handle:
        run = json.load(handle)
    validate_network_schema(run, NETWORKS)
    return run


def _diffs_for_seed(seed: int) -> dict:
    expert = _load("expert", seed)
    novice = _load("novice", seed)
    e_tail = compute_tail_statistics(expert, STATES, PLOT_STEPS)
    n_tail = compute_tail_statistics(novice, STATES, PLOT_STEPS)
    e_events, n_events = event_measure_summary(expert, PLOT_STEPS), event_measure_summary(novice, PLOT_STEPS)

    out = {}
    for label, kind, key, _ in MEASURES:
        if kind == "dwell":
            e_val, n_val = (None if tail["dwell_times"][key] is None else DEFAULT_DT * tail["dwell_times"][key]
                            for tail in (e_tail, n_tail))
        elif kind == "transition":
            source, destination = key
            e_val = e_tail["transition_matrix"][source][destination]
            n_val = n_tail["transition_matrix"][source][destination]
        elif kind == "access":
            e_val = access_summary(expert, PLOT_STEPS)[key]
            n_val = access_summary(novice, PLOT_STEPS)[key]
        elif kind == "detection":
            e_val = detection_summary(expert, PLOT_STEPS)[key]
            n_val = detection_summary(novice, PLOT_STEPS)[key]
        elif kind == "effort":
            e_val = return_effort_summary(expert, PLOT_STEPS)[key]
            n_val = return_effort_summary(novice, PLOT_STEPS)[key]
        elif kind == "event":
            e_val, n_val = e_events[key], n_events[key]
        out[label] = _difference(e_val, n_val)
    return out


def _difference(e_val, n_val) -> float:
    """Expert minus novice; NaN if either is missing, 0 within roundoff."""
    if e_val is None or n_val is None:
        return np.nan
    tolerance = PLOT_STEPS * np.finfo(float).eps * max(abs(e_val), abs(n_val))
    return 0.0 if abs(e_val - n_val) <= tolerance else e_val - n_val


def _onset_outcome_table() -> list:
    """Table S5: recognition onsets by regime and M1 outcome, summed over the runs; causes by regime are printed."""
    causes = ("belief", "clarity", "both")
    outcomes = ("present", "returned", "ended", "censored")
    tables, cause_totals = {}, {cause: 0 for cause in causes}
    print("\nOnset outcomes by regime, summed over the runs, with regime exposure:")
    for phenotype in ("expert", "novice"):
        runs = [onset_outcomes(_load(phenotype, seed), PLOT_STEPS) for seed in SEEDS]
        tables[phenotype] = {}
        for state in STATES:
            raw = {o: sum(run["counts"][state][o] for run in runs) for o in ("present", "returned", *causes, "censored")}
            for cause in causes:
                cause_totals[cause] += raw[cause]
            tables[phenotype][state] = {"present": raw["present"], "returned": raw["returned"],
                                        "ended": sum(raw[c] for c in causes), "censored": raw["censored"]}
            print(f"  {phenotype:6s} {STATE_SHORT_NAMES[state]:3s} " + " ".join(f"{k}={v}" for k, v in raw.items())
                  + f" entries={sum(run['entries'][state] for run in runs)}"
                  + f" seconds={sum(run['seconds'][state] for run in runs):.0f}")
    if cause_totals["belief"] == cause_totals["both"] == 0:
        cause_note = r" Every onset that ended first did so through clarity ($m_t\le m_{\mathrm{rest}}$)."
    else:
        cause_note = (f" Of the onsets that ended first, {cause_totals['clarity']} ended through clarity,"
                      f" {cause_totals['belief']} through the belief and {cause_totals['both']} through both.")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Recognition onsets by outcome.} Final \StatsPlotSteps{} steps of training seeds 100--104,"
        r" with counts summed over the five runs. Every recognition onset is counted by the regime at its step and its"
        r" outcome in the definition of $\hat p_{\mathrm b}$ (Section~2.3 of the main text). The breath sensation was"
        r" already in the workspace at onset, the breath re-entered before or at recognition offset, recognition ended"
        r" first, or the recognition interval remained open at the window end without breath re-entry. The outcomes"
        r" are exclusive and sum to"
        r" the onsets $K$." + cause_note + r" Re-entries over $K$ give a pooled proportion, which differs from the"
        r" median of per-run proportions in Table~2 of the main text. The counts depend on the time spent in each regime"
        r" and are not detection rates.}",
        r"\small",
        r"\begin{tabular}{l c c c c c}",
        r"\hline",
        r"Regime & Present & Re-entry & Ended first & Open, no re-entry & $K$\\",
        r"\hline",
    ]
    for phenotype, rows in tables.items():
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{phenotype.title()}}}}}\\")
        for state, row in rows.items():
            lines.append(f"{STATE_SHORT_NAMES[state]} & " + " & ".join(str(row[o]) for o in outcomes)
                         + f" & {sum(row.values())}\\\\")
        totals = {o: sum(row[o] for row in rows.values()) for o in outcomes}
        lines.append("all & " + " & ".join(str(totals[o]) for o in outcomes) + f" & {sum(totals.values())}\\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\label{tab:onset_outcomes}", r"\end{table}"]
    return lines


def _reentry_offset_macros() -> list:
    """\\FiveRun...ReentryOffset macros: eligible and timed re-entries, median (range) of run medians, recurrences."""
    lines = []
    for phenotype in ("expert", "novice"):
        records = [breath_reentry_to_offset(_load(phenotype, seed), PLOT_STEPS) for seed in SEEDS]
        timed = [[r for r in run if r["seconds"] is not None] for run in records]
        medians = [float(np.median([r["seconds"] for r in run])) for run in timed if run]
        median, spread = ("--", "--") if not medians else (
            f"{np.median(medians):.1f}",
            f"{min(medians):.1f}--{max(medians):.1f}" + ("" if len(medians) == len(SEEDS) else f", $n={len(medians)}$"))
        pooled = [r for run in timed for r in run]
        prefix = f"FiveRun{phenotype.title()}ReentryOffset"
        lines += [
            rf"\newcommand{{\{prefix}Eligible}}{{{sum(map(len, records))}}}",
            rf"\newcommand{{\{prefix}Timed}}{{{len(pooled)}}}",
            rf"\newcommand{{\{prefix}Median}}{{{median}}}",
            rf"\newcommand{{\{prefix}Range}}{{{spread}}}",
            rf"\newcommand{{\{prefix}Recurred}}{{{sum(r['recurred'] for r in pooled)}}}",
        ]
    return lines


def _event_table() -> list:
    """Main-text table: per phenotype, the median (minimum--maximum) across runs of each within-run summary."""
    values = {phenotype: {key: [] for _, _, _, key, _ in EVENT_ROWS} for phenotype in ("expert", "novice")}
    for phenotype in values:
        for seed in SEEDS:
            run = _load(phenotype, seed)
            summaries = {"event": event_measure_summary(run, PLOT_STEPS), "detection": detection_summary(run, PLOT_STEPS)}
            for _, _, source, key, _ in EVENT_ROWS:
                values[phenotype][key].append(summaries[source][key])

    def cell(series, digits):
        present = [v for v in series if v is not None]
        if not present:
            return "--"
        text = f"{np.median(present):.{digits}f} ({min(present):.{digits}f}--{max(present):.{digits}f})"
        return text if len(present) == len(series) else f"{text}, $n={len(present)}$"

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{\textbf{Recognition and return across the five runs.} Each cell is the median of five within-run"
        r" summaries, one per training seed 100--104 over the evaluation window $\mathcal W$, with the range across"
        r" runs in parentheses. The proportion is a median of per-run proportions, not a pooled proportion. The two"
        r" latency rows are conditional on different eligible cases, named in each row. A summary without eligible"
        r" cases is unavailable. Definitions are in Section~\ref{sec:recognition}, eligibility in Supplementary"
        r" Section~S3, paired differences in Supplementary Table~S4 and event counts in Supplementary Table~S5.}",
        r"\label{tab:event_measures}",
        r"\small",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.40\textwidth} >{\raggedright\arraybackslash}p{0.18\textwidth} c c}",
        r"\hline",
        r"Measure & Within-run summary & Expert & Novice\\",
        r"\hline",
    ]
    for label, summary, _, key, digits in EVENT_ROWS:
        lines.append(f"{label} & {summary} & {cell(values['expert'][key], digits)} & "
                     f"{cell(values['novice'][key], digits)}\\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return lines


def _meta_awareness_detection_macros() -> list:
    """\\FiveRun macros: MA detection events pooled over the runs, and those with a distractor in the workspace."""
    lines = []
    for phenotype in ("expert", "novice"):
        runs = [meta_awareness_detection_access(_load(phenotype, seed), PLOT_STEPS) for seed in SEEDS]
        prefix = f"FiveRun{phenotype.title()}MetaAwarenessDetections"
        lines += [
            rf"\newcommand{{\{prefix}}}{{{sum(run['detections'] for run in runs)}}}",
            rf"\newcommand{{\{prefix}Distractor}}{{{sum(run['distractor'] for run in runs)}}}",
        ]
    return lines


PRACTICE_MEASURES = [
    ("BreathRun", "breath_run_seconds", 2),
    ("BreathAccess", "breath_access_percent", 1),
    ("HabitReturn", "habit_return", 3),
    ("ReturnEffort", "return_effort_nats", 3),
]


def _practice_curves() -> list:
    """Fig. S2 and the Up/Down \\FiveRun macros (runs whose last training block is above or below the first)."""
    blocks = {phenotype: [practice_blocks(_load(phenotype, seed)) for seed in SEEDS] for phenotype in ("expert", "novice")}
    plot_practice_curves(blocks, FIGURE)
    lines = []
    for phenotype, runs in blocks.items():
        fit = [[b for b in run if b["phase"] == "fit"] for run in runs]
        for name, key, digits in PRACTICE_MEASURES:
            first = np.array([np.nan if run[0][key] is None else run[0][key] for run in fit], dtype=float)
            last = np.array([np.nan if run[-1][key] is None else run[-1][key] for run in fit], dtype=float)
            paired = np.isfinite(first) & np.isfinite(last)
            prefix = f"FiveRun{phenotype.title()}{name}"
            print(f"{prefix}: first block {_fmt(np.nanmedian(first), digits)}, last block "
                  f"{_fmt(np.nanmedian(last), digits)}")
            lines += [
                rf"\newcommand{{\{prefix}Up}}{{{int(np.sum(last[paired] > first[paired]))}/{int(paired.sum())}}}",
                rf"\newcommand{{\{prefix}Down}}{{{int(np.sum(last[paired] < first[paired]))}/{int(paired.sum())}}}",
            ]
    return lines


def _fmt(value: float, digits: int) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    per_seed = {seed: _diffs_for_seed(seed) for seed in SEEDS}

    print(f"{'Measure':32s} {'median':>10s} {'min':>10s} {'max':>10s}  direction")
    rows = []
    for label, _, _, digits in MEASURES:
        if label in MEASURE_GROUPS:
            rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{MEASURE_GROUPS[label]}}}}}\\")
        values = np.array([per_seed[seed][label] for seed in SEEDS], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append(f"{label} & -- & 0/0\\\\")
            continue
        median = float(np.median(values))
        lo, hi = float(values.min()), float(values.max())
        same_direction = int(np.sum(np.sign(values) == np.sign(median))) if median != 0 else int(np.sum(values == 0))
        print(f"{label:32s} {median:10.3f} {lo:10.3f} {hi:10.3f}  {same_direction}/{len(values)}")

        def fmt(value: float, sign_digits: int) -> str:
            return f"${value:+.{sign_digits}f}$"

        rows.append(
            f"{label} & {fmt(median, digits)} ({fmt(lo, digits)}--{fmt(hi, digits)}) & {same_direction}/{len(values)}\\\\"
        )

    print("\nLaTeX table rows:\n")
    for row in rows:
        print(row)

    header = [
        "% Auto-generated by scripts/aggregate_robustness.py; do not edit by hand.",
        f"% Model version: {MODEL_VERSION}.",
        rf"\def\RobustnessModelVersion{{{MODEL_VERSION}}}",
    ]
    table = [
        *header,
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Five-run qualitative robustness check.} Paired expert-minus-novice differences over the final \StatsPlotSteps{} steps, given as the median (minimum--maximum) across training seeds 100--104. Positive dwell differences indicate longer expert mean dwell durations. Direction counts runs whose difference has the same sign as the median. When the median is zero, it counts zero differences. The latency and overlap rows compare within-run medians over each run's eligible events, with eligibility as in Table~2 of the main text and \S S3.}",
        r"\small",
        r"\begin{tabular}{p{0.56\textwidth} c c}",
        r"\hline",
        r"Measure & Paired difference & Direction\\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
        r"\label{tab:robustness}",
        r"\end{table}",
        *_onset_outcome_table(),
    ]
    macros = [
        *header,
        "% Detection events in MA pooled over the five plot windows, and those with a distractor in the workspace.",
        *_meta_awareness_detection_macros(),
        "% Practice curves (Fig. S2): runs in which the last training block is above or below the first.",
        *_practice_curves(),
        "% Time from first breath re-entry to recognition offset (Supplementary Section S4).",
        *_reentry_offset_macros(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(table) + "\n", encoding="utf-8")
    EVENTS.write_text("\n".join([*header[:2], *_event_table()]) + "\n", encoding="utf-8")
    MACROS.write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}, {EVENTS}, {MACROS} and {FIGURE}")


if __name__ == "__main__":
    main()
