"""Generate the manuscript figures from paired simulation results."""

from pathlib import Path

from viz.analysis_utils import (
    prepare_tail_data,
    plot_transitions,
    validate_network_schema,
)
from viz.convergence import plot_convergence
from viz.radar_plot import plot_comparison
from viz.hierarchy import plot_hierarchy_continuous
from viz.ignition import anatomy_seed, plot_lapse_anatomy
from experiments.training_pipeline import ROBUSTNESS_SEEDS, load_results
from config.defaults import STATES, NETWORKS, EVALUATION_STEPS, PLOT_STEPS

CURRENT_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = CURRENT_DIR / "figures"


def generate_plots(expert_results, novice_results) -> None:
    validate_network_schema(expert_results, NETWORKS)
    validate_network_schema(novice_results, NETWORKS)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    print("  - Processing plot window (final tail)...")
    expert_tail = prepare_tail_data(
        expert_results, STATES, NETWORKS, tail_steps=PLOT_STEPS
    )
    novice_tail = prepare_tail_data(
        novice_results, STATES, NETWORKS, tail_steps=PLOT_STEPS
    )
    print(f"    Plot window: {PLOT_STEPS} steps (final tail)")
    print(f"    Full run: {len(expert_results['state_history'])} steps")

    def spans_for(results):
        training_steps = int(results["training_steps"])
        return [
            {
                "start": training_steps,
                "end": training_steps + EVALUATION_STEPS - PLOT_STEPS,
                "label": "Frozen evaluation window",
                "color": "#c7d4f1",
                "alpha": 0.45,
            },
            {
                "start": training_steps + EVALUATION_STEPS - PLOT_STEPS,
                "end": training_steps + EVALUATION_STEPS,
                "label": "Frozen plot window",
                "color": "#f3d0c7",
                "alpha": 0.35,
            },
        ]

    print("\n--- Figures ---")
    print("  - FigS1_Convergence_Expert.pdf (fit, frozen evaluation, frozen plot window)")
    plot_convergence(
        expert_results,
        str(PLOT_DIR / "FigS1_Convergence_Expert.pdf"),
        highlight_spans=spans_for(expert_results),
    )
    print("  - FigS1_Convergence_Novice.pdf (fit, frozen evaluation, frozen plot window)")
    plot_convergence(
        novice_results,
        str(PLOT_DIR / "FigS1_Convergence_Novice.pdf"),
        highlight_spans=spans_for(novice_results),
    )

    print("  - fig4a.pdf")
    plot_comparison(novice_tail, expert_tail, str(PLOT_DIR / "fig4a.pdf"))

    print("  - fig4b.pdf")
    plot_transitions(novice_tail, expert_tail, str(PLOT_DIR / "fig4b.pdf"))

    print("  - fig5a.pdf")
    plot_hierarchy_continuous(novice_tail, str(PLOT_DIR / "fig5a.pdf"), "Novice")

    print("  - fig5b.pdf")
    plot_hierarchy_continuous(expert_tail, str(PLOT_DIR / "fig5b.pdf"), "Expert")

    pinned = expert_results["seed"]

    def load_pair(seed):
        if seed == pinned:
            return expert_results, novice_results
        pair = load_results(seed)
        for run in pair:
            validate_network_schema(run, NETWORKS)
        return pair

    seed = anatomy_seed(pinned, ROBUSTNESS_SEEDS, load_pair)
    print(f"  - fig6.pdf (one focal-distractor detection per phenotype, seed {seed})")
    anatomy_expert, anatomy_novice = load_pair(seed)
    plot_lapse_anatomy(anatomy_novice, anatomy_expert, str(PLOT_DIR / "fig6.pdf"))

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"All plots: {PLOT_DIR}/")
    print()
