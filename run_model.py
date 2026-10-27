"""Entry point: fit, frozen rollout and plot per phenotype.

Usage:
    python run_model.py run [--seed N]
    python run_model.py plot [--seed N]
"""

import argparse

from experiments.training_pipeline import SEED, run_training_and_simulation, load_results
from viz.report import generate_plots


def main():
    parser = argparse.ArgumentParser(
        description="Three-layer focused-attention meditation model: fit -> frozen rollout -> plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run             - Fit then run a frozen evaluation rollout; save and plot
  plot            - Generate plots from previously saved results
Structure: phenotype practice steps (fit) -> EVALUATION_STEPS (fresh frozen rollout).
Figures are only regenerated for the default seed; other runs save
JSON results and skip plotting so the figures stay at the pinned seed.
        """,
    )

    parser.add_argument(
        "command",
        type=str,
        choices=["run", "plot"],
        help='"run" for fit+frozen rollout+plot or "plot" for visualization only',
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"fit seed (rollout uses seed+1); default {SEED}",
    )

    args = parser.parse_args()

    if args.command == "run":
        expert_results, novice_results = run_training_and_simulation(seed=args.seed)
        if args.seed == SEED:
            generate_plots(expert_results, novice_results)
        else:
            print("Skipping plot generation (non-default seed); "
                  "results JSONs saved for aggregation.")
    elif args.command == "plot":
        expert_results, novice_results = load_results(seed=args.seed)
        if expert_results is None:
            return
        generate_plots(expert_results, novice_results)


if __name__ == "__main__":
    main()
