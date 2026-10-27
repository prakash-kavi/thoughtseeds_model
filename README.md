# Dual-Process Computational Model

A three-layer generative model of focused-attention meditation. It simulates expert and novice phenotypes across the cycle of Breath Focus, Mind Wandering, Meta Awareness and Redirect Attention, and produces the figures, tables and statistics reported in the manuscript.

Each run per phenotype is a 12,000-step fit followed by an independent 4,000-step frozen rollout. Figures and statistics use the final 2,000 rollout steps.

---

## Quick Start

Install dependencies (`pyproject.toml`/`uv.lock`):

```bash
uv sync
```

or, without `uv`:

```bash
pip install -e .
```

Run the model for both phenotypes at the pinned seed and draw the figures:

```bash
python run_model.py run
```

Run the five robustness seeds (saves results only):

```bash
python run_model.py run --seed 100   # likewise 101, 102, 103, 104
```

Regenerate figures and manuscript tables from the saved results:

```bash
python run_model.py plot
python scripts/export_stats.py
python scripts/aggregate_robustness.py
```

The model runs on CPU only.

---

## Model

- **Layer 1** (`substrate.py`, `duration.py`): a switching multivariate Ornstein-Uhlenbeck process over seven large-scale networks (DMN, VAN, DAN, FPN, VIS, SOM, LIM), with four regimes (BF, MW, MA, RA) and a Gamma dwell law per regime.
- **Layer 2** (`thoughtseeds.py`): a switching Kalman filter that infers five thoughtseed activations and a belief over the regime from network activity.
- **Layer 3** (`workspace.py`, `habits.py`, `policy.py`): single-slot workspace access, monitoring clarity, the lapse belief, the recognition indicator and the return policy.
- Layers exchange signals only through the Markov blankets in `markov_blankets.py`. No layer receives the regime from another.

Thoughtseeds: `breath_sensation`, `pain_discomfort`, `pending_tasks`, `recognition_cue`, `intention_to_return`.

---

## Outputs

`run` writes one JSON file per phenotype and seed to `data/` (`training_results_{expert,novice}_seed<N>.json`) holding the full run histories and learned quantities. Saved results for seeds 100--104 are included.

Figures and tables are generated into `figures/`. `scripts/export_stats.py` writes `stats.tex` (pinned-seed values). `scripts/aggregate_robustness.py` writes, from seeds 100--104, `event_measures.tex` (main Table 2), `robustness.tex` (Supplementary Tables S4 and S5), `robustness_macros.tex` and Fig. S2. `run_model.py plot` writes the remaining figures:

| File | Content |
|---|---|
| `fig4a.pdf`, `fig4b.pdf` | Fig. 4: regime network profiles, transition proportions |
| `fig5a.pdf`, `fig5b.pdf` | Fig. 5: novice and expert hierarchical traces |
| `fig6.pdf` | Fig. 6: detection while a distractor remains in the workspace |
| `FigS1_Convergence_{Expert,Novice}.pdf` | Fig. S1: L2 surprisal and regime occupancy over the full run |
| `figS2_practice.pdf` | Fig. S2: practice curves over training, seeds 100--104 |

---

## Repository Layout

```text
.
+-- run_model.py                   # Entry point (run | plot)
+-- experiments/
|   +-- training_pipeline.py       # Pinned seed, fit and frozen rollout
+-- config/
|   +-- defaults.py                # Shared constants
|   +-- profiles.py                # Phenotype dwell, transition and attractor tables
+-- model/
|   +-- training_loop.py           # Per-step update order and learning
|   +-- phenotype.py               # Expert and novice phenotypes
|   +-- substrate.py               # Layer 1 network dynamics
|   +-- duration.py                # Gamma exit probability and broadcast stabilization
|   +-- thoughtseeds.py            # Layer 2 switching filter
|   +-- workspace.py               # Layer 3 access, monitoring, lapse belief, detection
|   +-- habits.py                  # Layer 3 learned access prior
|   +-- policy.py                  # Layer 3 return policy and habit
|   +-- markov_blankets.py         # Interfaces between layers
+-- utils/                         # Math helpers
+-- viz/                           # Outcome measures and figures
+-- scripts/                       # Statistics and table export
+-- data/                          # Saved runs
```

---

## Configuration

Model parameters are in `config/defaults.py` (shared constants, thoughtseed priors `THOUGHTSEED_STATE_PRIORS`, coupling `THETA_BASE`, rollout length) and `config/profiles.py` (phenotype network attractors, dwell means, transition probabilities and practice length). Many constants are derived at import from the thoughtseed priors.

---

## Reproducibility

The pinned figure seed is 103 (`SEED` in `experiments/training_pipeline.py`), and the frozen rollout uses `SEED + 1`. All random number generators are seeded, so runs are reproducible under the same configuration and dependency environment.

---

## Citation

Kavi, P. C., Friedman, D. A., & Patow, G. (2026). Thoughtseeds as Latent Causes: A Dual-Process Computational Phenomenology of Focused-Attention Meditation. arXiv:2607.14833. https://doi.org/10.48550/arXiv.2607.14833

The preprint describes an earlier version of the model. This repository implements the revised model of the journal article. Citation metadata is in `CITATION.cff`.

## License

Apache License 2.0 (`LICENSE`). Redistributions and derivative works must keep the attribution in `NOTICE`.

---

This repository builds on earlier Thoughtseeds codebases:

- https://github.com/prakash-kavi/thoughtseeds_vipassana
- https://github.com/prakash-kavi/aif_iwai2025_thoughtseeds
