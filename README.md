# exemplar-abstraction-sims

Simulations accompanying the paper:

> **Exemplars in Disguise: Pure Exemplar Models Mimic Abstraction-First Learning**
> Zachary Nicholas Houghton & Vsevolod Kapatsinski (University of Oregon)

We show that the abstraction-first onset ordering reported by Jian & Manning (2026) for GPT-2 is reproduced by pure memorizer models with no class-level representations, no information flow between verbs, and no inductive bias toward generalization — provided sensitivity to individual observations is sufficiently low. The ordering reflects observation sensitivity, not learning strategy.

## Repository structure

```
exemplar-abstraction-sims/
├── scripts/                        # Simulation scripts
│   ├── zero-sensitivity-learner.R      # Zero-Sensitivity Learner (k → ∞ limit)
│   ├── variable-sensitivity-learner.R  # Variable-Sensitivity Learner (Dirichlet-Multinomial)
│   ├── hierarchical-bayesian-learner.R # Hierarchical Bayesian Learner
│   ├── zipfian-vsl.R                   # Zipfian VSL (unequal verb frequencies)
│   ├── plot_k_trajectories.R           # Trajectory visualization
│   ├── summarize_results.R             # Results summary
│   └── README.md                       # Full model documentation
├── data/                           # Simulation output (CSVs)
│   ├── grid_results_model1.csv         # ZSL results (108 combinations × 50 seeds)
│   ├── grid_results_model2.csv         # VSL results (540 combinations × 50 seeds)
│   ├── grid_results_model3.csv         # Zipfian VSL results (1620 combinations × 50 seeds)
│   └── k_trajectories_data.csv         # Trajectory data for visualization
└── writeup/                        # Paper (local only, not tracked)
    └── exemplars_in_disguise.qmd
```

## Quick start

```r
# Install dependencies (first time only)
install.packages(c("furrr", "progressr"))

# Run from the repo root
source("scripts/zero-sensitivity-learner.R")      # ~10 min with 6 workers
source("scripts/variable-sensitivity-learner.R")  # ~45 min with 6 workers
source("scripts/zipfian-vsl.R")                   # ~6 hr with 6 workers
```

Adjust `N_WORKERS` at the top of each script to match your machine. Results are written to `data/`.

## Models

| Script | Model | Key parameter |
|---|---|---|
| `zero-sensitivity-learner.R` | Deterministic interpolation ($k \to \infty$ limit) | $\alpha \in [0,1]$ |
| `variable-sensitivity-learner.R` | Dirichlet-Multinomial VSL | $k \in \{0.001, \ldots, 1.0\}$ |
| `hierarchical-bayesian-learner.R` | Hierarchical Bayesian pooling | $\gamma \in \{0.1, \ldots, 10000\}$ |
| `zipfian-vsl.R` | Zipfian VSL (unequal verb frequencies) | $k \in \{0.001, \ldots, 1.0\}$, Zipf $s \in \{0.5, 1.0, 1.5\}$ |

See `scripts/README.md` for full mathematical documentation of each model.
