# Simulation Scripts: Model Documentation

This document covers the mathematics, implementation details, and linking hypotheses for each simulation model, with worked numerical examples connecting each implementation step to the formula it computes. All four models are documented here.

---

## Shared components: data generation and evaluation

All models use the same procedure to (a) build verb token pools, (b) construct true verb distributions, and (c) evaluate onset criteria. This section describes each component in full.

### Background: what are we simulating?

Jian & Manning (2026) track how GPT-2's next-token distributions change over training for two verb classes: *to-Dative* verbs (e.g. *give*, *sell*, *award*; N = 35) and *Motion* verbs (e.g. *go*, *run*, *walk*; N = 36). They measure two onset events:

- **ob (between-class onset)**: the training step at which class-level structure first becomes statistically detectable — the model treats dative and motion verbs as systematically different.
- **ow (within-class onset)**: the training step at which within-class idiosyncratic differentiation first crosses an absolute divergence threshold — individual verbs start looking distinct from one another even within the same class.

J&M find **ob < ow**: class structure is detectable before item-level idiosyncrasy reaches threshold. They interpret this as evidence of an abstraction-first learning bias in GPT-2.

**Our counter-argument**: ob < ow is a structural consequence of *low sensitivity to individual observations*, not of abstraction-first learning. Any learner that requires many consistent observations before its estimate shifts will detect the stronger class-level signal before the noisier item-level signal. We demonstrate this with simulations.

---

### Token pool construction

Each verb has a set of "preferred" tokens partitioned into three groups:

| Group | Shared with | Linguistic analogue |
|---|---|---|
| **cross tokens** | All verbs in both classes | High-frequency function words, common syntactic frames |
| **within tokens** | All verbs in the same class only | Class-specific argument preferences |
| **idiosyncratic tokens** | Only this verb | Verb-specific collocates |

The key structural constraint is **item_overlap > class_overlap**: within-class sharing exceeds cross-class sharing. This is what makes class membership a real distributional distinction.

**Mathematics**: given `class_overlap`, `item_overlap`, and `n_preferred`:

$$n_\text{cross} = \text{round}(\texttt{class\_overlap} \times n_\text{preferred})$$
$$n_\text{within} = \text{round}((\texttt{item\_overlap} - \texttt{class\_overlap}) \times n_\text{preferred})$$
$$n_\text{idio} = n_\text{preferred} - n_\text{cross} - n_\text{within}$$

Token IDs are assigned contiguously. Cross tokens occupy IDs 1 through $n_\text{cross}$. Within-class tokens for each class occupy the next $n_\text{within}$ IDs. Each verb independently samples $n_\text{idio}$ idiosyncratic token IDs from the remaining vocabulary.

**Worked example** (n_preferred = 10, class_overlap = 0.3, item_overlap = 0.6):

```r
n_cross  <- round(0.3 * 10)          # 3 tokens
n_within <- round(0.3 * 10)          # 3 tokens per class
n_idio   <- 10 - n_cross - n_within  # 4 tokens

cross_tok    <- seq_len(n_cross)                         # IDs 1:3
within_A_tok <- n_cross + seq_len(n_within)              # IDs 4:6
within_B_tok <- n_cross + n_within + seq_len(n_within)   # IDs 7:9
idio_pool    <- (n_cross + 2 * n_within + 1):VOCAB_SIZE  # IDs 10:1000
```

A1 and A2 share cross + within-A tokens (6 of 10); A1 and B1 share only cross tokens (3 of 10), confirming item_overlap (0.6) > class_overlap (0.3).

---

### True distribution construction

Each verb's true distribution $P_v$ is what the learner would converge to given unlimited data. Preferred tokens receive high probability; all other tokens receive a low uniform background weight.

**Mathematics**: raw weights are:

$$w_t = \begin{cases} \text{LogNormal}(\mu_\text{log}, \sigma) & \text{if } t \in \text{preferred}(v) \\ 1.0 & \text{otherwise} \end{cases}$$

where $\mu_\text{log} = \log(\mu) - \frac{1}{2}\sigma^2$ ensures $\mathbb{E}[w_t] = \mu$ for preferred tokens. The distribution is then normalised: $P_v(t) = w_t / \sum_{t'} w_{t'}$.

The lognormal weighting creates heterogeneity within the preferred set — preferred tokens are not all equally preferred, just as in real language.

---

### Jensen-Shannon Divergence

We measure similarity between verb distributions using Jensen-Shannon Divergence (DJS). It is symmetric, bounded in $[0, \log 2]$, and zero if and only if the two distributions are identical.

**Mathematics**:

$$\text{DJS}(P, Q) = H\!\left(\frac{P + Q}{2}\right) - \frac{1}{2}\bigl[H(P) + H(Q)\bigr]$$

where $H(P) = -\sum_t P(t) \log P(t)$ is Shannon entropy.

The `pairwise_jsd()` function computes this for all 71 × 71 verb pairs simultaneously using matrix operations:

```r
pairwise_jsd <- function(P) {
  N   <- nrow(P)
  eps <- 1e-30
  H   <- -rowSums(P * log(P + eps))
  jsd_mat <- matrix(0.0, N, N)
  for (i in seq_len(N - 1L)) {
    j_idx <- (i + 1L):N
    M     <- 0.5 * (matrix(P[i, ], nrow = length(j_idx), ncol = ncol(P),
                            byrow = TRUE) + P[j_idx, , drop = FALSE])
    H_M   <- -rowSums(M * log(M + eps))
    d     <- pmax(0.0, H_M - 0.5 * (H[i] + H[j_idx]))
    jsd_mat[i, j_idx] <- d
    jsd_mat[j_idx, i] <- d
  }
  jsd_mat
}
```

---

### Onset criteria

#### ob: between-class onset

ob fires when the model has learned to treat the two verb classes as systematically different — not just when two specific verbs happen to diverge, but when the between > within pattern holds reliably across verbs.

For each verb $v$, we compare:
- **within**: DJS distances from $v$ to all other same-class verbs
- **between**: DJS distances from $v$ to all cross-class verbs

A one-tailed Mann-Whitney test asks whether between-class distances are stochastically greater than within-class distances. ob fires when ≥ 10% of all verbs pass this test (p < 0.001).

**Critical property — ob is a RELATIVE test**: it asks whether between > within, not whether either is large in absolute terms. ob can fire even when all DJS values are tiny, as long as the structural between > within asymmetry is present. This is key to why ob fires before ow under low-sensitivity conditions.

#### ow: within-class onset

ow fires when within-class idiosyncrasy is detectable — individual verbs within the same class have started to diverge from one another. This is measured as the mean DJS across all same-class verb pairs, with ow firing when this mean exceeds 0.01 for 3 consecutive checkpoints.

**Critical property — ow is an ABSOLUTE test**: it asks whether within-class DJS reaches 0.01, regardless of what between-class DJS is doing. The sustained-window requirement (SUSTAIN = 3 steps) prevents a single noisy spike from triggering ow prematurely. ow fires at the *start* of the sustained window, not the end.

---

## Model 1: Zero-Sensitivity Learner (`zero-sensitivity-learner.R`)

### Theoretical claim

If a learner has zero sensitivity to individual observations — if every new token observation moves the learner's estimate by an infinitesimal amount — then the learner will always show ob < ow.

This is the theoretical extreme case. It serves two purposes: (1) it shows the predicted result under the strongest version of the low-sensitivity hypothesis, and (2) it provides a clean analytical baseline that the more realistic models should converge to as sensitivity decreases.

### Mathematical formulation

At learning progress $\alpha \in [0, 1]$, each verb's estimated distribution is:

$$\hat{P}_v(\alpha) = (1 - \alpha) \cdot \text{Uniform} + \alpha \cdot P_v$$

- At $\alpha = 0$: all verbs look uniform (random-initialisation state)
- At $\alpha = 1$: each verb shows its true distributional profile
- At intermediate $\alpha$: a convex mixture

**Why does this correspond to zero sensitivity?** A learner with sensitivity controlled by $k$ (Model 2) has expected distribution:

$$\mathbb{E}[\hat{P}_v(n_\text{obs})] = \frac{n_\text{obs}}{n_\text{obs} + kV} \cdot P_v + \frac{kV}{n_\text{obs} + kV} \cdot \text{Uniform}$$

This is exactly the ZSL formula with $\alpha(n_\text{obs}) = n_\text{obs} / (n_\text{obs} + kV)$. As $k \to \infty$, the variance of the stochastic estimate around this expectation vanishes, and Model 2 collapses to this deterministic trajectory. **The ZSL is the $k \to \infty$ limit of the VSL.**

### Why ob < ow is guaranteed

**Step 1**: At any $\alpha > 0$, DJS between two verbs is proportional to $\alpha^2$ times the DJS between their true distributions. Because item_overlap > class_overlap, same-class verbs share more preferred tokens than cross-class verbs, so DJS(same class) < DJS(different class) by construction. The relative ob criterion detects this structural asymmetry.

**Step 2**: At very small $\alpha$, all DJS values are tiny (proportional to $\alpha^2$). The ow threshold requires mean within-class DJS to reach 0.01, which requires $\alpha$ to be large enough that $\alpha^2 \cdot \text{DJS}_\text{true, within} \geq 0.01$. Since the relative ob test has no minimum magnitude requirement, ob fires at a much smaller $\alpha$.

### Parameter grid

- $\mu \in \{10, 30, 60, 100\}$, $\sigma \in \{0.5, 1.0, 1.5\}$
- item_overlap $\in \{0.5, 0.6, 0.7\}$, class_overlap $\in \{0.2, 0.3, 0.4\}$ (class_overlap < item_overlap)
- 50 seeds per combination → 108 valid combinations × 50 seeds
- Output: `data/grid_results_model1.csv`

---

## Model 2: Variable-Sensitivity Learner (`variable-sensitivity-learner.R`)

### Theoretical claim

The ob < ow vs. ow < ob ordering is controlled entirely by $k$ — the smoothing parameter that determines how sensitive the learner is to individual observations. At high $k$, the learner is insensitive (like Model 1) and produces ob < ow. At low $k$, idiosyncratic early observations create within-class divergence before class structure is detectable, and the learner produces ow < ob.

### Mathematical formulation

The VSL is a Dirichlet-Multinomial conjugate Bayesian learner. The prior is a symmetric Dirichlet($k, k, \ldots, k$), encoding $k$ pseudo-observations of each token type. After observing $n_\text{obs}$ tokens for verb $v$, the posterior mean is:

$$\hat{P}_v(t) = \frac{\text{count}(t \mid v,\, n_\text{obs}) + k}{n_\text{obs} + k \cdot V}$$

**$k$ as inverse learning rate**: adding one new observation of token $t$ changes the estimate by approximately:

$$\Delta\hat{P}_v(t) \approx \frac{1 - \hat{P}_v(t)}{n_\text{obs} + kV}$$

The magnitude is inversely proportional to $n_\text{obs} + kV$. At large $k$, each observation barely moves the estimate. This is exactly analogous to a small learning rate in gradient descent.

### How the learner encounters tokens

At each observation, every verb $v$ receives one token — a vocabulary item drawn from $P_v$ — and that item's running count is incremented by 1. $\hat{P}_v$ can be evaluated at any point via the formula above; the simulation snapshots the learner's state at 20 log-spaced values of $n_\text{obs}$ from 1 to 5,000.

### Why low k produces ow < ob

At low $k$, a verb's first few observations strongly spike its estimated distribution around the specific tokens that happened to be observed. Two same-class verbs that observed *different* idiosyncratic tokens will look highly dissimilar — within-class DJS rises quickly. But this idiosyncratic spiking is random and the ob criterion (which requires between > within *reliably across all verbs*) has no signal to detect yet. ob fires only later, once sufficient data has accumulated for the shared class-level tokens to dominate.

### Why high k produces ob < ow

At high $k$, all distributions remain near-uniform for many observations. The structural asymmetry (within-class token sharing > cross-class token sharing) means that whenever DJS values are non-zero, between-class DJS is slightly larger than within-class DJS. The relative ob criterion detects this asymmetry even at tiny DJS values; the absolute ow criterion must wait for within-class DJS to reach 0.01.

### Parameter grid

Same structural parameters as Model 1, plus $k \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$ → 432 valid combinations × 50 seeds.
Output: `data/grid_results_model2.csv`.

---

## Hierarchical Bayesian Learner (`hierarchical-bayesian-learner.R`)

### Theoretical claim

The VSL's add-$k$ prior is uninformative: it smooths each verb toward the *uniform* distribution regardless of what other verbs look like. But in a neural network, parameters are shared across verbs that appear in similar contexts — observing *give* updates representations also used by *sell* and *award*. The Hierarchical Bayesian Learner captures this cross-verb sharing by replacing the uniform prior with a **class prototype**: each verb's estimate is smoothed toward the pooled distribution of all verbs in the same class.

The key theoretical question: does hierarchical pooling produce ob < ow across the full range of pooling strength, or only when pooling is strong enough to suppress idiosyncratic early observations?

### Mathematical formulation

Estimation proceeds in two steps at each $n_\text{obs}$ checkpoint.

**Step 1 — Class prototype** (pooled from all $N_\text{class}$ verbs in the same class):

$$\hat{P}_\text{class}(t) = \frac{\sum_{v \in \text{class}} \text{count}(t \mid v,\, n_\text{obs}) + k_0}{N_\text{class} \cdot n_\text{obs} + k_0 \cdot V}$$

The prototype has $N_\text{class}$ times more observations than any individual verb, so it is stable even when individual estimates are noisy.

**Step 2 — Hierarchical verb estimate**:

$$\hat{P}_v(t) = \frac{\text{count}(t \mid v,\, n_\text{obs}) + k_0 + \gamma \cdot \hat{P}_\text{class}(t)}{n_\text{obs} + k_0 \cdot V + \gamma}$$

$\gamma$ (gamma) is the pooling strength — how strongly each verb is pulled toward its class prototype:
- $\gamma \approx 0$: no pooling; each verb relies on its own observations only (equivalent to the VSL with $k = k_0$)
- $\gamma$ large: full pooling; each verb's estimate converges to the class prototype

$k_0 = 0.001$ is a weak flat floor that prevents division by zero and is not the primary smoothing parameter.

**Bayesian interpretation**: this is an empirical Bayes Dirichlet-Multinomial model. The prior over verb distributions is Dirichlet($k_0 + \gamma \hat{P}_\text{class}$), centered on the class prototype with total concentration $\gamma$. "Empirical Bayes" because the prior parameters (the prototype) are estimated from the data rather than fixed in advance.

**What $\gamma$ captures**: the functional effect of shared neural parameters. In a transformer, observing *give* updates weights also used by *sell* and *award* because they occur in similar contexts. $\gamma$ approximates this: information about one verb partially informs estimates for all verbs in the same class. The important caveat is that the prototype is computed from individual verb observations at each step — it is not given a priori. This is an **exemplar-driven** model that computes class-level statistics as an intermediate step; abstractions emerge from exemplars, not before.

### Why $\gamma$ controls ob/ow ordering

At large $\gamma$, each verb immediately looks like its class prototype: between-class DJS is large (prototypes differ), within-class DJS is small (verbs within a class resemble their shared prototype). The ob criterion detects this reliably and fires early; ow must wait for individual verbs to diverge from the prototype, which requires enough observations to overcome the prototype's pull. Result: ob < ow.

At small $\gamma$, the prototype exerts little pull and behavior approaches that of the VSL with $k = k_0 = 0.001$ — the low-sensitivity regime where idiosyncratic early observations spike within-class DJS before between-class structure is detectable. Result: ow < ob.

### Parameter grid

Same structural parameters as the VSL, plus $\gamma \in \{0.1, 1, 10, 100, 1000, 10000\}$ → 648 valid combinations × 50 seeds.
Output: `data/grid_results_model3_hierarchical.csv`.

---

## Zipfian VSL (`zipfian-vsl.R`)

### Theoretical claim

The VSL allocates exactly equal observations to every verb at each checkpoint. Real language is Zipfian: a small number of verbs are encountered extremely frequently, while most verbs are rare. The Zipfian VSL asks whether frequency inequality changes the ob/ow ordering — specifically, whether rare verbs (which have accumulated few observations and therefore behave like low-$k$ learners) destabilize the ob < ow result seen at high $k$ in the uniform VSL.

**Key prediction**: higher $k$ is required to produce ob < ow under Zipfian sampling compared to uniform sampling, because rare verbs generate noisy, idiosyncratic estimates that contribute within-class DJS even when frequent verbs have converged to stable class-consistent estimates. The transition from ow < ob to ob < ow should shift rightward in $k$ as the Zipf exponent $s$ increases.

### Mathematical formulation

Observation allocation follows a Zipfian distribution. Each verb is assigned a random rank $r \in \{1, \ldots, 71\}$, and its probability of being observed at any draw is:

$$P(\text{rank } r) \propto \frac{1}{r^s}$$

where $s$ is the Zipf exponent ($s = 1$ matches typical English frequency distributions; $s = 0.5$ is flatter; $s = 1.5$ is more skewed). Ranks are randomly permuted across seeds so that results average over which specific verbs happen to be frequent or rare.

At each total-token checkpoint $N_\text{total}$, verb $v$ has accumulated $n_v$ observations (not equal across verbs). The add-$k$ estimate is:

$$\hat{P}_v(t) = \frac{\text{count}(t \mid v,\, n_v) + k}{n_v + k \cdot V}$$

This is the same formula as the VSL, but $n_v$ varies across verbs at each checkpoint. The checkpoint axis is $N_\text{total}$ (total tokens across all verbs); $N_\text{total} / N_\text{verbs}$ gives the mean observations per verb, directly comparable to the VSL's $n_\text{obs}$ axis.

**Calibration**: `MAX_NOBS_TOTAL = N_VERBS × 5000`, so at the final checkpoint the mean observations per verb equals 5,000, matching the VSL.

### Implementation detail: efficient checkpoint accumulation

Rather than re-scanning verb assignments up to each checkpoint, the script makes a single O(`MAX_NOBS_TOTAL`) pass, incrementing per-verb counts and recording snapshots when each checkpoint value is reached. This avoids repeatedly calling `tabulate(verb_draws[1:n_total], ...)` and keeps runtime feasible for the full 1,620-combination grid.

### Parameter grid

Same structural parameters and $k$ values as the VSL, plus $s \in \{0.5, 1.0, 1.5\}$ → 1,620 valid combinations × 50 seeds.
Output: `data/grid_results_model3.csv`.

---

## Running the full grid searches

```r
install.packages(c("furrr", "progressr"))

source("scripts/zero-sensitivity-learner.R")        # ~10 min with 6 workers
source("scripts/variable-sensitivity-learner.R")    # ~45 min with 6 workers
source("scripts/hierarchical-bayesian-learner.R")   # ~2 hr with 6 workers
source("scripts/model3_zipfian.R")                  # ~6 hr with 6 workers
```

Adjust `N_WORKERS` at the top of each script (`parallel::detectCores() - 1` is a safe default).

### Output format

| File | Key columns |
|---|---|
| `data/grid_results_model1.csv` | `frac_ob_lt_ow`, `mean_ob_nobs`, `mean_ow_nobs`, `mean_log_ratio` |
| `data/grid_results_model2.csv` | `add_k`, `frac_ob_lt_ow`, `frac_ow_lt_ob` |
| `data/grid_results_model3_hierarchical.csv` | `gamma`, `frac_ob_lt_ow`, `frac_ow_lt_ob` |
| `data/grid_results_model3.csv` | `add_k`, `zipf_s`, `frac_ob_lt_ow`, `frac_ow_lt_ob` |

Each row is one parameter combination. `frac_ob_lt_ow` is the fraction of seeds (out of 50) where ob fired before ow.

---

## Summary of linking hypotheses

| Claim | Mathematical statement | Implemented as |
|---|---|---|
| Zero-Sensitivity Learner is the $k\to\infty$ limit of the VSL | $\mathbb{E}[\hat{P}_v(n)] = \alpha(n)\cdot P_v + (1-\alpha)\cdot\text{Uniform}$ | Equation (1) matches equation (2) expected value |
| $k$ controls sensitivity to observations | $\Delta\hat{P}(t) \approx 1/(n+kV)$ per observation | Denominator of the Dirichlet posterior mean in `variable-sensitivity-learner.R` |
| ob is a relative test; ow is absolute | ob: Mann-Whitney; ow: threshold 0.01 | `wilcox.test()` vs `DJS_THRESH` constant |
| item_overlap > class_overlap ensures class signal | DJS(same-class) < DJS(cross-class) by construction | `GRID` filters `class_overlap < item_overlap` |
| $\gamma = 0$ Hierarchical Bayesian Learner collapses to VSL with $k = k_0$ | $\hat{P}_v = (\text{count} + k_0) / (n + k_0 V)$ when $\gamma = 0$ | `denom = n_obs + K_0 * VOCAB_SIZE + gamma` with `gamma = 0` |
| Zipfian rare verbs behave like low-$k$ VSL learners | $\hat{P}_v$ with small $n_v$ = noisy, observation-driven estimate | `n_v` varies across verbs; per-verb formula identical to VSL |
