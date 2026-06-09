# =============================================================================
# Model 2: Add-k smoothing (corpus-size simulation)
#
# This model simulates a learner who accumulates actual token observations.
# At each corpus size n_obs, the estimated distribution is:
#
#   P̂_verb(t) = (count(t | verb, n_obs) + k) / (n_obs + k × V)
#
# where count(t | verb, n_obs) = the number of times token t was observed
# in the context of this verb across n_obs total observations.
#
# The k term is the ADD-K SMOOTHING PARAMETER. It adds k pseudo-observations
# of every token type, acting as a uniform (flat, uninformative) prior over
# the vocabulary. This is mathematically equivalent to a Bayesian model with
# a Dirichlet(k) prior: the posterior mean after observing token counts is
# exactly this formula.
#
# k CONTROLS SENSITIVITY TO INDIVIDUAL OBSERVATIONS:
#   Low k  (e.g. 0.001): k × V = 1 pseudo-count → real observations dominate
#       immediately → verb distributions become idiosyncratically shaped by
#       the specific tokens observed → within-class DJS rises fast (ow fires
#       before ob).
#   High k (e.g. 0.5): k × V = 500 pseudo-counts → the uniform prior swamps
#       individual observations → distributions stay near-uniform for longer →
#       only the consistent class-level signal accumulates (ob fires before ow).
#
# At k → ∞, Model 2 collapses to Model 1 (the deterministic interpolation).
# k is like the INVERSE OF A LEARNING RATE: high k = small effective learning
# rate = insensitive to individual observations.
#
# KEY RESULT: the ob < ow vs. ow < ob ordering is controlled entirely by k.
# The same model, same data structure, same onset criteria — different k
# produces both orderings. No abstraction mechanism is needed.
#
# DATA GENERATION: identical to Model 1.
# ESTIMATION: differs from Model 1 — uses stochastic token counts + add-k.
# =============================================================================

library(furrr)      # parallel map functions (future_map_dfr)
library(progressr)  # progress bar support

# =============================================================================
# SECTION 1: Constants
# =============================================================================

VOCAB_SIZE  <- 1000L  # Total vocabulary size (tokens 1 to 1000).

N_PREFERRED <- 50L    # Preferred tokens per verb (~5% of vocabulary).

N_A         <- 35L    # Dative verbs (Jian & Manning Appendix A).
N_B         <- 36L    # Motion verbs (Jian & Manning Appendix A).

N_SEEDS     <- 50L    # Random seeds per parameter combination.
N_WORKERS   <- 6L     # Parallel CPU cores to use.

BG_WEIGHT   <- 1.0    # Raw weight for non-preferred (background) tokens.
                      # All background tokens are equally probable before
                      # normalisation.

MAX_NOBS    <- 5000L  # Maximum corpus size simulated (tokens per verb).
                      # All interesting onset behaviour occurs well below this.

# Corpus-size grid: 20 log-spaced values from 1 to MAX_NOBS.
# We use LOG spacing because onsets typically occur at small n_obs values.
# unique() removes any duplicates that arise from rounding.
N_OBS_GRID  <- unique(round(exp(seq(log(1), log(MAX_NOBS), length.out = 20L))))

# Onset detection thresholds (same as Model 1)
P_THRESH   <- 0.001  # Mann-Whitney p-value threshold (ob criterion).
CLASS_FRAC <- 0.10   # Fraction of verbs that must pass Mann-Whitney for ob.
DJS_THRESH <- 0.01   # Absolute within-class DJS threshold for ow.
SUSTAIN    <- 3L     # Consecutive steps ow threshold must hold before declaring ow.

# =============================================================================
# SECTION 2: Parameter grid
# All combinations of structural parameters are tested. The key additional
# parameter vs. Model 1 is add_k — the smoothing strength.
# =============================================================================

GRID <- expand.grid(
  mu            = c(10, 30, 60, 100),  # Mean lognormal weight on preferred tokens.
  sigma         = c(0.5, 1.0, 1.5),   # Spread of lognormal weights.
  item_overlap  = c(0.5, 0.6, 0.7),   # Fraction of preferred tokens shared within class.
  class_overlap = c(0.2, 0.3, 0.4),   # Fraction of preferred tokens shared across classes.

  add_k = c(0.001, 0.01, 0.1, 0.5, 1.0),  # THE KEY VARIABLE. Smoothing strength.
                                       # Effective pseudo-counts = add_k × VOCAB_SIZE:
                                       #   0.001 →   1 pseudo-count  (near-zero smoothing)
                                       #   0.01  →  10 pseudo-counts
                                       #   0.1   → 100 pseudo-counts
                                       #   0.5   → 500 pseudo-counts (heavy smoothing)
                                       #   1.0   → 1000 pseudo-counts
                                       # Low add_k → ow < ob; high add_k → ob < ow.

  stringsAsFactors = FALSE
)

# Remove combinations where class_overlap >= item_overlap.
# The ob criterion requires within-class similarity to exceed cross-class
# similarity, which only holds when class_overlap < item_overlap.
GRID <- GRID[GRID$class_overlap < GRID$item_overlap, ]

# =============================================================================
# SECTION 3: Pairwise Jensen-Shannon Divergence
# Identical to Model 1. See model1_monotone.R for full explanation.
# =============================================================================

pairwise_jsd <- function(P) {
  N   <- nrow(P)    # Number of verbs.
  eps <- 1e-30      # Numerical stability: prevents log(0).

  # Per-verb Shannon entropy: H(P_i) = -sum_t P_i(t) * log(P_i(t))
  H <- -rowSums(P * log(P + eps))

  jsd_mat <- matrix(0.0, N, N)  # N × N output matrix, initialised to 0.

  for (i in seq_len(N - 1L)) {
    j_idx <- (i + 1L):N   # Indices of all verbs after verb i.

    # Mixture distribution M_ij = 0.5*(P_i + P_j) for all j > i simultaneously.
    # matrix(..., byrow = TRUE) tiles row i into a (N-i) × V matrix.
    M <- 0.5 * (matrix(P[i, ], nrow = length(j_idx), ncol = VOCAB_SIZE, byrow = TRUE)
                + P[j_idx, , drop = FALSE])

    # Entropy of each mixture.
    H_M <- -rowSums(M * log(M + eps))

    # DJS = H(mixture) - 0.5*(H(P_i) + H(P_j)), clamped to non-negative.
    d <- pmax(0.0, H_M - 0.5 * (H[i] + H[j_idx]))

    # Fill both upper and lower triangles (DJS is symmetric).
    jsd_mat[i, j_idx] <- d
    jsd_mat[j_idx, i] <- d
  }
  jsd_mat
}

# =============================================================================
# SECTION 4: Single simulation run
#
# The data-generation steps (token pool, true distributions) are identical
# to Model 1. The estimation step is new: instead of interpolating the true
# distribution, we sample tokens and apply add-k smoothing.
# =============================================================================

run_one <- function(mu, sigma, n_preferred, item_overlap, class_overlap, add_k, seed) {

  set.seed(seed)  # Fix RNG for reproducibility across seeds.

  # ---------------------------------------------------------------------------
  # Step 1: Build the token pool (identical to Model 1)
  # ---------------------------------------------------------------------------

  # How many preferred tokens fall in each category.
  n_cross  <- round(class_overlap * n_preferred)           # Shared by both classes.
  n_within <- round((item_overlap - class_overlap) * n_preferred)  # Shared within class only.
  n_idio   <- n_preferred - n_cross - n_within             # Unique per verb.

  # Assign contiguous token IDs to each category.
  cross_tok    <- seq_len(n_cross)
  within_A_tok <- n_cross + seq_len(n_within)
  within_B_tok <- n_cross + n_within + seq_len(n_within)
  idio_pool    <- (n_cross + 2L * n_within + 1L):VOCAB_SIZE  # Available idio IDs.

  # Build each verb's preferred token list (cross + class-shared + idiosyncratic).
  make_verb_tokens <- function(class_shared) {
    c(cross_tok,
      class_shared,
      sample(idio_pool, n_idio, replace = FALSE))  # Each verb draws its own unique idio tokens.
  }

  A_tokens <- replicate(N_A, make_verb_tokens(within_A_tok), simplify = FALSE)
  B_tokens <- replicate(N_B, make_verb_tokens(within_B_tok), simplify = FALSE)

  # ---------------------------------------------------------------------------
  # Step 2: Build true distributions P_verb (identical to Model 1)
  # ---------------------------------------------------------------------------

  # Location parameter for lognormal: ensures E[weight] = mu.
  log_mean <- log(mu) - 0.5 * sigma^2

  build_dists <- function(token_list) {
    # Start all tokens at background weight.
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)
    for (i in seq_along(token_list)) {
      # Assign lognormal weights to preferred tokens.
      P[i, token_list[[i]]] <- rlnorm(n_preferred, log_mean, sigma)
    }
    P / rowSums(P)  # Normalise each row to a probability distribution.
  }

  PA_true <- build_dists(A_tokens)  # True distributions for dative verbs: N_A × V.
  PB_true <- build_dists(B_tokens)  # True distributions for motion verbs: N_B × V.

  # ---------------------------------------------------------------------------
  # Step 3: Pre-sample token sequences (MODEL 2 SPECIFIC)
  #
  # For efficiency, we draw the full MAX_NOBS tokens per verb ONCE per seed.
  # At each corpus-size grid point n_obs, we use only the first n_obs draws.
  # This simulates incremental corpus accumulation without re-sampling.
  #
  # Each draw is a single token ID sampled with replacement from the verb's
  # true distribution. This is what a learner would observe in a real corpus.
  # ---------------------------------------------------------------------------

  presample <- function(P_true_rows) {
    lapply(seq_len(nrow(P_true_rows)), function(i) {
      # sample() draws MAX_NOBS token IDs with replacement, using P_true as weights.
      # Result: a vector of MAX_NOBS integers, each in 1:VOCAB_SIZE.
      sample(seq_len(VOCAB_SIZE), size = MAX_NOBS, replace = TRUE,
             prob = P_true_rows[i, ])
    })
  }

  A_draws <- presample(PA_true)  # List of N_A integer vectors, each of length MAX_NOBS.
  B_draws <- presample(PB_true)  # List of N_B integer vectors.

  # ---------------------------------------------------------------------------
  # Step 4: Sweep corpus size and check onset criteria (MODEL 2 SPECIFIC)
  # ---------------------------------------------------------------------------

  ob_nobs   <- NA_integer_  # Corpus size (n_obs) at which ob fires. NA until detected.
  ow_nobs   <- NA_integer_  # Corpus size (n_obs) at which ow fires.
  ow_streak <- 0L           # Consecutive steps above the ow threshold.
  ow_start  <- NA_integer_  # n_obs at which the current ow streak began.

  for (n_obs in N_OBS_GRID) {

    # Build the add-k smoothed distribution for a single verb's draw sequence.
    smooth_verb <- function(draws) {
      # tabulate() counts how many times each token ID (1 to VOCAB_SIZE) appears
      # in the first n_obs draws. Returns an integer vector of length VOCAB_SIZE.
      counts <- tabulate(draws[seq_len(n_obs)], nbins = VOCAB_SIZE)

      # Add-k smoothing: (count + k) / (n_obs + k * V)
      # This is the posterior mean of a Dirichlet-Multinomial model with
      # a flat Dirichlet(add_k) prior.
      (counts + add_k) / (n_obs + add_k * VOCAB_SIZE)
    }

    # Apply smooth_verb to every verb in each class, then stack into matrices.
    # do.call(rbind, lapply(...)) is a fast way to row-bind a list of vectors.
    P_hat <- rbind(
      do.call(rbind, lapply(A_draws, smooth_verb)),  # Estimated dists: N_A × V
      do.call(rbind, lapply(B_draws, smooth_verb))   # Estimated dists: N_B × V
    )

    # Compute all pairwise DJS values among the 71 estimated distributions.
    jsd_mat <- pairwise_jsd(P_hat)  # 71 × 71 matrix

    # ---- ob check (between-class onset) ----
    if (is.na(ob_nobs)) {
      n_sig <- 0L

      # Check each dative verb: is DJS to motion verbs > DJS to other dative verbs?
      for (i in seq_len(N_A)) {
        within  <- jsd_mat[i, setdiff(seq_len(N_A), i)]  # Distances within dative class.
        between <- jsd_mat[i, (N_A + 1L):(N_A + N_B)]    # Distances to motion class.
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH) {
          n_sig <- n_sig + 1L
        }
      }

      # Check each motion verb: is DJS to dative verbs > DJS to other motion verbs?
      for (j in seq_len(N_B)) {
        i       <- N_A + j
        within  <- jsd_mat[i, N_A + setdiff(seq_len(N_B), j)]
        between <- jsd_mat[i, seq_len(N_A)]
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH) {
          n_sig <- n_sig + 1L
        }
      }

      # Declare ob if ≥ 10% of verbs pass.
      if (n_sig / (N_A + N_B) >= CLASS_FRAC) ob_nobs <- n_obs
    }

    # ---- ow check (within-class onset) ----
    if (is.na(ow_nobs)) {
      jA          <- jsd_mat[seq_len(N_A), seq_len(N_A)]                      # Dative × Dative sub-matrix.
      jB          <- jsd_mat[(N_A + 1L):(N_A + N_B), (N_A + 1L):(N_A + N_B)] # Motion × Motion sub-matrix.

      # Average of upper-triangle entries (unique pairs) across both classes.
      within_mean <- (mean(jA[upper.tri(jA)]) + mean(jB[upper.tri(jB)])) / 2.0

      if (within_mean > DJS_THRESH) {
        if (is.na(ow_start)) ow_start <- n_obs  # Mark start of streak.
        ow_streak <- ow_streak + 1L
        if (ow_streak >= SUSTAIN) ow_nobs <- ow_start  # Declare ow after 3 consecutive steps.
      } else {
        ow_streak <- 0L        # Reset streak if threshold not met.
        ow_start  <- NA_integer_
      }
    }

    # Stop early once both onsets have been detected.
    if (!is.na(ob_nobs) && !is.na(ow_nobs)) break
  }

  # Return corpus sizes at which ob and ow fired (NA if never detected).
  c(ob_nobs = ob_nobs, ow_nobs = ow_nobs)
}

# =============================================================================
# SECTION 5: Grid search — run all parameter combinations
# =============================================================================

run_combo <- function(row_i) {
  p  <- GRID[row_i, ]    # Parameters for this combination.
  ob <- integer(N_SEEDS)  # Corpus size at ob onset, per seed.
  ow <- integer(N_SEEDS)  # Corpus size at ow onset, per seed.

  for (s in seq_len(N_SEEDS)) {
    res   <- run_one(p$mu, p$sigma, N_PREFERRED, p$item_overlap, p$class_overlap,
                     p$add_k, seed = s)
    ob[s] <- res["ob_nobs"]
    ow[s] <- res["ow_nobs"]
  }

  # both: seeds where BOTH onsets were detected.
  both     <- !is.na(ob) & !is.na(ow)

  # ob_lt_ow: J&M's pattern — ob fires before ow (between-class before within-class).
  ob_lt_ow <- both & (ob < ow)

  # ow_lt_ob: the reverse pattern — ow fires before ob.
  # Expected at low k (high sensitivity to individual observations).
  ow_lt_ob <- both & (ow < ob)

  data.frame(
    mu             = p$mu,
    sigma          = p$sigma,
    item_overlap   = p$item_overlap,
    class_overlap  = p$class_overlap,
    add_k          = p$add_k,
    n_preferred    = N_PREFERRED,
    frac_detected  = mean(both),          # Fraction of seeds with both onsets found.
    frac_ob_lt_ow  = mean(ob_lt_ow),      # Fraction with J&M's pattern (ob before ow).
    frac_ow_lt_ob  = mean(ow_lt_ob),      # Fraction with reverse pattern (ow before ob).
    mean_ob_nobs   = if (any(both)) mean(ob[both]) else NA_real_,  # Mean corpus size at ob.
    mean_ow_nobs   = if (any(both)) mean(ow[both]) else NA_real_,  # Mean corpus size at ow.

    # Mean log(ow/ob): positive = ob fired first (larger = bigger gap).
    mean_log_ratio = if (any(ob_lt_ow)) mean(log(ow[ob_lt_ow] / ob[ob_lt_ow])) else NA_real_
  )
}

# =============================================================================
# SECTION 6: Run the grid search in parallel
# =============================================================================

cat(sprintf("%d parameter combinations x %d seeds\nWorkers: %d\n",
            nrow(GRID), N_SEEDS, N_WORKERS))
cat(sprintf("n_obs grid: %d values from %d to %d\n\n",
            length(N_OBS_GRID), min(N_OBS_GRID), max(N_OBS_GRID)))

plan(multisession, workers = N_WORKERS)  # Spawn N_WORKERS parallel R processes.
t0 <- proc.time()[["elapsed"]]

with_progress({
  results <- future_map_dfr(seq_len(nrow(GRID)), run_combo, .progress = TRUE)
})

plan(sequential)  # Return to single-threaded execution.

write.csv(results, "../data/grid_results_model2.csv", row.names = FALSE)

cat(sprintf("Saved: grid_results_model2.csv\n"))
cat(sprintf("frac_ow_lt_ob == 1.0 (all seeds): %d\n",
            sum(results$frac_ow_lt_ob == 1.0, na.rm = TRUE)))
cat(sprintf("frac_ob_lt_ow == 1.0 (all seeds): %d\n",
            sum(results$frac_ob_lt_ow == 1.0, na.rm = TRUE)))
cat(sprintf("Elapsed: %.0fs\n", proc.time()[["elapsed"]] - t0))
