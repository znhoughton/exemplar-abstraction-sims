# =============================================================================
# Model 3: Hierarchical estimation (class prototype prior)
#
# Like Model 2, this model accumulates actual token observations. But instead
# of smoothing each verb toward the UNIFORM distribution (Model 2's flat,
# uninformative prior), it smooths each verb toward its CLASS PROTOTYPE —
# the pooled distribution of all verbs in the same class.
#
# Estimation proceeds in two steps:
#
#   Step 1 — Class prototype (from pooled observations):
#     P̂_class(t) = (Σ_{v ∈ class} count(t | v, n_obs) + k₀)
#                  / (N_class × n_obs + k₀ × V)
#     The prototype pools counts across all N_class verbs, so it has
#     N_class times more data than any individual verb. Even at small n_obs
#     it captures the class-level distributional profile.
#
#   Step 2 — Hierarchical verb estimate:
#     P̂_verb(t) = (count(t | v, n_obs) + k₀ + γ × P̂_class(t))
#                 / (n_obs + k₀ × V + γ)
#     Each verb's estimate is a mixture of its own observations, a weak
#     flat floor (k₀), and the class prototype weighted by γ.
#
# γ (GAMMA) controls how strongly each verb is pulled toward its class:
#   γ ≈ 0   : no pooling — each verb relies on its own observations only
#             (equivalent to Model 2 with add_k = k₀ = 0.001)
#   γ large : full pooling — verb estimate ≈ class prototype
#
# BAYESIAN INTERPRETATION: this is an empirical Bayes Dirichlet-Multinomial
# model. The prior over verb distributions is Dirichlet(k₀ + γ × P̂_class),
# centered on the class prototype with total concentration γ. "Empirical
# Bayes" because the prior parameters (the prototype) are estimated from
# the data rather than fixed in advance.
#
# WHAT THIS MODEL CAPTURES: the functional effect of shared neural parameters.
# In a transformer, observing "give" updates weights also used by "sell",
# "award" etc. (because they co-occur in similar contexts). γ approximates
# this cross-verb sharing: observing one verb's tokens partially informs
# estimates for all verbs in the same class.
#
# IMPORTANT — learning order: abstractions emerge FROM exemplars, not before.
# The prototype is computed from individual verb observations at each step.
# It is not given a priori. This is an EXEMPLAR-DRIVEN model that happens to
# compute class-level statistics as an intermediate step.
#
# DATA GENERATION: identical to Models 1 and 2.
# ESTIMATION: differs — uses class prototype prior instead of uniform prior.
# =============================================================================

library(furrr)      # parallel map functions (future_map_dfr)
library(progressr)  # progress bar support

# =============================================================================
# SECTION 1: Constants
# =============================================================================

VOCAB_SIZE  <- 1000L  # Total vocabulary size.
N_PREFERRED <- 50L    # Preferred tokens per verb.
N_A         <- 35L    # Dative verbs (Jian & Manning Appendix A).
N_B         <- 36L    # Motion verbs (Jian & Manning Appendix A).
N_SEEDS     <- 50L    # Random seeds per parameter combination.
N_WORKERS   <- 6L     # Parallel CPU cores to use.
BG_WEIGHT   <- 1.0    # Background token weight.
MAX_NOBS    <- 5000L  # Maximum corpus size simulated per verb.

# Corpus-size grid: 20 log-spaced values from 1 to MAX_NOBS.
N_OBS_GRID  <- unique(round(exp(seq(log(1), log(MAX_NOBS), length.out = 20L))))

K_0 <- 0.001  # Weak flat prior used INSIDE the hierarchical estimate.
              # This serves as a numerical floor — prevents divisions by zero
              # and adds a tiny amount of uniform smoothing. It is NOT the
              # main smoothing parameter (that is γ / gamma).

# Onset detection thresholds (same as Models 1 and 2)
P_THRESH   <- 0.001  # Mann-Whitney p-value threshold (ob criterion).
CLASS_FRAC <- 0.10   # Fraction of verbs that must pass Mann-Whitney for ob.
DJS_THRESH <- 0.01   # Absolute within-class DJS threshold for ow.
SUSTAIN    <- 3L     # Consecutive steps ow must be above threshold.

# =============================================================================
# SECTION 2: Parameter grid
# The key additional parameter vs. Models 1 and 2 is gamma — the pooling
# strength toward the class prototype.
# =============================================================================

GRID <- expand.grid(
  mu            = c(10, 30, 60, 100),  # Mean lognormal weight on preferred tokens.
  sigma         = c(0.5, 1.0, 1.5),   # Spread of lognormal weights.
  item_overlap  = c(0.5, 0.6, 0.7),   # Fraction of preferred tokens shared within class.
  class_overlap = c(0.2, 0.3, 0.4),   # Fraction of preferred tokens shared across classes.

  gamma = c(0.1, 1, 10, 100, 1000, 10000),  # THE KEY VARIABLE. Prototype pooling strength.
                                              # Small γ → verb relies on own observations
                                              #           (like low-k Model 2).
                                              # Large γ → verb looks like class prototype
                                              #           → between-class signal immediate
                                              #           → ob < ow.

  stringsAsFactors = FALSE
)

# Remove invalid combinations: class_overlap must be < item_overlap.
GRID <- GRID[GRID$class_overlap < GRID$item_overlap, ]

# =============================================================================
# SECTION 3: Pairwise Jensen-Shannon Divergence
# Identical to Models 1 and 2. See model1_monotone.R for full explanation.
# =============================================================================

pairwise_jsd <- function(P) {
  N   <- nrow(P)   # Number of verbs (71 = N_A + N_B).
  eps <- 1e-30     # Numerical stability: prevents log(0).

  # Per-verb Shannon entropy.
  H <- -rowSums(P * log(P + eps))

  jsd_mat <- matrix(0.0, N, N)

  for (i in seq_len(N - 1L)) {
    j_idx <- (i + 1L):N  # Indices of verbs after i.

    # Mixture distribution M_ij = 0.5*(P_i + P_j) for all j > i at once.
    M <- 0.5 * (matrix(P[i, ], nrow = length(j_idx), ncol = VOCAB_SIZE, byrow = TRUE)
                + P[j_idx, , drop = FALSE])

    # Entropy of each mixture.
    H_M <- -rowSums(M * log(M + eps))

    # DJS = H(mixture) - 0.5*(H(P_i) + H(P_j)), clamped to >= 0.
    d <- pmax(0.0, H_M - 0.5 * (H[i] + H[j_idx]))

    jsd_mat[i, j_idx] <- d
    jsd_mat[j_idx, i] <- d
  }
  jsd_mat
}

# =============================================================================
# SECTION 4: Single simulation run
#
# Token pool and true distributions are identical to Models 1 and 2.
# Estimation is new: hierarchical pooling toward the class prototype.
# =============================================================================

run_one <- function(mu, sigma, n_preferred, item_overlap, class_overlap, gamma, seed) {

  set.seed(seed)

  # ---------------------------------------------------------------------------
  # Step 1: Build the token pool (identical to Models 1 and 2)
  # ---------------------------------------------------------------------------

  n_cross  <- round(class_overlap * n_preferred)
  n_within <- round((item_overlap - class_overlap) * n_preferred)
  n_idio   <- n_preferred - n_cross - n_within

  cross_tok    <- seq_len(n_cross)
  within_A_tok <- n_cross + seq_len(n_within)
  within_B_tok <- n_cross + n_within + seq_len(n_within)
  idio_pool    <- (n_cross + 2L * n_within + 1L):VOCAB_SIZE

  make_verb_tokens <- function(class_shared) {
    c(cross_tok, class_shared, sample(idio_pool, n_idio, replace = FALSE))
  }

  A_tokens <- replicate(N_A, make_verb_tokens(within_A_tok), simplify = FALSE)
  B_tokens <- replicate(N_B, make_verb_tokens(within_B_tok), simplify = FALSE)

  # ---------------------------------------------------------------------------
  # Step 2: Build true distributions P_verb (identical to Models 1 and 2)
  # ---------------------------------------------------------------------------

  log_mean <- log(mu) - 0.5 * sigma^2  # Lognormal location: ensures E[weight] = mu.

  build_dists <- function(token_list) {
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)
    for (i in seq_along(token_list)) {
      P[i, token_list[[i]]] <- rlnorm(n_preferred, log_mean, sigma)
    }
    P / rowSums(P)
  }

  PA_true <- build_dists(A_tokens)  # N_A × V true distributions for dative verbs.
  PB_true <- build_dists(B_tokens)  # N_B × V true distributions for motion verbs.

  # ---------------------------------------------------------------------------
  # Step 3: Pre-sample token sequences (same as Model 2)
  #
  # Draw MAX_NOBS tokens per verb once per seed. At each corpus-size grid
  # point n_obs, we use only the first n_obs draws.
  # ---------------------------------------------------------------------------

  presample <- function(P_true_rows) {
    lapply(seq_len(nrow(P_true_rows)), function(i) {
      sample(seq_len(VOCAB_SIZE), size = MAX_NOBS, replace = TRUE,
             prob = P_true_rows[i, ])
    })
  }

  A_draws <- presample(PA_true)  # List of N_A integer vectors of length MAX_NOBS.
  B_draws <- presample(PB_true)  # List of N_B integer vectors.

  # ---------------------------------------------------------------------------
  # Step 4: Sweep corpus size and apply hierarchical estimation (MODEL 3 SPECIFIC)
  # ---------------------------------------------------------------------------

  ob_nobs   <- NA_integer_
  ow_nobs   <- NA_integer_
  ow_streak <- 0L
  ow_start  <- NA_integer_

  for (n_obs in N_OBS_GRID) {

    # ---- Compute raw token counts for each verb ----
    # tabulate(draws[1:n_obs], nbins=V) counts how many times each of the V
    # token IDs appears in the first n_obs draws for one verb.
    count_verb <- function(draws) {
      tabulate(draws[seq_len(n_obs)], nbins = VOCAB_SIZE)
    }

    # Stack counts into matrices: N_A × V (dative) and N_B × V (motion).
    counts_A <- do.call(rbind, lapply(A_draws, count_verb))
    counts_B <- do.call(rbind, lapply(B_draws, count_verb))

    # ---- Step 1: Estimate class prototypes ----
    # The prototype pools ALL verbs in the class into one big estimate.
    # This gives N_A * n_obs (or N_B * n_obs) total observations per class —
    # far more than any individual verb's n_obs, so the prototype is stable
    # even when individual verb estimates are noisy.

    smooth_A <- N_A * n_obs + K_0 * VOCAB_SIZE  # Total pseudo-counts denominator for class A.
    smooth_B <- N_B * n_obs + K_0 * VOCAB_SIZE  # Same for class B.

    # colSums(counts_A) sums the counts across all N_A verbs for each token.
    # Adding K_0 provides a weak flat floor.
    # Dividing by smooth_A gives a proper probability vector (sums to ~1).
    proto_A <- (colSums(counts_A) + K_0) / smooth_A  # Length-V probability vector.
    proto_B <- (colSums(counts_B) + K_0) / smooth_B  # Length-V probability vector.

    # ---- Step 2: Hierarchical verb estimates ----
    # Each verb's estimate = its own counts + flat floor (k_0) + prototype (gamma).
    # The denominator ensures each row sums to 1.
    denom <- n_obs + K_0 * VOCAB_SIZE + gamma

    # matrix(proto_A, nrow=N_A, ncol=V, byrow=TRUE) tiles the prototype across
    # N_A rows so it can be added element-wise to counts_A (N_A × V matrix).
    P_hat_A <- (counts_A + K_0 + gamma * matrix(proto_A, nrow = N_A,
                                                  ncol = VOCAB_SIZE, byrow = TRUE)) / denom

    P_hat_B <- (counts_B + K_0 + gamma * matrix(proto_B, nrow = N_B,
                                                  ncol = VOCAB_SIZE, byrow = TRUE)) / denom

    # Stack dative and motion estimates into one (N_A+N_B) × V matrix.
    P_hat   <- rbind(P_hat_A, P_hat_B)

    # Compute all pairwise DJS values.
    jsd_mat <- pairwise_jsd(P_hat)

    # ---- ob check (between-class onset) ----
    if (is.na(ob_nobs)) {
      n_sig <- 0L

      # Check each dative verb.
      for (i in seq_len(N_A)) {
        within  <- jsd_mat[i, setdiff(seq_len(N_A), i)]  # Distances to other dative verbs.
        between <- jsd_mat[i, (N_A + 1L):(N_A + N_B)]    # Distances to motion verbs.
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH) {
          n_sig <- n_sig + 1L
        }
      }

      # Check each motion verb.
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

      # ob fires when ≥ 10% of all verbs pass.
      if (n_sig / (N_A + N_B) >= CLASS_FRAC) ob_nobs <- n_obs
    }

    # ---- ow check (within-class onset) ----
    if (is.na(ow_nobs)) {
      jA          <- jsd_mat[seq_len(N_A), seq_len(N_A)]                      # Dative × Dative.
      jB          <- jsd_mat[(N_A + 1L):(N_A + N_B), (N_A + 1L):(N_A + N_B)] # Motion × Motion.
      within_mean <- (mean(jA[upper.tri(jA)]) + mean(jB[upper.tri(jB)])) / 2.0

      if (within_mean > DJS_THRESH) {
        if (is.na(ow_start)) ow_start <- n_obs
        ow_streak <- ow_streak + 1L
        if (ow_streak >= SUSTAIN) ow_nobs <- ow_start
      } else {
        ow_streak <- 0L
        ow_start  <- NA_integer_
      }
    }

    if (!is.na(ob_nobs) && !is.na(ow_nobs)) break
  }

  c(ob_nobs = ob_nobs, ow_nobs = ow_nobs)
}

# =============================================================================
# SECTION 5: Grid search — run all parameter combinations
# =============================================================================

run_combo <- function(row_i) {
  p  <- GRID[row_i, ]
  ob <- integer(N_SEEDS)
  ow <- integer(N_SEEDS)

  for (s in seq_len(N_SEEDS)) {
    res   <- run_one(p$mu, p$sigma, N_PREFERRED, p$item_overlap, p$class_overlap,
                     p$gamma, seed = s)
    ob[s] <- res["ob_nobs"]
    ow[s] <- res["ow_nobs"]
  }

  both     <- !is.na(ob) & !is.na(ow)
  ob_lt_ow <- both & (ob < ow)  # ob fires before ow (J&M's pattern).
  ow_lt_ob <- both & (ow < ob)  # ow fires before ob (exemplar idiosyncrasy pattern).

  data.frame(
    mu             = p$mu,
    sigma          = p$sigma,
    item_overlap   = p$item_overlap,
    class_overlap  = p$class_overlap,
    gamma          = p$gamma,
    n_preferred    = N_PREFERRED,
    frac_detected  = mean(both),          # Fraction of seeds with both onsets found.
    frac_ob_lt_ow  = mean(ob_lt_ow),      # Fraction with J&M's pattern (ob before ow).
    frac_ow_lt_ob  = mean(ow_lt_ob),      # Fraction with reverse pattern (ow before ob).
    mean_ob_nobs   = if (any(both)) mean(ob[both]) else NA_real_,
    mean_ow_nobs   = if (any(both)) mean(ow[both]) else NA_real_,
    mean_log_ratio = if (any(ob_lt_ow)) mean(log(ow[ob_lt_ow] / ob[ob_lt_ow])) else NA_real_
  )
}

# =============================================================================
# SECTION 6: Run the grid search in parallel
# =============================================================================

cat(sprintf("%d parameter combinations x %d seeds\nWorkers: %d\n",
            nrow(GRID), N_SEEDS, N_WORKERS))
cat(sprintf("n_obs grid: %d values from %d to %d\n",
            length(N_OBS_GRID), min(N_OBS_GRID), max(N_OBS_GRID)))
cat(sprintf("gamma values: %s\n\n",
            paste(sort(unique(GRID$gamma)), collapse = ", ")))

plan(multisession, workers = N_WORKERS)  # Spawn N_WORKERS parallel R processes.
t0 <- proc.time()[["elapsed"]]

with_progress({
  results <- future_map_dfr(seq_len(nrow(GRID)), run_combo, .progress = TRUE)
})

plan(sequential)  # Return to single-threaded execution.

write.csv(results, "../data/grid_results_model3_hierarchical.csv", row.names = FALSE)

cat(sprintf("Saved: grid_results_model3.csv\n"))
cat(sprintf("frac_ob_lt_ow == 1.0 (all seeds): %d\n",
            sum(results$frac_ob_lt_ow == 1.0, na.rm = TRUE)))
cat(sprintf("frac_ow_lt_ob == 1.0 (all seeds): %d\n",
            sum(results$frac_ow_lt_ob == 1.0, na.rm = TRUE)))

# Summary table: how does frac_ob_lt_ow change as gamma increases?
# This is the key comparison — larger gamma should produce more ob < ow.
cat("\n--- Mean frac_ob_lt_ow by gamma (averaged over structural parameters) ---\n")
gamma_summary <- aggregate(cbind(frac_ob_lt_ow, frac_ow_lt_ob, frac_detected) ~ gamma,
                            data = results, FUN = mean)
print(gamma_summary[order(gamma_summary$gamma), ], row.names = FALSE)

cat(sprintf("\nElapsed: %.0fs\n", proc.time()[["elapsed"]] - t0))
