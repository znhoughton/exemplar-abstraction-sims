# =============================================================================
# Model 1: Monotone simulation (the k → ∞ limit)
#
# Each verb's distribution at training step α is a linear interpolation
# between a uniform baseline and its true distribution:
#
#   P_verb(α) = (1 − α) × Uniform  +  α × P_true_verb
#
# At α = 0 every verb looks uniform (no information); at α = 1 every verb
# shows its true distribution. The trajectory is deterministic — there is
# no observation-level noise. This is the theoretical limit of a learner
# that is completely insensitive to individual token observations (k → ∞).
#
# KEY PREDICTION: ob < ow
#   Between-class divergence (ob) is detectable as soon as α > 0 because
#   the class structure is baked into the true distributions. Within-class
#   divergence (ow) must wait for the absolute DJS threshold (0.01) to be
#   exceeded, which happens later.
#
# DATA GENERATION (identical across Models 1, 2, and 3):
#   Each verb has n_preferred "preferred" tokens partitioned into:
#     cross tokens  : shared by ALL verbs in BOTH classes
#     within tokens : shared only within the same class
#     idiosyncratic : unique to that individual verb
#   Preferred token weights are drawn from a lognormal distribution.
#   All other vocabulary items get a low uniform background weight.
#
# ESTIMATION (unique to Model 1):
#   The learner does not sample tokens — it observes the interpolated
#   distribution directly. The "estimate" at step α IS the model output.
#   No prior, likelihood, or posterior is involved.
#
# ONSET CRITERIA (identical across Models 1, 2, and 3):
#   ob: first α where ≥ 10% of verbs show DJS(other-class) > DJS(same-class)
#       via one-tailed Mann-Whitney test (p < 0.001). RELATIVE test.
#   ow: first α where mean within-class DJS > 0.01 for 3 consecutive steps.
#       ABSOLUTE threshold.
# =============================================================================

library(furrr)      # parallel map functions (future_map_dfr)
library(progressr)  # progress bar support

# =============================================================================
# SECTION 1: Constants
# These values are fixed throughout all simulations in this script.
# =============================================================================

VOCAB_SIZE  <- 1000L  # Total number of distinct tokens in the vocabulary.
                      # GPT-2 uses 50,257; we use 1,000 for speed. The
                      # qualitative results (ob < ow ordering) are robust to
                      # vocabulary size.

N_PREFERRED <- 50L    # Number of "preferred" tokens per verb — tokens that
                      # appear with elevated probability in that verb's context.
                      # This is ~5% of the vocabulary, matching GPT-2's ratio.

N_A         <- 35L    # Number of dative verbs (e.g. give, sell, award).
                      # Taken directly from Jian & Manning (2026) Appendix A.

N_B         <- 36L    # Number of motion verbs (e.g. go, run, walk).
                      # Taken directly from Jian & Manning (2026) Appendix A.

N_SEEDS     <- 50L    # Number of random seeds to run per parameter combination.
                      # Each seed produces a different random token pool and
                      # distribution, giving us a distribution of results.

N_WORKERS   <- 6L     # Number of parallel CPU cores to use. Adjust to match
                      # your machine (check with parallel::detectCores()).

BG_WEIGHT   <- 1.0    # The raw weight assigned to non-preferred ("background")
                      # tokens. Preferred tokens get lognormal weights with
                      # mean mu >> 1, making them much more probable than
                      # background tokens.

# Log-spaced alpha values: 25 values from 0.001 to 1.0.
# α represents how far training has progressed (0 = random init, 1 = converged).
# We use LOG spacing (not linear) because interesting onset behaviour happens
# at very small α (near 0.001). Linear spacing would waste resolution at large α
# where both ob and ow have already fired.
ALPHAS <- exp(seq(log(0.001), log(1.0), length.out = 25L))

# Onset detection thresholds (matching Jian & Manning 2026)
P_THRESH   <- 0.001  # p-value cutoff for the per-verb Mann-Whitney test (ob).
                     # A verb "passes" if its between-class distances are
                     # stochastically greater than within-class distances at
                     # this significance level.

CLASS_FRAC <- 0.10   # Fraction of verbs that must pass the Mann-Whitney test
                     # before ob is declared. 10% of 71 verbs = ~7 verbs.
                     # This is conservative — J&M use ~1.4% (first departure
                     # from 0). Our stricter threshold makes the ob < ow result
                     # MORE impressive, not less.

DJS_THRESH <- 0.01   # Absolute mean within-class DJS threshold for ow.
                     # Taken from J&M's Experiment 2 breakpoint criterion.

SUSTAIN    <- 3L     # Number of consecutive steps the ow threshold must be
                     # exceeded before ow is declared. Prevents transient
                     # noise spikes from triggering a premature ow onset.

# =============================================================================
# SECTION 2: Parameter grid
# We test all combinations of these structural parameters to show the results
# are robust and not specific to one parameter setting.
# =============================================================================

GRID <- expand.grid(
  mu            = c(10, 30, 60, 100),  # Mean lognormal weight on preferred tokens.
                                        # Higher mu = preferred tokens are more
                                        # dominant relative to background tokens,
                                        # creating sharper verb-specific distributions.

  sigma         = c(0.5, 1.0, 1.5),   # Spread of lognormal weights across preferred
                                        # tokens. Higher sigma = more unequal weighting
                                        # among preferred tokens (some very high weight,
                                        # some low). Controls within-verb heterogeneity.

  item_overlap  = c(0.5, 0.6, 0.7),   # Fraction of preferred tokens SHARED within
                                        # the same class (e.g. fraction shared among
                                        # all dative verbs). Controls how similar same-
                                        # class verbs are to each other.

  class_overlap = c(0.2, 0.3, 0.4),   # Fraction of preferred tokens shared ACROSS
                                        # both classes (shared by dative AND motion
                                        # verbs). Controls how similar the two classes
                                        # are to each other.

  stringsAsFactors = FALSE
)

# Remove invalid combinations: class_overlap must be strictly less than
# item_overlap. This ensures within-class similarity always exceeds
# cross-class similarity — a prerequisite for the ob criterion to ever fire.
GRID <- GRID[GRID$class_overlap < GRID$item_overlap, ]

# =============================================================================
# SECTION 3: Pairwise Jensen-Shannon Divergence
#
# Computes the DJS between every pair of verb distributions simultaneously.
# DJS(P, Q) = H(0.5*(P+Q)) - 0.5*(H(P) + H(Q))
# where H is Shannon entropy. Ranges from 0 (identical) to log(2) ≈ 0.693.
#
# Input:  P — an N × V matrix where each row is one verb's probability
#             distribution over the V-token vocabulary (rows sum to 1).
# Output: an N × N symmetric matrix of pairwise DJS values.
#
# Implementation note: instead of a double loop over all (i, j) pairs
# (which would be slow in R), we loop over i only and compute all pairs
# (i, j>i) at once using matrix operations. This gives N-1 iterations
# instead of N*(N-1)/2 — much faster.
# =============================================================================

pairwise_jsd <- function(P) {
  N   <- nrow(P)   # Number of verbs (71 = N_A + N_B)
  eps <- 1e-30     # Small constant added before log() to avoid log(0) = -Inf.
                   # This is numerically safe because probabilities are never
                   # exactly 0 in practice with smoothing applied.

  # Compute Shannon entropy for each verb: H(P_i) = -sum_t P_i(t) * log(P_i(t))
  # rowSums operates on the N × V matrix, giving a length-N vector of entropies.
  H <- -rowSums(P * log(P + eps))

  # Initialise the N × N output matrix with zeros.
  # We fill in the upper triangle and mirror it to the lower triangle.
  jsd_mat <- matrix(0.0, N, N)

  for (i in seq_len(N - 1L)) {
    # j_idx: indices of all verbs j that come AFTER verb i (avoids double-counting).
    j_idx <- (i + 1L):N

    # Compute the mixture distribution M = 0.5*(P_i + P_j) for all j at once.
    # matrix(P[i, ], ..., byrow = TRUE) repeats row i to form a (N-i) × V matrix
    # so it can be added element-wise to P[j_idx, ].
    M <- 0.5 * (matrix(P[i, ], nrow = length(j_idx), ncol = VOCAB_SIZE, byrow = TRUE)
                + P[j_idx, , drop = FALSE])

    # Entropy of each mixture distribution: H(M_ij) for all j > i.
    # This gives a length-(N-i) vector.
    H_M <- -rowSums(M * log(M + eps))

    # DJS = H(mixture) - 0.5*(H(P_i) + H(P_j)).
    # pmax(0, ...) clamps small negative values from floating-point error to 0.
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
# Runs one full simulation for one set of parameters and one random seed.
# Returns the α values at which ob and ow fire, or NA if not found.
# =============================================================================

run_one <- function(mu, sigma, n_preferred, item_overlap, class_overlap, seed) {

  set.seed(seed)  # Fix the random seed for reproducibility. Different seeds
                  # give different random token pools and lognormal weights,
                  # letting us estimate variability across the parameter setting.

  # ---------------------------------------------------------------------------
  # Step 1: Build the token pool
  #
  # Each verb has n_preferred preferred tokens, split into three groups:
  #   n_cross  tokens shared by ALL verbs in BOTH classes
  #   n_within tokens shared by ALL verbs in ONE class only
  #   n_idio   tokens unique to each individual verb
  #
  # We assign contiguous integer IDs to tokens so that each group occupies
  # a known range: [1, n_cross], [n_cross+1, n_cross+n_within], etc.
  # ---------------------------------------------------------------------------

  # Number of cross-class tokens (shared by dative AND motion verbs).
  n_cross  <- round(class_overlap * n_preferred)

  # Number of within-class tokens (shared within one class but not the other).
  # The gap (item_overlap - class_overlap) is what makes classes distinct.
  n_within <- round((item_overlap - class_overlap) * n_preferred)

  # Number of idiosyncratic tokens (unique per verb, what makes verbs differ
  # even within the same class).
  n_idio   <- n_preferred - n_cross - n_within

  # Token ID ranges for each type.
  cross_tok    <- seq_len(n_cross)                            # e.g. 1:10
  within_A_tok <- n_cross + seq_len(n_within)                 # e.g. 11:25 (class A only)
  within_B_tok <- n_cross + n_within + seq_len(n_within)      # e.g. 26:40 (class B only)

  # The idiosyncratic pool: all remaining token IDs after cross and within tokens.
  # Each verb independently samples n_idio tokens from this shared pool.
  idio_pool <- (n_cross + 2L * n_within + 1L):VOCAB_SIZE

  # Helper function: assemble one verb's full preferred token list.
  # class_shared = the within-class tokens for this verb's class (A or B).
  make_verb_tokens <- function(class_shared) {
    c(cross_tok,                                        # tokens shared by all
      class_shared,                                     # tokens shared within class
      sample(idio_pool, n_idio, replace = FALSE))       # unique idiosyncratic tokens
  }

  # Build token lists for all verbs in each class.
  # replicate() calls make_verb_tokens N_A (or N_B) times, each with a fresh
  # random draw of idiosyncratic tokens.
  A_tokens <- replicate(N_A, make_verb_tokens(within_A_tok), simplify = FALSE)
  B_tokens <- replicate(N_B, make_verb_tokens(within_B_tok), simplify = FALSE)

  # ---------------------------------------------------------------------------
  # Step 2: Build true distributions P_verb
  #
  # Each verb's true distribution is a probability vector over all V tokens.
  # Preferred tokens get lognormal weights (high probability); all other
  # tokens get BG_WEIGHT = 1.0 (low, uniform background probability).
  # Rows are normalised so they sum to 1.
  #
  # The lognormal mean is set so that E[weight] = mu (the parameter).
  # Formula: if X ~ LogNormal(log_mean, sigma), then E[X] = exp(log_mean + sigma²/2)
  # Setting log_mean = log(mu) - 0.5*sigma² gives E[X] = mu.
  # ---------------------------------------------------------------------------

  log_mean <- log(mu) - 0.5 * sigma^2  # lognormal location parameter

  build_dists <- function(token_list) {
    # Start with a matrix where every token has background weight BG_WEIGHT = 1.0.
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)

    for (i in seq_along(token_list)) {
      # For each verb, assign lognormal weights to its preferred tokens.
      # rlnorm(n, log_mean, sigma) draws n values from LogNormal(log_mean, sigma).
      P[i, token_list[[i]]] <- rlnorm(n_preferred, log_mean, sigma)
    }

    # Normalise each row to sum to 1 (convert raw weights to probabilities).
    P / rowSums(P)
  }

  # Stack dative verbs (rows 1:N_A) on top of motion verbs (rows N_A+1:N_A+N_B).
  P_true <- rbind(build_dists(A_tokens),   # N_A × V matrix
                  build_dists(B_tokens))   # N_B × V matrix

  # ---------------------------------------------------------------------------
  # Step 3: Sweep α and check onset criteria
  #
  # MODEL 1 SPECIFIC: The "estimated" distribution at step α is the interpolation
  #   P_obs(α) = (1 - α) * Uniform + α * P_true
  # No token observations are drawn. The trajectory is deterministic.
  # ---------------------------------------------------------------------------

  uni <- 1.0 / VOCAB_SIZE  # Uniform probability: each of V tokens equally likely.
                            # This is the starting state at α = 0 (no learning).

  # Tracking variables for onset detection.
  ob_alpha  <- NA_real_  # α at which ob fires. NA until detected.
  ow_alpha  <- NA_real_  # α at which ow fires. NA until detected.
  ow_streak <- 0L        # Number of consecutive steps where within-class DJS > threshold.
  ow_start  <- NA_real_  # The α at which the current ow streak began.

  for (alpha in ALPHAS) {

    # Interpolate: at this α, every verb's distribution is a weighted average
    # of uniform (weight 1-α) and its true distribution (weight α).
    # Broadcasting: uni is a scalar, P_true is N × V; R broadcasts the scalar
    # across all rows and columns automatically.
    P_obs   <- (1.0 - alpha) * uni + alpha * P_true  # N × V matrix

    # Compute all 71×71 pairwise DJS values for this step.
    jsd_mat <- pairwise_jsd(P_obs)  # 71 × 71 matrix

    # ---- ob check (between-class onset) ----
    # Only run this check if ob has not yet been detected.
    if (is.na(ob_alpha)) {
      n_sig <- 0L  # Count of verbs that pass the Mann-Whitney test this step.

      # Check each dative verb (rows 1 to N_A in jsd_mat).
      for (i in seq_len(N_A)) {
        # within: DJS distances from verb i to all OTHER dative verbs.
        within  <- jsd_mat[i, setdiff(seq_len(N_A), i)]

        # between: DJS distances from verb i to all motion verbs.
        between <- jsd_mat[i, (N_A + 1L):(N_A + N_B)]

        # One-tailed Mann-Whitney U-test: is between stochastically greater
        # than within? exact = FALSE uses a normal approximation (faster).
        # suppressWarnings() silences tie-related warnings that are harmless here.
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH) {
          n_sig <- n_sig + 1L  # This verb passes — increment counter.
        }
      }

      # Check each motion verb (rows N_A+1 to N_A+N_B in jsd_mat).
      for (j in seq_len(N_B)) {
        i       <- N_A + j  # Row index in jsd_mat for this motion verb.
        within  <- jsd_mat[i, N_A + setdiff(seq_len(N_B), j)]  # Other motion verbs.
        between <- jsd_mat[i, seq_len(N_A)]                     # All dative verbs.
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH) {
          n_sig <- n_sig + 1L
        }
      }

      # ob fires when at least CLASS_FRAC (10%) of all verbs pass.
      if (n_sig / (N_A + N_B) >= CLASS_FRAC) ob_alpha <- alpha
    }

    # ---- ow check (within-class onset) ----
    # Only run this check if ow has not yet been detected.
    if (is.na(ow_alpha)) {
      # Extract the within-class DJS sub-matrices.
      jA <- jsd_mat[seq_len(N_A), seq_len(N_A)]                     # Dative × Dative
      jB <- jsd_mat[(N_A + 1L):(N_A + N_B), (N_A + 1L):(N_A + N_B)] # Motion × Motion

      # Mean within-class DJS: average of the upper triangle of each sub-matrix
      # (upper triangle avoids counting the zero diagonal and double-counting pairs).
      within_mean <- (mean(jA[upper.tri(jA)]) + mean(jB[upper.tri(jB)])) / 2.0

      if (within_mean > DJS_THRESH) {
        # Threshold exceeded: start or extend the streak.
        if (is.na(ow_start)) ow_start <- alpha  # Record where the streak began.
        ow_streak <- ow_streak + 1L

        # Declare ow only after SUSTAIN = 3 consecutive steps above threshold.
        # This prevents a single noisy step from triggering ow prematurely.
        if (ow_streak >= SUSTAIN) ow_alpha <- ow_start
      } else {
        # Threshold not met: reset the streak.
        ow_streak <- 0L
        ow_start  <- NA_real_
      }
    }

    # Early stopping: if both onsets have fired, no need to continue.
    if (!is.na(ob_alpha) && !is.na(ow_alpha)) break
  }

  # Return both onset values. NA means the onset was never detected within ALPHAS.
  c(ob_alpha = ob_alpha, ow_alpha = ow_alpha)
}

# =============================================================================
# SECTION 5: Grid search — run all parameter combinations
#
# run_combo() runs all N_SEEDS seeds for one row of GRID and summarises
# the results into a single data frame row.
# =============================================================================

run_combo <- function(row_i) {
  p  <- GRID[row_i, ]          # Extract this row's parameter values.
  ob <- numeric(N_SEEDS)        # Pre-allocate vectors for ob and ow onset values.
  ow <- numeric(N_SEEDS)        # One entry per seed.

  for (s in seq_len(N_SEEDS)) {
    res   <- run_one(p$mu, p$sigma, N_PREFERRED, p$item_overlap, p$class_overlap, seed = s)
    ob[s] <- res["ob_alpha"]    # α at which ob fired (NA if not found).
    ow[s] <- res["ow_alpha"]    # α at which ow fired (NA if not found).
  }

  # both: TRUE for seeds where BOTH ob and ow were detected.
  both     <- !is.na(ob) & !is.na(ow)

  # ob_lt_ow: TRUE for seeds where both detected AND ob fired before ow.
  # This is J&M's predicted pattern (between-class onset precedes within-class onset).
  ob_lt_ow <- both & (ob < ow)

  # Return a one-row data frame with summary statistics for this parameter combo.
  data.frame(
    mu             = p$mu,
    sigma          = p$sigma,
    item_overlap   = p$item_overlap,
    class_overlap  = p$class_overlap,
    n_preferred    = N_PREFERRED,

    # Fraction of seeds where both onsets were found (data quality check).
    frac_detected  = mean(both),

    # KEY RESULT: fraction of seeds with ob < ow (J&M's predicted pattern).
    frac_ob_lt_ow  = mean(ob_lt_ow),

    # Mean α at ob onset, averaged over seeds where both were detected.
    mean_ob        = if (any(both)) mean(ob[both]) else NA_real_,

    # Mean α at ow onset.
    mean_ow        = if (any(both)) mean(ow[both]) else NA_real_,

    # Mean log(ow/ob): positive values indicate ob fired before ow.
    # Larger values = larger gap between the two onsets.
    mean_log_ratio = if (any(ob_lt_ow)) mean(log(ow[ob_lt_ow] / ob[ob_lt_ow])) else NA_real_
  )
}

# =============================================================================
# SECTION 6: Run the grid search in parallel
# =============================================================================

cat(sprintf("%d parameter combinations x %d seeds\nWorkers: %d\n\n",
            nrow(GRID), N_SEEDS, N_WORKERS))

# Set up parallel workers. multisession spawns fresh R processes — works on
# both Windows and Linux/Mac without CUDA conflicts.
plan(multisession, workers = N_WORKERS)

t0 <- proc.time()[["elapsed"]]  # Record start time for elapsed-time reporting.

# with_progress() enables the progress bar display.
# future_map_dfr() applies run_combo() to each row index of GRID in parallel,
# then row-binds the resulting data frames into one combined data frame.
# .progress = TRUE activates the progressr progress bar.
with_progress({
  results <- future_map_dfr(seq_len(nrow(GRID)), run_combo, .progress = TRUE)
})

plan(sequential)  # Reset to single-threaded execution after the parallel block.

# Save results to CSV in the current working directory.
write.csv(results, "../data/grid_results_model1.csv", row.names = FALSE)

cat(sprintf("Saved: grid_results_model1.csv\n"))
cat(sprintf("frac_ob_lt_ow == 1.0 (all seeds): %d\n",
            sum(results$frac_ob_lt_ow == 1.0, na.rm = TRUE)))
cat(sprintf("frac_ob_lt_ow  > 0.9             : %d\n",
            sum(results$frac_ob_lt_ow > 0.9, na.rm = TRUE)))
cat(sprintf("Elapsed: %.0fs\n", proc.time()[["elapsed"]] - t0))
