# =============================================================================
# Model 3: Zipfian Verb Frequency
#
# Identical to Model 2 (Variable-Sensitivity Learner) except that verb
# observations are allocated according to a Zipfian distribution rather than
# uniformly. At each total-token checkpoint N_total, different verbs have
# accumulated different numbers of observations proportional to their Zipfian
# frequency rank.
#
# PROCEDURE:
#   For each draw in 1:MAX_NOBS_TOTAL:
#     1. Sample a verb from the Zipfian distribution over the 71 verbs.
#     2. Sample a single token from that verb's true distribution P_v.
#     3. Increment count(token | verb) by 1.
#   At each checkpoint N_total, apply add-k smoothing per verb:
#     P_hat_v(t) = (count(t | v, N_total) + k) / (obs_v(N_total) + k * V)
#   where obs_v(N_total) is the number of draws assigned to verb v so far.
#
# CALIBRATION: N_total is on a scale such that N_total / N_VERBS gives the
# MEAN observations per verb. This makes the N_total axis directly comparable
# to Model 2's n_obs axis (set N_total = n_obs * N_VERBS for a fair comparison
# at the same total observation budget).
#
# VERB RANKS: randomly assigned per seed, so results average over which
# specific verbs happen to be frequent or rare.
#
# KEY PREDICTION: rare verbs (few observations) behave like low-k learners
# (noisy, within-class DJS spikes early), while frequent verbs behave like
# high-k learners. Net effect: higher k is required to achieve o_b < o_w
# compared to the uniform-sampling Model 2, and the transition should shift
# rightward in k.
# =============================================================================

library(furrr)
library(progressr)

# =============================================================================
# SECTION 1: Constants
# =============================================================================

VOCAB_SIZE  <- 1000L
N_PREFERRED <- 50L
N_A         <- 35L
N_B         <- 36L
N_VERBS     <- N_A + N_B    # 71

N_SEEDS     <- 50L
N_WORKERS   <- 6L
BG_WEIGHT   <- 1.0

LOG_FILE    <- "model3_progress.log"

# Total tokens across all 71 verbs. At the maximum checkpoint, the MEAN
# observations per verb = MAX_NOBS_TOTAL / N_VERBS = 5,000, matching Model 2.
MAX_NOBS_TOTAL <- N_VERBS * 5000L   # = 355,000

# 20 log-spaced checkpoints (total tokens). Minimum = N_VERBS so that each
# verb has approximately 1 observation in expectation at the first checkpoint.
N_OBS_GRID <- unique(round(exp(seq(log(N_VERBS), log(MAX_NOBS_TOTAL),
                                    length.out = 20L))))

P_THRESH   <- 0.001
CLASS_FRAC <- 0.10
DJS_THRESH <- 0.01
SUSTAIN    <- 3L

# =============================================================================
# SECTION 2: Parameter grid
# =============================================================================

GRID <- expand.grid(
  mu            = c(10, 30, 60, 100),
  sigma         = c(0.5, 1.0, 1.5),
  item_overlap  = c(0.5, 0.6, 0.7),
  class_overlap = c(0.2, 0.3, 0.4),
  add_k         = c(0.001, 0.01, 0.1, 0.5, 1.0),
  zipf_s        = c(0.5, 1.0, 1.5),  # Zipf exponent. s=1 matches typical English;
                                       # s=0.5 is flatter; s=1.5 is more skewed.
  stringsAsFactors = FALSE
)
GRID <- GRID[GRID$class_overlap < GRID$item_overlap, ]

# =============================================================================
# SECTION 3: Pairwise Jensen-Shannon Divergence (identical to Models 1 and 2)
# =============================================================================

pairwise_jsd <- function(P) {
  N   <- nrow(P)
  eps <- 1e-30
  H   <- -rowSums(P * log(P + eps))
  jsd_mat <- matrix(0.0, N, N)
  for (i in seq_len(N - 1L)) {
    j_idx <- (i + 1L):N
    M     <- 0.5 * (matrix(P[i, ], nrow = length(j_idx), ncol = VOCAB_SIZE,
                            byrow = TRUE) + P[j_idx, , drop = FALSE])
    H_M   <- -rowSums(M * log(M + eps))
    d     <- pmax(0.0, H_M - 0.5 * (H[i] + H[j_idx]))
    jsd_mat[i, j_idx] <- d
    jsd_mat[j_idx, i] <- d
  }
  jsd_mat
}

# =============================================================================
# SECTION 4: Single simulation run
# =============================================================================

run_one <- function(mu, sigma, n_preferred, item_overlap, class_overlap,
                    add_k, zipf_s, seed) {

  set.seed(seed)

  # ---------------------------------------------------------------------------
  # Step 1: Build token pool (identical to Models 1 and 2)
  # ---------------------------------------------------------------------------

  n_cross  <- round(class_overlap * n_preferred)
  n_within <- round((item_overlap - class_overlap) * n_preferred)
  n_idio   <- n_preferred - n_cross - n_within

  cross_tok    <- seq_len(n_cross)
  within_A_tok <- n_cross + seq_len(n_within)
  within_B_tok <- n_cross + n_within + seq_len(n_within)
  idio_pool    <- (n_cross + 2L * n_within + 1L):VOCAB_SIZE

  make_verb_tokens <- function(class_shared)
    c(cross_tok, class_shared, sample(idio_pool, n_idio, replace = FALSE))

  A_tokens <- replicate(N_A, make_verb_tokens(within_A_tok), simplify = FALSE)
  B_tokens <- replicate(N_B, make_verb_tokens(within_B_tok), simplify = FALSE)

  # ---------------------------------------------------------------------------
  # Step 2: Build true distributions (identical to Models 1 and 2)
  # ---------------------------------------------------------------------------

  log_mean <- log(mu) - 0.5 * sigma^2

  build_dists <- function(token_list) {
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)
    for (i in seq_along(token_list))
      P[i, token_list[[i]]] <- rlnorm(n_preferred, log_mean, sigma)
    P / rowSums(P)
  }

  P_true <- rbind(build_dists(A_tokens), build_dists(B_tokens))  # N_VERBS x V

  # ---------------------------------------------------------------------------
  # Step 3: Build Zipfian verb distribution (MODEL 3 SPECIFIC)
  #
  # Randomly assign ranks 1:N_VERBS to verbs so that results average over
  # which specific verbs happen to be frequent or rare across seeds.
  # Verb at rank r gets probability proportional to 1/r^zipf_s.
  # ---------------------------------------------------------------------------

  ranks      <- sample(N_VERBS)               # Random rank assignment
  zipf_probs <- (1.0 / ranks^zipf_s)
  zipf_probs <- zipf_probs / sum(zipf_probs)  # Normalize to sum to 1

  # ---------------------------------------------------------------------------
  # Step 4: Pre-sample all observations (MODEL 3 SPECIFIC)
  #
  # Draw MAX_NOBS_TOTAL verb assignments from the Zipfian distribution.
  # Then, for each verb, independently sample its token draws from P_v.
  # Processing verb-by-verb (rather than draw-by-draw) keeps this vectorized.
  # ---------------------------------------------------------------------------

  verb_draws <- sample(N_VERBS, MAX_NOBS_TOTAL, replace = TRUE,
                       prob = zipf_probs)  # Which verb is observed at each step

  # For each verb, extract the positions in verb_draws where it was observed,
  # then draw the corresponding tokens from P_true[v, ].
  # verb_streams[[v]]: ordered sequence of token IDs seen for verb v.
  token_draws  <- integer(MAX_NOBS_TOTAL)
  verb_streams <- vector("list", N_VERBS)
  for (v in seq_len(N_VERBS)) {
    idx <- which(verb_draws == v)
    if (length(idx) > 0L) {
      toks <- sample(VOCAB_SIZE, length(idx), replace = TRUE, prob = P_true[v, ])
      token_draws[idx]  <- toks
      verb_streams[[v]] <- toks         # Already in observation order
    } else {
      verb_streams[[v]] <- integer(0L)  # Verb never observed (very unlikely)
    }
  }

  # ---------------------------------------------------------------------------
  # Step 5: Pre-compute per-verb observation counts at each checkpoint
  #
  # Single O(MAX_NOBS_TOTAL) pass: scan verb_draws once and record how many
  # observations each verb has accumulated each time a checkpoint is reached.
  # This avoids repeating tabulate(verb_draws[1:n_total], ...) at each step.
  # ---------------------------------------------------------------------------

  obs_at_chk <- matrix(0L, nrow = N_VERBS, ncol = length(N_OBS_GRID))
  cur_obs    <- integer(N_VERBS)
  chk_ptr    <- 1L

  for (idx in seq_len(MAX_NOBS_TOTAL)) {
    cur_obs[verb_draws[idx]] <- cur_obs[verb_draws[idx]] + 1L
    while (chk_ptr <= length(N_OBS_GRID) && N_OBS_GRID[chk_ptr] <= idx) {
      obs_at_chk[, chk_ptr] <- cur_obs
      chk_ptr <- chk_ptr + 1L
    }
    if (chk_ptr > length(N_OBS_GRID)) break
  }

  # ---------------------------------------------------------------------------
  # Step 6: Sweep checkpoints and check onset criteria
  # ---------------------------------------------------------------------------

  ob_nobs   <- NA_integer_
  ow_nobs   <- NA_integer_
  ow_streak <- 0L
  ow_start  <- NA_integer_

  for (ci in seq_along(N_OBS_GRID)) {
    n_total <- N_OBS_GRID[ci]

    # Build add-k smoothed estimate for each verb using its accumulated counts.
    P_hat <- matrix(0.0, N_VERBS, VOCAB_SIZE)
    for (v in seq_len(N_VERBS)) {
      n_v    <- obs_at_chk[v, ci]
      counts <- if (n_v > 0L)
        tabulate(verb_streams[[v]][seq_len(n_v)], nbins = VOCAB_SIZE)
      else
        integer(VOCAB_SIZE)
      P_hat[v, ] <- (counts + add_k) / (n_v + add_k * VOCAB_SIZE)
    }

    jsd_mat <- pairwise_jsd(P_hat)

    # ---- ob check ----
    if (is.na(ob_nobs)) {
      n_sig <- 0L
      for (i in seq_len(N_A)) {
        within  <- jsd_mat[i, setdiff(seq_len(N_A), i)]
        between <- jsd_mat[i, (N_A + 1L):(N_A + N_B)]
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH)
          n_sig <- n_sig + 1L
      }
      for (j in seq_len(N_B)) {
        i       <- N_A + j
        within  <- jsd_mat[i, N_A + setdiff(seq_len(N_B), j)]
        between <- jsd_mat[i, seq_len(N_A)]
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH)
          n_sig <- n_sig + 1L
      }
      if (n_sig / N_VERBS >= CLASS_FRAC) ob_nobs <- n_total
    }

    # ---- ow check ----
    if (is.na(ow_nobs)) {
      jA <- jsd_mat[seq_len(N_A), seq_len(N_A)]
      jB <- jsd_mat[(N_A + 1L):(N_A + N_B), (N_A + 1L):(N_A + N_B)]
      within_mean <- (mean(jA[upper.tri(jA)]) + mean(jB[upper.tri(jB)])) / 2.0

      if (within_mean > DJS_THRESH) {
        if (is.na(ow_start)) ow_start <- n_total
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
# SECTION 5: Grid search
# =============================================================================

run_combo <- function(row_i) {
  p  <- GRID[row_i, ]
  ob <- integer(N_SEEDS)
  ow <- integer(N_SEEDS)

  for (s in seq_len(N_SEEDS)) {
    res   <- run_one(p$mu, p$sigma, N_PREFERRED, p$item_overlap, p$class_overlap,
                     p$add_k, p$zipf_s, seed = s)
    ob[s] <- res["ob_nobs"]
    ow[s] <- res["ow_nobs"]
  }

  both     <- !is.na(ob) & !is.na(ow)
  ob_lt_ow <- both & (ob < ow)
  ow_lt_ob <- both & (ow < ob)

  out <- data.frame(
    mu            = p$mu,
    sigma         = p$sigma,
    item_overlap  = p$item_overlap,
    class_overlap = p$class_overlap,
    add_k         = p$add_k,
    zipf_s        = p$zipf_s,
    n_preferred   = N_PREFERRED,
    frac_detected  = mean(both),
    frac_ob_lt_ow  = mean(ob_lt_ow),
    frac_ow_lt_ob  = mean(ow_lt_ob),
    frac_tie       = mean(both & !ob_lt_ow & !ow_lt_ob),
    mean_ob_nobs   = if (any(both)) mean(ob[both])        else NA_real_,
    mean_ow_nobs   = if (any(both)) mean(ow[both])        else NA_real_,
    mean_log_ratio = if (any(ob_lt_ow)) mean(log(ow[ob_lt_ow] / ob[ob_lt_ow])) else NA_real_
  )
  cat(sprintf("%s  combo %4d / %d done\n", format(Sys.time(), "%H:%M:%S"),
              row_i, nrow(GRID)),
      file = LOG_FILE, append = TRUE)
  out
}

# =============================================================================
# SECTION 6: Run
# =============================================================================

cat(sprintf("%d parameter combinations x %d seeds\nWorkers: %d\n",
            nrow(GRID), N_SEEDS, N_WORKERS))
cat(sprintf("N_total grid: %d values from %d to %d (mean n_obs: %.0f to %.0f)\n\n",
            length(N_OBS_GRID), min(N_OBS_GRID), max(N_OBS_GRID),
            min(N_OBS_GRID) / N_VERBS, max(N_OBS_GRID) / N_VERBS))

cat(sprintf("Started: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    file = LOG_FILE, append = FALSE)

plan(multisession, workers = N_WORKERS)
t0 <- proc.time()[["elapsed"]]

with_progress({
  results <- future_map_dfr(seq_len(nrow(GRID)), run_combo, .progress = TRUE)
})

plan(sequential)

write.csv(results, "grid_results_model3.csv", row.names = FALSE)
cat(sprintf("Saved: grid_results_model3.csv\n"))
cat(sprintf("frac_ow_lt_ob == 1.0 (all seeds): %d\n",
            sum(results$frac_ow_lt_ob == 1.0, na.rm = TRUE)))
cat(sprintf("frac_ob_lt_ow == 1.0 (all seeds): %d\n",
            sum(results$frac_ob_lt_ow == 1.0, na.rm = TRUE)))
cat(sprintf("Elapsed: %.0fs\n", proc.time()[["elapsed"]] - t0))
