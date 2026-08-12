# =============================================================================
# Model 2 (4-CLASS REPLICATION): Variable-Sensitivity Learner (Dirichlet-Multinomial)
#
# Same model as ../variable-sensitivity-learner.R, generalized to 4
# verb classes. See zero-sensitivity-learner-4class.R in this directory for
# the full explanation of what changed (CLASS_SIZES/N_CLASSES/CLASS_ID, the
# pooled ob-test generalization, class-averaged ow). This is a SEPARATE
# script — ../variable-sensitivity-learner.R is untouched.
# =============================================================================

library(furrr)
library(progressr)

# =============================================================================
# SECTION 1: Constants
# =============================================================================

VOCAB_SIZE  <- 1000L
N_PREFERRED <- 50L

# Verb class sizes. Order: Dative, Motion, Reciprocal, Spray-load.
# Dative/Motion counts are Jian & Manning (2026) Appendix A. Reciprocal/
# Spray-load are PLACEHOLDERS pending real counts.
CLASS_SIZES <- c(35L, 36L, 35L, 36L)
N_CLASSES   <- length(CLASS_SIZES)
N_VERBS     <- sum(CLASS_SIZES)
CLASS_ID    <- rep(seq_len(N_CLASSES), CLASS_SIZES)
CLASS_MEMBERS <- split(seq_len(N_VERBS), CLASS_ID)

N_SEEDS     <- 50L
N_WORKERS   <- 15L

BG_WEIGHT   <- 1.0

MAX_NOBS    <- 5000L
N_OBS_GRID  <- unique(round(exp(seq(log(1), log(MAX_NOBS), length.out = 20L))))

P_THRESH   <- 0.001
CLASS_FRAC <- 0.10   # Fraction of verbs (now out of N_VERBS = 142) that must pass.
DJS_THRESH <- 0.01
SUSTAIN    <- 3L

# =============================================================================
# SECTION 2: Parameter grid (identical to the 2-class version)
# =============================================================================

GRID <- expand.grid(
  mu            = c(10, 30, 60, 100),
  sigma         = c(0.5, 1.0, 1.5),
  item_overlap  = c(0.5, 0.6, 0.7),
  class_overlap = c(0.2, 0.3, 0.4),
  add_k         = c(0.001, 0.01, 0.1, 0.5, 1.0),
  stringsAsFactors = FALSE
)
GRID <- GRID[GRID$class_overlap < GRID$item_overlap, ]

# =============================================================================
# SECTION 3: Pairwise Jensen-Shannon Divergence (identical to the 2-class version)
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

run_one <- function(mu, sigma, n_preferred, item_overlap, class_overlap, add_k, seed) {

  set.seed(seed)

  # ---------------------------------------------------------------------------
  # Step 1: Build the token pool (see zero-sensitivity-learner-4class.R for
  # the per-class-block generalization of within-class tokens).
  # ---------------------------------------------------------------------------

  n_cross  <- round(class_overlap * n_preferred)
  n_within <- round((item_overlap - class_overlap) * n_preferred)
  n_idio   <- n_preferred - n_cross - n_within

  cross_tok  <- seq_len(n_cross)
  within_tok <- lapply(seq_len(N_CLASSES), function(c)
    n_cross + (c - 1L) * n_within + seq_len(n_within))
  idio_pool  <- (n_cross + N_CLASSES * n_within + 1L):VOCAB_SIZE

  n_total_idio  <- N_VERBS * n_idio
  n_copies      <- ceiling(n_total_idio / length(idio_pool))
  idio_shuffled <- unlist(replicate(n_copies, sample(idio_pool), simplify = FALSE))[seq_len(n_total_idio)]
  idio_assignments <- split(idio_shuffled, ceiling(seq_len(n_total_idio) / n_idio))

  make_verb_tokens <- function(class_shared, idio_tokens) {
    c(cross_tok, class_shared, idio_tokens)
  }

  verb_tokens <- lapply(seq_len(N_VERBS), function(i)
    make_verb_tokens(within_tok[[CLASS_ID[i]]], idio_assignments[[i]]))

  # ---------------------------------------------------------------------------
  # Step 2: Build true distributions P_verb (identical formula to 2-class version)
  # ---------------------------------------------------------------------------

  log_mean <- log(mu) - 0.5 * sigma^2

  build_dists <- function(token_list) {
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)
    for (i in seq_along(token_list)) {
      P[i, token_list[[i]]] <- rlnorm(n_preferred, log_mean, sigma)
    }
    P / rowSums(P)
  }

  P_true <- build_dists(verb_tokens)  # N_VERBS x V

  # ---------------------------------------------------------------------------
  # Step 3: Pre-sample token sequences (identical mechanics to 2-class version,
  # just applied to all N_VERBS verbs in one list instead of two).
  # ---------------------------------------------------------------------------

  presample <- function(P_true_rows) {
    lapply(seq_len(nrow(P_true_rows)), function(i) {
      sample(seq_len(VOCAB_SIZE), size = MAX_NOBS, replace = TRUE,
             prob = P_true_rows[i, ])
    })
  }

  verb_draws <- presample(P_true)  # List of N_VERBS integer vectors, each length MAX_NOBS.

  # ---------------------------------------------------------------------------
  # Step 4: Sweep corpus size and check onset criteria
  # ---------------------------------------------------------------------------

  ob_nobs   <- NA_integer_
  ow_nobs   <- NA_integer_
  ow_streak <- 0L
  ow_start  <- NA_integer_

  for (n_obs in N_OBS_GRID) {

    smooth_verb <- function(draws) {
      counts <- tabulate(draws[seq_len(n_obs)], nbins = VOCAB_SIZE)
      (counts + add_k) / (n_obs + add_k * VOCAB_SIZE)
    }

    P_hat <- do.call(rbind, lapply(verb_draws, smooth_verb))  # N_VERBS x V

    jsd_mat <- pairwise_jsd(P_hat)

    # ---- ob check (between-class onset): POOLED generalization ----
    if (is.na(ob_nobs)) {
      n_sig <- 0L
      for (i in seq_len(N_VERBS)) {
        ci      <- CLASS_ID[i]
        within  <- jsd_mat[i, setdiff(CLASS_MEMBERS[[ci]], i)]
        between <- jsd_mat[i, unlist(CLASS_MEMBERS[-ci], use.names = FALSE)]
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH) {
          n_sig <- n_sig + 1L
        }
      }
      if (n_sig / N_VERBS >= CLASS_FRAC) ob_nobs <- n_obs
    }

    # ---- ow check (within-class onset): mean over N_CLASSES classes ----
    if (is.na(ow_nobs)) {
      within_dj <- vapply(seq_len(N_CLASSES), function(c) {
        idx <- CLASS_MEMBERS[[c]]
        jc  <- jsd_mat[idx, idx]
        mean(jc[upper.tri(jc)])
      }, numeric(1))
      within_mean <- mean(within_dj)

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
                     p$add_k, seed = s)
    ob[s] <- res["ob_nobs"]
    ow[s] <- res["ow_nobs"]
  }

  data.frame(
    mu            = p$mu,
    sigma         = p$sigma,
    item_overlap  = p$item_overlap,
    class_overlap = p$class_overlap,
    add_k         = p$add_k,
    n_preferred   = N_PREFERRED,
    seed          = seq_len(N_SEEDS),
    ob_nobs       = ob,
    ow_nobs       = ow
  )
}

summarize_grid <- function(per_seed) {
  group_cols <- c("mu", "sigma", "item_overlap", "class_overlap", "add_k")
  combo_key  <- do.call(paste, c(per_seed[group_cols], sep = "\r"))

  rows <- lapply(split(seq_len(nrow(per_seed)), combo_key), function(idx) {
    d  <- per_seed[idx, ]
    ob <- d$ob_nobs
    ow <- d$ow_nobs

    both      <- !is.na(ob) & !is.na(ow)
    ob_lt_ow  <- both & (ob < ow)
    ow_lt_ob  <- both & (ow < ob)
    log_ratio <- if (any(ob_lt_ow)) log(ow[ob_lt_ow] / ob[ob_lt_ow]) else numeric(0)

    data.frame(
      d[1, group_cols, drop = FALSE],
      n_preferred    = N_PREFERRED,
      frac_detected  = mean(both),
      frac_ob_lt_ow  = mean(ob_lt_ow),
      frac_ow_lt_ob  = mean(ow_lt_ob),
      mean_ob_nobs   = if (any(both)) mean(ob[both]) else NA_real_,
      mean_ow_nobs   = if (any(both)) mean(ow[both]) else NA_real_,
      sd_ob_nobs     = if (sum(both) > 1) sd(ob[both]) else NA_real_,
      sd_ow_nobs     = if (sum(both) > 1) sd(ow[both]) else NA_real_,
      n_both         = sum(both),
      mean_log_ratio = if (length(log_ratio) > 0) mean(log_ratio) else NA_real_,
      sd_log_ratio   = if (length(log_ratio) > 1) sd(log_ratio) else NA_real_,
      n_ob_lt_ow     = length(log_ratio)
    )
  })

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[order(out$mu, out$sigma, out$item_overlap, out$class_overlap, out$add_k), ]
}

# =============================================================================
# SECTION 6: Run the grid search in parallel
# =============================================================================

cat(sprintf("4-class replication: %d verbs (classes: %s)\n",
            N_VERBS, paste(CLASS_SIZES, collapse = ", ")))
cat(sprintf("%d parameter combinations x %d seeds\nWorkers: %d\n",
            nrow(GRID), N_SEEDS, N_WORKERS))
cat(sprintf("n_obs grid: %d values from %d to %d\n\n",
            length(N_OBS_GRID), min(N_OBS_GRID), max(N_OBS_GRID)))

plan(multisession, workers = N_WORKERS)
t0 <- proc.time()[["elapsed"]]

with_progress({
  per_seed <- future_map_dfr(seq_len(nrow(GRID)), run_combo, .progress = TRUE)
})

plan(sequential)

results <- summarize_grid(per_seed)

write.csv(per_seed, "../../data/four-class-replication/grid_results_model2_4class_per_seed.csv", row.names = FALSE)
write.csv(results,  "../../data/four-class-replication/grid_results_model2_4class.csv",          row.names = FALSE)

cat(sprintf("Saved: grid_results_model2_4class.csv, grid_results_model2_4class_per_seed.csv\n"))
cat(sprintf("frac_ow_lt_ob == 1.0 (all seeds): %d\n",
            sum(results$frac_ow_lt_ob == 1.0, na.rm = TRUE)))
cat(sprintf("frac_ob_lt_ow == 1.0 (all seeds): %d\n",
            sum(results$frac_ob_lt_ow == 1.0, na.rm = TRUE)))
cat(sprintf("Elapsed: %.0fs\n", proc.time()[["elapsed"]] - t0))
