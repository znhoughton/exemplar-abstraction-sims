# =============================================================================
# Model 1 (4-CLASS REPLICATION): Monotone simulation (the k → ∞ limit)
#
# Same model as scripts/zero-sensitivity-learner.R, generalized from 2 verb
# classes (Dative, Motion) to Jian & Manning (2026)'s full 4 classes (Dative,
# Motion, Reciprocal, Spray-load), as flagged as an open robustness check in
# the paper's Limitations section. This is a SEPARATE script — the original
# 2-class scripts in the parent directory are untouched.
#
# WHAT CHANGED vs. the 2-class version:
#   - CLASS_SIZES/N_CLASSES/CLASS_ID replace the hardcoded N_A/N_B split.
#     Dative=35, Motion=36 are Jian & Manning's real counts; Reciprocal=35,
#     Spray-load=36 are PLACEHOLDERS (real counts not available) — update
#     CLASS_SIZES below once known and rerun.
#   - Token sharing generalizes directly: item_overlap/class_overlap keep
#     their original meaning (item_overlap - class_overlap = within-class
#     sharing, now producing one disjoint token block per class instead of
#     two), just applied across N_CLASSES classes instead of 2.
#   - ow (within-class onset) generalizes trivially: mean within-class DJS
#     averaged over N_CLASSES classes instead of 2.
#   - ob (between-class onset) uses the POOLED generalization: for each verb,
#     "between" = DJS to ALL verbs outside its class (pooled across the other
#     3 classes), vs. "within" = DJS to same-class verbs. One Wilcoxon test
#     per verb, same as the 2-class version — just a larger between-set. This
#     was chosen over strict pairwise (test against each other class
#     separately) because it needs no extra design parameter (how many of the
#     3 pairwise tests must pass) and is the direct generalization of the
#     existing statistical test.
#
# Everything else (data-generation procedure, lognormal weights, DJS formula,
# onset thresholds, parameter grid) is unchanged from the 2-class version.
# =============================================================================

library(furrr)
library(progressr)

# =============================================================================
# SECTION 1: Constants
# =============================================================================

VOCAB_SIZE  <- 1000L
N_PREFERRED <- 50L

# Verb class sizes. Order: Dative, Motion, Reciprocal, Spray-load.
# Dative/Motion counts are Jian & Manning (2026) Appendix A (same as the
# 2-class scripts). Reciprocal/Spray-load are PLACEHOLDERS pending real counts.
CLASS_SIZES <- c(35L, 36L, 35L, 36L)
N_CLASSES   <- length(CLASS_SIZES)
N_VERBS     <- sum(CLASS_SIZES)                 # 142
CLASS_ID    <- rep(seq_len(N_CLASSES), CLASS_SIZES)  # length-N_VERBS class label per verb
CLASS_MEMBERS <- split(seq_len(N_VERBS), CLASS_ID)   # verb indices per class, precomputed once

N_SEEDS     <- 50L
N_WORKERS   <- 16L    # Adjust to match your machine (check with parallel::detectCores()).

BG_WEIGHT   <- 1.0

ALPHAS <- exp(seq(log(0.001), log(1.0), length.out = 25L))

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

run_one <- function(mu, sigma, n_preferred, item_overlap, class_overlap, seed) {

  set.seed(seed)

  # ---------------------------------------------------------------------------
  # Step 1: Build the token pool.
  #   n_cross  tokens shared by ALL verbs in ALL N_CLASSES classes
  #   n_within tokens shared within ONE class only (each class gets its own
  #            disjoint block of n_within tokens)
  #   n_idio   tokens unique to each individual verb
  # ---------------------------------------------------------------------------

  n_cross  <- round(class_overlap * n_preferred)
  n_within <- round((item_overlap - class_overlap) * n_preferred)
  n_idio   <- n_preferred - n_cross - n_within

  cross_tok <- seq_len(n_cross)

  # One disjoint within-class token block per class.
  within_tok <- lapply(seq_len(N_CLASSES), function(c)
    n_cross + (c - 1L) * n_within + seq_len(n_within))

  idio_pool <- (n_cross + N_CLASSES * n_within + 1L):VOCAB_SIZE

  # Assign idiosyncratic tokens GLOBALLY across all N_VERBS verbs (same
  # reuse-spreading approach as the 2-class version — see its comments).
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
  # Step 3: Sweep α and check onset criteria
  # ---------------------------------------------------------------------------

  uni <- 1.0 / VOCAB_SIZE

  ob_alpha  <- NA_real_
  ow_alpha  <- NA_real_
  ow_streak <- 0L
  ow_start  <- NA_real_

  for (alpha in ALPHAS) {

    P_obs   <- (1.0 - alpha) * uni + alpha * P_true
    jsd_mat <- pairwise_jsd(P_obs)  # N_VERBS x N_VERBS

    # ---- ob check (between-class onset): POOLED generalization ----
    if (is.na(ob_alpha)) {
      n_sig <- 0L
      for (i in seq_len(N_VERBS)) {
        ci      <- CLASS_ID[i]
        within  <- jsd_mat[i, setdiff(CLASS_MEMBERS[[ci]], i)]        # same-class verbs
        between <- jsd_mat[i, unlist(CLASS_MEMBERS[-ci], use.names = FALSE)]  # all other verbs, pooled
        if (suppressWarnings(
              wilcox.test(between, within, alternative = "greater",
                          exact = FALSE)$p.value) < P_THRESH) {
          n_sig <- n_sig + 1L
        }
      }
      if (n_sig / N_VERBS >= CLASS_FRAC) ob_alpha <- alpha
    }

    # ---- ow check (within-class onset): mean over N_CLASSES classes ----
    if (is.na(ow_alpha)) {
      within_dj <- vapply(seq_len(N_CLASSES), function(c) {
        idx <- CLASS_MEMBERS[[c]]
        jc  <- jsd_mat[idx, idx]
        mean(jc[upper.tri(jc)])
      }, numeric(1))
      within_mean <- mean(within_dj)

      if (within_mean > DJS_THRESH) {
        if (is.na(ow_start)) ow_start <- alpha
        ow_streak <- ow_streak + 1L
        if (ow_streak >= SUSTAIN) ow_alpha <- ow_start
      } else {
        ow_streak <- 0L
        ow_start  <- NA_real_
      }
    }

    if (!is.na(ob_alpha) && !is.na(ow_alpha)) break
  }

  c(ob_alpha = ob_alpha, ow_alpha = ow_alpha)
}

# =============================================================================
# SECTION 5: Grid search — run all parameter combinations
#
# run_combo() returns one row PER SEED (raw ob_alpha/ow_alpha), matching the
# CI-enabling pattern used in ../zero-sensitivity-learner.R.
# =============================================================================

run_combo <- function(row_i) {
  p  <- GRID[row_i, ]
  ob <- numeric(N_SEEDS)
  ow <- numeric(N_SEEDS)

  for (s in seq_len(N_SEEDS)) {
    res   <- run_one(p$mu, p$sigma, N_PREFERRED, p$item_overlap, p$class_overlap, seed = s)
    ob[s] <- res["ob_alpha"]
    ow[s] <- res["ow_alpha"]
  }

  data.frame(
    mu            = p$mu,
    sigma         = p$sigma,
    item_overlap  = p$item_overlap,
    class_overlap = p$class_overlap,
    n_preferred   = N_PREFERRED,
    seed          = seq_len(N_SEEDS),
    ob_alpha      = ob,
    ow_alpha      = ow
  )
}

summarize_grid <- function(per_seed) {
  group_cols <- c("mu", "sigma", "item_overlap", "class_overlap")
  combo_key  <- do.call(paste, c(per_seed[group_cols], sep = "\r"))

  rows <- lapply(split(seq_len(nrow(per_seed)), combo_key), function(idx) {
    d  <- per_seed[idx, ]
    ob <- d$ob_alpha
    ow <- d$ow_alpha

    both      <- !is.na(ob) & !is.na(ow)
    ob_lt_ow  <- both & (ob < ow)
    log_ratio <- if (any(ob_lt_ow)) log(ow[ob_lt_ow] / ob[ob_lt_ow]) else numeric(0)

    data.frame(
      d[1, group_cols, drop = FALSE],
      n_preferred    = N_PREFERRED,
      frac_detected  = mean(both),
      frac_ob_lt_ow  = mean(ob_lt_ow),
      mean_ob        = if (any(both)) mean(ob[both]) else NA_real_,
      mean_ow        = if (any(both)) mean(ow[both]) else NA_real_,
      sd_ob          = if (sum(both) > 1) sd(ob[both]) else NA_real_,
      sd_ow          = if (sum(both) > 1) sd(ow[both]) else NA_real_,
      n_both         = sum(both),
      mean_log_ratio = if (length(log_ratio) > 0) mean(log_ratio) else NA_real_,
      sd_log_ratio   = if (length(log_ratio) > 1) sd(log_ratio) else NA_real_,
      n_ob_lt_ow     = length(log_ratio)
    )
  })

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out[order(out$mu, out$sigma, out$item_overlap, out$class_overlap), ]
}

# =============================================================================
# SECTION 6: Run the grid search in parallel
# =============================================================================

cat(sprintf("4-class replication: %d verbs (classes: %s)\n",
            N_VERBS, paste(CLASS_SIZES, collapse = ", ")))
cat(sprintf("%d parameter combinations x %d seeds\nWorkers: %d\n\n",
            nrow(GRID), N_SEEDS, N_WORKERS))

plan(multisession, workers = N_WORKERS)
t0 <- proc.time()[["elapsed"]]

with_progress({
  per_seed <- future_map_dfr(seq_len(nrow(GRID)), run_combo, .progress = TRUE)
})

plan(sequential)

results <- summarize_grid(per_seed)

write.csv(per_seed, "../../data/four-class-replication/grid_results_model1_4class_per_seed.csv", row.names = FALSE)
write.csv(results,  "../../data/four-class-replication/grid_results_model1_4class.csv",          row.names = FALSE)

cat(sprintf("Saved: grid_results_model1_4class.csv, grid_results_model1_4class_per_seed.csv\n"))
cat(sprintf("frac_ob_lt_ow == 1.0 (all seeds): %d\n",
            sum(results$frac_ob_lt_ow == 1.0, na.rm = TRUE)))
cat(sprintf("frac_ob_lt_ow  > 0.9             : %d\n",
            sum(results$frac_ob_lt_ow > 0.9, na.rm = TRUE)))
cat(sprintf("Elapsed: %.0fs\n", proc.time()[["elapsed"]] - t0))
