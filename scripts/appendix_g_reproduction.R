# =============================================================================
# Attempt to reproduce Appendix G (Threshold Robustness) numbers.
#
# Case 1: Zero-Sensitivity Learner, mu=100, sigma=0.5, item_overlap=0.7,
#         class_overlap=0.2, 50 seeds, fine alpha grid.
# Case 2: Variable-Sensitivity Learner, k=0.001, same structural params,
#         50 seeds, fine n_obs grid.
#
# For each seed we record, at each step, frac_sig (fraction of verbs passing
# the one-tailed Mann-Whitney test at p < 0.001) and within_mean (mean
# within-class DJS). These two raw trajectories let us sweep CLASS_FRAC and
# DJS_THRESH post-hoc without re-running the tests, exactly matching how
# the paper describes precomputing "the full DJS-matrix trajectory" once
# and sweeping thresholds afterward.
# =============================================================================

VOCAB_SIZE  <- 1000L
N_PREFERRED <- 50L
N_A         <- 35L
N_B         <- 36L
BG_WEIGHT   <- 1.0

MU            <- 100
SIGMA         <- 0.5
ITEM_OVERLAP  <- 0.7
CLASS_OVERLAP <- 0.2

N_SEEDS  <- 50L
P_THRESH <- 0.001

# Fine grids: dense at the low end where onsets happen, log-spaced beyond that.
ALPHA_GRID  <- sort(unique(c(
  exp(seq(log(0.0001), log(0.01), length.out = 40)),
  exp(seq(log(0.01), log(1.0), length.out = 80))
)))
NOBS_GRID   <- sort(unique(round(c(
  1:30,
  exp(seq(log(31), log(5000), length.out = 120))
))))

cat(sprintf("Alpha grid: %d points | Nobs grid: %d points\n",
            length(ALPHA_GRID), length(NOBS_GRID)))

pairwise_jsd <- function(P) {
  N   <- nrow(P)
  eps <- 1e-30
  H   <- -rowSums(P * log(P + eps))
  jsd_mat <- matrix(0.0, N, N)
  for (i in seq_len(N - 1L)) {
    j_idx <- (i + 1L):N
    M     <- 0.5 * (matrix(P[i, ], nrow = length(j_idx), ncol = VOCAB_SIZE, byrow = TRUE)
                    + P[j_idx, , drop = FALSE])
    H_M   <- -rowSums(M * log(M + eps))
    d     <- pmax(0.0, H_M - 0.5 * (H[i] + H[j_idx]))
    jsd_mat[i, j_idx] <- d
    jsd_mat[j_idx, i] <- d
  }
  jsd_mat
}

frac_sig_and_within <- function(P_hat) {
  jsd_mat <- pairwise_jsd(P_hat)

  n_sig <- 0L
  for (i in seq_len(N_A)) {
    within  <- jsd_mat[i, setdiff(seq_len(N_A), i)]
    between <- jsd_mat[i, (N_A + 1L):(N_A + N_B)]
    if (suppressWarnings(wilcox.test(between, within, alternative = "greater",
                                      exact = FALSE)$p.value) < P_THRESH)
      n_sig <- n_sig + 1L
  }
  for (j in seq_len(N_B)) {
    i       <- N_A + j
    within  <- jsd_mat[i, N_A + setdiff(seq_len(N_B), j)]
    between <- jsd_mat[i, seq_len(N_A)]
    if (suppressWarnings(wilcox.test(between, within, alternative = "greater",
                                      exact = FALSE)$p.value) < P_THRESH)
      n_sig <- n_sig + 1L
  }
  frac_sig <- n_sig / (N_A + N_B)

  jA <- jsd_mat[seq_len(N_A), seq_len(N_A)]
  jB <- jsd_mat[(N_A + 1L):(N_A + N_B), (N_A + 1L):(N_A + N_B)]
  within_mean <- (mean(jA[upper.tri(jA)]) + mean(jB[upper.tri(jB)])) / 2.0

  c(frac_sig = frac_sig, within_mean = within_mean)
}

build_token_pool_and_true_dists <- function(seed) {
  set.seed(seed)
  n_cross  <- round(CLASS_OVERLAP * N_PREFERRED)
  n_within <- round((ITEM_OVERLAP - CLASS_OVERLAP) * N_PREFERRED)
  n_idio   <- N_PREFERRED - n_cross - n_within

  cross_tok    <- seq_len(n_cross)
  within_A_tok <- n_cross + seq_len(n_within)
  within_B_tok <- n_cross + n_within + seq_len(n_within)
  idio_pool    <- (n_cross + 2L * n_within + 1L):VOCAB_SIZE

  # Global idio-token assignment (avoids accidental cross-verb collisions); see
  # zipfian-vsl.R for the full rationale.
  n_total_idio  <- (N_A + N_B) * n_idio
  n_copies      <- ceiling(n_total_idio / length(idio_pool))
  idio_shuffled <- unlist(replicate(n_copies, sample(idio_pool), simplify = FALSE))[seq_len(n_total_idio)]
  idio_assignments <- split(idio_shuffled, ceiling(seq_len(n_total_idio) / n_idio))

  make_verb_tokens <- function(class_shared, idio_tokens)
    c(cross_tok, class_shared, idio_tokens)

  A_tokens <- lapply(seq_len(N_A), function(i)
    make_verb_tokens(within_A_tok, idio_assignments[[i]]))
  B_tokens <- lapply(seq_len(N_B), function(i)
    make_verb_tokens(within_B_tok, idio_assignments[[N_A + i]]))

  log_mean <- log(MU) - 0.5 * SIGMA^2
  build_dists <- function(token_list) {
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)
    for (i in seq_along(token_list))
      P[i, token_list[[i]]] <- rlnorm(N_PREFERRED, log_mean, SIGMA)
    P / rowSums(P)
  }
  rbind(build_dists(A_tokens), build_dists(B_tokens))
}

# ---------------------------------------------------------------------------
# Case 1: Zero-Sensitivity Learner
# ---------------------------------------------------------------------------
run_case1 <- function(seed) {
  P_true <- build_token_pool_and_true_dists(seed)
  uni <- 1.0 / VOCAB_SIZE
  out <- matrix(NA_real_, nrow = length(ALPHA_GRID), ncol = 2)
  for (i in seq_along(ALPHA_GRID)) {
    a <- ALPHA_GRID[i]
    P_obs <- (1 - a) * uni + a * P_true
    r <- frac_sig_and_within(P_obs)
    out[i, ] <- r
  }
  data.frame(step = ALPHA_GRID, frac_sig = out[,1], within_mean = out[,2], seed = seed)
}

# ---------------------------------------------------------------------------
# Case 2: Variable-Sensitivity Learner, k = 0.001
# ---------------------------------------------------------------------------
run_case2 <- function(seed, add_k = 0.001) {
  set.seed(seed)
  P_true <- build_token_pool_and_true_dists(seed)  # same seed -> same true dists as case 1
  MAX_NOBS <- max(NOBS_GRID)

  presample <- function(P_rows)
    lapply(seq_len(nrow(P_rows)), function(i)
      sample(seq_len(VOCAB_SIZE), size = MAX_NOBS, replace = TRUE, prob = P_rows[i, ]))

  draws <- presample(P_true)

  out <- matrix(NA_real_, nrow = length(NOBS_GRID), ncol = 2)
  for (i in seq_along(NOBS_GRID)) {
    n_obs <- NOBS_GRID[i]
    P_hat <- do.call(rbind, lapply(draws, function(d) {
      counts <- tabulate(d[seq_len(n_obs)], nbins = VOCAB_SIZE)
      (counts + add_k) / (n_obs + add_k * VOCAB_SIZE)
    }))
    r <- frac_sig_and_within(P_hat)
    out[i, ] <- r
  }
  data.frame(step = NOBS_GRID, frac_sig = out[,1], within_mean = out[,2], seed = seed)
}

t0 <- proc.time()[["elapsed"]]

cat("Running Case 1 (ZSL)...\n")
case1_results <- do.call(rbind, lapply(seq_len(N_SEEDS), function(s) {
  if (s %% 10 == 0) cat(sprintf("  seed %d/%d (%.0fs elapsed)\n", s, N_SEEDS, proc.time()[["elapsed"]] - t0))
  run_case1(s)
}))
write.csv(case1_results, "../data/appendix_g_case1_raw.csv", row.names = FALSE)
cat(sprintf("Case 1 done (%.0fs)\n", proc.time()[["elapsed"]] - t0))

cat("Running Case 2 (VSL, k=0.001)...\n")
case2_results <- do.call(rbind, lapply(seq_len(N_SEEDS), function(s) {
  if (s %% 10 == 0) cat(sprintf("  seed %d/%d (%.0fs elapsed)\n", s, N_SEEDS, proc.time()[["elapsed"]] - t0))
  run_case2(s)
}))
write.csv(case2_results, "../data/appendix_g_case2_raw.csv", row.names = FALSE)
cat(sprintf("Case 2 done (%.0fs)\n", proc.time()[["elapsed"]] - t0))

# =============================================================================
# Threshold sweep
# =============================================================================

CF_VALS <- c(0.01, 0.05, 0.10, 0.20, 0.30)
DT_VALS <- c(0.001, 0.005, 0.010, 0.020, 0.050)
SUSTAIN <- 3L

onset_ob <- function(frac_sig_seq, class_frac) {
  idx <- which(frac_sig_seq >= class_frac)
  if (length(idx) == 0) return(NA_integer_)
  idx[1]
}

onset_ow <- function(within_seq, djs_thresh, sustain = 3L) {
  streak <- 0L
  start  <- NA_integer_
  for (i in seq_along(within_seq)) {
    if (within_seq[i] > djs_thresh) {
      if (streak == 0L) start <- i
      streak <- streak + 1L
      if (streak >= sustain) return(start)
    } else {
      streak <- 0L
      start  <- NA_integer_
    }
  }
  NA_integer_
}

sweep <- function(results, label) {
  cat(sprintf("\n=== %s ===\n", label))
  for (cf in CF_VALS) {
    row_ob <- row_ow <- row_neither <- numeric(length(DT_VALS))
    for (di in seq_along(DT_VALS)) {
      dt <- DT_VALS[di]
      ob_lt_ow <- ow_lt_ob <- neither <- 0L
      seeds <- unique(results$seed)
      for (s in seeds) {
        g <- results[results$seed == s, ]
        g <- g[order(g$step), ]
        ob_i <- onset_ob(g$frac_sig, cf)
        ow_i <- onset_ow(g$within_mean, dt, SUSTAIN)
        if (is.na(ob_i) || is.na(ow_i)) {
          neither <- neither + 1L
        } else if (ow_i < ob_i) {
          ow_lt_ob <- ow_lt_ob + 1L
        } else if (ob_i < ow_i) {
          ob_lt_ow <- ob_lt_ow + 1L
        } else {
          neither <- neither + 1L  # tie
        }
      }
      n <- length(seeds)
      row_ob[di] <- ob_lt_ow / n
      row_ow[di] <- ow_lt_ob / n
      row_neither[di] <- neither / n
    }
    cat(sprintf("CLASS_FRAC=%.2f | ob<ow: %s | ow<ob: %s | neither/tie: %s\n",
                cf,
                paste(sprintf("%.2f", row_ob), collapse=" "),
                paste(sprintf("%.2f", row_ow), collapse=" "),
                paste(sprintf("%.2f", row_neither), collapse=" ")))
  }
}

sweep(case1_results, "CASE 1 (Zero-Sensitivity Learner)")
sweep(case2_results, "CASE 2 (Variable-Sensitivity Learner, k=0.001)")

cat(sprintf("\nTotal elapsed: %.0fs\n", proc.time()[["elapsed"]] - t0))
