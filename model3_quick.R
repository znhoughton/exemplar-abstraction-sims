# =============================================================================
# Model 3 quick run: fixed sigma=0.5, item_overlap=0.7, class_overlap=0.2
# (max overlap gap), all mu/k/zipf_s. Used to preview the k x zipf_s table
# while the full Model 3 grid runs in the background.
# =============================================================================

library(furrr)
library(progressr)

VOCAB_SIZE  <- 1000L
N_PREFERRED <- 50L
N_A         <- 35L
N_B         <- 36L
N_VERBS     <- N_A + N_B

N_SEEDS     <- 50L
N_WORKERS   <- 2L
BG_WEIGHT   <- 1.0

MAX_NOBS_TOTAL <- N_VERBS * 5000L
N_OBS_GRID <- unique(round(exp(seq(log(N_VERBS), log(MAX_NOBS_TOTAL),
                                    length.out = 20L))))

P_THRESH   <- 0.001
CLASS_FRAC <- 0.10
DJS_THRESH <- 0.01
SUSTAIN    <- 3L

LOG_FILE <- "model3_quick_progress.log"

GRID <- expand.grid(
  mu            = c(30, 60),
  sigma         = 0.5,
  item_overlap  = 0.7,
  class_overlap = 0.2,
  add_k         = c(0.01, 0.1, 3.0),
  zipf_s        = c(0.5, 1.5),
  stringsAsFactors = FALSE
)

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

run_one <- function(mu, sigma, n_preferred, item_overlap, class_overlap,
                    add_k, zipf_s, seed) {
  set.seed(seed)

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

  log_mean <- log(mu) - 0.5 * sigma^2

  build_dists <- function(token_list) {
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)
    for (i in seq_along(token_list))
      P[i, token_list[[i]]] <- rlnorm(n_preferred, log_mean, sigma)
    P / rowSums(P)
  }

  P_true <- rbind(build_dists(A_tokens), build_dists(B_tokens))

  ranks      <- sample(N_VERBS)
  zipf_probs <- (1.0 / ranks^zipf_s)
  zipf_probs <- zipf_probs / sum(zipf_probs)

  verb_draws   <- sample(N_VERBS, MAX_NOBS_TOTAL, replace = TRUE, prob = zipf_probs)
  token_draws  <- integer(MAX_NOBS_TOTAL)
  verb_streams <- vector("list", N_VERBS)
  for (v in seq_len(N_VERBS)) {
    idx <- which(verb_draws == v)
    if (length(idx) > 0L) {
      toks <- sample(VOCAB_SIZE, length(idx), replace = TRUE, prob = P_true[v, ])
      token_draws[idx]  <- toks
      verb_streams[[v]] <- toks
    } else {
      verb_streams[[v]] <- integer(0L)
    }
  }

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

  ob_nobs   <- NA_integer_
  ow_nobs   <- NA_integer_
  ow_streak <- 0L
  ow_start  <- NA_integer_

  for (ci in seq_along(N_OBS_GRID)) {
    n_total <- N_OBS_GRID[ci]

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
  cat(sprintf("%s  combo %3d / %d done\n", format(Sys.time(), "%H:%M:%S"),
              row_i, nrow(GRID)),
      file = LOG_FILE, append = TRUE)
  out
}

cat(sprintf("%d combos x %d seeds | workers: %d\n", nrow(GRID), N_SEEDS, N_WORKERS))
cat(sprintf("Started: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    file = LOG_FILE, append = FALSE)

plan(multisession, workers = N_WORKERS)
t0 <- proc.time()[["elapsed"]]

with_progress({
  results <- future_map_dfr(seq_len(nrow(GRID)), run_combo, .progress = TRUE)
})

plan(sequential)

write.csv(results, "grid_results_model3_quick.csv", row.names = FALSE)
cat(sprintf("Saved: grid_results_model3_quick.csv  (%.0fs)\n",
            proc.time()[["elapsed"]] - t0))
