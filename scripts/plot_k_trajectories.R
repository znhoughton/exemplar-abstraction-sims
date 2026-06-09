# =============================================================================
# plot_k_trajectories.R
#
# Generates a 4-row trajectory plot showing ob and ow criteria over corpus
# size for four values of k that straddle the ow<ob / ob<ow transition:
#
#   Before threshold (ow < ob):  k = 0.001,  k = 0.1
#   After  threshold (ob < ow):  k = 2.0,    k = 10.0
#
# Each row shows one k value. For each step n_obs we record:
#   - frac_sig:    fraction of verbs passing the Mann-Whitney ob test
#   - within_mean: mean within-class DJS (ow criterion)
#
# Results are averaged over N_SEEDS seeds with 95% CI ribbons.
# Horizontal dotted lines show the CLASS_FRAC and DJS_THRESH thresholds.
#
# Parameters fixed at mu=100, sigma=0.5, item_overlap=0.7, class_overlap=0.2
# (the combination with the clearest, cleanest signal).
# =============================================================================

library(tidyverse)

# =============================================================================
# Constants
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

K_VALUES <- c(0.001, 0.01, 1.0)
K_LABELS <- c(
  "k = 0.001",
  "k = 0.01",
  "k = 1.0"
)

N_SEEDS    <- 20L
MAX_NOBS   <- 5000L
N_OBS_GRID <- unique(round(exp(seq(log(1), log(MAX_NOBS), length.out = 40L))))

P_THRESH   <- 0.001
CLASS_FRAC <- 0.10
DJS_THRESH <- 0.01

# =============================================================================
# Pairwise JSD
# =============================================================================

pairwise_jsd <- function(P) {
  N   <- nrow(P)
  eps <- 1e-30
  H   <- -rowSums(P * log(P + eps))
  jsd_mat <- matrix(0.0, N, N)
  for (i in seq_len(N - 1L)) {
    j_idx <- (i + 1L):N
    M     <- 0.5 * (matrix(P[i, ], nrow = length(j_idx),
                            ncol = VOCAB_SIZE, byrow = TRUE) +
                    P[j_idx, , drop = FALSE])
    H_M   <- -rowSums(M * log(M + eps))
    d     <- pmax(0.0, H_M - 0.5 * (H[i] + H[j_idx]))
    jsd_mat[i, j_idx] <- d
    jsd_mat[j_idx, i] <- d
  }
  jsd_mat
}

# =============================================================================
# Single seed trajectory
# =============================================================================

run_trajectory <- function(add_k, seed) {
  set.seed(seed)

  n_cross  <- round(CLASS_OVERLAP * N_PREFERRED)
  n_within <- round((ITEM_OVERLAP - CLASS_OVERLAP) * N_PREFERRED)
  n_idio   <- N_PREFERRED - n_cross - n_within

  cross_tok    <- seq_len(n_cross)
  within_A_tok <- n_cross + seq_len(n_within)
  within_B_tok <- n_cross + n_within + seq_len(n_within)
  idio_pool    <- (n_cross + 2L * n_within + 1L):VOCAB_SIZE

  make_verb_tokens <- function(class_shared)
    c(cross_tok, class_shared, sample(idio_pool, n_idio, replace = FALSE))

  A_tokens <- replicate(N_A, make_verb_tokens(within_A_tok), simplify = FALSE)
  B_tokens <- replicate(N_B, make_verb_tokens(within_B_tok), simplify = FALSE)

  log_mean <- log(MU) - 0.5 * SIGMA^2

  build_dists <- function(token_list) {
    P <- matrix(BG_WEIGHT, nrow = length(token_list), ncol = VOCAB_SIZE)
    for (i in seq_along(token_list))
      P[i, token_list[[i]]] <- rlnorm(N_PREFERRED, log_mean, SIGMA)
    P / rowSums(P)
  }

  PA_true <- build_dists(A_tokens)
  PB_true <- build_dists(B_tokens)

  presample <- function(P_rows)
    lapply(seq_len(nrow(P_rows)), function(i)
      sample(seq_len(VOCAB_SIZE), size = MAX_NOBS, replace = TRUE,
             prob = P_rows[i, ]))

  A_draws <- presample(PA_true)
  B_draws <- presample(PB_true)

  rows <- vector("list", length(N_OBS_GRID))

  for (idx in seq_along(N_OBS_GRID)) {
    n_obs <- N_OBS_GRID[idx]

    smooth_verb <- function(draws) {
      counts <- tabulate(draws[seq_len(n_obs)], nbins = VOCAB_SIZE)
      (counts + add_k) / (n_obs + add_k * VOCAB_SIZE)
    }

    P_hat <- rbind(
      do.call(rbind, lapply(A_draws, smooth_verb)),
      do.call(rbind, lapply(B_draws, smooth_verb))
    )

    jsd_mat <- pairwise_jsd(P_hat)

    # ob criterion: fraction of verbs passing Mann-Whitney
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

    frac_sig <- n_sig / (N_A + N_B)

    # ow criterion: mean within-class DJS
    jA          <- jsd_mat[seq_len(N_A), seq_len(N_A)]
    jB          <- jsd_mat[(N_A + 1L):(N_A + N_B), (N_A + 1L):(N_A + N_B)]
    within_mean <- (mean(jA[upper.tri(jA)]) + mean(jB[upper.tri(jB)])) / 2.0

    rows[[idx]] <- data.frame(n_obs = n_obs, frac_sig = frac_sig,
                               within_mean = within_mean)
  }

  bind_rows(rows)
}

# =============================================================================
# Run all k values and seeds (cached)
# =============================================================================

CACHE_FILE <- "../data/k_trajectories_data.csv"

if (file.exists(CACHE_FILE)) {
  cat("Loading cached simulation data from", CACHE_FILE, "\n")
  all_results <- read_csv(CACHE_FILE, show_col_types = FALSE)
} else {
  cat(sprintf("Running %d k values x %d seeds ...\n", length(K_VALUES), N_SEEDS))
  all_results <- map_dfr(seq_along(K_VALUES), function(ki) {
    k <- K_VALUES[ki]
    cat(sprintf("  k = %.3f\n", k))
    map_dfr(seq_len(N_SEEDS), function(s) {
      run_trajectory(k, s) |>
        mutate(add_k = k, seed = s)
    })
  })
  write_csv(all_results, CACHE_FILE)
  cat("Saved simulation data to", CACHE_FILE, "\n")
}

# =============================================================================
# Summarise: mean and 95% CI over seeds
# =============================================================================

summary_df <- all_results |>
  group_by(add_k, n_obs) |>
  summarise(
    ob_mean = mean(frac_sig),
    ob_lo   = quantile(frac_sig, 0.025),
    ob_hi   = quantile(frac_sig, 0.975),
    ow_mean = mean(within_mean),
    ow_lo   = quantile(within_mean, 0.025),
    ow_hi   = quantile(within_mean, 0.975),
    .groups = "drop"
  ) |>
  mutate(
    k_label = factor(
      add_k,
      levels = K_VALUES,
      labels = K_LABELS
    )
  )

# =============================================================================
# Plot
# =============================================================================

# We use a dual-axis approach: ob on left (proportion), ow on right (DJS).
# Since ggplot2 doesn't support true dual axes elegantly, we rescale ow to
# the [0,1] range of ob and use a secondary axis transformation.
ow_max <- max(summary_df$ow_hi) * 1.05

p <- summary_df |>
  ggplot(aes(x = n_obs)) +

  # ob criterion (fraction of verbs passing, left axis)
  geom_ribbon(aes(ymin = ob_lo, ymax = ob_hi), fill = "#2171b5", alpha = 0.15) +
  geom_line(aes(y = ob_mean, colour = "ob criterion"),
            linewidth = 0.8) +

  # ow criterion (rescaled to [0,1] for display, right axis)
  geom_ribbon(aes(ymin = ow_lo / ow_max, ymax = ow_hi / ow_max),
              fill = "#e6550d", alpha = 0.15) +
  geom_line(aes(y = ow_mean / ow_max,
                colour = "ow criterion"),
            linewidth = 0.8, linetype = "dashed") +

  # Threshold lines
  geom_hline(yintercept = CLASS_FRAC, linetype = "dotted", linewidth = 0.5,
             colour = "#2171b5") +
  geom_hline(yintercept = DJS_THRESH / ow_max, linetype = "dotted",
             linewidth = 0.5, colour = "#e6550d") +

  scale_x_log10(
    breaks = c(1, 10, 100, 1000, 5000),
    labels = c("1", "10", "100", "1k", "5k")
  ) +
  scale_y_continuous(
    name   = "Fraction of verbs",
    limits = c(0, 1),
    breaks = c(0, 0.5, 1),
    labels = c("0", "0.50", "1"),
    sec.axis = sec_axis(~ . * ow_max,
                        name = "Mean within-class DJS")
  ) +
  scale_colour_manual(
    values = c("ob criterion" = "#2171b5",
               "ow criterion" = "#e6550d"),
    name = NULL
  ) +
  facet_wrap(~ k_label, ncol = 1, strip.position = "right") +
  labs(
    x = "Corpus size"
  ) +
  theme_bw(base_size = 16) +
  theme(
    strip.text.y     = element_text(angle = -90, hjust = 0.5, size = 16),
    legend.position  = "right",
    legend.text      = element_text(size = 18),
    axis.title.y     = element_text(size = 18),
    axis.title.y.right = element_text(size = 18),
    axis.ticks.length = unit(0.2, "cm"),
    panel.grid.minor = element_blank()
  )

ggsave("../data/k_trajectories.pdf", p, width = 7, height = 9)
ggsave("../data/k_trajectories.png", p, width = 7, height = 9, dpi = 150)
cat("Saved: ../data/k_trajectories.pdf / ../data/k_trajectories.png\n")
