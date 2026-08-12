# =============================================================================
# Confidence intervals for the 4-class replication grid results. Same method
# as ../scripts/compute_cis.R (Wilson intervals on frac_* proportions,
# t-intervals on mean_* values) — see that script for the rationale. Requires
# the *_4class_per_seed.csv files, written by the three *-4class.R scripts in
# this directory.
# =============================================================================

wilson_ci <- function(x, n, conf = 0.95) {
  z <- qnorm(1 - (1 - conf) / 2)
  p <- x / n
  denom  <- 1 + z^2 / n
  center <- (p + z^2 / (2 * n)) / denom
  margin <- (z / denom) * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))
  data.frame(lower = pmax(0, center - margin), upper = pmin(1, center + margin))
}

t_ci <- function(mean_val, sd_val, n, conf = 0.95) {
  ok <- !is.na(mean_val) & !is.na(sd_val) & !is.na(n) & n > 1
  lower <- upper <- rep(NA_real_, length(mean_val))
  margin <- qt(1 - (1 - conf) / 2, df = n[ok] - 1) * sd_val[ok] / sqrt(n[ok])
  lower[ok] <- mean_val[ok] - margin
  upper[ok] <- mean_val[ok] + margin
  data.frame(lower = lower, upper = upper)
}

pooled_proportion_ci <- function(per_seed, indicator, group_cols = NULL) {
  if (is.null(group_cols)) {
    x <- sum(indicator); n <- length(indicator)
    return(cbind(data.frame(n = n, x = x, proportion = x / n), wilson_ci(x, n)))
  }
  key <- do.call(paste, c(per_seed[group_cols], sep = "\r"))
  rows <- lapply(split(seq_along(indicator), key), function(idx) {
    x <- sum(indicator[idx]); n <- length(idx)
    cbind(per_seed[idx[1], group_cols, drop = FALSE],
          data.frame(n = n, x = x, proportion = x / n), wilson_ci(x, n))
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

DATA_DIR <- "../data/four-class-replication"

# =============================================================================
# MODEL 1 (Zero-Sensitivity Learner, 4-class)
# =============================================================================

ps1 <- read.csv(file.path(DATA_DIR, "grid_results_model1_4class_per_seed.csv"))
d1  <- read.csv(file.path(DATA_DIR, "grid_results_model1_4class.csv"))

d1 <- cbind(d1, setNames(wilson_ci(round(d1$frac_ob_lt_ow * 50), 50),
                          c("frac_ob_lt_ow_ci_lo", "frac_ob_lt_ow_ci_hi")))
d1 <- cbind(d1, setNames(t_ci(d1$mean_ob, d1$sd_ob, d1$n_both),
                          c("mean_ob_ci_lo", "mean_ob_ci_hi")))
d1 <- cbind(d1, setNames(t_ci(d1$mean_ow, d1$sd_ow, d1$n_both),
                          c("mean_ow_ci_lo", "mean_ow_ci_hi")))
d1 <- cbind(d1, setNames(t_ci(d1$mean_log_ratio, d1$sd_log_ratio, d1$n_ob_lt_ow),
                          c("mean_log_ratio_ci_lo", "mean_log_ratio_ci_hi")))
write.csv(d1, file.path(DATA_DIR, "grid_results_model1_4class_with_ci.csv"), row.names = FALSE)

both1     <- !is.na(ps1$ob_alpha) & !is.na(ps1$ow_alpha)
ob_lt_ow1 <- both1 & (ps1$ob_alpha < ps1$ow_alpha)
ow_lt_ob1 <- both1 & (ps1$ow_alpha < ps1$ob_alpha)
cat("=== Model 1 (4-class) overall (n =", nrow(ps1), "seed-runs) ===\n")
cat("ob < ow: "); print(pooled_proportion_ci(ps1, ob_lt_ow1))
cat("ow < ob: "); print(pooled_proportion_ci(ps1, ow_lt_ob1))
cat("neither: "); print(pooled_proportion_ci(ps1, !both1))

# =============================================================================
# MODEL 2 (Variable-Sensitivity Learner, 4-class)
# =============================================================================

ps2 <- read.csv(file.path(DATA_DIR, "grid_results_model2_4class_per_seed.csv"))
d2  <- read.csv(file.path(DATA_DIR, "grid_results_model2_4class.csv"))

d2 <- cbind(d2, setNames(wilson_ci(round(d2$frac_ob_lt_ow * 50), 50),
                          c("frac_ob_lt_ow_ci_lo", "frac_ob_lt_ow_ci_hi")))
d2 <- cbind(d2, setNames(wilson_ci(round(d2$frac_ow_lt_ob * 50), 50),
                          c("frac_ow_lt_ob_ci_lo", "frac_ow_lt_ob_ci_hi")))
d2 <- cbind(d2, setNames(t_ci(d2$mean_ob_nobs, d2$sd_ob_nobs, d2$n_both),
                          c("mean_ob_nobs_ci_lo", "mean_ob_nobs_ci_hi")))
d2 <- cbind(d2, setNames(t_ci(d2$mean_ow_nobs, d2$sd_ow_nobs, d2$n_both),
                          c("mean_ow_nobs_ci_lo", "mean_ow_nobs_ci_hi")))
d2 <- cbind(d2, setNames(t_ci(d2$mean_log_ratio, d2$sd_log_ratio, d2$n_ob_lt_ow),
                          c("mean_log_ratio_ci_lo", "mean_log_ratio_ci_hi")))
write.csv(d2, file.path(DATA_DIR, "grid_results_model2_4class_with_ci.csv"), row.names = FALSE)

both2     <- !is.na(ps2$ob_nobs) & !is.na(ps2$ow_nobs)
ob_lt_ow2 <- both2 & (ps2$ob_nobs < ps2$ow_nobs)
ow_lt_ob2 <- both2 & (ps2$ow_nobs < ps2$ob_nobs)
cat("\n=== Model 2 (4-class) by k (n = 108 combos x 50 seeds = 5,400 per row) ===\n")
cat("ob < ow by k:\n"); print(pooled_proportion_ci(ps2, ob_lt_ow2, "add_k"))
cat("ow < ob by k:\n"); print(pooled_proportion_ci(ps2, ow_lt_ob2, "add_k"))

# =============================================================================
# MODEL 3 (Zipfian VSL, 4-class)
# =============================================================================

ps3 <- read.csv(file.path(DATA_DIR, "grid_results_model3_4class_per_seed.csv"))
d3  <- read.csv(file.path(DATA_DIR, "grid_results_model3_4class.csv"))

d3 <- cbind(d3, setNames(wilson_ci(round(d3$frac_ob_lt_ow * 50), 50),
                          c("frac_ob_lt_ow_ci_lo", "frac_ob_lt_ow_ci_hi")))
d3 <- cbind(d3, setNames(wilson_ci(round(d3$frac_ow_lt_ob * 50), 50),
                          c("frac_ow_lt_ob_ci_lo", "frac_ow_lt_ob_ci_hi")))
d3 <- cbind(d3, setNames(t_ci(d3$mean_ob_nobs, d3$sd_ob_nobs, d3$n_both),
                          c("mean_ob_nobs_ci_lo", "mean_ob_nobs_ci_hi")))
d3 <- cbind(d3, setNames(t_ci(d3$mean_ow_nobs, d3$sd_ow_nobs, d3$n_both),
                          c("mean_ow_nobs_ci_lo", "mean_ow_nobs_ci_hi")))
d3 <- cbind(d3, setNames(t_ci(d3$mean_log_ratio, d3$sd_log_ratio, d3$n_ob_lt_ow),
                          c("mean_log_ratio_ci_lo", "mean_log_ratio_ci_hi")))
write.csv(d3, file.path(DATA_DIR, "grid_results_model3_4class_with_ci.csv"), row.names = FALSE)

both3     <- !is.na(ps3$ob_nobs) & !is.na(ps3$ow_nobs)
ob_lt_ow3 <- both3 & (ps3$ob_nobs < ps3$ow_nobs)
ow_lt_ob3 <- both3 & (ps3$ow_nobs < ps3$ob_nobs)
cat("\n=== Model 3 (4-class, Zipfian) by k (n = 324 combos x 50 seeds = 16,200 per row) ===\n")
cat("ob < ow by k:\n"); print(pooled_proportion_ci(ps3, ob_lt_ow3, "add_k"))
cat("ow < ob by k:\n"); print(pooled_proportion_ci(ps3, ow_lt_ob3, "add_k"))

cat("\nDone. Per-combination CIs written to", DATA_DIR, "/grid_results_model{1,2,3}_4class_with_ci.csv\n")
