# =============================================================================
# Summarize grid search results for all three models.
# Generates console output and markdown-formatted tables.
# =============================================================================

# Helper: add derived columns (frac_ow_lt_ob, frac_neither, gap)
add_derived <- function(d, has_ow_lt_ob = FALSE) {
  if (!has_ow_lt_ob) {
    # Model 1 CSV only has frac_detected + frac_ob_lt_ow; ties do not exist
    d$frac_ow_lt_ob <- d$frac_detected - d$frac_ob_lt_ow
    d$frac_tie      <- 0
  } else {
    # Models 2/3 have frac_ow_lt_ob explicit; ties = detected minus both strict orderings
    d$frac_tie <- d$frac_detected - d$frac_ob_lt_ow - d$frac_ow_lt_ob
    d$frac_tie <- pmax(0, round(d$frac_tie, 10))  # numerical floor
  }
  d$frac_neither <- 1 - d$frac_detected
  d$gap          <- round(d$item_overlap - d$class_overlap, 10)
  d
}

# Helper: format cell as "ob<ow / ow<ob / neither" (or with tie column)
fmt3 <- function(ob, ow, n) sprintf("%.2f / %.2f / %.2f", ob, ow, n)
fmt4 <- function(ob, ow, t, n) sprintf("%.2f / %.2f / %.2f / %.2f", ob, ow, t, n)

# Helper: print a markdown table with rows=gap, cols=mu
# sub: data frame with columns gap, mu, frac_ob_lt_ow, frac_ow_lt_ob, (frac_tie,) frac_neither
print_gap_mu_table <- function(sub, mu_vals, gap_vals, with_tie = FALSE) {
  hdr <- paste0("| gap | ", paste(sprintf("μ = %d", mu_vals), collapse = " | "), " |")
  sep <- paste0("|-----|", paste(rep(if (with_tie) "------------------------|" else "---------------------|",
                                    length(mu_vals)), collapse = ""), "")
  cat(hdr, "\n", sep, "\n", sep = "")
  for (g in gap_vals) {
    cells <- sapply(mu_vals, function(m) {
      r <- sub[round(sub$gap, 10) == round(g, 10) & sub$mu == m, ]
      if (nrow(r) == 0) return("—")
      if (with_tie)
        fmt4(r$frac_ob_lt_ow, r$frac_ow_lt_ob, r$frac_tie, r$frac_neither)
      else
        fmt3(r$frac_ob_lt_ow, r$frac_ow_lt_ob, r$frac_neither)
    })
    cat(sprintf("| %.1f | %s |\n", g, paste(cells, collapse = " | ")))
  }
  cat("\n")
}

# Helper: print a summary markdown table with rows=sigma, cols=gap
# Averages over mu (and over overlap combos yielding the same gap)
print_sigma_gap_table <- function(d, sigma_vals, gap_vals, with_tie = FALSE) {
  agg <- aggregate(cbind(frac_ob_lt_ow, frac_ow_lt_ob, frac_tie, frac_neither) ~ sigma + gap,
                   data = d, FUN = mean)
  hdr <- paste0("| σ | ", paste(sprintf("gap = %.1f", gap_vals), collapse = " | "), " |")
  sep <- paste0("|---|", paste(rep(if (with_tie) "------------------------|" else "---------------------|",
                                   length(gap_vals)), collapse = ""), "")
  cat(hdr, "\n", sep, "\n", sep = "")
  for (s in sigma_vals) {
    cells <- sapply(gap_vals, function(g) {
      r <- agg[agg$sigma == s & round(agg$gap, 10) == round(g, 10), ]
      if (nrow(r) == 0) return("—")
      if (with_tie)
        fmt4(r$frac_ob_lt_ow, r$frac_ow_lt_ob, r$frac_tie, r$frac_neither)
      else
        fmt3(r$frac_ob_lt_ow, r$frac_ow_lt_ob, r$frac_neither)
    })
    cat(sprintf("| **%.1f** | %s |\n", s, paste(cells, collapse = " | ")))
  }
  cat("\n")
}

# =============================================================================
# MODEL 1
# =============================================================================

d1 <- add_derived(read.csv("../data/grid_results_model1.csv"), has_ow_lt_ob = FALSE)

sigma_vals <- sort(unique(d1$sigma))
gap_vals   <- sort(unique(d1$gap))
mu_vals    <- sort(unique(d1$mu))

cat("=== MODEL 1: Zero-Sensitivity Limit ===\n\n")
cat("Summary by σ × gap (averaged over μ and overlap combinations):\n\n")
print_sigma_gap_table(d1, sigma_vals, gap_vals)
cat("Format: ob < ow / ow < ob / neither\n\n")

# Detailed: aggregate over overlap combos only (keep mu separate)
agg1 <- aggregate(cbind(frac_ob_lt_ow, frac_ow_lt_ob, frac_tie, frac_neither) ~ sigma + gap + mu,
                  data = d1, FUN = mean)

cat("Detailed by σ, gap, μ (averaged over overlap combinations):\n")
cat("(σ = 0.5 omitted: ob < ow = 1.00 for all combinations)\n\n")
for (s in c(1.0, 1.5)) {
  cat(sprintf("σ = %.1f\n\n", s))
  print_gap_mu_table(agg1[agg1$sigma == s, ], mu_vals, gap_vals)
}
cat("Format: ob < ow / ow < ob / neither\n\n")

cat("Overall means:\n")
cat(sprintf("  ob < ow: %.3f\n  ow < ob: %.3f\n  neither: %.3f\n\n",
            mean(d1$frac_ob_lt_ow), mean(d1$frac_ow_lt_ob), mean(d1$frac_neither)))

# =============================================================================
# MODEL 2
# =============================================================================

cat("=== MODEL 2: Add-k Smoothing ===\n\n")

if (file.exists("../data/grid_results_model2.csv")) {
  d2 <- add_derived(read.csv("../data/grid_results_model2.csv"), has_ow_lt_ob = TRUE)

  k_vals     <- sort(unique(d2$add_k))
  gap_vals2  <- sort(unique(d2$gap))
  sigma_vals2 <- sort(unique(d2$sigma))
  mu_vals2   <- sort(unique(d2$mu))

  cat("Summary by k × σ × gap (averaged over μ and overlap combinations):\n\n")
  for (k in k_vals) {
    label <- if (k <= 0.01) "ow < ob strongly expected" else
             if (k >= 0.5)  "ob < ow expected (→ Model 1)" else "transitional"
    cat(sprintf("k = %s (%s)\n\n", k, label))
    print_sigma_gap_table(d2[d2$add_k == k, ], sigma_vals2, gap_vals2)
  }
  cat("Format: ob < ow / ow < ob / neither\n\n")

  cat("Detailed by k, σ, gap, μ:\n\n")
  agg2 <- aggregate(cbind(frac_ob_lt_ow, frac_ow_lt_ob, frac_tie, frac_neither) ~ add_k + sigma + gap + mu,
                    data = d2, FUN = mean)
  for (k in k_vals) {
    cat(sprintf("k = %s\n\n", k))
    for (s in sigma_vals2) {
      cat(sprintf("σ = %.1f\n\n", s))
      print_gap_mu_table(agg2[agg2$add_k == k & agg2$sigma == s, ], mu_vals2, gap_vals2)
    }
  }
  cat("Format: ob < ow / ow < ob / neither\n\n")

  cat("Overall means by k:\n")
  for (k in k_vals) {
    sub <- d2[d2$add_k == k, ]
    cat(sprintf("  k = %s: ob<ow=%.3f  ow<ob=%.3f  neither=%.3f\n",
                k, mean(sub$frac_ob_lt_ow), mean(sub$frac_ow_lt_ob), mean(sub$frac_neither)))
  }
  cat("\n")
} else {
  cat("(../data/grid_results_model2.csv not yet available)\n\n")
}
