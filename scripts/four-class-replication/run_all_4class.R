# Driver: run all three 4-class grid simulations, then compute CIs.
# Run from the four-class-replication/ directory.
cat(sprintf("=== [%s] Starting Model 1 (Zero-Sensitivity Learner, 4-class) ===\n", Sys.time()))
source("zero-sensitivity-learner-4class.R")
rm(list = ls()); gc()

cat(sprintf("=== [%s] Starting Model 2 (Variable-Sensitivity Learner, 4-class) ===\n", Sys.time()))
source("variable-sensitivity-learner-4class.R")
rm(list = ls()); gc()

cat(sprintf("=== [%s] Starting Model 3 (Zipfian VSL, 4-class) ===\n", Sys.time()))
source("zipfian-vsl-4class.R")
rm(list = ls()); gc()

cat(sprintf("=== [%s] Computing CIs ===\n", Sys.time()))
source("compute_cis_4class.R")

cat(sprintf("=== [%s] ALL DONE ===\n", Sys.time()))
