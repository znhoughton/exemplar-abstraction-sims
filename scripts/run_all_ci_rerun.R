# Driver: rerun all three grid simulations (now emitting per-seed raw data)
# and then compute CIs. Run from the scripts/ directory.
cat(sprintf("=== [%s] Starting Model 1 (Zero-Sensitivity Learner) ===\n", Sys.time()))
source("zero-sensitivity-learner.R")
rm(list = setdiff(ls(), "run_all_marker")); gc()

cat(sprintf("=== [%s] Starting Model 2 (Variable-Sensitivity Learner) ===\n", Sys.time()))
source("variable-sensitivity-learner.R")
rm(list = setdiff(ls(), "run_all_marker")); gc()

cat(sprintf("=== [%s] Starting Model 3 (Zipfian VSL) ===\n", Sys.time()))
source("zipfian-vsl.R")
rm(list = setdiff(ls(), "run_all_marker")); gc()

cat(sprintf("=== [%s] Computing CIs ===\n", Sys.time()))
source("compute_cis.R")

cat(sprintf("=== [%s] ALL DONE ===\n", Sys.time()))
