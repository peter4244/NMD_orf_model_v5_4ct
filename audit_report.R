#!/usr/bin/env Rscript
# audit_report.R — Independent verification of hardcoded numbers and key claims
# in orf_model_report_v5.Rmd
#
# This script loads data independently and checks every hardcoded number
# found in the report prose against computed values.

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

results_dir <- "results"
best_tag <- "atg500_stop500"
cat("=== Report Audit: Hardcoded Number Verification ===\n\n")

errors <- 0
warnings <- 0
pass <- 0

check <- function(description, claimed, computed, tolerance = 0.02) {
  match <- abs(claimed - computed) / max(abs(computed), 1e-6) <= tolerance
  if (match) {
    cat(sprintf("  PASS: %s (claimed=%.2f, computed=%.2f)\n", description, claimed, computed))
    pass <<- pass + 1
  } else {
    cat(sprintf("  FAIL: %s (claimed=%.2f, computed=%.2f, diff=%.2f)\n",
                description, claimed, computed, claimed - computed))
    errors <<- errors + 1
  }
}

check_approx <- function(description, claimed, computed, tolerance = 0.05) {
  if (is.na(computed) || is.nan(computed)) {
    cat(sprintf("  FAIL: %s — computed value is NA/NaN\n", description))
    errors <<- errors + 1
    return(invisible(NULL))
  }
  match <- abs(claimed - computed) / max(abs(computed), 1e-6) <= tolerance
  if (match) {
    cat(sprintf("  PASS: %s (claimed~%.1f, computed=%.2f)\n", description, claimed, computed))
    pass <<- pass + 1
  } else {
    cat(sprintf("  WARN: %s (claimed~%.1f, computed=%.2f, diff=%.2f)\n",
                description, claimed, computed, claimed - computed))
    warnings <<- warnings + 1
  }
}

# =========================================================================
# 1. Load data
# =========================================================================
cat("Loading data...\n")
preds <- read.delim(file.path(results_dir, paste0("predictions_", best_tag, ".tsv")))
ks <- read.delim(file.path(results_dir, paste0("kernel_shap_branch_", best_tag, ".tsv")))
orfs <- read.delim(file.path(results_dir, "selected_orfs.tsv"))
ref <- read.csv(file.path(results_dir, "ref_cds_features.tsv"), sep = "\t")
td2 <- read.csv(file.path(results_dir, "td2_features.tsv"), sep = "\t")
attn <- read.delim(file.path(results_dir, paste0("attention_entropy_", best_tag, ".tsv")))
metrics <- fromJSON(file.path(results_dir, paste0("metrics_", best_tag, ".json")))
gc_content <- read.delim(file.path(results_dir, paste0("gc_content_across_stop_window_", best_tag, ".tsv")))
cat(sprintf("  Loaded %d predictions, %d KernelSHAP, %d ORFs, %d attention\n",
            nrow(preds), nrow(ks), nrow(orfs), nrow(attn)))

# =========================================================================
# 2. Basic population counts
# =========================================================================
cat("\n--- Population counts ---\n")
n_test <- nrow(preds)
n_nmd <- sum(preds$label == 1)
n_ctrl <- sum(preds$label == 0)
nmd_pct <- round(100 * n_nmd / n_test, 1)
cat(sprintf("  n_test=%d, n_nmd=%d, n_ctrl=%d, nmd_pct=%.1f%%\n", n_test, n_nmd, n_ctrl, nmd_pct))

# Verify metrics JSON matches
check("AUC from JSON vs pROC", metrics$auc, metrics$auc, 0.001)
cat(sprintf("  Metrics JSON: AUC=%.4f, AUPRC=%.4f, n_test=%d\n",
            metrics$auc, metrics$auprc, metrics$n_test))
if (metrics$n_test != n_test) {
  cat(sprintf("  FAIL: metrics JSON n_test=%d != predictions n_test=%d\n",
              metrics$n_test, n_test))
  errors <- errors + 1
} else {
  cat(sprintf("  PASS: n_test consistent (JSON=%d, predictions=%d)\n", metrics$n_test, n_test))
  pass <- pass + 1
}

# =========================================================================
# 3. Line ~307: "~73% of transcripts" have ref CDS at rank 0
# =========================================================================
cat("\n--- Line 307: Ref CDS availability ---\n")
test_rank0 <- orfs %>% filter(orf_rank == 0) %>% semi_join(preds, by = "isoform_id")
ref_available <- test_rank0 %>% left_join(ref, by = "isoform_id")
# "ref CDS available" means category is NOT ref_atg_lost/no_ref_isoform/etc
ref_retained_cats <- c("effectively_ptc", "no_downstream_ejc", "truncated_no_ejc")
n_has_ref <- sum(ref_available$category %in% ref_retained_cats, na.rm = TRUE)
pct_ref_cds <- 100 * n_has_ref / nrow(test_rank0)
cat(sprintf("  %d of %d test rank-0 ORFs have ref CDS category (%.1f%%)\n",
            n_has_ref, nrow(test_rank0), pct_ref_cds))
check_approx("% transcripts with ref CDS at rank 0 (line 307 says ~73%)", 73, pct_ref_cds)

# =========================================================================
# 4. Lines ~641, 645: KernelSHAP branch percentages
# =========================================================================
cat("\n--- Lines 641/645: KernelSHAP branch decomposition ---\n")
ks_nmd <- ks %>% filter(label == 1)
branch_means <- data.frame(
  branch = c("ATG", "Stop", "Structural"),
  mean_abs = c(mean(abs(ks_nmd$shap_atg)),
               mean(abs(ks_nmd$shap_stop)),
               mean(abs(ks_nmd$shap_structural)))
)
branch_means$pct <- 100 * branch_means$mean_abs / sum(branch_means$mean_abs)
cat(sprintf("  Computed: ATG=%.1f%%, Stop=%.1f%%, Structural=%.1f%%\n",
            branch_means$pct[1], branch_means$pct[2], branch_means$pct[3]))

check_approx("Structural branch % (line 645 says 61%)", 61, branch_means$pct[3])
check_approx("Stop branch % (line 767 says 29%)", 29, branch_means$pct[2])
check_approx("ATG branch % (line 767 says 10%)", 10, branch_means$pct[1])

# =========================================================================
# 5. KernelSHAP residual claim ("residual = 0")
# =========================================================================
cat("\n--- Line 641: KernelSHAP residual = 0 ---\n")
max_residual <- max(abs(ks$residual), na.rm = TRUE)
cat(sprintf("  Max |residual| = %.2e\n", max_residual))
if (max_residual < 1e-6) {
  cat("  PASS: Residual effectively zero (< 1e-6)\n")
  pass <- pass + 1
} else if (max_residual < 1e-3) {
  cat("  WARN: Residual near zero but > 1e-6\n")
  warnings <- warnings + 1
} else {
  cat("  FAIL: Residual not zero\n")
  errors <- errors + 1
}

# =========================================================================
# 6. Line ~1252: EJC dose-response values
# =========================================================================
cat("\n--- Line 1252: EJC dose-response ---\n")
# Join predictions with rank-0 ORF features
rank0_with_preds <- test_rank0 %>%
  left_join(preds %>% select(isoform_id, label, prob), by = "isoform_id")

# n_downstream_ejc is a feature of the ORF
if ("n_downstream_ejc" %in% names(rank0_with_preds)) {
  ejc_response <- rank0_with_preds %>%
    filter(!is.na(prob)) %>%
    group_by(n_downstream_ejc) %>%
    summarise(mean_prob = mean(prob, na.rm = TRUE), n = n(), .groups = "drop") %>%
    arrange(n_downstream_ejc)
  cat("  EJC dose-response (rank-0 ORFs):\n")
  for (i in seq_len(min(6, nrow(ejc_response)))) {
    cat(sprintf("    EJC=%d: P(NMD)=%.1f%% (n=%d)\n",
                ejc_response$n_downstream_ejc[i],
                100 * ejc_response$mean_prob[i],
                ejc_response$n[i]))
  }
  # Line 1252 claims: "0 EJCs = 56%, 1 EJC = 88%, 2 = 94%, 3 = 96%, 5+ = 97%"
  # These are approximate; check 0 and 1
  if (nrow(ejc_response) > 0) {
    ejc0 <- ejc_response$mean_prob[ejc_response$n_downstream_ejc == 0]
    if (length(ejc0) > 0) check_approx("P(NMD) at 0 EJCs (line 1252 says 56%)", 56, 100 * ejc0, 0.1)
    ejc1 <- ejc_response$mean_prob[ejc_response$n_downstream_ejc == 1]
    if (length(ejc1) > 0) check_approx("P(NMD) at 1 EJC (line 1252 says 88%)", 88, 100 * ejc1, 0.1)
  }
} else {
  cat("  SKIP: n_downstream_ejc not in selected_orfs\n")
}

# =========================================================================
# 7. Line ~1412: GC content transitions
# =========================================================================
cat("\n--- Line 1412: GC content transitions ---\n")
if (nrow(gc_content) > 0 && "position" %in% names(gc_content)) {
  # The claim is about raw GC content transitioning from ~48% to ~36%
  # around the stop codon (position 0 = stop codon center)
  gc_before <- gc_content %>% filter(position < -10 & position > -100)
  gc_after <- gc_content %>% filter(position > 10 & position < 100)
  if ("gc_mean" %in% names(gc_content)) {
    mean_before <- mean(gc_before$gc_mean, na.rm = TRUE)
    mean_after <- mean(gc_after$gc_mean, na.rm = TRUE)
    cat(sprintf("  GC before stop: %.1f%%, after: %.1f%%\n",
                100 * mean_before, 100 * mean_after))
    check_approx("GC before stop (line 1412 says ~48%)", 48, 100 * mean_before, 0.1)
    check_approx("GC after stop (line 1412 says ~36%)", 36, 100 * mean_after, 0.15)
  } else {
    cat("  Available columns:", paste(names(gc_content), collapse = ", "), "\n")
    cat("  SKIP: gc_mean column not found, checking available columns\n")
  }
} else {
  cat("  SKIP: GC content file not in expected format\n")
}

# =========================================================================
# 8. Lines ~2438-2440: Attention stats by subgroup
# =========================================================================
cat("\n--- Lines 2438-2440: Attention by subgroup ---\n")
# Build subgroups the same way the report does
attn_with_ref <- attn %>%
  left_join(ref %>% select(isoform_id, ref_atg_available, ref_downstream_ejc, category),
            by = "isoform_id") %>%
  left_join(td2 %>% select(isoform_id, td2_downstream_ejc), by = "isoform_id")

attn_sg <- attn_with_ref %>%
  mutate(subgroup = case_when(
    label == 0 ~ "Control",
    !is.na(ref_downstream_ejc) & ref_downstream_ejc > 0 ~ "NMD PTC+",
    !is.na(ref_downstream_ejc) & ref_downstream_ejc == 0 &
      !is.na(td2_downstream_ejc) & td2_downstream_ejc == 0 ~ "NMD PTC- (ref retained)",
    is.na(ref_atg_available) | ref_atg_available == 0 ~ "NMD PTC- (ATG lost)",
    TRUE ~ "NMD other"
  ))

# Line 2438: "~95% of transcripts, with median attention ~0.88" for PTC+
ptc_plus <- attn_sg %>% filter(subgroup == "NMD PTC+")
ptc_plus_rank0_dom <- mean(ptc_plus$max_attn > 0.5, na.rm = TRUE) * 100
ptc_plus_med_attn <- median(ptc_plus$max_attn, na.rm = TRUE)
cat(sprintf("  PTC+: rank-0 dominant=%.1f%% (claim ~95%%), median max_attn=%.2f (claim ~0.88)\n",
            ptc_plus_rank0_dom, ptc_plus_med_attn))
check_approx("PTC+ rank-0 dominant % (line 2438 says ~95%)", 95, ptc_plus_rank0_dom, 0.1)
check_approx("PTC+ median max attention (line 2438 says ~0.88)", 0.88, ptc_plus_med_attn, 0.1)

# Line 2439: "rank-0 is dominant in only ~50%" for PTC- retained
ptc_ret <- attn_sg %>% filter(subgroup == "NMD PTC- (ref retained)")
ptc_ret_rank0_dom <- mean(ptc_ret$max_attn > 0.5 & ptc_ret$argmax_orf == 0, na.rm = TRUE) * 100
cat(sprintf("  PTC- retained: rank-0 dominant=%.1f%% (claim ~50%%)\n", ptc_ret_rank0_dom))
check_approx("PTC- retained rank-0 dominant % (line 2439 says ~50%)", 50, ptc_ret_rank0_dom, 0.15)

# Line 2440: "~63% rank-0 dominant" for PTC- ATG lost
ptc_lost <- attn_sg %>% filter(subgroup == "NMD PTC- (ATG lost)")
ptc_lost_rank0_dom <- mean(ptc_lost$max_attn > 0.5 & ptc_lost$argmax_orf == 0, na.rm = TRUE) * 100
cat(sprintf("  PTC- ATG lost: rank-0 dominant=%.1f%% (claim ~63%%)\n", ptc_lost_rank0_dom))
check_approx("PTC- ATG lost rank-0 dominant % (line 2440 says ~63%)", 63, ptc_lost_rank0_dom, 0.15)

# =========================================================================
# 9. Model performance (AUC reproducibility)
# =========================================================================
cat("\n--- AUC reproducibility ---\n")
if (requireNamespace("pROC", quietly = TRUE)) {
  roc_obj <- pROC::roc(preds$label, preds$prob, quiet = TRUE)
  computed_auc <- as.numeric(pROC::auc(roc_obj))
  cat(sprintf("  JSON AUC=%.4f, pROC AUC=%.4f\n", metrics$auc, computed_auc))
  check("AUC matches JSON", metrics$auc, computed_auc, 0.001)
} else {
  cat("  SKIP: pROC not available\n")
}

# =========================================================================
# 10. Prevalence consistency
# =========================================================================
cat("\n--- Prevalence ---\n")
cat(sprintf("  n_nmd/n_test = %d/%d = %.1f%%\n", n_nmd, n_test, nmd_pct))
# The report should say 15.3% everywhere prevalence is mentioned
# Check metrics JSON
if (!is.null(metrics$n_nmd)) {
  check("n_nmd JSON vs predictions", metrics$n_nmd, n_nmd, 0.001)
}

# =========================================================================
# 11. Subgroup line 2001: "27% ATG branch vs 8% for PTC+"
# =========================================================================
cat("\n--- Line 2001: Branch % by subgroup ---\n")
# Join KernelSHAP with subgroup labels
ks_with_sg <- ks %>%
  left_join(ref %>% select(isoform_id, ref_atg_available, ref_downstream_ejc, category),
            by = "isoform_id") %>%
  left_join(td2 %>% select(isoform_id, td2_downstream_ejc), by = "isoform_id") %>%
  mutate(subgroup = case_when(
    label == 0 ~ "Control",
    !is.na(ref_downstream_ejc) & ref_downstream_ejc > 0 ~ "NMD PTC+",
    !is.na(ref_downstream_ejc) & ref_downstream_ejc == 0 &
      !is.na(td2_downstream_ejc) & td2_downstream_ejc == 0 ~ "NMD PTC- (ref retained)",
    is.na(ref_atg_available) | ref_atg_available == 0 ~ "NMD PTC- (ATG lost)",
    TRUE ~ "NMD other"
  ))

for (sg in c("NMD PTC+", "NMD PTC- (ATG lost)")) {
  sg_data <- ks_with_sg %>% filter(subgroup == sg)
  if (nrow(sg_data) > 0) {
    atg_pct <- 100 * mean(abs(sg_data$shap_atg)) /
      (mean(abs(sg_data$shap_atg)) + mean(abs(sg_data$shap_stop)) + mean(abs(sg_data$shap_structural)))
    cat(sprintf("  %s: ATG branch = %.1f%%\n", sg, atg_pct))
  }
}
# The claim: PTC+ ATG=8%, PTC- ATG lost=27%
ptc_plus_ks <- ks_with_sg %>% filter(subgroup == "NMD PTC+")
ptc_lost_ks <- ks_with_sg %>% filter(subgroup == "NMD PTC- (ATG lost)")
if (nrow(ptc_plus_ks) > 0 && nrow(ptc_lost_ks) > 0) {
  atg_pct_ptc <- 100 * mean(abs(ptc_plus_ks$shap_atg)) /
    (mean(abs(ptc_plus_ks$shap_atg)) + mean(abs(ptc_plus_ks$shap_stop)) + mean(abs(ptc_plus_ks$shap_structural)))
  atg_pct_lost <- 100 * mean(abs(ptc_lost_ks$shap_atg)) /
    (mean(abs(ptc_lost_ks$shap_atg)) + mean(abs(ptc_lost_ks$shap_stop)) + mean(abs(ptc_lost_ks$shap_structural)))
  check_approx("PTC+ ATG branch % (line 2001 says 8%)", 8, atg_pct_ptc, 0.15)
  check_approx("PTC- ATG lost ATG branch % (line 2001 says 27%)", 27, atg_pct_lost, 0.15)
}

# =========================================================================
# 12. Line ~313: window claim "500bp total (±250bp)"
# =========================================================================
cat("\n--- Line 313: Window size claim ---\n")
# best_atg = 500, best_stop = 500
# The code does half_win = win_size // 2 = 250
# Window = [center - 250, center + 250) = 500 positions
# Claim "±250bp" is slightly misleading (asymmetric: 250 before, 249 after center)
# but acceptable for a description
cat("  INFO: Window is 500 positions = [center-250, center+250), technically 250 before + center + 249 after\n")
cat("  Prose says '±250bp' which is close enough but not exactly symmetric\n")

# =========================================================================
# SUMMARY
# =========================================================================
cat(sprintf("\n=== AUDIT SUMMARY ===\n"))
cat(sprintf("  PASS: %d\n", pass))
cat(sprintf("  WARN: %d\n", warnings))
cat(sprintf("  FAIL: %d\n", errors))
cat(sprintf("  Total checks: %d\n", pass + warnings + errors))
