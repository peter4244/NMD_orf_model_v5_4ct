#!/usr/bin/env Rscript
# compute_uorf_attention_metrics.R
#
# Builds the per-isoform metrics table for the uORF-attention analysis.
# Joins v5_4ct inference output (attention weights + predictions) with the
# subgroup classifier inputs (ref_cds_features::category, td2_features::
# td2_downstream_ejc) and the per-ORF identity table (selected_orfs).
#
# Per-isoform outputs:
#   subgroup                  PTC+ / PTC- retained / PTC- lost / non-NMD
#   uorf_attention_frac       fraction of attention on ORFs flagged as uORFs
#   top_is_uorf               TRUE if argmax-attention ORF is a uORF
#   top_orf_rank              argmax-attention ORF rank (0..4)
#   main_orf_rank             which ORF was identified as the main CDS
#   prob                      model P(NMD)
#
# Source: results_4ct/{uorf_attention_predictions.tsv, selected_orfs.tsv,
#   ref_cds_features.tsv, td2_features.tsv, tx_summary.tsv}
# Output: results_4ct/uorf_attention_metrics.tsv
#         results_4ct/uorf_attention_subgroup_counts.tsv

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr)
})

PROJ <- "/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct"
RES  <- file.path(PROJ, "results_4ct")

# ── Load inference output ──
predictions <- read_tsv(file.path(RES, "uorf_attention_predictions.tsv"),
                        show_col_types = FALSE)
cat("Predictions: ", nrow(predictions), " isoforms\n")

# ── Load selected_orfs (per-ORF identity at model input) ──
sel <- read_tsv(file.path(RES, "selected_orfs.tsv"), show_col_types = FALSE) |>
  select(isoform_id, orf_rank, orf_start, orf_length,
         frac_position, is_ref_cds, is_sqanti_cds, kozak_score)
cat("selected_orfs: ", nrow(sel), " rows across ",
    n_distinct(sel$isoform_id), " isoforms\n")

# ── Identify the "main CDS" ORF per isoform (fallback chain) ──
# Priority: is_ref_cds==1 > is_sqanti_cds==1 > longest ORF.
# This handles ATF4-style "no_ref_isoform" cases where the model has neither
# ref nor SQANTI flag set; we fall back to the longest ORF as the candidate
# canonical CDS.
main_orf <- sel |>
  group_by(isoform_id) |>
  arrange(desc(is_ref_cds), desc(is_sqanti_cds), desc(orf_length),
          .by_group = TRUE) |>
  slice(1) |>
  ungroup() |>
  select(isoform_id,
         main_orf_rank      = orf_rank,
         main_frac_position = frac_position,
         main_orf_length    = orf_length,
         main_is_ref_cds    = is_ref_cds,
         main_is_sqanti_cds = is_sqanti_cds)

# ── Pivot attention to long form ──
attn_long <- predictions |>
  select(isoform_id, split, chr, label, prob,
         starts_with("attn_")) |>
  pivot_longer(starts_with("attn_"),
               names_to = "orf_rank_str", values_to = "attention") |>
  mutate(orf_rank = as.integer(str_remove(orf_rank_str, "attn_"))) |>
  select(-orf_rank_str)

# ── Join per-rank ORF features (only valid ranks have a row in `sel`) ──
joined <- attn_long |>
  left_join(sel |> select(isoform_id, orf_rank, frac_position, kozak_score),
            by = c("isoform_id", "orf_rank")) |>
  # Attach the per-isoform main-ORF info (keyed by isoform only so every
  # rank — including padding ranks with no `sel` row — inherits it)
  left_join(main_orf, by = "isoform_id") |>
  mutate(is_uorf = !is.na(frac_position) & !is.na(main_frac_position) &
                   (frac_position < main_frac_position) &
                   (orf_rank != main_orf_rank))

# ── Per-isoform metrics ──
per_iso <- joined |>
  group_by(isoform_id, split, chr, label, prob,
           main_orf_rank, main_frac_position) |>
  summarize(
    uorf_attention_frac = sum(attention[is_uorf], na.rm = TRUE),
    top_orf_rank        = orf_rank[which.max(attention)],
    top_attention       = max(attention),
    top_is_uorf         = is_uorf[which.max(attention)],
    n_uorf_slots        = sum(is_uorf, na.rm = TRUE),
    .groups = "drop"
  )

cat("Per-isoform table: ", nrow(per_iso), " rows\n")

# ── Apply subgroup classifier ──
# Logic from v5_4ct METHODS.md §"Subgroup definitions" (and
# 09b_export_subgroup_profiles.py::assign_subgroup).
# IMPORTANT: use the v5_4ct relabeled is_nmd from tx_summary (NOT the
# HDF5 `labels` field, which is v5's 6-cell-type labeling — they differ
# for the ~22 isoforms relabeled during the 4-CT mashr refit).
tx_labels <- read_tsv(file.path(RES, "tx_summary.tsv"),
                      show_col_types = FALSE) |>
  select(isoform_id, is_nmd_v5_4ct = is_nmd)
ref_cds <- read_tsv(file.path(RES, "ref_cds_features.tsv"),
                    show_col_types = FALSE) |>
  select(isoform_id, category)
td2     <- read_tsv(file.path(RES, "td2_features.tsv"),
                    show_col_types = FALSE) |>
  select(isoform_id, td2_downstream_ejc)

REF_LOST_CATS <- c("ref_atg_lost", "no_ref_isoform",
                   "not_atg_in_target", "no_stop_in_target")
PTC_NEG_RETAINED_CATS <- c("no_downstream_ejc", "truncated_no_ejc")

subgroup <- per_iso |>
  left_join(tx_labels, by = "isoform_id") |>
  left_join(ref_cds,   by = "isoform_id") |>
  left_join(td2,       by = "isoform_id") |>
  mutate(subgroup = case_when(
    is_nmd_v5_4ct == 0                                                ~ "non_NMD",
    category == "effectively_ptc"                                     ~ "NMD_PTC_pos",
    category %in% REF_LOST_CATS & !is.na(td2_downstream_ejc) &
      td2_downstream_ejc > 0                                          ~ "NMD_PTC_pos",
    category %in% PTC_NEG_RETAINED_CATS                               ~ "NMD_PTC_neg_retained",
    category %in% REF_LOST_CATS                                       ~ "NMD_PTC_neg_lost",
    TRUE                                                              ~ "other"
  ))

# ── Save ──
out_metrics  <- file.path(RES, "uorf_attention_metrics.tsv")
out_counts   <- file.path(RES, "uorf_attention_subgroup_counts.tsv")
write_tsv(subgroup, out_metrics)

counts <- subgroup |>
  count(subgroup, name = "n") |>
  mutate(pct = round(100 * n / sum(n), 1))
write_tsv(counts, out_counts)

cat("\n=== Subgroup counts ===\n")
print(counts)

cat("\n=== uORF attention fraction by subgroup (median, IQR) ===\n")
subgroup |>
  group_by(subgroup) |>
  summarize(
    n = n(),
    median_uorf_attn = round(median(uorf_attention_frac, na.rm = TRUE), 3),
    q25 = round(quantile(uorf_attention_frac, 0.25, na.rm = TRUE), 3),
    q75 = round(quantile(uorf_attention_frac, 0.75, na.rm = TRUE), 3),
    pct_top_is_uorf  = round(100 * mean(top_is_uorf, na.rm = TRUE), 1)
  ) |> print()

cat("\nSaved:\n  ", out_metrics, "\n  ", out_counts, "\n")
