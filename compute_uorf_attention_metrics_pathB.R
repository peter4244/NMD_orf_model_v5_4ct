#!/usr/bin/env Rscript
# ==============================================================================
# compute_uorf_attention_metrics_pathB.R
#
# TD2-INDEPENDENT uORF attention metric. Path B: identify candidate uORF slots
# by structural features alone — no reference to TD2 or the "main" CDS slot.
#
# Rationale: the original metric (compute_uorf_attention_metrics.R) defines
# is_uorf by relative position to "main", where "main" priority is
# ref_cds → TD2 CDS → longest. For novel transcripts without a recognizable
# ref CDS, main falls back to TD2 — which is biased against PTC-containing
# ORFs. This propagates the TD2 bias into the uORF metric: real-CDS variants
# that TD2 missed get labeled as "uORFs" simply because they start upstream
# of TD2's pick.
#
# Path B sidesteps the main-CDS problem entirely. A slot is a candidate uORF
# if it has the STRUCTURAL features of a classical mammalian uORF, with
# thresholds calibrated against the GENCODE-annotated CDS length distribution
# (hard floor at 270 nt — zero GENCODE CDSes shorter; see Methods):
#
#   - short ORF (orf_length < 200 nt; well below the GENCODE floor of 270)
#   - starts in the 5' half of the transcript (frac_position < 0.5)
#   - terminates 5'-leaning (frac_stop < 0.6, so the ORF doesn't span the
#     transcript midpoint)
#   - credible Kozak strength (kozak_score >= 1; excludes weak ATGs unlikely
#     to be translated)
#
# Path B-strict: all four criteria simultaneously. No reference to
# main_orf_rank, no reference to TD2. The length cutoff (200 nt) is
# defensible because 100% of GENCODE-annotated CDSes are ≥270 nt
# (n=37,293 ENSTs), so this can't false-positive on a real annotated CDS.
#
# Outputs (drop-in schema next to v1, distinguished by suffix):
#   uorf_attention_metrics_pathB.tsv          — per-isoform Path B metrics
#   uorf_attention_subgroup_counts_pathB.tsv  — subgroup totals
#
# Both v1 and Path B values are kept in the output for sanity comparison.
# Subgroup classifier is unchanged from v1 (relies on ref_cds_features::category
# from Isopair ref-AUG projection, which is methodologically sound).
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr)
})

# Repo root, from this script's own path -- so nothing depends on which machine it runs on.
REPO <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else normalizePath(getwd())
})

# Derived from this script's own location. Was a hardcoded Explorer path with a hardcoded
# Mac fallback -- two machines' absolute paths in tracked code, and wrong on any third.
PROJ <- REPO
# THE RESULTS TREE IS OVERRIDABLE. It was file.path(PROJ, "results_4ct") -- the PUBLISHED run --
# hardcoded with no override, so this script read the old universe whatever was being built.
# Same defect and same fix as audit_report.R; the 2026-08-13 audit listed neither this file
# nor compute_uorf_attention_metrics_pathB.R, both found by sweeping after Category A closed.
RES  <- Sys.getenv("NMD_RESULTS_DIR", unset = "")
RES  <- if (nzchar(RES)) RES else file.path(PROJ, "results_deposit_h5_2026-08-04")

# ── Path B-strict structural thresholds ─────────────────────────────────
MAX_UORF_LEN  <- 200L     # nt; below the GENCODE annotated CDS floor (270 nt)
MAX_FRAC_POS  <- 0.5      # ORF starts within first half of transcript
MAX_FRAC_STOP <- 0.6      # ORF terminates within first 60% of transcript
MIN_KOZAK     <- 1L       # excludes weak ATGs unlikely to be translated

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

# ── Main_orf — kept ONLY for cross-comparison with v1 (NOT used to define
# candidate uORFs under Path B) ──
main_orf <- sel |>
  group_by(isoform_id) |>
  arrange(desc(is_ref_cds), desc(is_sqanti_cds), desc(orf_length),
          .by_group = TRUE) |>
  slice(1) |>
  ungroup() |>
  select(isoform_id,
         main_orf_rank      = orf_rank,
         main_frac_position = frac_position)

# ── tx_length (needed for frac_stop) ──
tx_lengths <- read_tsv(file.path(RES, "tx_summary.tsv"),
                       show_col_types = FALSE) |>
  select(isoform_id, tx_length)

# ── Pivot attention to long form ──
attn_long <- predictions |>
  select(isoform_id, split, chr, label, prob,
         attn_0, attn_1, attn_2, attn_3, attn_4) |>
  pivot_longer(starts_with("attn_"), names_to = "orf_rank",
               values_to = "attention",
               names_transform = list(orf_rank = ~as.integer(sub("attn_", "", .x)))) |>
  filter(!is.na(attention))

# ── Join, apply Path B-strict + v1 flags ──
joined <- attn_long |>
  left_join(sel, by = c("isoform_id", "orf_rank")) |>
  left_join(main_orf, by = "isoform_id") |>
  left_join(tx_lengths, by = "isoform_id") |>
  mutate(
    frac_stop = (orf_start + orf_length) / tx_length,
    # Path B-strict — TD2-independent, structural, four criteria
    is_candidate_uorf = !is.na(orf_length) & !is.na(frac_position) &
                        !is.na(frac_stop) & !is.na(kozak_score) &
                        (orf_length < MAX_UORF_LEN) &
                        (frac_position < MAX_FRAC_POS) &
                        (frac_stop < MAX_FRAC_STOP) &
                        (kozak_score >= MIN_KOZAK),
    # v1 — original position-only relative to main (kept for comparison)
    is_uorf_v1        = !is.na(frac_position) & !is.na(main_frac_position) &
                        (frac_position < main_frac_position) &
                        (orf_rank != main_orf_rank)
  )

# ── Per-isoform metrics ──
per_iso <- joined |>
  group_by(isoform_id, split, chr, label, prob,
           main_orf_rank, main_frac_position) |>
  summarize(
    # Path B (this script's headline output)
    uorf_attention_frac        = sum(attention[is_candidate_uorf], na.rm = TRUE),
    n_candidate_uorf_slots     = sum(is_candidate_uorf, na.rm = TRUE),
    top_orf_rank               = orf_rank[which.max(attention)],
    top_attention              = max(attention),
    top_is_candidate_uorf      = is_candidate_uorf[which.max(attention)],
    # v1 (for direct comparison)
    uorf_attention_frac_v1     = sum(attention[is_uorf_v1], na.rm = TRUE),
    n_uorf_slots_v1            = sum(is_uorf_v1, na.rm = TRUE),
    top_is_uorf_v1             = is_uorf_v1[which.max(attention)],
    .groups = "drop"
  )

cat("Per-isoform table: ", nrow(per_iso), " rows\n")

# ── Apply subgroup classifier (UNCHANGED from v1 — sound by construction) ──
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
write_tsv(subgroup, file.path(RES, "uorf_attention_metrics_pathB.tsv"))

counts <- subgroup |>
  count(subgroup) |>
  mutate(pct = round(100 * n / sum(n), 1))
write_tsv(counts, file.path(RES, "uorf_attention_subgroup_counts_pathB.tsv"))

cat("\nWrote:\n  ", file.path(RES, "uorf_attention_metrics_pathB.tsv"), "\n",
    "  ", file.path(RES, "uorf_attention_subgroup_counts_pathB.tsv"), "\n", sep = "")

# ── Sanity contrast by subgroup ──
cat("\n=== Path B vs v1 — mean uorf_attention_frac by subgroup ===\n")
print(
  subgroup |>
    group_by(subgroup) |>
    summarize(n = n(),
              mean_v1     = round(mean(uorf_attention_frac_v1), 3),
              mean_pathB  = round(mean(uorf_attention_frac), 3),
              drop_pct    = round(100 * (1 - mean(uorf_attention_frac) /
                                         pmax(mean(uorf_attention_frac_v1), 1e-9)), 1),
              .groups = "drop")
)

cat("\n=== top_is_candidate_uorf rate by subgroup ===\n")
print(
  subgroup |>
    group_by(subgroup) |>
    summarize(n = n(),
              top_pathB_pct = round(100 * mean(top_is_candidate_uorf), 1),
              top_v1_pct    = round(100 * mean(top_is_uorf_v1), 1),
              .groups = "drop")
)
