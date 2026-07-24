#!/usr/bin/env Rscript
# audit_uorf_attention.R
#
# Step 1/Step 2 of the 5-step verification for the uORF-attention report.
# Independently recomputes every numerical claim in the Rmd from the
# upstream input TSVs (NOT the cached intermediate metrics file), and
# compares to the values rendered in the report.
#
# Each check prints PASS/WARN/FAIL with claimed vs computed values.

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr); library(effsize); library(tibble)
})

PROJ <- "/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct"
RES  <- file.path(PROJ, "results_4ct")

pass <- 0L; warn <- 0L; fail <- 0L
check <- function(name, claimed, computed, tol = 0.005) {
  ok <- if (is.numeric(claimed) && is.numeric(computed)) {
    abs(claimed - computed) <= tol * max(abs(claimed), 1e-9)
  } else identical(claimed, computed)
  if (isTRUE(ok)) {
    pass <<- pass + 1L
    cat(sprintf("PASS  %s  claimed=%s computed=%s\n",
                name, format(claimed), format(computed)))
  } else {
    fail <<- fail + 1L
    cat(sprintf("FAIL  %s  claimed=%s computed=%s\n",
                name, format(claimed), format(computed)))
  }
}

# ── Independent load from upstream sources ──
tx  <- read_tsv(file.path(RES, "tx_summary.tsv"),    show_col_types = FALSE)
ref <- read_tsv(file.path(RES, "ref_cds_features.tsv"), show_col_types = FALSE)
td2 <- read_tsv(file.path(RES, "td2_features.tsv"),  show_col_types = FALSE)
sel <- read_tsv(file.path(RES, "selected_orfs.tsv"), show_col_types = FALSE)
pred <- read_tsv(file.path(RES, "uorf_attention_predictions.tsv"),
                 show_col_types = FALSE)

cat("\n=== Step 1: Factual accuracy ===\n\n")

# Claim 1: n_total = 39,938
check("Universe n_total", 39938L, nrow(pred))
check("Universe n_NMD",   8840L,  sum(tx$is_nmd == 1))
check("Universe n_non_NMD", 31098L, sum(tx$is_nmd == 0))

# ── Recompute subgroup assignment INDEPENDENTLY ──
REF_LOST <- c("ref_atg_lost", "no_ref_isoform", "not_atg_in_target", "no_stop_in_target")
PTC_NEG_RETAINED <- c("no_downstream_ejc", "truncated_no_ejc")

sg <- pred |>
  select(isoform_id, prob) |>
  left_join(tx  |> select(isoform_id, is_nmd_v5_4ct = is_nmd), by = "isoform_id") |>
  left_join(ref |> select(isoform_id, category), by = "isoform_id") |>
  left_join(td2 |> select(isoform_id, td2_downstream_ejc), by = "isoform_id") |>
  mutate(subgroup = case_when(
    is_nmd_v5_4ct == 0                                                ~ "non_NMD",
    category == "effectively_ptc"                                     ~ "NMD_PTC_pos",
    category %in% REF_LOST & !is.na(td2_downstream_ejc) &
      td2_downstream_ejc > 0                                          ~ "NMD_PTC_pos",
    category %in% PTC_NEG_RETAINED                                    ~ "NMD_PTC_neg_retained",
    category %in% REF_LOST                                            ~ "NMD_PTC_neg_lost",
    TRUE                                                              ~ "other"
  ))

# Claim 2: Subgroup counts
sg_n <- sg |> count(subgroup) |> deframe()
check("Subgroup n  NMD_PTC_pos",          6888L, sg_n["NMD_PTC_pos"])
check("Subgroup n  NMD_PTC_neg_lost",     1252L, sg_n["NMD_PTC_neg_lost"])
check("Subgroup n  NMD_PTC_neg_retained",  700L, sg_n["NMD_PTC_neg_retained"])
check("Subgroup n  non_NMD",             31098L, sg_n["non_NMD"])
check("Subgroups sum to universe", 39938L, sum(sg_n))

# ── Recompute uORF flag + attention metric INDEPENDENTLY ──
# Main-CDS ORF per isoform via the priority chain
main_orf <- sel |>
  group_by(isoform_id) |>
  arrange(desc(is_ref_cds), desc(is_sqanti_cds), desc(orf_length), .by_group = TRUE) |>
  slice(1) |>
  ungroup() |>
  select(isoform_id, main_rank = orf_rank, main_frac = frac_position)

attn_long <- pred |>
  select(isoform_id, prob, starts_with("attn_")) |>
  pivot_longer(starts_with("attn_"), names_to = "rk", values_to = "att") |>
  mutate(orf_rank = as.integer(str_remove(rk, "attn_"))) |>
  select(-rk)

joined <- attn_long |>
  left_join(sel |> select(isoform_id, orf_rank, frac_position),
            by = c("isoform_id", "orf_rank")) |>
  left_join(main_orf, by = "isoform_id") |>
  mutate(is_uorf = !is.na(frac_position) & !is.na(main_frac) &
                   (frac_position < main_frac) &
                   (orf_rank != main_rank))

per_iso <- joined |>
  group_by(isoform_id) |>
  summarize(
    uorf_attn = sum(att[is_uorf], na.rm = TRUE),
    top_uorf  = is_uorf[which.max(att)],
    .groups = "drop"
  ) |>
  left_join(sg |> select(isoform_id, subgroup), by = "isoform_id")

# Claim 3: medians per subgroup
med <- per_iso |> group_by(subgroup) |>
  summarize(m = median(uorf_attn, na.rm = TRUE)) |> deframe()
check("Median uorf_attn  NMD_PTC_neg_lost",     0.550, round(med["NMD_PTC_neg_lost"], 3))
check("Median uorf_attn  NMD_PTC_neg_retained", 0.434, round(med["NMD_PTC_neg_retained"], 3))
check("Median uorf_attn  NMD_PTC_pos",          0.000, round(med["NMD_PTC_pos"], 3))
check("Median uorf_attn  non_NMD",              0.000, round(med["non_NMD"], 3))

# Claim 4: % top_is_uorf per subgroup
pct <- per_iso |> group_by(subgroup) |>
  summarize(p = round(100 * mean(top_uorf), 1)) |> deframe()
check("Pct top_is_uorf  NMD_PTC_neg_lost",      54.2, pct["NMD_PTC_neg_lost"])
check("Pct top_is_uorf  NMD_PTC_neg_retained",  44.1, pct["NMD_PTC_neg_retained"])
check("Pct top_is_uorf  NMD_PTC_pos",            4.8, pct["NMD_PTC_pos"])
check("Pct top_is_uorf  non_NMD",               11.2, pct["non_NMD"])

# Claim 5: Wilcoxon + Cliff's delta
ptc_pos_v <- per_iso |> filter(subgroup == "NMD_PTC_pos")         |> pull(uorf_attn)
ptc_neg_v <- per_iso |> filter(subgroup %in% c("NMD_PTC_neg_lost",
                                                "NMD_PTC_neg_retained")) |> pull(uorf_attn)
d_pn <- cliff.delta(ptc_neg_v, ptc_pos_v)
check("Cliff's delta PTC- vs PTC+", 0.61, signif(as.numeric(d_pn$estimate), 2))
check("Median PTC- (combined)",     0.51, round(median(ptc_neg_v), 2))
check("Median PTC+",                0.00, round(median(ptc_pos_v), 2))

# Claim 6: Fisher's OR
tab_ti <- per_iso |>
  filter(subgroup %in% c("NMD_PTC_pos","NMD_PTC_neg_lost","NMD_PTC_neg_retained")) |>
  mutate(grp = factor(if_else(subgroup == "NMD_PTC_pos", "PTC+","PTC-"),
                      levels = c("PTC-", "PTC+"))) |>  # PTC- as row 1 so OR>1 means PTC- has higher odds
  count(grp, top_uorf) |>
  pivot_wider(names_from = top_uorf, values_from = n, values_fill = 0) |>
  arrange(grp)
ft_mat <- as.matrix(tab_ti[, c("TRUE","FALSE")]); rownames(ft_mat) <- as.character(tab_ti$grp)
ft <- fisher.test(ft_mat)
check("Fisher OR PTC- vs PTC+",  20.3, signif(ft$estimate, 3))
check("Fisher 95% CI lower",     17.6, signif(ft$conf.int[1], 3))
check("Fisher 95% CI upper",     23.5, signif(ft$conf.int[2], 3))

# Claim 7: ATF4 specifics
atf4 <- per_iso |>
  filter(isoform_id %in% c("ENST00000674920.3","ENST00000337304.2")) |>
  left_join(pred |> select(isoform_id, prob), by = "isoform_id") |>
  arrange(isoform_id)
check("ATF4 ENST00000337304.2 P(NMD)", 0.240, round(atf4$prob[1], 3))
check("ATF4 ENST00000674920.3 P(NMD)", 0.252, round(atf4$prob[2], 3))
check("ATF4 337304 uORF attn",         0.558, round(atf4$uorf_attn[1], 3))
check("ATF4 674920 uORF attn",         0.542, round(atf4$uorf_attn[2], 3))
check("ATF4 337304 top_is_uORF",       TRUE,  atf4$top_uorf[1])
check("ATF4 674920 top_is_uORF",       TRUE,  atf4$top_uorf[2])

# Claim 8: prob means by label (sanity)
mean_lab1 <- mean(pred$prob[pred$label == 1])
mean_lab0 <- mean(pred$prob[pred$label == 0])
check("Mean prob | label==1", 0.785, round(mean_lab1, 3))
check("Mean prob | label==0", 0.171, round(mean_lab0, 3))

# ─────────────────────────────────────────────────────────────────────────
# uORF feature-signature section (Tables 5–6 in the Rmd)
# Recompute the per-uORF analysis from upstream sources, independent of the
# saved intermediate uorf_features_in_priority_slots.tsv.
# ─────────────────────────────────────────────────────────────────────────
cat("\n=== uORF feature-signature checks ===\n\n")

ptc_neg_ids <- sg |> filter(subgroup %in% c("NMD_PTC_neg_lost",
                                              "NMD_PTC_neg_retained")) |>
  pull(isoform_id)
main_orf2 <- sel |>
  group_by(isoform_id) |>
  arrange(desc(is_ref_cds), desc(is_sqanti_cds),
          desc(orf_length), .by_group = TRUE) |>
  slice(1) |> ungroup() |>
  select(isoform_id, main_rank = orf_rank, main_frac = frac_position)
attn_long2 <- pred |>
  select(isoform_id, starts_with("attn_")) |>
  pivot_longer(starts_with("attn_"), names_to = "rk", values_to = "attention") |>
  mutate(orf_rank = as.integer(str_remove(rk, "attn_")))
per_uorf2 <- attn_long2 |>
  left_join(sel |> select(isoform_id, orf_rank, frac_position,
                           orf_length, kozak_score, n_downstream_ejc),
            by = c("isoform_id", "orf_rank")) |>
  left_join(main_orf2, by = "isoform_id") |>
  filter(!is.na(frac_position), frac_position < main_frac,
         orf_rank != main_rank,
         isoform_id %in% ptc_neg_ids)

iso_uc <- per_uorf2 |> count(isoform_id, name = "n_uorfs")
check("PTC- NMD isoforms with >=1 uORF", 1428L, nrow(iso_uc))
check("PTC- NMD isoforms with >=2 uORFs (study pop)", 871L,
      sum(iso_uc$n_uorfs >= 2))

iso_conc2 <- per_uorf2 |>
  group_by(isoform_id) |>
  summarize(n_uorfs = n(),
            share = max(attention) / sum(attention),
            ne = if (n() > 1) {
              p <- attention / sum(attention); p <- p[p > 0]
              -sum(p * log(p)) / log(length(p))
            } else NA_real_,
            .groups = "drop") |>
  filter(n_uorfs >= 2)
check("Median top_uorf_share",     0.628, round(median(iso_conc2$share), 3))
check("Pct top_uorf_share >= 0.8", 24.5,  round(100 * mean(iso_conc2$share >= 0.8), 1))
check("Median normalized entropy", 0.861, round(median(iso_conc2$ne, na.rm = TRUE), 3))

# Within-isoform paired feature deltas
tv <- per_uorf2 |>
  group_by(isoform_id) |>
  filter(n() >= 2) |>
  mutate(is_top = attention == max(attention)) |>
  summarize(
    d_kozak  = first(kozak_score[is_top])      - mean(kozak_score[!is_top],      na.rm = TRUE),
    d_length = first(orf_length[is_top])       - mean(orf_length[!is_top],       na.rm = TRUE),
    d_ejc    = first(n_downstream_ejc[is_top]) - mean(n_downstream_ejc[!is_top], na.rm = TRUE),
    d_frac   = first(frac_position[is_top])    - mean(frac_position[!is_top],    na.rm = TRUE),
    .groups = "drop")
check("Median d_length",   57,    round(median(tv$d_length, na.rm = TRUE), 0))
check("Pct d_length > 0",  67.0,  round(100 * mean(tv$d_length > 0, na.rm = TRUE), 1))
check("Median d_frac",    -0.065, round(median(tv$d_frac,   na.rm = TRUE), 3))
check("Pct d_frac > 0",    25.8,  round(100 * mean(tv$d_frac > 0, na.rm = TRUE), 1))
check("Median d_kozak",    0,     round(median(tv$d_kozak,  na.rm = TRUE), 0))
check("Pct d_kozak > 0",   19.9,  round(100 * mean(tv$d_kozak > 0, na.rm = TRUE), 1))
check("Median d_ejc",      0,     round(median(tv$d_ejc,    na.rm = TRUE), 0))
check("Pct d_ejc > 0",     30.5,  round(100 * mean(tv$d_ejc > 0, na.rm = TRUE), 1))

cat("\n=== Summary ===\n")
cat(sprintf("PASS: %d   WARN: %d   FAIL: %d\n", pass, warn, fail))
if (fail > 0) quit(status = 1)
