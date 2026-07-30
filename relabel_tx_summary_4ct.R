#!/usr/bin/env Rscript
#
# Build tx_summary.tsv from tx_summary_prelabel.tsv using 4-cell-type mashr (AT, DD, FB, MV).
#
# - NMD = union of nmd_responsive == TRUE across 4 cell types
# - non-NMD = intersection of adj.P.Val > NON_NMD_ADJ_P across 4 cell types.
#   THE THRESHOLD IS 0.30, not the 0.50 this header claimed until 2026-07-30; it was lowered
#   for 4-CT mashr and the constant below has read 0.30 throughout. Header only, no behaviour.
# - Isoforms in neither set are dropped
# - Writes tx_summary_provenance.json alongside; data_prep.py requires it
#

# Repo root, from this script's own path -- so nothing depends on which machine it runs on.
REPO <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else normalizePath(getwd())
})

# External input: config.yaml `paths:` / $NMD_MASHR_DIR, not a baked-in absolute path.
mashr_dir <- (function() {
  ev <- Sys.getenv("NMD_MASHR_DIR", unset = "")
  if (nzchar(ev)) return(ev)
  yaml::read_yaml(file.path(REPO, "config.yaml"))$paths$mashr_dir
})()
# This repo's OWN output dir -- derive it, never hardcode one machine's home.
results_dir <- file.path(REPO, "results_4ct")
de_date <- "2026.3.10"

cts <- c("at", "dd", "fb", "mv")
NON_NMD_ADJ_P <- 0.30  # lowered from 0.50 for 4-cell-type mashr

# --- Load per-cell-type mashr results ---
nmd_per_ct <- list()
non_nmd_per_ct <- list()

for (ct in cts) {
  de <- read.csv(file.path(mashr_dir,
                            sprintf("nmd_mashr_die_%s_%s.csv", ct, de_date)))
  nmd_per_ct[[ct]]     <- de$txid[de$nmd_responsive == TRUE]
  non_nmd_per_ct[[ct]] <- de$txid[de$adj.P.Val > NON_NMD_ADJ_P]
  cat(sprintf("  %s: %d NMD, %d non-NMD\n", toupper(ct),
              length(nmd_per_ct[[ct]]), length(non_nmd_per_ct[[ct]])))
}

# --- Aggregate ---
all_nmd     <- unique(unlist(nmd_per_ct))
all_non_nmd <- Reduce(intersect, non_nmd_per_ct)

# Remove any overlap (should be 0)
overlap <- intersect(all_nmd, all_non_nmd)
if (length(overlap) > 0) {
  cat(sprintf("  WARNING: %d isoforms in both NMD and non-NMD — removing from non-NMD\n",
              length(overlap)))
  all_non_nmd <- setdiff(all_non_nmd, overlap)
}

cat(sprintf("\nAggregate: %d NMD, %d non-NMD\n", length(all_nmd), length(all_non_nmd)))

# --- GUARD: the non-NMD set must match the definition the paper documents ------------
#
# Supplemental Methods: "Non-NMD susceptible isoforms were defined as those with mashr
# lfsr >= 0.05 AND per-cell-type adj.P.Val > 0.30". The loop above applies only the
# adj.P.Val half. On the 2026.3.10 vintage the two are EXACTLY equivalent -- measured
# 2026-07-26, all 77,459 isoforms, zero difference -- because adj.P.Val > 0.30 in all four
# cell types already implies lfsr >= 0.05 in all four.
#
# That is a property of this vintage, not a theorem. mashr shrinkage borrows strength
# across conditions, so a re-run could produce a confidently-signed isoform (lfsr < 0.05)
# whose per-cell-type p-values all stay above 0.30 -- and it would enter the training set
# as a NEGATIVE while the paper's own definition excludes it. Silently, and only in the
# labels. Assert the equivalence rather than re-derive it; if it ever breaks, the fix is to
# apply both conditions, but that is a decision to take deliberately, not by drift.
#
# Columns are selected by PATTERN, not by a hardcoded ct->column map: the die files spell
# the cell types at/dd/fb/mv while the lfsr matrix spells them AT2/LAE/FB/MV, and a
# spelling mismatch of exactly this kind silently zeroed a published range on 2026-07-25.
# Requiring lfsr >= 0.05 in ALL cell types makes the mapping irrelevant.
lfsr_path <- file.path(mashr_dir, sprintf("mashr_isoform_lfsr_%s.csv", de_date))
if (!file.exists(lfsr_path)) {
  stop(sprintf("cannot verify the documented non-NMD definition: %s not found", lfsr_path))
}
lfsr <- read.csv(lfsr_path, check.names = FALSE)
lfsr_cols <- grep("^Smg1i_in_", names(lfsr), value = TRUE)
if (length(lfsr_cols) != length(cts)) {
  stop(sprintf("expected %d Smg1i_in_* columns in the lfsr matrix, found %d: %s",
               length(cts), length(lfsr_cols), paste(lfsr_cols, collapse = ", ")))
}
min_lfsr  <- do.call(pmin, c(lfsr[lfsr_cols], na.rm = TRUE))
lfsr_ok   <- lfsr$txid[min_lfsr >= 0.05]
violators <- setdiff(all_non_nmd, lfsr_ok)
if (length(violators) > 0) {
  stop(sprintf(paste0(
    "non-NMD set no longer matches the documented definition: %d of %d isoforms have ",
    "lfsr < 0.05 in at least one cell type.\n  e.g. %s\n",
    "  The Supplemental Methods require lfsr >= 0.05 AND adj.P.Val > 0.30; this script ",
    "applies only the second. Decide explicitly before proceeding."),
    length(violators), length(all_non_nmd),
    paste(utils::head(violators, 3), collapse = ", ")))
}
cat(sprintf("  guard: all %d non-NMD isoforms satisfy lfsr >= 0.05 (documented definition)\n",
            length(all_non_nmd)))

# --- Update tx_summary ---
# INPUT IS NOW tx_summary_prelabel.tsv, WRITTEN BY export_rds.R (2026-07-30, Pete's call).
#
# Supersedes an unstaged 2026-07-27 edit that repointed this from tx_summary_6ct.tsv to
# tx_summary_4ct.tsv and attributed that file to export_rds.R. export_rds.R wrote neither name
# -- it wrote tx_summary.tsv, the same path THIS script writes -- so the pipeline had two
# writers on one filename and read a bootstrap input that nothing in the repo produced.
#
# Now: export_rds.R -> tx_summary_prelabel.tsv -> (this script) -> tx_summary.tsv. One writer
# per file, and the repo regenerates its own inputs, which is what the deposit has to support.
prelabel_path <- file.path(results_dir, "tx_summary_prelabel.tsv")
if (!file.exists(prelabel_path)) {
  stop(sprintf(paste0(
    "%s not found.\n",
    "  It is written by export_rds.R, which must run first -- see README 'Build order'.\n",
    "  If you have a legacy tx_summary_6ct.tsv or tx_summary_4ct.tsv from before 2026-07-30,\n",
    "  do NOT rename it into place: neither was produced by this repo and neither has a\n",
    "  recorded row count, so its vintage is unknown. Re-run export_rds.R instead."),
    prelabel_path))
}
tx <- read.delim(prelabel_path, stringsAsFactors = FALSE)
nrow_in <- nrow(tx)
cat(sprintf("\nOriginal tx_summary: %d rows (NMD=%d, non-NMD=%d)\n",
            nrow_in, sum(tx$is_nmd == 1), sum(tx$is_nmd == 0)))

# Assign new labels
tx$is_nmd_new <- NA_integer_
tx$is_nmd_new[tx$isoform_id %in% all_nmd]     <- 1L
tx$is_nmd_new[tx$isoform_id %in% all_non_nmd] <- 0L

# Report changes
n_was_nmd     <- sum(tx$is_nmd == 1)
n_was_non     <- sum(tx$is_nmd == 0)
n_now_nmd     <- sum(tx$is_nmd_new == 1, na.rm = TRUE)
n_now_non     <- sum(tx$is_nmd_new == 0, na.rm = TRUE)
n_dropped     <- sum(is.na(tx$is_nmd_new))
n_flipped_to_nmd <- sum(tx$is_nmd == 0 & tx$is_nmd_new == 1, na.rm = TRUE)
n_flipped_to_non <- sum(tx$is_nmd == 1 & tx$is_nmd_new == 0, na.rm = TRUE)

cat(sprintf("\nNew labels:\n"))
cat(sprintf("  NMD:     %d → %d\n", n_was_nmd, n_now_nmd))
cat(sprintf("  non-NMD: %d → %d\n", n_was_non, n_now_non))
cat(sprintf("  Dropped (neither): %d\n", n_dropped))
cat(sprintf("  Flipped non-NMD → NMD: %d\n", n_flipped_to_nmd))
cat(sprintf("  Flipped NMD → non-NMD: %d\n", n_flipped_to_non))

# Drop isoforms in neither set, replace is_nmd
tx <- tx[!is.na(tx$is_nmd_new), ]
tx$is_nmd <- tx$is_nmd_new
tx$is_nmd_new <- NULL

cat(sprintf("\nFinal tx_summary: %d rows (NMD=%d, non-NMD=%d)\n",
            nrow(tx), sum(tx$is_nmd == 1), sum(tx$is_nmd == 0)))

# --- Write ---
out_path <- file.path(results_dir, "tx_summary.tsv")
write.table(tx, out_path, sep = "\t", row.names = FALSE, quote = TRUE)
cat(sprintf("\nWritten to %s\n", out_path))

# PROVENANCE SIDECAR (2026-07-30). data_prep.py reads tx_summary.tsv and takes its `is_nmd`
# column on faith -- there was no way for it to tell 4-CT labels from the scaffold's original
# ones, and the two were one filename apart. This records what produced the labels, and
# data_prep.py refuses to build an HDF5 unless it is present and its row count matches.
#
# A sidecar rather than a column: it survives being read by code that does not know about it,
# and it carries the mashr vintage, which no column could.
prov <- list(
  written_by      = "relabel_tx_summary_4ct.R",
  written_at      = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  input           = "tx_summary_prelabel.tsv",
  n_rows_in       = nrow_in,
  n_rows_out      = nrow(tx),
  n_nmd           = sum(tx$is_nmd == 1),
  n_non_nmd       = sum(tx$is_nmd == 0),
  mashr_dir       = mashr_dir,
  de_date         = de_date,
  cell_types      = cts,
  non_nmd_adj_p   = NON_NMD_ADJ_P
)
prov_path <- file.path(results_dir, "tx_summary_provenance.json")
jsonlite::write_json(prov, prov_path, auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("Provenance to %s (%d rows, NMD=%d, non-NMD=%d, mashr %s)\n",
            prov_path, prov$n_rows_out, prov$n_nmd, prov$n_non_nmd, de_date))
