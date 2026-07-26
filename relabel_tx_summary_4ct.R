#!/usr/bin/env Rscript
#
# Relabel tx_summary.tsv using new 4-cell-type mashr results (AT, DD, FB, MV).
#
# - NMD = union of nmd_responsive == TRUE across 4 cell types
# - non-NMD = intersection of adj.P.Val > 0.50 across 4 cell types
# - Isoforms in neither set are dropped from tx_summary
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

# --- Update tx_summary ---
tx <- read.delim(file.path(results_dir, "tx_summary_6ct.tsv"),
                 stringsAsFactors = FALSE)
cat(sprintf("\nOriginal tx_summary: %d rows (NMD=%d, non-NMD=%d)\n",
            nrow(tx), sum(tx$is_nmd == 1), sum(tx$is_nmd == 0)))

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
