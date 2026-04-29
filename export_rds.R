#!/usr/bin/env Rscript
# =============================================================================
# 01a_export_rds.R — Export RDS files to TSV for Python consumption
#
# Inputs:  orfik_scan.rds, ref_cds_features_all.rds, paralog_genes.rds
# Outputs: results/orf_features.tsv       — per-ORF features (2.4M rows)
#          results/tx_summary.tsv         — per-transcript ORF summary (61K rows)
#          results/ref_cds_features.tsv   — structural features (61K rows)
#          results/synthetic_cds.tsv      — CDS coordinates
#          results/paralog_genes.tsv      — paralog gene list
#          results/orf_scan_metadata.json — scan parameters
#
# Note: cnn_data.tsv is already in TSV format and read directly by Python.
# =============================================================================

library(jsonlite)

cache_dir <- "/projects/talisman/shared-data/nmd/isoform_transitions/Version_6.0/isopair_wrapper/data_mashr/analysis_cache"
out_dir   <- "results"
dir.create(out_dir, showWarnings = FALSE)

# Helper: convert logical columns to 0/1 integers for clean Python consumption
logicals_to_int <- function(df) {
  bool_cols <- sapply(df, is.logical)
  if (any(bool_cols)) {
    cat("  converting logical cols to 0/1:",
        paste(names(which(bool_cols)), collapse = ", "), "\n")
    df[bool_cols] <- lapply(df[bool_cols], as.integer)
  }
  df
}

# Helper: write TSV with safe defaults (na="", quote only when needed)
safe_write_tsv <- function(df, path) {
  df <- logicals_to_int(df)
  write.table(df, file = path, sep = "\t", row.names = FALSE,
              quote = TRUE, qmethod = "double", na = "")
}

# ===========================================================================
# orfik_scan.rds
# ===========================================================================
rds_path <- file.path(cache_dir, "orfik_scan.rds")
stopifnot("orfik_scan.rds not found" = file.exists(rds_path))

cat("=== Loading orfik_scan.rds ===\n")
orfik <- readRDS(rds_path)

stopifnot("orf_features missing" = is.data.frame(orfik$orf_features))
stopifnot("tx_summary missing"   = is.data.frame(orfik$tx_summary))

cat("  orf_features:", nrow(orfik$orf_features), "rows x",
    ncol(orfik$orf_features), "cols\n")
cat("  tx_summary:  ", nrow(orfik$tx_summary), "rows x",
    ncol(orfik$tx_summary), "cols\n")

safe_write_tsv(orfik$orf_features, file.path(out_dir, "orf_features.tsv"))
cat("  -> wrote orf_features.tsv\n")

orfik$orf_features <- NULL  # free before writing next table

safe_write_tsv(orfik$tx_summary, file.path(out_dir, "tx_summary.tsv"))
cat("  -> wrote tx_summary.tsv\n")

write_json(orfik$metadata, file.path(out_dir, "orf_scan_metadata.json"),
           auto_unbox = TRUE, pretty = TRUE)
cat("  -> wrote orf_scan_metadata.json\n")

rm(orfik); gc(verbose = FALSE)

# ===========================================================================
# ref_cds_features_all.rds
# ===========================================================================
rds_path <- file.path(cache_dir, "ref_cds_features_all.rds")
stopifnot("ref_cds_features_all.rds not found" = file.exists(rds_path))

cat("\n=== Loading ref_cds_features_all.rds ===\n")
ref <- readRDS(rds_path)

stopifnot("features missing" = is.data.frame(ref$features))

cat("  features:    ", nrow(ref$features), "rows x",
    ncol(ref$features), "cols\n")

safe_write_tsv(ref$features, file.path(out_dir, "ref_cds_features.tsv"))
cat("  -> wrote ref_cds_features.tsv\n")

if (!is.null(ref$synthetic_cds) && is.data.frame(ref$synthetic_cds)) {
  safe_write_tsv(ref$synthetic_cds, file.path(out_dir, "synthetic_cds.tsv"))
  cat("  -> wrote synthetic_cds.tsv (", nrow(ref$synthetic_cds), "rows)\n")
}

rm(ref); gc(verbose = FALSE)

# ===========================================================================
# TD2-based transcript features (utr5_features_all.rds + ptc.rds + cds.rds)
# ===========================================================================
data_dir <- "/projects/talisman/shared-data/nmd/isoform_transitions/Version_6.0/isopair_wrapper/data_mashr"

cat("\n=== Building TD2 features ===\n")

# --- a) TD2 5'UTR features (6 of 8) from utr5_features_all.rds ---
utr5_path <- file.path(data_dir, "analysis_cache/utr5_features_all.rds")
stopifnot("utr5_features_all.rds not found" = file.exists(utr5_path))

utr5_all <- readRDS(utr5_path)$isoform_features
cat("  utr5_features_all:", nrow(utr5_all), "rows,", sum(!utr5_all$excluded), "non-excluded\n")

td2_utr5 <- utr5_all[!utr5_all$excluded, ]
td2_utr5 <- data.frame(
  isoform_id                = td2_utr5$isoform_id,
  td2_atg_density           = td2_utr5$atg_density,
  td2_atg_strong_kozak      = td2_utr5$n_strong_kozak_atg,
  td2_uorf_count_overlapping = td2_utr5$n_orfs_overlapping,
  td2_uorf_count_outframe   = td2_utr5$n_orfs_outframe,
  td2_utr5_orf_coverage     = td2_utr5$pct_utr5_in_orfs,
  td2_stop_density           = td2_utr5$stop_density,
  td2_utr5_length           = td2_utr5$utr5_length,
  stringsAsFactors = FALSE
)
rm(utr5_all); gc(verbose = FALSE)

# --- b) TD2 downstream_ejc (1 of 8) from ptc.rds ---
ptc_path <- file.path(data_dir, "ptc.rds")
stopifnot("ptc.rds not found" = file.exists(ptc_path))

ptc <- readRDS(ptc_path)
cat("  ptc:", nrow(ptc), "rows\n")

td2_ejc <- data.frame(
  isoform_id       = ptc$isoform_id,
  td2_downstream_ejc = pmin(ptc$n_downstream_ejcs, 5L),
  stringsAsFactors = FALSE
)
rm(ptc); gc(verbose = FALSE)

# --- c) TD2 log_utr3_length (1 of 8) from cds.rds + structures.rds ---
cds_path <- file.path(data_dir, "cds.rds")
structs_path <- file.path(data_dir, "structures.rds")
stopifnot("cds.rds not found" = file.exists(cds_path))
stopifnot("structures.rds not found" = file.exists(structs_path))

cds_all <- readRDS(cds_path)
structs_all <- readRDS(structs_path)

cds_coding <- cds_all[cds_all$coding_status == "coding",
                       c("isoform_id", "cds_start", "cds_stop", "strand")]

td2_utr3 <- merge(structs_all, cds_coding, by = c("isoform_id", "strand"))
cat("  structures x cds_coding:", nrow(td2_utr3), "rows\n")

# Compute 3'UTR length (same logic as 05l_unified_model.R lines 140-164)
td2_utr3$utr3_length <- mapply(function(starts, ends, cds_start, cds_stop, strand) {
  if (strand == "+") {
    sum(pmax(0L, ends - pmax(starts, cds_stop)))
  } else {
    sum(pmax(0L, pmin(ends, cds_start) - starts))
  }
}, td2_utr3$exon_starts, td2_utr3$exon_ends,
   td2_utr3$cds_start, td2_utr3$cds_stop, td2_utr3$strand)

td2_utr3 <- data.frame(
  isoform_id          = td2_utr3$isoform_id,
  td2_log_utr3_length = log1p(td2_utr3$utr3_length),
  stringsAsFactors = FALSE
)

# --- d) Junction positions in transcript space (for Python data_prep.py) ---
cat("\n  Computing junction positions from structures...\n")
junc_list <- vector("list", nrow(structs_all))
for (i in seq_len(nrow(structs_all))) {
  starts <- structs_all$exon_starts[[i]]
  ends <- structs_all$exon_ends[[i]]
  strand <- structs_all$strand[i]
  n_exons <- length(starts)
  if (n_exons <= 1) { junc_list[[i]] <- ""; next }
  exon_lengths <- ends - starts + 1L
  if (strand == "-") exon_lengths <- rev(exon_lengths)
  junctions <- cumsum(exon_lengths)[-n_exons]
  junc_list[[i]] <- paste(junctions, collapse = ",")
}

junc_df <- data.frame(
  isoform_id = structs_all$isoform_id,
  junctions = unlist(junc_list),
  stringsAsFactors = FALSE
)
safe_write_tsv(junc_df, file.path(out_dir, "junctions.tsv"))
cat("  -> wrote junctions.tsv (", nrow(junc_df), "rows)\n")

rm(cds_all, structs_all, cds_coding, junc_df, junc_list); gc(verbose = FALSE)

# --- Join all three and write ---
td2_features <- merge(td2_ejc, td2_utr3, by = "isoform_id", all = TRUE)
td2_features <- merge(td2_features, td2_utr5, by = "isoform_id", all = TRUE)
cat("  Final td2_features:", nrow(td2_features), "rows,",
    ncol(td2_features), "cols\n")
cat("  Non-NA counts:\n")
for (col in setdiff(names(td2_features), "isoform_id")) {
  cat(sprintf("    %s: %d\n", col, sum(!is.na(td2_features[[col]]))))
}

safe_write_tsv(td2_features, file.path(out_dir, "td2_features.tsv"))
cat("  -> wrote td2_features.tsv\n")

rm(td2_ejc, td2_utr3, td2_utr5, td2_features); gc(verbose = FALSE)

# ===========================================================================
# paralog_genes.rds
# ===========================================================================
rds_path <- file.path(cache_dir, "paralog_genes.rds")
stopifnot("paralog_genes.rds not found" = file.exists(rds_path))

cat("\n=== Loading paralog_genes.rds ===\n")
paralogs <- readRDS(rds_path)

cat("  class:", paste(class(paralogs), collapse = ", "), "\n")
if (is.data.frame(paralogs)) {
  cat("  dim:", nrow(paralogs), "x", ncol(paralogs), "\n")
  safe_write_tsv(paralogs, file.path(out_dir, "paralog_genes.tsv"))
  cat("  -> wrote paralog_genes.tsv\n")
} else if (is.character(paralogs)) {
  cat("  length:", length(paralogs), "gene IDs\n")
  safe_write_tsv(data.frame(gene_id = paralogs),
                 file.path(out_dir, "paralog_genes.tsv"))
  cat("  -> wrote paralog_genes.tsv\n")
} else {
  df <- data.frame(gene_id = unlist(paralogs))
  cat("  coerced to data.frame:", nrow(df), "rows\n")
  safe_write_tsv(df, file.path(out_dir, "paralog_genes.tsv"))
  cat("  -> wrote paralog_genes.tsv\n")
}

# ===========================================================================
# Verification
# ===========================================================================
cat("\n=== Verification ===\n")
exported <- list.files(out_dir, pattern = "\\.tsv$|\\.json$")
for (f in exported) {
  sz <- file.size(file.path(out_dir, f))
  cat(sprintf("  %-30s %s\n", f,
              ifelse(sz > 1e6, sprintf("%.1f MB", sz / 1e6),
                     sprintf("%.1f KB", sz / 1e3))))
}

cat("\nDone.\n")
