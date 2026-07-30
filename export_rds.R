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

# ---------------------------------------------------------------------------
# nmd_path(key) — external inputs, resolved the same way data_prep.py resolves them:
# environment variable > config.yaml `paths:`. ONE source of truth across both languages.
#
# cache_dir and data_dir below were absolute /projects/talisman/... literals. That pinned
# the feature tables -- and therefore the model's entire isoform universe -- to one
# machine's copy of the LEGACY isopair tree. It was invisible until measured: 0 of 24
# isoforms that the rebuilt structures.rds added appear in the resulting tx_summary.tsv
# (2026-07-26). This script is the one that reads the analysis_cache, so it is the point
# where a rerun either picks up the rebuilt universe or silently repeats the old one.
# ---------------------------------------------------------------------------
nmd_path <- function(key, must_exist = FALSE) {
  env_of <- c(isopair_cache = "NMD_ISOPAIR_CACHE", isopair_data = "NMD_ISOPAIR_DATA",
              sqanti_class  = "NMD_SQANTI_CLASS",  sqanti_fasta = "NMD_SQANTI_FASTA",
              mashr_dir     = "NMD_MASHR_DIR")
  ev <- Sys.getenv(env_of[[key]], unset = "")
  src <- if (nzchar(ev)) paste0("$", env_of[[key]]) else "config.yaml"
  val <- if (nzchar(ev)) ev else {
    cfg_file <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
                                                           value = TRUE)[1])), "config.yaml")
    if (!file.exists(cfg_file)) cfg_file <- "config.yaml"
    yaml::read_yaml(cfg_file)$paths[[key]]
  }
  if (is.null(val) || !nzchar(val))
    stop(sprintf("no path configured for '%s': set $%s or add paths.%s to config.yaml",
                 key, env_of[[key]], key), call. = FALSE)
  val <- path.expand(val)
  if (must_exist && !file.exists(val))
    stop(sprintf("%s does not exist:\n    %s\n  (from %s)\n  Set $%s or edit paths.%s.",
                 key, val, src, env_of[[key]], key), call. = FALSE)
  val
}

cache_dir <- nmd_path("isopair_cache")

# ---------------------------------------------------------------------------
# out_dir was the literal "results_4ct" and it is the WORST place in this repo for that,
# because this script WRITES the eight feature tables the whole model chain reads. Two
# consequences, both silent:
#
#   1. data_prep.py is invoked with --results-dir results_4ct_dn for the deposit-native
#      rebuild and reads orf_features / tx_summary / ref_cds_features / td2_features /
#      junctions / paralog_genes from THERE. With the output hardcoded, the tables the
#      rebuild needs are written somewhere else and never arrive.
#   2. Writing into results_4ct OVERWRITES THE PUBLISHED FEATURE TABLES -- the artifacts the
#      deposit-native run exists to be compared against. That is the exact hazard
#      REPRODUCTION.md records for --results-dir on 03_train/evaluate/11/deepshap.
#
# Resolution order matches the rest of the project: flag > NMD_RESULTS_DIR > default. The
# default preserves the published behaviour, so no existing invocation changes. W76.
#
# ANCHORED TO THE SCRIPT'S OWN DIRECTORY, not the working directory. The first version of
# this fix left a relative out_dir CWD-relative, and REPRODUCTION.md invokes this script as
# `Rscript <model repo>/export_rds.R` FROM THE ANALYSIS REPO -- so the eight feature tables
# would have been written into the analysis repo while data_prep.py, which anchors to the
# model repo, looked for them here and found nothing. infer_uorf_attention.py anchors the
# same flag to its own REPO; this now matches. An ABSOLUTE --results-dir is honoured as
# given -- file.path() does NOT absorb an absolute second component, so it must be tested
# for rather than blindly joined.
# ---------------------------------------------------------------------------
out_dir <- local({
  a <- commandArgs(TRUE)
  # Accept both `--results-dir X` and `--results-dir=X`; ignore a value-less flag rather
  # than silently taking the next argument or falling through to the published default.
  eq <- grep("^--results-dir=", a, value = TRUE)
  i  <- which(a == "--results-dir")
  v <- if (length(eq)) sub("^--results-dir=", "", eq[1])
       else if (length(i) && length(a) > i[1]) a[i[1] + 1L]
       else Sys.getenv("NMD_RESULTS_DIR", unset = "")
  if (!nzchar(v)) v <- "results_4ct"          # nzchar guard: an exported-but-empty var
  v <- path.expand(v)                          # is not an override
  script_dir <- dirname(sub("^--file=", "",
                            grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  is_abs <- grepl("^(/|~|[A-Za-z]:)", v)
  if (is_abs || is.na(script_dir) || !nzchar(script_dir)) v else file.path(script_dir, v)
})
cat(sprintf("[results-dir] %s\n", out_dir))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

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

# LABEL PROVENANCE SIDECAR (2026-07-30). This script is the SOLE writer of tx_summary.tsv
# (D18 retired relabel_tx_summary_4ct.R for exactly that reason), so `is_nmd` here is whatever
# orfik_scan.rds carried -- and data_prep.py reads it straight into the training labels with no
# record of which scan it came from.
#
# That is the unguarded half of the hazard this file's own header documents: 0 of 24 isoforms
# added by the rebuilt structures.rds appear in the resulting tx_summary.tsv (2026-07-26), and
# nothing said so. A stale scan yields plausible labels, exit 0, and a model trained on the wrong
# universe. Recording the scan's identity and class counts makes the vintage checkable downstream
# instead of assumed.
#
# Deliberately NOT a claim about mashr: this script never reads the mashr CSVs, so it records only
# what it can observe. It does not reinstate the relabel step and does not change tx_summary.tsv.
local({
  tx_prov <- list(
    written_by  = "export_rds.R",
    written_at  = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    source_rds  = rds_path,
    source_mtime = format(file.mtime(rds_path), "%Y-%m-%dT%H:%M:%S%z"),
    source_bytes = as.numeric(file.size(rds_path)),
    n_rows      = nrow(orfik$tx_summary),
    n_nmd       = if ("is_nmd" %in% names(orfik$tx_summary))
                    sum(orfik$tx_summary$is_nmd == 1, na.rm = TRUE) else NA_integer_,
    n_non_nmd   = if ("is_nmd" %in% names(orfik$tx_summary))
                    sum(orfik$tx_summary$is_nmd == 0, na.rm = TRUE) else NA_integer_
  )
  write_json(tx_prov, file.path(out_dir, "tx_summary_provenance.json"),
             auto_unbox = TRUE, pretty = TRUE)
  cat(sprintf("  -> wrote tx_summary_provenance.json (%d rows, NMD=%s)\n",
              tx_prov$n_rows, format(tx_prov$n_nmd)))
})

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
data_dir <- nmd_path("isopair_data")

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
} else if (is.list(paralogs) && !is.null(paralogs$leakage_genes)) {
  # NAME THE ELEMENT; DO NOT unlist() THE WHOLE OBJECT (2026-07-29).
  #
  # paralog_genes.rds is a 5-element list from 05u_paralog_annotation.R: leakage_genes
  # (56 versioned gene ids -- the screen), leakage_pairs (161x5), all_expressed_pairs
  # (82,590x3), paralog_pairs (131,280x3) and metadata. The previous branch was
  # `data.frame(gene_id = unlist(paralogs))`, which flattened ALL FIVE into one column
  # labelled gene_id: 642,485 rows in which only 56 values were the gene list and the
  # rest were unversioned ids from the pair tables plus percent-identity NUMBERS like
  # 47.2152, all presented under a column name asserting they were gene ids.
  #
  # It happened not to change the screen's result -- data_prep matches versioned ids, and
  # noise cannot match -- so test_paralog = 122 is correct and claim 5.6.4 stands. That is
  # luck, not design: the file was wrong by four orders of magnitude and read as if right.
  #
  # leakage_genes is what the screen means: >=80% protein identity, both genes expressed,
  # and the pair straddling the train/test split (05u:69,196,211). It is versioned, which
  # is what data_prep.py's gene_lookup compares against.
  lg <- paralogs$leakage_genes
  stopifnot("leakage_genes must be a character vector" = is.character(lg))
  cat("  leakage_genes:", length(lg), "versioned gene IDs",
      sprintf("(%d carry a version suffix)\n", sum(grepl(".", lg, fixed = TRUE))))
  if (!is.null(paralogs$metadata$holdout_chrs))
    cat("  screen defined against holdout chrs:",
        paste(paralogs$metadata$holdout_chrs, collapse = ", "), "\n")
  safe_write_tsv(data.frame(gene_id = lg), file.path(out_dir, "paralog_genes.tsv"))
  cat("  -> wrote paralog_genes.tsv\n")

  # THE VALIDATION SIDE IS A SEPARATE FILE, because it is a separate SET (2026-07-29, D35).
  # 05u defines leakage as a pair straddling a split boundary, so the test-side and val-side
  # sets are computed against different boundaries and are disjoint -- 56 genes on chr1/3/5/7
  # and 19 on chr2/chr4. data_prep.py consults one per branch; a chr2 gene tested against the
  # test-side set can never match, which is why val_paralog was 0.
  if (!is.null(paralogs$val_leakage_genes)) {
    vlg <- paralogs$val_leakage_genes
    stopifnot("val_leakage_genes must be a character vector" = is.character(vlg))
    cat("  val_leakage_genes:", length(vlg), "versioned gene IDs\n")
    if (!is.null(paralogs$metadata$val_chrs))
      cat("  screen defined against val chrs:",
          paste(paralogs$metadata$val_chrs, collapse = ", "), "\n")
    safe_write_tsv(data.frame(gene_id = vlg),
                   file.path(out_dir, "val_paralog_genes.tsv"))
    cat("  -> wrote val_paralog_genes.tsv\n")
  } else {
    # Not fatal here -- an older RDS predates D35. data_prep.py raises loudly if the file is
    # missing, which is the right place for it: that is where the screen is actually applied.
    cat("  NOTE: this paralog_genes.rds has no val_leakage_genes (predates D35).\n")
    cat("        Re-run 05u_paralog_annotation.R --from-cache to add it (offline).\n")
  }
} else {
  # Refusing rather than coercing. A silent unlist() is what produced the defect above.
  stop("paralog_genes.rds is a ", paste(class(paralogs), collapse = "/"),
       " with elements [", paste(names(paralogs), collapse = ", "),
       "] and no `leakage_genes`. Name the element that holds the gene list; do not ",
       "flatten the object and hope the consumer's matching filters the rest out.")
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
