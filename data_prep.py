#!/usr/bin/env python3
"""
01_data_prep.py — Build HDF5 dataset for ORF-centric NMD model.

Reads exported TSVs (from export_rds.R) and full-length sequences from
SQANTI FASTA, extracts per-ORF sequence windows with 10-channel
encoding, assembles structural features, assigns train/val/test splits,
computes normalization stats, and writes a single HDF5.

Also produces ORF coverage diagnostics to validate K=10 cutoff.

Usage:
    python 01_data_prep.py [--results-dir results]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================
CACHE_DIR = Path("/projects/talisman/shared-data/nmd/isoform_transitions/"
                 "Version_6.0/isopair_wrapper/data_mashr/analysis_cache")
SQANTI_CLASS_PATH = Path("/projects/talisman/shared-data/nmd/sqanti/results/"
                         "nmd_lungcells_classification.txt")
FASTA_PATH = Path("/projects/talisman/shared-data/nmd/sqanti/results/"
                  "nmd_lungcells_corrected.fasta")

MAX_ORFS = 5
WINDOW_SIZES = [100, 500, 1000, 2000]
N_SEQ_CHANNELS = 9  # 4 nuc + junction + rolling GC + 3 frame position (one-hot)
HOLDOUT_CHRS = {"chr1", "chr3", "chr5", "chr7"}
VAL_CHRS = {"chr2", "chr4"}

# v5: minimal per-ORF structural features
# n_downstream_ejc included because junctions beyond the stop window are invisible to the CNN
ORF_FEATURE_COLS = [
    "frac_start",       # orf_start / tx_length (fractional start position)
    "frac_stop",        # orf_end / tx_length (fractional stop position)
    "is_ref_cds",       # binary: matches reference CDS ATG
    "is_sqanti_cds",    # binary: matches SQANTI/TD2 CDS ATG
    "n_downstream_ejc", # count of EJCs downstream of this ORF's stop codon
]

GC_ROLLING_WINDOW = 50  # bp for rolling GC computation

NUC_TO_IDX = np.zeros(256, dtype=np.int8)
NUC_TO_IDX[:] = -1
NUC_TO_IDX[ord("A")] = 0
NUC_TO_IDX[ord("C")] = 1
NUC_TO_IDX[ord("G")] = 2
NUC_TO_IDX[ord("T")] = 3

STOP_CODONS = {b"TAA", b"TAG", b"TGA"}


# =============================================================================
# v5: 9-channel sequence encoding
# =============================================================================
def seq_to_uint8(seq):
    """Convert sequence string to numpy uint8 array."""
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)


def parse_junctions(junc_str):
    """Parse comma-separated 1-based junction positions to 0-based set."""
    if pd.isna(junc_str) or junc_str == "":
        return set()
    return {int(x) - 1 for x in str(junc_str).split(",")}


def compute_rolling_gc(seq_uint8, window=GC_ROLLING_WINDOW):
    """Compute rolling GC fraction across the transcript (vectorized).

    Returns a float32 array of length len(seq_uint8) with local GC fraction.
    At edges, shrinks the window to available bases.
    """
    seq_len = len(seq_uint8)
    if seq_len == 0:
        return np.zeros(0, dtype=np.float32)

    is_gc = ((seq_uint8 == ord("G")) | (seq_uint8 == ord("C"))).astype(np.float32)
    cumsum = np.concatenate([[0.0], np.cumsum(is_gc)])

    half = window // 2
    # Vectorized: compute lo and hi for every position at once
    positions = np.arange(seq_len)
    lo = np.maximum(0, positions - half)
    hi = np.minimum(seq_len, positions + half + 1)
    gc_arr = ((cumsum[hi] - cumsum[lo]) / (hi - lo)).astype(np.float32)

    return gc_arr


def encode_window_v5(seq_uint8, junctions_set, center, half_win,
                     orf_start_0based, rolling_gc):
    """
    v5 9-channel window encoding (vectorized).

    Channels:
      0-3: one-hot A/C/G/T
      4:   exon-exon junction positions
      5:   rolling GC fraction (continuous, ~50bp sliding window)
      6-8: reading frame one-hot — codon position (0/1/2) relative to this ORF's ATG

    Returns (9, win_size) float16 array.
    """
    win_size = 2 * half_win
    encoded = np.zeros((N_SEQ_CHANNELS, win_size), dtype=np.float16)
    seq_len = len(seq_uint8)

    start = center - half_win
    end = center + half_win

    # Determine valid range within sequence
    w_start = max(0, start)
    w_end = min(seq_len, end)
    if w_start >= w_end:
        return encoded

    # Offset into the window array
    arr_start = w_start - start
    arr_end = arr_start + (w_end - w_start)
    n_valid = w_end - w_start

    # Extract the subsequence
    subseq = seq_uint8[w_start:w_end]
    nuc_idx = NUC_TO_IDX[subseq]

    # Channels 0-3: one-hot nucleotides (vectorized)
    valid_mask = nuc_idx >= 0
    valid_positions = np.arange(arr_start, arr_end)[valid_mask]
    valid_nuc = nuc_idx[valid_mask]
    encoded[valid_nuc, valid_positions] = 1.0

    # Channel 4: junctions (sparse — loop is fine, typically < 20 junctions)
    for j in junctions_set:
        idx_in_win = j - start
        if 0 <= idx_in_win < win_size:
            encoded[4, idx_in_win] = 1.0

    # Channel 5: rolling GC fraction (vectorized slice)
    if len(rolling_gc) > 0:
        encoded[5, arr_start:arr_end] = rolling_gc[w_start:w_end].astype(np.float16)

    # Channels 6-8: reading frame one-hot (vectorized)
    genomic_positions = np.arange(w_start, w_end)
    frames = (genomic_positions - orf_start_0based) % 3  # 0, 1, or 2
    win_positions = np.arange(arr_start, arr_end)
    encoded[6 + frames, win_positions] = 1.0

    return encoded


def encode_transcript_orfs(args):
    """
    v5: Encode all ORF windows for a single transcript across all window sizes.
    Designed for use with multiprocessing.

    args: (seq_str, junctions_str, atg_positions_list, stop_positions_list,
           orf_starts_list, orf_ends_list, orf_mask_list, window_sizes)

    Returns dict: {win_size: (atg_array, stop_array)} where each is (K, 9, W).
    """
    (seq_str, junc_str, atg_centers, stop_centers,
     orf_starts, orf_ends, orf_mask_arr, window_sizes) = args

    seq_uint8 = seq_to_uint8(seq_str) if seq_str else np.array([], dtype=np.uint8)
    junctions = parse_junctions(junc_str)
    K = len(orf_mask_arr)

    # Compute rolling GC once per transcript
    rolling_gc = compute_rolling_gc(seq_uint8)

    result = {}
    for win_size in window_sizes:
        half_win = win_size // 2
        atg_wins = np.zeros((K, N_SEQ_CHANNELS, win_size), dtype=np.float16)
        stop_wins = np.zeros((K, N_SEQ_CHANNELS, win_size), dtype=np.float16)

        for k in range(K):
            if not orf_mask_arr[k]:
                continue
            atg_pos = atg_centers[k]
            stop_pos = stop_centers[k]
            orf_s = orf_starts[k]

            if atg_pos >= 0:
                atg_wins[k] = encode_window_v5(
                    seq_uint8, junctions, atg_pos, half_win,
                    orf_s, rolling_gc)
            if stop_pos >= 0:
                stop_wins[k] = encode_window_v5(
                    seq_uint8, junctions, stop_pos, half_win,
                    orf_s, rolling_gc)

        result[win_size] = (atg_wins, stop_wins)

    return result


# =============================================================================
# Data loading
# =============================================================================
def load_orf_features(results_dir):
    path = results_dir / "orf_features.tsv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype={"isoform_id": str})
    print(f"  {len(df):,} ORFs across {df['isoform_id'].nunique():,} transcripts")
    return df


def load_tx_summary(results_dir):
    path = results_dir / "tx_summary.tsv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype={"isoform_id": str})
    print(f"  {len(df):,} transcripts")
    return df


def load_ref_features(results_dir):
    path = results_dir / "ref_cds_features.tsv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype={"isoform_id": str})
    df = df.drop_duplicates("isoform_id")
    print(f"  {len(df):,} isoforms (deduplicated), {len(df.columns)} columns")
    return df


def load_td2_features(results_dir):
    path = results_dir / "td2_features.tsv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype={"isoform_id": str})
    df = df.drop_duplicates("isoform_id")
    print(f"  {len(df):,} isoforms (deduplicated), {len(df.columns)} columns")
    return df


def load_fasta(fasta_path, target_ids=None):
    """Parse multi-line FASTA, optionally filtering to target_ids."""
    print(f"Loading {fasta_path} ...")
    sequences = {}
    target_set = set(target_ids) if target_ids is not None else None
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                # Save previous record
                if current_id is not None and (target_set is None or current_id in target_set):
                    sequences[current_id] = "".join(current_seq).upper()
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        # Save last record
        if current_id is not None and (target_set is None or current_id in target_set):
            sequences[current_id] = "".join(current_seq).upper()

    print(f"  {len(sequences):,} sequences loaded"
          f" (filtered from {target_set and len(target_set) or 'all'} target IDs)")
    return sequences


def load_junctions(results_dir):
    """Load junction positions from junctions.tsv."""
    path = results_dir / "junctions.tsv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype={"isoform_id": str, "junctions": str})
    junc_map = dict(zip(df["isoform_id"], df["junctions"].fillna("")))
    print(f"  {len(junc_map):,} isoforms with junction data")
    return junc_map


def load_paralog_genes(results_dir):
    path = results_dir / "paralog_genes.tsv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t")
    col = df.columns[0]
    genes = set(df[col].dropna().astype(str))
    print(f"  {len(genes):,} paralog genes")
    return genes


def load_cds_atg_positions(results_dir):
    """Load reference and SQANTI CDS ATG positions (1-based, transcript-relative)."""
    # Reference CDS ATG: ref_utr5_length is 0-based → ATG at ref_utr5_length + 1
    ref = pd.read_csv(results_dir / "ref_cds_features.tsv", sep="\t",
                       dtype={"isoform_id": str},
                       usecols=["isoform_id", "ref_atg_available", "ref_utr5_length"])
    ref = ref.drop_duplicates("isoform_id")
    ref_atg = ref[(ref["ref_atg_available"] == 1) & ref["ref_utr5_length"].notna()].copy()
    ref_atg["ref_atg_tx_pos"] = (ref_atg["ref_utr5_length"] + 1).astype(int)
    ref_map = dict(zip(ref_atg["isoform_id"], ref_atg["ref_atg_tx_pos"]))
    print(f"  Reference CDS ATG positions: {len(ref_map):,} isoforms")

    # SQANTI CDS ATG: CDS_start is 1-based transcript-relative
    sqanti = pd.read_csv(SQANTI_CLASS_PATH, sep="\t", engine="python",
                          dtype={"isoform": str},
                          usecols=["isoform", "coding", "CDS_start"])
    sqanti = sqanti[sqanti["coding"] == "coding"].drop_duplicates("isoform")
    sqanti_map = dict(zip(sqanti["isoform"],
                          sqanti["CDS_start"].astype(int)))
    print(f"  SQANTI CDS ATG positions: {len(sqanti_map):,} isoforms")

    return ref_map, sqanti_map


# =============================================================================
# ORF selection and diagnostics
# =============================================================================
def select_top_orfs(orf_df, max_k=MAX_ORFS):
    """Legacy: top-K by Kozak only. Replaced by select_priority_orfs()."""
    print(f"\nSelecting top-{max_k} ORFs per transcript by kozak_score ...")
    orf_df = orf_df.sort_values(["isoform_id", "kozak_score"],
                                ascending=[True, False])
    orf_df["orf_rank"] = orf_df.groupby("isoform_id").cumcount()
    selected = orf_df[orf_df["orf_rank"] < max_k].copy()
    n_tx = selected["isoform_id"].nunique()
    print(f"  Kept {len(selected):,} ORFs across {n_tx:,} transcripts")
    sizes = selected.groupby("isoform_id").size()
    print(f"  ORFs per transcript: median={sizes.median():.0f}, mean={sizes.mean():.1f}")
    return selected


def select_priority_orfs(orf_df, ref_atg_map, sqanti_atg_map, max_k=MAX_ORFS):
    """
    Select up to max_k ORFs per transcript with priority inclusion of CDS ORFs.

    Priority order:
      1. Reference CDS ORF (if available and found in orfik scan)
      2. SQANTI/TD2 CDS ORF (if available, different from ref, and found)
      3. Remaining slots filled by top Kozak-scored ORFs

    Returns DataFrame with orf_rank assigned and selection_reason column.
    """
    print(f"\nSelecting up to {max_k} ORFs per transcript (priority CDS + Kozak fill) ...")
    assert "orf_start" in orf_df.columns, "orf_start column required"

    # Sort by Kozak within each transcript (for tie-breaking and filling)
    orf_df = orf_df.sort_values(["isoform_id", "kozak_score"],
                                ascending=[True, False]).copy()

    # Pre-compute CDS matches per transcript (vectorized)
    orf_df["_is_ref_match"] = (
        orf_df["isoform_id"].map(ref_atg_map).fillna(-1) == orf_df["orf_start"])
    orf_df["_is_sqanti_match"] = (
        orf_df["isoform_id"].map(sqanti_atg_map).fillna(-1) == orf_df["orf_start"])

    selected_rows = []
    n_ref_included = 0
    n_sqanti_included = 0
    n_ref_available = 0
    n_sqanti_available = 0
    n_ref_not_found = 0
    n_sqanti_not_found = 0

    for tid, grp in orf_df.groupby("isoform_id", sort=False):
        chosen_indices = []
        reasons = []

        # 1. Reference CDS ORF
        has_ref_atg = tid in ref_atg_map
        if has_ref_atg:
            n_ref_available += 1
            ref_matches = grp[grp["_is_ref_match"]]
            if len(ref_matches) > 0:
                # Take highest Kozak among matches (already sorted)
                chosen_indices.append(ref_matches.index[0])
                reasons.append("ref_cds")
                n_ref_included += 1
            else:
                n_ref_not_found += 1

        # 2. SQANTI CDS ORF (if different from ref)
        has_sqanti_atg = tid in sqanti_atg_map
        if has_sqanti_atg:
            n_sqanti_available += 1
            sqanti_matches = grp[grp["_is_sqanti_match"]]
            if len(sqanti_matches) > 0:
                # Take highest Kozak among matches
                best_sqanti = sqanti_matches.index[0]
                if best_sqanti not in chosen_indices:
                    chosen_indices.append(best_sqanti)
                    reasons.append("sqanti_cds")
                    n_sqanti_included += 1
                else:
                    # Same ORF as ref CDS — already included
                    n_sqanti_included += 1
            else:
                n_sqanti_not_found += 1

        # 3. Fill remaining slots with top Kozak ORFs
        remaining = max_k - len(chosen_indices)
        if remaining > 0:
            kozak_candidates = grp[~grp.index.isin(chosen_indices)]
            fill = kozak_candidates.head(remaining)
            chosen_indices.extend(fill.index.tolist())
            reasons.extend(["kozak"] * len(fill))

        # Assign ranks and collect
        for rank, (idx, reason) in enumerate(zip(chosen_indices, reasons)):
            row = grp.loc[idx].copy()
            row["orf_rank"] = rank
            row["selection_reason"] = reason
            row["is_ref_cds"] = int(grp.loc[idx, "_is_ref_match"])
            row["is_sqanti_cds"] = int(grp.loc[idx, "_is_sqanti_match"])
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    selected.drop(columns=["_is_ref_match", "_is_sqanti_match"], inplace=True)

    n_tx = selected["isoform_id"].nunique()
    sizes = selected.groupby("isoform_id").size()

    print(f"  Kept {len(selected):,} ORFs across {n_tx:,} transcripts")
    print(f"  ORFs per transcript: median={sizes.median():.0f}, mean={sizes.mean():.1f}")
    print(f"\n  CDS inclusion diagnostics:")
    print(f"    Ref CDS available: {n_ref_available:,}, included: {n_ref_included:,}, "
          f"no matching ORF: {n_ref_not_found:,}")
    print(f"    SQANTI CDS available: {n_sqanti_available:,}, included: {n_sqanti_included:,}, "
          f"no matching ORF: {n_sqanti_not_found:,}")
    n_neither = n_tx - len(set(
        selected[selected["selection_reason"].isin(["ref_cds", "sqanti_cds"])]["isoform_id"]))
    print(f"    Transcripts with no CDS ORF in top-{max_k}: {n_neither:,}")

    return selected


def orf_coverage_diagnostics(orf_df, tx_summary, max_k=MAX_ORFS):
    print(f"\n{'='*60}")
    print(f"ORF Coverage Diagnostics (K={max_k})")
    print(f"{'='*60}")

    n_orfs = tx_summary["n_orfs"]
    print(f"\nORFs per transcript:")
    print(f"  min={n_orfs.min()}, median={n_orfs.median():.0f}, "
          f"mean={n_orfs.mean():.1f}, max={n_orfs.max()}")
    pct_over_k = (n_orfs > max_k).mean() * 100
    print(f"  Transcripts with >{max_k} ORFs: {pct_over_k:.1f}%")

    diag = pd.DataFrame({
        "metric": [
            f"transcripts_with_gt_{max_k}_orfs_pct",
            "median_orfs_per_tx",
            "mean_orfs_per_tx",
            "max_orfs_per_tx",
        ],
        "value": [
            round(pct_over_k, 2),
            round(n_orfs.median(), 1),
            round(n_orfs.mean(), 1),
            int(n_orfs.max()),
        ]
    })
    return diag


# =============================================================================
# Main dataset assembly
# =============================================================================
def build_dataset(results_dir, n_workers=8):
    # Load all data
    orf_df = load_orf_features(results_dir)
    tx_summary = load_tx_summary(results_dir)
    ref_features = load_ref_features(results_dir)  # still needed for gene_id + ref_atg_map
    junc_map = load_junctions(results_dir)
    paralog_genes = load_paralog_genes(results_dir)

    # Load sequences from FASTA (filtered to transcripts in tx_summary)
    target_ids = set(tx_summary["isoform_id"].values)
    seq_dict = load_fasta(FASTA_PATH, target_ids)

    # Diagnostics
    diag = orf_coverage_diagnostics(orf_df, tx_summary, MAX_ORFS)
    diag_path = results_dir / "orf_coverage_analysis.tsv"
    diag.to_csv(diag_path, sep="\t", index=False)
    print(f"\n  -> Saved diagnostics to {diag_path}")

    # Load CDS ATG positions BEFORE ORF selection (needed for priority inclusion)
    print("\nLoading CDS ATG positions ...")
    ref_atg_map, sqanti_atg_map = load_cds_atg_positions(results_dir)

    # Select ORFs with priority CDS inclusion
    orf_selected = select_priority_orfs(orf_df, ref_atg_map, sqanti_atg_map, MAX_ORFS)
    del orf_df

    # Export selected ORFs for downstream interpretation scripts
    selected_orfs_path = results_dir / "selected_orfs.tsv"
    orf_selected.to_csv(selected_orfs_path, sep="\t", index=False)
    print(f"\n  -> Saved selected ORFs to {selected_orfs_path}")

    # Master transcript list: intersection of tx_summary and FASTA
    fasta_ids = set(seq_dict.keys())
    tx_summary_ids = set(tx_summary["isoform_id"].values)
    master_ids = fasta_ids & tx_summary_ids
    missing_fasta = tx_summary_ids - fasta_ids
    missing_tx = fasta_ids & target_ids - tx_summary_ids
    if missing_fasta:
        print(f"  WARNING: {len(missing_fasta):,} tx_summary isoforms not in FASTA (dropped)")
    if missing_tx:
        print(f"  WARNING: {len(missing_tx):,} FASTA isoforms not in tx_summary (dropped)")

    # Use tx_summary order, filtered to master list
    tx_ids = tx_summary["isoform_id"].values
    tx_ids = np.array([tid for tid in tx_ids if tid in master_ids])
    n_tx = len(tx_ids)
    tx_id_to_idx = {tid: i for i, tid in enumerate(tx_ids)}
    print(f"\nMaster transcript list: {n_tx:,} isoforms")

    # Verify sequences: ATG at orf_start for rank-0 ORFs
    print("\nVerifying sequences ...")
    n_verified = 0
    n_atg_fail = 0
    for _, row in orf_selected[orf_selected["orf_rank"] == 0].iterrows():
        tid = row["isoform_id"]
        if tid not in seq_dict:
            continue
        seq = seq_dict[tid]
        orf_start_1based = int(row["orf_start"])
        codon = seq[orf_start_1based - 1 : orf_start_1based + 2]
        if codon == "ATG":
            n_verified += 1
        else:
            n_atg_fail += 1
            if n_atg_fail <= 5:
                print(f"  ATG verification FAIL: {tid} orf_start={orf_start_1based} "
                      f"codon={codon} start_codon={row.get('start_codon', '?')}")
    print(f"  ATG verified: {n_verified:,}, failed: {n_atg_fail:,}")
    if n_atg_fail > 0:
        print(f"  WARNING: {n_atg_fail} rank-0 ORFs do not have ATG at orf_start")

    # -------------------------------------------------------------------------
    # Assign splits: train / val / test / test_paralog
    # -------------------------------------------------------------------------
    print("\nAssigning splits ...")
    chr_map = dict(zip(tx_summary["isoform_id"], tx_summary["chr"]))
    gene_lookup = dict(zip(ref_features["isoform_id"], ref_features["gene_id"])) \
        if "gene_id" in ref_features.columns else {}

    splits = []
    for tid in tx_ids:
        ch = chr_map.get(tid, "")
        if ch in HOLDOUT_CHRS:
            gene = gene_lookup.get(tid, "")
            if gene in paralog_genes:
                splits.append("test_paralog")
            else:
                splits.append("test")
        elif ch in VAL_CHRS:
            splits.append("val")
        else:
            splits.append("train")

    splits = np.array(splits)
    for s in ["train", "val", "test", "test_paralog"]:
        print(f"  {s}: {(splits == s).sum():,}")

    # -------------------------------------------------------------------------
    # Per-ORF structural features (vectorized)
    # -------------------------------------------------------------------------
    print("\nAssembling per-ORF structural features ...")
    orf_feat_array = np.zeros((n_tx, MAX_ORFS, len(ORF_FEATURE_COLS)), dtype=np.float32)
    orf_mask = np.zeros((n_tx, MAX_ORFS), dtype=bool)
    atg_centers = np.full((n_tx, MAX_ORFS), -1, dtype=np.int32)
    stop_centers = np.full((n_tx, MAX_ORFS), -1, dtype=np.int32)
    orf_starts = np.full((n_tx, MAX_ORFS), -1, dtype=np.int32)
    orf_ends = np.full((n_tx, MAX_ORFS), -1, dtype=np.int32)

    orf_sel = orf_selected[orf_selected["isoform_id"].isin(tx_id_to_idx)].copy()
    orf_sel["_tx_idx"] = orf_sel["isoform_id"].map(tx_id_to_idx)
    tx_idx_arr = orf_sel["_tx_idx"].values
    rank_arr = orf_sel["orf_rank"].values

    orf_mask[tx_idx_arr, rank_arr] = True
    # orf_start/orf_end are 1-based. Store 0-based for encoding.
    orf_starts[tx_idx_arr, rank_arr] = (orf_sel["orf_start"].values - 1).astype(np.int32)
    orf_ends[tx_idx_arr, rank_arr] = (orf_sel["orf_end"].values - 1).astype(np.int32)
    # Center on middle nucleotide of codon
    atg_centers[tx_idx_arr, rank_arr] = (orf_sel["orf_start"].values - 1 + 1).astype(np.int32)
    stop_centers[tx_idx_arr, rank_arr] = (orf_sel["orf_end"].values - 1 - 1).astype(np.int32)

    # v5: compute frac_start and frac_stop per ORF
    tx_lengths = orf_sel["tx_length"].values.astype(np.float32)
    tx_lengths[tx_lengths < 1] = 1.0  # avoid division by zero
    orf_sel["frac_start"] = orf_sel["orf_start"].values / tx_lengths
    orf_sel["frac_stop"] = orf_sel["orf_end"].values / tx_lengths

    for j, col in enumerate(ORF_FEATURE_COLS):
        vals = orf_sel[col].fillna(0.0).values.astype(np.float32)
        orf_feat_array[tx_idx_arr, rank_arr, j] = vals

    print(f"  Transcripts with >=1 ORF: {orf_mask.any(axis=1).sum():,}")
    print(f"  ORF features ({len(ORF_FEATURE_COLS)}): {ORF_FEATURE_COLS}")
    del orf_sel, orf_selected

    # -------------------------------------------------------------------------
    # Compute normalization stats on training set
    # -------------------------------------------------------------------------
    print("\nComputing normalization stats on training set ...")
    train_mask_split = splits == "train"

    orf_feat_train = orf_feat_array[train_mask_split]
    orf_mask_train = orf_mask[train_mask_split]
    # Flatten valid ORFs only
    valid_orf_feats = orf_feat_train[orf_mask_train]
    orf_feat_mean = valid_orf_feats.mean(axis=0).astype(np.float32)
    orf_feat_std = valid_orf_feats.std(axis=0).astype(np.float32)
    orf_feat_std[orf_feat_std < 1e-8] = 1.0  # avoid division by zero

    print(f"  ORF feature means: {orf_feat_mean}")

    # -------------------------------------------------------------------------
    # Encode sequence windows — parallel across transcripts
    # -------------------------------------------------------------------------
    print(f"\nEncoding {N_SEQ_CHANNELS}-channel windows with {n_workers} workers ...")

    # Prepare arguments for parallel encoding (using FASTA sequences + junctions)
    encode_args = []
    for i in range(n_tx):
        tid = tx_ids[i]
        seq_str = seq_dict.get(tid, "")
        junc_str = junc_map.get(tid, "")
        encode_args.append((
            seq_str,
            junc_str,
            atg_centers[i].tolist(),
            stop_centers[i].tolist(),
            orf_starts[i].tolist(),
            orf_ends[i].tolist(),
            orf_mask[i].tolist(),
            WINDOW_SIZES,
        ))

    # -------------------------------------------------------------------------
    # Write HDF5 incrementally
    # -------------------------------------------------------------------------
    h5_path = results_dir / "nmd_orf_data.h5"
    print(f"\nBuilding HDF5 at {h5_path} ...")

    with h5py.File(h5_path, "w") as f:
        # Pre-create window datasets
        for win_size in WINDOW_SIZES:
            grp = f.create_group(f"w{win_size}")
            # Scale chunk size down for large windows to keep chunks < ~10 MB
            chunk_tx = min(256, n_tx) if win_size <= 500 else min(32, n_tx)
            grp.create_dataset("atg_windows",
                               shape=(n_tx, MAX_ORFS, N_SEQ_CHANNELS, win_size),
                               dtype=np.float16, compression="lzf",
                               chunks=(chunk_tx, MAX_ORFS, N_SEQ_CHANNELS, win_size))
            grp.create_dataset("stop_windows",
                               shape=(n_tx, MAX_ORFS, N_SEQ_CHANNELS, win_size),
                               dtype=np.float16, compression="lzf",
                               chunks=(chunk_tx, MAX_ORFS, N_SEQ_CHANNELS, win_size))

        # Process in batches with multiprocessing
        batch_size = 1000
        n_batches = (n_tx + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="  encoding"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_tx)
            batch_args = encode_args[start:end]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(encode_transcript_orfs, batch_args))

            for win_size in WINDOW_SIZES:
                atg_batch = np.stack([r[win_size][0] for r in results])
                stop_batch = np.stack([r[win_size][1] for r in results])
                f[f"w{win_size}/atg_windows"][start:end] = atg_batch
                f[f"w{win_size}/stop_windows"][start:end] = stop_batch

        for win_size in WINDOW_SIZES:
            shape = f[f"w{win_size}/atg_windows"].shape
            print(f"  w{win_size}: atg={shape}, stop={shape}")

        # Non-window data
        f.create_dataset("orf_features", data=orf_feat_array, compression="lzf")
        f.create_dataset("orf_mask", data=orf_mask, compression="lzf")
        # v5: no tx_features dataset
        # Labels and chr from tx_summary (keyed by master tx_ids order)
        tx_sum_indexed = tx_summary.set_index("isoform_id")
        labels = tx_sum_indexed.loc[tx_ids, "is_nmd"].values.astype(np.int8)
        chrs = tx_sum_indexed.loc[tx_ids, "chr"].values.astype(str)
        f.create_dataset("labels", data=labels)

        str_dt = h5py.string_dtype()
        f.create_dataset("chr", data=np.array(chrs, dtype="S"), dtype=str_dt)
        f.create_dataset("isoform_id", data=np.array(tx_ids, dtype="S"), dtype=str_dt)
        f.create_dataset("split", data=np.array(splits, dtype="S"), dtype=str_dt)

        # Normalization stats
        norm_grp = f.create_group("normalization")
        norm_grp.create_dataset("orf_feat_mean", data=orf_feat_mean)
        norm_grp.create_dataset("orf_feat_std", data=orf_feat_std)

        # Metadata attributes
        f.attrs["orf_feature_cols"] = json.dumps(ORF_FEATURE_COLS)
        f.attrs["window_sizes"] = json.dumps(WINDOW_SIZES)
        f.attrs["max_orfs"] = MAX_ORFS
        f.attrs["n_seq_channels"] = N_SEQ_CHANNELS
        f.attrs["holdout_chrs"] = json.dumps(sorted(HOLDOUT_CHRS))
        f.attrs["val_chrs"] = json.dumps(sorted(VAL_CHRS))
        f.attrs["n_transcripts"] = n_tx

    file_size_mb = os.path.getsize(h5_path) / 1e6
    print(f"\n  -> {h5_path} ({file_size_mb:.0f} MB)")
    print("Done.")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Build HDF5 dataset for NMD ORF model")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    required = ["orf_features.tsv", "tx_summary.tsv", "ref_cds_features.tsv",
                "junctions.tsv", "paralog_genes.tsv"]
    for fname in required:
        p = args.results_dir / fname
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run export_rds.R first.")
    if not FASTA_PATH.exists():
        sys.exit(f"ERROR: {FASTA_PATH} not found.")

    build_dataset(args.results_dir, n_workers=args.workers)


if __name__ == "__main__":
    main()
