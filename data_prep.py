#!/usr/bin/env python3
"""
01_data_prep.py — Build HDF5 dataset for ORF-centric NMD model.

Reads exported TSVs (from export_rds.R) and full-length sequences from
SQANTI FASTA, extracts per-ORF sequence windows with 9-channel
encoding, assembles structural features, assigns train/val/test splits,
computes normalization stats, and writes a single HDF5.

Also produces ORF coverage diagnostics to validate K=5 cutoff.

Usage:
    python data_prep.py [--results-dir results_4ct]
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

from paths_config import resolve_path

# =============================================================================
# Constants
# =============================================================================
# Resolved from config.yaml `paths:`, overridable per machine with $NMD_ISOPAIR_CACHE /
# $NMD_SQANTI_CLASS / $NMD_SQANTI_FASTA. These were absolute /projects/talisman/... literals,
# which pinned the model's entire isoform universe to one machine's copy of the LEGACY isopair
# tree -- and invisibly: 0 of 24 isoforms that the rebuilt structures.rds added appear in the
# resulting tx_summary.tsv (measured 2026-07-26). A path only changeable by editing code is a
# path nobody changes.
# RESOLVED AT CALL TIME, AGAINST THE CONFIG THE CALLER NAMED (2026-08-03).
#
# These were module-level constants, so they were computed at IMPORT -- before argparse ran -- and
# resolve_path() with no config argument always reads config.yaml. The effect was that
# `--config config_dn.yaml` could not influence a path no matter what it said: the deposit-native
# config was decorative, and a clean-room run died on
# "/projects/talisman/.../nmd_lungcells_corrected.fasta not found" while that file sat in the
# deposit. A flag that cannot change what it names is worse than no flag, because it reads as
# configured.
#
# CACHE_DIR is gone rather than deferred: it was assigned here and referenced nowhere, so it only
# ever served to fail early on a path this script does not use.
def _paths(config):
    """The input paths for one run, from the config that run was given."""
    return {"sqanti_class": resolve_path("sqanti_class", config_path=config),
            "sqanti_fasta": resolve_path("sqanti_fasta", config_path=config)}

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
                     orf_start_0based, limit_lo=None, limit_hi=None):
    """
    v5 9-channel window encoding (vectorized).

    Channels:
      0-3: one-hot A/C/G/T
      4:   exon-exon junction positions
      5:   rolling GC fraction (continuous, ~50bp sliding window)
      6-8: reading frame one-hot — codon position (0/1/2) relative to this ORF's ATG

    limit_lo / limit_hi bound the region that may be FILLED, without moving the window frame.
    Used to stop the ATG and stop windows crossing the ORF midpoint -- see
    encode_transcript_orfs. The anchor stays at array index half_win either way, so a clipped
    window is zero-padded on its inward side rather than shifted; positional channels keep
    meaning the same thing across ORFs of every length.

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
    # Never fill past the inward boundary. Applied to the VALID RANGE, not to `start`, so the
    # array offset below and the anchor's index are unchanged.
    if limit_lo is not None:
        w_start = max(w_start, limit_lo)
    if limit_hi is not None:
        w_end = min(w_end, limit_hi)
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
    # BOUNDED BY THE VALID RANGE, NOT THE WINDOW FRAME (fixed 2026-07-29). This tested
    # `0 <= j - start < win_size`, i.e. the frame, while every other channel fills only
    # arr_start:arr_end. Harmless while the sole clipping was the transcript edge -- a junction
    # is always inside the transcript, so it could not land in the padded region. NOT harmless
    # once the ORF-midpoint clip was added: junctions beyond the midpoint kept being marked in
    # the ATG window, and junctions before it in the stop window, so the two branches stayed
    # coupled through channel 4 after the sequence channels had been separated. The overlap fix
    # would have been 8/9 complete and looked finished, because the verification measured
    # nucleotide channels only.
    for j in junctions_set:
        if not (w_start <= j < w_end):
            continue
        encoded[4, j - start] = 1.0

    # Channel 5: rolling GC COMPUTED FROM THIS WINDOW'S OWN SEQUENCE (2026-07-29, Pete).
    #
    # It used to be a slice of a transcript-wide rolling_gc. The slice respected the clip but
    # the VALUES did not: GC_ROLLING_WINDOW is 50 with a centred support, so a value just
    # inside the midpoint averaged 25 bases on the far side of it. Verified by perturbation --
    # flipping a base at or after the midpoint changed the start window's channel 5, and
    # symmetrically. So after the sequence channels were separated, the two heads stayed
    # coupled through channel 5, over ~25 positions at each inner edge.
    #
    # That coupling is not confined to a minor channel. conv1 is Conv1d(9, 32, k), so every
    # filter sums all nine channels at once (model.py:49-50): channel 5 enters the same dot
    # product as the nucleotides at layer 1, and after BatchNorm/ReLU/pooling the branch
    # embedding is not additively separable. A small leak in channel 5 can move the whole
    # 32-dim embedding, which is exactly what the midpoint clip existed to prevent.
    #
    # Computing it from `seq_uint8[w_start:w_end]` makes the channel mean "GC of the sequence
    # this head can see", which is the only definition consistent with having separate heads.
    # compute_rolling_gc already shrinks its support at the edges of what it is given, so
    # padding is treated as absent rather than as GC-free -- zeros would dilute the value.
    #
    # KNOWN CONSEQUENCE, stated rather than discovered later: this also changes the OUTER
    # (UTR-side) edge, where there was no leak. Values in the first ~25 positions previously
    # included real transcript sequence beyond the window; now they do not. That is a small
    # loss of real information, accepted for a channel that means one thing everywhere instead
    # of two things at its two edges.
    if w_end > w_start:
        encoded[5, arr_start:arr_end] = compute_rolling_gc(
            seq_uint8[w_start:w_end]).astype(np.float16)

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

            # THE TWO WINDOWS MAY NOT CROSS THE ORF MIDPOINT (2026-07-29, Pete).
            #
            # Both windows are CENTRED on their anchor and span win_size, so they overlap
            # whenever the ORF is shorter than the window -- by (win_size - orf_length)
            # positions. ORFik is called with minimumLength=9 and longestORF=FALSE, so ORFs
            # reach ~30 nt; at win_size=500 such an ORF has ~94% of each window in common, and
            # at win_size=2000 the majority of ORFs overlap at all.
            #
            # That is not merely redundant input. The published branch decomposition
            # (60.7% structural / 28.8% stop / 10.5% ATG) treats the ATG branch and the stop
            # branch as two independent Shapley players. When they read the same bases, "how
            # much did the stop window contribute rather than the ATG window" has no clean
            # answer -- the attribution is split between inputs that are partly the same input.
            # It also means window SIZE changes how confounded that decomposition is, which no
            # selection criterion currently accounts for.
            #
            # The midpoint between the two anchors is the only boundary that guarantees
            # disjointness. Note this is the ORF's midpoint, not the transcript's: an ORF near
            # either end of the transcript would still overlap if clipped at the transcript's
            # centre.
            #
            # Applied UNCONDITIONALLY, which is safe because it is inert for long ORFs: if
            # stop - atg >= win_size then mid >= atg + half_win, i.e. at or beyond the ATG
            # window's own end, so the bound never binds. No branch, no special case.
            mid = (atg_pos + stop_pos) // 2 if (atg_pos >= 0 and stop_pos >= 0) else None
            if atg_pos >= 0:
                atg_wins[k] = encode_window_v5(
                    seq_uint8, junctions, atg_pos, half_win,
                    orf_s, limit_hi=mid)
            if stop_pos >= 0:
                stop_wins[k] = encode_window_v5(
                    seq_uint8, junctions, stop_pos, half_win,
                    orf_s, limit_lo=mid)

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
    _assert_labels_are_relabeled(results_dir, df)
    return df


def _assert_labels_are_relabeled(results_dir, tx_summary):
    """Refuse to build an HDF5 from a tx_summary whose labels have no recorded provenance.

    THE FAILURE THIS EXISTS TO PREVENT (2026-07-30). `is_nmd` is read straight into the training
    labels at the bottom of build_dataset with no check of any kind, and export_rds.R is its sole
    writer (D18 retired relabel_tx_summary_4ct.R precisely so there would be one writer). So the
    labels are whatever orfik_scan.rds carried, and nothing records WHICH scan that was.

    That is the unguarded half of the hazard export_rds.R's own header documents: 0 of 24 isoforms
    added by the rebuilt structures.rds appear in the resulting tx_summary.tsv (2026-07-26), and
    nothing said so. A stale scan gives plausible labels, exit 0, and a model trained on the wrong
    universe -- the same shape as C44 and the 2026-07-28 hybrid.

    export_rds.R now writes tx_summary_provenance.json beside the table; this requires it to exist
    and to describe the file actually on disk. NOTE it does NOT reinstate the relabel step and
    asserts nothing about mashr: export_rds.R never reads the mashr CSVs, so the sidecar records
    only the scan's identity and class counts.

    Escape hatch matches the house pattern -- an environment variable that PRINTS what it is
    conceding, not a quiet argument default:

        NMD_ALLOW_UNVERIFIED_LABELS=1
    """
    prov_path = Path(results_dir) / "tx_summary_provenance.json"
    allow = os.environ.get("NMD_ALLOW_UNVERIFIED_LABELS") == "1"

    if not prov_path.exists():
        msg = (f"{prov_path} not found, so the NMD labels in tx_summary.tsv have no recorded "
               f"provenance: there is no way to tell which orfik_scan.rds produced them.\n"
               f"  Re-run export_rds.R --results-dir <this dir> (it writes both files), or set "
               f"NMD_ALLOW_UNVERIFIED_LABELS=1 to build anyway.\n"
               f"  Trees built before 2026-07-30 have no sidecar; use the variable there, and "
               f"note in the log that the label vintage is unverified.")
        if not allow:
            raise FileNotFoundError(msg)
        print(f"\n  WARNING: NMD_ALLOW_UNVERIFIED_LABELS=1 -- {msg}")
        return

    with open(prov_path) as f:
        prov = json.load(f)

    # FIELD NAMES, and why this reads two of them. The sidecar was first written by
    # relabel_tx_summary_4ct.R as `n_rows_out`; when D18 moved the job to export_rds.R (the sole
    # writer of tx_summary.tsv) the field became `n_rows`, and this consumer was not updated.
    # Result, found by running it 2026-07-30: prov.get("n_rows_out") returned None, the print
    # raised TypeError on {None:,}, and -- far worse -- the `is not None` escape below meant the
    # ROW-COUNT CHECK WOULD HAVE SILENTLY NO-OPPED had the print not crashed. A guard whose
    # checks evaporate when a field is missing is not a guard. So:
    #   * both spellings are accepted, newest first;
    #   * a MISSING field is now fatal rather than a skipped check.
    n_rows = prov.get("n_rows", prov.get("n_rows_out"))
    n_nmd = prov.get("n_nmd")
    missing = [k for k, v in (("n_rows", n_rows), ("n_nmd", n_nmd)) if v is None]
    if missing:
        raise ValueError(
            f"{prov_path} is missing {missing}, so it cannot be checked against "
            f"tx_summary.tsv and the label vintage is unverifiable. This is a MALFORMED sidecar, "
            f"not an absent one -- re-run export_rds.R rather than deleting it, and do not reach "
            f"for NMD_ALLOW_UNVERIFIED_LABELS, which is for trees that never had a sidecar.")

    # `de_date` exists only on the legacy relabel sidecar: export_rds.R never reads the mashr CSVs
    # and so cannot honestly claim a DE vintage. Reported when present, omitted when not.
    de = prov.get("de_date")
    print(f"  label provenance: {prov.get('written_by')} @ {prov.get('written_at')}"
          + (f", mashr {de}" if de else "")
          + f", {n_rows:,} rows, NMD={n_nmd:,}")
    if prov.get("source_rds"):
        print(f"    scan: {prov['source_rds']}")
        print(f"          mtime {prov.get('source_mtime')}, {prov.get('source_bytes', 0):,} bytes")

    # The sidecar must describe THIS file, not a previous vintage of it. A stale sidecar beside a
    # regenerated tx_summary is exactly the "documented but not true" state the check exists for.
    if n_rows != len(tx_summary):
        raise ValueError(
            f"tx_summary_provenance.json says {n_rows:,} rows but tx_summary.tsv has "
            f"{len(tx_summary):,}. The sidecar is stale, or the two files came from different "
            f"runs. Re-run export_rds.R rather than editing either.")
    if "is_nmd" not in tx_summary.columns:
        raise ValueError("tx_summary.tsv has no is_nmd column; the labels cannot be checked.")
    actual = int((tx_summary["is_nmd"] == 1).sum())
    if actual != n_nmd:
        raise ValueError(
            f"tx_summary_provenance.json says NMD={n_nmd:,} but tx_summary.tsv has "
            f"{actual:,}. Re-run export_rds.R.")


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


def load_paralog_genes(results_dir, which="test"):
    """Load ONE side's leakage-gene set. The two sides are different sets, not one shared set.

    WHY THIS TAKES AN ARGUMENT NOW (2026-07-29). 05u defines leakage as a pair straddling a
    split boundary, so "leaks into the test set" and "leaks into the validation set" are
    computed against DIFFERENT boundaries and yield disjoint gene sets (56 on chr1/3/5/7, 19
    on chr2/chr4, zero overlap -- they are different chromosomes). Testing a chr2 gene against
    the test-side set can never match, which is exactly why val_paralog came out 0 after the
    val branch was added: the branch was right, the set it consulted was the wrong one.
    """
    fname = {"test": "paralog_genes.tsv", "val": "val_paralog_genes.tsv"}[which]
    path = results_dir / fname
    if which == "val" and not path.exists():
        # Loud, not silent: without this file the validation split cannot be screened, and a
        # sweep scored on an unscreened val set relocates the published leak (D31).
        raise FileNotFoundError(
            f"{path} not found. The validation leakage screen needs it. Re-run "
            f"05u_paralog_annotation.R (--from-cache is offline) so paralog_genes.rds carries "
            f"val_leakage_genes, then export_rds.R to write the TSV.")
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t")
    col = df.columns[0]
    genes = set(df[col].dropna().astype(str))
    print(f"  {len(genes):,} {which}-side leakage genes")
    return genes


def load_cds_atg_positions(results_dir, sqanti_class_path):
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
    sqanti = pd.read_csv(sqanti_class_path, sep="\t", engine="python",
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
def build_dataset(results_dir, n_workers=8, paths=None, config_path=None):
    # Load all data
    orf_df = load_orf_features(results_dir)
    tx_summary = load_tx_summary(results_dir)
    ref_features = load_ref_features(results_dir)  # still needed for gene_id + ref_atg_map
    junc_map = load_junctions(results_dir)
    paralog_genes = load_paralog_genes(results_dir, which="test")
    val_paralog_genes = load_paralog_genes(results_dir, which="val")

    # Load sequences from FASTA (filtered to transcripts in tx_summary)
    target_ids = set(tx_summary["isoform_id"].values)
    seq_dict = load_fasta(paths["sqanti_fasta"], target_ids)

    # Diagnostics
    diag = orf_coverage_diagnostics(orf_df, tx_summary, MAX_ORFS)
    diag_path = results_dir / "orf_coverage_analysis.tsv"
    diag.to_csv(diag_path, sep="\t", index=False)
    print(f"\n  -> Saved diagnostics to {diag_path}")

    # Load CDS ATG positions BEFORE ORF selection (needed for priority inclusion)
    print("\nLoading CDS ATG positions ...")
    ref_atg_map, sqanti_atg_map = load_cds_atg_positions(results_dir, paths["sqanti_class"])

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

    # READ-THROUGH LOCI ARE EXCLUDED ENTIRELY (2026-07-29, Pete's call).
    #
    # 233 gene_ids in ref_cds_features are composites of the form ENSGa.v::ENSGb.v -- two
    # adjacent genes transcribed as one unit. They are not one gene, so EVERY gene-level
    # operation on them is ill-defined: the split is assigned per gene (below), the paralog
    # leakage screen keys on gene, and 05u_paralog_annotation.R strips only the TRAILING
    # version before querying Ensembl, so a composite becomes a mangled string that matches
    # nothing and the locus is invisible to the paralog search.
    #
    # That last part had a measured consequence, which is why removal beats patching the
    # parser: two chr2 validation genes (ENSG00000144118.15, ENSG00000138069.19) have a
    # >=80%-identity paralog whose ONLY presence in this universe is inside a composite locus,
    # so the screen never flagged them -- real leakage, one into train and one into test.
    # Patching the ID parsing would flag them. Removing the composites deletes the twin
    # sequence instead, so the leakage does not exist to be flagged.
    #
    # THE PARALOG SPLIT COUNTS DO NOT MOVE, and an earlier version of this comment said they
    # would ("Two more on the holdout side, which is why test_paralog will not stay at 122").
    # That cannot happen, and the reason is the same defect that motivated the exclusion: a
    # composite id is `ENSGa.v::ENSGb.v`, which never equals a plain `ENSG` in paralog_genes,
    # so a composite-locus transcript was NEVER in test_paralog to begin with. Excluding it can
    # only shrink train/val/test. Measured on the 2026-07-30 tables before this ran: of the 278
    # excluded transcripts, 176 are train-side, 77 holdout-side and 25 val-side, and ZERO have a
    # gene in either paralog list -- test_paralog stays 122 and val_paralog stays 56.
    #
    # So the leakage is removed by deleting the twin SEQUENCE, not by reassigning a split. Do not
    # read an unchanged test_paralog as evidence the exclusion failed; unchanged is correct.
    #
    # Cost is 278 transcripts on 233 loci. Against the 42,043-row deposit-native scaffold that is
    # 0.66%; the published tree's 39,938 would give 0.70%, and the two are different populations
    # (see METHODS "Isoform universe"). Cheap for removing a whole class of gene-level ambiguity
    # rather than one of its symptoms.
    _gene_of = dict(zip(ref_features["isoform_id"], ref_features["gene_id"])) \
        if "gene_id" in ref_features.columns else {}
    # COUNT THE TRANSCRIPTS THAT HAVE NO GENE (2026-07-30). `_gene_of.get(tid, "")` defaults to
    # the empty string, and "" contains no "::" and is in no paralog set -- so a transcript
    # missing from ref_cds_features.tsv silently skips BOTH gene-level screens: the read-through
    # exclusion here and the paralog leakage screen below. Nothing counted them, so their number
    # was unknown, and "0 flagged" was indistinguishable from "0 checked".
    _n_no_gene = sum(1 for tid in tx_ids if not str(_gene_of.get(tid, "")))
    if _n_no_gene:
        print(f"\n  {_n_no_gene:,} of {len(tx_ids):,} transcripts have no gene_id in "
              f"ref_cds_features.tsv. These are NOT screened for read-through composites and "
              f"NOT screened for paralog leakage -- they cannot be, there is no gene to key on.")
    else:
        print(f"\n  gene_id present for all {len(tx_ids):,} transcripts "
              f"(both gene-level screens see the whole universe).")
    _readthrough = np.array([("::" in str(_gene_of.get(tid, ""))) for tid in tx_ids])
    if _readthrough.any():
        _loci = {_gene_of[t] for t, f in zip(tx_ids, _readthrough) if f}
        print(f"\nExcluding read-through loci: {_readthrough.sum():,} transcripts "
              f"on {len(_loci):,} composite gene ids "
              f"({100 * _readthrough.sum() / len(tx_ids):.2f}% of {len(tx_ids):,})")
        tx_ids = tx_ids[~_readthrough]
    else:
        # Loud rather than silent: if the pattern stops matching, the exclusion has silently
        # become a no-op and the ambiguity is back without anything saying so.
        print("\nWARNING: no read-through (::) gene ids found. Expected ~233 -- has the "
              "gene_id format changed? The exclusion is doing nothing.")

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

    # A TRANSCRIPT WITH NO CHROMOSOME SILENTLY BECOMES TRAINING DATA (counted from 2026-07-30).
    # chr_map.get(tid, "") returns "", which is in neither HOLDOUT_CHRS nor VAL_CHRS, so it falls
    # through to the `else` and joins train. That may well be the right default -- but it was
    # happening unrecorded, so the training set could absorb transcripts of unknown location
    # without the log showing a thing.
    n_no_chr = 0
    splits = []
    for tid in tx_ids:
        ch = chr_map.get(tid, "")
        if not str(ch):
            n_no_chr += 1
        gene = gene_lookup.get(tid, "")
        # EACH BRANCH CONSULTS ITS OWN SIDE'S SET. A single `is_paralog` computed against the
        # test-side set was the bug: it can never fire for a chr2/chr4 gene.
        if ch in HOLDOUT_CHRS:
            splits.append("test_paralog" if gene in paralog_genes else "test")
        elif ch in VAL_CHRS:
            # THE VAL SCREEN IS NOW SYMMETRIC WITH THE TEST SCREEN (2026-07-29).
            # It was not: the paralog test sat inside the HOLDOUT_CHRS branch only, so every
            # chr2/chr4 transcript went to `val` unscreened. That was harmless while val was
            # used only for early stopping, and it stops being harmless the moment val becomes
            # the SELECTION set -- which is exactly what re-running the window sweep on
            # train+val does. A config chosen on a paralog-contaminated val set carries the
            # same leak the sweep is being re-run to remove, just relocated.
            splits.append("val_paralog" if gene in val_paralog_genes else "val")
        else:
            splits.append("train")

    splits = np.array(splits)
    if n_no_chr:
        print(f"  {n_no_chr:,} transcripts had NO chromosome and were assigned to train "
              f"by fall-through. Verify that is intended.")
    for s in ["train", "val", "val_paralog", "test", "test_paralog"]:
        print(f"  {s}: {(splits == s).sum():,}")
    # THE SPLITS MUST ACCOUNT FOR EVERY TRANSCRIPT. Cheap, and it turns the five counts above
    # into a closed set rather than five numbers a reader has to add up and trust.
    _named = sum((splits == s).sum()
                 for s in ["train", "val", "val_paralog", "test", "test_paralog"])
    if _named != len(tx_ids):
        raise ValueError(f"splits account for {_named:,} transcripts but the master list has "
                         f"{len(tx_ids):,}; an unexpected split label was assigned: "
                         f"{sorted(set(splits.tolist()))}")
    print(f"  total: {_named:,}  (= master transcript list, checked)")

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
        # ── PROVENANCE STAMP (2026-08-04) ─────────────────────────────────────────────────
        # An H5 could not say which config built it, and that is why a dataset built from
        # CHANNING inputs sat in a directory named results_4ct_dn for five days while every
        # downstream number inherited the wrong universe. Finding it required reading a
        # five-day-old SLURM log; by then four different isoform universes were in play
        # (10,131 / 10,597 / 10,520 / 10,522 test rows) and no artifact could be matched to
        # the run that produced it.
        #
        # So the file now carries its own provenance. This is the DETECTION half of the fix --
        # load_config()/resolve_path() removing their defaults is the PREVENTION half, and
        # prevention alone leaves every artifact already on disk unattributable.
        #
        # Deliberately recorded: the config actually used, the resolved input paths (not the
        # configured ones -- an env var can override), the counts, and the code vintage.
        _paths_used = paths or {}
        f.attrs["built_by"] = "data_prep.py"
        f.attrs["config_path"] = str(config_path) if config_path else "UNKNOWN"
        for _k, _v in sorted(_paths_used.items()):
            f.attrs[f"input_{_k}"] = str(_v)
        f.attrs["n_transcripts"] = int(n_tx)
        try:
            import subprocess
            f.attrs["git_commit"] = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10).stdout.strip() or "UNKNOWN"
        except Exception:
            f.attrs["git_commit"] = "UNKNOWN"
        print(f"  [provenance] config={f.attrs['config_path']}")
        for _k in sorted(_paths_used):
            print(f"  [provenance] {_k} -> {_paths_used[_k]}")

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
    parser.add_argument("--results-dir", type=Path, default=Path("results_4ct"))
    parser.add_argument("--workers", type=int, default=8)
    # --config REACHES THE PATHS NOW. It did not exist here at all, so every path came from
    # config.yaml regardless of which config a caller believed they had selected.
    parser.add_argument("--config", required=True,
                        help="Config whose paths: block names the inputs. config_dn.yaml points "
                             "at the deposit; config.yaml at the Channing tree.")
    args = parser.parse_args()
    paths = _paths(args.config)

    required = ["orf_features.tsv", "tx_summary.tsv", "ref_cds_features.tsv",
                "junctions.tsv", "paralog_genes.tsv"]
    for fname in required:
        p = args.results_dir / fname
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run export_rds.R first.")
    if not paths["sqanti_fasta"].exists():
        sys.exit(f"ERROR: {paths['sqanti_fasta']} not found (from --config {args.config}).")

    build_dataset(args.results_dir, n_workers=args.workers, paths=paths,
                  config_path=args.config)


if __name__ == "__main__":
    main()
