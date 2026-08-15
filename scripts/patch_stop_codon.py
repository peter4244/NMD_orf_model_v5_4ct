#!/usr/bin/env python3
"""
patch_stop_codon.py — Replace the stop_codon column in selected_orfs.tsv with
the bug-free codon decoded from the HDF5 one-hot stop window.

Background: BUGFIX_STOP_CODON_2026-03-31.md documents that the upstream
05s_orfik_scan.R was reading positions orf_end+1..orf_end+3 (post-stop) instead
of orf_end-2..orf_end (the actual stop codon). The fix was applied upstream on
2026-03-31, but the 4ct selected_orfs.tsv (Apr 29) was generated against an
already-buggy intermediate and never re-run. This script applies the fix
directly without re-running the upstream R pipeline.

The HDF5 one-hot encoding is authoritative (bug doc explicitly states the
window center is the middle nucleotide of the stop codon, and the model and
all SHAP analyses derive stop codon identity from this encoding).
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


NUCS = np.array(['A', 'C', 'G', 'T'])


def decode_stop_codon(stop_window_orf):
    """Decode the 3-nt stop codon from positions [-1, 0, +1] of a stop window.

    stop_window_orf: array of shape (n_channels >= 4, n_positions=500)
    Returns 3-letter string (e.g. 'TGA') or 'NNN' if any position is padded.
    """
    # Positions 249, 250, 251 (0-indexed) = relative positions -1, 0, +1
    # Channels 0,1,2,3 = A,C,G,T
    triplet = stop_window_orf[:4, 249:252].astype(np.float32)
    codon = []
    for p in range(3):
        col = triplet[:, p]
        if col.sum() < 0.5:
            codon.append('N')
        else:
            codon.append(NUCS[int(np.argmax(col))])
    return ''.join(codon)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # NO DEFAULT RESULTS TREE. It was "results_4ct" -- the PUBLISHED run -- so a bare call
    # silently read the old universe and exited 0. Every wrapper already passes
    # --results-dir, so requiring it costs nothing and removes the one path that could
    # produce published-tree numbers while looking deposit-native. Same reasoning as
    # deepshap.py's --split: a default of any value is how the wrong population goes invisible.
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite selected_orfs.tsv. Otherwise writes selected_orfs_fixed.tsv.")
    ap.add_argument("--filter-to-h5", action="store_true",
                    help="Drop rows whose isoform_id is not in the HDF5 (e.g. 6ct→4ct trim).")
    args = ap.parse_args()

    results = Path(args.results_dir)
    h5_path = results / "nmd_orf_data.h5"
    sel_path = results / "selected_orfs.tsv"
    out_path = sel_path if args.in_place else results / "selected_orfs_fixed.tsv"

    if not h5_path.exists():
        sys.exit(f"HDF5 not found: {h5_path}")
    if not sel_path.exists():
        sys.exit(f"selected_orfs not found: {sel_path}")

    print(f"Reading {sel_path} ...")
    sel = pd.read_csv(sel_path, sep="\t")
    print(f"  {len(sel)} rows")
    n_iso_unique = sel['isoform_id'].nunique()
    print(f"  {n_iso_unique} unique isoforms, ranks: {sorted(sel['orf_rank'].unique())}")

    # Diagnostic: pre-fix canonical stop rate
    pre_canonical = sel['stop_codon'].isin(['TGA', 'TAA', 'TAG']).mean()
    print(f"  Pre-fix canonical (TGA/TAA/TAG) stop rate: {100*pre_canonical:.1f}%")

    print(f"\nReading HDF5 stop windows (bulk) from {h5_path} ...")
    with h5py.File(h5_path, "r") as h5:
        iso_ids = np.array([i.decode() if isinstance(i, bytes) else i
                            for i in h5["isoform_id"][:]])
        iso_to_row = {iid: i for i, iid in enumerate(iso_ids)}
        # Bulk read: stop_windows[:, :, 0:4, 249:252] -> (39938, 5, 4, 3) ~ 2.4M float16 = 4.6MB
        stop_triplets = h5["w500/stop_windows"][:, :, 0:4, 249:252].astype(np.float32)
        print(f"  Bulk-loaded triplets shape: {stop_triplets.shape}, dtype={stop_triplets.dtype}")

    # Vectorized decode: for each (iso, rank, position), find argmax channel
    # If sum at a position < 0.5 -> N (padding)
    sums = stop_triplets.sum(axis=2)              # (39938, 5, 3)
    argmax = stop_triplets.argmax(axis=2)         # (39938, 5, 3) -> 0..3
    is_pad = sums < 0.5                           # (39938, 5, 3)

    nuc_idx_to_char = np.array(['A', 'C', 'G', 'T'])
    chars = nuc_idx_to_char[argmax]               # (39938, 5, 3)
    chars[is_pad] = 'N'

    # Concatenate the 3 positions into a string per (iso, rank)
    codons_grid = np.array(['' for _ in range(stop_triplets.shape[0] * stop_triplets.shape[1])],
                           dtype=object).reshape(stop_triplets.shape[0], stop_triplets.shape[1])
    for p in range(3):
        codons_grid = codons_grid + chars[:, :, p]
    print(f"  Decoded codons grid shape: {codons_grid.shape}")

    # Look up each (isoform_id, orf_rank) row in selected_orfs and assign the codon
    n_total = len(sel)
    new_codons = np.empty(n_total, dtype=object)
    rows = np.array([iso_to_row.get(iid, -1) for iid in sel["isoform_id"].values])
    ranks = sel["orf_rank"].values
    missing = (rows == -1).sum()
    valid = rows >= 0
    new_codons[valid] = codons_grid[rows[valid], ranks[valid]]
    new_codons[~valid] = sel["stop_codon"].values[~valid]  # leave as-is for missing
    print(f"  Done. Missing isoforms: {missing}")

    # Diagnostics: post-fix
    sel["stop_codon"] = new_codons
    if args.filter_to_h5:
        before = len(sel)
        sel = sel[sel["isoform_id"].isin(iso_to_row)].copy()
        print(f"  Filtered to HDF5 isoforms: {before} -> {len(sel)} rows")
    post_canonical = sel['stop_codon'].isin(['TGA', 'TAA', 'TAG']).mean()
    print(f"\nPost-fix canonical stop rate: {100*post_canonical:.1f}%")
    print(f"  N-containing codons (padded windows): {(sel['stop_codon'].str.contains('N', na=False)).sum()}")
    print(f"  Top 5 codons in patched data:")
    print(sel['stop_codon'].value_counts().head().to_string())

    print(f"\nWriting {out_path} ...")
    sel.to_csv(out_path, sep="\t", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
