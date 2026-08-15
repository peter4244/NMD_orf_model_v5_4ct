#!/usr/bin/env python3
"""
v5: Export per-sample ORF feature importance (grad × input) from NPZ to TSV.
Exports rank-0 ORF importance for each sample.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results-dir", required=True)
    # DERIVED FROM --results-dir, NOT DEFAULTED TO A TREE. This was
    # default="results_4ct/nmd_orf_data.h5" -- the PUBLISHED run -- so a caller that passed
    # --results-dir correctly still read the old HDF5, which is worse than a wrong default
    # because every visible flag looks right. Requiring it instead would break
    # slurm_export_importance_dn.sh, which passes --results-dir and not --hdf5-path; deriving
    # it keeps that caller working and makes the two impossible to disagree.
    parser.add_argument("--hdf5-path", default=None,
                        help="Defaults to <results-dir>/nmd_orf_data.h5.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    npz_path = results_dir / f"sample_importance_{args.tag}.npz"

    print(f"Loading {npz_path} ...")
    d = np.load(npz_path, allow_pickle=True)
    orf_gxi = d["orf_gxi"]       # (n_samples, K, n_orf_features)
    isoform_ids = d["isoform_ids"]
    labels = d["labels"]

    # Load feature names from HDF5 metadata
    hdf5_path = args.hdf5_path or str(Path(args.results_dir) / "nmd_orf_data.h5")
    print(f"[hdf5] {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        orf_names = json.loads(f.attrs["orf_feature_cols"])

    print(f"  orf_gxi shape: {orf_gxi.shape}")
    print(f"  ORF features ({len(orf_names)}): {orf_names}")

    # Export rank-0 ORF importance
    rank0_gxi = orf_gxi[:, 0, :]  # (n_samples, n_orf_features)
    df = pd.DataFrame(rank0_gxi, columns=orf_names)
    df["isoform_id"] = isoform_ids
    df["label"] = labels

    out_path = results_dir / f"sample_importance_tx_{args.tag}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path} ({len(df)} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
