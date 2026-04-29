#!/usr/bin/env python3
"""
09_export_polya.py — Export poly(A) annotations for test transcripts.

Joins SQANTI3 poly(A) annotations with model predictions to produce
polya_sqanti_test_{tag}.tsv expected by the report.

Columns: isoform_id, label, prob, polyA_motif_found, polyA_motif, polyA_dist
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="atg500_stop500")
    parser.add_argument("--sqanti-classification",
                        default="/projects/talisman/shared-data/nmd/sqanti/results/nmd_lungcells_classification.txt")
    args = parser.parse_args()

    results_dir = Path("results")

    # Load predictions (test set)
    preds = pd.read_csv(results_dir / f"predictions_{args.tag}.tsv", sep="\t",
                        dtype={"isoform_id": str})
    print(f"Predictions: {len(preds)} test transcripts ({int(preds['label'].sum())} NMD)")

    # Load SQANTI classification (poly(A) columns)
    print(f"Loading SQANTI classification: {args.sqanti_classification}")
    sqanti = pd.read_csv(args.sqanti_classification, sep="\t",
                         usecols=["isoform", "polyA_motif_found", "polyA_motif", "polyA_dist"],
                         dtype={"isoform": str})
    sqanti = sqanti.rename(columns={"isoform": "isoform_id"})
    print(f"  SQANTI: {len(sqanti)} isoforms, {sqanti['polyA_motif_found'].value_counts().to_dict()}")

    # Join
    merged = preds.merge(sqanti, on="isoform_id", how="left")
    n_with_polya_info = merged["polyA_motif_found"].notna().sum()
    print(f"  Merged: {len(merged)} rows, {n_with_polya_info} with poly(A) annotation")

    # Save
    out_cols = ["isoform_id", "label", "prob", "polyA_motif_found", "polyA_motif", "polyA_dist"]
    out_path = results_dir / f"polya_sqanti_test_{args.tag}.tsv"
    merged[out_cols].to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path}")

    # Sanity check
    for cls_label, cls_val in [("NMD", 1), ("Control", 0)]:
        sub = merged[merged["label"] == cls_val]
        n_found = sub["polyA_motif_found"].astype(str).str.lower().eq("true").sum()
        print(f"  {cls_label}: {n_found}/{len(sub)} ({100*n_found/len(sub):.1f}%) have poly(A) motif")

    print("Done.")


if __name__ == "__main__":
    main()
