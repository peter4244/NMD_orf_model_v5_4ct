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

from paths_config import resolve_path

from paths_config import load_config, selected_tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None,
                                            help="Window-config tag. Default: the `selected:` block in --config. "
                                                 "Never a hardcoded literal -- see utils.selected_tag.")
    parser.add_argument("--config", default="config.yaml",
                           help="Where the selected window configuration is read from")
    # Default resolved from config.yaml `paths:` / $NMD_SQANTI_CLASS rather than baked in.
    parser.add_argument("--sqanti-classification",
                        default=str(resolve_path("sqanti_class")))
    # --results-dir, the ninth script in this repo to need it. Without it this hardcoded
    # results_4ct, so pointed at a deposit-native question it read the PUBLISHED run and exited 0
    # reporting it -- the failure mode is silence, not an error.
    parser.add_argument("--results-dir", default="results_4ct")
    parser.add_argument("--split", default="all",
                        help="Which universe the predictions file describes. 'all' is the full "
                             "cohort (D74/D77). The split is part of the filename evaluate.py "
                             "writes, so the wrong value reads a different population.")
    args = parser.parse_args()

    # Resolve the tag from the ONE place that names the selected configuration.
    if args.tag is None:
        args.tag = selected_tag(load_config(args.config))
    results_dir = Path(args.results_dir)

    # Load predictions for the requested universe -- the file IS the population, so nothing is
    # filtered here.
    preds = pd.read_csv(results_dir / f"predictions_{args.tag}_{args.split}.tsv", sep="\t",
                        dtype={"isoform_id": str})
    print(f"Predictions: {len(preds)} transcripts in split={args.split} "
          f"({int(preds['label'].sum())} NMD)")

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
