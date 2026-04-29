#!/usr/bin/env python3
"""
04_interpret_attention.py — Attention weight analysis for NMD ORF model.

Joins attention weights with ORF-level features to understand which ORFs
the model attends to and how attention patterns differ between NMD and
non-NMD transcripts.

Outputs:
  - attention_by_orf_type.tsv: mean attention by ORF type (ref-CDS, SQANTI, etc.)
  - attention_entropy.tsv: per-transcript attention entropy, stratified by class
  - attention_vs_features.tsv: correlation of attention with ORF features
  - attention_noptc_analysis.tsv: attention patterns for no-PTC NMD isoforms
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, mannwhitneyu, spearmanr


def load_attention_and_predictions(results_dir, tag):
    attn = pd.read_csv(results_dir / f"attention_weights_{tag}.tsv", sep="\t")
    preds = pd.read_csv(results_dir / f"predictions_{tag}.tsv", sep="\t")
    return attn, preds


def load_orf_features_with_rank(results_dir, max_k=10):
    """Load selected ORFs with pre-assigned ranks from data_prep.py.

    Uses selected_orfs.tsv which has priority-based ranking (ref CDS first,
    then SQANTI CDS, then Kozak fill) matching the HDF5 orf_rank indices.
    """
    path = results_dir / "selected_orfs.tsv"
    orf = pd.read_csv(path, sep="\t")
    # Strip quotes from column names if present
    orf.columns = [c.strip('"') for c in orf.columns]
    orf["isoform_id"] = orf["isoform_id"].str.strip('"')
    orf = orf[orf["orf_rank"] < max_k].copy()
    return orf



def analysis_1_attention_by_orf_type(attn_merged, results_dir, tag=""):
    """Mean attention weight by ORF type, stratified by NMD/non-NMD."""
    print("\n=== Attention by ORF type ===")

    rows = []
    for label_name, label_val in [("NMD", 1), ("non-NMD", 0)]:
        subset = attn_merged[attn_merged["label"] == label_val]

        for orf_type, col in [("ref_CDS", "is_ref_cds"),
                               ("sqanti_CDS", "is_sqanti_cds"),
                               ("has_downstream_EJC", "has_downstream_ejc")]:
            if col not in subset.columns:
                continue
            for val in [0, 1]:
                s = subset[subset[col] == val]["attn_weight"]
                if len(s) > 0:
                    rows.append({
                        "class": label_name,
                        "orf_type": orf_type,
                        "value": val,
                        "mean_attn": s.mean(),
                        "median_attn": s.median(),
                        "std_attn": s.std(),
                        "n_orfs": len(s),
                    })

        # By ORF rank
        for rank in range(10):
            s = subset[subset["orf_rank"] == rank]["attn_weight"]
            if len(s) > 0:
                rows.append({
                    "class": label_name,
                    "orf_type": f"rank_{rank}",
                    "value": rank,
                    "mean_attn": s.mean(),
                    "median_attn": s.median(),
                    "std_attn": s.std(),
                    "n_orfs": len(s),
                })

    df = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"attention_by_orf_type{suffix}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")

    # Print summary
    type_df = df[~df["orf_type"].str.startswith("rank_")]
    print(type_df.to_string(index=False))

    # Statistical tests
    print("\n  Mann-Whitney U tests (NMD class, is_ref_cds=1 vs 0):")
    nmd = attn_merged[attn_merged["label"] == 1]
    if "is_ref_cds" in nmd.columns:
        g1 = nmd[nmd["is_ref_cds"] == 1]["attn_weight"]
        g0 = nmd[nmd["is_ref_cds"] == 0]["attn_weight"]
        if len(g1) > 10 and len(g0) > 10:
            stat, p = mannwhitneyu(g1, g0, alternative="two-sided")
            print(f"    ref_CDS=1 mean={g1.mean():.4f} vs =0 mean={g0.mean():.4f}, p={p:.2e}")

    return df


def analysis_2_attention_entropy(attn, preds, results_dir, tag=""):
    """Per-transcript attention entropy — concentrated vs diffuse."""
    print("\n=== Attention entropy ===")

    # Pivot to get attention vector per transcript
    attn_pivot = attn.pivot(index="isoform_id", columns="orf_rank",
                            values="attn_weight").fillna(0)
    attn_matrix = attn_pivot.values

    # Compute Shannon entropy (base 2)
    ent = np.array([entropy(row[row > 0], base=2) for row in attn_matrix])

    # Max entropy per transcript = log2(n_valid_orfs)
    n_valid = np.array([(row > 0).sum() for row in attn_matrix])
    max_ent = np.log2(np.maximum(n_valid, 1))  # avoid log2(0)

    ent_df = pd.DataFrame({
        "isoform_id": attn_pivot.index,
        "attn_entropy": ent,
        "n_valid_orfs": n_valid,
        "normalized_entropy": np.where(max_ent > 0, ent / max_ent, 0.0),
        "max_attn": attn_matrix.max(axis=1),
        "argmax_orf": attn_matrix.argmax(axis=1),
    })
    ent_df = ent_df.merge(preds[["isoform_id", "label", "prob"]], on="isoform_id")

    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"attention_entropy{suffix}.tsv"
    ent_df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")

    # Summary by class
    for label_name, label_val in [("NMD", 1), ("non-NMD", 0)]:
        s = ent_df[ent_df["label"] == label_val]
        print(f"  {label_name}: entropy mean={s['attn_entropy'].mean():.3f}, "
              f"median={s['attn_entropy'].median():.3f}, "
              f"max_attn mean={s['max_attn'].mean():.3f}")

    # Test: is attention more concentrated for NMD?
    nmd_ent = ent_df[ent_df["label"] == 1]["attn_entropy"]
    ctrl_ent = ent_df[ent_df["label"] == 0]["attn_entropy"]
    if len(nmd_ent) > 10 and len(ctrl_ent) > 10:
        stat, p = mannwhitneyu(nmd_ent, ctrl_ent, alternative="two-sided")
        print(f"  Mann-Whitney (NMD vs non-NMD entropy): p={p:.2e}")

    # Argmax ORF distribution
    print("\n  Most-attended ORF rank distribution:")
    for label_name, label_val in [("NMD", 1), ("non-NMD", 0)]:
        s = ent_df[ent_df["label"] == label_val]
        counts = s["argmax_orf"].value_counts().sort_index()
        pcts = (counts / len(s) * 100).round(1)
        print(f"    {label_name}: {dict(pcts)}")

    return ent_df


def analysis_3_attention_feature_correlation(attn_merged, results_dir, tag=""):
    """Spearman correlation between attention weight and ORF features."""
    print("\n=== Attention vs ORF feature correlations ===")

    feature_cols = ["orf_length", "frac_position", "frac_tx_covered",
                    "kozak_score", "n_upstream_atgs", "n_downstream_ejc",
                    "has_downstream_ejc"]
    available = [c for c in feature_cols if c in attn_merged.columns]

    rows = []
    for label_name, label_val in [("all", None), ("NMD", 1), ("non-NMD", 0)]:
        if label_val is not None:
            subset = attn_merged[attn_merged["label"] == label_val]
        else:
            subset = attn_merged

        for col in available:
            valid = subset[[col, "attn_weight"]].dropna()
            if len(valid) < 20:
                continue
            rho, p = spearmanr(valid[col], valid["attn_weight"])
            rows.append({
                "class": label_name,
                "feature": col,
                "spearman_rho": round(rho, 4),
                "p_value": p,
                "n": len(valid),
            })

    df = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"attention_vs_features{suffix}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")
    print(df[df["class"] == "NMD"].to_string(index=False))
    return df


def analysis_4_noptc_attention(attn_merged, preds, results_dir, tag=""):
    """Attention patterns for NMD isoforms with no downstream EJC (no-PTC cases)."""
    print("\n=== No-PTC NMD isoform attention ===")

    nmd_isoforms = set(preds[preds["label"] == 1]["isoform_id"])

    # For each NMD transcript, check if ANY attended ORF has downstream EJC
    nmd_attn = attn_merged[attn_merged["isoform_id"].isin(nmd_isoforms)].copy()

    # Per-transcript: does the top-attended ORF have downstream EJC?
    if "has_downstream_ejc" not in nmd_attn.columns:
        print("  has_downstream_ejc not available in merged data, skipping")
        return None

    # Group by transcript, find ORF with max attention
    top_orf = nmd_attn.loc[nmd_attn.groupby("isoform_id")["attn_weight"].idxmax()]

    # Check if ANY ORF in top-K has downstream EJC
    any_ejc = nmd_attn.groupby("isoform_id")["has_downstream_ejc"].max()
    top_orf = top_orf.merge(any_ejc.rename("any_orf_has_ejc"),
                            left_on="isoform_id", right_index=True)

    n_total = len(top_orf)
    n_top_has_ejc = (top_orf["has_downstream_ejc"] == 1).sum()
    n_any_has_ejc = (top_orf["any_orf_has_ejc"] == 1).sum()
    n_no_ejc = n_total - n_any_has_ejc

    print(f"  NMD isoforms in test set: {n_total}")
    print(f"  Top-attended ORF has downstream EJC: {n_top_has_ejc} ({100*n_top_has_ejc/n_total:.1f}%)")
    print(f"  Any ORF in top-K has downstream EJC: {n_any_has_ejc} ({100*n_any_has_ejc/n_total:.1f}%)")
    print(f"  No ORF has downstream EJC (no-PTC): {n_no_ejc} ({100*n_no_ejc/n_total:.1f}%)")

    # For no-PTC cases: what does attention look like?
    noptc_ids = set(any_ejc[any_ejc == 0].index)
    if len(noptc_ids) > 0:
        noptc_attn = nmd_attn[nmd_attn["isoform_id"].isin(noptc_ids)]
        noptc_preds = preds[preds["isoform_id"].isin(noptc_ids)]

        print(f"\n  No-PTC NMD isoforms ({len(noptc_ids)}):")
        print(f"    Mean prediction probability: {noptc_preds['prob'].mean():.3f}")
        print(f"    Correctly predicted (prob > 0.5): "
              f"{(noptc_preds['prob'] > 0.5).sum()}/{len(noptc_preds)}")

        # Attention distribution for no-PTC
        noptc_top = noptc_attn.loc[
            noptc_attn.groupby("isoform_id")["attn_weight"].idxmax()]
        print(f"    Top-attended ORF rank distribution: "
              f"{dict(noptc_top['orf_rank'].value_counts().sort_index())}")

        # Feature comparison: no-PTC top ORFs vs PTC top ORFs
        ptc_ids = nmd_isoforms - noptc_ids
        ptc_top = nmd_attn[nmd_attn["isoform_id"].isin(ptc_ids)]
        ptc_top = ptc_top.loc[ptc_top.groupby("isoform_id")["attn_weight"].idxmax()]

        compare_cols = ["orf_length", "kozak_score", "frac_position", "n_upstream_atgs"]
        available = [c for c in compare_cols if c in noptc_top.columns]
        if available:
            print(f"\n    Feature comparison (top-attended ORF):")
            print(f"    {'feature':<20} {'no-PTC mean':>12} {'PTC mean':>12}")
            for col in available:
                v1 = noptc_top[col].mean()
                v2 = ptc_top[col].mean()
                print(f"    {col:<20} {v1:>12.3f} {v2:>12.3f}")

    # Save
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"attention_noptc_analysis{suffix}.tsv"
    top_orf.to_csv(path, sep="\t", index=False)
    print(f"\n  -> {path}")
    return top_orf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", default="atg20_stop500")
    args = parser.parse_args()

    results_dir = args.results_dir
    tag = args.tag

    print(f"Attention analysis for model: {tag}")
    print(f"Results dir: {results_dir}")

    # Load data
    attn, preds = load_attention_and_predictions(results_dir, tag)
    orf_features = load_orf_features_with_rank(results_dir)

    # Merge attention with predictions
    attn = attn.merge(preds[["isoform_id", "label", "prob"]], on="isoform_id")

    # Merge with ORF features (includes CDS indicators and selection_reason)
    orf_features["isoform_id"] = orf_features["isoform_id"].str.strip('"')
    merge_cols = ["isoform_id", "orf_rank", "orf_length", "frac_position",
                  "frac_tx_covered", "kozak_score", "n_upstream_atgs",
                  "n_downstream_ejc", "has_downstream_ejc",
                  "is_ref_cds", "is_sqanti_cds"]
    if "selection_reason" in orf_features.columns:
        merge_cols.append("selection_reason")
    attn_merged = attn.merge(
        orf_features[merge_cols],
        on=["isoform_id", "orf_rank"], how="left")
    attn_merged["is_ref_cds"] = attn_merged["is_ref_cds"].fillna(0).astype(int)
    attn_merged["is_sqanti_cds"] = attn_merged["is_sqanti_cds"].fillna(0).astype(int)

    print(f"\nMerged data: {len(attn_merged):,} ORF-level rows, "
          f"{attn_merged['isoform_id'].nunique():,} transcripts")

    # Run analyses
    analysis_1_attention_by_orf_type(attn_merged, results_dir, tag)
    analysis_2_attention_entropy(attn, preds, results_dir, tag)
    analysis_3_attention_feature_correlation(attn_merged, results_dir, tag)
    analysis_4_noptc_attention(attn_merged, preds, results_dir, tag)

    print("\n=== Attention analysis complete ===")


if __name__ == "__main__":
    main()
