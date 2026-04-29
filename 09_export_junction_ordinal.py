#!/usr/bin/env python3
"""
09_export_junction_ordinal.py — Export per-junction DeepSHAP by ordinal position.

For each NMD sample in the DeepSHAP stop branch, finds all junction positions
downstream of the stop codon, assigns ordinals (EJC 1, EJC 2, ...), and extracts
the junction channel |SHAP| at each position.

Outputs:
  - junction_by_ordinal_{tag}.tsv: per-junction rows (for loess plots)
  - junction_ordinal_summary_{tag}.tsv: mean |SHAP| per ordinal (summary table)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="atg500_stop500")
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    results_dir = Path("results")
    run_tag = f"{args.tag}_run{args.run}"

    # Load stop branch DeepSHAP
    npz_path = results_dir / f"deepshap_stop_{run_tag}.npz"
    print(f"Loading {npz_path}")
    data = np.load(npz_path)
    shap_vals = data["shap_values"]   # (N, 9, W)
    inputs = data["inputs"]           # (N, 9, W)
    labels = data["labels"]           # (N,)
    channel_names = list(data["channel_names"])

    # Squeeze trailing output dimension if present (model outputs (batch, 1))
    if shap_vals.ndim == 4:
        shap_vals = shap_vals.squeeze(-1)
    N, C, W = shap_vals.shape
    print(f"  {N} samples, {C} channels, {W} positions")

    # Find junction channel index
    junc_idx = channel_names.index("junction")
    print(f"  Junction channel index: {junc_idx}")

    # Stop codon is at window center
    stop_pos = W // 2
    print(f"  Stop codon center: position {stop_pos}")

    # Process each sample: find junctions downstream of stop, assign ordinals
    rows = []
    for i in range(N):
        class_label = "NMD" if labels[i] == 1 else "Control"

        # Junction positions: where junction channel > 0.5, downstream of stop
        junc_channel = inputs[i, junc_idx, :]
        junc_positions = np.where((junc_channel > 0.5) & (np.arange(W) > stop_pos))[0]

        if len(junc_positions) == 0:
            continue

        # Sort by position (ascending = ordinal order)
        junc_positions = np.sort(junc_positions)

        for j, pos in enumerate(junc_positions):
            ordinal = j + 1  # 1-indexed
            ejc_label = f"EJC {min(ordinal, 5)}" if ordinal < 5 else "EJC 5+"
            distance = pos - stop_pos

            # Junction channel SHAP at this position
            abs_shap = float(np.abs(shap_vals[i, junc_idx, pos]))
            signed_shap = float(shap_vals[i, junc_idx, pos] * inputs[i, junc_idx, pos])

            rows.append({
                "sample_idx": i,
                "class": class_label,
                "ejc_ordinal": ordinal,
                "ejc_label": ejc_label,
                "position": int(pos),
                "distance_from_stop": int(distance),
                "abs_shap": abs_shap,
                "signed_shap": signed_shap,
            })

    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} junction observations across {df['sample_idx'].nunique()} samples")
    print(f"  NMD junctions: {(df['class'] == 'NMD').sum()}")
    print(f"  Control junctions: {(df['class'] == 'Control').sum()}")

    # Per-ordinal summary
    summary = df.groupby(["class", "ejc_label"]).agg(
        n_junctions=("abs_shap", "count"),
        mean_abs_shap=("abs_shap", "mean"),
        mean_signed_shap=("signed_shap", "mean"),
        mean_distance=("distance_from_stop", "mean"),
    ).reset_index()

    # Save
    raw_path = results_dir / f"junction_by_ordinal_{run_tag}.tsv"
    df.to_csv(raw_path, sep="\t", index=False)
    print(f"  -> {raw_path}")

    summary_path = results_dir / f"junction_ordinal_summary_{run_tag}.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"  -> {summary_path}")

    # Quick sanity check
    nmd_df = df[df["class"] == "NMD"]
    if len(nmd_df) > 0:
        print("\n  NMD junction SHAP by ordinal:")
        for label in ["EJC 1", "EJC 2", "EJC 3", "EJC 4", "EJC 5+"]:
            sub = nmd_df[nmd_df["ejc_label"] == label]
            if len(sub) > 0:
                print(f"    {label}: n={len(sub)}, mean |SHAP|={sub['abs_shap'].mean():.5f}, "
                      f"mean dist={sub['distance_from_stop'].mean():.0f}bp")

    print("\nDone.")


if __name__ == "__main__":
    main()
