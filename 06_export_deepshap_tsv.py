#!/usr/bin/env python3
"""
06_export_deepshap_tsv.py — Export DeepSHAP NPZ arrays to TSV for R consumption.

Processes deepshap_{atg,stop}_{tag}[_runN].npz files and exports:
  - Positional SHAP summaries (mean |SHAP| and signed SHAP×input by position/channel)
  - Per-sample SHAP×input aggregates for clustering
  - Kozak context table
  - Stop codon composition table
  - Regional SHAP bins

This avoids the need for reticulate in the RMarkdown report.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_npz(path):
    """Load NPZ and squeeze trailing singleton dim from shap_values."""
    data = np.load(path)
    shap = data["shap_values"]
    if shap.ndim == 4 and shap.shape[-1] == 1:
        shap = shap.squeeze(-1)
    return shap, data["inputs"], data["labels"], data["channel_names"]


def export_atg_positional(shap, inputs, labels, channels, out_dir, tag):
    """Positional nucleotide SHAP for ATG branch."""
    nmd = labels == 1
    ctrl = labels == 0
    nucs = ["A", "C", "G", "T"]
    W = shap.shape[2]

    # ATG is at window center
    atg_pos = W // 2

    rows = []
    for pos in range(W):
        rel = pos - atg_pos
        for ch_idx, ch in enumerate(nucs):
            nmd_abs = np.mean(np.abs(shap[nmd, ch_idx, pos]))
            ctrl_abs = np.mean(np.abs(shap[ctrl, ch_idx, pos]))
            nmd_contrib = np.mean(shap[nmd, ch_idx, pos] * inputs[nmd, ch_idx, pos])
            ctrl_contrib = np.mean(shap[ctrl, ch_idx, pos] * inputs[ctrl, ch_idx, pos])
            nmd_freq = np.mean(inputs[nmd, ch_idx, pos])
            ctrl_freq = np.mean(inputs[ctrl, ch_idx, pos])
            rows.append({
                "position": pos, "relative_position": rel,
                "channel": ch,
                "nmd_abs_shap": nmd_abs, "ctrl_abs_shap": ctrl_abs,
                "nmd_signed_contrib": nmd_contrib, "ctrl_signed_contrib": ctrl_contrib,
                "nmd_freq": nmd_freq, "ctrl_freq": ctrl_freq,
            })

    df = pd.DataFrame(rows)
    path = out_dir / f"deepshap_atg_positional_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} rows)")
    return atg_pos


def export_stop_regional(shap, inputs, labels, channels, out_dir, tag):
    """Regional nucleotide SHAP for STOP branch in 25bp bins."""
    nmd = labels == 1
    ctrl = labels == 0
    W = shap.shape[2]

    # Stop codon is at window center
    stop_pos = W // 2

    rows = []
    bin_size = 25
    for start in range(0, W, bin_size):
        end = min(start + bin_size, W)
        rel_start = start - stop_pos
        rel_end = end - stop_pos
        for ch_idx, ch in enumerate(list(channels)[:5]):  # A,C,G,T,junction
            nmd_abs = np.mean(np.abs(shap[nmd, ch_idx, start:end]))
            ctrl_abs = np.mean(np.abs(shap[ctrl, ch_idx, start:end]))
            nmd_signed = np.mean(shap[nmd, ch_idx, start:end] * inputs[nmd, ch_idx, start:end])
            ctrl_signed = np.mean(shap[ctrl, ch_idx, start:end] * inputs[ctrl, ch_idx, start:end])
            rows.append({
                "bin_start": start, "bin_end": end,
                "rel_start": rel_start, "rel_end": rel_end,
                "channel": ch,
                "nmd_abs_shap": nmd_abs, "ctrl_abs_shap": ctrl_abs,
                "nmd_signed_contrib": nmd_signed, "ctrl_signed_contrib": ctrl_signed,
            })

    df = pd.DataFrame(rows)
    path = out_dir / f"deepshap_stop_regional_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} rows)")
    return stop_pos


def export_stop_positional_near(shap, inputs, labels, channels, stop_pos,
                                 out_dir, tag, window=50):
    """Per-position nucleotide SHAP near the stop codon."""
    nmd = labels == 1
    ctrl = labels == 0
    nucs = ["A", "C", "G", "T"]

    start = max(0, stop_pos - window)
    end = min(shap.shape[2], stop_pos + window)

    rows = []
    for pos in range(start, end):
        rel = pos - stop_pos
        for ch_idx, ch in enumerate(nucs):
            nmd_abs = np.mean(np.abs(shap[nmd, ch_idx, pos]))
            ctrl_abs = np.mean(np.abs(shap[ctrl, ch_idx, pos]))
            nmd_contrib = np.mean(shap[nmd, ch_idx, pos] * inputs[nmd, ch_idx, pos])
            nmd_freq = np.mean(inputs[nmd, ch_idx, pos])
            ctrl_freq = np.mean(inputs[ctrl, ch_idx, pos])
            rows.append({
                "position": pos, "relative_position": rel,
                "channel": ch,
                "nmd_abs_shap": nmd_abs, "ctrl_abs_shap": ctrl_abs,
                "nmd_signed_contrib": nmd_contrib,
                "nmd_freq": nmd_freq, "ctrl_freq": ctrl_freq,
            })

    df = pd.DataFrame(rows)
    path = out_dir / f"deepshap_stop_near_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} rows)")


def export_stop_codon_identity(inputs, labels, channels, stop_pos, out_dir, tag):
    """Stop codon type (TGA vs TAA) frequency and composition."""
    nmd = labels == 1
    ctrl = labels == 0
    nucs = ["A", "C", "G", "T"]

    rows = []
    # Stop codon is at window center (stop_pos)
    for offset in range(-3, 4):
        pos = stop_pos + offset
        if pos < 0 or pos >= inputs.shape[2]:
            continue
        for ch_idx, ch in enumerate(nucs):
            rows.append({
                "offset_from_stop": offset,
                "channel": ch,
                "nmd_freq": float(np.mean(inputs[nmd, ch_idx, pos])),
                "ctrl_freq": float(np.mean(inputs[ctrl, ch_idx, pos])),
            })

    df = pd.DataFrame(rows)
    path = out_dir / f"deepshap_stop_codon_context_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} rows)")


def export_junction_shap(shap, inputs, labels, channels, out_dir, tag):
    """Junction channel SHAP across the stop window."""
    nmd = labels == 1
    ctrl = labels == 0
    junc_idx = list(channels).index("junction")
    W = shap.shape[2]

    rows = []
    for pos in range(W):
        rows.append({
            "position": pos,
            "nmd_freq": float(np.mean(inputs[nmd, junc_idx, pos])),
            "ctrl_freq": float(np.mean(inputs[ctrl, junc_idx, pos])),
            "nmd_abs_shap": float(np.mean(np.abs(shap[nmd, junc_idx, pos]))),
            "ctrl_abs_shap": float(np.mean(np.abs(shap[ctrl, junc_idx, pos]))),
            "nmd_signed_shap": float(np.mean(shap[nmd, junc_idx, pos] * inputs[nmd, junc_idx, pos])),
        })

    df = pd.DataFrame(rows)
    path = out_dir / f"deepshap_junction_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} rows)")


def export_gc_by_cds_status(inputs, labels, channels, stop_pos, out_dir, tag,
                             selected_orfs_path=None, predictions_path=None):
    """GC content downstream of stop, stratified by is_ref_cds of rank-0 ORF.

    Requires selected_orfs.tsv and predictions TSV to identify which DeepSHAP
    samples have ref_CDS at rank 0.
    """
    if selected_orfs_path is None or predictions_path is None:
        print("  Skipping GC-by-CDS (no selected_orfs or predictions path)")
        return

    # Load explain_indices to map DeepSHAP samples back to test set
    # We need to match DeepSHAP samples to their isoform IDs
    # For now, compute overall GC stats without CDS stratification
    nmd = labels == 1
    ctrl = labels == 0
    W = inputs.shape[2]

    # GC = C + G channels
    gc = inputs[:, 1, :] + inputs[:, 2, :]  # C + G

    # Downstream of stop codon
    if stop_pos < W:
        gc_downstream = gc[:, stop_pos:]
        rows = []
        for name, mask in [("NMD", nmd), ("Control", ctrl)]:
            vals = gc_downstream[mask].mean(axis=1)
            rows.append({
                "class": name,
                "mean_gc": float(vals.mean()),
                "std_gc": float(vals.std()),
                "median_gc": float(np.median(vals)),
                "n": int(mask.sum()),
            })
        df = pd.DataFrame(rows)
        path = out_dir / f"deepshap_gc_downstream_{tag}.tsv"
        df.to_csv(path, sep="\t", index=False)
        print(f"  -> {path}")


def process_replicates(results_dir, tag):
    """Check for replicate runs and compute stability statistics."""
    run_files = sorted(results_dir.glob(f"deepshap_summary_{tag}_run*.tsv"))
    if not run_files:
        return

    print(f"\n=== Replicate stability ({len(run_files)} runs) ===")
    dfs = []
    for f in run_files:
        df = pd.read_csv(f, sep="\t")
        run_id = f.stem.split("_run")[-1]
        df["run_id"] = int(run_id)
        dfs.append(df)

    combined = pd.concat(dfs)

    # Compute mean and CV across runs per (branch, channel)
    stability = combined.groupby(["branch", "channel"]).agg(
        mean_abs_shap_mean=("mean_abs_shap", "mean"),
        mean_abs_shap_std=("mean_abs_shap", "std"),
        mean_abs_shap_nmd_mean=("mean_abs_shap_nmd", "mean"),
        mean_abs_shap_nmd_std=("mean_abs_shap_nmd", "std"),
        n_runs=("run_id", "nunique"),
    ).reset_index()

    stability["cv_all"] = stability["mean_abs_shap_std"] / stability["mean_abs_shap_mean"]
    stability["cv_nmd"] = stability["mean_abs_shap_nmd_std"] / stability["mean_abs_shap_nmd_mean"]

    path = results_dir / f"deepshap_stability_{tag}.tsv"
    stability.to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")

    for _, row in stability.iterrows():
        print(f"  {row['branch']:>4} {row['channel']:<12} "
              f"mean={row['mean_abs_shap_nmd_mean']:.6f} "
              f"std={row['mean_abs_shap_nmd_std']:.6f} "
              f"CV={row['cv_nmd']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="atg100_stop500")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", type=int, default=None,
                        help="If set, process a specific replicate run")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tag = args.tag
    run_suffix = f"_run{args.run_id}" if args.run_id is not None else ""
    file_tag = f"{tag}{run_suffix}"

    atg_path = results_dir / f"deepshap_atg_{file_tag}.npz"
    stop_path = results_dir / f"deepshap_stop_{file_tag}.npz"

    if not atg_path.exists() or not stop_path.exists():
        print(f"NPZ files not found for tag={file_tag}")
        print(f"  Checked: {atg_path}")
        print(f"  Checked: {stop_path}")
        return

    print(f"Exporting DeepSHAP TSVs for {file_tag}")

    # ATG branch
    print("\n--- ATG branch ---")
    atg_shap, atg_inp, labels, channels = load_npz(atg_path)
    nmd_n = (labels == 1).sum()
    ctrl_n = (labels == 0).sum()
    print(f"  Samples: {len(labels)} (NMD={nmd_n}, ctrl={ctrl_n})")
    print(f"  Shape: {atg_shap.shape}")

    atg_pos = export_atg_positional(atg_shap, atg_inp, labels, channels,
                                     results_dir, file_tag)
    print(f"  ATG marker at position {atg_pos}")

    # STOP branch
    print("\n--- STOP branch ---")
    stop_shap, stop_inp, labels_s, channels_s = load_npz(stop_path)
    print(f"  Shape: {stop_shap.shape}")

    stop_pos = export_stop_regional(stop_shap, stop_inp, labels_s, channels_s,
                                     results_dir, file_tag)
    print(f"  Stop codon at position {stop_pos}")

    export_stop_positional_near(stop_shap, stop_inp, labels_s, channels_s,
                                 stop_pos, results_dir, file_tag)
    export_stop_codon_identity(stop_inp, labels_s, channels_s, stop_pos,
                                results_dir, file_tag)
    export_junction_shap(stop_shap, stop_inp, labels_s, channels_s,
                          results_dir, file_tag)
    export_gc_by_cds_status(stop_inp, labels_s, channels_s, stop_pos,
                             results_dir, file_tag)

    # Process replicates if available
    process_replicates(results_dir, tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
