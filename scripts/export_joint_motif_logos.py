#!/usr/bin/env python3
"""
export_joint_motif_logos.py — Pool joint DeepSHAP NPZs across 5 runs and
produce ATG and stop motif logo TSVs covering the full test set.

The joint NPZs (deepshap_joint_atg{A}_stop{S}_run{N}.npz, N=1..5) each contain
shap_values over a 9005-dim joint vector (9 channels × 500 ATG + 9 channels ×
500 stop + 5 structural features) for all n_test test transcripts.

This script:
  1. Loads the 5 NPZs.
  2. Slices the ATG-window block (channels A,C,G,T, positions 0..499) and the
     stop-window block.
  3. Pools across runs by averaging shap_values per sample-position-channel.
  4. Computes per-position-per-channel mean signed SHAP*input attribution for
     NMD vs Control, plus per-position-per-channel input frequency.
  5. Writes:
       motif_logo_atg_joint_{tag}.tsv
       motif_logo_stop_joint_{tag}.tsv
     in the schema expected by orf_model_report_v5.Rmd:
       relative_position, channel, nmd_mean_contrib, ctrl_mean_contrib,
       nmd_freq, ctrl_freq, n_nmd, n_ctrl

The window's relative_position 0 corresponds to the middle nucleotide of the
codon (T of ATG, or middle of stop codon). Specifically:
  index in window         relative_position
  0                       -250
  ...
  249                       -1
  250                        0  (middle nt of codon)
  251                       +1
  ...
  499                      +249
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# member_tag is IMPORTED, not restated. Its own docstring makes the argument: "a naming rule
# restated at seven call sites is a rule that will be seven different rules within a month." This
# script was effectively an eighth site -- it built the stem inline, seedlessly, and looked for
# deepshap_joint_atg1000_stop1000_run1.npz while deepshap.py had written
# deepshap_joint_atg1000_stop1000_seed42_run1.npz. scripts/ had no import-from-root pattern, so
# this establishes one rather than copying the rule in.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import member_tag  # noqa: E402


def load_run(npz_path):
    d = np.load(npz_path)
    sv = d["shap_values"]            # (n, 9005, 1)
    inp = d["inputs"]                # (n, 9005)
    labels = d["labels"].astype(int) # (n,)
    n_ch = int(d["n_channels"])      # 9
    w_atg = int(d["w_atg"])          # 500
    w_stop = int(d["w_stop"])        # 500
    return sv.squeeze(-1), inp, labels, n_ch, w_atg, w_stop


def slice_window(arr_2d, n_ch, w, offset):
    """arr_2d: (n, 9005). Returns (n, n_ch, w) for the chosen block."""
    block = arr_2d[:, offset:offset + n_ch * w]
    return block.reshape(arr_2d.shape[0], n_ch, w)


def export_logo(shap_block, input_block, labels, channels, out_path,
                relative_offset=-250, channels_keep=("A", "C", "G", "T")):
    """Compute motif-logo TSV for one window block.

    shap_block, input_block: (n_samples, n_channels, w)
    labels: (n_samples,) 0/1
    channels: list of channel names of length n_channels
    Writes columns: relative_position, channel, nmd_mean_contrib,
    ctrl_mean_contrib, nmd_freq, ctrl_freq, n_nmd, n_ctrl.
    """
    n, n_ch, w = shap_block.shape
    contrib = shap_block * input_block       # SHAP × input
    nmd_mask = labels == 1
    ctrl_mask = labels == 0
    n_nmd = int(nmd_mask.sum())
    n_ctrl = int(ctrl_mask.sum())

    rows = []
    for ch_idx, ch_name in enumerate(channels):
        if ch_name not in channels_keep:
            continue
        for pos in range(w):
            rel = relative_offset + pos
            rows.append({
                "relative_position": rel,
                "channel": ch_name,
                "nmd_mean_contrib": float(contrib[nmd_mask, ch_idx, pos].mean()),
                "ctrl_mean_contrib": float(contrib[ctrl_mask, ch_idx, pos].mean()),
                "nmd_freq": float(input_block[nmd_mask, ch_idx, pos].mean()),
                "ctrl_freq": float(input_block[ctrl_mask, ch_idx, pos].mean()),
                "n_nmd": n_nmd,
                "n_ctrl": n_ctrl,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  Wrote {out_path}: {len(df)} rows, n_nmd={n_nmd}, n_ctrl={n_ctrl}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # NO DEFAULT RESULTS TREE. It was "results_4ct" -- the PUBLISHED run -- so a bare call
    # silently read the old universe and exited 0. Every wrapper already passes
    # --results-dir, so requiring it costs nothing and removes the one path that could
    # produce published-tree numbers while looking deposit-native. Same reasoning as
    # deepshap.py's --split: a default of any value is how the wrong population goes invisible.
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--atg", type=int, default=500)
    ap.add_argument("--stop", type=int, default=500)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--member-seed", type=int, default=None,
                    help="Ensemble member, by training seed. Omitted reproduces the legacy "
                         "un-seeded name so already-deposited artifacts still resolve.")
    args = ap.parse_args()

    results = Path(args.results_dir)
    tag = member_tag(f"atg{args.atg}_stop{args.stop}", args.member_seed)

    runs = []
    for r in range(1, args.n_runs + 1):
        p = results / f"deepshap_joint_{tag}_run{r}.npz"
        if not p.exists():
            sys.exit(f"Joint NPZ missing: {p}")
        print(f"Loading {p.name} ...")
        runs.append(load_run(p))

    # Sanity: all runs should have same labels/inputs (same test set)
    sv0, inp0, lab0, n_ch, w_atg, w_stop = runs[0]
    for i, (sv, inp, lab, _, _, _) in enumerate(runs[1:], start=2):
        if not np.array_equal(lab, lab0):
            sys.exit(f"Run {i} labels differ from run 1")
        if sv.shape != sv0.shape:
            sys.exit(f"Run {i} shap_values shape {sv.shape} differs")

    # Pool shap_values: mean across runs, per sample × dim
    print(f"\nPooling shap_values across {args.n_runs} runs ...")
    shap_pooled = np.mean(np.stack([r[0] for r in runs], axis=0), axis=0)
    inputs = inp0  # identical across runs
    labels = lab0
    print(f"  pooled shape: {shap_pooled.shape}")

    # Slice ATG and stop blocks
    atg_off = 0
    stop_off = n_ch * w_atg
    shap_atg = slice_window(shap_pooled, n_ch, w_atg, atg_off)
    inp_atg  = slice_window(inputs,      n_ch, w_atg, atg_off)
    shap_stop = slice_window(shap_pooled, n_ch, w_stop, stop_off)
    inp_stop  = slice_window(inputs,      n_ch, w_stop, stop_off)
    print(f"  atg block: {shap_atg.shape}, stop block: {shap_stop.shape}")

    channels = ['A', 'C', 'G', 'T', 'junction', 'rolling_gc', 'frame_0', 'frame_1', 'frame_2']

    print("\nExporting motif logo TSVs ...")
    export_logo(shap_atg, inp_atg, labels, channels,
                results / f"motif_logo_atg_joint_{tag}.tsv",
                relative_offset=-(w_atg // 2))
    export_logo(shap_stop, inp_stop, labels, channels,
                results / f"motif_logo_stop_joint_{tag}.tsv",
                relative_offset=-(w_stop // 2))

    print("\nDone.")


if __name__ == "__main__":
    main()
