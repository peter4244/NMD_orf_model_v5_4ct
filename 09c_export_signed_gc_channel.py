#!/usr/bin/env python3
"""09c_export_signed_gc_channel.py — Signed SHAP × input for the rolling_gc channel.

Companion to 09b_export_subgroup_profiles.py. 09b writes per-position
|SHAP| — i.e., channel importance — which is the wrong metric for SF40:
the paper's SF40 claim ("GC content differentiates NMD from Control in
the STOP but not the START window") is about the model's *directional*
use of GC content, not raw importance.

This script emits signed SHAP × input for the `rolling_gc` channel, per
position, per class (NMD vs Control), pooled across the 5 joint DeepSHAP
runs. Signed SHAP × input is the canonical directional attribution
(positive = pushes the prediction toward NMD, negative = away from NMD).

Outputs (in --results-dir):
  shap_signed_gc_atg_joint_{tag}.tsv
  shap_signed_gc_stop_joint_{tag}.tsv

Schema: relative_position, nmd_signed, ctrl_signed, nmd_n, ctrl_n
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths_config import load_config, selected_tag


def load_joint_runs(results_dir, tag, split, n_runs=5):
    """Load and mean-pool signed SHAP × input for the rolling_gc channel
    across all runs, returning per-window (atg, stop) matrices."""
    # Match the 9005-column layout used by 09b: 9 channels × 500 atg + 9 × 500 stop + 5 structural.
    n_ch = 9
    w = 500
    per_window = n_ch * w

    atg_signed_runs, stop_signed_runs = [], []
    labels = None
    gc_idx = None
    for r in range(1, n_runs + 1):
        path = results_dir / f"deepshap_joint_{tag}_{split}_run{r}.npz"
        d = np.load(path, allow_pickle=True)
        sv = d["shap_values"].squeeze(-1)
        inp = d["inputs"]
        N = sv.shape[0]

        if gc_idx is None:
            channel_names = list(d["channel_names"])
            gc_idx = channel_names.index("rolling_gc")
            labels = d["labels"]

        atg_sv = sv[:, :per_window].reshape(N, n_ch, w)
        stop_sv = sv[:, per_window:2 * per_window].reshape(N, n_ch, w)
        atg_in = inp[:, :per_window].reshape(N, n_ch, w)
        stop_in = inp[:, per_window:2 * per_window].reshape(N, n_ch, w)

        atg_signed_runs.append(atg_sv[:, gc_idx, :] * atg_in[:, gc_idx, :])   # (N, w)
        stop_signed_runs.append(stop_sv[:, gc_idx, :] * stop_in[:, gc_idx, :])
        print(f"  loaded run{r}: {path.name}  N={N}")

    atg_signed = np.mean(atg_signed_runs, axis=0)
    stop_signed = np.mean(stop_signed_runs, axis=0)
    return atg_signed, stop_signed, labels


def write_signed_gc_profile(signed_arr, labels, window, out_path):
    """Per-position mean signed SHAP × input for the GC channel, split by class."""
    rel_pos = np.arange(window) - (window // 2)  # -250..249 for window=500
    nmd_mask = labels > 0.5
    ctrl_mask = ~nmd_mask
    nmd_mean = signed_arr[nmd_mask].mean(axis=0)
    ctrl_mean = signed_arr[ctrl_mask].mean(axis=0)
    nmd_n = int(nmd_mask.sum())
    ctrl_n = int(ctrl_mask.sum())

    out = pd.DataFrame({
        "relative_position": rel_pos.astype(int),
        "nmd_signed":  nmd_mean.astype(float),
        "ctrl_signed": ctrl_mean.astype(float),
        "nmd_n":  nmd_n,
        "ctrl_n": ctrl_n,
    })
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path.name}  ({len(out)} rows, NMD n={nmd_n}, Control n={ctrl_n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                                        help="Window-config tag. Default: the `selected:` block in --config. "
                                             "Never a hardcoded literal -- see utils.selected_tag.")
    ap.add_argument("--config", default="config.yaml",
                       help="Where the selected window configuration is read from")
    ap.add_argument("--results-dir", default="results_4ct")
    ap.add_argument("--n-runs", type=int, default=5)
    # REQUIRED, matching deepshap.py: the split now travels in the artifact name, so a consumer
    # that guesses it reads the wrong population's attributions or nothing at all.
    ap.add_argument("--split", required=True,
                    help="the split deepshap.py explained; part of the artifact name")
    args = ap.parse_args()

    # Resolve the tag from the ONE place that names the selected configuration.
    if args.tag is None:
        args.tag = selected_tag(load_config(args.config))
    rdir = Path(args.results_dir)
    print(f"Loading {args.n_runs} joint DeepSHAP NPZs for tag={args.tag}, "
          f"extracting signed SHAP × input for rolling_gc channel ...")
    atg_signed, stop_signed, labels = load_joint_runs(rdir, args.tag, args.split, args.n_runs)
    print(f"\nWriting outputs ...")
    write_signed_gc_profile(atg_signed,  labels, window=500,
                             out_path=rdir / f"shap_signed_gc_atg_joint_{args.tag}.tsv")
    write_signed_gc_profile(stop_signed, labels, window=500,
                             out_path=rdir / f"shap_signed_gc_stop_joint_{args.tag}.tsv")
    print("\nDone.")


if __name__ == "__main__":
    main()
