#!/usr/bin/env python3
"""
09_export_gc_content.py — Export GC content across the AUG or stop window.

Computes sliding-window GC content from a branch's one-hot nucleotide
channels for NMD and Control samples. Emits
gc_content_across_{branch}_window_{tag}.tsv, where {branch} is `atg`
or `stop` (defaults to `stop` for backwards compatibility with the
Rmd report).

Columns: rel_mid, mean_gc, se_gc, class
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths_config import load_config, selected_tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_4ct",
                        help="results tree to read and write "
                             "(results_4ct_dn for the deposit-native rebuild)")
    parser.add_argument("--tag", default=None,
                                            help="Window-config tag. Default: the `selected:` block in --config. "
                                                 "Never a hardcoded literal -- see utils.selected_tag.")
    parser.add_argument("--config", default="config.yaml",
                           help="Where the selected window configuration is read from")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--branch", choices=("atg", "stop"), default="stop",
                        help="Which branch NPZ to read from (default: stop, "
                             "for backwards compatibility with the original "
                             "gc_content_across_stop_window_*.tsv path).")
    parser.add_argument("--gc-window", type=int, default=50,
                        help="Sliding window size for GC computation")
    parser.add_argument("--gc-step", type=int, default=10,
                        help="Step size for sliding window")
    args = parser.parse_args()

    # Resolve the tag from the ONE place that names the selected configuration.
    if args.tag is None:
        args.tag = selected_tag(load_config(args.config))
    # --results-dir, matching 03_train / evaluate / 11_kernel_shap_branches / deepshap /
    # 10_export. Hardcoded, this silently measures the PUBLISHED run when pointed at the
    # deposit-native rebuild -- the reader sees a clean exit and numbers from the wrong tree.
    results_dir = Path(args.results_dir)
    run_tag = f"{args.tag}_run{args.run}"
    branch = args.branch

    # Load requested branch's inputs
    npz_path = results_dir / f"deepshap_{branch}_{run_tag}.npz"
    print(f"Loading {npz_path}")
    data = np.load(npz_path)
    inputs = data["inputs"]   # (N, 9, W)
    labels = data["labels"]   # (N,)
    channel_names = list(data["channel_names"])

    N, C, W = inputs.shape
    codon_pos = W // 2

    # G and C channels
    g_idx = channel_names.index("G")
    c_idx = channel_names.index("C")

    # Per-sample GC at each position: G + C (since one-hot, this is 0 or 1)
    gc_per_pos = inputs[:, g_idx, :] + inputs[:, c_idx, :]  # (N, W)

    # Sliding window
    win = args.gc_window
    step = args.gc_step
    rows = []

    for class_label, mask in [("NMD", labels == 1), ("Control", labels == 0)]:
        gc_class = gc_per_pos[mask]  # (n_class, W)
        n_class = gc_class.shape[0]
        if n_class == 0:
            continue

        for start in range(0, W - win + 1, step):
            end = start + win
            rel_mid = ((start + end) / 2) - codon_pos

            # Mean GC in this window for each sample, then aggregate
            sample_gc = gc_class[:, start:end].mean(axis=1)  # (n_class,)
            mean_gc = float(sample_gc.mean())
            se_gc = float(sample_gc.std() / np.sqrt(n_class))

            rows.append({
                "rel_mid": rel_mid,
                "mean_gc": mean_gc,
                "se_gc": se_gc,
                "class": class_label,
            })

    df = pd.DataFrame(rows)

    # Save with the tag format expected by the report (no _run suffix)
    out_path = results_dir / f"gc_content_across_{branch}_window_{args.tag}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path} ({len(df)} rows)")

    # Sanity check
    for cls in ["NMD", "Control"]:
        sub = df[df["class"] == cls]
        print(f"  {cls}: {len(sub)} bins, GC range {sub['mean_gc'].min():.3f}-{sub['mean_gc'].max():.3f}")

    print("Done.")


if __name__ == "__main__":
    main()
