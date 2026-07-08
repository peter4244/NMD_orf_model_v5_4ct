#!/usr/bin/env python3
"""09d_export_gc_content_refaug_only.py — GC content restricted to NMD isoforms
whose priority ORF is the reference CDS (ref-AUG-traced), plus all Controls.

Companion to 09_export_gc_content.py. That script anchors each transcript's
stop window on the *priority* ORF, which for the majority of NMD test
transcripts is the reference CDS ORF (unbiased against PTCs). But a minority
of NMD transcripts fall through to the TD2 fallback, whose CDS calls
systematically avoid PTC-containing ORFs (feedback_sqanti_cds_ptc_bias). Those
TD2-anchored NMD transcripts drag the post-stop GC signal toward Control by
placing "position 0" at a downstream non-PTC stop instead of the true PTC.

This addendum reads the joint DeepSHAP NPZ, uses its structural feature
`is_ref_cds` (ORF0's) to filter the NMD population to ref-AUG-anchored
isoforms only, keeps all Control isoforms, then re-emits the same GC-content
TSVs on the filtered cohort.

Outputs (in --results-dir):
  gc_content_across_atg_window_refaug_only_{tag}.tsv
  gc_content_across_stop_window_refaug_only_{tag}.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


N_CHANNELS = 9
WINDOW = 500
PER_WINDOW = N_CHANNELS * WINDOW
IS_REF_CDS_STRUCT_IDX = 2   # 5-feature block: frac_start, frac_stop, is_ref_cds, is_sqanti_cds, n_downstream_ejc


def load_joint_run(results_dir, tag, run=1):
    """Load one joint DeepSHAP NPZ. The joint file lays out shap_values and inputs
    as (N, 9005) where the first 9*500 cols are the ATG window, the next 9*500
    are the stop window, and the trailing 5 cols are per-ORF structural features
    (for ORF0 = priority ORF)."""
    path = results_dir / f"deepshap_joint_{tag}_run{run}.npz"
    print(f"Loading {path}")
    d = np.load(path, allow_pickle=True)
    inputs = d["inputs"]
    labels = d["labels"]
    channel_names = list(d["channel_names"])
    N = inputs.shape[0]

    atg_inp  = inputs[:, :PER_WINDOW].reshape(N, N_CHANNELS, WINDOW)
    stop_inp = inputs[:, PER_WINDOW:2 * PER_WINDOW].reshape(N, N_CHANNELS, WINDOW)
    struct   = inputs[:, 2 * PER_WINDOW:]     # (N, 5)
    is_ref_cds = struct[:, IS_REF_CDS_STRUCT_IDX] > 0.5

    return dict(atg_inp=atg_inp, stop_inp=stop_inp,
                labels=labels, is_ref_cds=is_ref_cds,
                channel_names=channel_names, N=N)


def emit_gc(bundle, branch, gc_window, gc_step, out_path):
    """Compute rolling GC content per position per class from one branch's
    inputs, restricted to (NMD with is_ref_cds) ∪ (all Controls). Same
    algorithm as 09_export_gc_content but with a subset mask."""
    inp = bundle[f"{branch}_inp"]         # (N, 9, W)
    labels = bundle["labels"]
    is_ref_cds = bundle["is_ref_cds"]
    channel_names = bundle["channel_names"]

    g_idx = channel_names.index("G")
    c_idx = channel_names.index("C")
    gc_per_pos = inp[:, g_idx, :] + inp[:, c_idx, :]   # (N, W), one-hot so 0/1

    codon_pos = WINDOW // 2
    rows = []
    for class_label, mask in [
        ("NMD",     (labels == 1) & is_ref_cds),
        ("Control", (labels == 0)),
    ]:
        gc_class = gc_per_pos[mask]
        n_class = gc_class.shape[0]
        if n_class == 0:
            continue
        for start in range(0, WINDOW - gc_window + 1, gc_step):
            end = start + gc_window
            rel_mid = ((start + end) / 2) - codon_pos
            sample_gc = gc_class[:, start:end].mean(axis=1)
            rows.append({
                "rel_mid":  rel_mid,
                "mean_gc":  float(sample_gc.mean()),
                "se_gc":    float(sample_gc.std() / np.sqrt(n_class)),
                "class":    class_label,
                "n_transcripts": int(n_class),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path.name}  ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="atg500_stop500")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--results-dir", default="results_4ct")
    parser.add_argument("--gc-window", type=int, default=50)
    parser.add_argument("--gc-step", type=int, default=10)
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    bundle = load_joint_run(rdir, args.tag, args.run)

    n_total_nmd = int((bundle["labels"] == 1).sum())
    n_nmd_refaug = int(((bundle["labels"] == 1) & bundle["is_ref_cds"]).sum())
    n_ctrl = int((bundle["labels"] == 0).sum())
    print(f"\nCohort after ref-AUG filter:")
    print(f"  NMD (is_ref_cds=1): {n_nmd_refaug} / {n_total_nmd} "
          f"({100*n_nmd_refaug/max(n_total_nmd,1):.1f}%)")
    print(f"  Control (unfiltered): {n_ctrl}")

    print(f"\nWriting outputs ...")
    emit_gc(bundle, "atg",  args.gc_window, args.gc_step,
            rdir / f"gc_content_across_atg_window_refaug_only_{args.tag}.tsv")
    emit_gc(bundle, "stop", args.gc_window, args.gc_step,
            rdir / f"gc_content_across_stop_window_refaug_only_{args.tag}.tsv")
    print("\nDone.")


if __name__ == "__main__":
    main()
