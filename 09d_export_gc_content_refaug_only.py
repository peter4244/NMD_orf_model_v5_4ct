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

from paths_config import load_config, selected_tag


N_CHANNELS = 9
N_STRUCT = 5
IS_REF_CDS_STRUCT_IDX = 2   # 5-feature block: frac_start, frac_stop, is_ref_cds, is_sqanti_cds, n_downstream_ejc

# WINDOW IS DERIVED FROM THE ARRAY, NEVER TYPED (fixed 2026-08-06).
#
# It was `WINDOW = 500`, and on the 1000nt model that silently produced GARBAGE rather than an
# error. The joint array is (N, 9*W + 9*W + 5), so at W=1000 it is (N, 18005) while PER_WINDOW
# still said 4500. Three things then went wrong at once, none of them raising:
#
#   * `inputs[:, :4500].reshape(N, 9, 500)` takes the first half of the ATG window and reshapes it
#     against the wrong stride. The real layout is channel-major -- cols 0:1000 are channel A's
#     1000 positions -- so row 1 of the result is A[500:1000] and row 2 is C[0:500]. The G and C
#     channel indices then address two DIFFERENT channels at different offsets.
#   * G+C is therefore the sum of two independent one-hot values, which is 0, 1 or 2. That is
#     precisely how mean_gc came out at 1.066 in
#     gc_content_across_stop_window_refaug_only_atg1000_stop1000_seed42.tsv (11 of 92 rows > 1.0).
#   * `struct = inputs[:, 9000:]` picked up 9,005 columns of window data instead of the 5
#     structural features, so `is_ref_cds` -- the entire point of this script -- was noise.
#
# The ATG output looked plausible (0.306-0.548) and was equally wrong; only the stop branch
# happened to land outside [0,1] and give the game away. Deriving W removes the whole class.
def _derive_window(n_cols):
    w, rem = divmod(n_cols - N_STRUCT, 2 * N_CHANNELS)
    if rem or w <= 0:
        raise SystemExit(
            f"cannot derive the window size: {n_cols} columns is not 2*{N_CHANNELS}*W + {N_STRUCT} "
            f"for any integer W. Refusing to reshape on a guess -- a wrong stride here does not "
            f"raise, it returns scrambled channels.")
    return w


def load_joint_run(results_dir, tag, run=1):
    """Load one joint DeepSHAP NPZ. The joint file lays out shap_values and inputs as
    (N, 9*W + 9*W + 5): the ATG window, then the stop window, then the trailing 5 per-ORF
    structural features (for ORF0 = priority ORF). W is read off the array, not assumed."""
    path = results_dir / f"deepshap_joint_{tag}_run{run}.npz"
    print(f"Loading {path}")
    d = np.load(path, allow_pickle=True)
    inputs = d["inputs"]
    labels = d["labels"]
    channel_names = list(d["channel_names"])
    N = inputs.shape[0]

    window = _derive_window(inputs.shape[1])
    per_window = N_CHANNELS * window
    print(f"  derived window = {window} nt ({inputs.shape[1]} cols, {N:,} rows)")

    atg_inp  = inputs[:, :per_window].reshape(N, N_CHANNELS, window)
    stop_inp = inputs[:, per_window:2 * per_window].reshape(N, N_CHANNELS, window)
    struct   = inputs[:, 2 * per_window:]     # (N, 5)
    if struct.shape[1] != N_STRUCT:
        raise SystemExit(f"structural block is {struct.shape[1]} wide, expected {N_STRUCT}")
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
    # The one-hot claim above is an ASSUMPTION, and when it broke it broke silently -- a
    # mis-strided reshape made this the sum of two different channels, giving values of 2.
    # Assert it here, where it is cheap, rather than discovering it as mean_gc = 1.066.
    if gc_per_pos.max() > 1.0 + 1e-6:
        raise SystemExit(
            f"{branch}: G+C reaches {float(gc_per_pos.max()):.3f} at some position, so the "
            f"channels are not one-hot as assumed -- almost always a wrong window/stride.")

    # W from the array, not a module constant. See _derive_window.
    window = inp.shape[2]
    codon_pos = window // 2
    rows = []
    for class_label, mask in [
        ("NMD",     (labels == 1) & is_ref_cds),
        ("Control", (labels == 0)),
    ]:
        gc_class = gc_per_pos[mask]
        n_class = gc_class.shape[0]
        if n_class == 0:
            continue
        for start in range(0, window - gc_window + 1, gc_step):
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
    parser.add_argument("--tag", default=None,
                                            help="Window-config tag. Default: the `selected:` block in --config. "
                                                 "Never a hardcoded literal -- see utils.selected_tag.")
    parser.add_argument("--config", required=True,
                           help="Where the selected window configuration is read from")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--gc-window", type=int, default=50)
    parser.add_argument("--gc-step", type=int, default=10)
    args = parser.parse_args()

    # Resolve the tag from the ONE place that names the selected configuration.
    if args.tag is None:
        args.tag = selected_tag(load_config(args.config))
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
