#!/usr/bin/env python
"""
test85_matched_pair_initiation.py — §8.5, the pre-registered matched-pair test.

Implements section 8.5 of analysis_plans/RETRAIN_PLAN_2026-08-01.md exactly. The
question: does the capture head prefer the annotated start codon over a competing
start codon in the SAME transcript, once every geometric property of the reading
window is held fixed?

THIS IS CONFIRMATORY. The exploratory version ran on validation this morning
(probe_position_matched_pairs.py, probe_pair_baselines.py) and §8.5 says in terms
that it is not reported: two geometric leaks were found in it, each inflating the
estimate, and the corrected residual did not survive gene-clustered inference at
that sample size. Nothing here is adapted from those scripts. Pointing them at the
test split would be substituting the nearest convenient artifact for the thing
itself, and the pre-registration is the only reason a test-split read is allowed
at all.

--split test reads `test_clean` and is a ONE-SHOT. --split val runs the identical
code on validation and is for shaking the mechanics out; validation is already
spent, so nothing is lost by running it there first.

THE TWO LEAKS THIS DESIGN CONTROLS, both in WHICH POSITIONS OF THE WINDOW ARE
FILLED, neither visible to an ablation over channels:

  5' padding     the ATG window reaches 900 upstream, so its left fill boundary is
                 max(901 - orf_start, 0) -- distance to the 5' end, exactly.
                 Controlled by matching orf_start to +/-50 AND by the direction
                 split, since matching alone still leaves one member upstream.
  midpoint clip  fill stops at the ORF midpoint, so right-hand fill reads out ORF
                 length, and reference coding sequences are long while competing
                 ORFs are usually very short. Controlled by requiring BOTH members
                 to carry the full 100 bases of right-hand fill.

GEOMETRY COMES FROM window_spans, NOT FROM A RESTATEMENT OF IT. §8.5 gives
right-hand fill as `min(100, (orf_length // 2) + 1)`. That is a description of
what build_tensor did; the authority is the function the tensor and the bank
actually call. This script imports it and CHECKS the plan's formula against it,
reporting any disagreement rather than silently preferring one.

ONE PLACE THE SPEC NEEDS A READING, flagged rather than decided quietly. §8.5's
last baseline is "the tabular gradient-boosted model of §8.3 with `is_ref_cds`
withheld, since that column is the label of this test." §8.3's model is a
TRANSCRIPT-level NMD classifier, and every other baseline here is a per-CANDIDATE
score used to pick the winner within a pair, so the §8.3 model cannot be applied
as written. The reading taken is the one that makes the sentence cohere: a
gradient-boosted model trained to predict `is_ref_cds` from the other tabular
columns -- which is why `is_ref_cds` must be withheld as an INPUT, being the label
of this test -- fitted on `train` candidates and scored on the test pairs. It asks
whether the capture head beats a tabular model built expressly to find the
reference start. Run with --no-gbm to produce every other baseline while that
reading is confirmed.

Usage:
    python test85_matched_pair_initiation.py --split val          # mechanics
    python test85_matched_pair_initiation.py --split test         # the one shot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from build_ism_bank import window_spans          # noqa: E402  the geometry authority
from model_v6 import ScanningNMDModel            # noqa: E402
from tensor_io import decode_windows             # noqa: E402

ATG_LEFT, ATG_RIGHT = 900, 100
KOZAK_FLOOR = -1.2508          # §3, the MANE floor both arms clear
CALIPER = 50                   # §8.5, orf_start match in nucleotides
FULL_RIGHT_FILL = ATG_RIGHT    # §8.5, both members must carry all 100


# --------------------------------------------------------------- the statistic
def win_rate(a, b):
    """Fraction of pairs where a > b, ties counted as one half. §8.5 throughout."""
    return float(np.mean((a > b) + 0.5 * (a == b)))


def boot_ci(values, groups, rng, n_boot, statistic):
    """Percentile 95% interval, resampling GROUPS with replacement.

    §8.5 requires the interval to resample genes, not pairs: the pairs of one gene
    are not independent, and the design effect measured on this statistic in the
    exploratory run was 3.15 against 1.71 for the label-level ICC. Passing pair
    indices as their own groups gives the pair-resampled interval, which §8.5 also
    requires so that ratio is visible again on this split.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by_group[p] for p in pick])
        out[b] = statistic(idx)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), out


# --------------------------------------------------------------- p_capture
def capture_scores(codes, o_s, off, cnt, n_cand, which_tx, ckpts, device):
    """p_k of §6.2 step 2, per candidate, one column per seed.

    The initiation head's own output on the ATG window. No aggregation: p_select
    would confound "this is a strong initiation context" with "everything upstream
    of it was weak", and this test is about the former.

    Scored only for the transcripts of `which_tx`. Every other candidate is left
    NaN rather than zero, so a candidate from outside the split that reached a pair
    would propagate a NaN into the statistic and be seen, instead of silently
    contributing a score of nothing.
    """
    out = np.full((n_cand, len(ckpts)), np.nan, dtype=np.float64)
    for c, cp in enumerate(ckpts):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1)
        m.load_state_dict(ck["model"])
        m.to(device).eval()
        with torch.no_grad():
            for i in which_tx:
                lo = int(off[i]); hi = lo + int(cnt[i])
                s0 = o_s[lo:hi].astype(np.int64)
                x = torch.as_tensor(
                    decode_windows(codes[lo:hi][:, 0], s0, ATG_LEFT, s0), device=device)
                out[lo:hi, c] = torch.sigmoid(
                    m.init_head(m.enc_init(x)).squeeze(-1)).cpu().numpy()
        print(f"    {cp.parent.name if cp.name == 'best.pt' else cp.name}"
              f": {int(np.isfinite(out[:, c]).sum()):,} candidates scored", flush=True)
    return out          # (n_cand, n_seed), NaN outside `which_tx`


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="results_tensor_v6")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="test reads test_clean and is a one-shot")
    ap.add_argument("--ckpt-glob", default="results_interp_all/v6_checkpoints/b8_s*.pt")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--device", default="")
    ap.add_argument("--no-gbm", action="store_true",
                    help="skip the withheld-column GBM baseline while its reading "
                         "of §8.5 is being confirmed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print(f"§8.5 matched-pair initiation test — split={args.split}  device={device}")
    if args.split == "test":
        print("  READING test_clean. Pre-registered in RETRAIN_PLAN_2026-08-01.md §8.5.")

    # ------------------------------------------------------------- the data
    with h5py.File(REPO / args.tensor / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        gene = np.array([s.decode() for s in f["gene_id"][:]])
        split = np.array([s.decode() for s in f["split"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:], f["orf_end"][:]
        keep_tx = np.flatnonzero(split == args.split)
        codes = f["codes"][:]

    print(f"  {args.split}_clean: {len(keep_tx):,} transcripts of {len(iso):,}")

    # candidate-level frames, in the tensor's own candidate order
    tx_of = np.concatenate([np.full(int(cnt[i]), i) for i in range(len(iso))])
    start = o_s.astype(np.int64)
    end = o_e.astype(np.int64)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "orf_start", "orf_end",
                                "orf_length", "n_downstream_ejc", "kozak_score",
                                "is_ref_cds", "is_sqanti_cds", "frac_start",
                                "frac_stop"])
    key = {s: i for i, s in enumerate(iso)}
    pool = pool[pool.isoform_id.isin(key)].copy()
    pool["tx"] = pool.isoform_id.map(key)
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)

    # THE POOL AND THE TENSOR MUST BE THE SAME CANDIDATES IN THE SAME ORDER. The
    # pool supplies kozak_score and is_ref_cds and the tensor supplies the windows,
    # and they are joined by position. A silent misalignment would attach one
    # candidate's Kozak score to another candidate's window and every number below
    # would be wrong in a way nothing downstream could detect.
    assert len(pool) == len(start), f"pool {len(pool):,} vs tensor {len(start):,}"
    assert np.array_equal(pool.orf_start.to_numpy().astype(np.int64), start), \
        "pool and tensor disagree on orf_start after sorting by (tx, slot)"
    assert np.array_equal(pool.orf_end.to_numpy().astype(np.int64), end), \
        "pool and tensor disagree on orf_end after sorting by (tx, slot)"
    print("  pool and tensor agree candidate-for-candidate on both ORF bounds")

    # ------------------------------------------------------------- geometry
    # tx_length is needed by window_spans; the ATG window's right edge is clipped
    # by it as well as by the midpoint, which the plan's formula does not mention.
    # It is READ, not derived. Taking it as the largest orf_end would be the
    # nearest convenient artifact rather than the transcript's length, and it is
    # wrong by construction for every transcript whose 3'-most ORF ends before the
    # transcript does -- which is most of them.
    tx_tab = pd.read_csv(REPO / "results_ism_v6" / "discovery_confirmation_split.tsv",
                         sep="\t", usecols=["isoform_id", "tx_length"])
    tx_map = dict(zip(tx_tab.isoform_id, tx_tab.tx_length))
    missing_len = [s for s in iso if s not in tx_map]
    if missing_len:
        sys.exit(f"FATAL: {len(missing_len):,} transcripts have no tx_length "
                 f"(first: {missing_len[0]})")
    tx_len = np.array([tx_map[s] for s in iso], dtype=np.int64)
    a_lo, a_hi, _, _ = window_spans(start, end, tx_len[tx_of])
    right_fill = np.maximum(0, a_hi - start + 1)

    plan_formula = np.minimum(ATG_RIGHT, (pool.orf_length.to_numpy() // 2) + 1)
    disagree = int((plan_formula != right_fill).sum())
    print(f"  right-hand fill from window_spans; §8.5's stated formula disagrees "
          f"on {disagree:,} of {len(right_fill):,} candidates "
          f"({100*disagree/len(right_fill):.2f}%)")
    if disagree:
        print("    window_spans is the authority — it is what built the tensor. The")
        print("    formula omits the transcript-length clip, so it overstates fill")
        print("    for candidates near the 3' end.")

    # ------------------------------------------------------------- p_capture
    ckpts = sorted((REPO).glob(args.ckpt_glob))
    if not ckpts:
        sys.exit(f"FATAL: no checkpoints matched {args.ckpt_glob}")
    print(f"  {len(ckpts)} checkpoints:")
    cap_seeds = capture_scores(codes, o_s, off, cnt, len(start), keep_tx,
                               ckpts, device)
    cap = cap_seeds.mean(1)

    # ------------------------------------------------------------- the pairs
    in_split = np.isin(tx_of, keep_tx)
    is_ref = pool.is_ref_cds.to_numpy() == 1
    kozak = pool.kozak_score.to_numpy()
    eligible = in_split & (kozak >= KOZAK_FLOOR) & (right_fill >= FULL_RIGHT_FILL)

    idx = np.flatnonzero(eligible)
    ref_i = idx[is_ref[idx]]
    order = np.argsort(tx_of[idx], kind="stable")
    idx_sorted = idx[order]
    bounds = np.searchsorted(tx_of[idx_sorted], np.arange(len(iso) + 1))

    pairs_r, pairs_c = [], []
    for i in ref_i:
        t = tx_of[i]
        sib = idx_sorted[bounds[t]:bounds[t + 1]]
        sib = sib[(~is_ref[sib]) & (np.abs(start[sib] - start[i]) <= CALIPER)]
        pairs_r.append(np.full(len(sib), i))
        pairs_c.append(sib)
    if not pairs_r:
        sys.exit("no pairs")
    R = np.concatenate(pairs_r)
    C = np.concatenate(pairs_c)
    # Nothing outside the split may reach a pair. Both members are drawn from
    # `eligible`, which is masked by `in_split`, so this holds by construction --
    # and it is asserted because a NaN here would otherwise travel silently into
    # a win rate as a comparison that is neither true nor false.
    assert np.isfinite(cap[R]).all() and np.isfinite(cap[C]).all(), \
        "a pair member has no capture score: a candidate outside the split got in"
    pair_gene = gene[tx_of[R]]
    n_genes = len(np.unique(pair_gene))
    print(f"\n  pairs {len(R):,}   genes {n_genes:,}   "
          f"pairs per gene {len(R)/max(n_genes,1):.2f}   "
          f"reference candidates {len(np.unique(R)):,}")

    # ------------------------------------------------------------- the result
    def stat_on(a, b):
        return lambda ix: win_rate(a[ix], b[ix])

    results = {}
    primary = win_rate(cap[R], cap[C])
    lo_g, hi_g, _ = boot_ci(None, pair_gene, rng, args.n_boot,
                            stat_on(cap[R], cap[C]))
    lo_p, hi_p, _ = boot_ci(None, np.arange(len(R)), rng, args.n_boot,
                            stat_on(cap[R], cap[C]))
    width_g, width_p = hi_g - lo_g, hi_p - lo_p
    print(f"\n=== PRIMARY: p_capture prefers the annotated start ===")
    print(f"  win rate                     {primary:.4f}")
    print(f"  95% CI, genes resampled      [{lo_g:.4f}, {hi_g:.4f}]   "
          f"{'EXCLUDES' if (lo_g > 0.5 or hi_g < 0.5) else 'INCLUDES'} 0.5")
    print(f"  95% CI, pairs resampled      [{lo_p:.4f}, {hi_p:.4f}]")
    print(f"  design effect (width ratio)^2 {(width_g/max(width_p,1e-12))**2:.2f}")
    print(f"  per seed: " + "  ".join(
        f"{win_rate(cap_seeds[R, s], cap_seeds[C, s]):.4f}"
        for s in range(cap_seeds.shape[1])))
    supported = lo_g > 0.5 or hi_g < 0.5
    print(f"  THE CLAIM IS {'SUPPORTED' if supported else 'NOT SUPPORTED'} "
          f"— §8.5 requires the gene-clustered interval to exclude 0.5")
    results["primary"] = dict(win_rate=primary, ci_gene=[lo_g, hi_g],
                              ci_pair=[lo_p, hi_p], supported=bool(supported),
                              n_pairs=int(len(R)), n_genes=int(n_genes))

    # ------------------------------------------- pre-specified direction split
    up = start[R] < start[C]        # the reference is the upstream member
    print(f"\n=== PRE-SPECIFIED SECONDARY: the direction split ===")
    print("  A monotone positional preference must push one of these below 0.5.")
    for name, mask in (("reference UPSTREAM", up), ("reference DOWNSTREAM", ~up)):
        if mask.sum() == 0:
            print(f"  {name:<24} no pairs")
            continue
        w = win_rate(cap[R][mask], cap[C][mask])
        print(f"  {name:<24} {w:.4f}   n {int(mask.sum()):,}")
        results.setdefault("direction", {})[name] = dict(win_rate=w,
                                                         n=int(mask.sum()))
    both_above = all(win_rate(cap[R][m], cap[C][m]) > 0.5
                     for m in (up, ~up) if m.sum())
    if both_above:
        print("  BOTH above 0.5 — evidence no positional account can produce.")

    # ------------------------------------------------- pre-specified baselines
    print(f"\n=== PRE-SPECIFIED BASELINES, on the identical pairs ===")
    baselines = [
        ("orf_length", pool.orf_length.to_numpy().astype(float)),
        ("right-hand window fill", right_fill.astype(float)),
        ("kozak_score", kozak),
        ("5' proximity (-orf_start)", -start.astype(float)),
        ("n_downstream_ejc", pool.n_downstream_ejc.to_numpy().astype(float)),
    ]
    for name, v in baselines:
        w = win_rate(v[R], v[C])
        print(f"  {name:<28} {w:.4f}")
        results.setdefault("baselines", {})[name] = w
    if not args.no_gbm:
        print("  tabular GBM, is_ref_cds withheld: see --no-gbm note in the header;")
        print("    the reading of §8.5 this implements is stated there and is")
        print("    pending confirmation before it is run.")

    print(f"\n  {time.time()-t0:.0f}s")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
