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

GEOMETRY COMES FROM window_spans, NOT FROM A RESTATEMENT OF IT. The authority is
the function the tensor and the bank actually call, and §8.5 no longer restates it
(amended 2026-08-01). The plan's former wording, read with the pool's INCLUSIVE
`orf_length`, overstated right-hand fill by one on every odd span under the cap --
36% of candidates -- which would have admitted candidates whose true fill is 99 as
though it were 100, in the one place the midpoint-clip control is enforced. Both
readings are still computed here and the span reading disagreeing at all is fatal.

WHAT p_capture READS, since it bounds what this test can conclude. enc_init sees
the ATG window ONLY -- 900 upstream of orf_start and up to 100 into the ORF, "up
to" because fill stops at the ORF midpoint. It does not see the stop window
(asserted in model_v6.py's own checks) and it cannot see ORF length. That is why
§8.5 as amended compares capture to kozak_score, which has strictly less context,
and reports orf_length and 5' proximity as bounding context rather than as
baselines: those use information the model was denied by design.

THE TABULAR MODEL, reading confirmed by the interpretability window. §8.3's model
is TRANSCRIPT-level and cannot pick a within-pair winner, so the clause "with
`is_ref_cds` withheld, since that column is the label of this test" only coheres
if the baseline PREDICTS is_ref_cds from the other tabular columns, fitted on
`train` and scored on the test pairs. Run twice, with and without `is_sqanti_cds`:
that column fingers the same candidate about two-thirds of the time, so with it
the number is a ceiling on any tabular model and without it an honest baseline.

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

    # §8.5 states right-hand fill as `min(100, (orf_length // 2) + 1)`. Read with
    # the pool's `orf_length`, which is INCLUSIVE (orf_end - orf_start + 1,
    # build_orf_pool.py:22), that overstates fill by exactly one whenever the span
    # is odd and the value is under the cap. Read with the span it is exact.
    #
    # The transcript-length clip in window_spans never binds -- orf_end <= tx_len
    # for all 796,584 candidates -- so it is not the cause, and an earlier version
    # of this script said it was. Both readings are printed so the discrepancy is
    # attributed rather than merely counted.
    span = end - start
    fill_span = np.minimum(ATG_RIGHT, (span // 2) + 1)
    fill_inclusive = np.minimum(ATG_RIGHT, (pool.orf_length.to_numpy() // 2) + 1)
    d_span = int((fill_span != right_fill).sum())
    d_incl = int((fill_inclusive != right_fill).sum())
    print(f"  right-hand fill: window_spans is the authority (it built the tensor)")
    print(f"    §8.5's formula on the SPAN            : {d_span:,} disagreements")
    print(f"    §8.5's formula on `orf_length`        : {d_incl:,} disagreements "
          f"({100*d_incl/len(right_fill):.2f}%)")
    if d_span:
        sys.exit("FATAL: window_spans disagrees with the span formula — the "
                 "geometry is not what §8.5 describes at all, and this test's "
                 "midpoint-clip control cannot be trusted until that is resolved")
    if d_incl:
        print("    -> the gap is the inclusive/exclusive length, not the 3' clip:")
        print("       every disagreement is span-odd and larger by exactly one, so")
        print("       the plan's reading would admit candidates whose true fill is")
        print("       99 as though it were 100. This code uses window_spans and")
        print("       does not.")

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
    # §8.5 as AMENDED 2026-08-01 (before the test read): the claim needs capture to
    # beat chance AND to beat kozak_score on the identical pairs, both gene-clustered.
    # Clearing 0.5 alone does not answer "compared to what".
    print(f"\n=== PRIMARY, two parts, both gene-clustered ===")
    # part 1 — capture against chance
    primary = win_rate(cap[R], cap[C])
    lo_g, hi_g, _ = boot_ci(None, pair_gene, rng, args.n_boot,
                            stat_on(cap[R], cap[C]))
    lo_p, hi_p, _ = boot_ci(None, np.arange(len(R)), rng, args.n_boot,
                            stat_on(cap[R], cap[C]))
    width_g, width_p = hi_g - lo_g, hi_p - lo_p
    print(f"  [1] capture vs chance          {primary:.4f}")
    print(f"      95% CI, genes resampled    [{lo_g:.4f}, {hi_g:.4f}]   "
          f"{'EXCLUDES' if (lo_g > 0.5 or hi_g < 0.5) else 'INCLUDES'} 0.5")
    print(f"      95% CI, pairs resampled    [{lo_p:.4f}, {hi_p:.4f}]")
    print(f"      design effect (width^2)    {(width_g/max(width_p,1e-12))**2:.2f}")
    print(f"      per seed: " + "  ".join(
        f"{win_rate(cap_seeds[R, s], cap_seeds[C, s]):.4f}"
        for s in range(cap_seeds.shape[1])))
    part1 = lo_g > 0.5 or hi_g < 0.5

    # part 2 — capture against the Kozak matrix, PAIRED on the identical pairs.
    # The statistic is the rate at which capture picks the reference and the matrix
    # does not, against the reverse; 0.5 means the two agree as often as they differ
    # in each direction. Ties in either score count as one half, as everywhere here.
    cap_win = (cap[R] > cap[C]) + 0.5 * (cap[R] == cap[C])
    koz_win = (kozak[R] > kozak[C]) + 0.5 * (kozak[R] == kozak[C])
    disagree = cap_win != koz_win
    def head_to_head(ix):
        d = ix[disagree[ix]] if len(ix) else ix
        if not len(d):
            return 0.5
        return float(np.mean(cap_win[d] > koz_win[d]))
    h2h = head_to_head(np.arange(len(R)))
    lo_h, hi_h, _ = boot_ci(None, pair_gene, rng, args.n_boot, head_to_head)
    print(f"  [2] capture vs kozak_score     {h2h:.4f}   "
          f"(on the {int(disagree.sum()):,} pairs where they differ)")
    print(f"      95% CI, genes resampled    [{lo_h:.4f}, {hi_h:.4f}]   "
          f"{'EXCLUDES' if (lo_h > 0.5 or hi_h < 0.5) else 'INCLUDES'} 0.5")
    part2 = lo_h > 0.5 or hi_h < 0.5

    supported = part1 and part2
    print(f"\n  THE CLAIM IS {'SUPPORTED' if supported else 'NOT SUPPORTED'} — the "
          f"amended §8.5 requires BOTH gene-clustered intervals to exclude 0.5 "
          f"(chance {'yes' if part1 else 'no'}, kozak {'yes' if part2 else 'no'})")
    results["primary"] = dict(vs_chance=primary, ci_gene=[lo_g, hi_g],
                              ci_pair=[lo_p, hi_p], vs_kozak=h2h,
                              ci_kozak=[lo_h, hi_h], n_disagree=int(disagree.sum()),
                              supported=bool(supported),
                              n_pairs=int(len(R)), n_genes=int(n_genes))

    # ------------------------------------------- pre-specified direction split
    up = start[R] < start[C]        # the reference is the upstream member
    print(f"\n=== PRE-SPECIFIED SECONDARY: the direction split ===")
    print("  A monotone positional preference must push one of these below 0.5.")
    print("  AMENDED: each arm carries its own gene-clustered interval. The")
    print("  downstream arm was n=45 on validation; point estimates are not a claim.")
    excl = []
    for name, mask in (("reference UPSTREAM", up), ("reference DOWNSTREAM", ~up)):
        if mask.sum() == 0:
            print(f"  {name:<24} no pairs")
            excl.append(False)
            continue
        m = np.flatnonzero(mask)
        w = win_rate(cap[R][mask], cap[C][mask])
        lo_d, hi_d, _ = boot_ci(None, pair_gene[mask], rng, args.n_boot,
                                lambda ix, m=m: win_rate(cap[R][m[ix]], cap[C][m[ix]]))
        e = lo_d > 0.5 or hi_d < 0.5
        excl.append(bool(e))
        print(f"  {name:<24} {w:.4f}   n {int(mask.sum()):,}   "
              f"95% CI [{lo_d:.4f}, {hi_d:.4f}] {'EXCLUDES' if e else 'INCLUDES'} 0.5")
        results.setdefault("direction", {})[name] = dict(
            win_rate=w, n=int(mask.sum()), ci_gene=[lo_d, hi_d], excludes=bool(e))
    if all(excl) and len(excl) == 2:
        print("  BOTH intervals exclude 0.5 — evidence no MONOTONE positional")
        print("  account can produce. It does not exclude a non-monotone one: a")
        print("  preference peaked at a fixed distance from the midpoint cap is")
        print("  positional and puts both arms above 0.5, and references sit at a")
        print("  characteristic distance from that cap by construction.")
    else:
        print("  Not both intervals exclude 0.5 — the monotone-positional")
        print("  alternative is NOT excluded by this split.")

    # ------------------------------------------------- pre-specified baselines
    print(f"\n=== BOUNDING CONTEXT — information p_capture CANNOT access ===")
    print("  The ATG window carries 900 upstream and 100 into the ORF, and both")
    print("  members carry all 100 by the filter, so ORF length is invisible to the")
    print("  model and visible to these. A model against an oracle, not a failure.")
    for name, v in (("orf_length", pool.orf_length.to_numpy().astype(float)),
                    ("5' proximity (-orf_start)", -start.astype(float))):
        w = win_rate(v[R], v[C])
        print(f"  {name:<28} {w:.4f}")
        results.setdefault("context", {})[name] = w

    print(f"\n=== ALSO REPORTED ===")
    w_ejc = win_rate(pool.n_downstream_ejc.to_numpy().astype(float)[R],
                     pool.n_downstream_ejc.to_numpy().astype(float)[C])
    print(f"  n_downstream_ejc             {w_ejc:.4f}")
    print("    Expected BELOW 0.5: reference ORFs are long, so their stops sit")
    print("    3'-proximal and carry fewer downstream junctions than their short")
    print("    competitors. Signal in reverse, not a failed baseline.")
    results.setdefault("context", {})["n_downstream_ejc"] = w_ejc

    print(f"\n=== DESIGN GUARD, not a baseline ===")
    w_fill = win_rate(right_fill.astype(float)[R], right_fill.astype(float)[C])
    print(f"  right-hand window fill       {w_fill:.4f}")
    if w_fill != 0.5:
        sys.exit(f"FATAL: right-hand fill is constant within every pair by "
                 f"construction, so this must be exactly 0.5000. It is {w_fill!r}, "
                 f"which means the full-right-fill filter did not engage and the "
                 f"midpoint-clip control is not in force.")
    print("    exactly 0.5000 — constant within every pair, filter engaged")
    results["fill_guard"] = w_fill
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
