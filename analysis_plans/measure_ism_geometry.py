#!/usr/bin/env python
"""
measure_ism_geometry.py — how much work is one transcript's in-silico mutagenesis?

Written BEFORE the mutagenesis bank's plan section, because the plan's subset size and
its arm list both depend on what a transcript actually costs, and the only estimate in
hand is a laptop measurement of a different architecture.

WHAT DECIDES THE COST. A substitution at transcript position p changes the input of
exactly those candidate windows whose FILLED range contains p. Every other candidate's
capture and decay are unchanged, so they are computed once and reused; the stick-breaking
product still has to be re-aggregated over the whole transcript, but that is arithmetic
over K numbers, not an encoder pass.

So the cost is not "positions x candidates". It is

    encoder passes = 3 x sum over candidates of (2 x filled ATG positions
                                                 + 1 x filled stop positions)

ATG counts twice because two encoders read that window — enc_init and enc_atg — and one
counts because only enc_stop reads the stop window. The 3 is the three substitutions per
position; the observed base is a no-op and is sampled separately.

GEOMETRY, read from build_tensor.py rather than from the plan's prose:

    mid       = (orf_start + orf_end) // 2                  # mid belongs to the ATG window
    ATG  window: anchor orf_start,  left 900, right 100, fill bounded to [1, mid]
    stop window: anchor orf_end-1,  left 500, right 500, fill bounded to [mid+1, tx_length]

    filled ATG  = [max(1, orf_start - 900),   min(tx_length, mid, orf_start + 99)]
    filled stop = [max(mid + 1, orf_end - 501), min(tx_length, orf_end + 498)]

Both are clipped again at the array bounds, which is what the min/max above already do.

No model, no torch: this is pool arithmetic and it runs in seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TABLES = Path.home() / "claude_projects" / "nmd_w69_tables_2026-07-30"
POOL = REPO / "results_pool_v6" / "orf_pool.tsv"

ATG_LEFT, ATG_RIGHT = 900, 100
STOP_LEFT, STOP_RIGHT = 500, 500
WINDOW = ATG_LEFT + ATG_RIGHT

N_SUBSET = 1000          # the size the interpretation window asked for
SUBSET_SEED = 20260801


def pct(x, q):
    return np.percentile(np.asarray(x, dtype=np.float64), q)


def describe(name, v, unit="", fmt=",.0f"):
    v = np.asarray(v, dtype=np.float64)
    print(f"  {name:<46} mean {format(v.mean(), fmt):>12}  median "
          f"{format(np.median(v), fmt):>12}  p90 {format(pct(v, 90), fmt):>12}  "
          f"max {format(v.max(), fmt):>12} {unit}")


def main():
    sys.stdout.reconfigure(line_buffering=True)

    tx = pd.read_csv(TABLES / "tx_summary.tsv", sep="\t")
    ref = pd.read_csv(TABLES / "ref_cds_features.tsv", sep="\t",
                      usecols=["isoform_id", "gene_id"])
    pool = pd.read_csv(POOL, sep="\t",
                       usecols=["isoform_id", "slot", "orf_start", "orf_end"])

    tx = tx.merge(ref.drop_duplicates("isoform_id"), on="isoform_id", how="left")
    print(f"transcripts in TX          : {len(tx):,}")
    print(f"candidates in the pool     : {len(pool):,}")
    print(f"transcripts in the pool    : {pool['isoform_id'].nunique():,}")

    # ------------------------------------------------------------- geometry
    L = tx.set_index("isoform_id")["tx_length"]
    pool = pool.join(L.rename("tx_length"), on="isoform_id")
    s = pool["orf_start"].to_numpy(np.int64)
    e = pool["orf_end"].to_numpy(np.int64)
    tl = pool["tx_length"].to_numpy(np.int64)
    mid = (s + e) // 2

    a_lo = np.maximum(1, s - ATG_LEFT)
    a_hi = np.minimum(np.minimum(tl, mid), s + ATG_RIGHT - 1)
    s_lo = np.maximum(mid + 1, (e - 1) - STOP_LEFT)
    s_hi = np.minimum(tl, (e - 1) + STOP_RIGHT - 1)

    a_n = np.maximum(0, a_hi - a_lo + 1)
    s_n = np.maximum(0, s_hi - s_lo + 1)
    pool["atg_filled"], pool["stop_filled"] = a_n, s_n
    pool["a_lo"], pool["a_hi"], pool["s_lo"], pool["s_hi"] = a_lo, a_hi, s_lo, s_hi

    print(f"\n=== per candidate, filled window positions (of {WINDOW} each) ===")
    describe("ATG window filled positions", a_n)
    describe("stop window filled positions", s_n)
    print(f"  candidates with an empty ATG window            : {(a_n == 0).sum():,}")
    print(f"  candidates with an empty stop window           : {(s_n == 0).sum():,}")
    print(f"  candidates whose ATG window is full ({ATG_LEFT+ATG_RIGHT})    : "
          f"{(a_n == WINDOW).sum():,} ({100*(a_n == WINDOW).mean():.1f}%)")
    print(f"  candidates whose stop window is full ({WINDOW})   : "
          f"{(s_n == WINDOW).sum():,} ({100*(s_n == WINDOW).mean():.1f}%)")

    # ---------------------------------------------- per-transcript coverage
    # The union of filled ranges over a transcript's candidates: how many transcript
    # positions are readable by the model at all. A position covered by no window
    # cannot be probed, and the bank must say so rather than report a zero effect.
    print(f"\n=== per transcript ===", flush=True)
    grp = pool.groupby("isoform_id", sort=True)
    per_tx = grp.agg(K=("slot", "size"),
                     atg_sum=("atg_filled", "sum"),
                     stop_sum=("stop_filled", "sum"),
                     tx_length=("tx_length", "first"))

    covered, n_intervals = [], []
    for iso, g in grp[["a_lo", "a_hi", "s_lo", "s_hi", "atg_filled", "stop_filled"]]:
        iv = []
        gl = g[["a_lo", "a_hi", "atg_filled"]].to_numpy()
        for lo, hi, n in gl:
            if n > 0:
                iv.append((lo, hi))
        gs = g[["s_lo", "s_hi", "stop_filled"]].to_numpy()
        for lo, hi, n in gs:
            if n > 0:
                iv.append((lo, hi))
        iv.sort()
        tot, cur_lo, cur_hi = 0, None, None
        for lo, hi in iv:
            if cur_lo is None:
                cur_lo, cur_hi = lo, hi
            elif lo <= cur_hi + 1:
                cur_hi = max(cur_hi, hi)
            else:
                tot += cur_hi - cur_lo + 1
                cur_lo, cur_hi = lo, hi
        if cur_lo is not None:
            tot += cur_hi - cur_lo + 1
        covered.append(tot)
        n_intervals.append(len(iv))
    per_tx["covered"] = covered
    per_tx["n_windows"] = n_intervals

    describe("transcript length", per_tx["tx_length"], "bases")
    describe("candidates K", per_tx["K"])
    describe("transcript positions covered by >=1 window", per_tx["covered"], "bases")
    frac = per_tx["covered"] / per_tx["tx_length"]
    describe("fraction of the transcript covered", frac, "", ".3f")
    print(f"  transcripts fully covered                      : "
          f"{int((frac >= 0.999).sum()):,} ({100*(frac >= 0.999).mean():.1f}%)")

    # WHERE THE UNCOVERED POSITIONS ARE. Candidates must start in the first half of
    # the transcript (§3 step 4), so the 3' reach of the pool ends at the last stop
    # window, which extends 498 bases past the 3'-most candidate's stop codon.
    # Everything beyond that is read by no window and cannot be probed at all.
    last_cov = grp.apply(lambda g: max(g["s_hi"].max(), g["a_hi"].max()),
                         include_groups=False)
    tail = np.maximum(0, per_tx["tx_length"] - last_cov.reindex(per_tx.index))
    uncov = per_tx["tx_length"] - per_tx["covered"]
    print(f"  mean uncovered bases per transcript            : {uncov.mean():,.0f}")
    print(f"  mean bases 3' of the last window               : {tail.mean():,.0f}")
    print(f"  share of the uncovered mass that is that tail  : "
          f"{100*tail.sum()/uncov.sum():.1f}%")
    per_tx["last_covered"] = last_cov.reindex(per_tx.index)

    # ------------------------------------------------------------- the cost
    # encoder passes, cached: only the windows containing p are recomputed
    enc_cached = 3 * (2 * per_tx["atg_sum"] + per_tx["stop_sum"])
    # encoder passes, naive: every candidate re-encoded for every substitution
    enc_naive = 3 * per_tx["covered"] * (2 * per_tx["K"] + per_tx["K"])
    per_tx["enc_cached"] = enc_cached
    per_tx["enc_naive"] = enc_naive

    print(f"\n=== encoder passes per transcript (one pass = one window through one encoder) ===")
    describe("cached  (recompute only affected windows)", enc_cached)
    describe("naive   (recompute every candidate)", enc_naive)
    print(f"  saving factor, on the totals                   : "
          f"{enc_naive.sum()/enc_cached.sum():.1f}x")
    describe("saving factor, per transcript", enc_naive / enc_cached, "", ".1f")

    # -------------------------------------------- what a 1,000-transcript bank costs
    rng = np.random.default_rng(SUBSET_SEED)
    lab = tx.set_index("isoform_id")["is_nmd"]
    per_tx["is_nmd"] = lab.reindex(per_tx.index).to_numpy()
    per_tx["gene_id"] = tx.set_index("isoform_id")["gene_id"].reindex(per_tx.index).to_numpy()

    print(f"\n=== a {N_SUBSET:,}-transcript bank, label-stratified at the pool's own prevalence ===")
    prev = float(per_tx["is_nmd"].mean())
    n_pos = int(round(N_SUBSET * prev))
    pos = per_tx.index[per_tx["is_nmd"] == 1].to_numpy()
    neg = per_tx.index[per_tx["is_nmd"] == 0].to_numpy()
    pick = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, N_SUBSET - n_pos, replace=False)])
    sub = per_tx.loc[pick]
    print(f"  prevalence in the pool {prev:.3f} -> {n_pos} positive, "
          f"{N_SUBSET - n_pos} negative")
    for name, col in (("cached", "enc_cached"), ("naive", "enc_naive")):
        tot = float(sub[col].sum())
        print(f"  {name:<7} total encoder passes            : {tot:,.0f}")
        for rate in (20_000, 85_000, 200_000):
            print(f"      at {rate:>7,}/s : {tot/rate/60:8.1f} min per arm, "
                  f"{3*tot/rate/60:8.1f} min for 3 window arms")

    # ------------------------------------------------------------ the memory wall
    # What is resident while one transcript is processed, if every perturbation of
    # that transcript is batched at once. This is the quantity probe_ism_cost.py
    # caught at 35.3 GiB on the old architecture.
    print(f"\n=== input-tensor bytes if ONE transcript's substitutions are batched at once ===")
    rows = 3 * (per_tx["atg_sum"] + per_tx["stop_sum"])      # window-encodes, not passes
    gb = rows * 9 * WINDOW * 4 / 1e9                          # float32, 9 channels
    describe("float32 input, whole transcript at once (GB)", gb, "", ".2f")
    print(f"  transcripts over 8 GB of input alone           : {int((gb > 8).sum()):,}")
    print(f"  transcripts over 32 GB of input alone          : {int((gb > 32).sum()):,}")
    print("  -> the bank chunks by perturbation, not by transcript; the chunk size is")
    print("     what bounds memory and it is set from a measured per-row cost, not this.")

    # ---------------------------------------------------------- gene structure
    print(f"\n=== gene structure, for the discovery / confirmation split ===")
    g = per_tx.groupby("gene_id")
    print(f"  genes over the {len(per_tx):,} pooled transcripts : "
          f"{per_tx['gene_id'].nunique():,}")
    print(f"  transcripts with no gene_id                    : "
          f"{int(per_tx['gene_id'].isna().sum()):,}")
    sizes = g.size()
    describe("transcripts per gene", sizes, "", ",.1f")
    mixed = g["is_nmd"].nunique()
    print(f"  genes carrying both labels                     : "
          f"{int((mixed == 2).sum()):,} of {len(sizes):,}")

    out = REPO / "analysis_plans" / "ism_geometry_per_transcript.tsv"
    per_tx.reset_index().to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}  ({len(per_tx):,} rows)")


if __name__ == "__main__":
    main()
