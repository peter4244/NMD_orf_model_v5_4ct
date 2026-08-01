#!/usr/bin/env python
"""
probe_geometry_baseline.py — how much of `is_ref_cds` does WINDOW GEOMETRY give
you for free, with no model at all?

WHY THIS IS THE CONTROL THAT MATTERS. The capture head separates reference starts
from other candidates at AUC 0.8382 (probe_junction_ablation2). That is only
evidence about initiation if a trivial non-sequence property of the window does
not do the same job. Three measurements say it might:

    keep_phase   AUC 0.8132   channels 6-8 alone -- and those are a FIXED
                              period-3 carrier times the fill mask, so the only
                              thing varying across candidates is WHERE THE WINDOW
                              IS FILLED
    keep_geom    AUC 0.8353   junction + fill, no sequence identity whatsoever
    keep_seq     AUC 0.9060   sequence alone, better than the full model

Two disjoint channel groups are each sufficient. So nothing in the ablation suite
can attribute the gap to sequence, and the question becomes: is the geometry route
real, or is it an artifact of feeding the model inputs it never saw in training?

A MODEL-FREE BASELINE ANSWERS THAT WITHOUT ANY MANIFOLD CONCERN AT ALL. These are
properties of the stored window. If distance-to-5'-end predicts is_ref_cds about
as well as the trained capture head does, then the head's 0.8382 is not evidence
of learned initiation -- it is evidence that annotated starts sit at a
characteristic place in a transcript, which was already known.

Also corrects a control of my own. In pass 2 I stratified on MEAN upstream fill in
deciles, saw AUC rise from 0.8382 to 0.8748, and told the model window the
geometry confound had come back clean. Mean fill is a scalar; the fill MASK is a
1000-position pattern whose informative content is the position of the contiguous
filled run's edges. Deciles of the scalar do not control the pattern. That control
was too coarse and its reassurance did not hold.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tensor_io import decode_windows           # noqa: E402


def auc(score, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    r = pd.Series(np.asarray(score, float)).rank().to_numpy()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_auc(score, pos, strat, n_min=8):
    num = den = 0.0
    for s in np.unique(strat):
        m = strat == s
        a, b = m & pos, m & ~pos
        if a.sum() >= n_min and b.sum() >= n_min:
            num += a.sum() * auc(score[m], a[m]); den += a.sum()
    return (num / den if den else float("nan")), int(den)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    with h5py.File(REPO / "results_tensor_chr21" / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        cnt, o_s, codes = f["count"][:], f["orf_start"][:], f["codes"][:]
        attrs = dict(f.attrs)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))

    ref = pool["is_ref_cds"].to_numpy() == 1
    slot = pool["slot"].to_numpy()
    kz = pool["kozak_score"].to_numpy()
    tx = pool["tx"].to_numpy()

    L = int(attrs["atg_left"])
    anchor = o_s.astype(np.int64)
    win = decode_windows(codes[:, 0], anchor, L, anchor)
    filled = win[:, 6:9].sum(1) > 0

    first = np.argmax(filled, axis=1).astype(float)          # first filled index
    nfill = filled.sum(1).astype(float)
    last = first + nfill                                     # exclusive
    up_fill = filled[:, :L].mean(1)
    dn_fill = filled[:, L:].mean(1)
    njunc = win[:, 4].sum(1)
    gc = np.where(filled, win[:, 5], np.nan)
    gc_mean = np.nanmean(np.where(filled, win[:, 5], np.nan), axis=1)

    print(f"chr21: {len(pool):,} candidates, {int(ref.sum())} reference starts")
    print(f"ATG window {L}+{attrs['atg_right']}; 'first' is the index of the first")
    print(f"filled position, so 900-first is how much real 5' context exists.\n")

    print("=== every feature is a property of the STORED WINDOW. No model. ===")
    print(f"  {'feature':<28} {'AUC':>7} {'AUC|slot':>9} {'AUC|kozak':>10}")
    dec = np.asarray(pd.qcut(kz, 10, labels=False, duplicates="drop"))
    feats = {
        "first filled index": first,
        "n filled positions": nfill,
        "upstream fill fraction": up_fill,
        "downstream fill fraction": dn_fill,
        "junction marks in window": njunc,
        "mean GC over filled": gc_mean,
        "slot (candidate rank)": slot.astype(float),
        "kozak_score (PWM)": kz,
    }
    for nm, v in feats.items():
        v = np.asarray(v, float)
        a = auc(v, ref)
        # a feature can point either way; report the informative direction
        a = max(a, 1 - a)
        s, _ = strat_auc(v, ref, slot)
        k, _ = strat_auc(v, ref, dec)
        print(f"  {nm:<28} {a:>7.4f} {max(s,1-s):>9.4f} {max(k,1-k):>10.4f}")

    print("\n=== the geometry-only combination ===")
    X = np.column_stack([first, nfill, up_fill, dn_fill, njunc]).astype(float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    X = np.column_stack([X, np.ones(len(X))])
    y = ref.astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(300):                       # plain IRLS-free gradient fit
        p = 1 / (1 + np.exp(-X @ w))
        w -= 0.5 * (X.T @ (p - y)) / len(X)
    geo = X @ w
    a_geo = auc(geo, ref)
    print(f"  logistic on 5 geometry features, IN-SAMPLE (so an upper bound):")
    print(f"    AUC {a_geo:.4f}")
    s, _ = strat_auc(geo, ref, slot)
    print(f"    within slot {s:.4f}")

    print("\n=== side by side with the trained capture head ===")
    npz = REPO / "results_interp_all" / "junction_ablation_capture_chr21.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        cap = z["full"].mean(0)
        print(f"  trained capture head (5-seed mean)   AUC {auc(cap, ref):.4f}")
        print(f"  geometry-only logistic, in-sample    AUC {a_geo:.4f}")
        print(f"  best single geometry feature         AUC "
              f"{max(max(auc(np.asarray(v,float), ref), 1-auc(np.asarray(v,float), ref)) for k2, v in feats.items() if k2 not in ('kozak_score (PWM)','mean GC over filled')):.4f}")
        print(f"  kozak_score (PWM) alone              AUC "
              f"{max(auc(kz, ref), 1-auc(kz, ref)):.4f}")

    print("\n=== the comparison that geometry cannot confound ===")
    print("  Within a transcript, every candidate shares the same 5' end and the")
    print("  same exon structure. Restricting to transcripts that hold a reference")
    print("  start AND at least one non-reference candidate, and asking whether the")
    print("  reference start outranks its OWN transcript's competitors, removes")
    print("  between-transcript geometry entirely.")
    if npz.exists():
        cap = np.load(npz, allow_pickle=True)["full"].mean(0)
        wins = ties = tot = 0
        for t in np.unique(tx):
            m = tx == t
            a, b = m & ref, m & ~ref
            if a.sum() == 1 and b.sum() >= 1:
                r_, o_ = cap[a][0], cap[b]
                wins += int((r_ > o_).sum()); ties += int((r_ == o_).sum())
                tot += len(o_)
        print(f"    within-transcript AUC (capture) {(wins + .5*ties)/tot:.4f}   "
              f"over {tot:,} reference-vs-competitor pairs")
        for nm, v in (("first filled index", first), ("slot", slot.astype(float)),
                      ("kozak_score", kz)):
            v = np.asarray(v, float)
            wins = ties = tot = 0
            for t in np.unique(tx):
                m = tx == t
                a, b = m & ref, m & ~ref
                if a.sum() == 1 and b.sum() >= 1:
                    r_, o_ = v[a][0], v[b]
                    wins += int((r_ > o_).sum()); ties += int((r_ == o_).sum())
                    tot += len(o_)
            g = (wins + .5 * ties) / tot
            print(f"    within-transcript AUC ({nm}) {max(g,1-g):.4f}")


if __name__ == "__main__":
    main()
