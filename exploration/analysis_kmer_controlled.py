#!/usr/bin/env python
"""
analysis_kmer_controlled.py — the AU-enrichment claim, with both confounds closed.

The finding: positions whose decay-branch ISM effect is elevated are enriched for
U/A-rich 5-mers (`TTTTT` +2.34) and depleted for GC-rich ones (−2.75), and the five
seeds agree on that enrichment vector at r = 0.75 while agreeing on the *positions*
at only 0.125. That is the signature of a motif the members share and place
differently.

It is not yet an AU-rich element, because two things reproduce it with no motif:

  1. GC COMPOSITION. The enriched-depleted axis IS a GC axis. If what makes a
     substitution's effect large is the GC shift it causes, then elevated positions
     are selected for GC-shiftability and the k-mer enrichment follows with nothing
     learned about sequence. CONTROL: rank positions using ONLY the GC-preserving
     substitution (A<->T, C<->G, `dgc == 0`), so the elevation criterion cannot see
     GC at all.

  2. REGIONAL COMPOSITION, the more dangerous one. Elevated positions concentrate 3'
     of the stop codon; the usual background is every valid position of the same
     transcripts, 5'UTR and CDS included. 3'UTRs are AU-rich AS A CLASS, so a
     positional bias toward the 3'UTR reproduces the entire enrichment. CONTROL:
     compute the enrichment SEPARATELY WITHIN each region, so elevated and
     background positions are drawn from the same compositional pool.

CONDITIONING, NOT MATCHING. A region-matched background would have to choose whether
regions are defined by the model-selected ORF or the annotated one — different sets,
different answers. Conditioning sidesteps that: if the enrichment survives inside the
downstream-of-stop stratum, regional composition is excluded rather than adjusted
for, and the choice of region definition can only change which stratum the evidence
comes from.

THE TWO CONTROLS ARE NOT SUBSTITUTES. Holding GC constant does not move where the
elevated positions are, so a clean result on (1) says nothing about (2). Both are
applied here, together and separately, so their contributions are visible.

Sequence comes from `obs` in the bank (ACGT = 0123), which is the observed base at
each transcript position and was verified against the FASTA with zero mismatches by
verify_ism_bank.py. k-mers spanning a non-ACGT or invalid position are dropped.

    python analysis_kmer_controlled.py results_ism_v6/bank_interp_s100.h5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# uint8 codes, so a k-mer is a contiguous byte slice and `.tobytes()` gives the
# string. np.array(list("ACGT")) is dtype '<U1' -- 4 bytes per character -- and
# .tobytes() on a slice of that returns null-padded UTF-32, so every k-mer would be
# a distinct unmatchable string and every count would be 1. Caught by unit test.
NT = np.frombuffer(b"ACGT", dtype=np.uint8)


def kmers_at(seq, ok, centres, k):
    """k-mers centred on `centres`, dropped if they span an invalid position."""
    half = k // 2
    out = []
    n = len(seq)
    for p in centres:
        lo, hi = p - half, p - half + k
        if lo < 0 or hi > n:
            continue
        if not ok[lo:hi].all():
            continue
        out.append(seq[lo:hi].tobytes().decode("ascii"))
    return out


def enrichment(fg, bg, min_count=50):
    """log2( fg share / bg share ) per k-mer, on k-mers seen enough in background.

    A symmetric half-count on both sides, not add-one on the foreground alone. With
    a small foreground, add-one is a large prior applied to one side only and it
    pushes every rare k-mer positive -- on a 3-transcript toy it made the "most
    depleted" k-mer read +1.00. Half on both keeps an absent k-mer bounded without
    biasing the ratio.
    """
    nf, nb = sum(fg.values()), sum(bg.values())
    if not nf or not nb:
        return {}
    K = len(bg)
    out = {}
    for km, c in bg.items():
        if c < min_count:
            continue
        f = fg.get(km, 0)
        out[km] = np.log2(((f + 0.5) / (nf + 0.5 * K)) / ((c + 0.5) / (nb + 0.5 * K)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("--column", default="vals_decay")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-bg", type=int, default=50)
    ap.add_argument("--region-anchor", default="selected",
                    choices=["selected", "reference"],
                    help="which ORF's stop defines downstream. selected: the "
                         "max-p_select candidate, the ORF whose decay is being "
                         "scored. reference: the annotated candidate where one "
                         "exists, max-p_select otherwise. They disagree for the "
                         "3,422 of 4,999 transcripts that have a reference, and "
                         "neither is obviously right -- run both.")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    print(f"k-mer enrichment with both confounds controlled — {Path(a.bank).name}")
    print(f"  column {a.column}   k = {a.k}   elevated = top {100*a.top_frac:g}% "
          f"of each transcript's valid positions")

    # four conditions: {all, GC-neutral scoring} x {pooled, region-conditioned}
    fg = {(s, r): Counter() for s in ("all", "neutral")
          for r in ("pooled", "upstream", "downstream")}
    bg = {k2: Counter() for k2 in fg}
    n_used = 0
    n_no_stop = 0

    with h5py.File(a.bank, "r") as f:
        n = f[a.column].shape[0]
        c_off, c_cnt = f["cand_offset"][:], f["cand_count"][:]
        c_end = f["cand_orf_end"][:]
        p_sel = f["p_select"][:]
        c_ref = f["cand_is_ref_cds"][:] if "cand_is_ref_cds" in f else None
        n_ref_anchor = 0
        take = range(n if not a.limit else min(a.limit, n))
        if a.limit:
            print(f"  --limit {a.limit}: a PREFIX of the stratified order, not a sample")

        for i in take:
            valid = f["valid"][i]
            obs = f["obs"][i]
            ok = valid & (obs >= 0)
            if ok.sum() < 200:
                continue
            seq = np.where(ok, NT[np.clip(obs, 0, 3)], ord("N")).astype(np.uint8)

            # the region boundary is the stop of the candidate the model actually
            # commits to, not the annotated one: "downstream of the stop" has to mean
            # downstream of the ORF whose decay the branch is scoring.
            sl = slice(int(c_off[i]), int(c_off[i]) + int(c_cnt[i]))
            ps = p_sel[sl]
            if not len(ps) or not np.isfinite(ps).any():
                n_no_stop += 1
                continue
            # WHICH ORF'S STOP DEFINES "DOWNSTREAM". The selected candidate is the
            # ORF whose decay the branch is actually scoring; the reference is the
            # annotation. They differ for most transcripts and neither is obviously
            # right, so the flag exists and both are run. If the result depends on
            # which, that dependence is the finding.
            j = int(np.argmax(ps))
            if a.region_anchor == "reference" and c_ref is not None:
                r_ = np.flatnonzero(c_ref[sl] == 1)
                if len(r_):
                    j = int(r_[0])
                    n_ref_anchor += 1
            stop = int(c_end[sl][j])
            pos1 = np.arange(1, len(valid) + 1)          # 1-based transcript position
            downstream = pos1 > stop

            x = np.abs(f[a.column][i])
            dgc = f["dgc"][i]
            n_used += 1
            for scoring in ("all", "neutral"):
                xx = x if scoring == "all" else np.where(dgc == 0, x, np.nan)
                with np.errstate(all="ignore"):
                    eff = np.nanmax(np.where(np.isfinite(xx), xx, np.nan), axis=1)
                good = ok & np.isfinite(eff)
                if good.sum() < 200:
                    continue
                for region, rmask in (("pooled", np.ones_like(good)),
                                      ("upstream", ~downstream),
                                      ("downstream", downstream)):
                    sel = good & rmask
                    m = int(sel.sum())
                    if m < 200:
                        continue
                    # elevated: top fraction WITHIN this region, so the comparison
                    # is against that region's own positions and nothing else
                    kk = max(1, int(round(a.top_frac * m)))
                    idx = np.flatnonzero(sel)
                    cut = np.partition(eff[idx], -kk)[-kk]
                    hi = idx[eff[idx] >= cut]
                    fg[(scoring, region)].update(kmers_at(seq, ok, hi, a.k))
                    bg[(scoring, region)].update(kmers_at(seq, ok, idx, a.k))

    print(f"\n  transcripts used {n_used:,}"
          + (f"   (skipped, no selectable stop: {n_no_stop:,})" if n_no_stop else ""))
    print(f"  region anchor: {a.region_anchor}"
          + (f"   ({n_ref_anchor:,} transcripts anchored on an annotated ORF, "
             f"the rest on max p_select)" if a.region_anchor == "reference" else
             "   (every transcript on the ORF the model commits to)"))

    ref = None
    print(f"\n  {'scoring':<9} {'region':<11} {'fg k-mers':>11} {'bg k-mers':>12} "
          f"{'top enriched':>26} {'top depleted':>26}  {'r vs pooled/all':>15}")
    for scoring in ("all", "neutral"):
        for region in ("pooled", "upstream", "downstream"):
            e = enrichment(fg[(scoring, region)], bg[(scoring, region)], a.min_bg)
            if not e:
                continue
            srt = sorted(e.items(), key=lambda t: -t[1])
            up = ", ".join(f"{m} {v:+.2f}" for m, v in srt[:2])
            dn = ", ".join(f"{m} {v:+.2f}" for m, v in srt[-2:])
            if ref is None:
                ref = e
                r = 1.0
            else:
                shared = sorted(set(ref) & set(e))
                r = (float(np.corrcoef([ref[m] for m in shared],
                                       [e[m] for m in shared])[0, 1])
                     if len(shared) > 10 else float("nan"))
            print(f"  {scoring:<9} {region:<11} {sum(fg[(scoring,region)].values()):>11,} "
                  f"{sum(bg[(scoring,region)].values()):>12,} {up:>26} {dn:>26}  {r:>15.3f}")

    print("\n  r is the enrichment vector against pooled/all, on shared k-mers.")
    print("  If AU enrichment is REGIONAL, the downstream rows lose it (r drops, and")
    print("  the top-enriched k-mers stop being U/A-rich). If it is GC-DRIVEN, the")
    print("  neutral rows lose it. Both controls are independent and both are here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
