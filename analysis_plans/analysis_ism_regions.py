#!/usr/bin/env python
"""
analysis_ism_regions.py — Pete's three questions, answered directly.

  1. Are there regions, one to ten bases, where the ISM signal is clearly
     elevated above baseline?
  2. If so, where are they relative to the start codon and the stop codon?
  3. If so, do they carry a sequence signature?

WHY THIS SCRIPT EXISTS RATHER THAN A RE-CUT OF WHAT WE HAD. The existing profile
(`analysis3_capture_ism_profile.py` -> `capture_ism_profile.npz`) stores four
arrays of length 1,000: cohort SUMS and COUNTS at each fixed window index. That is
the cross-isoform average at a fixed offset, which this project measured as
diluting the per-isoform maximum about ninefold, and it retains no per-base
values. So it cannot say whether any single transcript has an elevated region, and
it cannot say what base sits there. It answers none of the three.

WHAT "BASELINE" MEANS HERE, because the word carries two different jobs:

  the METHOD floor   below which a number is arithmetic, not signal. Shipped per
                     position by the bank as `chunk_offset`; measured
                     independently as a few float32 ulp of the transcript's own
                     logit (probe_bank_floor_chunk_invariance.py). A position must
                     clear this before it is discussed at all.
  the POSITIONAL     what an ordinary position of the same transcript does.
  baseline           Elevation is reported as a fold over the transcript's own
                     median, never as an absolute, because transcripts differ in
                     overall responsiveness by orders of magnitude.

THE TWO NULLS, one per question.

  Q1 asks whether elevated positions form RUNS or are scattered. Rotating the
  track cannot answer that -- a rotation preserves the distribution exactly. The
  right null is to keep the same number of elevated positions and place them at
  random, then compare run lengths. That is what separates "a region" from "a
  handful of unconnected bases".

  Q2 asks whether elevation sits at particular offsets. That null is the circular
  shift: rotate each transcript's effect track against its own anchor and rebuild
  the same profile. A peak the shifted data reproduces is not a location.

PER-ISOFORM FIRST, ALWAYS. Every statistic is computed within a transcript and
only then combined, for the dilution reason above.

THE POOLING-BIN CAVEAT, which has already produced one artifact here. The encoder
pools into 8 bins over the 1,000-position axis, so ATG-window indices 875-999 --
offsets -25 to +99 relative to the start codon -- are ONE bin. A previous run
found its top offsets at +58 to +97, inside that final bin and within 41 positions
of the padded array end, and the matched controls showed the same thing. Any peak
in that range is flagged rather than reported.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from build_ism_bank import ATG_LEFT, STOP_LEFT      # noqa: E402

NT = "ACGT"
FLOOR_ABS = 3e-6          # measured; see probe_bank_floor_chunk_invariance.py
LAST_BIN_LO = -25         # ATG-window offsets inside the final pooling bin
N_SHIFT = 5               # circular-shift draws averaged into the null


def per_position_effect(vals):
    """Largest |effect| over the three substitutions at each position.

    Not the mean: `mean_abs` was measured on this project to be unable to
    discriminate when many weak contributors are present, and it ranked a
    composition-confounded operator above a clean one. The maximum is what the
    SpliceAI validation used and it is what a motif would move.
    """
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.abs(vals), axis=1)


def runs_of(mask):
    """Lengths of contiguous True runs."""
    if not mask.any():
        return np.zeros(0, dtype=int)
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def load_shards(shard_dir, pool):
    """One record per transcript, with the reference candidate as the anchor."""
    out = []
    for p in sorted(Path(shard_dir).glob("*.npz")):
        z = np.load(p)
        iso = p.stem
        g = pool[pool.isoform_id == iso].sort_values("slot", kind="stable")
        if g.empty or not (g.is_ref_cds == 1).any():
            continue
        k = int(np.flatnonzero(g.is_ref_cds.to_numpy() == 1)[0])
        spans = z["spans"]
        if k >= len(spans):
            continue
        out.append(dict(iso=iso, k=k,
                        vals=z["vals"], cap=z["vals_capture"],
                        dec=z["vals_decay"], obs=z["obs"], valid=z["valid"],
                        floor=z["chunk_offset"], fill=z["fill_count"],
                        dgc=z["dgc"], spans=spans,
                        orf_start=int(g.orf_start.to_numpy()[k]),
                        orf_end=int(g.orf_end.to_numpy()[k]),
                        base_logit=float(z["base_logit"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--column", default="vals",
                    choices=["vals", "cap", "dec"],
                    help="vals = the transcript logit; dec = the decay branch")
    ap.add_argument("--fold", type=float, default=10.0,
                    help="elevation threshold, as a fold over the transcript median")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "orf_start", "orf_end",
                                "is_ref_cds"])
    recs = load_shards(args.shards, pool)
    if not recs:
        sys.exit(f"no usable shards in {args.shards}")

    key = {"vals": "vals", "cap": "cap", "dec": "dec"}[args.column]
    name = {"vals": "the transcript logit", "cap": "the capture branch",
            "dec": "the decay branch"}[args.column]
    print(f"{len(recs)} transcripts, effect column = {name}\n")

    # ================================================================== Q1
    print("=" * 78)
    print("Q1  Are there regions where the signal is clearly above baseline?\n")
    rows, run_real, run_null = [], [], []
    for r in recs:
        eff = per_position_effect(r[key])
        ok = r["valid"] & np.isfinite(eff)
        if ok.sum() < 50:
            continue
        e = eff[ok]
        med = np.median(e)
        above_floor = (e > np.maximum(r["floor"][ok] * 2, FLOOR_ABS)).mean()
        hi = eff >= args.fold * med
        hi &= ok
        run_real.append(runs_of(hi))
        # NULL for "is it a region": same COUNT of elevated positions, placed at
        # random among the valid ones. Preserves how many, destroys where.
        idx = np.flatnonzero(ok)
        pick = rng.choice(idx, size=int(hi.sum()), replace=False) if hi.sum() else []
        nm = np.zeros_like(hi)
        nm[pick] = True
        run_null.append(runs_of(nm))
        rows.append((r["iso"], int(ok.sum()), med, np.percentile(e, 99), e.max(),
                     e.max() / med if med > 0 else np.nan,
                     100 * above_floor, int(hi.sum())))

    df = pd.DataFrame(rows, columns=["iso", "n_pos", "median", "p99", "max",
                                     "max_fold", "pct_above_floor", "n_elev"])
    print(f"  per-transcript effect distribution, {len(df)} transcripts")
    print(f"    median position          {df['median'].median():.3e}")
    print(f"    99th percentile          {df['p99'].median():.3e}")
    print(f"    largest position         {df['max'].median():.3e}")
    print(f"    largest / median         {df['max_fold'].median():.1f}x   "
          f"(range {df['max_fold'].min():.1f} to {df['max_fold'].max():.1f})")
    print(f"    positions above floor    {df['pct_above_floor'].median():.1f}%")
    print(f"\n  So the answer to 'is anything elevated' is a fold, not a yes/no:")
    print(f"  the strongest single position of a transcript runs "
          f"{df['max_fold'].median():.0f}x its own median.\n")

    rr = np.concatenate(run_real) if run_real else np.zeros(0)
    rn = np.concatenate(run_null) if run_null else np.zeros(0)
    print(f"  ARE THEY REGIONS OR SCATTERED BASES?  positions at >= "
          f"{args.fold:.0f}x the transcript median,")
    print(f"  against the same count placed at random in the same transcripts:\n")
    print(f"    {'run length':<12} {'observed':>10} {'random':>10}")
    for L in range(1, 11):
        print(f"    {L:<12} {int((rr == L).sum()):>10,} {int((rn == L).sum()):>10,}")
    print(f"    {'>= 11':<12} {int((rr >= 11).sum()):>10,} {int((rn >= 11).sum()):>10,}")
    print(f"\n    total elevated positions {int(rr.sum()):,}  in {len(rr):,} runs "
          f"(random: {len(rn):,} runs)")
    print(f"    mean run length  observed {rr.mean() if len(rr) else 0:.2f}   "
          f"random {rn.mean() if len(rn) else 0:.2f}")

    # ================================================================== Q2
    # FILL-CONDITIONED IS THE HEADLINE, RAW IS THE SECONDARY. In the raw profile
    # the set of transcripts contributing changes with offset -- a transcript
    # appears at +150 only if its window is filled that far -- so a rise with
    # offset can be produced entirely by which isoforms are still present. The
    # conditioned profile fixes ONE transcript set that is filled across the whole
    # reported range, so every offset is an average over the same isoforms and a
    # difference between offsets is a difference in position.
    print("\n" + "=" * 78)
    print("Q2  Where are they, relative to the start codon and the stop codon?\n")
    for anchor, lo, hi_ in (("start codon", -300, 99), ("stop codon", -300, 300)):
        width = hi_ - lo + 1
        for conditioned in (True, False):
            prof = np.zeros(width); prof_n = np.zeros(width); shuf = np.zeros(width)
            n_tx = 0
            for r in recs:
                eff = per_position_effect(r[key])
                ok = r["valid"] & np.isfinite(eff)
                if ok.sum() < 50:
                    continue
                med = np.median(eff[ok])
                if med <= 0:
                    continue
                a = r["orf_start"] if anchor == "start codon" else r["orf_end"] - 1
                pos = np.flatnonzero(ok) + 1             # 1-based transcript pos
                off = pos - a
                keep = (off >= lo) & (off <= hi_)
                if not keep.any():
                    continue
                # the condition: this transcript must cover the WHOLE range
                if conditioned and len(np.unique(off[keep])) < width:
                    continue
                n_tx += 1
                np.add.at(prof, off[keep] - lo, eff[ok][keep] / med)
                np.add.at(prof_n, off[keep] - lo, 1)
                # THE SHIFT ROTATES WITHIN THE REPORTED RANGE, NOT ACROSS THE
                # TRANSCRIPT. Rotating the whole transcript moves ATG-window
                # values into stop-window offsets, and the two windows differ in
                # overall responsiveness by 1.5-2x here -- so the null sat at the
                # transcript-wide average instead of at the level of the region
                # being profiled, and every positional excess came out about
                # twice its true size. Rotating inside the range preserves the
                # multiset of effects there and destroys only their arrangement,
                # which is exactly what a positional claim needs tested.
                #
                # A ROTATION, not a permutation: positions are autocorrelated
                # (far-field lag-1 0.60), and a permutation would destroy that
                # too, making the null easy to beat for a reason unrelated to
                # position.
                sel = np.flatnonzero(keep)
                v_in = eff[ok][sel]
                if len(v_in) > 2:
                    # several draws, because one rotation is one draw from a
                    # distribution and the top-of-table is chosen on the null's
                    # noise as much as on its level
                    acc_s = np.zeros(len(v_in))
                    for _ in range(N_SHIFT):
                        acc_s += np.roll(v_in, int(rng.integers(1, len(v_in))))
                    np.add.at(shuf, off[sel] - lo, acc_s / (N_SHIFT * med))
            m = prof_n > 0
            if not m.any():
                continue
            mean_fold = np.where(m, prof / np.maximum(prof_n, 1), np.nan)
            mean_shuf = np.where(m, shuf / np.maximum(prof_n, 1), np.nan)
            tag = ("FILL-CONDITIONED — one transcript set across every offset"
                   if conditioned else "raw — the transcript set varies with offset")
            print(f"  relative to the {anchor}, {tag}")
            print(f"    {n_tx} transcripts cover {lo:+} to {hi_:+}")
            order = np.argsort(-np.nan_to_num(mean_fold))
            print(f"    {'offset':>7} {'mean fold':>10} {'shifted':>9} "
                  f"{'n tx':>6}   note")
            for i in order[:8]:
                o = i + lo
                note = ""
                if anchor == "start codon" and o >= LAST_BIN_LO:
                    note = "FINAL POOLING BIN — flagged"
                if abs(o) <= 2:
                    note = "the codon itself"
                print(f"    {o:>+7} {mean_fold[i]:>10.2f} {mean_shuf[i]:>9.2f} "
                      f"{int(prof_n[i]):>6}   {note}")
            good = m.copy()
            if anchor == "start codon":
                good &= (np.arange(width) + lo) < LAST_BIN_LO
            if good.any():
                b = int(np.nanargmax(np.where(good, mean_fold, -np.inf)))
                print(f"    best unflagged offset {b + lo:+}: {mean_fold[b]:.2f} "
                      f"against shifted {mean_shuf[b]:.2f} "
                      f"({mean_fold[b] - mean_shuf[b]:+.2f})")
            print()

    # ============================================= Q2b, the fill-boundary check
    # THE CHECK THE LAST POSITIONAL FINDING DIED FOR. A window's filled region
    # ends somewhere, and near that edge two things change that have nothing to do
    # with sequence: channel 5 averages GC over a clipped span, and the
    # convolutions see padding. So an elevation that sits at the fill boundary is
    # the encoding, not a motif.
    #
    # This is decisive because it does not depend on knowing where the boundary
    # falls: plot effect against DISTANCE TO THE NEAREST BOUNDARY of the window
    # the position sits in. A boundary artifact spikes at small distances. A motif
    # has no reason to care.
    #
    # DISTANCE TO THE BOUNDARY AND DISTANCE TO THE ANCHOR ARE NOT INDEPENDENT and
    # the first version of this table conflated them. min(p - w_lo, w_hi - p) is
    # maximised at the window's centre -- which IS the anchor -- so "far from any
    # boundary" and "at the stop codon" were the same bin, and the deep bin read
    # high for a reason that had nothing to do with edges. They are separated
    # here: the boundary profile excludes the anchor neighbourhood outright, and
    # the anchor profile is reported on its own.
    print("\n" + "=" * 78)
    print("Q2b Is the elevation a fill-boundary artifact?\n")
    ANCHOR_EXCL = 30
    for wname, (lo_i, hi_i), anch in (("ATG window", (0, 1), "orf_start"),
                                      ("stop window", (2, 3), "orf_end")):
        d_bins = np.arange(0, 200, 20)
        acc = np.zeros(len(d_bins) + 1); cnt = np.zeros(len(d_bins) + 1)
        a_bins = np.arange(0, 200, 20)
        aacc = np.zeros(len(a_bins) + 1); acnt = np.zeros(len(a_bins) + 1)
        for r in recs:
            eff = per_position_effect(r[key])
            ok = r["valid"] & np.isfinite(eff)
            if ok.sum() < 50:
                continue
            med = np.median(eff[ok])
            if med <= 0:
                continue
            w_lo, w_hi = int(r["spans"][r["k"]][lo_i]), int(r["spans"][r["k"]][hi_i])
            if w_hi < w_lo:
                continue
            a = r["orf_start"] if anch == "orf_start" else r["orf_end"] - 1
            pos = np.flatnonzero(ok) + 1
            inw = (pos >= w_lo) & (pos <= w_hi)
            if not inw.any():
                continue
            p, f = pos[inw], eff[ok][inw] / med
            d_edge = np.minimum(p - w_lo, w_hi - p)
            d_anch = np.abs(p - a)
            far = d_anch > ANCHOR_EXCL                   # boundary profile only
            b = np.clip(np.digitize(d_edge[far], d_bins) - 1, 0, len(d_bins))
            np.add.at(acc, b, f[far]); np.add.at(cnt, b, 1)
            ab = np.clip(np.digitize(d_anch, a_bins) - 1, 0, len(a_bins))
            np.add.at(aacc, ab, f); np.add.at(acnt, ab, 1)
        lab = [f"{d}-{d+19}" for d in d_bins] + [f">= {d_bins[-1]+20}"]
        print(f"  {wname}: fold-elevation by distance from the nearest fill "
              f"boundary,\n  excluding positions within {ANCHOR_EXCL} nt of the "
              f"anchor so the two are not confounded")
        print(f"    {'distance':>10} {'mean fold':>10} {'n':>9}")
        for i in range(len(acc)):
            if cnt[i]:
                print(f"    {lab[i]:>10} {acc[i]/cnt[i]:>10.2f} {int(cnt[i]):>9,}")
        alab = [f"{d}-{d+19}" for d in a_bins] + [f">= {a_bins[-1]+20}"]
        print(f"\n  {wname}: fold-elevation by distance from the ANCHOR, reported "
              f"separately")
        print(f"    {'distance':>10} {'mean fold':>10} {'n':>9}")
        for i in range(len(aacc)):
            if acnt[i]:
                print(f"    {alab[i]:>10} {aacc[i]/acnt[i]:>10.2f} {int(acnt[i]):>9,}")
        print()

    # ========================================== Q2c, does the peak track the
    # stop codon or the 3' fill edge?
    #
    # PETE'S QUESTION: could the downstream signal just be 3' UTR length? Two
    # mechanisms, one test. A transcript contributes at offset +150 only if its
    # window is filled that far, so large offsets are computed on a LENGTH-SELECTED
    # subset; and near the fill edge channel 5 divides its GC count by a clipped
    # denominator, so each substitution has more leverage there. Both would put
    # apparent signal at large offsets without anything being at a position.
    #
    # THE TEST. Split transcripts by where their stop window's fill actually ends,
    # and find the peak offset within each stratum.
    #
    #   peak at a fixed OFFSET across strata      -> anchored on the stop codon
    #   peak tracking each stratum's own EDGE     -> it is the edge, i.e. length
    #
    # This is the same logic that killed the period-3 explanation: a real local
    # feature does not move when you change something it should not depend on.
    print("\n" + "=" * 78)
    print("Q2c Does the downstream peak track the stop codon or the 3' fill edge?\n")
    per_tx = []
    for r in recs:
        eff = per_position_effect(r[key])
        ok = r["valid"] & np.isfinite(eff)
        if ok.sum() < 50:
            continue
        med = np.median(eff[ok])
        s_lo, s_hi = int(r["spans"][r["k"]][2]), int(r["spans"][r["k"]][3])
        if med <= 0 or s_hi < s_lo:
            continue
        a = r["orf_end"] - 1
        pos = np.flatnonzero(ok) + 1
        m = (pos >= s_lo) & (pos <= s_hi) & (pos > a)      # downstream of the stop
        if m.sum() < 30:
            continue
        per_tx.append((s_hi - a, pos[m] - a, eff[ok][m] / med))
    edges = np.array([e for e, _, _ in per_tx])
    print(f"  {len(per_tx)} transcripts; downstream fill ends at offset "
          f"{edges.min()} to {edges.max()} (median {int(np.median(edges))})")

    # EVERY STRATUM IS SEARCHED OVER THE SAME OFFSETS. The first version stratified
    # first and searched each stratum as far as its own shortest member reached,
    # so the short arm was searched over +1..+34 and the long arm over +1..+334 --
    # and their peaks were printed side by side as though that were a comparison.
    # It was not: the short arm never looked where the long arm peaked. Fixing the
    # range first and dropping the transcripts that cannot cover it costs sample
    # size and buys the only thing that makes the strata comparable.
    for COMMON in (int(np.quantile(edges, 0.5)), int(np.quantile(edges, 0.25))):
        elig = [t for t in per_tx if t[0] >= COMMON]
        if len(elig) < 12:
            continue
        el_edges = np.array([e for e, _, _ in elig])
        print(f"\n  common search range +1..+{COMMON}: {len(elig)} of "
              f"{len(per_tx)} transcripts reach it "
              f"({len(per_tx) - len(elig)} dropped, and they are the short ones)")
        if el_edges.max() == el_edges.min():
            print(f"    every eligible transcript ends at {el_edges.max()} — "
                  f"no length contrast left to stratify on")
            continue
        cuts = np.quantile(el_edges, [1 / 2])
        print(f"    {'stratum':<20} {'n':>4} {'fill ends at':>13} "
              f"{'peak offset':>12} {'peak fold':>10} {'dist to edge':>13}")
        for b, lab in enumerate(["shorter 3' fill", "longer 3' fill"]):
            sel = [t for t in elig if np.searchsorted(cuts, t[0]) == b]
            if len(sel) < 5:
                continue
            acc = np.zeros(COMMON + 1); cnt = np.zeros(COMMON + 1)
            for _, off, fold in sel:
                k_ = off <= COMMON
                np.add.at(acc, off[k_], fold[k_])
                np.add.at(cnt, off[k_], 1)
            prof = np.where(cnt >= len(sel), acc / np.maximum(cnt, 1), -np.inf)
            if not np.isfinite(prof).any():
                continue
            pk = int(np.argmax(prof))
            med_edge = int(np.median([e for e, _, _ in sel]))
            print(f"    {lab:<20} {len(sel):>4} {med_edge:>13} {pk:>+12} "
                  f"{prof[pk]:>10.2f} {med_edge - pk:>13}")
    print("\n  Both strata now search the same offsets. A peak at the same offset")
    print("  in both is anchored on the stop codon; a peak that shifts with the")
    print("  stratum's own fill extent is 3' UTR length.")

    # =========================================== Q3a, motif or composition
    #
    # THE SCREEN THAT DECIDES WHAT KIND OF THING THIS IS. Channel 5 is a rolling
    # GC fraction derived from the bases, so a substitution that changes GC status
    # moves it as well as the one-hot -- on about two thirds of substitutions. Base
    # identity and local composition are therefore confounded in the discovery pass
    # itself, and an "important position" can be important only because
    # substituting there moved the GC channel.
    #
    # The bank ships `dgc` per substitution: +1 for non-GC -> GC, -1 for the
    # reverse, 0 when GC status is unchanged. Every position has exactly ONE
    # GC-neutral alternative (A<->T or C<->G) and two GC-changing ones, so the
    # comparison is available at every position without matching anything.
    #
    #   GC-neutral effect ~ GC-changing effect   the model reads base IDENTITY,
    #                                            which is what a motif is
    #   GC-neutral effect << GC-changing         the model reads COMPOSITION and
    #                                            the position is incidental
    #
    # This is not the anagram control and does not replace it -- anagrams hold
    # composition exactly while reordering, and they need forward passes the bank
    # does not contain. This is the cheap screen that says whether spending them
    # is worth it.
    print("=" * 78)
    print("Q3a Is the signal base identity, or local GC?\n")
    stat = {k2: {"neutral": [], "changing": []} for k2 in ("elevated", "all")}
    ratio_tx = []
    for r in recs:
        eff = per_position_effect(r[key])
        ok = r["valid"] & np.isfinite(eff)
        if ok.sum() < 50:
            continue
        med = np.median(eff[ok])
        if med <= 0:
            continue
        hi = ok & (eff >= args.fold * med)
        v, g = np.abs(r[key]), r["dgc"]
        neutral = (g == 0) & np.isfinite(v)
        changing = (g != 0) & np.isfinite(v)
        for grp, rows in (("elevated", hi), ("all", ok)):
            n_ = v[rows & neutral.any(1)][neutral[rows & neutral.any(1)]]
            c_ = v[rows & changing.any(1)][changing[rows & changing.any(1)]]
            if len(n_):
                stat[grp]["neutral"].append(n_)
            if len(c_):
                stat[grp]["changing"].append(c_)
        if hi.sum() >= 5:
            n_ = v[hi & neutral.any(1)][neutral[hi & neutral.any(1)]]
            c_ = v[hi & changing.any(1)][changing[hi & changing.any(1)]]
            if len(n_) and len(c_) and np.median(c_) > 0:
                ratio_tx.append(np.median(n_) / np.median(c_))
    print(f"  {'positions':<12} {'GC-neutral':>13} {'GC-changing':>13} "
          f"{'neutral/changing':>18}")
    for grp in ("all", "elevated"):
        if not stat[grp]["neutral"] or not stat[grp]["changing"]:
            continue
        n_ = np.concatenate(stat[grp]["neutral"])
        c_ = np.concatenate(stat[grp]["changing"])
        print(f"  {grp:<12} {np.median(n_):>13.3e} {np.median(c_):>13.3e} "
              f"{np.median(n_)/np.median(c_):>18.2f}")
    if ratio_tx:
        ratio_tx = np.array(ratio_tx)
        print(f"\n  per transcript, at elevated positions only "
              f"({len(ratio_tx)} transcripts):")
        print(f"    median ratio {np.median(ratio_tx):.2f}   "
              f"quartiles {np.percentile(ratio_tx, 25):.2f} to "
              f"{np.percentile(ratio_tx, 75):.2f}")
        print(f"    transcripts where the GC-neutral substitution is at least "
              f"half the GC-changing one: "
              f"{100*(ratio_tx >= 0.5).mean():.0f}%")

    # ================================================================== Q3
    print("\n" + "=" * 78)
    print("Q3  Do the elevated positions carry a sequence signature?\n")
    base_hi = np.zeros(4)
    base_all = np.zeros(4)
    ctx = []
    for r in recs:
        eff = per_position_effect(r[key])
        ok = r["valid"] & np.isfinite(eff)
        if ok.sum() < 50:
            continue
        med = np.median(eff[ok])
        hi = ok & (eff >= args.fold * med)
        o = r["obs"]
        for b in range(4):
            base_hi[b] += int(((o == b) & hi).sum())
            base_all[b] += int(((o == b) & ok).sum())
        for p in np.flatnonzero(hi):
            w = o[max(0, p - 4):p + 5]
            if len(w) == 9 and (w >= 0).all():
                ctx.append("".join(NT[i] for i in w))
    print(f"  base at elevated positions, against all measured positions")
    print(f"    {'base':<6} {'elevated':>10} {'all':>10} {'enrichment':>12}")
    for b in range(4):
        pe = base_hi[b] / max(base_hi.sum(), 1)
        pa = base_all[b] / max(base_all.sum(), 1)
        print(f"    {NT[b]:<6} {100*pe:>9.1f}% {100*pa:>9.1f}% "
              f"{pe/pa if pa else np.nan:>11.2f}x")
    print(f"\n  {len(ctx):,} elevated positions with a full 9-mer context")
    if ctx:
        s = pd.Series(ctx).value_counts()
        print(f"  most frequent contexts (centre base is the elevated one):")
        for k9, n in s.head(8).items():
            print(f"    {k9[:4]} [{k9[4]}] {k9[5:]}   {n}")
        print(f"    distinct 9-mers {s.size:,} over {len(ctx):,} sites — "
              f"{'no repetition, so no motif at this sample size' if s.max() < 3 else 'some repetition, worth a larger sample'}")


if __name__ == "__main__":
    main()
