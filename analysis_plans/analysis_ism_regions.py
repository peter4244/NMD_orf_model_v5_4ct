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
                        spans=spans,
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
    print("\n" + "=" * 78)
    print("Q2  Where are they, relative to the start codon and the stop codon?\n")
    for anchor, lo, hi_, left in (("start codon", -900, 99, ATG_LEFT),
                                  ("stop codon", -500, 499, STOP_LEFT)):
        prof = np.zeros(hi_ - lo + 1)
        prof_n = np.zeros(hi_ - lo + 1)
        shuf = np.zeros(hi_ - lo + 1)
        for r in recs:
            eff = per_position_effect(r[key])
            ok = r["valid"] & np.isfinite(eff)
            if ok.sum() < 50:
                continue
            med = np.median(eff[ok])
            if med <= 0:
                continue
            a = r["orf_start"] if anchor == "start codon" else r["orf_end"] - 1
            pos = np.flatnonzero(ok) + 1                 # 1-based transcript pos
            off = pos - a
            keep = (off >= lo) & (off <= hi_)
            if not keep.any():
                continue
            fold = eff[ok][keep] / med
            np.add.at(prof, off[keep] - lo, fold)
            np.add.at(prof_n, off[keep] - lo, 1)
            # circular-shift null: rotate the track against its own anchor
            roll = rng.integers(1, ok.sum())
            fold_s = np.roll(eff[ok], roll)[keep] / med
            np.add.at(shuf, off[keep] - lo, fold_s)
        m = prof_n > 0
        mean_fold = np.where(m, prof / np.maximum(prof_n, 1), np.nan)
        mean_shuf = np.where(m, shuf / np.maximum(prof_n, 1), np.nan)
        order = np.argsort(-np.nan_to_num(mean_fold))
        print(f"  relative to the {anchor} — top 12 offsets by mean fold-elevation")
        print(f"    {'offset':>7} {'mean fold':>10} {'shifted':>9} "
              f"{'n tx':>6}   note")
        for i in order[:12]:
            o = i + lo
            note = ""
            if anchor == "start codon" and o >= LAST_BIN_LO:
                note = "FINAL POOLING BIN — flagged, not reported"
            if abs(o) <= 2:
                note = "the codon itself"
            print(f"    {o:>+7} {mean_fold[i]:>10.2f} {mean_shuf[i]:>9.2f} "
                  f"{int(prof_n[i]):>6}   {note}")
        good = m & ~((np.arange(len(m)) + lo >= LAST_BIN_LO)
                     if anchor == "start codon" else np.zeros(len(m), bool))
        if good.any():
            ex = np.nanmax(mean_fold[good]) - np.nanmax(mean_shuf[good])
            print(f"    best offset outside the flagged bin exceeds the shifted "
                  f"null by {ex:+.2f} fold\n")

    # ================================================================== Q3
    print("=" * 78)
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
