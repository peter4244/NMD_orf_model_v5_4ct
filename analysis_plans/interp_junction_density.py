#!/usr/bin/env python
"""
interp_junction_density.py — is the regression-discontinuity design available?

STEP 2, AND IT IS A GATE. The proposed claim is a discontinuity in the model's
decay probability at 500 nt downstream of the stop -- the stop window's downstream
half-width, where a junction stops being visible positionally in channel 4 and
becomes only +1 in `n_downstream_ejc`. An RD design lives or dies on how much data
sits near the cutoff. Measuring that BEFORE writing the row is the step this project
has walked past for two days.

WHAT WOULD MAKE THE DESIGN UNAVAILABLE, stated before the numbers:
  - too few junctions within a usable bandwidth of 500
  - the RD sample (transcripts extending past stop+500, so the window is not
    fill-limited) being a small or peculiar subset
  - a density discontinuity at 500 itself, which would break the design's own
    assumption -- biology has no reason to produce one, so if it appears it is
    evidence of something upstream and the design is off.

NO MODEL, NO BANK, NO CLUSTER. Annotation only: junctions.tsv, orf_pool.tsv,
ism_subset.tsv. This is a claim about our DATA's geometry, not about the model.

CONVENTION: junction positions are 1-based (data_prep.py:81). `main_orf_stop` and
`orf_end` are taken on the same 1-based transcript coordinate. Distances are
junction - stop, so positive means downstream of the stop.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path.home() / "claude_projects/nmd_w69_tables_2026-07-30"
REPO = Path(__file__).resolve().parent.parent
CUTOFF = 500                      # stop window downstream half-width
BANDS = [(25, "±25"), (50, "±50"), (100, "±100"), (200, "±200")]


def main():
    sub = pd.read_csv(REPO / "results_ism_v6/ism_subset.tsv", sep="\t")
    junc = pd.read_csv(TABLES / "junctions.tsv", sep="\t", quotechar='"')
    junc.columns = [c.strip('"') for c in junc.columns]
    jmap = {r.isoform_id: r.junctions for r in junc.itertuples()
            if isinstance(r.junctions, str) and r.junctions}

    print(f"ISM subset            {len(sub):,} transcripts")
    print(f"junctions.tsv         {len(jmap):,} transcripts with >=1 junction")
    print(f"cutoff                {CUTOFF} nt downstream of the stop "
          f"(stop window half-width)\n")

    have_stop = sub["main_orf_stop"].notna()
    print(f"  with an annotation-derived stop (main_orf_stop)   "
          f"{int(have_stop.sum()):,}  ({have_stop.mean():.1%})")

    rows, no_junc = [], 0
    for r in sub[have_stop].itertuples():
        js = jmap.get(r.isoform_id)
        if not js:
            no_junc += 1
            continue
        stop = float(r.main_orf_stop)
        for j in (int(x) for x in js.split(",") if x):
            d = j - stop
            if d > 0:
                rows.append((r.isoform_id, d, float(r.tx_length), int(r.is_nmd)))
    D = pd.DataFrame(rows, columns=["isoform_id", "dist", "tx_length", "is_nmd"])
    print(f"  ...of those, with no junction record                 {no_junc:,}")
    print(f"  downstream junctions total                       {len(D):,}"
          f"   over {D.isoform_id.nunique():,} transcripts\n")

    # ---- THE RD SAMPLE. A window is only bounded at 500 if the transcript
    # actually extends that far; otherwise the boundary is the transcript end and
    # the running variable means something different. This selects on 3'UTR
    # length, which is NMD-associated -- it applies equally on BOTH sides of the
    # cutoff so it bounds generalization rather than biasing the contrast, but it
    # is a selection and it is reported rather than adjusted.
    reach = D.tx_length >= (D.dist + 0)          # placeholder, refined below
    stop_plus = D.groupby("isoform_id").tx_length.first()
    print("  RD SAMPLE — transcripts whose 3' end extends past stop+500,")
    print("  so the 500 boundary is architectural rather than fill-limited:")
    sub_ok = sub[have_stop].copy()
    sub_ok["reaches"] = sub_ok["tx_length"] >= (sub_ok["main_orf_stop"] + CUTOFF)
    print(f"    reaches stop+{CUTOFF}   {int(sub_ok.reaches.sum()):,}"
          f"  ({sub_ok.reaches.mean():.1%} of annotated)")
    print(f"    NMD fraction, reaching {sub_ok[sub_ok.reaches].is_nmd.mean():.3f}"
          f"   against not-reaching {sub_ok[~sub_ok.reaches].is_nmd.mean():.3f}")
    keep = set(sub_ok[sub_ok.reaches].isoform_id)
    R = D[D.isoform_id.isin(keep)]
    print(f"    downstream junctions in the RD sample  {len(R):,}\n")

    # ---- DENSITY NEAR THE CUTOFF, which is the gate itself.
    print(f"  JUNCTIONS BY DISTANCE FROM THE STOP (RD sample, n={len(R):,})")
    print(f"    {'band':>8} {'below 500':>10} {'above 500':>10} {'total':>8}")
    for h, name in BANDS:
        lo = R[(R.dist >= CUTOFF - h) & (R.dist < CUTOFF)]
        hi = R[(R.dist >= CUTOFF) & (R.dist < CUTOFF + h)]
        print(f"    {name:>8} {len(lo):>10,} {len(hi):>10,} {len(lo)+len(hi):>8,}")

    # ---- IS THE DENSITY ITSELF SMOOTH AT 500? If it jumps, the design's own
    # assumption fails and any discontinuity in the outcome is unreadable.
    print(f"\n  DENSITY SMOOTHNESS AT THE CUTOFF (the design's own assumption)")
    w = 50
    edges = np.arange(CUTOFF - 250, CUTOFF + 251, w)
    cnt, _ = np.histogram(R.dist, bins=edges)
    print(f"    counts in {w}-nt bins from {edges[0]} to {edges[-1]}:")
    print("      " + "  ".join(f"{c:,}" for c in cnt))
    left = cnt[len(cnt) // 2 - 1]
    right = cnt[len(cnt) // 2]
    ratio = right / left if left else float("nan")
    print(f"    bin just below 500: {left:,}   just above: {right:,}"
          f"   ratio {ratio:.3f}")
    print("    (a ratio far from 1.0 means the running variable is itself")
    print("     discontinuous at the cutoff and the design is not usable)")

    print(f"\n  VERDICT INPUTS — decide the design against these, not against a")
    print(f"  hoped-for number. Reported before any outcome is looked at.")


if __name__ == "__main__":
    main()
