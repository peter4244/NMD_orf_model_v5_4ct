#!/usr/bin/env python
"""
EXPERIMENT 12 -- what is it about the front of the transcript?

THE LEAD, AND WHY IT IS THE ONLY ONE LEFT WORTH CHASING
  Every candidate at the back of the transcript has now died or shrunk to
  nothing: three 3'UTR motifs were composition, the +4 base failed against its
  own control, and the stop codon survives at odds ratio ~1.2 with no mechanism
  that holds up. Meanwhile, measured in passing and never followed:

      5'UTR length, long vs short        odds ratio 2.82
      upstream AUG count, >=4 vs <=1     odds ratio 2.42
      stop codon TGA vs TAG              odds ratio 1.22
      any 3'UTR motif                    odds ratio ~1.0

  Roughly ten to one, front against back.

THE QUESTION THIS ANSWERS
  "The 5'UTR is long" is not a sequence element -- you cannot point at it in a
  transcript and say that is the thing. So the point of this is to find out
  WHICH part of the front carries the signal, and whether any of it is
  nameable:

    is it LENGTH, or is it CONTENT? Longer 5'UTRs hold more upstream AUGs, so
      the two are confounded by construction and have to be separated.
    is it COUNT of upstream start codons, or their STRENGTH (Kozak context)?
    does it matter WHERE the upstream ORF ends -- inside the 5'UTR, where the
      ribosome can resume scanning, or overlapping the coding sequence out of
      frame, where it cannot?

  The last one is the only one of the three that is a mechanism rather than a
  correlate, and it is the one Pete's uORF interest points at.

POPULATION
  Transcripts whose MAIN coding sequence has a clean stop -- no junction beyond
  stop+50. In transcripts that already carry a junction trigger, that trigger
  dominates and anything at the front is noise on top of it. This is the
  population where the front of the transcript is the candidate cause.

Everything is reported on both the percentage-point and odds scales, because a
percentage-point difference cannot be compared across groups with different
base rates -- established the hard way three times today.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp12_front_of_transcript.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast_lib import boot_diff

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(df["isoform_id"], df["junctions"])}


def n_beyond(j, pos):
    return len(j) - int(np.searchsorted(j, pos, side="right"))


def cell(g, label="is_nmd", lo=25):
    if len(g) < lo:
        return f"{len(g):>5,} {'--':>8}"
    return f"{len(g):>5,} {g[label].mean()*100:>7.1f}%"


def main():
    print("=" * 100)
    print("EXPERIMENT 12 -- the front of the transcript")
    print("=" * 100)

    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t")
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_start", "orf_end", "is_ref_cds"])

    slot = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id")
    d = ref.drop_duplicates("isoform_id").merge(slot, on="isoform_id", how="inner")
    d = d.merge(tx, on="isoform_id", how="inner")
    d = d[d["ref_atg_available"].eq(1) & d["ref_utr5_length"].notna()].copy()

    d["main_ptc"] = [int(n_beyond(junc.get(i, np.empty(0, dtype=np.int64)),
                                  int(e) + 50) > 0)
                     for i, e in zip(d["isoform_id"], d["orf_end"])]
    clean = d[d["main_ptc"].eq(0)].copy()
    print(f"\n  {len(d):,} isoforms with a projected reference start codon")
    print(f"  {len(clean):,} of those have a CLEAN main stop -- no junction beyond")
    print(f"  stop+50. That is the population where the front can be the cause.")
    print(f"  NMD+ in it: {clean['is_nmd'].mean()*100:.1f}%")

    clean["u5"] = clean["ref_utr5_length"].astype(int)
    clean = clean[clean["u5"] >= 30].copy()
    clean["u5_q"] = pd.qcut(clean["u5"], 4, labels=False,
                            duplicates="drop").astype(int)
    clean["n_atg"] = clean["ref_atg_count"].fillna(0).astype(int)
    clean["n_strong"] = clean["ref_atg_strong_kozak"].fillna(0).astype(int)
    clean["n_oorf"] = clean["ref_uorf_count_outframe"].fillna(0).astype(int)
    clean["n_uorf5"] = (clean["ref_uorf_count_overlapping"].fillna(0)
                        - clean["ref_uorf_count_outframe"].fillna(0)).clip(lower=0)
    clean["cov"] = clean["ref_utr5_orf_coverage"].fillna(0)

    # ------------------------------------------------------------------- 1
    print("\n" + "=" * 100)
    print("1. THE TWO RAW SIGNALS")
    print("=" * 100)
    print(f"\n  {'5UTR length quartile':<24} {'n':>7} {'median nt':>10} {'NMD+':>8}")
    print(f"  {'-'*24} {'-'*7} {'-'*10} {'-'*8}")
    for q in range(4):
        g = clean[clean["u5_q"].eq(q)]
        print(f"  Q{q+1:<23} {len(g):>7,} {g['u5'].median():>10,.0f} "
              f"{g['is_nmd'].mean()*100:>7.1f}%")
    print(f"\n  {'upstream AUG count':<24} {'n':>7} {'median 5UTR':>12} {'NMD+':>8}")
    print(f"  {'-'*24} {'-'*7} {'-'*12} {'-'*8}")
    for k in [0, 1, 2, 3]:
        g = clean[clean["n_atg"].eq(k)]
        if len(g) >= 25:
            print(f"  {k:<24} {len(g):>7,} {g['u5'].median():>12,.0f} "
                  f"{g['is_nmd'].mean()*100:>7.1f}%")
    g = clean[clean["n_atg"] >= 4]
    print(f"  {'>=4':<24} {len(g):>7,} {g['u5'].median():>12,.0f} "
          f"{g['is_nmd'].mean()*100:>7.1f}%")
    print("\n  Note the median-5'UTR column: the two are confounded by")
    print("  construction, because a longer 5'UTR has more room for AUGs.")

    # ------------------------------------------------------------------- 2
    print("\n" + "=" * 100)
    print("2. LENGTH OR CONTENT? -- the two crossed")
    print("=" * 100)
    print(f"\n  {'':<12}" + "".join(f"{'5UTR Q'+str(q+1):>15}" for q in range(4)))
    print(f"  {'-'*12}" + "".join(f"{'-'*15:>15}" for _ in range(4)))
    for k, lab in [(0, "0 uAUG"), (1, "1"), (2, "2"), (3, "3"), (4, ">=4")]:
        g = clean[clean["n_atg"].eq(k)] if k < 4 else clean[clean["n_atg"] >= 4]
        print(f"  {lab:<12}" + "".join(
            f"{cell(g[g['u5_q'].eq(q)]):>15}" for q in range(4)))
    print("""
  Read DOWN a column: 5'UTR length held roughly fixed, upstream AUGs varying.
  Read ACROSS a row: upstream AUGs held fixed, length varying.
  Whichever direction carries the gradient is the one that matters.""")

    print("\n  Same thing as adjusted contrasts, gene-clustered, both scales:")
    clean["c_len"] = np.where(clean["u5_q"].eq(3), "long",
                              np.where(clean["u5_q"].eq(0), "short", None))
    clean["c_atg"] = np.where(clean["n_atg"] >= 4, "many",
                              np.where(clean["n_atg"] <= 1, "few", None))
    for nm, col, a, b, adj in (
            ("5'UTR length long vs short, adjusted for uAUG count",
             "c_len", "long", "short", ["n_atg_b"]),
            ("uAUG count >=4 vs <=1, adjusted for 5'UTR length",
             "c_atg", "many", "few", ["u5_q"])):
        clean["n_atg_b"] = np.clip(clean["n_atg"], 0, 5)
        g = clean[clean[col].isin([a, b])]
        if len(g) < 300:
            continue
        r = boot_diff(g, col, a, b, adj, "is_nmd", "gene_id", n=800)
        orr = (f"{r['mh_or']:.2f} [{r.get('or_lo', float('nan')):.2f}, "
               f"{r.get('or_hi', float('nan')):.2f}]")
        print(f"    {nm}")
        print(f"      {r['diff_common']:+.2f}pp [{r['lo']:+.2f}, {r['hi']:+.2f}]"
              f"   odds ratio {orr}   n {r['n1']:,} vs {r['n0']:,}")

    # ------------------------------------------------------------------- 3
    print("\n" + "=" * 100)
    print("3. COUNT OR STRENGTH? -- Kozak context at fixed upstream AUG count")
    print("=" * 100)
    print("\n  A strong-Kozak start codon captures more scanning ribosomes. If")
    print("  strength matters at fixed count, the element is the CONTEXT and not")
    print("  merely the presence of an AUG.\n")
    print(f"  {'uAUG count':<12}" + "".join(
        f"{'strong=' + str(s):>15}" for s in range(4)))
    print(f"  {'-'*12}" + "".join(f"{'-'*15:>15}" for _ in range(4)))
    for k in [1, 2, 3, 4]:
        g = clean[clean["n_atg"].eq(k)] if k < 4 else clean[clean["n_atg"] >= 4]
        print(f"  {k if k < 4 else '>=4':<12}" + "".join(
            f"{cell(g[g['n_strong'].eq(s)]):>15}" for s in range(4)))

    # ------------------------------------------------------------------- 4
    print("\n" + "=" * 100)
    print("4. WHERE DOES THE UPSTREAM ORF END? -- the only mechanism here")
    print("=" * 100)
    print("""
  An upstream ORF that stops inside the 5'UTR lets the ribosome resume scanning
  and start again at the real start codon -- a leaky trigger. One that reads
  PAST the real start codon out of frame and stops inside the coding sequence
  has no such rescue. Mechanistically the second should be much the more
  potent, and it is the one distinction here that is a sequence ARRANGEMENT
  rather than a count.""")
    clean["uclass"] = np.where(clean["n_oorf"] > 0, "oORF (overlaps CDS)",
                               np.where(clean["n_uorf5"] > 0,
                                        "uORF (ends in 5'UTR)", "no upstream ORF"))
    print(f"\n  {'class':<24} {'n':>8} {'NMD+':>8} {'median 5UTR':>13}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*13}")
    for c in ["no upstream ORF", "uORF (ends in 5'UTR)", "oORF (overlaps CDS)"]:
        g = clean[clean["uclass"].eq(c)]
        print(f"  {c:<24} {len(g):>8,} {g['is_nmd'].mean()*100:>7.1f}% "
              f"{g['u5'].median():>13,.0f}")
    print("\n  and with 5'UTR length and upstream AUG count held fixed:")
    clean["c_class"] = np.where(clean["n_oorf"] > 0, "oORF",
                                np.where(clean["n_uorf5"] > 0, "uORF5", None))
    g = clean[clean["c_class"].isin(["oORF", "uORF5"])]
    if len(g) >= 300:
        r = boot_diff(g, "c_class", "oORF", "uORF5", ["u5_q", "n_atg_b"],
                      "is_nmd", "gene_id", n=800)
        print(f"    oORF minus uORF-in-5'UTR: {r['diff_common']:+.2f}pp "
              f"[{r['lo']:+.2f}, {r['hi']:+.2f}]   odds ratio {r['mh_or']:.2f} "
              f"[{r.get('or_lo', float('nan')):.2f}, "
              f"{r.get('or_hi', float('nan')):.2f}]")
        print(f"    n {r['n1']:,} oORF vs {r['n0']:,} uORF")

    # ------------------------------------------------------------------- 5
    print("\n" + "=" * 100)
    print("5. EVERYTHING SIDE BY SIDE, ONE SCALE")
    print("=" * 100)
    print("\n  odds ratios in the clean-main-stop population, each adjusted for")
    print("  the others where they are confounded:\n")
    tests = [
        ("5'UTR length, top vs bottom quartile", "c_len", "long", "short",
         ["n_atg_b"]),
        ("upstream AUG count, >=4 vs <=1", "c_atg", "many", "few", ["u5_q"]),
        ("oORF vs uORF ending in 5'UTR", "c_class", "oORF", "uORF5",
         ["u5_q", "n_atg_b"]),
    ]
    clean["c_cov"] = np.where(clean["cov"] > 0.5, "high",
                              np.where(clean["cov"].eq(0), "zero", None))
    tests.append(("5'UTR covered >50% by ORFs vs 0%", "c_cov", "high", "zero",
                  ["u5_q"]))
    print(f"  {'contrast':<42} {'odds ratio':>22} {'pp':>10} {'n':>16}")
    print(f"  {'-'*42} {'-'*22} {'-'*10} {'-'*16}")
    for nm, col, a, b, adj in tests:
        g = clean[clean[col].isin([a, b])]
        if len(g) < 300:
            print(f"  {nm:<42} too few")
            continue
        r = boot_diff(g, col, a, b, adj, "is_nmd", "gene_id", n=800)
        orr = (f"{r['mh_or']:.2f} [{r.get('or_lo', float('nan')):.2f},"
               f"{r.get('or_hi', float('nan')):.2f}]")
        print(f"  {nm:<42} {orr:>22} {r['diff_common']:>+9.2f}p "
              f"{r['n1']:>7,}/{r['n0']:<8,}")
    print("\n  for comparison, from the back of the transcript:")
    print(f"  {'stop codon TGA vs TAG':<42} {'1.24 [1.06,1.47]':>22}")
    print(f"  {'every 3UTR motif tested':<42} {'~1.0':>22}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
