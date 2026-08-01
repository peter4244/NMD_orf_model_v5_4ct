#!/usr/bin/env python
"""
EXPERIMENT 2c -- is my "six-fold larger in PTC+" a mechanism, or is it the scale?

WHAT I CLAIMED
  After the retraction, the argument for the stop-codon effect rested partly on
  a gradient: TGA - TAG is +4.36pp where the transcript has a downstream
  junction and +1.05pp where it does not. I wrote that this is "six-fold larger
  where a downstream junction exists for termination efficiency to act on" and
  that "matching cannot manufacture it".

TRACK A'S OBJECTION
  Matching didn't manufacture it -- the SCALE did. Percentage-point differences
  compress as the base rate approaches 0 or 1. The PTC+ arm sits near 70% NMD
  and the PTC- arm near 5%. The same MULTIPLICATIVE effect must produce a much
  smaller pp difference in the second. They report odds ratios of 1.21 and 1.26
  on their anchor -- the pp gradient is 3.7-fold and the odds gradient is
  absent, with PTC- nominally larger.

  Checked here on my own matched population rather than accepted, and with the
  arithmetic demonstrated from scratch rather than argued.

THREE PARTS
  A  the arithmetic, from first principles: what pp difference does a FIXED
     odds ratio produce at each base rate? If it reproduces 4.4 and 1.0, the
     gradient carries no information about mechanism.
  B  the measurement: pp difference AND Mantel-Haenszel odds ratio in each
     subgroup, on the matched strata, gene-clustered.
  C  the homogeneity test the mechanism claim actually needs: is the odds ratio
     DIFFERENT between PTC+ and PTC-?

WHY THE ANSWER MATTERS BEYOND BOOKKEEPING
  If the effect is the same on the odds scale in both strata, then it is NOT
  specific to transcripts with a downstream junction. A termination-efficiency
  mechanism acting through EJC-dependent decay predicts the effect should be
  concentrated where an EJC exists -- ideally absent without one. Equal odds
  ratios in both strata is evidence against that specific story, whatever it
  says about the association.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp2c_scale_check.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast_lib import boot_diff, code_strata, describe, std_diff

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260801)


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(df["isoform_id"], df["junctions"])}


def load_seqs():
    z = np.load(os.path.join(HERE, "seq_store.npz"), allow_pickle=False)
    return z["blob"], z["offsets"], {s: i for i, s in enumerate(z["ids"])}


def sub(blob, off, i, a, b):
    lo, hi = int(off[i]), int(off[i + 1])
    a0, b0 = lo + a - 1, lo + b - 1
    if a0 < lo or b0 > hi or b0 <= a0:
        return ""
    return blob[a0:b0].tobytes().decode("ascii")


def pp_from_or(p_ref, odds_ratio):
    """pp difference produced by a fixed odds ratio at reference rate p_ref."""
    o = p_ref / (1 - p_ref)
    return ((o * odds_ratio) / (1 + o * odds_ratio) - p_ref) * 100


def main():
    print("=" * 98)
    print("EXPERIMENT 2c -- mechanism or scale?")
    print("=" * 98)

    print("\n" + "=" * 98)
    print("A. THE ARITHMETIC, BEFORE ANY DATA")
    print("=" * 98)
    print("\n  What percentage-point difference does a FIXED odds ratio produce,")
    print("  at the base rates of my two subgroups?\n")
    print(f"  {'odds ratio':<12} {'at TAG rate 66.5% (PTC+)':>26} "
          f"{'at TAG rate 4.4% (PTC-)':>26} {'pp ratio':>10}")
    print(f"  {'-'*12} {'-'*26} {'-'*26} {'-'*10}")
    for orr in (1.15, 1.20, 1.25, 1.30, 1.40):
        hi = pp_from_or(0.665, orr)
        lo = pp_from_or(0.044, orr)
        print(f"  {orr:<12.2f} {hi:>25.2f}p {lo:>25.2f}p {hi/lo:>10.1f}x")
    print("\n  A single constant odds ratio, applied to both subgroups with NO")
    print("  difference between them, generates a 4-5x gradient in percentage")
    print("  points. My 'six-fold' claim is inside that range. Track A is right")
    print("  that the pp gradient on its own carries no mechanistic information.")

    # ------------------------------------------------------------------ data
    blob, off, idx = load_seqs()
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id"]).drop_duplicates("isoform_id")
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t")
    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.drop(columns=[c for c in ("tx_length",) if c in d.columns])
    d = d.merge(tx, on="isoform_id").merge(ref, on="isoform_id", how="left")
    d["_i"] = d["isoform_id"].map(idx)
    d = d[d["_i"].notna() & d["gene_id"].notna()].copy()
    d["_i"] = d["_i"].astype(int)
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d["ptc"] = [int(len(junc.get(i, np.empty(0, dtype=np.int64)))
                    - int(np.searchsorted(junc.get(i, np.empty(0, dtype=np.int64)),
                                          int(e) + 50, side="right")) > 0)
                for i, e in zip(d["isoform_id"], d["orf_end"])]
    d = d[d["utr3"] >= 60].copy()
    d["gc100"] = [(lambda t: (t.count("G") + t.count("C")) / len(t) if t else np.nan)(
        sub(blob, off, i, e + 1, e + 101)) for i, e in zip(d["_i"], d["orf_end"])]
    d = d[d["gc100"].notna()].copy()
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop").astype(int)
    d["gc_q"] = pd.qcut(d["gc100"], 4, labels=False, duplicates="drop").astype(int)

    print("\n" + "=" * 98)
    print("B. THE MEASUREMENT ON MY MATCHED STRATA -- both scales, side by side")
    print("=" * 98)
    res = {}
    for v, nm in ((1, "PTC+"), (0, "PTC-")):
        g = d[d["ptc"].eq(v)]
        r = boot_diff(g, "stop_codon", "TGA", "TAG", ["utr3_q", "gc_q"],
                      "is_nmd", "gene_id", n=1500)
        res[nm] = r
        describe(r, f"\n  {nm}  (n = {len(g):,})")

    pp_ratio = res["PTC+"]["diff_common"] / res["PTC-"]["diff_common"]
    or_ratio = res["PTC+"]["mh_or"] / res["PTC-"]["mh_or"]
    print(f"\n  {'':<22} {'PTC+':>10} {'PTC-':>10} {'ratio':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'percentage points':<22} {res['PTC+']['diff_common']:>+9.2f}p "
          f"{res['PTC-']['diff_common']:>+9.2f}p {pp_ratio:>9.1f}x")
    print(f"  {'odds ratio':<22} {res['PTC+']['mh_or']:>10.2f} "
          f"{res['PTC-']['mh_or']:>10.2f} {or_ratio:>9.2f}x")
    print(f"  {'base rate':<22} {res['PTC+']['base_rate']:>9.1f}% "
          f"{res['PTC-']['base_rate']:>9.1f}%")

    print("\n  Predicted pp difference if BOTH subgroups had PTC+'s odds ratio:")
    o_hi = res["PTC+"]["mh_or"]
    pred = pp_from_or(res["PTC-"]["rate0"] / 100, o_hi)
    print(f"    PTC- would show {pred:+.2f}pp; it shows "
          f"{res['PTC-']['diff_common']:+.2f}pp.")

    print("\n" + "=" * 98)
    print("C. THE TEST THE MECHANISM CLAIM ACTUALLY NEEDS")
    print("   Is the ODDS RATIO different between PTC+ and PTC-?")
    print("=" * 98)
    a, b = res["PTC+"].get("or_draws"), res["PTC-"].get("or_draws")
    if a is not None and b is not None and len(a) > 50 and len(b) > 50:
        k = min(len(a), len(b))
        lr = np.log(a[:k]) - np.log(b[:k])
        lo, hi = np.percentile(lr, [2.5, 97.5])
        p = 2 * min((lr <= 0).mean(), (lr >= 0).mean())
        print(f"\n  log odds-ratio difference (PTC+ minus PTC-):")
        print(f"    {lr.mean():+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
              f"p = {max(p, 1/len(lr)):.3f}")
        print(f"    ratio of odds ratios {np.exp(lr.mean()):.2f} "
              f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}]")
        print("\n  The two bootstrap series are independent draws over different")
        print("  gene sets, so this interval is approximate. It is here to show")
        print("  the order of magnitude, not as a formal interaction test.")
        if lo < 0 < hi:
            print("\n  >> The interval spans 1. There is NO evidence the effect")
            print("     differs between PTC+ and PTC- on the odds scale.")

    print("\n" + "=" * 98)
    print("D. WHAT THIS DOES TO THE CLAIM")
    print("=" * 98)
    print("""
  WITHDRAWN: "six-fold larger where a downstream junction exists for
  termination efficiency to act on -- matching cannot manufacture it."
  Matching didn't. The scale did, and section A shows a constant odds ratio
  reproduces a 4-5x pp gradient with no group difference at all.

  WHAT SURVIVES: the direction and the overall effect. TGA carries more decay
  than TAG at matched composition, PTC status, and 3'UTR length.

  WHAT CHANGES MECHANISTICALLY, and it is not a small change: on the odds
  scale the effect is present in BOTH strata. A termination-efficiency
  mechanism acting through EJC-dependent decay predicts concentration where an
  EJC exists. An effect of similar multiplicative size in transcripts WITHOUT a
  downstream junction is not that -- it points at something common to both,
  which could be EJC-independent decay, or a compositional property of TGA-
  ending transcripts that PTC status does not capture.

  So the association stands and the mechanism story I attached to it does not.
  That is the second time today I have reached for a mechanism ahead of the
  evidence for it -- the first was the readthrough prediction, which came out
  backwards.
""")

    print("=" * 98)
    print("DONE")
    print("=" * 98)


if __name__ == "__main__":
    main()
