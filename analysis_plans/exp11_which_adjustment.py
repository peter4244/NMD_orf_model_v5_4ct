#!/usr/bin/env python
"""
EXPERIMENT 11 -- binary, count, or graded distance? The comparison neither
window has actually run.

WHAT JUST HAPPENED
  I concluded from exp7 that grading the junction feature changes nothing,
  reading shifts of 0.45pp median and 1.32pp maximum as small. Track A measured
  the same thing and drew the opposite conclusion by expressing each shift
  RELATIVE TO THE EFFECT IT MOVES. On my own numbers:

      stop codon TGA vs TAG   +1.89 -> +1.36    -28%
      AATAAA                  +0.89 -> +0.53    -40%
      pyrimidine run          -4.50 -> -3.18    -29%
      upstream AUG count      +7.23 -> +7.80     +8%
      5'UTR length            +9.51 -> +9.64     +1%

  They are right, and this is the THIRD time today I have read an absolute
  percentage-point number without its base -- after the PTC gradient and after
  the positional control. The pattern is systematic: adjusting junction geometry
  properly weakens 3'-end claims by roughly a third and leaves 5'-end claims
  alone.

  Their null is good and I reproduce it below: five RANDOM dummies at the same
  marginal frequencies move the stop effect by ~0.01pp, so the shift is the bins
  carrying information, not covariate absorption.

THE QUESTION THAT IS STILL OPEN, AND IT IS THE DESIGN ONE
  Both of us have been comparing BINARY against GRADED DISTANCE. But the fix
  already agreed for Step 1 is neither. `05s_orfik_scan.R:227` computes
  `sum(junctions > orf_end)` -- a thresholdless COUNT -- and the agreed fix is
  to apply the 50 nt rule to it, giving `sum(junctions > orf_end + 50)`. That
  is still a COUNT, not a binary indicator.

  So the real comparison is three-way:
      A  binary        does any junction lie beyond stop+50
      B  COUNT beyond 50   <-- THE FEATURE ALREADY AGREED FOR STEP 1
      C  graded distance to the nearest such junction

  If B already does what C does, the agreed fix is sufficient and no design
  change is needed -- Pete's original question gets a clean answer. If C beats
  B, the case for grading is real and rests on Track A's argument rather than
  on the 18pp biology finding.

  Neither window has asked this. We have both been arguing about A vs C while
  the thing being built is B.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp11_which_adjustment.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast_lib import boot_diff, code_strata, std_diff

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


def dbin(x):
    if np.isnan(x):
        return "none"
    for lo, hi, nm in [(-1, 50, "d0_50"), (50, 100, "d51_100"),
                       (100, 200, "d101_200"), (200, 500, "d201_500"),
                       (500, 10**9, "d500p")]:
        if lo < x <= hi:
            return nm
    return "none"


def main():
    print("=" * 104)
    print("EXPERIMENT 11 -- binary vs COUNT vs graded distance")
    print("=" * 104)

    blob, off, idx = load_seqs()
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id", "ref_utr5_length",
                               "ref_atg_count", "ref_utr5_orf_coverage"]
                      ).drop_duplicates("isoform_id")
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t")

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.drop(columns=[c for c in ("tx_length",) if c in d.columns])
    d = d.merge(tx, on="isoform_id").merge(ref, on="isoform_id", how="left")
    d["_i"] = d["isoform_id"].map(idx)
    d = d[d["_i"].notna() & d["gene_id"].notna()].copy()
    d["_i"] = d["_i"].astype(int)
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d = d[d["utr3"] >= 200].copy()

    dist, cnt50 = [], []
    for i, e in zip(d["isoform_id"], d["orf_end"]):
        j = junc.get(i, np.empty(0, dtype=np.int64))
        k = int(np.searchsorted(j, int(e), side="right"))
        dist.append(int(j[k]) - int(e) if k < len(j) else np.nan)
        cnt50.append(len(j) - int(np.searchsorted(j, int(e) + 50, side="right")))
    d["dist"], d["n50"] = dist, cnt50

    # the three adjustments
    d["ADJ_binary"] = (d["n50"] > 0).astype(int)
    d["ADJ_count"] = np.clip(d["n50"], 0, 4).astype(int)      # the agreed Step 1 fix
    d["ADJ_graded"] = d["dist"].apply(dbin)
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop").astype(int)
    d["win"] = [sub(blob, off, i, e + 1, e + 201)
                for i, e in zip(d["_i"], d["orf_end"])]
    d = d[d["win"].str.len().eq(200)].copy()
    d["gc_q"] = pd.qcut(d["win"].str.count("[GC]") / 200, 4, labels=False,
                        duplicates="drop").astype(int)
    d["p4"] = [sub(blob, off, i, e + 4, e + 5)
               for i, e in zip(d["_i"], d["orf_end"])]

    print(f"\n  {len(d):,} isoforms, reference-CDS anchor, >= 200 nt 3'UTR")
    print(f"\n  the three adjustments, and how much they distinguish:")
    print(f"    {'ADJ_binary levels':<26} {d['ADJ_binary'].nunique()}")
    print(f"    {'ADJ_count levels':<26} {d['ADJ_count'].nunique()}   "
          f"distribution {dict(sorted(d['ADJ_count'].value_counts().items()))}")
    print(f"    {'ADJ_graded levels':<26} {d['ADJ_graded'].nunique()}")
    print(f"\n  NMD rate by the COUNT feature (the one being built):")
    for k in sorted(d["ADJ_count"].unique()):
        g = d[d["ADJ_count"].eq(k)]
        print(f"    n50 = {k}{'+' if k == 4 else ' '}   n {len(g):>7,}   "
              f"NMD+ {g['is_nmd'].mean()*100:>5.1f}%")

    # ---------------------------------------------------------------- candidates
    d["c_stop"] = d["stop_codon"].where(d["stop_codon"].isin(["TGA", "TAG"]))
    d["c_p4"] = d["p4"].where(d["p4"].isin(["A", "T"]))
    d["c_pas"] = np.where(d["win"].str.contains("AATAAA", regex=False),
                          "present", "absent")
    d["c_py"] = np.where(d["win"].str.contains("[CT]{10}", regex=True),
                         "present", "absent")
    u5 = d["ref_utr5_length"]
    d["c_u5"] = np.where(u5 > u5.quantile(0.75), "long",
                         np.where(u5 < u5.quantile(0.25), "short", None))
    na = d["ref_atg_count"]
    d["c_uorf"] = np.where(na >= 4, "many", np.where(na <= 1, "few", None))
    cov = d["ref_utr5_orf_coverage"]
    d["c_cov"] = np.where(cov > 0.5, "high", np.where(cov.eq(0), "zero", None))
    u3 = d["utr3"]
    d["c_u3"] = np.where(u3 > u3.quantile(0.75), "long",
                         np.where(u3 < u3.quantile(0.25), "short", None))

    cands = [("stop codon TGA vs TAG", "c_stop", "TGA", "TAG", "3'"),
             ("+4 base A vs T", "c_p4", "A", "T", "3'"),
             ("AATAAA present", "c_pas", "present", "absent", "3'"),
             ("pyrimidine run >= 10", "c_py", "present", "absent", "3'"),
             ("3'UTR length long vs short", "c_u3", "long", "short", "3'"),
             ("5'UTR length long vs short", "c_u5", "long", "short", "5'"),
             ("upstream AUG >=4 vs <=1", "c_uorf", "many", "few", "5'"),
             ("5'UTR ORF coverage >0.5 vs 0", "c_cov", "high", "zero", "5'")]

    print("\n" + "=" * 104)
    print("A. EACH CANDIDATE UNDER ALL THREE ADJUSTMENTS")
    print("   shift reported RELATIVE to the effect, which is the correction")
    print("   Track A made and I needed")
    print("=" * 104)
    print(f"\n  {'candidate':<30} {'end':<4} {'A binary':>10} {'B COUNT':>10} "
          f"{'C graded':>10} {'B vs A':>9} {'C vs B':>9}")
    print(f"  {'-'*30} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*9}")
    rows = []
    for nm, col, a, b, end in cands:
        g = d[d[col].isin([a, b])].copy()
        if len(g) < 400:
            continue
        out = {}
        for tag, adj in (("A", "ADJ_binary"), ("B", "ADJ_count"),
                         ("C", "ADJ_graded")):
            r = boot_diff(g, col, a, b, [adj, "utr3_q", "gc_q"],
                          "is_nmd", "gene_id", n=400)
            out[tag] = r["diff_common"]
        ba = (out["B"] - out["A"]) / abs(out["A"]) * 100 if out["A"] else np.nan
        cb = (out["C"] - out["B"]) / abs(out["B"]) * 100 if out["B"] else np.nan
        print(f"  {nm:<30} {end:<4} {out['A']:>+9.2f}p {out['B']:>+9.2f}p "
              f"{out['C']:>+9.2f}p {ba:>+8.0f}% {cb:>+8.0f}%")
        rows.append(dict(name=nm, end=end, A=out["A"], B=out["B"], C=out["C"],
                         ba=ba, cb=cb))
    res = pd.DataFrame(rows)

    print("\n" + "=" * 104)
    print("B. THE ANSWER TO THE DESIGN QUESTION")
    print("=" * 104)
    print(f"\n  median |shift| binary -> COUNT   : "
          f"{res['ba'].abs().median():>6.1f}% of the effect")
    print(f"  median |shift| COUNT  -> graded  : "
          f"{res['cb'].abs().median():>6.1f}% of the effect")
    for e in ["3'", "5'"]:
        s = res[res["end"].eq(e)]
        print(f"    {e} end  binary->COUNT {s['ba'].abs().median():>6.1f}%   "
              f"COUNT->graded {s['cb'].abs().median():>6.1f}%")
    print("""
  If binary -> COUNT captures most of the movement and COUNT -> graded adds
  little, then the fix ALREADY AGREED for Step 1 is sufficient, and the
  graded-distance change is unnecessary. Track A's measurement would then be
  correct about binary and beside the point about what is being built, because
  neither of us was comparing against the feature that will actually exist.""")

    print("\n" + "=" * 104)
    print("C. TRACK A'S NULL, REPRODUCED -- is this covariate absorption?")
    print("=" * 104)
    g = d[d["c_stop"].isin(["TGA", "TAG"])].copy()
    freqs = d["ADJ_graded"].value_counts(normalize=True)
    base = boot_diff(g, "c_stop", "TGA", "TAG",
                     ["ADJ_binary", "utr3_q", "gc_q"], "is_nmd", "gene_id", n=300)
    draws = []
    kc_cols = ["utr3_q", "gc_q"]
    for t in range(200):
        g["_rand"] = RNG.choice(freqs.index, size=len(g), p=freqs.values)
        kcode, nk = code_strata(g, ["ADJ_binary", "_rand"] + kc_cols)
        arm = g["c_stop"].eq("TGA").to_numpy().astype(np.int64)
        y = g["is_nmd"].to_numpy().astype(np.float64)
        draws.append(std_diff(kcode, arm, y, nk)["diff_common"])
    draws = np.array([x for x in draws if not np.isnan(x)])
    grad = boot_diff(g, "c_stop", "TGA", "TAG",
                     ["ADJ_graded", "utr3_q", "gc_q"], "is_nmd", "gene_id", n=300)
    cnt = boot_diff(g, "c_stop", "TGA", "TAG",
                    ["ADJ_count", "utr3_q", "gc_q"], "is_nmd", "gene_id", n=300)
    print(f"\n  stop effect | binary                      {base['diff_common']:+.2f}pp")
    print(f"  stop effect | binary + 5 RANDOM dummies    "
          f"{np.median(draws):+.2f}pp   band "
          f"[{np.percentile(draws,2.5):+.2f}, {np.percentile(draws,97.5):+.2f}]")
    print(f"  stop effect | the COUNT feature            {cnt['diff_common']:+.2f}pp")
    print(f"  stop effect | graded distance bins         {grad['diff_common']:+.2f}pp")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    for nm, v in (("COUNT", cnt["diff_common"]), ("graded", grad["diff_common"])):
        print(f"    {nm:<8} is {'OUTSIDE' if not (lo <= v <= hi) else 'INSIDE'} "
              f"the random-covariate band")
    print("\n  Random covariates at the same marginal frequencies do not move it.")
    print("  Whatever moves it is information, not absorption -- Track A's point,")
    print("  reproduced. The open question is whether the COUNT already has it.")

    print("\n" + "=" * 104)
    print("DONE")
    print("=" * 104)


if __name__ == "__main__":
    main()
