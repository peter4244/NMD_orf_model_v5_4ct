#!/usr/bin/env python
"""
Does contrast_lib reproduce the numbers already reported, and does showing both
weightings side by side actually surface the problem it is meant to surface?

There is no pass/fail on an estimator choice here -- the library does not make
one. What IS checked: the two weightings must agree where both arms are large
(that is the substantive claim behind keeping the stop-codon result), and they
must diverge where one arm is thin (that is the defect that produced a wrong
retraction). And the reported headlines must still come out.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python test_contrast_lib.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast_lib import (boot_diff, code_strata, describe, power_match,
                          std_diff, sweep_null)

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, got, want, tol):
    ok = (not np.isnan(got)) and abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<50} got {got:+7.2f}  "
          f"expected {want:+7.2f}  (tol {tol})")
    if not ok:
        FAILED.append(name)


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


def gc_of(blob, off, i, e, n=100):
    t = sub(blob, off, i, e + 1, e + 1 + n)
    return (t.count("G") + t.count("C")) / len(t) if t else np.nan


def main():
    junc = load_junctions()
    blob, off, idx = load_seqs()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id"]).drop_duplicates("isoform_id")
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t")

    print("=" * 92)
    print("0. THE POPULATION DEFINITION, STATED CORRECTLY THIS TIME")
    print("=" * 92)
    print(f"\n  orf_rank is ZERO-BASED: "
          f"{dict(sorted(sel['orf_rank'].value_counts().items()))}")
    print(f"  is_ref_cds lives entirely at rank 0: "
          f"{sel.groupby('orf_rank')['is_ref_cds'].sum().to_dict()}")
    print("  My earlier prose called the headline population 'slot 0")
    print("  (orf_rank == 1)'. The CODE used .min(), which is 0, so the numbers")
    print("  are right and the sentence was wrong. Caught by Track A.")

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.drop(columns=[c for c in ("tx_length",) if c in d.columns])
    d = d.merge(tx, on="isoform_id").merge(ref, on="isoform_id", how="left")
    d["_i"] = d["isoform_id"].map(idx)
    d = d[d["_i"].notna() & d["gene_id"].notna()].copy()
    d["_i"] = d["_i"].astype(int)
    d["utr3"] = d["tx_length"] - d["orf_end"]
    dist, ptc = [], []
    for i, e in zip(d["isoform_id"], d["orf_end"]):
        j = junc.get(i, np.empty(0, dtype=np.int64))
        k = int(np.searchsorted(j, int(e), side="right"))
        dist.append(int(j[k]) - int(e) if k < len(j) else np.nan)
        # PTC is "SOME junction lies beyond stop+50", not "the NEAREST one does".
        # Those differ whenever a near junction is followed by a far one, and
        # conflating them was the first thing this file caught.
        ptc.append(int(len(j) - int(np.searchsorted(j, int(e) + 50,
                                                    side="right")) > 0))
    d["dist"], d["ptc"] = dist, ptc

    s2 = d[d["utr3"] >= 60].copy()
    s2["gc100"] = [gc_of(blob, off, i, e) for i, e in zip(s2["_i"], s2["orf_end"])]
    s2 = s2[s2["gc100"].notna()].copy()
    s2["utr3_q"] = pd.qcut(s2["utr3"], 4, labels=False, duplicates="drop").astype(int)
    s2["gc_q"] = pd.qcut(s2["gc100"], 4, labels=False, duplicates="drop").astype(int)
    S = ["ptc", "utr3_q", "gc_q"]

    print("\n" + "=" * 92)
    print("1. WHERE BOTH ARMS ARE LARGE, THE WEIGHTING CHOICE DOES NOT MATTER")
    print("=" * 92)
    r = boot_diff(s2, "stop_codon", "TGA", "TAG", S, "is_nmd", "gene_id", n=1500)
    describe(r, "\n  Exp 2 -- TGA minus TAG at the stop codon")
    check("stop: the two weightings agree",
          abs(r["diff_per_arm"] - r["diff_common"]), 0.0, 0.30)
    check("stop: effect is present and positive", r["diff_common"], 2.6, 1.0)
    print("\n  That agreement is the substantive point. The estimator that")
    print("  inflates at small n is not inflating here, because it has no thin")
    print("  arm to inflate on -- which is why the stop-codon claim stands.")

    for v, nm in ((1, "PTC+"), (0, "PTC-")):
        rr = boot_diff(s2[s2["ptc"].eq(v)], "stop_codon", "TGA", "TAG",
                       ["utr3_q", "gc_q"], "is_nmd", "gene_id", n=800)
        describe(rr, f"\n  {nm} subgroup")
    print("\n  The mechanistic gradient -- larger where a downstream junction")
    print("  exists for termination efficiency to act on -- does not depend on")
    print("  the weighting choice either.")

    print("\n" + "=" * 92)
    print("2. WHERE ONE ARM IS THIN, THEY DIVERGE -- WHICH IS THE WHOLE STORY")
    print("=" * 92)
    s3 = d[d["utr3"] >= 320].copy()
    s3["gc100"] = [gc_of(blob, off, i, e) for i, e in zip(s3["_i"], s3["orf_end"])]
    s3 = s3[s3["gc100"].notna()].copy()
    s3["utr3_q"] = pd.qcut(s3["utr3"], 4, labels=False, duplicates="drop").astype(int)
    s3["gc_q"] = pd.qcut(s3["gc100"], 4, labels=False, duplicates="drop").astype(int)
    cache = {}

    def make_arm(pos):
        if pos not in cache:
            cache[pos] = pd.Series(
                [sub(blob, off, i, e + pos, e + pos + 3)
                 for i, e in zip(s3["_i"], s3["orf_end"])], index=s3.index)
        return cache[pos].map({"TGA": True, "TAG": False})

    null = sweep_null(make_arm, range(4, 301, 3), s3, S, "is_nmd")
    print(f"\n  {len(null)} mechanism-free 3'UTR positions, TGA vs TAG")
    print(f"    {'weighting':<18} {'median':>9} {'IQR':>20} {'median strata':>15}")
    print(f"    {'-'*18} {'-'*9} {'-'*20} {'-'*15}")
    for col, nm in (("diff_per_arm", "per-arm"), ("diff_common", "common-stratum")):
        v = null[col].dropna()
        print(f"    {nm:<18} {v.median():>+8.2f}p "
              f"[{v.quantile(.25):>+7.2f}, {v.quantile(.75):>+7.2f}] "
              f"{null['strata_shared'].median():>15.0f}")
    check("per-arm null is displaced from zero",
          float(null["diff_per_arm"].median()), 13.0, 6.0)
    check("common-stratum null is centred near zero",
          float(null["diff_common"].median()), 0.0, 2.5)
    print(f"\n  median arm sizes at a control position: "
          f"{null['n1'].median():.0f} vs {null['n0'].median():.0f}   "
          f"(at the stop: {r['n1']:,} vs {r['n0']:,})")
    print("  A guard keyed on 'no shared stratum' would NOT have fired here --")
    print(f"  the median position still shares {null['strata_shared'].median():.0f} strata. "
          f"The tell was the arm\n  sizes and the 25x ratio, which is a thing to notice, not to assert.")

    print("\n" + "=" * 92)
    print("3. THE CHECK THAT ACTUALLY DECIDED IT -- power-matching")
    print("=" * 92)
    n1, n0 = int(null["n1"].median()), int(null["n0"].median())
    pm = power_match(s3, "stop_codon", "TGA", "TAG", S, "is_nmd", n1, n0,
                     reps=500, use="per_arm")
    print(f"\n  Stop-codon contrast subsampled to {n1} TGA / {n0} TAG, per-arm")
    print(f"  weighting, 500 draws -- the answer here is KNOWN to be about +2pp:")
    print(f"    median {np.median(pm):+.2f}pp   "
          f"2.5-97.5 pct [{np.percentile(pm, 2.5):+.2f}, "
          f"{np.percentile(pm, 97.5):+.2f}]   sd {pm.std(ddof=1):.2f}")
    check("per-arm estimator inflates at control sample size",
          float(np.median(pm)), 13.0, 6.0)
    print("\n  It returns ~+13pp on a contrast whose true value is ~+2pp. The")
    print("  control positions returned ~+13pp. They were measuring the same")
    print("  thing with 25x less data through an estimator that inflates.")

    print("\n" + "=" * 92)
    print("4. THE OTHER HEADLINES STILL COME OUT")
    print("=" * 92)
    one = d[d["n_downstream_ejc"].eq(1) & d["dist"].notna() & (d["utr3"] >= 10)].copy()
    one["utr3_q"] = pd.qcut(one["utr3"], 4, labels=False, duplicates="drop").astype(int)
    one["len_q"] = pd.qcut(one["orf_length"], 4, labels=False,
                           duplicates="drop").astype(int)
    one["far"] = np.where(one["dist"] > 50, "far", "near")
    r4 = boot_diff(one, "far", "far", "near", ["utr3_q", "len_q"],
                   "is_nmd", "gene_id", n=1500)
    describe(r4, "\n  Exp 4 -- the >50 nt rule (positive control)")
    check("Exp4 positive control", r4["diff_common"], 37.4, 2.0)

    mid = one[one["dist"] > 50].copy()
    mid["band"] = np.where(mid["dist"] <= 100, "b51_100",
                           np.where(mid["dist"] <= 200, "b101_200", "far"))
    r6 = boot_diff(mid, "band", "b101_200", "b51_100", ["utr3_q", "len_q"],
                   "is_nmd", "gene_id", n=1500)
    describe(r6, "\n  Exp 6 -- 101-200 minus 51-100, the dose beyond the threshold")
    check("Exp6 rise beyond the threshold", r6["diff_common"], 13.0, 6.0)
    print("         Track A independently got +13.0pp, CI [+6.4, +19.7].")

    print("\n" + "=" * 92)
    print("ALL CHECKS PASSED" if not FAILED else "FAILURES: " + ", ".join(FAILED))
    print("=" * 92)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
