#!/usr/bin/env python
"""
EXPERIMENT 2b -- was the control that killed the stop-codon claim any good?

WHAT I CONCLUDED, AND WHY IT IS UNDER REVIEW
  Exp 2 found TGA - TAG = +2.00pp (CI [+0.84, +3.18], p = 0.001, gene-clustered)
  at the stop codon, standardised over PTC status x 3'UTR quartile x GC quartile.
  I then ran the same 3-mer contrast at three 3'UTR positions -- +25, +50, +100 --
  got +11.66, +9.11, +11.13pp, and concluded the stop result did not stand
  outside its own null.

  Pete pushed back that this is too strong. Three things are wrong with that
  control and I did not check any of them:

  1. THREE DRAWS ARE NOT A DISTRIBUTION. The control CIs were [+3.67, +23.48],
     [+2.67, +16.15] and [-2.02, +19.64]. Estimates that imprecise cannot
     establish where a null sits; I read three noisy numbers as a null and they
     are not one.

  2. THE SAMPLE SIZES ARE NOT COMPARABLE. At the stop, TGA n = 14,477 and
     TAG n = 5,818 -- every transcript carries one of three stop codons. At +25,
     TGA n = 541 and TAG n = 214, roughly the 1/64 chance rate for a specific
     3-mer. A 25-fold difference in precision was never accounted for.

  3. THE ESTIMANDS DIFFER. standardised() drops any stratum with fewer than
     min_cell members. At the stop every stratum is populated and all 32
     contribute. At a control position only the largest few can, so the control
     is a differently-weighted average over a different subset of strata. That
     is not the same statistic.

WHAT THIS SCRIPT DOES INSTEAD
  A. Runs the contrast at ~99 3'UTR positions, not 3, to get a real null.
  B. Power-matches: subsamples the STOP population down to a control position's
     sample sizes, 500 times, so the two are compared at equal precision.
  C. Reports how many strata actually contribute at each position, which is the
     defect in 3 above, and repeats the null with a common-stratum estimator.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp2b_control_validity.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "seq_store.npz")
RNG = np.random.default_rng(20260801)


def load_seqs():
    z = np.load(STORE, allow_pickle=False)
    return z["blob"], z["offsets"], {s: i for i, s in enumerate(z["ids"])}


def sub(blob, off, i, a, b):
    lo, hi = int(off[i]), int(off[i + 1])
    a0, b0 = lo + a - 1, lo + b - 1
    if a0 < lo or b0 > hi or b0 <= a0:
        return ""
    return blob[a0:b0].tobytes().decode("ascii")


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                  if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for iso, j in zip(df["isoform_id"], df["junctions"])}


def std_diff(kcode, arm, y, nk, rows=None, min_cell=10):
    """Standardised NMD difference, arm 1 minus arm 0. Returns (diff, n_strata_used)."""
    if rows is not None:
        kcode, arm, y = kcode[rows], arm[rows], y[rows]
    n_k = np.bincount(kcode, minlength=nk).astype(np.float64)
    if n_k.sum() == 0:
        return np.nan, 0
    w = n_k / n_k.sum()
    vals, used = [], None
    goods = []
    for side in (1, 0):
        m = arm == side
        cnt = np.bincount(kcode[m], minlength=nk).astype(np.float64)
        s = np.bincount(kcode[m], weights=y[m], minlength=nk)
        good = cnt >= min_cell
        goods.append(good)
        den = w[good].sum()
        rate = np.divide(s, cnt, out=np.zeros(nk), where=cnt > 0)
        vals.append((w[good] * rate[good]).sum() / den if den > 0 else np.nan)
    used = int((goods[0] & goods[1]).sum())
    return (vals[0] - vals[1]) * 100, used


def std_diff_common(kcode, arm, y, nk, min_cell=10):
    """Same, but restricted to strata where BOTH arms clear min_cell -- so the
    two arms are averaged over an identical set of strata."""
    cnt1 = np.bincount(kcode[arm == 1], minlength=nk).astype(np.float64)
    cnt0 = np.bincount(kcode[arm == 0], minlength=nk).astype(np.float64)
    good = (cnt1 >= min_cell) & (cnt0 >= min_cell)
    if not good.any():
        return np.nan, 0
    n_k = np.bincount(kcode, minlength=nk).astype(np.float64)
    w = n_k * good
    w = w / w.sum()
    s1 = np.bincount(kcode[arm == 1], weights=y[arm == 1], minlength=nk)
    s0 = np.bincount(kcode[arm == 0], weights=y[arm == 0], minlength=nk)
    r1 = (w[good] * (s1[good] / cnt1[good])).sum()
    r0 = (w[good] * (s0[good] / cnt0[good])).sum()
    return (r1 - r0) * 100, int(good.sum())


def main():
    print("=" * 100)
    print("EXPERIMENT 2b -- validating the control that killed the stop-codon claim")
    print("=" * 100)

    blob, off, idx = load_seqs()
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_end", "is_ref_cds", "stop_codon"])

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.merge(tx, on="isoform_id")
    d["_i"] = d["isoform_id"].map(idx)
    d = d[d["_i"].notna()].copy()
    d["_i"] = d["_i"].astype(int)
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d = d[d["utr3"] >= 320].copy()          # every scanned position readable
    d["ptc"] = [len(junc.get(i, np.empty(0, dtype=np.int64)))
                - int(np.searchsorted(junc.get(i, np.empty(0, dtype=np.int64)),
                                      int(e) + 50, side="right")) > 0
                for i, e in zip(d["isoform_id"], d["orf_end"])]
    d["ptc"] = d["ptc"].astype(int)
    g = []
    for i, e in zip(d["_i"], d["orf_end"]):
        s = sub(blob, off, i, e + 1, e + 101)
        g.append((s.count("G") + s.count("C")) / len(s) if s else np.nan)
    d["gc100"] = g
    d = d[d["gc100"].notna()].copy()
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop").astype(int)
    d["gc_q"] = pd.qcut(d["gc100"], 4, labels=False, duplicates="drop").astype(int)
    print(f"\n  population: {len(d):,} isoforms (reference-CDS slot, >= 320 nt 3'UTR")
    print(f"  so that every scanned control position is readable in every one)")
    print(f"  NMD+ {d['is_nmd'].mean()*100:.1f}%   PTC+ {d['ptc'].mean()*100:.1f}%")

    kcode, _ = pd.factorize(
        d[["ptc", "utr3_q", "gc_q"]].astype(str).agg("|".join, axis=1))
    nk = int(kcode.max()) + 1
    y = d["is_nmd"].to_numpy().astype(np.float64)
    print(f"  strata: {nk}")

    def contrast_at(seq_series):
        m = seq_series.isin(["TGA", "TAG"]).to_numpy()
        if m.sum() == 0:
            return None
        arm = seq_series.eq("TGA").to_numpy().astype(np.int64)[m]
        kk, yy = kcode[m], y[m]
        diff, used = std_diff(kk, arm, yy, nk)
        dc, usedc = std_diff_common(kk, arm, yy, nk)
        crude = (yy[arm == 1].mean() - yy[arm == 0].mean()) * 100 \
            if (arm == 1).any() and (arm == 0).any() else np.nan
        return dict(n_tga=int((arm == 1).sum()), n_tag=int((arm == 0).sum()),
                    crude=crude, std=diff, strata=used, common=dc,
                    strata_common=usedc)

    print("\n" + "=" * 100)
    print("A. THE STOP CODON ITSELF")
    print("=" * 100)
    at_stop = contrast_at(d["stop_codon"])
    print(f"\n  n TGA {at_stop['n_tga']:,}   n TAG {at_stop['n_tag']:,}")
    print(f"  crude          TGA - TAG = {at_stop['crude']:+.2f}pp")
    print(f"  standardised   TGA - TAG = {at_stop['std']:+.2f}pp   "
          f"({at_stop['strata']} of {nk} strata contribute)")
    print(f"  common-stratum TGA - TAG = {at_stop['common']:+.2f}pp   "
          f"({at_stop['strata_common']} strata)")

    print("\n" + "=" * 100)
    print("B. THE REAL NULL -- 99 control positions instead of 3")
    print("=" * 100)
    pos = list(range(4, 301, 3))
    rows = []
    for k in pos:
        s = pd.Series([sub(blob, off, i, e + k, e + k + 3)
                       for i, e in zip(d["_i"], d["orf_end"])], index=d.index)
        r = contrast_at(s)
        if r and min(r["n_tga"], r["n_tag"]) >= 30:
            r["pos"] = k
            rows.append(r)
    ctl = pd.DataFrame(rows)
    print(f"\n  {len(ctl)} control positions with >= 30 in each arm, +4 to +300 nt")
    print(f"\n  {'statistic':<22} {'median':>9} {'IQR':>20} {'2.5-97.5 pct':>22} "
          f"{'mean':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*20} {'-'*22} {'-'*9}")
    for col, nm in (("crude", "crude"), ("std", "standardised"),
                    ("common", "common-stratum")):
        v = ctl[col].dropna()
        lo, hi = np.percentile(v, [2.5, 97.5])
        q1, q3 = np.percentile(v, [25, 75])
        print(f"  {nm:<22} {v.median():>+8.2f}p [{q1:>+7.2f}, {q3:>+7.2f}] "
              f"[{lo:>+8.2f}, {hi:>+8.2f}]  {v.mean():>+8.2f}p")

    print(f"\n  median n per arm at a control position: "
          f"TGA {ctl['n_tga'].median():.0f}, TAG {ctl['n_tag'].median():.0f}   "
          f"(at the stop: {at_stop['n_tga']:,} / {at_stop['n_tag']:,})")
    print(f"  median strata contributing: {ctl['strata'].median():.0f} of {nk} "
          f"(at the stop: {at_stop['strata']})")
    print("\n  ^ THIS is defect 3. The control statistic is averaged over a")
    print("    handful of strata; the stop statistic over all of them.")

    for col, nm in (("std", "standardised"), ("common", "common-stratum")):
        v = ctl[col].dropna()
        pct = (v < at_stop[col]).mean() * 100
        print(f"\n  where the stop codon sits in the {nm} null: "
              f"{at_stop[col]:+.2f}pp = {pct:.0f}th percentile of {len(v)} controls")

    print("\n" + "=" * 100)
    print("C. POWER-MATCHED -- the stop contrast subsampled to a control's size")
    print("=" * 100)
    n1, n0 = int(ctl["n_tga"].median()), int(ctl["n_tag"].median())
    m = d["stop_codon"].isin(["TGA", "TAG"]).to_numpy()
    arm_s = d["stop_codon"].eq("TGA").to_numpy().astype(np.int64)[m]
    kk_s, yy_s = kcode[m], y[m]
    i1 = np.flatnonzero(arm_s == 1)
    i0 = np.flatnonzero(arm_s == 0)
    draws = []
    for _ in range(500):
        pick = np.concatenate([RNG.choice(i1, n1, replace=False),
                               RNG.choice(i0, n0, replace=False)])
        dd, _u = std_diff(kk_s[pick], arm_s[pick], yy_s[pick], nk)
        draws.append(dd)
    draws = np.array([x for x in draws if not np.isnan(x)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"\n  Stop-codon contrast, subsampled to n = {n1} TGA / {n0} TAG,")
    print(f"  500 draws, identical estimator:")
    print(f"    median {np.median(draws):+.2f}pp   "
          f"2.5-97.5 pct [{lo:+.2f}, {hi:+.2f}]   sd {draws.std(ddof=1):.2f}")
    v = ctl["std"].dropna()
    clo, chi = np.percentile(v, [2.5, 97.5])
    print(f"  Control positions at their native n:")
    print(f"    median {v.median():+.2f}pp   2.5-97.5 pct [{clo:+.2f}, {chi:+.2f}]   "
          f"sd {v.std(ddof=1):.2f}")
    print("\n  If these two spreads are similar, the control positions were never")
    print("  measuring a different thing -- they were measuring the same thing")
    print("  with 25x less data, and the +9 to +11pp I reported is what this")
    print("  estimator does at n = 200-500, not what a null looks like.")

    print("\n" + "=" * 100)
    print("D. THE THREE POSITIONS I ORIGINALLY PICKED, IN CONTEXT")
    print("=" * 100)
    print(f"\n  {'position':<12} {'n TGA':>7} {'n TAG':>7} {'crude':>9} "
          f"{'standardised':>14} {'strata':>8} {'pctile of null':>16}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*9} {'-'*14} {'-'*8} {'-'*16}")
    v = ctl["std"].dropna()
    for k in (25, 49, 100):
        r = ctl[ctl["pos"].eq(k)]
        if not len(r):
            continue
        r = r.iloc[0]
        print(f"  +{k:<11} {r['n_tga']:>7,.0f} {r['n_tag']:>7,.0f} "
              f"{r['crude']:>+8.2f}p {r['std']:>+13.2f}p {r['strata']:>8.0f} "
              f"{(v < r['std']).mean()*100:>15.0f}%")
    print(f"  {'STOP':<12} {at_stop['n_tga']:>7,} {at_stop['n_tag']:>7,} "
          f"{at_stop['crude']:>+8.2f}p {at_stop['std']:>+13.2f}p "
          f"{at_stop['strata']:>8} {(v < at_stop['std']).mean()*100:>15.0f}%")

    print("\n" + "=" * 100)
    print("E. THE SUBGROUP SPLIT I QUOTED AS SUPPORTING EVIDENCE")
    print("=" * 100)
    print("\n  Exp 2 reported TGA-TAG = +4.41pp in PTC+ and +1.04pp in PTC-, and I")
    print("  used that gradient as mechanistic support. Those were computed with")
    print("  the separate-stratum estimator too, so they need re-checking. If all")
    print("  strata are populated in both arms the two estimators must agree.\n")
    print(f"  {'subgroup':<12} {'n TGA':>7} {'n TAG':>7} {'crude':>9} "
          f"{'separate':>10} {'common':>9} {'strata':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*9} {'-'*10} {'-'*9} {'-'*8}")
    for v, nm in ((1, "PTC+"), (0, "PTC-")):
        g = d[d["ptc"].eq(v)]
        m = g["stop_codon"].isin(["TGA", "TAG"]).to_numpy()
        kk2, _ = pd.factorize(
            g[["utr3_q", "gc_q"]].astype(str).agg("|".join, axis=1))
        nk2 = int(kk2.max()) + 1
        arm = g["stop_codon"].eq("TGA").to_numpy().astype(np.int64)[m]
        yy = g["is_nmd"].to_numpy().astype(np.float64)[m]
        kk2 = kk2[m]
        sep, used = std_diff(kk2, arm, yy, nk2)
        com, usedc = std_diff_common(kk2, arm, yy, nk2)
        crude = (yy[arm == 1].mean() - yy[arm == 0].mean()) * 100
        print(f"  {nm:<12} {int((arm==1).sum()):>7,} {int((arm==0).sum()):>7,} "
              f"{crude:>+8.2f}p {sep:>+9.2f}p {com:>+8.2f}p "
              f"{usedc:>4d}/{nk2:<3d}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
