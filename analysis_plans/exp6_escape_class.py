#!/usr/bin/env python
"""
EXPERIMENT 6 -- the transcripts that satisfy the canonical PTC rule and are not
degraded.

Holding the junction count at exactly one, NMD rate by stop-to-junction distance
runs 10.2 / 13.8 / 57.6 / 70.6 / 22.2 %. The last bin is the anomaly: a single
junction MORE than 200 nt past the stop is a textbook NMD trigger, and these
transcripts survive.

WHY THIS VERSION IS WORTH CHECKING WHEN THE EARLIER ONE WAS NOT
  Track A withdrew an earlier version of this drop as a metric artifact: taking
  the NEAREST junction selects transcripts with no junction anywhere in 50-200
  nt, and switching to the LAST junction moved a related figure from 32.8% to
  66.0%. That explanation cannot apply here. At n_downstream_ejc == 1 there is
  exactly one junction past the stop, so nearest IS last, and the artifact is
  impossible by construction.

THE CONFOUND NEITHER WINDOW HAS NAMED, AND IT COULD KILL THE WHOLE THING
  The NMD label is not a measurement of decay. It is the outcome of a
  differential-expression test between SMG1-inhibitor and DMSO across four cell
  types, and the NEGATIVE class is defined as adj.P > 0.30 in ALL FOUR. A
  lowly-expressed isoform cannot reach significance anywhere, so it is assigned
  to the negative class BY LACK OF POWER.

  If the > 200 nt bin is enriched for lowly-expressed transcripts, its low NMD
  rate is a detection artifact and there is no escape class. This is the first
  thing tested, before anything else is believed.

  Baseline expression is computed from the DMSO samples ONLY -- never from a
  pooled or treatment-averaged value, since the Smg1i arm is precisely where NMD
  substrates are stabilised and pooling would inflate the substrates' baseline.

ANCHOR
  The reference-CDS slot is primary, per Track A's argument that slot 0 is
  filled by our own ranking code (annotation priority, agreeing with transcript
  position 17.5% of the time) whereas the reference-CDS slot is a population a
  reader can reconstruct without our code. Slot 0 is reported alongside.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp6_escape_class.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
DEPOSIT = os.path.expanduser("~/claude_projects/nmd_deposit_2026/source_data")

BINS = [(-1, 37, "<=37"), (37, 50, "38-50"), (50, 100, "51-100"),
        (100, 200, "101-200"), (200, 500, "201-500"), (500, 10**9, ">500")]


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                  if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for iso, j in zip(df["isoform_id"], df["junctions"])}


def baseline_expression():
    """mean log2(CPM + 1) over the DMSO samples only."""
    pheno = pd.read_csv(os.path.join(DEPOSIT, "pheno_4ct.csv"))
    dmso = pheno.loc[pheno["treatment"].eq("DMSO"), "sample_name"].tolist()
    cnt = pd.read_csv(os.path.join(DEPOSIT, "nmd_lungcells_counts_4ct.csv"),
                      index_col=0)
    print(f"  count matrix: {cnt.shape[0]:,} isoforms x {cnt.shape[1]} samples")
    print(f"  DMSO samples used for baseline ({len(dmso)}): "
          f"{', '.join(sorted(dmso)[:4])} ...")
    missing = [s for s in dmso if s not in cnt.columns]
    if missing:
        print(f"  WARNING: DMSO samples absent from the matrix: {missing}")
        dmso = [s for s in dmso if s in cnt.columns]
    lib = cnt.sum(axis=0)
    cpm = cnt[dmso].div(lib[dmso], axis=1) * 1e6
    out = np.log2(cpm + 1).mean(axis=1).rename("expr_dmso")
    det = (cnt[dmso] > 0).sum(axis=1).rename("n_dmso_detected")
    print(f"  baseline log2(CPM+1): median {out.median():.2f}, "
          f"IQR [{out.quantile(.25):.2f}, {out.quantile(.75):.2f}]")
    return pd.concat([out, det], axis=1)


def standardised(d, col, strata, label="is_nmd", min_cell=15):
    d = d.dropna(subset=[col] + strata + [label])
    key = d[strata].astype(str).agg("|".join, axis=1)
    d = d.assign(_k=key)
    w = d["_k"].value_counts(normalize=True)
    out = {}
    for g, gg in d.groupby(col):
        num = den = 0.0
        used = 0
        for k, h in gg.groupby("_k"):
            if len(h) < min_cell:
                continue
            num += w[k] * h[label].mean()
            den += w[k]
            used += len(h)
        out[g] = (len(gg), gg[label].mean() * 100,
                  num / den * 100 if den > 0 else np.nan, used)
    return out


def bin_of(x):
    for lo, hi, nm in BINS:
        if lo < x <= hi:
            return nm
    return None


def main():
    print("=" * 98)
    print("EXPERIMENT 6 -- the >200 nt escape class, and whether it is real")
    print("=" * 98)

    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length", "n_junctions"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id"])
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_start", "orf_end", "orf_length",
                               "is_ref_cds", "orf_rank", "n_downstream_ejc"])

    print("\n  baseline expression")
    expr = baseline_expression()

    def build(mask, name):
        d = sel[mask].drop_duplicates("isoform_id").copy()
        d = (d.merge(tx, on="isoform_id")
               .merge(ref.drop_duplicates("isoform_id"), on="isoform_id", how="left")
               .merge(expr, left_on="isoform_id", right_index=True, how="left"))
        d = d[d["n_downstream_ejc"].eq(1)].copy()
        dist, jpos = [], []
        for i, e in zip(d["isoform_id"], d["orf_end"]):
            j = junc.get(i, np.empty(0, dtype=np.int64))
            k = int(np.searchsorted(j, int(e), side="right"))
            jpos.append(int(j[k]) if k < len(j) else np.nan)
            dist.append(int(j[k]) - int(e) if k < len(j) else np.nan)
        d["dist"] = dist
        d["jpos"] = jpos
        d["utr3"] = d["tx_length"] - d["orf_end"]
        d["j_to_end"] = d["tx_length"] - d["jpos"]
        d["bin"] = d["dist"].apply(lambda x: bin_of(x) if pd.notna(x) else None)
        print(f"\n  {name}: {len(d):,} isoforms with exactly one downstream junction"
              f"   (expression missing for {int(d['expr_dmso'].isna().sum()):,})")
        return d[d["bin"].notna()].copy()

    d = build(sel["is_ref_cds"].astype(bool), "reference-CDS anchor (primary)")
    d0 = build(sel["orf_rank"].eq(sel["orf_rank"].min()), "slot-0 anchor")

    print("\n" + "=" * 98)
    print("1. THE CONFOUND FIRST -- is the >200 bin simply lowly expressed?")
    print("=" * 98)
    print(f"\n  {'bin':<10} {'n':>6} {'NMD+':>7} {'med expr':>10} {'q25':>7} {'q75':>7} "
          f"{'% expr<median':>14} {'med DMSO detected':>18}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*10} {'-'*7} {'-'*7} {'-'*14} {'-'*18}")
    med_all = d["expr_dmso"].median()
    for lo, hi, nm in BINS:
        g = d[d["bin"].eq(nm)]
        if not len(g):
            continue
        print(f"  {nm:<10} {len(g):>6,} {g['is_nmd'].mean()*100:>6.1f}% "
              f"{g['expr_dmso'].median():>10.2f} {g['expr_dmso'].quantile(.25):>7.2f} "
              f"{g['expr_dmso'].quantile(.75):>7.2f} "
              f"{(g['expr_dmso'] < med_all).mean()*100:>13.1f}% "
              f"{g['n_dmso_detected'].median():>18.0f}")

    print("\n  Does expression predict the label at all, in this population?")
    dd = d[d["expr_dmso"].notna()].copy()
    dd["expr_q"] = pd.qcut(dd["expr_dmso"], 5, labels=False, duplicates="drop")
    print(f"    {'expr quintile':<16} {'n':>7} {'med log2CPM':>13} {'NMD+':>8}")
    for q in sorted(dd["expr_q"].unique()):
        g = dd[dd["expr_q"].eq(q)]
        print(f"    Q{int(q)+1:<15} {len(g):>7,} {g['expr_dmso'].median():>13.2f} "
              f"{g['is_nmd'].mean()*100:>7.1f}%")

    print("\n" + "=" * 98)
    print("2. THE DROP, WITH EXPRESSION HELD FIXED")
    print("=" * 98)
    r = standardised(dd, "bin", ["expr_q"])
    print(f"\n  {'bin':<10} {'n':>7} {'crude':>8} {'std. over expr quintile':>25}")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*25}")
    for lo, hi, nm in BINS:
        if nm in r:
            n, c, s, u = r[nm]
            ss = f"{s:>24.1f}%" if not np.isnan(s) else f"{'--':>25}"
            print(f"  {nm:<10} {n:>7,} {c:>7.1f}% {ss}")

    print("\n  And restricted to WELL-EXPRESSED isoforms only (top two quintiles),")
    print("  where the DE test has power and the negative class means something:")
    hi_e = dd[dd["expr_q"].ge(3)]
    print(f"    n = {len(hi_e):,}, median log2(CPM+1) {hi_e['expr_dmso'].median():.2f}")
    print(f"    {'bin':<10} {'n':>7} {'NMD+':>8}")
    for lo, hi, nm in BINS:
        g = hi_e[hi_e["bin"].eq(nm)]
        if len(g):
            print(f"    {nm:<10} {g.shape[0]:>7,} {g['is_nmd'].mean()*100:>7.1f}%")

    print("\n" + "=" * 98)
    print("3. IF IT SURVIVES -- what else is different about these transcripts?")
    print("=" * 98)
    print(f"\n  {'bin':<10} {'n':>6} {'NMD+':>7} {'med 3UTR':>10} {'med junc->end':>14} "
          f"{'med n_junc':>11} {'med ORF len':>12} {'med tx len':>11}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*10} {'-'*14} {'-'*11} {'-'*12} {'-'*11}")
    for lo, hi, nm in BINS:
        g = d[d["bin"].eq(nm)]
        if not len(g):
            continue
        print(f"  {nm:<10} {len(g):>6,} {g['is_nmd'].mean()*100:>6.1f}% "
              f"{g['utr3'].median():>10,.0f} {g['j_to_end'].median():>14,.0f} "
              f"{g['n_junctions'].median():>11,.0f} {g['orf_length'].median():>12,.0f} "
              f"{g['tx_length'].median():>11,.0f}")

    print("\n  The junction-to-3'-end distance is the structural variable that")
    print("  separates a junction sitting in the middle of a long 3'UTR from one")
    print("  sitting just before the poly-A site. NMD rate by BOTH distances:")
    dd2 = d[d["dist"] > 50].copy()
    dd2["jend_q"] = pd.qcut(dd2["j_to_end"], 4, labels=False, duplicates="drop")
    print(f"\n    {'stop->junction':<16}" + "".join(
        f"{'j->end Q'+str(q+1):>16}" for q in range(4)))
    print(f"    {'-'*16}" + "".join(f"{'-'*16:>16}" for _ in range(4)))
    for lo, hi, nm in BINS:
        if lo < 50:
            continue
        g = dd2[dd2["bin"].eq(nm)]
        cells = []
        for q in range(4):
            gg = g[g["jend_q"].eq(q)]
            cells.append(f"{len(gg):>5,} {gg['is_nmd'].mean()*100:>7.1f}%"
                         if len(gg) >= 20 else f"{len(gg):>5,} {'--':>8}")
        print(f"    {nm:<16}" + "".join(f"{c:>16}" for c in cells))

    print("\n" + "=" * 98)
    print("4. THE 3'UTR SIGN FLIP, WITH EXPRESSION HELD FIXED")
    print("=" * 98)
    print("\n  Earlier: long 3'UTR was +16pp at 101-200 nt and -19pp at >200 nt.")
    print("  A sign flip is either a real interaction or a confound. Re-run inside")
    print("  expression quintiles.\n")
    print(f"  {'bin':<10} {'3UTR<=1kb n':>13} {'NMD+':>8} {'3UTR>1kb n':>12} {'NMD+':>8} "
          f"{'diff':>9} {'diff | expr fixed':>19}")
    print(f"  {'-'*10} {'-'*13} {'-'*8} {'-'*12} {'-'*8} {'-'*9} {'-'*19}")
    for lo, hi, nm in BINS:
        g = dd[dd["bin"].eq(nm)].copy()
        if len(g) < 60:
            continue
        g["long3"] = (g["utr3"] > 1000).astype(int)
        s, l = g[g["long3"].eq(0)], g[g["long3"].eq(1)]
        if len(s) < 20 or len(l) < 20:
            continue
        rr = standardised(g, "long3", ["expr_q"])
        adj = (rr[1][2] - rr[0][2]) if (1 in rr and 0 in rr) else np.nan
        adjs = f"{adj:>+18.1f}pp" if not np.isnan(adj) else f"{'--':>19}"
        print(f"  {nm:<10} {len(s):>13,} {s['is_nmd'].mean()*100:>7.1f}% "
              f"{len(l):>12,} {l['is_nmd'].mean()*100:>7.1f}% "
              f"{(l['is_nmd'].mean()-s['is_nmd'].mean())*100:>+8.1f}p {adjs}")

    print("\n" + "=" * 98)
    print("5. SAME TEST ON THE SLOT-0 ANCHOR, so the conclusion is not")
    print("   an artifact of which ORF we anchored on")
    print("=" * 98)
    e0 = d0[d0["expr_dmso"].notna()].copy()
    e0["expr_q"] = pd.qcut(e0["expr_dmso"], 5, labels=False, duplicates="drop")
    r0 = standardised(e0, "bin", ["expr_q"])
    print(f"\n  {'bin':<10} {'n':>7} {'crude':>8} {'std. over expr quintile':>25}")
    for lo, hi, nm in BINS:
        if nm in r0:
            n, c, s, u = r0[nm]
            ss = f"{s:>24.1f}%" if not np.isnan(s) else f"{'--':>25}"
            print(f"  {nm:<10} {n:>7,} {c:>7.1f}% {ss}")

    print("\n" + "=" * 98)
    print("DONE")
    print("=" * 98)


if __name__ == "__main__":
    main()
