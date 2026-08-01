#!/usr/bin/env python
"""
EXPERIMENT 6b -- the last surviving alternative explanation for the window.

Experiment 6 found that the effect of a downstream junction is not a threshold
but a WINDOW: standardised over baseline expression, NMD rate runs
10.2 / 9.8 / 53.1 / 63.7 / 32.2 / 10.0 % across stop-to-junction distances of
<=37 / 38-50 / 51-100 / 101-200 / 201-500 / >500 nt. It rises to 64% and comes
all the way back down to 10%.

Five explanations are already excluded (exp6): baseline expression, the
nearest-vs-last artifact, slot-0 selection, junction-to-3'-end position, and
3'UTR length on its own.

ONE REMAINS, AND IT IS THE OBVIOUS ONE.
  A transcript in the >500 nt bin has a median of 3 junctions total, a 2.3 kb
  3'UTR, and its single downstream junction sits deep inside that 3'UTR. If a
  meaningful fraction of those junction CALLS are wrong -- long-read splice
  calls in long 3'UTRs are the weakest calls in the dataset -- then those
  transcripts are not PTC-bearing at all, they are simply mis-annotated, and
  they would show exactly the low NMD rate observed.

  If the window is an artifact of junction-call quality, it should disappear
  among transcripts whose junctions are well supported. If it survives there,
  it is not that.

SQANTI's own junction QC is used, none of it computed by us:
  all_canonical    every junction has canonical (GT-AG) dinucleotides
  min_cov          short-read coverage of the least-covered junction
  min_sample_cov   number of samples supporting the least-covered junction
  RTS_stage        flagged as a reverse-transcriptase switching artifact
  bite             flagged as an intron-retention-adjacent call

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp6b_junction_quality.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
DEPOSIT = os.path.expanduser("~/claude_projects/nmd_deposit_2026/source_data")
SQ = os.path.join(DEPOSIT, "sqanti", "nmd_lungcells_classification.txt")

BINS = [(-1, 37, "<=37"), (37, 50, "38-50"), (50, 100, "51-100"),
        (100, 200, "101-200"), (200, 500, "201-500"), (500, 10**9, ">500")]


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                  if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for iso, j in zip(df["isoform_id"], df["junctions"])}


def baseline_expression():
    pheno = pd.read_csv(os.path.join(DEPOSIT, "pheno_4ct.csv"))
    dmso = pheno.loc[pheno["treatment"].eq("DMSO"), "sample_name"].tolist()
    cnt = pd.read_csv(os.path.join(DEPOSIT, "nmd_lungcells_counts_4ct.csv"),
                      index_col=0)
    dmso = [s for s in dmso if s in cnt.columns]
    lib = cnt.sum(axis=0)
    cpm = cnt[dmso].div(lib[dmso], axis=1) * 1e6
    return np.log2(cpm + 1).mean(axis=1).rename("expr_dmso")


def standardised(d, col, strata, label="is_nmd", min_cell=15):
    d = d.dropna(subset=[col] + strata + [label])
    key = d[strata].astype(str).agg("|".join, axis=1)
    d = d.assign(_k=key)
    w = d["_k"].value_counts(normalize=True)
    out = {}
    for g, gg in d.groupby(col):
        num = den = 0.0
        for k, h in gg.groupby("_k"):
            if len(h) < min_cell:
                continue
            num += w[k] * h[label].mean()
            den += w[k]
        out[g] = (len(gg), gg[label].mean() * 100,
                  num / den * 100 if den > 0 else np.nan)
    return out


def bin_of(x):
    for lo, hi, nm in BINS:
        if lo < x <= hi:
            return nm
    return None


def show(title, d, strata=None):
    print(f"\n  {title}")
    print(f"    {'bin':<10} {'n':>7} {'NMD+':>8}"
          + (f"  {'standardised':>13}" if strata else ""))
    print(f"    {'-'*10} {'-'*7} {'-'*8}" + ("  " + "-"*13 if strata else ""))
    r = standardised(d, "bin", strata) if strata else None
    for lo, hi, nm in BINS:
        g = d[d["bin"].eq(nm)]
        if not len(g):
            continue
        line = f"    {nm:<10} {len(g):>7,} {g['is_nmd'].mean()*100:>7.1f}%"
        if strata:
            s = r.get(nm, (0, 0, np.nan))[2]
            line += f"  {s:>12.1f}%" if not np.isnan(s) else f"  {'--':>13}"
        print(line)


def main():
    print("=" * 98)
    print("EXPERIMENT 6b -- is the window an artifact of junction-call quality?")
    print("=" * 98)

    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length", "n_junctions"]]
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_end", "orf_length", "is_ref_cds",
                               "n_downstream_ejc"])
    expr = baseline_expression()

    cols = ["isoform", "structural_category", "subcategory", "exons", "RTS_stage",
            "all_canonical", "min_sample_cov", "min_cov", "bite"]
    print(f"\n  reading SQANTI classification ({os.path.getsize(SQ)/1e6:.0f} MB) ...")
    sq = pd.read_csv(SQ, sep="\t", usecols=cols, low_memory=False)
    print(f"  {len(sq):,} rows")

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = (d.merge(tx, on="isoform_id")
           .merge(expr, left_on="isoform_id", right_index=True, how="left")
           .merge(sq, left_on="isoform_id", right_on="isoform", how="left"))
    d = d[d["n_downstream_ejc"].eq(1)].copy()
    dist = []
    for i, e in zip(d["isoform_id"], d["orf_end"]):
        j = junc.get(i, np.empty(0, dtype=np.int64))
        k = int(np.searchsorted(j, int(e), side="right"))
        dist.append(int(j[k]) - int(e) if k < len(j) else np.nan)
    d["dist"] = dist
    d["bin"] = d["dist"].apply(lambda x: bin_of(x) if pd.notna(x) else None)
    d = d[d["bin"].notna()].copy()
    d["expr_q"] = pd.qcut(d["expr_dmso"], 5, labels=False,
                          duplicates="drop").astype("Int64")
    print(f"  analysis set: {len(d):,};  matched to SQANTI: "
          f"{int(d['isoform'].notna().sum()):,}")

    print("\n" + "=" * 98)
    print("1. IS JUNCTION QUALITY WORSE IN THE FAR BINS?")
    print("=" * 98)
    print(f"\n  {'bin':<10} {'n':>6} {'NMD+':>7} {'all_canonical':>14} {'RTS_stage':>11} "
          f"{'bite':>8} {'med min_cov':>12} {'min_sample_cov=0':>18}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*14} {'-'*11} {'-'*8} {'-'*12} {'-'*18}")
    for lo, hi, nm in BINS:
        g = d[d["bin"].eq(nm)]
        if not len(g):
            continue
        can = g["all_canonical"].astype(str).str.lower().isin(["canonical", "true"])
        rts = g["RTS_stage"].astype(str).str.lower().isin(["true"])
        bite = g["bite"].astype(str).str.lower().isin(["true"])
        mc = pd.to_numeric(g["min_cov"], errors="coerce")
        msc = pd.to_numeric(g["min_sample_cov"], errors="coerce")
        print(f"  {nm:<10} {len(g):>6,} {g['is_nmd'].mean()*100:>6.1f}% "
              f"{can.mean()*100:>13.1f}% {rts.mean()*100:>10.1f}% "
              f"{bite.mean()*100:>7.1f}% {mc.median():>12.1f} "
              f"{(msc.fillna(0) == 0).mean()*100:>17.1f}%")

    print("\n" + "=" * 98)
    print("2. THE WINDOW AMONG WELL-SUPPORTED TRANSCRIPTS ONLY")
    print("=" * 98)
    can = d["all_canonical"].astype(str).str.lower().isin(["canonical", "true"])
    rts = d["RTS_stage"].astype(str).str.lower().isin(["true"])
    mc = pd.to_numeric(d["min_cov"], errors="coerce")
    msc = pd.to_numeric(d["min_sample_cov"], errors="coerce")

    show("ALL transcripts (the exp6 result, for reference)", d, ["expr_q"])
    show("all junctions canonical, not RTS-flagged",
         d[can & ~rts], ["expr_q"])
    good = can & ~rts & (msc.fillna(0) >= 3) & (mc.fillna(0) >= 5)
    show(f"...and short-read support: min_sample_cov >= 3 AND min_cov >= 5 "
         f"(n = {int(good.sum()):,})", d[good], ["expr_q"])

    print("\n" + "=" * 98)
    print("3. AND AMONG REFERENCE-MATCHING TRANSCRIPTS ONLY")
    print("   (full-splice-match to an annotated transcript -- every junction is")
    print("    one GENCODE already contains, so a wrong call is not possible)")
    print("=" * 98)
    fsm = d[d["structural_category"].eq("full-splice_match")]
    show(f"full-splice_match only (n = {len(fsm):,})", fsm, ["expr_q"])
    print(f"\n  structural category composition by bin:")
    print(f"    {'bin':<10}" + "".join(f"{c[:14]:>16}" for c in
          ["full-splice_ma", "incomplete-spl", "novel_in_catal", "novel_not_in_c"]))
    for lo, hi, nm in BINS:
        g = d[d["bin"].eq(nm)]
        if not len(g):
            continue
        vc = g["structural_category"].value_counts(normalize=True) * 100
        cells = "".join(
            f"{vc.get(c, 0.0):>15.1f}%" for c in
            ["full-splice_match", "incomplete-splice_match",
             "novel_in_catalog", "novel_not_in_catalog"])
        print(f"    {nm:<10}{cells}")

    print("\n" + "=" * 98)
    print("4. EXON COUNT -- the far bins have few junctions; is that the driver?")
    print("=" * 98)
    d["nj_b"] = pd.cut(d["n_junctions"], [-1, 2, 4, 7, 10**6],
                       labels=["<=2", "3-4", "5-7", ">=8"])
    print(f"\n  {'bin':<10}" + "".join(f"{c:>16}" for c in ["<=2", "3-4", "5-7", ">=8"]))
    print(f"  {'-'*10}" + "".join(f"{'-'*16:>16}" for _ in range(4)))
    for lo, hi, nm in BINS:
        g = d[d["bin"].eq(nm)]
        cells = []
        for c in ["<=2", "3-4", "5-7", ">=8"]:
            gg = g[g["nj_b"].eq(c)]
            cells.append(f"{len(gg):>5,} {gg['is_nmd'].mean()*100:>7.1f}%"
                         if len(gg) >= 20 else f"{len(gg):>5,} {'--':>8}")
        print(f"  {nm:<10}" + "".join(f"{c:>16}" for c in cells))
    show("standardised over junction-count band AND expression quintile",
         d, ["nj_b", "expr_q"])

    print("\n" + "=" * 98)
    print("DONE")
    print("=" * 98)


if __name__ == "__main__":
    main()
