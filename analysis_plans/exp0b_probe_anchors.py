#!/usr/bin/env python
"""
The headline did not reproduce under either anchor tried in exp0_verify_headline.py.
Before calling it wrong, enumerate every plausible definition of "the transcripts
with exactly one downstream junction" and see whether ANY of them yields the
published n = 1,389 / 2,084 with NMD+ 10.8% / 46.8%.

Also test the second table in the same document -- the gradient
45.8 / 61.4 / 75.5 / 76.3 / 32.8 % over <=37 / 38-50 / 51-100 / 101-200 / >200 --
which cannot be a partition of the same population as the headline (a weighted
mean of 45.8 and 61.4 cannot be 10.8).

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp0b_probe_anchors.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")

TARGET = {"n_near": 1389, "n_far": 2084, "p_near": 10.8, "p_far": 46.8}
TARGET_GRAD = [45.8, 61.4, 75.5, 76.3, 32.8]


def load_junctions():
    df = pd.read_csv(
        os.path.join(TABLES, "junctions.tsv"), sep="\t", dtype=str, keep_default_na=False
    )
    return {
        iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
              if j not in ("", "NA") else np.empty(0, dtype=np.int64))
        for iso, j in zip(df["isoform_id"], df["junctions"])
    }


def dist_to_next(junc, stop_end):
    i = int(np.searchsorted(junc, stop_end, side="right"))
    return int(junc[i]) - stop_end if i < len(junc) else np.nan


def report(name, sub, cut=50):
    if not len(sub):
        print(f"  {name:<52}  EMPTY")
        return
    near = sub[sub["dist"] <= cut]
    far = sub[sub["dist"] > cut]
    pn = near["is_nmd"].mean() * 100 if len(near) else float("nan")
    pf = far["is_nmd"].mean() * 100 if len(far) else float("nan")
    hit = (
        abs(len(near) - TARGET["n_near"]) <= 25
        and abs(len(far) - TARGET["n_far"]) <= 25
        and abs(pn - TARGET["p_near"]) <= 1.0
        and abs(pf - TARGET["p_far"]) <= 1.0
    )
    mark = "  <== MATCHES PUBLISHED" if hit else ""
    print(f"  {name:<52}  n={len(sub):>6,}  near {len(near):>5,}/{pn:5.1f}%   "
          f"far {len(far):>5,}/{pf:5.1f}%{mark}")


def gradient(sub):
    bins = [(-1, 37), (37, 50), (50, 100), (100, 200), (200, 10**9)]
    out = []
    for lo, hi in bins:
        g = sub[(sub["dist"] > lo) & (sub["dist"] <= hi)]
        out.append((len(g), g["is_nmd"].mean() * 100 if len(g) else float("nan")))
    return out


def show_grad(name, sub):
    g = gradient(sub)
    cells = "  ".join(f"{n:>5,}/{p:5.1f}%" for n, p in g)
    vals = [p for _, p in g]
    hit = all(
        not np.isnan(v) and abs(v - t) <= 2.0 for v, t in zip(vals, TARGET_GRAD)
    )
    print(f"  {name:<40} {cells}{'   <== MATCHES PUBLISHED GRADIENT' if hit else ''}")


def main():
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length", "n_junctions", "max_downstream_ejc"]
    ]
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t")
    allorf = pd.read_csv(
        os.path.join(TABLES, "orf_features.tsv"), sep="\t",
        usecols=["isoform_id", "orf_idx", "orf_start", "orf_end", "orf_length",
                 "n_downstream_ejc", "start_codon", "stop_codon"],
    )
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t")

    print(f"selected_orfs slots {len(sel):,} over {sel['isoform_id'].nunique():,} isoforms")
    print(f"full scan ORFs      {len(allorf):,} over {allorf['isoform_id'].nunique():,} isoforms")
    print(f"tx_summary          {len(tx):,}\n")
    print(f"PUBLISHED TARGET: near n=1,389 @ 10.8%   far n=2,084 @ 46.8%   "
          f"(total 3,473)\n")

    def with_dist(df, end_col):
        d = df.copy()
        d["dist"] = [
            dist_to_next(junc.get(i, np.empty(0, dtype=np.int64)), int(e))
            for i, e in zip(d["isoform_id"], d[end_col])
        ]
        return d.merge(tx, on="isoform_id", how="inner")

    print("=" * 100)
    print("A. ANCHORS THAT PICK ONE ORF PER ISOFORM, then condition on that ORF's count == 1")
    print("=" * 100)

    cands = {}
    cands["slot flagged is_ref_cds"] = sel[sel["is_ref_cds"].astype(bool)]
    cands["slot flagged is_sqanti_cds"] = sel[sel["is_sqanti_cds"].astype(bool)]
    cands["slot 0 (orf_rank == 1)"] = sel[sel["orf_rank"].eq(sel["orf_rank"].min())]
    cands["longest ORF in the 5 slots"] = sel.sort_values(
        "orf_length", ascending=False).drop_duplicates("isoform_id")
    cands["longest ORF in the full scan"] = allorf.sort_values(
        "orf_length", ascending=False).drop_duplicates("isoform_id")
    cands["first ORF in the full scan (5'-most)"] = allorf.sort_values(
        "orf_start").drop_duplicates("isoform_id")

    for name, df in cands.items():
        d = df.drop_duplicates("isoform_id", keep="first")
        d = with_dist(d, "orf_end")
        report(name + ", count==1", d[d["n_downstream_ejc"].eq(1)])

    r = ref[ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()].copy()
    r["orf_end"] = (r["ref_utr5_length"] + r["ref_orf_length"]).astype("Int64")
    r = r[r["orf_end"].notna()]
    r = with_dist(r, "orf_end")
    report("reference-AUG projection, ref_downstream_ejc==1",
           r[r["ref_downstream_ejc"].eq(1)])

    print("\n" + "=" * 100)
    print("B. POOLED OVER SLOTS -- every (isoform, ORF) pair with count == 1")
    print("=" * 100)
    s1 = with_dist(sel[sel["n_downstream_ejc"].eq(1)], "orf_end")
    report("all 5-slot ORFs with count==1 (slot-level rows)", s1)
    report("...deduplicated to one row per isoform",
           s1.drop_duplicates("isoform_id", keep="first"))
    a1 = with_dist(allorf[allorf["n_downstream_ejc"].eq(1)], "orf_end")
    report("all full-scan ORFs with count==1 (ORF-level rows)", a1)
    report("...deduplicated to one row per isoform",
           a1.drop_duplicates("isoform_id", keep="first"))

    print("\n" + "=" * 100)
    print("C. TRANSCRIPT-LEVEL COUNT")
    print("=" * 100)
    t1 = tx[tx["max_downstream_ejc"].eq(1)].merge(
        sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id")[
            ["isoform_id", "orf_end"]], on="isoform_id", how="left")
    t1 = t1[t1["orf_end"].notna()].copy()
    t1["dist"] = [dist_to_next(junc.get(i, np.empty(0, dtype=np.int64)), int(e))
                  for i, e in zip(t1["isoform_id"], t1["orf_end"])]
    report("tx_summary max_downstream_ejc==1, ref-CDS stop", t1)

    print("\n" + "=" * 100)
    print("D. THE PUBLISHED GRADIENT -- which population, if any, produces")
    print("   45.8 / 61.4 / 75.5 / 76.3 / 32.8 % over <=37 / 38-50 / 51-100 / 101-200 / >200")
    print("=" * 100)
    print(f"  {'population':<40} {'<=37':>13}  {'38-50':>13}  {'51-100':>13}  "
          f"{'101-200':>13}  {'>200':>13}")
    refslot = with_dist(
        sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id"), "orf_end")
    show_grad("ref-CDS slot, count==1", refslot[refslot["n_downstream_ejc"].eq(1)])
    show_grad("ref-CDS slot, count>=1", refslot[refslot["n_downstream_ejc"].ge(1)])
    show_grad("ref-CDS slot, any (incl. count==0)", refslot)
    slot0 = with_dist(
        sel[sel["orf_rank"].eq(sel["orf_rank"].min())].drop_duplicates("isoform_id"),
        "orf_end")
    show_grad("slot 0, count>=1", slot0[slot0["n_downstream_ejc"].ge(1)])
    show_grad("slot 0, count==1", slot0[slot0["n_downstream_ejc"].eq(1)])
    show_grad("full-scan ORFs, count>=1 (ORF rows)",
              with_dist(allorf[allorf["n_downstream_ejc"].ge(1)], "orf_end"))
    show_grad("ref-AUG projection, count>=1", r[r["ref_downstream_ejc"].ge(1)])

    print("\n" + "=" * 100)
    print("E. INTERNAL CONSISTENCY OF THE PUBLISHED PAIR OF TABLES")
    print("=" * 100)
    print("  If the gradient partitions the same population as the headline, then")
    print("  pooling its two lowest bins must reproduce the headline's near cell.")
    print("  Published near cell : n = 1,389 at 10.8% NMD+")
    print("  Published <=37 bin  : 45.8%      published 38-50 bin : 61.4%")
    print("  Any weighted mean of 45.8 and 61.4 lies in [45.8, 61.4].")
    print("  10.8 is not in that interval, so the two tables CANNOT describe one")
    print("  population under any bin sizes. At most one of them is right.")


if __name__ == "__main__":
    main()
