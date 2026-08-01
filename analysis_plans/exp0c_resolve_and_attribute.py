#!/usr/bin/env python
"""
exp0b showed both published tables reproduce, but on DIFFERENT populations:

  headline (1,389 @ 10.8% / 2,084 @ 46.8%)  ->  slot 0, n_downstream_ejc == 1
  gradient (45.8/61.4/75.5/76.3/32.8)       ->  slot 0, n_downstream_ejc >= 1

The documents present them as one analysis. They are not. This script:

  1. Restricts to the model's own universe (the 41,765 isoforms in the HDF5,
     not the 42,043 in the tables) to see whether the residual count gap
     1,397 vs 1,389 and 2,107 vs 2,084 closes exactly.

  2. Reports the honest gradient -- the one computed on the SAME population as
     the headline, count == 1 -- against the published one, since the plan's
     architectural argument ("the action is at 38-200 nt") cites the published
     one.

  3. Attributes the terminal drop at > 200 nt, which DAY_SUMMARY flags as
     "must be attributed before this is reported" and does not attribute.
     The hypothesis under test: in that bin slot 0 is disproportionately NOT
     the real coding sequence, so its stop is not the stop that decides the
     transcript's fate, and the NMD rate falls back toward cohort baseline.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp0c_resolve_and_attribute.py
"""

import os

import h5py
import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
H5 = os.path.join(DN, "nmd_orf_data.h5")

BINS = [(-1, 37, "<=37"), (37, 50, "38-50"), (50, 100, "51-100"),
        (100, 200, "101-200"), (200, 10**9, ">200")]


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                  if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for iso, j in zip(df["isoform_id"], df["junctions"])}


def dist_to_next(junc, stop_end):
    i = int(np.searchsorted(junc, stop_end, side="right"))
    return int(junc[i]) - stop_end if i < len(junc) else np.nan


def grad(sub, label="is_nmd"):
    rows = []
    for lo, hi, nm in BINS:
        g = sub[(sub["dist"] > lo) & (sub["dist"] <= hi)]
        rows.append((nm, len(g), g[label].mean() * 100 if len(g) else float("nan")))
    return rows


def print_grad(title, rows, published=None):
    print(f"\n  {title}")
    hdr = f"    {'bin':<10} {'n':>7}  {'NMD+':>7}"
    if published:
        hdr += f"  {'published':>10}"
    print(hdr)
    print(f"    {'-'*10} {'-'*7}  {'-'*7}" + ("  " + "-"*10 if published else ""))
    for i, (nm, n, p) in enumerate(rows):
        line = f"    {nm:<10} {n:>7,}  {p:>6.1f}%"
        if published:
            line += f"  {published[i]:>9.1f}%"
        print(line)


def main():
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length", "n_junctions"]]
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t")

    with h5py.File(H5, "r") as f:
        h5_iso = np.array([x.decode() if isinstance(x, bytes) else str(x)
                           for x in f["isoform_id"][:]])
    h5_set = set(h5_iso)
    print(f"HDF5 model universe : {len(h5_iso):,} isoforms "
          f"({len(h5_set):,} unique)")
    print(f"Table universe      : {tx['isoform_id'].nunique():,} isoforms")
    print(f"In tables, not in H5: {tx['isoform_id'].nunique() - len(h5_set & set(tx['isoform_id'])):,}")

    slot0 = sel[sel["orf_rank"].eq(sel["orf_rank"].min())].drop_duplicates(
        "isoform_id", keep="first").copy()
    slot0["dist"] = [dist_to_next(junc.get(i, np.empty(0, dtype=np.int64)), int(e))
                     for i, e in zip(slot0["isoform_id"], slot0["orf_end"])]
    slot0 = slot0.merge(tx, on="isoform_id", how="inner", suffixes=("", "_tx"))

    print("\n" + "=" * 90)
    print("1. DOES THE HEADLINE CLOSE EXACTLY ON THE MODEL'S OWN UNIVERSE?")
    print("=" * 90)
    print(f"  {'population':<34} {'near n':>8} {'near %':>8} {'far n':>8} {'far %':>8}")
    print(f"  {'-'*34} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'PUBLISHED':<34} {1389:>8,} {10.8:>7.1f}% {2084:>8,} {46.8:>7.1f}%")
    for nm, pop in (("all 42,043 table isoforms", slot0),
                    ("restricted to the 41,765 H5", slot0[slot0["isoform_id"].isin(h5_set)])):
        one = pop[pop["n_downstream_ejc"].eq(1)]
        near, far = one[one["dist"] <= 50], one[one["dist"] > 50]
        print(f"  {nm:<34} {len(near):>8,} {near['is_nmd'].mean()*100:>7.1f}% "
              f"{len(far):>8,} {far['is_nmd'].mean()*100:>7.1f}%")

    h5only = slot0[slot0["isoform_id"].isin(h5_set)]

    print("\n" + "=" * 90)
    print("2. THE GRADIENT -- published vs. the one on the headline's own population")
    print("=" * 90)
    print_grad("slot 0, count >= 1  (what the published gradient actually is)",
               grad(h5only[h5only["n_downstream_ejc"].ge(1)]),
               published=[45.8, 61.4, 75.5, 76.3, 32.8])
    print_grad("slot 0, count == 1  (the headline's population -- count held fixed)",
               grad(h5only[h5only["n_downstream_ejc"].eq(1)]))
    print("\n  The published gradient does NOT hold the model's feature fixed. In it")
    print("  the junction count varies with distance, and count is the dominant")
    print("  predictor -- so part of that gradient is count, not distance.")
    cnt = h5only[h5only["n_downstream_ejc"].ge(1)]
    print(f"\n  mean n_downstream_ejc by bin, in the count>=1 population:")
    print(f"    {'bin':<10} {'n':>7}  {'mean count':>11}")
    for lo, hi, nm in BINS:
        g = cnt[(cnt["dist"] > lo) & (cnt["dist"] <= hi)]
        if len(g):
            print(f"    {nm:<10} {len(g):>7,}  {g['n_downstream_ejc'].mean():>11.2f}")

    print("\n" + "=" * 90)
    print("3. ATTRIBUTING THE TERMINAL DROP AT > 200 nt  (count == 1 population)")
    print("=" * 90)
    one = h5only[h5only["n_downstream_ejc"].eq(1)].copy()
    print(f"\n  {'bin':<10} {'n':>6} {'NMD+':>7} {'is_ref_cds':>11} {'is_sqanti':>10} "
          f"{'med ORF len':>12} {'med tx len':>11} {'med orf_end':>12} {'med n_junc':>11}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*11} {'-'*10} {'-'*12} {'-'*11} {'-'*12} {'-'*11}")
    for lo, hi, nm in BINS:
        g = one[(one["dist"] > lo) & (one["dist"] <= hi)]
        if not len(g):
            continue
        print(f"  {nm:<10} {len(g):>6,} {g['is_nmd'].mean()*100:>6.1f}% "
              f"{g['is_ref_cds'].astype(bool).mean()*100:>10.1f}% "
              f"{g['is_sqanti_cds'].astype(bool).mean()*100:>9.1f}% "
              f"{g['orf_length'].median():>12,.0f} {g['tx_length'].median():>11,.0f} "
              f"{g['orf_end'].median():>12,.0f} {g['n_junctions'].median():>11,.0f}")

    print("\n  Same gradient, split by whether slot 0 IS the reference CDS:")
    for flag, nm2 in ((True, "slot 0 IS the reference CDS"),
                      (False, "slot 0 is NOT the reference CDS")):
        sub = one[one["is_ref_cds"].astype(bool).eq(flag)]
        print_grad(f"{nm2}   (n = {len(sub):,})", grad(sub))

    print("\n" + "=" * 90)
    print("4. THE CLAIM THAT WAS MADE ABOUT THE DROP")
    print("=" * 90)
    print('  DAY_SUMMARY: "it falls within class (PTC+ 78.7% -> 34.1%), so it is a')
    print('  real ceiling on the rule". PTC+ here means junction > 50 nt past the stop,')
    print('  which every isoform in the 51-100, 101-200 and >200 bins satisfies BY')
    print('  CONSTRUCTION. Conditioning on PTC+ therefore removes nothing from those')
    print('  three bins -- it is not a control. Recomputed, count >= 1, PTC+ only:')
    ptc = cnt[cnt["dist"] > 50]
    print_grad("count >= 1 and dist > 50 nt (the 'within PTC+' comparison)",
               grad(ptc))

    print("\n" + "=" * 90)
    print("5. WHAT ELSE CHANGES AT > 200 nt -- 3'UTR length")
    print("=" * 90)
    one = one.copy()
    one["utr3"] = one["tx_length"] - one["orf_end"]
    print(f"  {'bin':<10} {'n':>6} {'NMD+':>7} {'med 3UTR':>10} "
          f"{'frac 3UTR>1kb':>14} {'NMD+ | 3UTR<=1kb':>18} {'NMD+ | 3UTR>1kb':>17}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*10} {'-'*14} {'-'*18} {'-'*17}")
    for lo, hi, nm in BINS:
        g = one[(one["dist"] > lo) & (one["dist"] <= hi)]
        if not len(g):
            continue
        s, l = g[g["utr3"] <= 1000], g[g["utr3"] > 1000]
        ss = f"{s['is_nmd'].mean()*100:.1f}%" if len(s) else "--"
        ls = f"{l['is_nmd'].mean()*100:.1f}%" if len(l) else "--"
        print(f"  {nm:<10} {len(g):>6,} {g['is_nmd'].mean()*100:>6.1f}% "
              f"{g['utr3'].median():>10,.0f} {l.shape[0]/len(g)*100:>13.1f}% "
              f"{ss:>18} {ls:>17}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
