#!/usr/bin/env python
"""
Independent recomputation of the 2026-07-31 headline result.

The claim, as written in DAY_SUMMARY_2026-07-31.md and repeated in three other
committed documents:

    Holding n_downstream_ejc == 1 exactly constant, and varying only the
    stop-to-junction distance the feature cannot express:
        <= 50 nt : n = 1,389   NMD+ 10.8%
        >  50 nt : n = 2,084   NMD+ 46.8%
    Full gradient (<=37 / 38-50 / 51-100 / 101-200 / >200 nt):
        45.8 / 61.4 / 75.5 / 76.3 / 32.8 %

No committed script in either repository produces these numbers, and no worklog
entry records them (W162 records a DIFFERENT contrast on the same defect). They
are therefore untraced under this project's own rule. This script recomputes
them from the input TSVs with independent code.

Two anchors are computed because the source documents do not say which was used:

  A  reference-AUG projection  -- ref_cds_features.tsv, ref_downstream_ejc,
     written by 05t_ref_cds_features.R. This is the anchor used for the
     mechanism classes in build_mechanism_classes.py.
  B  the model's own slot      -- selected_orfs.tsv where is_ref_cds is true,
     n_downstream_ejc, written by 05s_orfik_scan.R. This is literally the
     model's input feature.

Both counts are thresholdless by the defect under investigation, so conditioning
on == 1 holds the model's input exactly fixed in either case.

Coordinate convention is not assumed -- it is VERIFIED. The script reconstructs
each anchor's stop coordinate, recounts junctions beyond it, and requires the
reconstruction to reproduce the file's own count before any contrast is read.

Outputs to stdout. Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp0_verify_headline.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
SELECTED = os.path.expanduser(
    "~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn/selected_orfs.tsv"
)


def rule(msg=""):
    print("\n" + "=" * 78)
    if msg:
        print(msg)
        print("=" * 78)


def load_junctions():
    """isoform_id -> np.array of junction positions, sorted ascending."""
    df = pd.read_csv(
        os.path.join(TABLES, "junctions.tsv"), sep="\t", dtype=str, keep_default_na=False
    )
    out = {}
    for iso, j in zip(df["isoform_id"], df["junctions"]):
        if j == "" or j == "NA":
            out[iso] = np.empty(0, dtype=np.int64)
        else:
            out[iso] = np.sort(np.fromstring(j, sep=",", dtype=np.int64))
    return out


def count_beyond(junc, stop_end, offset=0):
    """#{ j : j > stop_end + offset }.  offset=0 reproduces the thresholdless
    count in 05s_orfik_scan.R:227 `sum(junctions > oe)`."""
    return int(np.searchsorted(junc, stop_end + offset, side="right").__rsub__(len(junc)))


def nearest_beyond(junc, stop_end):
    """Position of the first junction strictly greater than stop_end, or None."""
    i = int(np.searchsorted(junc, stop_end, side="right"))
    return int(junc[i]) if i < len(junc) else None


def contrast(sub, dist_col, label_col, cut, name):
    """The 2x2 the headline reports, plus the full gradient."""
    near = sub[sub[dist_col] <= cut]
    far = sub[sub[dist_col] > cut]
    print(f"\n  {name}")
    print(f"    {'stop -> nearest downstream junction':<38} {'n':>7}  {'NMD+':>7}")
    print(f"    {'-' * 38} {'-' * 7}  {'-' * 7}")
    for tag, g in ((f"<= {cut} nt", near), (f">  {cut} nt", far)):
        if len(g):
            print(f"    {tag:<38} {len(g):>7,}  {g[label_col].mean() * 100:>6.1f}%")
        else:
            print(f"    {tag:<38} {0:>7,}  {'--':>7}")
    if len(near) and len(far):
        r0, r1 = near[label_col].mean(), far[label_col].mean()
        if r0 > 0:
            print(f"    ratio far/near: {r1 / r0:.2f}x")

    bins = [(-1, 37), (37, 50), (50, 100), (100, 200), (200, 10**9)]
    names = ["<=37", "38-50", "51-100", "101-200", ">200"]
    print(f"\n    full gradient")
    print(f"    {'bin (nt)':<12} {'n':>7}  {'NMD+':>7}")
    print(f"    {'-' * 12} {'-' * 7}  {'-' * 7}")
    for (lo, hi), nm in zip(bins, names):
        g = sub[(sub[dist_col] > lo) & (sub[dist_col] <= hi)]
        if len(g):
            print(f"    {nm:<12} {len(g):>7,}  {g[label_col].mean() * 100:>6.1f}%")
        else:
            print(f"    {nm:<12} {0:>7,}  {'--':>7}")


def main():
    rule("PROVENANCE")
    prov = json.load(open(os.path.join(TABLES, "tx_summary_provenance.json")))
    for k, v in prov.items():
        print(f"  {k}: {v}")
    meta = json.load(open(os.path.join(TABLES, "orf_scan_metadata.json")))
    print(f"  orf scan: n_transcripts={meta['n_transcripts']:,} "
          f"n_orfs={meta['n_orfs']:,} min_orf_length={meta['min_orf_length']}")
    print(f"  tables dir: {TABLES}")
    print(f"  selected_orfs: {SELECTED}")

    junc = load_junctions()
    print(f"  junction table: {len(junc):,} isoforms, "
          f"{sum(len(v) for v in junc.values()):,} junctions total")

    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")
    tx = tx[["isoform_id", "is_nmd", "tx_length", "chr", "n_junctions"]]
    print(f"  tx_summary: {len(tx):,} isoforms, NMD+ {tx['is_nmd'].mean() * 100:.1f}%")

    # ---------------------------------------------------------------- anchor A
    rule("ANCHOR A -- reference-AUG projection (ref_cds_features.tsv)")
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t")
    print(f"  rows: {len(ref):,}")
    print(f"  ref_atg_available: {int(ref['ref_atg_available'].sum()):,}")
    ref = ref[
        ref["ref_atg_available"].eq(1)
        & ref["ref_utr5_length"].notna()
        & ref["ref_orf_length"].notna()
    ].copy()
    print(f"  with usable 5'UTR + ORF length: {len(ref):,}")

    # stop coordinate: the ORF spans [utr5+1, utr5+orf_length] in tx coords,
    # so the last base of the stop codon is at utr5_length + orf_length.
    ref["stop_end"] = (ref["ref_utr5_length"] + ref["ref_orf_length"]).astype(np.int64)
    ref["recount"] = [
        count_beyond(junc.get(i, np.empty(0, dtype=np.int64)), s)
        for i, s in zip(ref["isoform_id"], ref["stop_end"])
    ]

    ok = ref["recount"].eq(ref["ref_downstream_ejc"])
    print(f"\n  COORDINATE CHECK: recomputed #{{j > stop_end}} vs the file's "
          f"ref_downstream_ejc")
    print(f"    agree: {int(ok.sum()):,} / {len(ref):,}  ({ok.mean() * 100:.2f}%)")
    if ok.mean() < 0.999:
        d = (ref["recount"] - ref["ref_downstream_ejc"])
        print(f"    disagreement distribution: "
              f"{d[~ok].value_counts().head(6).to_dict()}")
        print("    -> convention NOT confirmed; probing alternatives")
        for off in (-3, -2, -1, 1, 2, 3):
            alt = [
                count_beyond(junc.get(i, np.empty(0, dtype=np.int64)), s + off)
                for i, s in zip(ref["isoform_id"], ref["stop_end"])
            ]
            a = np.mean(np.asarray(alt) == ref["ref_downstream_ejc"].values)
            print(f"      stop_end {off:+d}: {a * 100:.2f}%")

    ref = ref.merge(tx, on="isoform_id", how="inner")
    print(f"  joined to labels: {len(ref):,}")

    one = ref[ref["ref_downstream_ejc"].eq(1) & ok.reindex(ref.index, fill_value=False)]
    # recompute the mask on the merged frame instead of reindexing across a merge
    one = ref[ref["ref_downstream_ejc"].eq(1) & ref["recount"].eq(1)].copy()
    one["dist"] = [
        nearest_beyond(junc[i], s) - s
        for i, s in zip(one["isoform_id"], one["stop_end"])
    ]
    print(f"\n  isoforms with EXACTLY ONE junction past the reference stop: {len(one):,}")
    contrast(one, "dist", "is_nmd", 50, "anchor A, cut at d > 50")
    contrast(one, "dist", "is_nmd", 49, "anchor A, cut at d >= 50 (Isopair's own rule)")

    # ---------------------------------------------------------------- anchor B
    rule("ANCHOR B -- the model's own slot (selected_orfs.tsv, is_ref_cds)")
    sel = pd.read_csv(SELECTED, sep="\t")
    print(f"  rows: {len(sel):,}  isoforms: {sel['isoform_id'].nunique():,}")
    slot = sel[sel["is_ref_cds"].astype(bool)].copy()
    print(f"  slots flagged is_ref_cds: {len(slot):,} "
          f"over {slot['isoform_id'].nunique():,} isoforms")
    slot = slot.drop_duplicates("isoform_id", keep="first")

    slot["recount"] = [
        count_beyond(junc.get(i, np.empty(0, dtype=np.int64)), int(e))
        for i, e in zip(slot["isoform_id"], slot["orf_end"])
    ]
    okb = slot["recount"].eq(slot["n_downstream_ejc"])
    print(f"\n  COORDINATE CHECK: recomputed #{{j > orf_end}} vs the file's "
          f"n_downstream_ejc")
    print(f"    agree: {int(okb.sum()):,} / {len(slot):,}  ({okb.mean() * 100:.2f}%)")

    slot = slot.merge(tx, on="isoform_id", how="inner", suffixes=("", "_tx"))
    oneb = slot[slot["n_downstream_ejc"].eq(1) & slot["recount"].eq(1)].copy()
    oneb["dist"] = [
        nearest_beyond(junc[i], int(e)) - int(e)
        for i, e in zip(oneb["isoform_id"], oneb["orf_end"])
    ]
    print(f"\n  isoforms with EXACTLY ONE junction past the model's ref-CDS slot stop: "
          f"{len(oneb):,}")
    contrast(oneb, "dist", "is_nmd", 50, "anchor B, cut at d > 50")

    # ------------------------------------------------------- the whole cohort
    rule("CONTEXT -- the same cut without conditioning on the count")
    allref = ref[ref["recount"].eq(ref["ref_downstream_ejc"])].copy()
    allref["nearest"] = [
        nearest_beyond(junc.get(i, np.empty(0, dtype=np.int64)), s)
        for i, s in zip(allref["isoform_id"], allref["stop_end"])
    ]
    allref["dist"] = allref["nearest"].astype("float") - allref["stop_end"]
    nojunc = allref[allref["nearest"].isna()]
    hasj = allref[allref["nearest"].notna()]
    print(f"  no junction past the stop        : {len(nojunc):>7,}  "
          f"NMD+ {nojunc['is_nmd'].mean() * 100:.1f}%")
    print(f"  >=1 junction past the stop       : {len(hasj):>7,}  "
          f"NMD+ {hasj['is_nmd'].mean() * 100:.1f}%")
    for lo, hi, nm in ((0, 50, "  nearest <= 50 nt"), (50, 10**9, "  nearest >  50 nt")):
        g = hasj[(hasj["dist"] > lo) & (hasj["dist"] <= hi)]
        print(f"  {nm:<32} : {len(g):>7,}  NMD+ {g['is_nmd'].mean() * 100:.1f}%")

    rule("COUNT CONFOUND -- is the 'exactly one' stratum representative?")
    print(f"  {'ref_downstream_ejc':<20} {'n':>7}  {'NMD+':>7}  "
          f"{'NMD+ | d<=50':>13}  {'NMD+ | d>50':>12}")
    print(f"  {'-' * 20} {'-' * 7}  {'-' * 7}  {'-' * 13}  {'-' * 12}")
    for k in [0, 1, 2, 3, 4, 5]:
        g = allref[allref["ref_downstream_ejc"].eq(k)]
        if not len(g):
            continue
        near = g[g["dist"] <= 50]
        far = g[g["dist"] > 50]
        n_s = f"{near['is_nmd'].mean() * 100:.1f}%" if len(near) else "--"
        f_s = f"{far['is_nmd'].mean() * 100:.1f}%" if len(far) else "--"
        print(f"  {k:<20} {len(g):>7,}  {g['is_nmd'].mean() * 100:>6.1f}%  "
              f"{n_s:>13}  {f_s:>12}")
    g = allref[allref["ref_downstream_ejc"] >= 6]
    if len(g):
        near, far = g[g["dist"] <= 50], g[g["dist"] > 50]
        n_s = f"{near['is_nmd'].mean() * 100:.1f}%" if len(near) else "--"
        f_s = f"{far['is_nmd'].mean() * 100:.1f}%" if len(far) else "--"
        print(f"  {'>=6':<20} {len(g):>7,}  {g['is_nmd'].mean() * 100:>6.1f}%  "
              f"{n_s:>13}  {f_s:>12}")

    rule("DONE")


if __name__ == "__main__":
    main()
