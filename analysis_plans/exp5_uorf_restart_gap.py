#!/usr/bin/env python
"""
EXPERIMENT 5 -- after a short upstream ORF, can the ribosome restart in time?

A 40S that has just terminated at a uORF stop can resume scanning and reinitiate
at the main AUG, but it has to reacquire an initiation factor on the way. Short
gap -> reinitiation fails -> the main protein is not made and the transcript
behaves as though its coding potential were absent. Long gap -> it recovers.

Unlike the exon-junction rule this is a genuine dose-response, not a threshold,
so "does the rate vary with distance" is a meaningful question here.

DESIGN
  population   isoforms with a projected reference AUG (ref_atg_available) and a
               5'UTR long enough to hold an ORF
  exposure     gap = (main AUG first base) - (uORF stop last base) - 1, for the
               MOST 3' uORF that terminates before the main AUG -- the one whose
               reinitiation actually decides the outcome
  outcome      is_nmd
  matched on   number of uORFs, and 5'UTR length, both of which mechanically
               constrain the gap. Reported as a direct-standardised rate over
               the joint strata, not as a raw contrast.

THE CONFOUND THAT MATTERS MOST
  If the main CDS is itself a PTC (a junction > 50 nt past its stop) the
  transcript is degraded for that reason and the uORF is irrelevant. Every
  contrast is therefore repeated on the subset whose reference CDS stop is
  clean, which is the only population where a uORF is the candidate trigger.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp5_uorf_restart_gap.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")

GAP_BINS = [(-1, 0, "0 (abutting)"), (0, 10, "1-10"), (10, 25, "11-25"),
            (25, 50, "26-50"), (50, 100, "51-100"), (100, 250, "101-250"),
            (250, 10**9, ">250")]


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                  if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for iso, j in zip(df["isoform_id"], df["junctions"])}


def n_beyond(junc, pos):
    return len(junc) - int(np.searchsorted(junc, pos, side="right"))


def binned(sub, col, bins, label="is_nmd"):
    rows = []
    for lo, hi, nm in bins:
        g = sub[(sub[col] > lo) & (sub[col] <= hi)]
        rows.append((nm, len(g), g[label].mean() * 100 if len(g) else np.nan))
    return rows


def show(title, rows, extra=None):
    print(f"\n  {title}")
    hdr = f"    {'gap (nt)':<14} {'n':>7}  {'NMD+':>7}"
    if extra:
        hdr += "".join(f"  {k:>12}" for k in extra[0])
    print(hdr)
    print(f"    {'-'*14} {'-'*7}  {'-'*7}" + ("".join(f"  {'-'*12}" for _ in extra[0]) if extra else ""))
    for i, (nm, n, p) in enumerate(rows):
        line = f"    {nm:<14} {n:>7,}  " + (f"{p:>6.1f}%" if not np.isnan(p) else f"{'--':>7}")
        if extra:
            line += "".join(f"  {v:>12}" for v in extra[1][i])
        print(line)


def standardised(sub, col, bins, strata_cols, label="is_nmd", min_cell=15):
    """Direct standardisation: NMD rate per gap bin, averaged over strata with
    the strata weighted by their overall size, so the comparison is not driven
    by gap bins sitting in different strata."""
    sub = sub.dropna(subset=strata_cols + [col, label]).copy()
    key = sub[strata_cols].astype(str).agg("|".join, axis=1)
    sub = sub.assign(_k=key)
    w = sub["_k"].value_counts(normalize=True)
    out = []
    for lo, hi, nm in bins:
        g = sub[(sub[col] > lo) & (sub[col] <= hi)]
        if not len(g):
            out.append((nm, 0, np.nan, 0))
            continue
        num, den = 0.0, 0.0
        used = 0
        for k, gg in g.groupby("_k"):
            if len(gg) < min_cell:
                continue
            num += w[k] * gg[label].mean()
            den += w[k]
            used += len(gg)
        out.append((nm, len(g), (num / den * 100) if den > 0 else np.nan, used))
    return out


def show_std(title, rows):
    print(f"\n  {title}")
    print(f"    {'gap (nt)':<14} {'n':>7}  {'crude':>7}  {'standardised':>13}  {'n in cells':>11}")
    print(f"    {'-'*14} {'-'*7}  {'-'*7}  {'-'*13}  {'-'*11}")
    for nm, n, p, used in rows:
        ps = f"{p:>12.1f}%" if not np.isnan(p) else f"{'--':>13}"
        print(f"    {nm:<14} {n:>7,}  {'':>7}  {ps}  {used:>11,}")


def main():
    print("=" * 92)
    print("EXPERIMENT 5 -- uORF-to-main-AUG gap and reinitiation")
    print("=" * 92)

    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length", "n_junctions", "chr"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t")
    orf = pd.read_csv(
        os.path.join(TABLES, "orf_features.tsv"), sep="\t",
        usecols=["isoform_id", "orf_start", "orf_end", "orf_length",
                 "kozak_score", "kozak_m3", "kozak_p4", "stop_codon"])

    print(f"\n  ORF scan: {len(orf):,} ORFs.  orf_length min {orf['orf_length'].min()}, "
          f"5th pct {orf['orf_length'].quantile(0.05):.0f}, median "
          f"{orf['orf_length'].median():.0f}")
    print(f"  ORFs shorter than 33 nt: {int((orf['orf_length'] < 33).sum()):,} "
          f"({(orf['orf_length'] < 33).mean() * 100:.1f}%)")
    print("  LANDMINE: orf_scan_metadata.json for these very tables records")
    print("  \"min_orf_length\": 9, but the data it describes has a hard floor of 33")
    print("  and not one ORF below it. The parameter was set and did not take")
    print("  effect. Part 2 Step 1 turns on lowering this floor, so whoever does")
    print("  that must verify the OUTPUT, not the parameter.")

    # ------------------------------------------------------------------ setup
    r = ref[ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()
            & ref["ref_orf_length"].notna()].copy()
    r["main_atg"] = r["ref_utr5_length"].astype(np.int64) + 1
    r["cds_stop_end"] = (r["ref_utr5_length"] + r["ref_orf_length"]).astype(np.int64)
    r = r[["isoform_id", "gene_id", "main_atg", "cds_stop_end", "ref_utr5_length",
           "ref_utr3_length", "ref_orf_length"]]
    print(f"\n  isoforms with a projected reference AUG: {len(r):,}")

    # the model's own ref-CDS slot stop, used for the PTC filter, because its
    # coordinate convention was verified exactly in exp0 (28,723/28,723)
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_end", "is_ref_cds",
                               "n_downstream_ejc"])
    slot = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id")
    slot = slot.rename(columns={"orf_end": "slot_stop_end"})
    slot["cds_ptc"] = [
        n_beyond(junc.get(i, np.empty(0, dtype=np.int64)), int(e) + 50) > 0
        for i, e in zip(slot["isoform_id"], slot["slot_stop_end"])]
    print(f"  of which the model carries a ref-CDS slot for: "
          f"{slot['isoform_id'].nunique():,}")
    print(f"  reference CDS is itself a PTC (junction >= 50 nt past its stop): "
          f"{int(slot['cds_ptc'].sum()):,}")

    # ----------------------------------------------------------------- uORFs
    o = orf.merge(r[["isoform_id", "main_atg"]], on="isoform_id", how="inner")
    u = o[o["orf_end"] < o["main_atg"]].copy()      # terminates before the main AUG
    u["gap"] = u["main_atg"] - u["orf_end"] - 1
    print(f"\n  ORFs terminating entirely before the main AUG: {len(u):,} "
          f"over {u['isoform_id'].nunique():,} isoforms")

    n_u = u.groupby("isoform_id").size().rename("n_uorf")
    # the MOST 3' one -- the ribosome's last obstacle before the main AUG
    last = u.sort_values("orf_end").drop_duplicates("isoform_id", keep="last")
    last = last[["isoform_id", "gap", "orf_start", "orf_end", "orf_length",
                 "kozak_score", "stop_codon"]].rename(
        columns={"orf_length": "uorf_length", "kozak_score": "uorf_kozak",
                 "stop_codon": "uorf_stop"})

    d = (r.merge(last, on="isoform_id", how="inner")
           .merge(n_u, on="isoform_id", how="left")
           .merge(slot[["isoform_id", "cds_ptc"]], on="isoform_id", how="left")
           .merge(tx, on="isoform_id", how="inner"))
    print(f"  analysis set: {len(d):,} isoforms, NMD+ {d['is_nmd'].mean() * 100:.1f}%")

    d["utr5_q"] = pd.qcut(d["ref_utr5_length"], 5, labels=False, duplicates="drop")
    d["nu_b"] = np.clip(d["n_uorf"], 1, 5)

    # ------------------------------------------------------------- contrasts
    print("\n" + "=" * 92)
    print("A. CRUDE -- gap vs NMD rate, no matching")
    print("=" * 92)
    ex_hdr = ["med 5'UTR", "med n_uORF", "med uORF len"]
    rows = binned(d, "gap", GAP_BINS)
    ex = []
    for lo, hi, nm in GAP_BINS:
        g = d[(d["gap"] > lo) & (d["gap"] <= hi)]
        ex.append([f"{g['ref_utr5_length'].median():,.0f}" if len(g) else "--",
                   f"{g['n_uorf'].median():.0f}" if len(g) else "--",
                   f"{g['uorf_length'].median():,.0f}" if len(g) else "--"])
    show("all isoforms with a uORF", rows, (ex_hdr, ex))
    print("\n  Note the covariate columns: gap is mechanically bounded by 5'UTR")
    print("  length, so the crude contrast is partly a 5'UTR-length contrast.")

    print("\n" + "=" * 92)
    print("B. THE POPULATION WHERE A uORF CAN BE THE TRIGGER")
    print("   (reference CDS stop is clean -- no junction >= 50 nt past it)")
    print("=" * 92)
    clean = d[d["cds_ptc"].eq(False)]
    ptcpos = d[d["cds_ptc"].eq(True)]
    print(f"\n  clean-CDS   n = {len(clean):,}  NMD+ {clean['is_nmd'].mean()*100:.1f}%")
    print(f"  PTC+ CDS    n = {len(ptcpos):,}  NMD+ {ptcpos['is_nmd'].mean()*100:.1f}%")
    show("clean reference CDS -- uORF is the candidate trigger",
         binned(clean, "gap", GAP_BINS))
    show("PTC+ reference CDS -- uORF should NOT matter here (negative control)",
         binned(ptcpos, "gap", GAP_BINS))

    print("\n" + "=" * 92)
    print("C. MATCHED -- standardised over (n_uORF, 5'UTR-length quintile)")
    print("=" * 92)
    show_std("all isoforms with a uORF",
             standardised(d, "gap", GAP_BINS, ["nu_b", "utr5_q"]))
    show_std("clean reference CDS only",
             standardised(clean, "gap", GAP_BINS, ["nu_b", "utr5_q"]))

    print("\n" + "=" * 92)
    print("D. DOES uORF STRENGTH MATTER AT FIXED GAP?")
    print("   (a strong-Kozak uORF captures more scanning ribosomes)")
    print("=" * 92)
    print(f"\n  {'gap bin':<14} {'kozak 0':>16} {'kozak 1':>16} {'kozak 2':>16}")
    print(f"  {'-'*14} {'-'*16} {'-'*16} {'-'*16}")
    for lo, hi, nm in GAP_BINS:
        g = clean[(clean["gap"] > lo) & (clean["gap"] <= hi)]
        cells = []
        for k in (0, 1, 2):
            gg = g[g["uorf_kozak"].eq(k)]
            cells.append(f"{len(gg):>5,} {gg['is_nmd'].mean()*100:>8.1f}%"
                         if len(gg) >= 15 else f"{len(gg):>5,} {'--':>9}")
        print(f"  {nm:<14} " + " ".join(f"{c:>16}" for c in cells))

    print("\n" + "=" * 92)
    print("E. SANITY -- is the gradient just 5'UTR length?")
    print("=" * 92)
    print(f"\n  {'5UTR quintile':<16} {'n':>7} {'med 5UTR':>10} {'NMD+':>8}  "
          f"{'NMD+ gap<=25':>13} {'NMD+ gap>100':>13}")
    print(f"  {'-'*16} {'-'*7} {'-'*10} {'-'*8}  {'-'*13} {'-'*13}")
    for q in sorted(clean["utr5_q"].dropna().unique()):
        g = clean[clean["utr5_q"].eq(q)]
        s, l = g[g["gap"] <= 25], g[g["gap"] > 100]
        ss = f"{s['is_nmd'].mean()*100:.1f}%" if len(s) >= 15 else "--"
        ls = f"{l['is_nmd'].mean()*100:.1f}%" if len(l) >= 15 else "--"
        print(f"  Q{int(q)+1:<15} {len(g):>7,} {g['ref_utr5_length'].median():>10,.0f} "
              f"{g['is_nmd'].mean()*100:>7.1f}%  {ss:>13} {ls:>13}")

    print("\n" + "=" * 92)
    print("F. POSITIVE CONTROLS -- can this population and these labels detect")
    print("   an effect that is already known to be there?")
    print("=" * 92)
    print("\n  A flat result is only informative if the same machinery, on the same")
    print("  rows, moves when something real is varied. Two knowns are used:")
    print("  (i) Track A W161 -- NMD rises with the NUMBER of upstream start codons;")
    print("  (ii) build_mechanism_classes -- an upstream ORF that is itself a PTC")
    print("       carries 12.6% NMD+ against 2.2% with no trigger.")

    print(f"\n  (i) number of uORFs, clean reference CDS")
    print(f"    {'n uORFs':<12} {'n':>7}  {'NMD+':>7}  {'std. over 5UTR quintile':>24}")
    print(f"    {'-'*12} {'-'*7}  {'-'*7}  {'-'*24}")
    w5 = clean["utr5_q"].value_counts(normalize=True)
    for k in [1, 2, 3, 4, 5]:
        g = clean[clean["n_uorf"].eq(k)] if k < 5 else clean[clean["n_uorf"].ge(5)]
        if not len(g):
            continue
        num = den = 0.0
        for q, gg in g.groupby("utr5_q"):
            if len(gg) >= 15:
                num += w5[q] * gg["is_nmd"].mean()
                den += w5[q]
        s = f"{num/den*100:.1f}%" if den > 0 else "--"
        lbl = f"{k}" if k < 5 else ">=5"
        print(f"    {lbl:<12} {len(g):>7,}  {g['is_nmd'].mean()*100:>6.1f}%  {s:>24}")

    print(f"\n  (ii) does the LAST uORF's own stop have a junction >= 50 nt past it?")
    lastu = last.merge(clean[["isoform_id", "is_nmd", "utr5_q", "n_uorf"]],
                       on="isoform_id", how="inner")
    lastu["uorf_ptc"] = [
        n_beyond(junc.get(i, np.empty(0, dtype=np.int64)), int(e) + 50) > 0
        for i, e in zip(lastu["isoform_id"], lastu["orf_end"])]
    for v, nm in ((True, "uORF stop IS a PTC"), (False, "uORF stop is NOT a PTC")):
        g = lastu[lastu["uorf_ptc"].eq(v)]
        print(f"    {nm:<26} n = {len(g):>6,}   NMD+ {g['is_nmd'].mean()*100:>5.1f}%")

    print("\n" + "=" * 92)
    print("G. THE THING THAT DID MOVE -- 5'UTR length at fixed uORF count")
    print("=" * 92)
    print("\n  Logged, not opened. Section E showed a 9-fold gradient across 5'UTR")
    print("  quintiles. The obvious explanation is that a longer 5'UTR simply holds")
    print("  more uORFs, so it is tested here at FIXED uORF count.")
    print(f"\n  {'n uORFs':<10}" + "".join(f"{'Q'+str(q+1):>14}" for q in range(5)))
    print(f"  {'-'*10}" + "".join(f"{'-'*14:>14}" for _ in range(5)))
    for k in [1, 2, 3, 4]:
        g = clean[clean["n_uorf"].eq(k)] if k < 4 else clean[clean["n_uorf"].ge(4)]
        cells = []
        for q in range(5):
            gg = g[g["utr5_q"].eq(q)]
            cells.append(f"{len(gg):>5,} {gg['is_nmd'].mean()*100:>6.1f}%"
                         if len(gg) >= 25 else f"{len(gg):>5,} {'--':>7}")
        lbl = f"{k}" if k < 4 else ">=4"
        print(f"  {lbl:<10}" + "".join(f"{c:>14}" for c in cells))

    print("\n" + "=" * 92)
    print("DONE")
    print("=" * 92)


if __name__ == "__main__":
    main()
