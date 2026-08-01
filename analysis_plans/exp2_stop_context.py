#!/usr/bin/env python
"""
EXPERIMENT 2 -- does it matter which stop codon a transcript uses, and what
follows it?

WHY THIS ONE IS CLEANLY TESTABLE
  TAG and TGA are anagrams: {T, A, G} both. Swapping one for the other holds
  base composition, GC content and length exactly fixed BY CONSTRUCTION. Almost
  no other sequence contrast in this project has that property. TAA is not
  composition-matched to either and is reported separately rather than pooled.

THE PUBLISHED CLAIM UNDER TEST
  Manuscript section 5.3.3 states that TGA turns up more often in degraded
  transcripts. That is a specific, falsifiable statement.

PRE-REGISTERED PREDICTION, WRITTEN BEFORE LOOKING
  Termination efficiency is known to run TAA > TAG > TGA, and a +4 purine
  terminates more efficiently than a +4 pyrimidine; TGA followed by C is the
  leakiest context known. Leaky termination means readthrough, and a ribosome
  that reads through displaces the downstream exon junction complexes that
  would otherwise trigger decay. So the mechanistic prediction is that the
  LEAKY contexts carry LESS NMD, not more:
      TGA < TAG < TAA   and   +4 C < +4 T < +4 A,G
  This runs OPPOSITE to the published claim. Whichever way it comes out, one of
  the two is wrong, which is the point of running it.

THE CONTROL THAT MAKES THE +4 RESULT MEAN ANYTHING
  A base identity effect at +4 is only interpretable against the same
  measurement at positions where no mechanism exists. Positions +10, +25 and
  +50 into the 3'UTR are measured identically. If +4 behaves like them, there
  is no stop-context effect -- only a regional composition effect.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp2_stop_context.py
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
    ids, blob, off = z["ids"], z["blob"], z["offsets"]
    idx = {s: i for i, s in enumerate(ids)}
    return ids, blob, off, idx


def sub(blob, off, i, a, b):
    """bases [a, b) in 1-based transcript coordinates, '' if out of range"""
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


def standardised(d, group_col, strata, label="is_nmd", min_cell=20):
    """NMD rate per group, direct-standardised over the joint strata."""
    d = d.dropna(subset=[group_col] + strata + [label])
    key = d[strata].astype(str).agg("|".join, axis=1)
    d = d.assign(_k=key)
    w = d["_k"].value_counts(normalize=True)
    out = {}
    for g, gg in d.groupby(group_col):
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


def boot_diff(d, col, a, b, strata, gene_col="gene_id", n=2000, label="is_nmd",
              min_cell=10):
    """Gene-clustered bootstrap of the standardised a-minus-b difference.

    Clustered by gene because transcripts of one gene are not independent.
    Magnitude, measured 2026-08-01: is_nmd ICC within gene 0.300 at an effective
    cluster size of 3.36, so the design effect is 1.71 and an interval widens by
    1.31x. The 5.47 this docstring used to cite (from the brief) was never a
    variance inflation -- it is `sd 5.47`, a percentage-point standard deviation
    from the power-matching section of exp2b's runlog.

    Implemented on integer-coded arrays with bincount rather than pandas
    groupby: a resample has to be taken thousands of times and rebuilding a
    DataFrame per draw is what made the first version unusable.
    """
    d = d[d[col].isin([a, b])].dropna(subset=strata + [label, gene_col]).copy()
    if not len(d):
        return (np.nan,) * 4 + (0,)
    kcode, _ = pd.factorize(d[strata].astype(str).agg("|".join, axis=1))
    gcode, guniq = pd.factorize(d[gene_col])
    arm = d[col].eq(a).to_numpy().astype(np.int64)      # 1 = a, 0 = b
    y = d[label].to_numpy().astype(np.float64)
    nk = int(kcode.max()) + 1

    # rows belonging to each gene, as one flat array + offsets
    order = np.argsort(gcode, kind="stable")
    gsorted = gcode[order]
    starts = np.searchsorted(gsorted, np.arange(len(guniq)), side="left")
    ends = np.searchsorted(gsorted, np.arange(len(guniq)), side="right")
    gene_rows = [order[s:e] for s, e in zip(starts, ends)]

    def stat(rows):
        kk, aa, yy = kcode[rows], arm[rows], y[rows]
        n_k = np.bincount(kk, minlength=nk).astype(np.float64)
        w = n_k / n_k.sum()
        vals = []
        for side in (1, 0):
            m = aa == side
            cnt = np.bincount(kk[m], minlength=nk).astype(np.float64)
            s = np.bincount(kk[m], weights=yy[m], minlength=nk)
            good = cnt >= min_cell
            den = w[good].sum()
            vals.append((w[good] * (s[good] / cnt[good])).sum() / den
                        if den > 0 else np.nan)
        return (vals[0] - vals[1]) * 100

    point = stat(np.arange(len(d)))
    draws = np.empty(n)
    ng = len(guniq)
    for t in range(n):
        pick = RNG.integers(0, ng, size=ng)
        draws[t] = stat(np.concatenate([gene_rows[g] for g in pick]))
    draws = draws[~np.isnan(draws)]
    if len(draws) < 20:
        return point, np.nan, np.nan, np.nan, len(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return point, lo, hi, max(p, 1.0 / len(draws)), len(draws)


def main():
    print("=" * 96)
    print("EXPERIMENT 2 -- stop codon identity and the base that follows it")
    print("=" * 96)

    ids, blob, off, idx = load_seqs()
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length", "chr"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id"])
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_end", "orf_start", "orf_length",
                               "is_ref_cds", "n_downstream_ejc", "stop_codon",
                               "orf_rank"])

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.merge(tx, on="isoform_id").merge(ref.drop_duplicates("isoform_id"),
                                           on="isoform_id", how="left")
    print(f"\n  anchor: the model's reference-CDS slot, {len(d):,} isoforms")

    # ------------------------------------------------- coordinate verification
    rows = d["isoform_id"].map(idx)
    ok = rows.notna()
    d = d[ok].copy()
    d["_i"] = rows[ok].astype(int)
    d["stop_seq"] = [sub(blob, off, i, e - 2, e + 1)
                     for i, e in zip(d["_i"], d["orf_end"])]
    agree = d["stop_seq"].eq(d["stop_codon"])
    print(f"  COORDINATE CHECK: bases [orf_end-2, orf_end] from the FASTA vs the")
    print(f"    table's stop_codon column: {int(agree.sum()):,} / {len(d):,} "
          f"({agree.mean()*100:.2f}%)")
    if agree.mean() < 0.99:
        print(f"    mismatch examples: "
              f"{d.loc[~agree, ['stop_codon', 'stop_seq']].head(5).values.tolist()}")
        for shift in (-2, -1, 1, 2):
            alt = [sub(blob, off, i, e - 2 + shift, e + 1 + shift)
                   for i, e in zip(d["_i"], d["orf_end"])]
            print(f"    shift {shift:+d}: "
                  f"{np.mean(np.array(alt) == d['stop_codon'].values)*100:.2f}%")
    d = d[agree].copy()
    print(f"  proceeding on the {len(d):,} with a verified stop coordinate")

    # ------------------------------------------------------------ covariates
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d["ptc"] = [len(junc.get(i, np.empty(0, dtype=np.int64)))
                - int(np.searchsorted(junc.get(i, np.empty(0, dtype=np.int64)),
                                      int(e) + 50, side="right")) > 0
                for i, e in zip(d["isoform_id"], d["orf_end"])]
    d = d[d["utr3"] >= 60].copy()          # need +50 of real 3'UTR to read
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop")
    for k in (4, 10, 25, 50):
        d[f"p{k}"] = [sub(blob, off, i, e + k, e + k + 1)
                      for i, e in zip(d["_i"], d["orf_end"])]
    gc = []
    for i, e in zip(d["_i"], d["orf_end"]):
        s = sub(blob, off, i, e + 1, e + 101)
        gc.append((s.count("G") + s.count("C")) / len(s) if s else np.nan)
    d["gc100"] = gc
    n_before = len(d)
    d = d[d["gc100"].notna()].copy()
    d["gc_q"] = pd.qcut(d["gc100"], 4, labels=False, duplicates="drop")
    d["ptc"] = d["ptc"].astype(int)
    d["utr3_q"] = d["utr3_q"].astype(int)
    d["gc_q"] = d["gc_q"].astype(int)
    print(f"  with >= 60 nt of 3'UTR to read: {n_before:,}, of which "
          f"{len(d):,} have a readable +100 window")
    print(f"  PTC+ {d['ptc'].mean()*100:.1f}%   NMD+ {d['is_nmd'].mean()*100:.1f}%")

    STRATA = ["ptc", "utr3_q", "gc_q"]
    print(f"  strata for every contrast below: {STRATA}  "
          f"({d[STRATA].astype(str).agg('|'.join, axis=1).nunique()} cells)")

    # ------------------------------------------------------ the published claim
    print("\n" + "=" * 96)
    print("A. THE PUBLISHED CLAIM -- is TGA more common in degraded transcripts?")
    print("=" * 96)
    ct = pd.crosstab(d["stop_codon"], d["is_nmd"], normalize="columns") * 100
    n_ct = pd.crosstab(d["stop_codon"], d["is_nmd"])
    print(f"\n  {'stop':<6} {'n non-NMD':>11} {'% non-NMD':>11} {'n NMD+':>9} "
          f"{'% NMD+':>9} {'difference':>12}")
    print(f"  {'-'*6} {'-'*11} {'-'*11} {'-'*9} {'-'*9} {'-'*12}")
    for s in ["TAA", "TAG", "TGA"]:
        if s in ct.index:
            print(f"  {s:<6} {n_ct.loc[s, 0]:>11,} {ct.loc[s, 0]:>10.1f}% "
                  f"{n_ct.loc[s, 1]:>9,} {ct.loc[s, 1]:>8.1f}% "
                  f"{ct.loc[s, 1] - ct.loc[s, 0]:>+11.1f}pp")
    print("\n  This is the comparison the manuscript makes. It is confounded --")
    print("  see B, where the same contrast is run inside matched strata.")

    print("\n" + "=" * 96)
    print("B. NMD RATE BY STOP CODON, CRUDE AND MATCHED")
    print("=" * 96)
    res = standardised(d, "stop_codon", STRATA)
    print(f"\n  {'stop':<6} {'n':>8} {'crude NMD+':>12} {'standardised':>14} "
          f"{'n in cells':>11}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*14} {'-'*11}")
    for s in ["TAA", "TAG", "TGA"]:
        if s in res:
            n, c, st, u = res[s]
            print(f"  {s:<6} {n:>8,} {c:>11.1f}% {st:>13.1f}% {u:>11,}")

    print("\n  TAG vs TGA -- the composition-matched pair, gene-clustered bootstrap")
    pt, lo, hi, p, nb = boot_diff(d, "stop_codon", "TGA", "TAG", STRATA)
    print(f"    standardised NMD+(TGA) - NMD+(TAG) = {pt:+.2f}pp   "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]   p = {p:.3f}   ({nb} draws)")
    print(f"    pre-registered prediction was NEGATIVE (TGA leakier -> less NMD).")
    print(f"    the manuscript's claim implies POSITIVE.")

    # ------------------------------------------------------------ the +4 base
    print("\n" + "=" * 96)
    print("C. THE BASE AFTER THE STOP, AGAINST THREE CONTROL POSITIONS")
    print("=" * 96)
    print("\n  standardised NMD+ (%) by base identity, same strata throughout.")
    print("  +4 is the mechanistic position. +10, +25, +50 are controls with no")
    print("  proposed mechanism -- they measure what this contrast returns when")
    print("  nothing is there.\n")
    print(f"  {'position':<10}" + "".join(f"{b:>10}" for b in "ACGT")
          + f"{'spread':>10}{'n':>10}")
    print(f"  {'-'*10}" + "".join(f"{'-'*10:>10}" for _ in "ACGT")
          + f"{'-'*10:>10}{'-'*10:>10}")
    for k in (4, 10, 25, 50):
        r = standardised(d, f"p{k}", STRATA)
        vals = {b: r[b][2] for b in "ACGT" if b in r}
        cells = "".join(f"{vals.get(b, np.nan):>9.1f}%" if b in vals
                        else f"{'--':>10}" for b in "ACGT")
        v = [x for x in vals.values() if not np.isnan(x)]
        spread = max(v) - min(v) if v else np.nan
        tot = sum(r[b][0] for b in "ACGT" if b in r)
        lbl = f"+{k}" + ("  <-- stop" if k == 4 else "")
        print(f"  {lbl:<10}{cells}{spread:>9.1f}p{tot:>10,}")

    print("\n  Composition-matched single contrast at each position, A vs T")
    print("  (equal GC, so the recomputed GC channel cannot drive it):")
    for k in (4, 10, 25, 50):
        pt, lo, hi, p, _ = boot_diff(d, f"p{k}", "A", "T", STRATA, n=800)
        flag = "  <-- stop" if k == 4 else ""
        print(f"    +{k:<3} A - T = {pt:+6.2f}pp  95% CI [{lo:+6.2f}, {hi:+6.2f}]  "
              f"p = {p:.3f}{flag}")

    # ------------------------------------------------------- the tetranucleotide
    print("\n" + "=" * 96)
    print("D. THE STOP TETRANUCLEOTIDE -- stop codon x +4 jointly")
    print("=" * 96)
    d["tetra"] = d["stop_codon"] + d["p4"]
    r = standardised(d, "tetra", STRATA, min_cell=15)
    print(f"\n  {'tetra':<8} {'n':>7} {'crude':>8} {'standardised':>14}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*14}")
    for t in sorted(r, key=lambda x: (np.isnan(r[x][2]), r[x][2])):
        n, c, st, u = r[t]
        if n < 100:
            continue
        s = f"{st:>13.1f}%" if not np.isnan(st) else f"{'--':>14}"
        print(f"  {t:<8} {n:>7,} {c:>7.1f}% {s}")
    print("\n  TGA-C is the leakiest context in the termination literature.")

    print("\n" + "=" * 96)
    print("E. SPLIT BY PTC STATUS -- readthrough can only matter where there is")
    print("   something downstream to read through")
    print("=" * 96)
    for v, nm in ((True, "PTC+ (junction >= 50 nt past the stop)"),
                  (False, "PTC- (no such junction)")):
        g = d[d["ptc"].eq(v)]
        r = standardised(g, "stop_codon", ["utr3_q", "gc_q"])
        line = "  ".join(
            f"{s} {r[s][2]:.1f}% (n={r[s][0]:,})" for s in ["TAA", "TAG", "TGA"]
            if s in r)
        print(f"\n  {nm}  n = {len(g):,}, NMD+ {g['is_nmd'].mean()*100:.1f}%")
        print(f"    {line}")
        pt, lo, hi, p, _ = boot_diff(g, "stop_codon", "TGA", "TAG",
                                     ["utr3_q", "gc_q"], n=800)
        print(f"    TGA - TAG = {pt:+.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]  p = {p:.3f}")

    print("\n" + "=" * 96)
    print("F. THE CONTROL THAT DECIDES WHETHER B IS REAL")
    print("=" * 96)
    print("\n  Added after Experiment 4 measured this machinery's noise floor and")
    print("  found it is NOT zero: null arms with no possible mechanism returned")
    print("  point estimates of 2-5pp, one of them at p = 0.027. B's +2.00pp sits")
    print("  inside that range, so B cannot be read as real on its own.")
    print("\n  The decisive control: run the IDENTICAL contrast -- the 3-mer TGA")
    print("  against the 3-mer TAG, same anagram logic, same strata, same")
    print("  bootstrap -- at positions in the 3'UTR where neither string is a stop")
    print("  codon and no mechanism exists. If those return ~0, the stop-codon")
    print("  result stands above the floor. If they return ~2pp, it does not.\n")
    for k in (0, 10, 25, 50, 100):
        if k == 0:
            col, lbl = "stop_codon", "the stop codon itself"
        else:
            col, lbl = f"t{k}", f"3'UTR +{k}..+{k+2}"
            d[col] = [sub(blob, off, i, e + k, e + k + 3)
                      for i, e in zip(d["_i"], d["orf_end"])]
        n_tga = int(d[col].eq("TGA").sum())
        n_tag = int(d[col].eq("TAG").sum())
        if min(n_tga, n_tag) < 200:
            print(f"    {lbl:<24} too few: TGA {n_tga:,} TAG {n_tag:,}")
            continue
        pt, lo, hi, p, _ = boot_diff(d, col, "TGA", "TAG", STRATA, n=1500)
        flag = "  <-- the claim" if k == 0 else ""
        print(f"    {lbl:<24} n TGA {n_tga:>6,} TAG {n_tag:>6,}   "
              f"TGA-TAG = {pt:+6.2f}pp  CI [{lo:+6.2f}, {hi:+6.2f}]  "
              f"p = {p:.3f}{flag}")
    print("\n  Read the +10/+25/+50/+100 rows as the null distribution of this")
    print("  exact statistic. The stop-codon row has to stand outside it.")

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
