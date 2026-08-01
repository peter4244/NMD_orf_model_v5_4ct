#!/usr/bin/env python
"""
EXPERIMENT 4 -- does it matter whether a junction falls on a codon boundary?

REFRAMED, AND THE REFRAMING IS THE POINT.

As written in the plan this experiment asks whether a junction that lands
cleanly between codons behaves differently from one that splits a codon. For
the junction that decides NMD -- the first one after the stop -- there is no
mechanism for that to be true, because the 3'UTR is not translated and has no
reading frame. A junction inside the coding region is displaced by the
elongating ribosome regardless of its phase.

So this is not a hypothesis with a plausible positive. It is a NULL, and that
makes it the most useful thing available: every other experiment today rests on
the same machinery -- direct standardisation over matched strata, gene-clustered
bootstrap -- and nothing has measured what that machinery returns when there is
provably nothing there. Experiment 2 called a +2.00pp difference real and a
+1.24pp difference noise. This is what licenses that distinction, or doesn't.

Three arms:
  NULL 1   phase of the first junction after the stop, (j - stop) mod 3
  NULL 2   phase of junctions inside the CDS, (j - orf_start) mod 3
  POSITIVE the >50 nt rule, run through the identical code path, to show the
           machinery is not simply flat on everything

If a null arm returns an effect comparable to Experiment 2's, Experiment 2 is
not safe. If the nulls sit near zero and the positive is large, the readings
stand.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp4_junction_phase.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
RNG = np.random.default_rng(20260801)


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                  if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for iso, j in zip(df["isoform_id"], df["junctions"])}


def standardised(d, col, strata, label="is_nmd", min_cell=20):
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


def boot_diff(d, col, a, b, strata, gene_col="gene_id", n=2000, label="is_nmd",
              min_cell=10):
    """Identical code path to exp2.boot_diff -- that is the entire point."""
    d = d[d[col].isin([a, b])].dropna(subset=strata + [label, gene_col]).copy()
    if not len(d):
        return (np.nan,) * 4 + (0,)
    kcode, _ = pd.factorize(d[strata].astype(str).agg("|".join, axis=1))
    gcode, guniq = pd.factorize(d[gene_col])
    arm = d[col].eq(a).to_numpy().astype(np.int64)
    y = d[label].to_numpy().astype(np.float64)
    nk = int(kcode.max()) + 1
    order = np.argsort(gcode, kind="stable")
    gs = gcode[order]
    starts = np.searchsorted(gs, np.arange(len(guniq)), side="left")
    ends = np.searchsorted(gs, np.arange(len(guniq)), side="right")
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
        draws[t] = stat(np.concatenate([gene_rows[g] for g in RNG.integers(0, ng, ng)]))
    draws = draws[~np.isnan(draws)]
    if len(draws) < 20:
        return point, np.nan, np.nan, np.nan, len(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return point, lo, hi, max(p, 1.0 / len(draws)), len(draws)


def main():
    print("=" * 96)
    print("EXPERIMENT 4 -- junction phase, run as a calibration null")
    print("=" * 96)

    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id"])
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_start", "orf_end", "orf_length",
                               "is_ref_cds", "n_downstream_ejc"])

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.merge(tx, on="isoform_id").merge(ref.drop_duplicates("isoform_id"),
                                           on="isoform_id", how="left")
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d = d[d["utr3"] >= 10].copy()

    nxt, n_cds_junc, cds_phase = [], [], []
    for i, s, e in zip(d["isoform_id"], d["orf_start"], d["orf_end"]):
        j = junc.get(i, np.empty(0, dtype=np.int64))
        k = int(np.searchsorted(j, int(e), side="right"))
        nxt.append(int(j[k]) - int(e) if k < len(j) else np.nan)
        inside = j[(j >= int(s)) & (j < int(e))]
        n_cds_junc.append(len(inside))
        cds_phase.append(float(np.mean((inside - int(s)) % 3 == 0))
                         if len(inside) else np.nan)
    d["dist"] = nxt
    d["n_cds_junc"] = n_cds_junc
    d["frac_cds_junc_in_phase"] = cds_phase

    d["ptc"] = (d["dist"] >= 50).fillna(False).astype(int)
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop").astype(int)
    d["len_q"] = pd.qcut(d["orf_length"], 4, labels=False,
                         duplicates="drop").astype(int)
    d["nej_b"] = np.clip(d["n_downstream_ejc"], 0, 4).astype(int)
    print(f"\n  {len(d):,} isoforms with a reference-CDS slot and >= 10 nt of 3'UTR")
    print(f"  NMD+ {d['is_nmd'].mean()*100:.1f}%")

    # ------------------------------------------------------------- NULL 1
    print("\n" + "=" * 96)
    print("NULL 1 -- phase of the first junction after the stop, (j - stop) mod 3")
    print("=" * 96)
    print("\n  The 3'UTR is not translated. There is no reading frame there, so")
    print("  this quantity is defined but meaningless. Anything it returns is the")
    print("  machinery's own noise.")
    one = d[d["n_downstream_ejc"].eq(1) & d["dist"].notna()].copy()
    one["phase"] = (one["dist"] % 3).astype(int)
    STRATA = ["ptc", "utr3_q", "len_q"]
    r = standardised(one, "phase", STRATA)
    print(f"\n  population: exactly one junction past the stop, n = {len(one):,}")
    print(f"  strata: {STRATA}")
    print(f"\n  {'phase':<8} {'n':>7} {'crude':>8} {'standardised':>14}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*14}")
    for p in sorted(r):
        n, c, s = r[p]
        ss = f"{s:>13.1f}%" if not np.isnan(s) else f"{'--':>14}"
        print(f"  {p:<8} {n:>7,} {c:>7.1f}% {ss}")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pt, lo, hi, pv, _ = boot_diff(one, "phase", a, b, STRATA, n=1500)
        print(f"    phase {a} - phase {b} = {pt:+6.2f}pp  "
              f"95% CI [{lo:+6.2f}, {hi:+6.2f}]  p = {pv:.3f}")

    print("\n  Same null on the whole cohort, not just the count==1 stratum:")
    allj = d[d["dist"].notna()].copy()
    allj["phase"] = (allj["dist"] % 3).astype(int)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pt, lo, hi, pv, _ = boot_diff(allj, "phase", a, b,
                                      ["ptc", "utr3_q", "len_q", "nej_b"], n=1500)
        print(f"    phase {a} - phase {b} = {pt:+6.2f}pp  "
              f"95% CI [{lo:+6.2f}, {hi:+6.2f}]  p = {pv:.3f}")

    # ------------------------------------------------------------- NULL 2
    print("\n" + "=" * 96)
    print("NULL 2 -- phase of junctions INSIDE the coding sequence")
    print("=" * 96)
    print("\n  An exon junction complex inside the CDS is displaced by the")
    print("  elongating ribosome whatever its phase. Also a null.")
    cds = d[d["n_cds_junc"].ge(1)].copy()
    cds["all_in_phase"] = (cds["frac_cds_junc_in_phase"] == 1.0).astype(int)
    cds["none_in_phase"] = (cds["frac_cds_junc_in_phase"] == 0.0).astype(int)
    cds["arm"] = np.where(cds["all_in_phase"].eq(1), "all",
                          np.where(cds["none_in_phase"].eq(1), "none", "mixed"))
    r = standardised(cds, "arm", ["ptc", "utr3_q", "len_q", "nej_b"])
    print(f"\n  n = {len(cds):,} isoforms with >= 1 junction inside the CDS")
    print(f"\n  {'arm':<8} {'n':>7} {'crude':>8} {'standardised':>14}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*14}")
    for k in ["all", "mixed", "none"]:
        if k in r:
            n, c, s = r[k]
            ss = f"{s:>13.1f}%" if not np.isnan(s) else f"{'--':>14}"
            print(f"  {k:<8} {n:>7,} {c:>7.1f}% {ss}")
    pt, lo, hi, pv, _ = boot_diff(cds, "arm", "all", "none",
                                  ["ptc", "utr3_q", "len_q", "nej_b"], n=1500)
    print(f"    all-in-phase - none-in-phase = {pt:+.2f}pp  "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]  p = {pv:.3f}")

    # ---------------------------------------------------------- POSITIVE
    print("\n" + "=" * 96)
    print("POSITIVE CONTROL -- the >50 nt rule through the identical code path")
    print("=" * 96)
    one2 = d[d["n_downstream_ejc"].eq(1) & d["dist"].notna()].copy()
    one2["far"] = np.where(one2["dist"] > 50, "far", "near")
    pt, lo, hi, pv, _ = boot_diff(one2, "far", "far", "near",
                                  ["utr3_q", "len_q"], n=1500)
    print(f"\n  n = {len(one2):,}, count held at exactly 1")
    print(f"    >50 nt - <=50 nt = {pt:+.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"p = {pv:.3f}")

    print("\n" + "=" * 96)
    print("READING")
    print("=" * 96)
    print("  Compare the null arms above against the effects this project is")
    print("  calling real:")
    print("    Experiment 2, TGA - TAG          +2.00pp  CI [+0.75, +3.24]")
    print("    Experiment 2, +10 A - T (control) +1.24pp  CI [-0.01, +2.55]")
    print("  A null arm reaching the second number means +1.24pp is the floor.")
    print("  A null arm reaching the first means Experiment 2 is not safe.")

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
