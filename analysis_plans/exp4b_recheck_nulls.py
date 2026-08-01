#!/usr/bin/env python
"""
EXPERIMENT 4b -- the noise floor was measured with the same broken estimator.

exp2b established that std_diff() -- which selects contributing strata
SEPARATELY FOR EACH ARM -- is biased upward at small n. Subsampling the
stop-codon contrast, where the truth is +1.80pp, down to n = 509/246 returns a
median of +14.04pp. The bias is not noise: the two arms end up averaged over
different sets of strata.

Experiment 4's "noise floor" of 2-5pp was measured with that estimator, and I
then used that floor to argue Experiment 2's +2.00pp was unreadable. If the
floor is itself an artifact, that argument collapses and so does the headline
conclusion I gave Pete -- "this design resolves +37pp and does not resolve
+2pp".

So every arm of Experiment 4 is re-run here under BOTH estimators:
  separate-stratum   the one used before; each arm keeps its own strata
  common-stratum     a stratum contributes only if BOTH arms clear min_cell,
                     so the two arms are averaged over identical strata

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp4b_recheck_nulls.py
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


def _sep(kcode, arm, y, nk, min_cell=10):
    n_k = np.bincount(kcode, minlength=nk).astype(np.float64)
    if n_k.sum() == 0:
        return np.nan
    w = n_k / n_k.sum()
    vals = []
    for side in (1, 0):
        m = arm == side
        cnt = np.bincount(kcode[m], minlength=nk).astype(np.float64)
        s = np.bincount(kcode[m], weights=y[m], minlength=nk)
        good = cnt >= min_cell
        den = w[good].sum()
        rate = np.divide(s, cnt, out=np.zeros(nk), where=cnt > 0)
        vals.append((w[good] * rate[good]).sum() / den if den > 0 else np.nan)
    return (vals[0] - vals[1]) * 100


def _common(kcode, arm, y, nk, min_cell=10):
    c1 = np.bincount(kcode[arm == 1], minlength=nk).astype(np.float64)
    c0 = np.bincount(kcode[arm == 0], minlength=nk).astype(np.float64)
    good = (c1 >= min_cell) & (c0 >= min_cell)
    if not good.any():
        return np.nan
    n_k = np.bincount(kcode, minlength=nk).astype(np.float64)
    w = n_k * good
    w = w / w.sum()
    s1 = np.bincount(kcode[arm == 1], weights=y[arm == 1], minlength=nk)
    s0 = np.bincount(kcode[arm == 0], weights=y[arm == 0], minlength=nk)
    return ((w[good] * (s1[good] / c1[good])).sum()
            - (w[good] * (s0[good] / c0[good])).sum()) * 100


def boot(d, col, a, b, strata, est, gene_col="gene_id", n=1500, label="is_nmd"):
    d = d[d[col].isin([a, b])].dropna(subset=strata + [label, gene_col]).copy()
    if not len(d):
        return np.nan, np.nan, np.nan, np.nan
    kcode, _ = pd.factorize(d[strata].astype(str).agg("|".join, axis=1))
    gcode, guniq = pd.factorize(d[gene_col])
    arm = d[col].eq(a).to_numpy().astype(np.int64)
    y = d[label].to_numpy().astype(np.float64)
    nk = int(kcode.max()) + 1
    order = np.argsort(gcode, kind="stable")
    gs = gcode[order]
    st = np.searchsorted(gs, np.arange(len(guniq)), side="left")
    en = np.searchsorted(gs, np.arange(len(guniq)), side="right")
    rows_by_gene = [order[s:e] for s, e in zip(st, en)]
    point = est(kcode, arm, y, nk)
    draws = np.empty(n)
    ng = len(guniq)
    for t in range(n):
        r = np.concatenate([rows_by_gene[g] for g in RNG.integers(0, ng, ng)])
        draws[t] = est(kcode[r], arm[r], y[r], nk)
    draws = draws[~np.isnan(draws)]
    if len(draws) < 20:
        return point, np.nan, np.nan, np.nan
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return point, lo, hi, max(p, 1.0 / len(draws))


def line(nm, d, col, a, b, strata):
    ps, los, his, pvs = boot(d, col, a, b, strata, _sep)
    pc, loc, hic, pvc = boot(d, col, a, b, strata, _common)
    f = lambda v: f"{v:+6.2f}" if not np.isnan(v) else "    --"
    g = lambda v: f"{v:.3f}" if not np.isnan(v) else "  --"
    print(f"  {nm:<34} {f(ps)}pp [{f(los)},{f(his)}] p={g(pvs)}   "
          f"|  {f(pc)}pp [{f(loc)},{f(hic)}] p={g(pvc)}")


def main():
    print("=" * 108)
    print("EXPERIMENT 4b -- re-running the noise floor under a corrected estimator")
    print("=" * 108)

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
    nxt, ncj, ph = [], [], []
    for i, s, e in zip(d["isoform_id"], d["orf_start"], d["orf_end"]):
        j = junc.get(i, np.empty(0, dtype=np.int64))
        k = int(np.searchsorted(j, int(e), side="right"))
        nxt.append(int(j[k]) - int(e) if k < len(j) else np.nan)
        ins = j[(j >= int(s)) & (j < int(e))]
        ncj.append(len(ins))
        ph.append(float(np.mean((ins - int(s)) % 3 == 0)) if len(ins) else np.nan)
    d["dist"], d["n_cds_junc"], d["frac_ip"] = nxt, ncj, ph
    d["ptc"] = (d["dist"] >= 50).fillna(False).astype(int)
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop").astype(int)
    d["len_q"] = pd.qcut(d["orf_length"], 4, labels=False,
                         duplicates="drop").astype(int)
    d["nej_b"] = np.clip(d["n_downstream_ejc"], 0, 4).astype(int)
    print(f"\n  {len(d):,} isoforms, NMD+ {d['is_nmd'].mean()*100:.1f}%")
    print(f"\n  {'arm':<34} {'SEPARATE-stratum (what I used)':<44} "
          f"|  {'COMMON-stratum (corrected)'}")
    print(f"  {'-'*34} {'-'*44} |  {'-'*44}")

    one = d[d["n_downstream_ejc"].eq(1) & d["dist"].notna()].copy()
    one["phase"] = (one["dist"] % 3).astype(int)
    S1 = ["ptc", "utr3_q", "len_q"]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        line(f"NULL 1 count==1, phase {a} - {b}", one, "phase", a, b, S1)

    allj = d[d["dist"].notna()].copy()
    allj["phase"] = (allj["dist"] % 3).astype(int)
    S2 = ["ptc", "utr3_q", "len_q", "nej_b"]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        line(f"NULL 1 full cohort, phase {a} - {b}", allj, "phase", a, b, S2)

    cds = d[d["n_cds_junc"].ge(1)].copy()
    cds["arm"] = np.where(cds["frac_ip"].eq(1.0), "all",
                          np.where(cds["frac_ip"].eq(0.0), "none", "mixed"))
    line("NULL 2 (withdrawn) all - none", cds, "arm", "all", "none", S2)
    cds["ncj_b"] = np.clip(cds["n_cds_junc"], 1, 6).astype(int)
    line("NULL 2 + CDS-junction count", cds, "arm", "all", "none",
         S2 + ["ncj_b"])

    one2 = one.copy()
    one2["far"] = np.where(one2["dist"] > 50, "far", "near")
    line("POSITIVE  >50nt - <=50nt", one2, "far", "far", "near",
         ["utr3_q", "len_q"])

    print("\n" + "=" * 108)
    print("READING")
    print("=" * 108)
    print("  Under the corrected estimator, if the null arms collapse toward zero")
    print("  while the positive control holds, then the 'floor of 2-5pp' was a")
    print("  property of my estimator, not of the design -- and Experiment 2's")
    print("  +2.00pp never needed to clear it.")

    print("\n" + "=" * 108)
    print("DONE")
    print("=" * 108)


if __name__ == "__main__":
    main()
