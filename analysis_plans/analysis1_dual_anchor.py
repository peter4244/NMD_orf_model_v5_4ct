#!/usr/bin/env python
"""
analysis1_dual_anchor.py — the upstream-ORF result under BOTH reading-frame definitions.

WHY BOTH. "Degraded without a premature stop in the main reading frame" is not a
measurement, it is a definition, and the definition moves the answer from 8% to
63% of degraded isoforms depending on how the reading frame is called. Two
definitions are live and they trade off against each other:

  GENCODE-anchored   the AUG of the isoform's OWN associated GENCODE transcript,
                     projected into its exon coordinates. External to our
                     expression data, no TransDecoder2 — but classifies only
                     ~14.5% of degraded isoforms, and systematically the
                     ANNOTATED ones, which the paper says is not where most NMD is.
  reference-AUG      the start codon of the gene's dominant non-NMD coding
                     isoform, projected. Covers ~63% of degraded isoforms — but is
                     TransDecoder2-derived and defined by expression in the same
                     data the NMD labels come from.

Clean and representative are in direct conflict and no option gives both. So
Pete's instruction is to report both side by side and let the disagreement be
visible. If the biological conclusion holds under both, the disagreement stops
mattering. If it holds under only one, we need that before writing rather than
after a reviewer asks.

In both cases the ANCHOR is the start; the STOP is computed on the observed
transcript by the pool's own enumeration, and "premature" is the same 50-nt rule
Section 4 applies to main ORFs.

NULL HANDLING. The GENCODE flags are read through their `has_gencode_cds` mask
column, never by interpreting empty cells. A NaN cast to an integer type becomes
0 with only a warning, which would silently turn "this isoform has no annotation"
into "this candidate is not upstream" — the model window hit exactly that bug
consuming this file this afternoon.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
TABLES = Path.home() / "claude_projects" / "nmd_w69_tables_2026-07-30"
sys.path.insert(0, str(TRACK_A / "tools"))
from claim_emit import emit                                  # noqa: E402

FLAGS = REPO / "results_ism_v6" / "gencode_candidate_flags.tsv"
SQANTI = (Path.home() / "claude_projects" / "nmd" / "sqanti" / "nmd_lungcells"
          / "results" / "nmd_lungcells_classification.txt")


def gene_boot(df, col, groups, n=4000, seed=20260801):
    d = df.dropna(subset=[col])
    a = (d.grp == groups[0]).to_numpy(float); b = (d.grp == groups[1]).to_numpy(float)
    v = d[col].to_numpy(float)
    gi, ug = pd.factorize(d.gene); G = len(ug)
    sa = np.bincount(gi, weights=v * a, minlength=G); na = np.bincount(gi, weights=a, minlength=G)
    sb = np.bincount(gi, weights=v * b, minlength=G); nb = np.bincount(gi, weights=b, minlength=G)
    idx = np.random.default_rng(seed).integers(0, G, size=(n, G))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (sa[idx].sum(1) / na[idx].sum(1)) - (sb[idx].sum(1) / nb[idx].sum(1))
    return np.nanpercentile(out, [2.5, 97.5])


def descriptors(pool, anchor_col):
    """Per-transcript upstream-ORF descriptors relative to `anchor_col`."""
    d = pool[pool[anchor_col].notna()].copy()
    a = d[anchor_col]
    d["upstream"] = d.orf_start < a
    d["capable"] = d.upstream & (d.n_downstream_ejc > 0)
    d["overlapping"] = d.upstream & (d.orf_end >= a)
    g = d.groupby("isoform_id")
    per = pd.DataFrame({
        "utr5_len": g[anchor_col].first() - 1,
        "n_upstream": g.upstream.sum(),
        "n_capable": g.capable.sum(),
        "n_overlapping": g.overlapping.sum(),
        "max_kozak_up": d[d.upstream].groupby("isoform_id").kozak_score.max(),
        "longest_up": d[d.upstream].groupby("isoform_id").orf_length.max(),
    }).reset_index()
    main = d[d.orf_start == a].groupby("isoform_id").n_downstream_ejc.max()
    per["main_ejc"] = per.isoform_id.map(main)
    per = per[per.main_ejc.notna()].copy()
    per["main_ptc"] = per.main_ejc > 0
    per["density_capable"] = 1000 * per.n_capable.fillna(0) / per.utr5_len.clip(lower=1)
    return per


def report(per, tx, label, tag):
    per = per.merge(tx, on="isoform_id", how="inner")
    per = per[per.gene.notna()]
    per["grp"] = np.where(per.is_nmd.eq(1) & ~per.main_ptc, "NMD, no main-ORF stop",
                  np.where(per.is_nmd.eq(1), "NMD, main-ORF stop",
                  np.where(~per.main_ptc, "control, no main-ORF stop",
                           "control, main-ORF stop")))
    nmd = per[per.is_nmd.eq(1)]
    nostop = int((nmd.grp == "NMD, no main-ORF stop").sum())
    print(f"\n{'='*74}\n{label}\n{'='*74}")
    print(f"  transcripts classified : {len(per):,}   genes {per.gene.nunique():,}")
    print(f"  NMD classified         : {len(nmd):,}  "
          f"({100*len(nmd)/9425:.1f}% of the 9,425 NMD isoforms in the pool)")
    print(f"  non-premature-stop route among them : {nostop:,} / {len(nmd):,} = "
          f"{100*nostop/max(len(nmd),1):.1f}%")
    print(f"\n  {'group':<28}{'n':>7}{'5UTR':>7}{'#capab':>8}{'>=1':>8}{'per kb':>8}{'overlap':>9}")
    for g_, s in per.groupby("grp"):
        print(f"  {g_:<28}{len(s):>7,}{s.utr5_len.median():>7.0f}"
              f"{s.n_capable.median():>8.0f}{100*(s.n_capable>0).mean():>7.1f}%"
              f"{s.density_capable.median():>8.2f}{100*(s.n_overlapping>0).mean():>8.1f}%")

    A, B = "NMD, no main-ORF stop", "control, no main-ORF stop"
    two = per[per.grp.isin([A, B])]
    print(f"\n  primary contrast, both arms with an intact main ORF: "
          f"n = {int((two.grp==A).sum()):,} vs {int((two.grp==B).sum()):,}, "
          f"{two.gene.nunique():,} genes")
    out = {}
    for col, lab in (("n_capable", "decay-capable upstream ORFs (count)"),
                     ("density_capable", "same, per kb of 5'UTR"),
                     ("n_overlapping", "overlapping upstream ORFs (count)"),
                     ("max_kozak_up", "strongest upstream initiation score")):
        a_, b_ = two.loc[two.grp == A, col], two.loc[two.grp == B, col]
        lo, hi = gene_boot(two, col, (A, B))
        diff = a_.mean() - b_.mean()
        out[lab] = (diff, lo, hi)
        star = "" if (lo < 0 < hi) else "  *"
        print(f"    {lab:<38}{diff:>+8.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]{star}")
        emit(tag, lab, float(diff), n=int(len(two)),
             population=f"{label}; '{A}' minus '{B}'; gene-clustered interval",
             sd_between=float((hi - lo) / 3.92))

    print(f"\n  conditioned on 5'UTR length (quartiles), % with >=1 decay-capable:")
    q = pd.qcut(two.utr5_len, 4, labels=False, duplicates="drop")
    for qq in sorted(set(q.dropna())):
        s = two[q == qq]
        a_, b_ = s[s.grp == A], s[s.grp == B]
        if len(a_) and len(b_):
            print(f"    {s.utr5_len.min():>5.0f}-{s.utr5_len.max():<7.0f} "
                  f"NMD {100*(a_.n_capable>0).mean():>5.1f}% (n={len(a_):>4,})   "
                  f"control {100*(b_.n_capable>0).mean():>5.1f}% (n={len(b_):>5,})")
    return per, out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "orf_start", "orf_end",
                                "orf_length", "n_downstream_ejc", "kozak_score",
                                "is_ref_cds"])
    fl = pd.read_csv(FLAGS, sep="\t",
                     usecols=["isoform_id", "slot", "has_gencode_cds", "gencode_start_tx"])
    # mask, never the nulls: NaN cast to int becomes 0 with only a warning
    fl.loc[fl.has_gencode_cds != 1, "gencode_start_tx"] = np.nan
    pool = pool.merge(fl[["isoform_id", "slot", "gencode_start_tx"]],
                      on=["isoform_id", "slot"], how="left")

    ref = pool.loc[pool.is_ref_cds == 1, ["isoform_id", "orf_start"]] \
              .rename(columns={"orf_start": "refaug_start_tx"})
    pool = pool.merge(ref, on="isoform_id", how="left")

    tx = pd.read_csv(TABLES / "tx_summary.tsv", sep="\t",
                     usecols=["isoform_id", "is_nmd"])
    gm = pd.read_csv(REPO / "results_4ct_dn" / "ref_cds_features.tsv", sep="\t",
                     usecols=["isoform_id", "gene_id"]).drop_duplicates("isoform_id")
    tx = tx.merge(gm, on="isoform_id", how="left").rename(columns={"gene_id": "gene"})

    # FULL SPLICE MATCHES ONLY, Pete 2026-08-01. An incomplete splice match is a
    # truncated observation of a known transcript, so its annotated start is
    # frequently absent for reasons that are technical rather than biological.
    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category"])
    fsm = set(sq.loc[sq.structural_category.eq("full-splice_match"), "isoform"])
    pool_fsm = pool[pool.isoform_id.isin(fsm)]
    print(f"full splice matches in the pool: {pool_fsm.isoform_id.nunique():,} "
          f"transcripts of {pool.isoform_id.nunique():,}")

    pg, og = report(descriptors(pool_fsm, "gencode_start_tx"), tx,
                    "PANEL A — GENCODE-anchored, full splice matches only",
                    "5.90.11")
    pr, orf = report(descriptors(pool_fsm, "refaug_start_tx"), tx,
                     "PANEL B — reference-AUG-traced, SAME full-splice-match population",
                     "5.90.12")
    _, oall = report(descriptors(pool, "refaug_start_tx"), tx,
                     "PANEL C (context) — reference-AUG-traced, full pool "
                     "including novel isoforms",
                     "5.90.14")

    print(f"\n{'='*74}\nDO THE TWO DEFINITIONS AGREE WHERE BOTH APPLY?\n{'='*74}")
    j = pg[["isoform_id", "main_ptc", "grp"]].merge(
        pr[["isoform_id", "main_ptc", "grp"]], on="isoform_id",
        suffixes=("_gencode", "_refaug"))
    print(f"  transcripts classified by both : {len(j):,}")
    ag = (j.main_ptc_gencode == j.main_ptc_refaug).mean()
    print(f"  agree on premature-stop status : {100*ag:.1f}%")
    print(pd.crosstab(j.main_ptc_gencode, j.main_ptc_refaug,
                      rownames=["GENCODE stop"], colnames=["ref-AUG stop"]).to_string())
    emit("5.90.13", "agreement between the two reading-frame definitions on "
         "premature-stop status", float(ag), n=int(len(j)),
         population="pooled transcripts classified by BOTH the GENCODE-anchored "
                    "and reference-AUG-traced definitions")

    print(f"\n{'='*74}\nSIDE BY SIDE — the primary contrast under each\n{'='*74}")
    print(f"  matched population (FSM) — so this isolates the reading-frame call")
    print(f"  {'quantity':<36}{'A GENCODE':>21}{'B ref-AUG':>21}{'C ref-AUG all':>21}")
    for k in og:
        a, b, c = og[k], orf[k], oall[k]
        print(f"  {k:<36}{a[0]:>+8.3f} [{a[1]:+.2f},{a[2]:+.2f}]"
              f"{b[0]:>+9.3f} [{b[1]:+.2f},{b[2]:+.2f}]"
              f"{c[0]:>+9.3f} [{c[1]:+.2f},{c[2]:+.2f}]")


if __name__ == "__main__":
    main()
