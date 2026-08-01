#!/usr/bin/env python
"""
EXPERIMENT 7 -- would a graded junction-distance feature actually help us find
OTHER sequence elements, or is a binary ">50 nt junction present" enough?

PETE'S QUESTION, WHICH I HAD NOT ANSWERED
  Both windows converged on "the junction feature should carry graded distance
  rather than a second threshold". Pete: how would that help identify other
  sequence features that trigger NMD, as opposed to a binary indicator?

  Fair, and I had slid two things together:
    - the 18pp rise from 51-100 to 101-200 nt is a FINDING about biology
    - encoding it in the supplied feature is a DESIGN DECISION, and does not
      follow from the finding
  The only discovery-side argument for the design change is variance-soaking:
  supply the known geometry so the residual left for stop context, uORF strength
  and 3'UTR motifs is cleaner. But the threshold already does the bulk of the
  work -- 10% -> 54% at the boundary against 54% -> 71% across the graded part.

THE TEST THAT ANSWERS IT
  A supplied feature is inadequate for discovery only if the variance it leaves
  behind can be MISTAKEN for a sequence element. So: measure each candidate
  sequence feature's association with NMD twice --

      adjustment A   strata = BINARY (junction >50 nt: yes/no) x 3'UTR quartile
      adjustment B   strata = GRADED (6 distance bins)        x 3'UTR quartile

  If the estimates agree, the binary indicator already absorbs everything that
  could contaminate a sequence claim, and graded distance buys nothing for
  discovery -- whatever it is worth as biology.

  If a candidate SHRINKS under B, then binary adjustment was leaving residual
  distance signal that the candidate was picking up, and grading matters.

  This is the right test because it asks about the thing we care about --
  contamination of sequence claims -- rather than about model fit, which is not
  the goal.

CANDIDATES TESTED
  the surviving stop-codon association, the three 3'UTR motif classes from
  Exp 3, 3'UTR GC content, 5'UTR length, and uORF count. Deliberately a mix of
  things believed real, things shown to be composition, and things unresolved.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp7_does_graded_help.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast_lib import boot_diff

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(df["isoform_id"], df["junctions"])}


def load_seqs():
    z = np.load(os.path.join(HERE, "seq_store.npz"), allow_pickle=False)
    return z["blob"], z["offsets"], {s: i for i, s in enumerate(z["ids"])}


def sub(blob, off, i, a, b):
    lo, hi = int(off[i]), int(off[i + 1])
    a0, b0 = lo + a - 1, lo + b - 1
    if a0 < lo or b0 > hi or b0 <= a0:
        return ""
    return blob[a0:b0].tobytes().decode("ascii")


def dbin(x):
    if np.isnan(x):
        return "none"
    for lo, hi, nm in [(-1, 50, "d0_50"), (50, 100, "d51_100"),
                       (100, 200, "d101_200"), (200, 500, "d201_500"),
                       (500, 10**9, "d500p")]:
        if lo < x <= hi:
            return nm
    return "none"


def main():
    print("=" * 100)
    print("EXPERIMENT 7 -- does grading the junction feature help find OTHER")
    print("                sequence elements, or is the binary indicator enough?")
    print("=" * 100)

    blob, off, idx = load_seqs()
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id", "ref_utr5_length",
                               "ref_uorf_count_overlapping",
                               "ref_atg_count"]).drop_duplicates("isoform_id")
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t")

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.drop(columns=[c for c in ("tx_length",) if c in d.columns])
    d = d.merge(tx, on="isoform_id").merge(ref, on="isoform_id", how="left")
    d["_i"] = d["isoform_id"].map(idx)
    d = d[d["_i"].notna() & d["gene_id"].notna()].copy()
    d["_i"] = d["_i"].astype(int)
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d = d[d["utr3"] >= 200].copy()

    dist, nbeyond = [], []
    for i, e in zip(d["isoform_id"], d["orf_end"]):
        j = junc.get(i, np.empty(0, dtype=np.int64))
        k = int(np.searchsorted(j, int(e), side="right"))
        dist.append(int(j[k]) - int(e) if k < len(j) else np.nan)
        nbeyond.append(len(j) - int(np.searchsorted(j, int(e) + 50, side="right")))
    d["dist"], d["n_beyond50"] = dist, nbeyond

    # the two adjustments under comparison
    d["ADJ_binary"] = (d["n_beyond50"] > 0).astype(int)          # the canonical rule
    d["ADJ_graded"] = d["dist"].apply(dbin)                       # distance bins
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop").astype(int)

    win = [sub(blob, off, i, e + 1, e + 201) for i, e in zip(d["_i"], d["orf_end"])]
    d["win"] = win
    d = d[d["win"].str.len().eq(200)].copy()
    d["gc200"] = d["win"].str.count("[GC]") / 200
    d["gc_q"] = pd.qcut(d["gc200"], 4, labels=False, duplicates="drop").astype(int)

    print(f"\n  population: {len(d):,} isoforms, reference-CDS anchor, "
          f">= 200 nt of 3'UTR")
    print(f"  NMD+ {d['is_nmd'].mean()*100:.1f}%   "
          f"binary rule fires in {d['ADJ_binary'].mean()*100:.1f}%")
    print(f"\n  what the two adjustments look like:")
    print(f"    {'distance bin':<12} {'n':>7} {'NMD+':>8} {'binary says':>13}")
    for b in ["none", "d0_50", "d51_100", "d101_200", "d201_500", "d500p"]:
        g = d[d["ADJ_graded"].eq(b)]
        if not len(g):
            continue
        print(f"    {b:<12} {len(g):>7,} {g['is_nmd'].mean()*100:>7.1f}% "
              f"{g['ADJ_binary'].mean():>12.2f}")
    print("\n  Note the binary column: it is NOT a coarsening of the distance")
    print("  bins. A transcript whose NEAREST junction is within 50 nt can still")
    print("  have a LATER one beyond 50, so 'd0_50' is a mix. That is the honest")
    print("  version of the comparison and it is what the model would see.")

    # ------------------------------------------------------------- candidates
    cands = []
    d["c_stop"] = d["stop_codon"].where(d["stop_codon"].isin(["TGA", "TAG"]))
    cands.append(("stop codon TGA vs TAG", "c_stop", "TGA", "TAG"))

    d["c_pas"] = np.where(d["win"].str.contains("AATAAA", regex=False),
                          "present", "absent")
    cands.append(("AATAAA in first 200 nt", "c_pas", "present", "absent"))

    d["c_are"] = np.where(d["win"].str.contains("TTATTTATT", regex=False),
                          "present", "absent")
    cands.append(("TTATTTATT (ARE)", "c_are", "present", "absent"))

    d["c_py"] = np.where(d["win"].str.contains("[CT]{10}", regex=True),
                         "present", "absent")
    cands.append(("pyrimidine run >= 10", "c_py", "present", "absent"))

    d["c_gc"] = np.where(d["gc_q"].eq(3), "high",
                         np.where(d["gc_q"].eq(0), "low", None))
    cands.append(("3'UTR GC, top vs bottom quartile", "c_gc", "high", "low"))

    u5 = d["ref_utr5_length"]
    d["c_u5"] = np.where(u5 > u5.quantile(0.75), "long",
                         np.where(u5 < u5.quantile(0.25), "short", None))
    cands.append(("5'UTR length, long vs short", "c_u5", "long", "short"))

    na = d["ref_atg_count"]
    d["c_uorf"] = np.where(na >= 4, "many", np.where(na <= 1, "few", None))
    cands.append(("upstream AUG count, >=4 vs <=1", "c_uorf", "many", "few"))

    print("\n" + "=" * 100)
    print("THE COMPARISON -- each candidate, adjusted two ways")
    print("=" * 100)
    print("\n  A = binary junction rule x 3'UTR quartile x GC quartile")
    print("  B = graded distance bin  x 3'UTR quartile x GC quartile")
    print("  Both gene-clustered. If A and B agree, grading buys nothing here.\n")
    print(f"  {'candidate':<34} {'A: binary':>22} {'B: graded':>22} {'shift':>9}")
    print(f"  {'-'*34} {'-'*22} {'-'*22} {'-'*9}")

    rows = []
    for nm, col, a, b in cands:
        g = d[d[col].isin([a, b])].copy()
        if len(g) < 300:
            print(f"  {nm:<34} too few rows ({len(g)})")
            continue
        try:
            rA = boot_diff(g, col, a, b, ["ADJ_binary", "utr3_q", "gc_q"],
                           "is_nmd", "gene_id", n=600)
            rB = boot_diff(g, col, a, b, ["ADJ_graded", "utr3_q", "gc_q"],
                           "is_nmd", "gene_id", n=600)
        except Exception as exc:
            print(f"  {nm:<34} FAILED: {exc}")
            continue
        sa = f"{rA['diff_common']:+6.2f} [{rA['lo']:+5.2f},{rA['hi']:+5.2f}]"
        sb = f"{rB['diff_common']:+6.2f} [{rB['lo']:+5.2f},{rB['hi']:+5.2f}]"
        shift = rB["diff_common"] - rA["diff_common"]
        print(f"  {nm:<34} {sa:>22} {sb:>22} {shift:>+8.2f}p")
        rows.append(dict(name=nm, A=rA["diff_common"], B=rB["diff_common"],
                         orA=rA["mh_or"], orB=rB["mh_or"], shift=shift))

    res = pd.DataFrame(rows)
    print(f"\n  On the odds scale, where the pp scale can mislead:")
    print(f"  {'candidate':<34} {'A: binary OR':>14} {'B: graded OR':>14} "
          f"{'ratio':>8}")
    print(f"  {'-'*34} {'-'*14} {'-'*14} {'-'*8}")
    for _, r in res.iterrows():
        print(f"  {r['name']:<34} {r['orA']:>14.3f} {r['orB']:>14.3f} "
              f"{r['orB']/r['orA']:>8.3f}")

    print("\n" + "=" * 100)
    print("READING")
    print("=" * 100)
    mx = res["shift"].abs().max()
    med = res["shift"].abs().median()
    orshift = (res["orB"] / res["orA"]).sub(1).abs()
    print(f"\n  largest absolute shift in any candidate: {mx:.2f}pp")
    print(f"  median absolute shift:                    {med:.2f}pp")
    print(f"  largest change in odds ratio:             "
          f"{orshift.max()*100:.1f}%")
    print("""
  If those are small, the answer to Pete's question is: grading the junction
  feature does NOT help find other sequence elements. The binary rule already
  absorbs the part of junction geometry that could contaminate a sequence
  claim, and the 18pp gradient beyond it -- real as it is -- lives in a
  direction that no candidate sequence feature is aligned with.

  That would make the graded feature a BIOLOGY finding to report, not a design
  change to make. The two are separable and I had conflated them.
""")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
