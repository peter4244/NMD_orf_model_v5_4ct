#!/usr/bin/env python
"""
EXPERIMENT 10 -- what actually IS the group of 2,334 transcripts?

Pete's question. I have described them procedurally -- "their gene has no
non-NMD coding isoform, so the category is forced" -- without ever saying what
kind of transcript they are. This answers that with measurements.

THE ONE THAT MATTERS MOST, AND I HAD NOT ASKED IT
  The model universe is NMD-positive UNION non-NMD, where non-NMD requires
  adj.P > 0.30 in ALL FOUR cell types. A transcript that is neither -- adj.P
  between the thresholds anywhere -- is in NEITHER class and is dropped.

  So a gene can lose its reference two very different ways:
    (a) every coding isoform it has really is an NMD substrate
    (b) its reference candidate fell in the EXCLUDED MIDDLE and was dropped
        from the universe, leaving only the NMD-positive ones behind
  Under (b) these are ordinary genes with an ordinary non-decayed isoform that
  the labelling threshold happened to discard, and "the gene has no non-NMD
  coding isoform" is a statement about our filter rather than about the gene.

  Distinguishing (a) from (b) needs the isoform count per gene BEFORE the
  universe restriction, which is in the count matrix.

ALSO MEASURED
  which decay route they carry (the 4-way partition from exp9), SQANTI
  structural category, exon count, baseline expression, and -- as a proxy for
  how robust the label is -- in how many of the four cell types the transcript
  actually rises under SMG1 inhibition. A positive called in one cell type is
  weaker evidence than one called in four, and the label is a UNION across the
  four, so single-cell-type calls are admitted.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp10_what_are_the_2334.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
DEPOSIT = os.path.expanduser("~/claude_projects/nmd_deposit_2026/source_data")
SQ = os.path.join(DEPOSIT, "sqanti", "nmd_lungcells_classification.txt")


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(df["isoform_id"], df["junctions"])}


def n_beyond(j, pos):
    return len(j) - int(np.searchsorted(j, pos, side="right"))


def side_by_side(d, col, groups, fmt="{:.1f}", pct=False):
    out = []
    for nm, m in groups:
        v = d.loc[m, col]
        out.append(fmt.format(v.mean() * 100 if pct else v.median()))
    return out


def main():
    print("=" * 96)
    print("EXPERIMENT 10 -- what are the 2,334?")
    print("=" * 96)

    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length", "n_junctions", "chr"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id", "category"]
                      ).drop_duplicates("isoform_id")
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_rank", "orf_start", "orf_end",
                               "orf_length", "is_ref_cds"])

    d = ref.merge(tx, on="isoform_id", how="inner")
    slot0 = sel[sel["orf_rank"].eq(sel["orf_rank"].min())].drop_duplicates("isoform_id")
    d = d.merge(slot0[["isoform_id", "orf_start", "orf_end", "orf_length"]],
                on="isoform_id", how="left")
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d["grp"] = np.where(d["category"].eq("no_ref_isoform"), "the 2,334",
                        np.where(d["is_nmd"].eq(1), "other positives", "negatives"))
    G = [("the 2,334", d["grp"].eq("the 2,334")),
         ("other positives", d["grp"].eq("other positives")),
         ("negatives", d["grp"].eq("negatives"))]
    print(f"\n  {'group':<20} {'n':>8}")
    for nm, m in G:
        print(f"  {nm:<20} {int(m.sum()):>8,}")

    # ------------------------------------------------------------------ (a) vs (b)
    print("\n" + "=" * 96)
    print("1. DID THE GENE HAVE NO NON-DECAYED ISOFORM, OR DID WE DROP IT?")
    print("=" * 96)
    print("\n  reading the count matrix for isoform counts BEFORE the universe "
          "restriction ...", flush=True)
    cnt = pd.read_csv(os.path.join(DEPOSIT, "nmd_lungcells_counts_4ct.csv"),
                      index_col=0)
    pheno = pd.read_csv(os.path.join(DEPOSIT, "pheno_4ct.csv"))
    sq = pd.read_csv(SQ, sep="\t",
                     usecols=["isoform", "associated_gene", "structural_category",
                              "exons", "coding"], low_memory=False)
    sq = sq.drop_duplicates("isoform")

    gene_of = dict(zip(sq["isoform"], sq["associated_gene"]))
    # isoforms present in the expression data at all, per gene
    expressed = cnt.index[cnt.sum(axis=1) > 0]
    ge = pd.Series([gene_of.get(i) for i in expressed], index=expressed,
                   name="gene").dropna()
    n_expr_per_gene = ge.value_counts()

    d["sq_gene"] = d["isoform_id"].map(gene_of)
    n_univ = d.groupby("gene_id")["isoform_id"].size()
    d["n_in_universe"] = d["gene_id"].map(n_univ)
    d["n_expressed"] = d["sq_gene"].map(n_expr_per_gene)
    d["n_dropped"] = d["n_expressed"] - d["n_in_universe"]

    tgt = d[d["grp"].eq("the 2,334")]
    print(f"\n  {'':<34} {'the 2,334':>12} {'other pos':>12} {'negatives':>12}")
    print(f"  {'-'*34} {'-'*12} {'-'*12} {'-'*12}")
    for lab, col in (("median isoforms in the universe", "n_in_universe"),
                     ("median isoforms expressed at all", "n_expressed"),
                     ("median dropped by the label filter", "n_dropped")):
        vals = side_by_side(d, col, G)
        print(f"  {lab:<34} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
    only1 = (tgt["n_in_universe"] == 1)
    print(f"\n  of the 2,334, {int(only1.sum()):,} ({only1.mean()*100:.0f}%) are the "
          f"ONLY isoform of their gene in the universe")
    hasdrop = tgt["n_dropped"] > 0
    print(f"  {int(hasdrop.sum()):,} ({hasdrop.mean()*100:.0f}%) belong to a gene that "
          f"HAS other expressed isoforms which the label filter dropped")
    print(f"  median dropped, among those: "
          f"{tgt.loc[hasdrop, 'n_dropped'].median():.0f}")
    print("""
  >> This is the (a)-versus-(b) answer. Where the gene has other expressed
     isoforms that simply failed the adj.P > 0.30 test in some cell type, the
     absent reference is a property of our threshold, not of the gene.""")

    # ------------------------------------------------------------------- routes
    print("\n" + "=" * 96)
    print("2. WHICH DECAY ROUTE DO THEY CARRY?")
    print("=" * 96)
    refslot = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id")
    has_ref_slot = set(refslot["isoform_id"])
    d["ejc"] = [int(n_beyond(junc.get(i, np.empty(0, dtype=np.int64)),
                             int(e) + 50) > 0) if pd.notna(e) else 0
                for i, e in zip(d["isoform_id"], d["orf_end"])]
    orf = pd.read_csv(os.path.join(TABLES, "orf_features.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_start", "orf_end"])
    st = dict(zip(d["isoform_id"], d["orf_start"]))
    orf = orf[orf["isoform_id"].isin(st)].copy()
    orf["cs"] = orf["isoform_id"].map(st)
    up = orf[orf["orf_start"] < orf["cs"]]
    hits = set()
    for i, e in zip(up["isoform_id"].to_numpy(), up["orf_end"].to_numpy()):
        if i not in hits and n_beyond(junc.get(i, np.empty(0, dtype=np.int64)),
                                      int(e) + 50) > 0:
            hits.add(i)
    d["uorf"] = d["isoform_id"].isin(hits).astype(int)
    d["route"] = np.where(d["ejc"].eq(1), "EJC junction",
                          np.where(d["uorf"].eq(1), "upstream ORF",
                                   np.where(d["utr3"] >= 1000, "long 3'UTR only",
                                            "no route")))
    print(f"\n  {'route':<20} {'the 2,334':>18} {'other positives':>18} "
          f"{'negatives':>18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    for r in ["EJC junction", "upstream ORF", "long 3'UTR only", "no route"]:
        cells = []
        for nm, m in G:
            g = d[m]
            cells.append(f"{g['route'].eq(r).mean()*100:>17.1f}%")
        print(f"  {r:<20} " + " ".join(cells))

    # -------------------------------------------------------- structure, expression
    print("\n" + "=" * 96)
    print("3. WHAT KIND OF TRANSCRIPT ARE THEY?")
    print("=" * 96)
    d = d.merge(sq.rename(columns={"isoform": "isoform_id"}),
                on="isoform_id", how="left")
    G = [(nm, d["grp"].eq(nm.replace("the ", "the "))) for nm, _ in
         [("the 2,334", None), ("other positives", None), ("negatives", None)]]
    print(f"\n  {'':<32} {'the 2,334':>14} {'other pos':>14} {'negatives':>14}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*14}")
    for lab, col in (("median transcript length", "tx_length"),
                     ("median exon count", "exons"),
                     ("median junction count", "n_junctions"),
                     ("median slot-0 ORF length", "orf_length"),
                     ("median 3'UTR", "utr3")):
        vals = side_by_side(d, col, G, fmt="{:,.0f}")
        print(f"  {lab:<32} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")
    print(f"\n  {'structural category':<32} {'the 2,334':>14} {'other pos':>14} "
          f"{'negatives':>14}")
    for cat in ["full-splice_match", "incomplete-splice_match", "novel_in_catalog",
                "novel_not_in_catalog", "fusion", "antisense", "genic",
                "intergenic"]:
        cells = []
        any_ = False
        for nm, m in G:
            v = d.loc[m, "structural_category"].eq(cat).mean() * 100
            any_ = any_ or v >= 2
            cells.append(f"{v:>13.1f}%")
        if any_:
            print(f"  {cat:<32} " + " ".join(cells))
    for nm, m in G:
        pass
    print(f"\n  {'SQANTI calls it coding':<32} " + " ".join(
        f"{d.loc[m, 'coding'].astype(str).eq('coding').mean()*100:>13.1f}%"
        for nm, m in G))

    # ----------------------------------------------------- robustness of the label
    print("\n" + "=" * 96)
    print("4. HOW ROBUST IS THE LABEL? -- response in how many cell types")
    print("=" * 96)
    print("""
  The NMD label is a UNION across four cell types, so a transcript called
  responsive in ONE cell type is admitted on the same footing as one called in
  four. mashr's own per-cell-type calls are not on this machine, so this is a
  direct proxy computed from the counts: in how many cell types does the
  transcript's mean CPM rise under SMG1 inhibition?""")
    lib = cnt.sum(axis=0)
    cpm = cnt.div(lib, axis=1) * 1e6
    ups = pd.DataFrame(index=cpm.index)
    for ct in sorted(pheno["cell_type"].unique()):
        s = pheno[pheno["cell_type"].eq(ct)]
        dm = [x for x in s.loc[s["treatment"].eq("DMSO"), "sample_name"]
              if x in cpm.columns]
        sm = [x for x in s.loc[s["treatment"].eq("Smg1i"), "sample_name"]
              if x in cpm.columns]
        ups[ct] = (cpm[sm].mean(axis=1) > cpm[dm].mean(axis=1) * 1.2).astype(int)
    ups["n_up"] = ups.sum(axis=1)
    d["n_ct_up"] = d["isoform_id"].map(ups["n_up"])
    dmso_cols = [x for x in pheno.loc[pheno["treatment"].eq("DMSO"), "sample_name"]
                 if x in cpm.columns]
    d["expr"] = d["isoform_id"].map(np.log2(cpm[dmso_cols].mean(axis=1) + 1))
    print(f"\n  {'cell types with >20% rise':<32} {'the 2,334':>14} "
          f"{'other pos':>14} {'negatives':>14}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*14}")
    for k in range(5):
        cells = [f"{d.loc[m, 'n_ct_up'].eq(k).mean()*100:>13.1f}%" for nm, m in G]
        print(f"  {k:<32} " + " ".join(cells))
    print(f"  {'median':<32} " + " ".join(
        f"{d.loc[m, 'n_ct_up'].median():>14.0f}" for nm, m in G))
    print(f"\n  {'median baseline log2(CPM+1), DMSO':<32} " + " ".join(
        f"{d.loc[m, 'expr'].median():>14.2f}" for nm, m in G))

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
