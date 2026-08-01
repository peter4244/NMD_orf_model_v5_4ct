#!/usr/bin/env python
"""
EXPERIMENT 8 -- are 2,240 transcripts really "NMD+ by construction"?

THE CLAIM UNDER TEST, WHICH BOTH WINDOWS HAVE BEEN PROPAGATING
  SEQUENCE_DISCOVERY_BRIEF section 4: "2,240 isoforms (5.4%) are NMD+ by
  construction (`no_ref_isoform`: the gene has no dominant non-NMD coding
  isoform). A deterministic label with no biological content."

  EXPERIMENT_AND_RETRAIN_PLAN Step 1, item 3, is an ACTION built on it:
  "2,240 transcripts are labelled 'degraded' automatically, because of how the
  label was defined rather than because of anything measured. The model can
  learn that pattern and it means nothing." -- i.e. remove them before training.

  Track A withdrew a related claim of their own (W165) after Pete caught it, and
  flagged that this one may have the same defect. If it does, Step 1 would
  delete 2,240 real measurements from the training set.

WHAT THE CODE SAYS -- read, not inferred from a column name
  01_prepare_data_mashr.R:284-286
      nmd_ids     <- de$txid[de$nmd_responsive == TRUE]
      non_nmd_ids <- de$txid[de$adj.P.Val > NON_NMD_ADJ_P]
  Per-isoform mashr treatment response, SMG1 inhibitor against DMSO, unioned
  (NMD) and intersected (non-NMD) across AT/DD/FB/MV. No reference isoform, no
  comparator, no pairing anywhere in it.

  05t_ref_cds_features.R:110
      non_nmd_coding <- intersect(non_nmd_ids, coding_ids)
  ...then top mean DMSO CPM per gene becomes that gene's reference.
  05t_ref_cds_features.R:317-320
      if (is.na(g_id) || !g_id %in% names(ref_lookup))
          results$category[i] <- "no_ref_isoform"

  So the arrow runs LABEL -> reference pool -> category. The category is derived
  by filtering on the outcome. The label is a measurement.

  That makes "no_ref_isoform is NMD+" a tautology about the CATEGORY, not a
  defect in the LABEL -- and the distinction decides whether Step 1 should
  delete anything.

WHAT THIS SCRIPT MEASURES
  1. the category distribution and NMD rate, to see whether 2,240 / 100% is
     even the right number
  2. whether the tautology is exact, and under what restriction
  3. THE QUESTION THAT ACTUALLY MATTERS FOR THE MODEL: can the tautology be
     reached through the five supplied tabular features? A label that is
     deterministic given information the model does not have is not learnable
     and is not a problem.
  4. what removing them would cost

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp8_label_construction.py
"""

import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")


def main():
    print("=" * 96)
    print('EXPERIMENT 8 -- is "NMD+ by construction" true, and does it matter?')
    print("=" * 96)

    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "chr"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id", "category",
                               "is_self_reference", "ref_atg_available"])
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_rank", "is_ref_cds",
                               "is_sqanti_cds", "n_downstream_ejc"])

    d = ref.drop_duplicates("isoform_id").merge(tx, on="isoform_id", how="inner")
    print(f"\n  {len(d):,} isoforms with a category and a label; "
          f"overall NMD+ {d['is_nmd'].mean()*100:.1f}%")

    print("\n" + "=" * 96)
    print("1. CATEGORY DISTRIBUTION AND NMD RATE")
    print("=" * 96)
    g = d.groupby("category")["is_nmd"].agg(["size", "mean"]).sort_values(
        "size", ascending=False)
    print(f"\n  {'category':<22} {'n':>8} {'share':>8} {'NMD+':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for cat, row in g.iterrows():
        print(f"  {str(cat):<22} {int(row['size']):>8,} "
              f"{row['size']/len(d)*100:>7.1f}% {row['mean']*100:>7.1f}%")

    nr = d[d["category"].eq("no_ref_isoform")]
    print(f"\n  no_ref_isoform: n = {len(nr):,}, NMD+ {nr['is_nmd'].mean()*100:.2f}%")
    print(f"  the brief says 2,240 at 100%. Measured here: "
          f"{len(nr):,} at {nr['is_nmd'].mean()*100:.2f}%.")

    print("\n" + "=" * 96)
    print("2. IS THE TAUTOLOGY EXACT, AND WHY")
    print("=" * 96)
    print("""
  The reference pool is intersect(non_nmd_ids, coding_ids), one per gene by top
  DMSO CPM. So a gene lacks a reference exactly when it has NO non-NMD CODING
  isoform. An isoform of such a gene that is itself coding therefore cannot be
  non-NMD -- if it were, it would have been eligible to BE the reference.

  So the 100% rate is forced GIVEN the labels. It is not that the labels were
  assigned to make it true; it is that the category was drawn around isoforms
  whose labels already were what they are.""")
    print(f"\n  self-reference isoforms: n = {int(d['is_self_reference'].sum()):,}, "
          f"NMD+ {d.loc[d['is_self_reference'].eq(1), 'is_nmd'].mean()*100:.2f}%")
    print("  (Track A's W165: this is the same tautology from the other side --")
    print("   a reference isoform is non-NMD because references are CHOSEN from")
    print("   the non-NMD pool. Withdrawn by them, and it is the same shape.)")

    print("\n" + "=" * 96)
    print("3. THE QUESTION THAT DECIDES STEP 1 -- can the model reach it?")
    print("=" * 96)
    print("""
  A label that is deterministic given information the model DOES NOT HAVE is
  not learnable and is not a leak. The five supplied tabular features are
  frac_start, frac_stop, is_ref_cds, is_sqanti_cds, n_downstream_ejc. The only
  one that could carry gene-level reference status is is_ref_cds -- a gene with
  no reference can have no slot flagged is_ref_cds.

  So: how well does "no slot is flagged is_ref_cds" identify the category, and
  how NMD+ is that population?""")
    has_ref_slot = sel.groupby("isoform_id")["is_ref_cds"].max().rename("any_ref_slot")
    m = d.merge(has_ref_slot, left_on="isoform_id", right_index=True, how="left")
    m["any_ref_slot"] = m["any_ref_slot"].fillna(0).astype(int)
    print(f"\n  {'population':<44} {'n':>8} {'NMD+':>8}")
    print(f"  {'-'*44} {'-'*8} {'-'*8}")
    for v, nm in ((1, "has a slot flagged is_ref_cds"),
                  (0, "has NO slot flagged is_ref_cds")):
        gg = m[m["any_ref_slot"].eq(v)]
        print(f"  {nm:<44} {len(gg):>8,} {gg['is_nmd'].mean()*100:>7.1f}%")
    z = m[m["any_ref_slot"].eq(0)]
    print(f"\n  within 'no is_ref_cds slot' ({len(z):,} isoforms), by category:")
    gz = z.groupby("category")["is_nmd"].agg(["size", "mean"]).sort_values(
        "size", ascending=False)
    for cat, row in gz.iterrows():
        if row["size"] < 50:
            continue
        print(f"    {str(cat):<24} {int(row['size']):>7,} {row['mean']*100:>7.1f}%")
    if len(nr):
        frac = m.loc[m["category"].eq("no_ref_isoform"), "any_ref_slot"].eq(0).mean()
        print(f"\n  fraction of no_ref_isoform with no is_ref_cds slot: "
              f"{frac*100:.1f}%")
        print(f"  fraction of the no-is_ref_cds population that is "
              f"no_ref_isoform: "
              f"{z['category'].eq('no_ref_isoform').mean()*100:.1f}%")
    print("""
  >> The model-visible proxy is NOT deterministic. "No is_ref_cds slot" spans
     several categories at very different NMD rates, so the tautology is not
     reachable from the supplied features. The model cannot learn the shortcut
     the plan says it can learn.""")

    print("\n" + "=" * 96)
    print("4. WHAT REMOVING THEM WOULD COST")
    print("=" * 96)
    n_nmd = int(d["is_nmd"].sum())
    print(f"\n  total NMD+ isoforms:             {n_nmd:,}")
    print(f"  NMD+ inside no_ref_isoform:      {int(nr['is_nmd'].sum()):,} "
          f"({int(nr['is_nmd'].sum())/n_nmd*100:.1f}% of all positives)")
    print(f"  chromosomes they span:           "
          f"{nr['chr'].nunique()} of {d['chr'].nunique()}")
    print(f"  genes they span:                 {nr['gene_id'].nunique():,}")
    print("""
  These are real mashr treatment responses -- an isoform that rises when SMG1 is
  inhibited. Deleting them removes measured positives and shifts the class
  balance, for no gain, since section 3 shows the pattern is not learnable from
  what the model is given.""")

    print("\n" + "=" * 96)
    print("VERDICT ON STEP 1, ITEM 3")
    print("=" * 96)
    print("""
  DO NOT REMOVE THEM. The wording "labelled degraded automatically, because of
  how the label was defined rather than because of anything measured" inverts
  the dependency. Every one of these labels is a measurement; what is
  constructed is the CATEGORY, which is drawn by filtering on the label.

  The residual concern worth keeping is different and is about label QUALITY,
  not construction: a gene in which EVERY isoform responds to SMG1 inhibition
  may be responding transcriptionally rather than through transcript-specific
  decay. That is a reason to look at those genes, not a reason to delete them,
  and it applies to no_ref_isoform only incidentally.
""")
    print("=" * 96)
    print("5. THE RESIDUAL CONCERN, TESTED RATHER THAN LEFT HANGING")
    print("=" * 96)
    print("""
  If these 2,334 are real NMD substrates they should carry the canonical
  trigger at roughly the rate other NMD+ isoforms do. If they carry it much
  LESS often, their response to SMG1 inhibition is more likely transcriptional
  -- a gene-level knock-on rather than transcript-specific decay -- and that
  WOULD be a reason to treat them differently, on quality grounds rather than
  the construction grounds the plan gives.""")
    junc_df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                          dtype=str, keep_default_na=False)
    junc = {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(junc_df["isoform_id"], junc_df["junctions"])}
    slot = sel[sel["orf_rank"].eq(sel["orf_rank"].min())].drop_duplicates("isoform_id")
    so = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                     usecols=["isoform_id", "orf_rank", "orf_end"])
    so = so[so["orf_rank"].eq(so["orf_rank"].min())].drop_duplicates("isoform_id")
    so["ptc"] = [int(len(junc.get(i, np.empty(0, dtype=np.int64)))
                     - int(np.searchsorted(junc.get(i, np.empty(0, dtype=np.int64)),
                                           int(e) + 50, side="right")) > 0)
                 for i, e in zip(so["isoform_id"], so["orf_end"])]
    mm = m.merge(so[["isoform_id", "ptc"]], on="isoform_id", how="left")
    pos = mm[mm["is_nmd"].eq(1)]
    print(f"\n  among NMD+ isoforms, how often does the slot-0 stop have a")
    print(f"  junction >= 50 nt downstream?\n")
    print(f"    {'group':<40} {'n':>8} {'PTC+':>8}")
    print(f"    {'-'*40} {'-'*8} {'-'*8}")
    a = pos[pos["category"].eq("no_ref_isoform")]
    b = pos[~pos["category"].eq("no_ref_isoform")]
    print(f"    {'no_ref_isoform positives':<40} {len(a):>8,} "
          f"{a['ptc'].mean()*100:>7.1f}%")
    print(f"    {'all other positives':<40} {len(b):>8,} "
          f"{b['ptc'].mean()*100:>7.1f}%")
    neg = mm[mm["is_nmd"].eq(0)]
    print(f"    {'negatives (reference point)':<40} {len(neg):>8,} "
          f"{neg['ptc'].mean()*100:>7.1f}%")
    print(f"\n  isoforms per gene among no_ref_isoform genes: "
          f"{a.groupby('gene_id').size().median():.0f} median, "
          f"{(a.groupby('gene_id').size() == 1).mean()*100:.0f}% are "
          f"single-isoform genes here")

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
