#!/usr/bin/env python
"""
build_ism_split.py — the gene-level discovery/confirmation partition, and the fixed
order in which transcripts enter the mutagenesis bank.

Implements section 9 step 1 and step 2 of analysis_plans/RETRAIN_PLAN_2026-08-01.md.

TWO THINGS, BOTH FIXED ONCE AND SHARED BY BOTH WINDOWS.

  arm    every gene is assigned to `discovery` or `confirmation` by a fair draw.
         Assigned over the WHOLE universe, before any subset is taken, so growing
         the bank does not move a single transcript from one arm to the other.

  rank   one fixed order over transcripts, drawn within each label stratum and
         interleaved so that EVERY PREFIX holds the pool's own prevalence. The
         bank of size n is rank < n. A bank of 2,000 is the bank of 1,000 with
         rows appended.

POPULATION. The 233 read-through composite loci — a `gene_id` of the form
ENSGa::ENSGb — are excluded, exactly as build_tensor.py excludes them from every
split. A composite is two genes transcribed as one unit, so an arm assignment on
it is ill-defined, and a transcript that is not in the tensor cannot be in a bank
built from the tensor. Excluding them here is what makes `rank < n` hold exactly n
transcripts instead of silently fewer.

Usage:
    python build_ism_split.py --out results_ism_v6
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
TABLES = Path(os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30"))
POOL = REPO / "results_pool_v6" / "orf_pool.tsv"

SEED = 20260801
EXPECTED_N = 41765          # the tensor's population, plan §5.3 step 5


def interleave_at_prevalence(pos, neg, prev):
    """One order whose every prefix holds `prev` positives.

    Position i takes a positive when the running target floor((i+1)*prev) exceeds
    floor(i*prev). Deterministic given the two shuffled lists, and the prefix count
    of positives never drifts from i*prev by more than one.
    """
    n = len(pos) + len(neg)
    out = np.empty(n, dtype=object)
    pi = ni = 0
    for i in range(n):
        want_pos = int((i + 1) * prev) > int(i * prev)
        if want_pos and pi < len(pos):
            out[i] = pos[pi]; pi += 1
        elif ni < len(neg):
            out[i] = neg[ni]; ni += 1
        else:
            out[i] = pos[pi]; pi += 1
    assert pi == len(pos) and ni == len(neg)
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_ism_v6")
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--pool", default=str(POOL))
    args = ap.parse_args()
    tables = Path(args.tables)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tx = pd.read_csv(tables / "tx_summary.tsv", sep="\t",
                     usecols=["isoform_id", "is_nmd", "chr", "tx_length"])
    ref = pd.read_csv(tables / "ref_cds_features.tsv", sep="\t",
                      usecols=["isoform_id", "gene_id"]).drop_duplicates("isoform_id")
    pool_iso = pd.read_csv(args.pool, sep="\t", usecols=["isoform_id"])["isoform_id"]

    df = tx.merge(ref, on="isoform_id", how="left")
    df = df[df["isoform_id"].isin(set(pool_iso))].copy()
    print(f"transcripts in TX and in the pool : {len(df):,}")
    missing = int(df["gene_id"].isna().sum())
    print(f"  without a gene_id               : {missing:,}")

    # -------------------------------------------------- composite read-through loci
    comp = df["gene_id"].astype(str).str.contains("::", regex=False)
    print(f"  on a composite locus (ENSGa::ENSGb)")
    print(f"    transcripts excluded          : {int(comp.sum()):,}")
    print(f"    composite loci                : {df.loc[comp, 'gene_id'].nunique():,}")
    df = df[~comp].copy()
    print(f"  population carried forward      : {len(df):,}")
    if len(df) != EXPECTED_N:
        print(f"  WARNING: expected {EXPECTED_N:,} (plan §5.3 step 5), got {len(df):,}")
    else:
        print(f"  matches the tensor's population in plan §5.3 step 5")

    # ------------------------------------------------------------------ step 1, arm
    # One draw per gene, over the whole universe. Sorted first so the assignment
    # depends on the gene set and the seed, and not on row order in any input file.
    genes = np.sort(df["gene_id"].astype(str).unique())
    rng = np.random.default_rng(SEED)
    arm_of_gene = dict(zip(genes,
                           np.where(rng.random(len(genes)) < 0.5,
                                    "discovery", "confirmation")))
    df["arm"] = df["gene_id"].astype(str).map(arm_of_gene)
    print(f"\nSTEP 1 — gene-level arm assignment, seed {SEED}")
    print(f"  genes                           : {len(genes):,}")
    for a in ("discovery", "confirmation"):
        g = sum(1 for v in arm_of_gene.values() if v == a)
        t = int((df["arm"] == a).sum())
        print(f"  {a:<14} {g:>7,} genes  {t:>7,} transcripts  "
              f"({100*t/len(df):.1f}%)  prevalence "
              f"{df.loc[df['arm'] == a, 'is_nmd'].mean():.4f}")

    # a gene must not span the two arms, and must not span chromosomes
    span_arm = df.groupby("gene_id")["arm"].nunique()
    span_chr = df.groupby("gene_id")["chr"].nunique()
    print(f"  genes spanning both arms        : {int((span_arm > 1).sum()):,}")
    print(f"  genes spanning >1 chromosome    : {int((span_chr > 1).sum()):,}")

    # ----------------------------------------------------------------- step 2, rank
    prev = float(df["is_nmd"].mean())
    pos = df.loc[df["is_nmd"] == 1, "isoform_id"].to_numpy()
    neg = df.loc[df["is_nmd"] == 0, "isoform_id"].to_numpy()
    pos = pos[rng.permutation(len(pos))]
    neg = neg[rng.permutation(len(neg))]
    order = interleave_at_prevalence(pos, neg, prev)
    rank_of = {iso: i for i, iso in enumerate(order)}
    df["rank"] = df["isoform_id"].map(rank_of)
    df = df.sort_values("rank", kind="stable").reset_index(drop=True)

    print(f"\nSTEP 2 — one fixed order, prevalence {prev:.4f}")
    print(f"  {'prefix n':>10} {'positives':>10} {'prevalence':>11} "
          f"{'discovery':>10} {'genes':>7}")
    for n in (120, 250, 500, 1000, 2000, 5000, len(df)):
        h = df.iloc[:n]
        print(f"  {n:>10,} {int(h['is_nmd'].sum()):>10,} "
              f"{h['is_nmd'].mean():>11.4f} "
              f"{int((h['arm'] == 'discovery').sum()):>10,} "
              f"{h['gene_id'].nunique():>7,}")

    # the balance the interpretation window will actually be powered by
    print(f"\n  discovery/confirmation balance at n = 1,000, within (label, arm):")
    h = df.iloc[:1000]
    for lab in (1, 0):
        row = h[h["is_nmd"] == lab]
        d = int((row["arm"] == "discovery").sum())
        print(f"    is_nmd={lab}  discovery {d:>4,}  confirmation "
              f"{len(row)-d:>4,}  genes {row['gene_id'].nunique():>4,}")

    # ------------------------------------------------------------------- provenance
    out = outdir / "discovery_confirmation_split.tsv"
    df[["isoform_id", "gene_id", "chr", "tx_length", "is_nmd", "arm", "rank"]] \
        .to_csv(out, sep="\t", index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    prov = dict(
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="build_ism_split.py",
        seed=SEED,
        sha256=sha,
        n_transcripts=int(len(df)),
        n_genes=int(len(genes)),
        prevalence=prev,
        composite_loci_excluded=int(comp.sum()),
        arm_counts={a: int((df["arm"] == a).sum()) for a in ("discovery", "confirmation")},
        gene_counts={a: int(sum(1 for v in arm_of_gene.values() if v == a))
                     for a in ("discovery", "confirmation")},
        paralog_screened_across_arms=False,
        note=("Arms are assigned per gene over the whole universe before any subset "
              "is taken, so a bank of n is a prefix of a bank of n' > n. Paralogy is "
              "NOT screened across the arm boundary: the two paralog lists this "
              "project holds are computed against the test and validation "
              "boundaries and carry no information about this one."),
        inputs=dict(tx_summary=str(tables / "tx_summary.tsv"),
                    ref_cds_features=str(tables / "ref_cds_features.tsv"),
                    pool=str(args.pool)))
    (outdir / "discovery_confirmation_split_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n")

    print(f"\nwrote {out}")
    print(f"  sha256 {sha}")
    print(f"  {len(df):,} transcripts, {len(genes):,} genes, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
