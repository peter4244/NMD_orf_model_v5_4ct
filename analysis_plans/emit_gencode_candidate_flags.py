#!/usr/bin/env python
"""
emit_gencode_candidate_flags.py — the per-candidate GENCODE flags the ISM bank reads.

WHAT AND WHY. The bank ships `cand_upstream_of_ref`, which uses the
EXPRESSION-DERIVED reference start: the highest-DMSO-CPM non-NMD coding isoform
of the gene, projected. That is label-adjacent by construction, because the NMD
labels come from the same expression data. This file adds the alternative that is
not: the AUG indicated by the GENCODE CDS of **this isoform's own associated
transcript**, projected into this isoform's exon coordinates.

Both belong in the bank as SEPARATE fields. They agree 84.5% of the time on chr2,
so they are neither interchangeable nor independent, and merging them would hide
which one an analysis used.

THE CONSTRAINT THAT MUST TRAVEL WITH THIS FILE. The GENCODE flags are NULL where
no GENCODE CDS exists for the isoform's associated transcript — 36.5% of pooled
transcripts have no associated GENCODE transcript at all. Among those that do, the
annotated AUG is admitted to the pool for 95.0% of full splice matches but only
30.8% of incomplete ones. So a stratification keyed on these columns silently
drops the novel isoforms, which is where most NMD lives. Null is written as empty,
never as False, so "no annotation" cannot be read as "not upstream".

COLUMNS, one row per pool candidate:

    isoform_id, slot          the pool key
    has_gencode_cds           0/1 — does this isoform have an annotated CDS at all
    gencode_start_tx          the annotated AUG in this isoform's transcript
                              coordinates, empty where absent
    is_gencode_start          1 if this candidate IS that AUG
    upstream_of_gencode_start 1 if this candidate starts before it
    overlaps_gencode_start    1 if it starts before and ends at or past it
                              (blocks reinitiation; the ATF4 configuration)

The exon-block table is persisted out of session scratchpad on the way through —
it was sitting in /private/tmp, which is the same ephemeral-storage problem this
window flagged in the model checkpoints this morning.
"""

import gzip
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NMD = Path.home() / "claude_projects" / "nmd"
GTF = NMD / "reference_files" / "gencode.v49.primary_assembly.annotation.gtf.gz"
SQANTI = NMD / "sqanti" / "nmd_lungcells" / "results" / "nmd_lungcells_classification.txt"
SCRATCH = Path("/private/tmp/claude-502/-Users-petecastaldi/"
               "6ea6b1ee-b03e-42c4-8a85-487850841c94/scratchpad/structures.tsv")
DURABLE = REPO / "results_interp_all" / "isoform_exon_blocks.tsv"
OUT = REPO / "results_ism_v6" / "gencode_candidate_flags.tsv"
TX_RE = re.compile(r'transcript_id "([^"]+)"')


def gencode_atg():
    lo, hi, sd = {}, {}, {}
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 9)
            if f[2] != "CDS":
                continue
            m = TX_RE.search(f[8])
            if not m:
                continue
            t = m.group(1).split(".")[0]
            s, e = int(f[3]), int(f[4])
            if t in lo:
                lo[t] = min(lo[t], s); hi[t] = max(hi[t], e)
            else:
                lo[t], hi[t], sd[t] = s, e, f[6]
    return {t: (lo[t] if sd[t] == "+" else hi[t]) for t in lo}


def to_tx(g, starts, ends, strand):
    if strand == "+":
        o = 0
        for s, e in zip(starts, ends):
            if s <= g <= e:
                return o + (g - s) + 1
            o += e - s + 1
    else:
        o = 0
        for s, e in zip(reversed(starts), reversed(ends)):
            if s <= g <= e:
                return o + (e - g) + 1
            o += e - s + 1
    return None


def main():
    sys.stdout.reconfigure(line_buffering=True)

    DURABLE.parent.mkdir(parents=True, exist_ok=True)
    if not DURABLE.exists():
        if not SCRATCH.exists():
            raise SystemExit(f"exon blocks missing from both {DURABLE} and {SCRATCH}; "
                             "regenerate from data_mashr/structures.rds")
        shutil.copy2(SCRATCH, DURABLE)
        print(f"persisted exon blocks out of session scratchpad -> {DURABLE}")
    _s = pd.read_csv(DURABLE, sep="\t")
    st = {r.isoform_id: (r.strand,
                         [int(x) for x in str(r.starts).split(",")],
                         [int(x) for x in str(r.ends).split(",")])
          for r in _s.itertuples()}
    print(f"exon blocks for {len(st):,} isoforms")

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "orf_start", "orf_end"])
    print(f"pool: {len(pool):,} candidates over {pool.isoform_id.nunique():,} transcripts")

    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category", "associated_transcript"])
    sq = sq[sq.isoform.isin(set(pool.isoform_id))].copy()
    sq["enst"] = (sq.associated_transcript.astype(str).str.split(".").str[0]
                  .where(sq.associated_transcript.astype(str).str.startswith("ENST")))
    atg = gencode_atg()
    print(f"GENCODE transcripts with a CDS: {len(atg):,}")

    start_tx, n_noenst, n_nocds, n_notexonic = {}, 0, 0, 0
    for r in sq.itertuples():
        if not isinstance(r.enst, str):
            n_noenst += 1
            continue
        g = atg.get(r.enst)
        if g is None:
            n_nocds += 1
            continue
        e = st.get(r.isoform)
        if e is None:
            continue
        t = to_tx(int(g), e[1], e[2], e[0])
        if t is None:
            n_notexonic += 1
            continue
        start_tx[r.isoform] = t

    print(f"  annotated start projected for {len(start_tx):,} transcripts")
    print(f"  no GENCODE transcript      : {n_noenst:,}")
    print(f"  associated transcript non-coding: {n_nocds:,}")
    print(f"  annotated AUG not exonic here   : {n_notexonic:,}")

    pool["gencode_start_tx"] = pool.isoform_id.map(start_tx)
    has = pool.gencode_start_tx.notna()
    pool["has_gencode_cds"] = has.astype(int)
    pool["is_gencode_start"] = pd.NA
    pool["upstream_of_gencode_start"] = pd.NA
    pool["overlaps_gencode_start"] = pd.NA
    c = pool.gencode_start_tx
    pool.loc[has, "is_gencode_start"] = (pool.orf_start[has] == c[has]).astype(int)
    pool.loc[has, "upstream_of_gencode_start"] = (pool.orf_start[has] < c[has]).astype(int)
    pool.loc[has, "overlaps_gencode_start"] = (
        (pool.orf_start[has] < c[has]) & (pool.orf_end[has] >= c[has])).astype(int)

    out = pool[["isoform_id", "slot", "has_gencode_cds", "gencode_start_tx",
                "is_gencode_start", "upstream_of_gencode_start",
                "overlaps_gencode_start"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)

    print(f"\nwrote {OUT}")
    print(f"  rows {len(out):,}  (one per pool candidate)")
    print(f"  candidates with a GENCODE CDS on their transcript: "
          f"{int(has.sum()):,} ({100*has.mean():.1f}%)")
    print(f"  transcripts with one : {pool.loc[has,'isoform_id'].nunique():,} of "
          f"{pool.isoform_id.nunique():,}")
    print(f"  is_gencode_start          = 1 : {int(pool.is_gencode_start.sum()):,}")
    print(f"  upstream_of_gencode_start = 1 : {int(pool.upstream_of_gencode_start.sum()):,}")
    print(f"  overlaps_gencode_start    = 1 : {int(pool.overlaps_gencode_start.sum()):,}")
    print(f"  NULL (no annotation, written empty not 0) : "
          f"{int((~has).sum()):,} candidates")


if __name__ == "__main__":
    main()
