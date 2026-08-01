#!/usr/bin/env python
"""
probe_sqanti_cds_source.py — where does the SQANTI CDS actually come from?

THE QUESTION, Pete's, 2026-08-01. I established that the reference AUG traces to
SQANTI3's corrected CDS (`cds.rds` matches SQANTI `CDS_genomic_start/end` exactly
on 93,971 isoforms) and concluded it is TransDecoder2-predicted and therefore
never GENCODE. Pete's correction: SQANTI very likely **carries over the GENCODE
CDS for isoforms that match a known transcript**, and falls back to TD2 only for
novel isoforms. If so, my blanket claim is wrong, and the CDS for a full splice
match is exactly the reliable annotation we want.

This decides it from the data rather than from either of our expectations.

THE TEST. For every SQANTI isoform with an `associated_transcript` that is a real
GENCODE accession, compare SQANTI's CDS genomic bounds against the GENCODE CDS
bounds for that accession. Stratified by structural category, because the
hypothesis is that the source DIFFERS by category:

  full-splice_match       expect agreement if GENCODE is carried over
  incomplete-splice_match partial match to a known transcript; agreement unclear
  novel_in_catalog        no associated transcript; must be predicted
  novel_not_in_catalog    same

WHY IT MATTERS. TD2 is known to avoid ORFs terminating at premature stop codons,
which is the bias reference-AUG tracing exists to repair. If the CDS for annotated
isoforms is GENCODE's, that bias does not reach them, and a GENCODE-anchored
target for §5 is already partly in hand rather than needing to be built.

Version suffixes are stripped on both sides before joining: GENCODE carries
`ENST00000456328.2` and the two sources need not agree on the version.
"""

import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NMD = Path.home() / "claude_projects" / "nmd"
TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
sys.path.insert(0, str(TRACK_A / "tools"))
from claim_emit import emit                                  # noqa: E402

GTF = NMD / "reference_files" / "gencode.v49.primary_assembly.annotation.gtf.gz"
SQANTI = NMD / "sqanti" / "nmd_lungcells" / "results" / "nmd_lungcells_classification.txt"

TX_RE = re.compile(r'transcript_id "([^"]+)"')


def gencode_cds_bounds():
    """min CDS start and max CDS end per GENCODE transcript, plus strand."""
    lo, hi, strand = {}, {}, {}
    n = 0
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
            tx = m.group(1).split(".")[0]
            s, e = int(f[3]), int(f[4])
            if tx in lo:
                if s < lo[tx]:
                    lo[tx] = s
                if e > hi[tx]:
                    hi[tx] = e
            else:
                lo[tx], hi[tx], strand[tx] = s, e, f[6]
            n += 1
    print(f"GENCODE: {n:,} CDS lines over {len(lo):,} transcripts")
    return pd.DataFrame({"enst": list(lo), "g_lo": list(lo.values()),
                         "g_hi": list(hi.values()),
                         "g_strand": [strand[t] for t in lo]})


def main():
    sys.stdout.reconfigure(line_buffering=True)
    g = gencode_cds_bounds()

    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category",
                              "associated_transcript", "coding",
                              "CDS_genomic_start", "CDS_genomic_end"])
    print(f"SQANTI: {len(sq):,} isoforms")
    sq = sq[sq.CDS_genomic_start.notna() & sq.CDS_genomic_end.notna()].copy()
    print(f"  with a CDS call: {len(sq):,}")

    sq["enst"] = (sq.associated_transcript.astype(str)
                  .str.split(".").str[0]
                  .where(sq.associated_transcript.astype(str).str.startswith("ENST")))
    sq["s_lo"] = sq[["CDS_genomic_start", "CDS_genomic_end"]].min(axis=1)
    sq["s_hi"] = sq[["CDS_genomic_start", "CDS_genomic_end"]].max(axis=1)

    m = sq.merge(g, on="enst", how="left")
    has_ref = m.enst.notna() & m.g_lo.notna()
    m["cds_matches_gencode"] = has_ref & (m.s_lo == m.g_lo) & (m.s_hi == m.g_hi)
    m["start_matches"] = has_ref & np.where(
        m.g_strand.eq("+"), m.s_lo == m.g_lo, m.s_hi == m.g_hi)

    print(f"\n{'structural_category':<26}{'n':>8}{'has ENST':>10}"
          f"{'CDS==GENCODE':>14}{'start==GENCODE':>16}")
    for cat, sub in m.groupby("structural_category"):
        n = len(sub)
        nref = int(sub.enst.notna().sum())
        if nref:
            full = 100 * sub.cds_matches_gencode.sum() / nref
            st = 100 * sub.start_matches.sum() / nref
            print(f"{cat:<26}{n:>8,}{nref:>10,}{full:>13.1f}%{st:>15.1f}%")
        else:
            print(f"{cat:<26}{n:>8,}{nref:>10,}{'—':>14}{'—':>16}")

    fsm = m[m.structural_category.eq("full-splice_match") & m.enst.notna()]
    if len(fsm):
        emit("5.90.4", "SQANTI CDS identical to the GENCODE CDS, full splice matches",
             float(fsm.cds_matches_gencode.mean()), n=int(len(fsm)),
             population="SQANTI full-splice_match isoforms with a GENCODE "
                        "associated_transcript and a SQANTI CDS call; both CDS "
                        "bounds compared, version suffixes stripped")
        emit("5.90.4", "SQANTI CDS start identical to the GENCODE CDS start, "
             "full splice matches", float(fsm.start_matches.mean()),
             n=int(len(fsm)),
             population="same population; strand-aware start codon coordinate only")

    novel = m[m.structural_category.str.startswith("novel")]
    emit("5.90.4", "novel isoforms with a SQANTI CDS call and no GENCODE "
         "associated transcript", int(novel.enst.isna().sum()), n=int(len(novel)),
         population="SQANTI novel_in_catalog and novel_not_in_catalog isoforms "
                    "carrying a CDS call")
    print(f"\nnovel isoforms with a CDS call: {len(novel):,}, "
          f"of which {int(novel.enst.isna().sum()):,} have no GENCODE transcript "
          f"to inherit from")

    # ---- restricted to the model's own universe ----------------------------
    # The 645k SQANTI isoforms are not the model's population. What matters for
    # what the model was SHOWN is the pooled set.
    pool_iso = set(pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                               usecols=["isoform_id"]).isoform_id.unique())
    p = m[m.isoform.isin(pool_iso)].copy()
    print(f"\n=== restricted to the {len(pool_iso):,} transcripts in the model pool ===")
    print(f"  of these, {len(p):,} carry a SQANTI CDS call")
    print(f"\n{'structural_category':<26}{'n':>8}{'has ENST':>10}{'start==GENCODE':>16}"
          f"{'start DIFFERS':>15}")
    for cat, sub in p.groupby("structural_category"):
        nref = int(sub.enst.notna().sum())
        if nref:
            ok = int(sub.start_matches.sum())
            print(f"{cat:<26}{len(sub):>8,}{nref:>10,}{100*ok/nref:>15.1f}%"
                  f"{nref-ok:>15,}")
        else:
            print(f"{cat:<26}{len(sub):>8,}{nref:>10,}{'—':>16}{'—':>15}")

    annot = p[p.enst.notna()]
    n_diff = int((~annot.start_matches).sum())
    emit("5.90.5", "pooled transcripts with a GENCODE associated transcript",
         int(len(annot)), n=int(len(pool_iso)),
         population="transcripts in the v6 candidate pool carrying a SQANTI CDS "
                    "call and a GENCODE associated_transcript (FSM and ISM)")
    emit("5.90.5", "pooled annotated transcripts whose SQANTI CDS start differs "
         "from the GENCODE start", n_diff, n=int(len(annot)),
         population="pooled transcripts with a GENCODE associated_transcript; "
                    "strand-aware start codon coordinate compared")
    emit("5.90.5", "pooled transcripts with no GENCODE transcript to inherit a "
         "CDS from", int(len(p) - len(annot)), n=int(len(p)),
         population="pooled transcripts carrying a SQANTI CDS call; novel, "
                    "fusion and genic categories have no associated_transcript")
    print(f"\n  annotated pooled transcripts: {len(annot):,}")
    print(f"    SQANTI start differs from GENCODE: {n_diff:,} "
          f"({100*n_diff/max(len(annot),1):.1f}% of annotated, "
          f"{100*n_diff/len(pool_iso):.1f}% of the pool)")
    print(f"  pooled transcripts with no GENCODE transcript at all: "
          f"{len(p)-len(annot):,}")


if __name__ == "__main__":
    main()
