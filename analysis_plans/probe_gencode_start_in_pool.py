#!/usr/bin/env python
"""
probe_gencode_start_in_pool.py — was the GENCODE start codon ever shown to the model?

THE QUESTION, Pete's. For each pooled isoform, "the GENCODE start" means the AUG
indicated by the GENCODE CDS **of that specific isoform** — i.e. of the GENCODE
transcript SQANTI assigned it as `associated_transcript`, not the gene's canonical
or MANE CDS. Two things must hold for that AUG to be usable at all:

  1. the annotated start codon's genomic position must lie inside one of THIS
     isoform's exons — a 5'-truncated full splice match or an incomplete splice
     match may simply not contain it;
  2. having been mapped into transcript coordinates, it must be among the
     candidates the pool admitted, or the model could not select it however
     good its sequence.

An earlier pass compared SQANTI's CDS start against GENCODE's in GENOMIC
coordinates and found they differ for 12,007 of the 42,043 pooled transcripts.
That says the labels differ. It does NOT say the annotated start was withheld
from the model, because the ORF scan enumerates every AUG and may have admitted
it anyway under a different slot. This resolves which.

Projection is strand-aware and done against the isoform's own exon blocks, the
same operation `05t_ref_cds_features.R` performs with `is_codon_exonic` and
`genomic_to_transcript`.
"""

import gzip
import re
import subprocess
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
STRUCT = (NMD / "results/isoform_transitions/Version_6.0/isopair_wrapper"
          / "data_mashr" / "structures.rds")
TX_RE = re.compile(r'transcript_id "([^"]+)"')
CACHE = Path("/private/tmp/claude-502/-Users-petecastaldi/"
             "6ea6b1ee-b03e-42c4-8a85-487850841c94/scratchpad/structures.tsv")


def gencode_start_codons():
    """Strand-aware genomic position of the A of each GENCODE transcript's start
    codon, taken from the CDS block bounds."""
    lo, hi, strand = {}, {}, {}
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
                lo[tx] = min(lo[tx], s); hi[tx] = max(hi[tx], e)
            else:
                lo[tx], hi[tx], strand[tx] = s, e, f[6]
    atg = {t: (lo[t] if strand[t] == "+" else hi[t]) for t in lo}
    return pd.DataFrame({"enst": list(atg), "g_atg": list(atg.values()),
                         "g_strand": [strand[t] for t in atg]})


def load_structures():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["Rscript", "-e", f'''
            s <- readRDS("{STRUCT}")
            out <- data.frame(isoform_id=s$isoform_id, strand=s$strand,
              starts=sapply(s$exon_starts, paste, collapse=","),
              ends=sapply(s$exon_ends, paste, collapse=","))
            write.table(out, "{CACHE}", sep="\\t", row.names=FALSE, quote=FALSE)
        '''], check=True, capture_output=True)
    return pd.read_csv(CACHE, sep="\t")


def to_transcript(g, starts, ends, strand):
    """Genomic -> 1-based transcript position, or None if not exonic.

    Exon blocks arrive in ascending genomic order. On the minus strand the
    transcript runs the other way, so the offset accumulates from the LAST block.
    """
    if strand == "+":
        off = 0
        for s, e in zip(starts, ends):
            if s <= g <= e:
                return off + (g - s) + 1
            off += e - s + 1
    else:
        off = 0
        for s, e in zip(reversed(starts), reversed(ends)):
            if s <= g <= e:
                return off + (e - g) + 1
            off += e - s + 1
    return None


def main():
    sys.stdout.reconfigure(line_buffering=True)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "orf_start", "is_ref_cds",
                                "is_sqanti_cds"])
    starts_by_iso = pool.groupby("isoform_id").orf_start.apply(set).to_dict()
    pool_iso = set(starts_by_iso)
    print(f"pool: {len(pool_iso):,} transcripts, {len(pool):,} candidates")

    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category",
                              "associated_transcript"])
    sq = sq[sq.isoform.isin(pool_iso)].copy()
    sq["enst"] = (sq.associated_transcript.astype(str).str.split(".").str[0]
                  .where(sq.associated_transcript.astype(str).str.startswith("ENST")))
    g = gencode_start_codons()
    sq = sq.merge(g, on="enst", how="left")
    annot = sq[sq.g_atg.notna()].copy()
    print(f"pooled transcripts with a GENCODE CDS for their own associated "
          f"transcript: {len(annot):,}")

    st = load_structures().set_index("isoform_id")
    exonic = mapped = in_pool = is_flagged_cds = 0
    missing_struct = 0
    rows = []
    for r in annot.itertuples():
        if r.isoform not in st.index:
            missing_struct += 1
            continue
        row = st.loc[r.isoform]
        s = [int(x) for x in str(row.starts).split(",")]
        e = [int(x) for x in str(row.ends).split(",")]
        tpos = to_transcript(int(r.g_atg), s, e, row.strand)
        ok_ex = tpos is not None
        ok_pool = bool(ok_ex and tpos in starts_by_iso[r.isoform])
        exonic += ok_ex
        in_pool += ok_pool
        rows.append((r.isoform, r.structural_category, ok_ex, ok_pool))
    d = pd.DataFrame(rows, columns=["isoform", "cat", "exonic", "in_pool"])

    print(f"\n{'structural_category':<26}{'n':>8}{'AUG present':>14}"
          f"{'AND in the pool':>18}")
    for cat, sub in d.groupby("cat"):
        print(f"{cat:<26}{len(sub):>8,}{100*sub.exonic.mean():>13.1f}%"
              f"{100*sub.in_pool.mean():>17.1f}%")
    print(f"{'ALL ANNOTATED':<26}{len(d):>8,}{100*d.exonic.mean():>13.1f}%"
          f"{100*d.in_pool.mean():>17.1f}%")
    if missing_struct:
        print(f"  ({missing_struct:,} skipped: no exon structure on file)")

    POP = ("pooled transcripts whose SQANTI associated_transcript is a GENCODE "
           "accession carrying a CDS; the annotated start codon of THAT "
           "transcript, projected into this isoform's own exon coordinates")
    emit("5.90.6", "annotated start codon present in the isoform's exons",
         int(d.exonic.sum()), n=int(len(d)), population=POP)
    emit("5.90.6", "annotated start codon admitted as a candidate in the pool",
         int(d.in_pool.sum()), n=int(len(d)), population=POP)
    emit("5.90.6", "annotated start codon present but NOT admitted",
         int((d.exonic & ~d.in_pool).sum()), n=int(len(d)),
         population=POP + "; present in the transcript but failing the "
                          "initiation floor or the 5'-half position rule")
    print(f"\npresent but NOT admitted: {int((d.exonic & ~d.in_pool).sum()):,}")
    print(f"not present in the isoform at all: {int((~d.exonic).sum()):,}")


if __name__ == "__main__":
    main()
