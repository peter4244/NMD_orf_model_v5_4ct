#!/usr/bin/env python
"""
analysis1_upstream_orf_decay_capacity.py — can the upstream ORFs actually trigger decay?

THE GAP THIS FILLS. Section 4 reports that isoforms degraded WITHOUT a premature
stop in their main reading frame carry longer 5'UTRs and longer upstream ORFs than
controls. It never asks whether any of those upstream ORFs *terminates* somewhere
that would trigger decay — which is the entire mechanism. A stop codon causes
decay only when a junction lies more than 50 nt downstream of it, and Section 4
applies that rule rigorously to main ORFs and never once to upstream ones.

No model is involved. Every upstream ORF is already a candidate in the pool
carrying its own downstream-junction count, so this is a property of the
transcripts.

DEFINITIONS, stated because the published ones do not survive inspection.

  upstream        orf_start strictly less than the GENCODE-annotated start codon
                  of THIS isoform's own associated transcript, projected into this
                  isoform's exon coordinates. NOT "in the 5' half of the
                  transcript" -- that criterion mislabels 71.7% of what it selects,
                  because annotated starts are themselves 5'-proximal.
  decay-capable   the upstream ORF's own stop codon has at least one junction more
                  than 50 nt downstream, i.e. the same rule Section 4 uses for main
                  ORFs.
  overlapping     an upstream ORF whose stop lies at or past the annotated start.
                  It blocks reinitiation, and it is the ATF4 configuration.
  premature stop  computed on the GENCODE-annotated ORF, not on a TD2 call.

THE CONTRAST MUST BE AGAINST CONTROLS, NOT AGAINST PTC+ ISOFORMS. 70.8% of short
upstream ORFs carry downstream junctions simply because they terminate early, so
"has an upstream ORF" is close to "has a premature stop" under the 50 nt rule.
Comparing degraded-without-a-premature-stop against degraded-with-one would
measure that definitional overlap. The informative comparison is between isoforms
that BOTH have an intact main ORF and differ only in whether they are degraded.

AND IT MUST BE CONDITIONED ON 5'UTR LENGTH. A longer 5'UTR contains more start
codons by chance, so it contains more upstream ORFs and a longer one by chance.
Section 4 reports 5'UTR length and longest-upstream-ORF length as two findings
without conditioning either on the other; they may be one observation. Counts are
therefore reported as densities and within 5'UTR-length strata.
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
TABLES = Path.home() / "claude_projects" / "nmd_w69_tables_2026-07-30"
sys.path.insert(0, str(TRACK_A / "tools"))
from claim_emit import emit                                  # noqa: E402

GTF = NMD / "reference_files" / "gencode.v49.primary_assembly.annotation.gtf.gz"
SQANTI = NMD / "sqanti" / "nmd_lungcells" / "results" / "nmd_lungcells_classification.txt"
CACHE = Path("/private/tmp/claude-502/-Users-petecastaldi/"
             "6ea6b1ee-b03e-42c4-8a85-487850841c94/scratchpad/structures.tsv")
TX_RE = re.compile(r'transcript_id "([^"]+)"')


def gencode_atg():
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
    return {t: (lo[t] if strand[t] == "+" else hi[t]) for t in lo}


def to_transcript(g, starts, ends, strand):
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


def gene_boot(df, col, groups, n=4000, seed=20260801):
    """Gene-clustered bootstrap of a difference in means.

    Resampling whole genes is the inference; doing it by concatenating frames is
    unusably slow, so each gene is reduced to four numbers first (sum and count
    within each arm) and the bootstrap is a matrix product over gene indices.
    Identical estimator, three orders of magnitude faster.
    """
    d = df.dropna(subset=[col])
    a_m = (d.grp == groups[0]).to_numpy()
    b_m = (d.grp == groups[1]).to_numpy()
    v = d[col].to_numpy(float)
    gi, ug = pd.factorize(d.gene)
    G = len(ug)
    sa = np.bincount(gi, weights=v * a_m, minlength=G)
    na = np.bincount(gi, weights=a_m.astype(float), minlength=G)
    sb = np.bincount(gi, weights=v * b_m, minlength=G)
    nb = np.bincount(gi, weights=b_m.astype(float), minlength=G)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n, G))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (sa[idx].sum(1) / na[idx].sum(1)) - (sb[idx].sum(1) / nb[idx].sum(1))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    sys.stdout.reconfigure(line_buffering=True)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "orf_start", "orf_end", "orf_length",
                                "n_downstream_ejc", "kozak_score"])
    tx = pd.read_csv(TABLES / "tx_summary.tsv", sep="\t",
                     usecols=["isoform_id", "chr", "is_nmd"])
    ref = pd.read_csv(REPO / "results_4ct_dn" / "ref_cds_features.tsv", sep="\t",
                      usecols=["isoform_id", "gene_id"]).drop_duplicates("isoform_id")
    tx = tx.merge(ref, on="isoform_id", how="left")

    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category", "associated_transcript"])
    sq = sq[sq.structural_category.eq("full-splice_match")].copy()
    sq["enst"] = sq.associated_transcript.astype(str).str.split(".").str[0]
    atg = gencode_atg()
    _st = pd.read_csv(CACHE, sep="\t")
    st = {r.isoform_id: (r.strand,
                         [int(x) for x in str(r.starts).split(",")],
                         [int(x) for x in str(r.ends).split(",")])
          for r in _st.itertuples()}

    # ---- project the annotated start into each isoform ----------------------
    rows = []
    for r in sq.itertuples():
        g = atg.get(r.enst)
        e = st.get(r.isoform)
        if g is None or e is None:
            continue
        t = to_transcript(int(g), e[1], e[2], e[0])
        if t is not None:
            rows.append((r.isoform, t))
    anno = pd.DataFrame(rows, columns=["isoform_id", "cds_tx_start"])
    print(f"full splice matches with the annotated start projected: {len(anno):,}")

    d = pool.merge(anno, on="isoform_id")
    print(f"pool candidates in those transcripts: {len(d):,}")

    # ---- per-transcript upstream-ORF descriptors ----------------------------
    d["upstream"] = d.orf_start < d.cds_tx_start
    d["capable"] = d.upstream & (d.n_downstream_ejc > 0)
    d["overlapping"] = d.upstream & (d.orf_end >= d.cds_tx_start)
    d["ov_capable"] = d.overlapping & (d.n_downstream_ejc > 0)
    d["strong_up"] = d.upstream & (d.kozak_score >= 1)

    g = d.groupby("isoform_id")
    per = pd.DataFrame({
        "utr5_len": g.cds_tx_start.first() - 1,
        "n_upstream": g.upstream.sum(),
        "n_capable": g.capable.sum(),
        "n_overlapping": g.overlapping.sum(),
        "n_ov_capable": g.ov_capable.sum(),
        "n_strong_up": g.strong_up.sum(),
        "max_kozak_up": d[d.upstream].groupby("isoform_id").kozak_score.max(),
        "longest_up": d[d.upstream].groupby("isoform_id").orf_length.max(),
    }).reset_index()

    # premature stop status of the ANNOTATED ORF itself
    main = d[d.orf_start == d.cds_tx_start].groupby("isoform_id").n_downstream_ejc.max()
    per["main_ejc"] = per.isoform_id.map(main)
    per = per[per.main_ejc.notna()].copy()          # annotated start admitted
    per["main_ptc"] = per.main_ejc > 0
    per = per.merge(tx, on="isoform_id", how="inner")
    per = per[per.gene_id.notna()].rename(columns={"gene_id": "gene"})
    per["n_capable"] = per.n_capable.fillna(0)
    per["density_capable"] = 1000 * per.n_capable / per.utr5_len.clip(lower=1)

    per["grp"] = np.where(per.is_nmd.eq(1) & ~per.main_ptc, "NMD, no main-ORF stop",
                  np.where(per.is_nmd.eq(1) & per.main_ptc, "NMD, main-ORF stop",
                  np.where(~per.is_nmd.eq(1) & ~per.main_ptc, "control, no main-ORF stop",
                           "control, main-ORF stop")))
    print(f"\ntranscripts analysed: {len(per):,}   genes: {per.gene.nunique():,}")
    print(per.grp.value_counts().to_string())

    print(f"\n{'group':<28}{'n':>7}{'5UTR nt':>9}{'#upstr':>8}{'#capable':>10}"
          f"{'%with>=1':>10}{'per kb':>8}{'#overlap':>10}")
    for grp, s in per.groupby("grp"):
        print(f"{grp:<28}{len(s):>7,}{s.utr5_len.median():>9.0f}"
              f"{s.n_upstream.median():>8.0f}{s.n_capable.median():>10.0f}"
              f"{100*(s.n_capable>0).mean():>9.1f}%{s.density_capable.median():>8.2f}"
              f"{100*(s.n_overlapping>0).mean():>9.1f}%")

    # ---- the primary contrast, conditioned on 5'UTR length ------------------
    A, B = "NMD, no main-ORF stop", "control, no main-ORF stop"
    two = per[per.grp.isin([A, B])].copy()
    print(f"\n=== primary contrast: both have an intact main ORF ===")
    print(f"    {A}  vs  {B}")
    for col, lab in (("n_capable", "decay-capable upstream ORFs (count)"),
                     ("density_capable", "same, per kb of 5'UTR"),
                     ("n_overlapping", "overlapping upstream ORFs (count)"),
                     ("utr5_len", "5'UTR length (nt)"),
                     ("longest_up", "longest upstream ORF (nt)"),
                     ("max_kozak_up", "strongest upstream initiation score")):
        a, b = two.loc[two.grp == A, col], two.loc[two.grp == B, col]
        lo, hi = gene_boot(two.dropna(subset=[col]), col, (A, B))
        print(f"  {lab:<38} {a.median():>8.2f} vs {b.median():>8.2f}   "
              f"mean diff {a.mean()-b.mean():>+7.3f}   gene-clustered 95% CI "
              f"[{lo:+.3f}, {hi:+.3f}]")
        emit("5.90.8", lab, float(a.mean() - b.mean()), n=int(len(two)),
             population=f"full-splice-match transcripts with the GENCODE start "
                        f"admitted to the pool and no premature stop in the "
                        f"annotated ORF; difference is '{A}' minus '{B}'",
             sd_between=float((hi - lo) / 3.92))

    print(f"\n=== conditioned on 5'UTR length (quartiles) ===")
    two["q"] = pd.qcut(two.utr5_len, 4, labels=False, duplicates="drop")
    print(f"  {'5UTR quartile':<18}{'n NMD':>7}{'n ctrl':>8}"
          f"{'%NMD with capable':>20}{'%ctrl with capable':>21}")
    for q, s in two.groupby("q"):
        a, b = s[s.grp == A], s[s.grp == B]
        rng_ = f"{s.utr5_len.min():.0f}-{s.utr5_len.max():.0f}"
        print(f"  {rng_:<18}{len(a):>7,}{len(b):>8,}"
              f"{100*(a.n_capable>0).mean():>19.1f}%{100*(b.n_capable>0).mean():>20.1f}%")


if __name__ == "__main__":
    main()
