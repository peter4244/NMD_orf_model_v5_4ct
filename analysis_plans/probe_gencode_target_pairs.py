#!/usr/bin/env python
"""
probe_gencode_target_pairs.py — start codon recovery against the GENCODE target.

WHAT CHANGES. Every start codon result so far has scored capture against
`is_ref_cds`, which is gene-level, TransDecoder2-derived, and defined by
expression in our own data — so it is not independent of the NMD labels. This
scores against the AUG indicated by the GENCODE CDS **of the isoform's own
associated transcript**, projected into that isoform's exon coordinates. That
target is external to our expression data, independent of the labels, and free of
the TD2 bias against ORFs terminating at premature stops.

Scoped to full splice matches, where the annotated AUG is present in 100% of
isoforms and admitted to the candidate pool in 95%. Incomplete splice matches are
excluded: the annotated AUG is absent from 64% of them because the transcript was
only partially observed, so a recovery rate there would mostly measure truncation.

WHAT DOES NOT CHANGE. Both geometry leaks are properties of the ENCODING, not of
the target, so every control carries over unchanged: the +/-50 nt caliper on
orf_start, the initiation floor on both arms, the requirement that both windows be
untruncated downstream, ties as one half, and the gene-clustered bootstrap as the
inference. The direction split is reported because a monotone positional
preference must favour one arm and disfavour the other.

ONE THING THIS SCRIPT MUST REPORT HONESTLY. If the GENCODE-annotated candidate is
usually the same candidate as `is_ref_cds`, the two targets are not independent
and agreement between them is not corroboration. The overlap is measured and
printed before any win rate.

chr2 is VALIDATION -- held out from gradient updates, used for early stopping.
"""

import gzip
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NMD = Path.home() / "claude_projects" / "nmd"
TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
sys.path.insert(0, str(TRACK_A / "tools"))
sys.path.insert(0, str(REPO))
from claim_emit import emit                                  # noqa: E402
from tensor_io import decode_windows                         # noqa: E402

GTF = NMD / "reference_files" / "gencode.v49.primary_assembly.annotation.gtf.gz"
SQANTI = NMD / "sqanti" / "nmd_lungcells" / "results" / "nmd_lungcells_classification.txt"
CACHE = Path("/private/tmp/claude-502/-Users-petecastaldi/"
             "6ea6b1ee-b03e-42c4-8a85-487850841c94/scratchpad/structures.tsv")
FLOOR = -1.2507921188400943
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


def ci_wilson(k, n):
    if n == 0:
        return (float("nan"),) * 2
    p, z = k / n, 1.959964
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main():
    sys.stdout.reconfigure(line_buffering=True)

    with h5py.File(REPO / "results_tensor_chr2" / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        gene = np.array([s.decode() for s in f["gene_id"][:]])
        cnt = f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        codes = f["codes"][:, 0]
        L, R = int(f.attrs["atg_left"]), int(f.attrs["atg_right"])

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))
    ref = pool.is_ref_cds.to_numpy() == 1
    kz = pool.kozak_score.to_numpy()
    tx = pool["tx"].to_numpy()
    cap = np.load(REPO / "results_interp_all" / "capture_chr2.npz")["cap"]

    # ---- the GENCODE target -------------------------------------------------
    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category", "associated_transcript"])
    sq = sq[sq.isoform.isin(set(iso)) &
            sq.structural_category.eq("full-splice_match")].copy()
    sq["enst"] = sq.associated_transcript.astype(str).str.split(".").str[0]
    atg = gencode_atg()
    st = pd.read_csv(CACHE, sep="\t").set_index("isoform_id")

    target = np.zeros(len(pool), dtype=bool)
    n_fsm = n_present = 0
    idx_by_iso = {k: v.to_numpy() for k, v in pool.groupby("isoform_id").groups.items()} \
        if False else None
    grp = {k: np.asarray(v) for k, v in pool.groupby("isoform_id").indices.items()}
    for r in sq.itertuples():
        g = atg.get(r.enst)
        if g is None or r.isoform not in st.index:
            continue
        n_fsm += 1
        row = st.loc[r.isoform]
        s = [int(x) for x in str(row.starts).split(",")]
        e = [int(x) for x in str(row.ends).split(",")]
        t = to_transcript(int(g), s, e, row.strand)
        if t is None:
            continue
        n_present += 1
        rows = grp.get(r.isoform)
        if rows is None:
            continue
        hit = rows[o_s[rows] == t]
        if len(hit) == 1:
            target[hit[0]] = True

    print(f"chr2: {len(iso):,} transcripts, {len(pool):,} candidates")
    print(f"  full splice matches with a GENCODE CDS : {n_fsm:,}")
    print(f"  annotated AUG present in the isoform   : {n_present:,}")
    print(f"  annotated AUG admitted as a candidate  : {int(target.sum()):,}")

    both = target & ref
    print(f"\n=== independence of the two targets ===")
    print(f"  annotated candidate that is ALSO the reference-AUG candidate: "
          f"{int(both.sum()):,} of {int(target.sum()):,} "
          f"({100*both.sum()/max(target.sum(),1):.1f}%)")
    emit("5.90.7", "GENCODE-annotated candidate coincides with the reference-AUG "
         "candidate", float(both.sum() / max(target.sum(), 1)),
         n=int(target.sum()),
         population="chr2 full-splice-match transcripts whose GENCODE-annotated "
                    "AUG is admitted to the candidate pool")

    # ---- geometry ----------------------------------------------------------
    right = np.empty(len(pool), dtype=np.int32)
    for i in range(0, len(pool), 4096):
        w = decode_windows(codes[i:i + 4096], o_s[i:i + 4096], L, o_s[i:i + 4096])
        right[i:i + 4096] = (w[:, 6:9].sum(1) > 0)[:, L:].sum(1)
    full = right >= R

    # ---- paired comparison, same controls as the reference-AUG version -----
    def pairs_for(pos, D=50):
        out = []
        for t in np.unique(tx[pos]):
            idx = np.where((tx == t) & (kz >= FLOOR) & full)[0]
            a, b = idx[pos[idx]], idx[~pos[idx]]
            for ai in a:
                for bi in b[np.abs(o_s[b] - o_s[ai]) <= D]:
                    out.append((ai, bi))
        return np.array(out) if out else np.empty((0, 2), int)

    for name, pos in (("GENCODE-annotated start", target),
                      ("reference-AUG start (comparison)", ref)):
        P = pairs_for(pos)
        if not len(P):
            print(f"\n{name}: no pairs"); continue
        won = (cap[P[:, 0]] > cap[P[:, 1]]).astype(float)
        won += 0.5 * (cap[P[:, 0]] == cap[P[:, 1]])
        g_ = gene[tx[P[:, 0]]]
        ug = np.unique(g_)
        rng = np.random.default_rng(20260801)
        bs = np.array([np.concatenate([won[g_ == x] for x in
                                       rng.choice(ug, len(ug), replace=True)]).mean()
                       for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        up = o_s[P[:, 0]] < o_s[P[:, 1]]
        print(f"\n=== {name} ===")
        print(f"  {len(P):,} pairs over {len(np.unique(P[:,0])):,} targets, "
              f"{len(ug):,} genes")
        print(f"  win rate {100*won.mean():.1f}%   "
              f"gene-clustered 95% CI [{100*lo:.1f}, {100*hi:.1f}]   "
              f"p(<=0.5) {(bs<=0.5).mean():.4f}")
        for lab, m in (("target upstream", up), ("target downstream", ~up)):
            if m.sum():
                k, n = won[m].sum(), int(m.sum())
                c = ci_wilson(k, n)
                print(f"    {lab:<20} {100*k/n:>5.1f}%  n {n:>5,}  "
                      f"95% CI [{c[0]:.1f}, {c[1]:.1f}]")
        if name.startswith("GENCODE"):
            emit("5.90.7", "capture prefers the GENCODE-annotated start over a "
                 "position-matched competitor", float(won.mean()), n=int(len(P)),
                 population="chr2 (validation) full-splice-match transcripts; "
                            "pairs within 50 nt of orf_start, both arms at or "
                            "above the initiation floor and with an untruncated "
                            "downstream start window; ties count one half",
                 sd_between=float(bs.std()))


if __name__ == "__main__":
    main()
