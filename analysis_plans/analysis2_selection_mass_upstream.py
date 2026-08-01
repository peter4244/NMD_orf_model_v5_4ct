#!/usr/bin/env python
"""
analysis2_selection_mass_upstream.py — where does the model send the ribosome?

Analysis 1 established that isoforms degraded without a premature stop in their
main reading frame carry more upstream ORFs that are *capable* of triggering
decay: denser per kb of 5'UTR, and 2.4x more likely to overlap the main start.
It also established the limit of that: 42.2% of undegraded controls carry the
same elements. Presence is enriched and nowhere near sufficient.

This asks the next thing. Not whether a decay-capable upstream ORF exists, but
whether the model ROUTES TRANSLATION TO IT. Under the scanning formulation
P(select k) is a distribution over initiation sites summing to at most one, so
"how much of the ribosome population is captured upstream of the annotated start"
is a direct readout requiring no attribution method.

QUANTITIES, per transcript:

  upstream_mass       sum of P(select k) over candidates starting before the
                      GENCODE-annotated start of this isoform's own associated
                      transcript, projected into its exon coordinates
  capable_mass        the same, restricted to upstream candidates whose own stop
                      codon has a junction more than 50 nt downstream
  overlap_mass        the same, restricted to upstream candidates whose stop lies
                      at or past the annotated start (the ATF4 configuration)
  annotated_mass      P(select) of the candidate at the annotated start itself

PILOT SCOPE, STATED UP FRONT. Only results_tensor_chr2 exists locally, and chr2
is the VALIDATION chromosome -- held out from gradient updates but used for early
stopping, so it is model-selection data. Worse for this analysis, chr2 carries
roughly 25 transcripts in the group of interest against 381 genome-wide. The
group contrast here is a pilot and is reported with its n; the distributional
readout over all annotated transcripts is the part worth reading.
"""

import gzip
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
NMD = Path.home() / "claude_projects" / "nmd"
TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
TABLES = Path.home() / "claude_projects" / "nmd_w69_tables_2026-07-30"
sys.path.insert(0, str(TRACK_A / "tools"))
sys.path.insert(0, str(REPO))
from claim_emit import emit                                  # noqa: E402
from model_v6 import ScanningNMDModel                        # noqa: E402
from tensor_io import decode_windows                         # noqa: E402

GTF = NMD / "reference_files" / "gencode.v49.primary_assembly.annotation.gtf.gz"
SQANTI = NMD / "sqanti" / "nmd_lungcells" / "results" / "nmd_lungcells_classification.txt"
CACHE = Path("/private/tmp/claude-502/-Users-petecastaldi/"
             "6ea6b1ee-b03e-42c4-8a85-487850841c94/scratchpad/structures.tsv")
CKPT = REPO / "results_interp_all" / "ckpt_interp_c32_b8"
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
    d = df.dropna(subset=[col])
    a = (d.grp == groups[0]).to_numpy(float)
    b = (d.grp == groups[1]).to_numpy(float)
    v = d[col].to_numpy(float)
    gi, ug = pd.factorize(d.gene)
    G = len(ug)
    sa = np.bincount(gi, weights=v * a, minlength=G)
    na = np.bincount(gi, weights=a, minlength=G)
    sb = np.bincount(gi, weights=v * b, minlength=G)
    nb = np.bincount(gi, weights=b, minlength=G)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n, G))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (sa[idx].sum(1) / na[idx].sum(1)) - (sb[idx].sum(1) / nb[idx].sum(1))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    sys.stdout.reconfigure(line_buffering=True)

    with h5py.File(REPO / "results_tensor_chr2" / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        gene = np.array([s.decode() for s in f["gene_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        o_e = f["orf_end"][:].astype(np.int64)
        struct = f["structural"][:]
        codes = f["codes"][:]
        L, SL = int(f.attrs["atg_left"]), int(f.attrs["stop_left"])

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "n_downstream_ejc"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))
    ejc = pool.n_downstream_ejc.to_numpy()

    # ---- selection mass, five seeds, full forward pass ----------------------
    ckpts = sorted(CKPT.glob("b8_s*.pt"))
    sel = np.zeros(len(pool), dtype=np.float64)
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1, permute_bins=False)
        m.load_state_dict(ck["model"]); m.eval()
        out = np.empty(len(pool), dtype=np.float64)
        with torch.no_grad():
            for i in range(len(iso)):
                sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
                s0, e0 = o_s[sl], o_e[sl]
                K = len(s0)
                atg = torch.as_tensor(decode_windows(codes[sl][:, 0], s0, L, s0))
                stp = torch.as_tensor(decode_windows(codes[sl][:, 1], e0 - 1, SL, s0))
                u = torch.as_tensor(struct[sl][:, [0]], dtype=torch.float32)
                _, parts = m(atg[None], stp[None], u[None],
                             torch.ones(1, K, dtype=torch.bool), return_parts=True)
                out[sl] = parts["p_select"][0].numpy()
        sel += out / len(ckpts)
        print(f"  {cp.name}: done", flush=True)

    # ---- the annotated start, projected -------------------------------------
    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category", "associated_transcript"])
    sq = sq[sq.isoform.isin(set(iso)) &
            sq.structural_category.eq("full-splice_match")].copy()
    sq["enst"] = sq.associated_transcript.astype(str).str.split(".").str[0]
    atg = gencode_atg()
    _st = pd.read_csv(CACHE, sep="\t")
    st = {r.isoform_id: (r.strand,
                         [int(x) for x in str(r.starts).split(",")],
                         [int(x) for x in str(r.ends).split(",")])
          for r in _st.itertuples()}
    cds_start = {}
    for r in sq.itertuples():
        g, e = atg.get(r.enst), st.get(r.isoform)
        if g is None or e is None:
            continue
        t = to_transcript(int(g), e[1], e[2], e[0])
        if t is not None:
            cds_start[r.isoform] = t
    print(f"\nchr2 full splice matches with the annotated start projected: "
          f"{len(cds_start):,}")

    rows = []
    for i, nm in enumerate(iso):
        c = cds_start.get(nm)
        if c is None:
            continue
        sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
        s0, e0, m_, ej = o_s[sl], o_e[sl], sel[sl], ejc[sl]
        tot = m_.sum()
        if tot <= 0:
            continue
        up = s0 < c
        rows.append(dict(
            isoform_id=nm, gene=gene[i],
            upstream_mass=m_[up].sum() / tot,
            capable_mass=m_[up & (ej > 0)].sum() / tot,
            overlap_mass=m_[up & (e0 >= c)].sum() / tot,
            annotated_mass=m_[s0 == c].sum() / tot,
            main_ejc=(ej[s0 == c].max() if (s0 == c).any() else np.nan)))
    per = pd.DataFrame(rows)
    per = per[per.main_ejc.notna()].copy()
    tx = pd.read_csv(TABLES / "tx_summary.tsv", sep="\t",
                     usecols=["isoform_id", "is_nmd"])
    per = per.merge(tx, on="isoform_id", how="inner")
    per["grp"] = np.where(per.is_nmd.eq(1) & per.main_ejc.eq(0), "NMD, no main-ORF stop",
                  np.where(per.is_nmd.eq(1), "NMD, main-ORF stop",
                  np.where(per.main_ejc.eq(0), "control, no main-ORF stop",
                           "control, main-ORF stop")))

    print(f"\ntranscripts with a selection distribution and an annotated start: "
          f"{len(per):,}   genes: {per.gene.nunique():,}")
    print(per.grp.value_counts().to_string())
    print(f"\n{'group':<28}{'n':>6}{'upstream':>11}{'capable':>10}{'overlap':>10}"
          f"{'annotated':>11}")
    for g_, s in per.groupby("grp"):
        print(f"{g_:<28}{len(s):>6,}{s.upstream_mass.mean():>11.3f}"
              f"{s.capable_mass.mean():>10.3f}{s.overlap_mass.mean():>10.3f}"
              f"{s.annotated_mass.mean():>11.3f}")

    A, B = "NMD, no main-ORF stop", "control, no main-ORF stop"
    two = per[per.grp.isin([A, B])]
    print(f"\n=== pilot contrast (both have an intact main ORF) ===")
    print(f"    n = {int((two.grp==A).sum())} vs {int((two.grp==B).sum())}, "
          f"{two.gene.nunique()} genes — UNDERPOWERED, see header")
    for col, lab in (("upstream_mass", "mass upstream of the annotated start"),
                     ("capable_mass", "mass on decay-capable upstream ORFs"),
                     ("overlap_mass", "mass on overlapping upstream ORFs"),
                     ("annotated_mass", "mass on the annotated start itself")):
        a, b = two.loc[two.grp == A, col], two.loc[two.grp == B, col]
        lo, hi = gene_boot(two, col, (A, B))
        print(f"  {lab:<40} {a.mean():>6.3f} vs {b.mean():>6.3f}   "
              f"diff {a.mean()-b.mean():>+7.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
        emit("5.90.9", lab, float(a.mean() - b.mean()), n=int(len(two)),
             population="chr2 (VALIDATION) full-splice-match transcripts with the "
                        "GENCODE start admitted and no premature stop in the "
                        f"annotated ORF; '{A}' minus '{B}'; PILOT, underpowered",
             sd_between=float((hi - lo) / 3.92))


if __name__ == "__main__":
    main()
