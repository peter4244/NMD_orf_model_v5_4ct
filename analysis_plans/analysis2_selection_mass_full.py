#!/usr/bin/env python
"""
analysis2_selection_mass_full.py — where does the model send the ribosome?
Full universe, replacing the chr2 pilot, and the ATF4 case study in the same pass.

WHAT CHANGED FROM THE PILOT. `results_tensor_v6` now exists locally: 796,584
candidates over 41,765 transcripts, all five split labels. The chr2 pilot carried
21 transcripts in the group of interest against 381 genome-wide, so its group
contrast was a code check rather than a result.

SCOPE IS ALL DATA, NOT TEST-ONLY, and that is a standing decision rather than a
convenience: only AUC and AUPRC are test-restricted on this project;
interpretability and descriptive analyses use everything, with a test-only
sensitivity reported alongside. Both are printed here.

TWO OUTPUTS FROM ONE SWEEP. The selection distribution is expensive to compute
and cheap to re-cut, so this caches it and then answers two questions:

  1. the group contrast — does the model route more initiation upstream in
     isoforms degraded WITHOUT a premature stop in their main reading frame
  2. ATF4 — a case whose mechanism is established independently of anything here

WHY ATF4 IS THE RIGHT CASE. Its main ORF is intact, so every CDS-based classifier
calls it "no premature stop". Its decay comes from architecture: a short permissive
upstream ORF, then a second one that OVERLAPS the main start out of frame, so
ribosomes captured there terminate past the main AUG rather than translating it.
Neither 5'UTR length nor longest-upstream-ORF length encodes that. If the model
has learned the mechanism, mass should land on an overlapping upstream candidate.
Verified in scope beforehand: ATF4 is NMD-responsive in all four manuscript cell
types, three isoforms, logFC 1.49-3.22, adj.P to 1.3e-33.

Batched rather than per-transcript: the pilot's transcript-at-a-time loop would
take hours at this size.
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

TENSOR = REPO / "results_tensor_v6" / "nmd_tensor.h5"
CKPT = REPO / "results_interp_all" / "ckpt_interp_c32_b8"
CACHEF = REPO / "results_interp_all" / "p_select_v6.npz"
GTF = NMD / "reference_files" / "gencode.v49.primary_assembly.annotation.gtf.gz"
SQANTI = NMD / "sqanti" / "nmd_lungcells" / "results" / "nmd_lungcells_classification.txt"
STRUCTS = Path("/private/tmp/claude-502/-Users-petecastaldi/"
               "6ea6b1ee-b03e-42c4-8a85-487850841c94/scratchpad/structures.tsv")
TX_RE = re.compile(r'transcript_id "([^"]+)"')


def batches(counts, max_padded=2048):
    order = np.argsort(counts, kind="stable")
    out, cur, cmax = [], [], 0
    for i in order:
        k = int(counts[i]); nm = max(cmax, k)
        if cur and (len(cur) + 1) * nm > max_padded:
            out.append(np.array(cur)); cur, cmax = [i], k
        else:
            cur.append(i); cmax = nm
    if cur:
        out.append(np.array(cur))
    return out


def compute_p_select():
    if CACHEF.exists():
        z = np.load(CACHEF)
        print(f"loaded cached selection mass from {CACHEF.name}")
        return z["sel"]
    with h5py.File(TENSOR, "r") as f:
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:].astype(np.int64), f["orf_end"][:].astype(np.int64)
        struct = f["structural"][:]
        codes = f["codes"][:]
        L, SL = int(f.attrs["atg_left"]), int(f.attrs["stop_left"])
    n_cand = codes.shape[0]
    print(f"tensor: {len(off):,} transcripts, {n_cand:,} candidates")

    ckpts = sorted(CKPT.glob("b8_s*.pt"))
    sel = np.zeros(n_cand, dtype=np.float64)
    bl = batches(cnt)
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1, permute_bins=False)
        m.load_state_dict(ck["model"]); m.eval()
        acc = np.zeros(n_cand, dtype=np.float64)
        with torch.no_grad():
            for bi, b in enumerate(bl):
                c = cnt[b].astype(int)
                K = int(c.max())
                rows = np.concatenate([np.arange(off[i], off[i] + cnt[i]) for i in b])
                s0, e0 = o_s[rows], o_e[rows]
                atg = decode_windows(codes[rows][:, 0], s0, L, s0)
                stp = decode_windows(codes[rows][:, 1], e0 - 1, SL, s0)
                st = struct[rows][:, [0]]
                W = atg.shape[2]
                A = np.zeros((len(b), K, 9, W), np.float32)
                S = np.zeros_like(A)
                U = np.zeros((len(b), K, 1), np.float32)
                M = np.zeros((len(b), K), bool)
                at = 0
                for j, cc in enumerate(c):
                    A[j, :cc] = atg[at:at + cc]; S[j, :cc] = stp[at:at + cc]
                    U[j, :cc] = st[at:at + cc]; M[j, :cc] = True
                    at += cc
                _, parts = m(torch.as_tensor(A), torch.as_tensor(S),
                             torch.as_tensor(U), torch.as_tensor(M),
                             return_parts=True)
                ps = parts["p_select"].numpy()
                at = 0
                for j, i in enumerate(b):
                    cc = int(cnt[i])
                    acc[off[i]:off[i] + cc] = ps[j, :cc]
                    at += cc
                if bi % 200 == 0:
                    print(f"    {cp.name} batch {bi}/{len(bl)}", flush=True)
        sel += acc / len(ckpts)
        print(f"  {cp.name}: done", flush=True)
    np.savez_compressed(CACHEF, sel=sel)
    print(f"cached -> {CACHEF}")
    return sel


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


def gene_boot(df, col, groups, n=4000, seed=20260801):
    d = df.dropna(subset=[col])
    a = (d.grp == groups[0]).to_numpy(float); b = (d.grp == groups[1]).to_numpy(float)
    v = d[col].to_numpy(float)
    gi, ug = pd.factorize(d.gene); G = len(ug)
    sa = np.bincount(gi, weights=v * a, minlength=G); na = np.bincount(gi, weights=a, minlength=G)
    sb = np.bincount(gi, weights=v * b, minlength=G); nb = np.bincount(gi, weights=b, minlength=G)
    idx = np.random.default_rng(seed).integers(0, G, size=(n, G))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (sa[idx].sum(1) / na[idx].sum(1)) - (sb[idx].sum(1) / nb[idx].sum(1))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sel = compute_p_select()

    with h5py.File(TENSOR, "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        gene = np.array([s.decode() for s in f["gene_id"][:]])
        split = np.array([s.decode() for s in f["split"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:].astype(np.int64), f["orf_end"][:].astype(np.int64)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "n_downstream_ejc", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))
    ejc = pool.n_downstream_ejc.to_numpy()
    koz = pool.kozak_score.to_numpy()

    sq = pd.read_csv(SQANTI, sep="\t", low_memory=False,
                     usecols=["isoform", "structural_category", "associated_transcript"])
    sq = sq[sq.isoform.isin(set(iso)) & sq.structural_category.eq("full-splice_match")].copy()
    sq["enst"] = sq.associated_transcript.astype(str).str.split(".").str[0]
    atg = gencode_atg()
    _s = pd.read_csv(STRUCTS, sep="\t")
    st = {r.isoform_id: (r.strand, [int(x) for x in str(r.starts).split(",")],
                         [int(x) for x in str(r.ends).split(",")]) for r in _s.itertuples()}
    cds = {}
    for r in sq.itertuples():
        g, e = atg.get(r.enst), st.get(r.isoform)
        if g is None or e is None:
            continue
        t = to_tx(int(g), e[1], e[2], e[0])
        if t is not None:
            cds[r.isoform] = t
    print(f"\nannotated start projected for {len(cds):,} transcripts in the tensor")

    rows = []
    for i, nm in enumerate(iso):
        c = cds.get(nm)
        if c is None:
            continue
        sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
        s0, e0, m_, ej = o_s[sl], o_e[sl], sel[sl], ejc[sl]
        tot = m_.sum()
        if tot <= 0:
            continue
        up = s0 < c
        rows.append(dict(isoform_id=nm, gene=gene[i], split=split[i],
                         upstream_mass=m_[up].sum() / tot,
                         capable_mass=m_[up & (ej > 0)].sum() / tot,
                         overlap_mass=m_[up & (e0 >= c)].sum() / tot,
                         annotated_mass=m_[s0 == c].sum() / tot,
                         main_ejc=(ej[s0 == c].max() if (s0 == c).any() else np.nan)))
    per = pd.DataFrame(rows)
    per = per[per.main_ejc.notna()]
    tx = pd.read_csv(TABLES / "tx_summary.tsv", sep="\t", usecols=["isoform_id", "is_nmd"])
    per = per.merge(tx, on="isoform_id", how="inner")
    per["grp"] = np.where(per.is_nmd.eq(1) & per.main_ejc.eq(0), "NMD, no main-ORF stop",
                  np.where(per.is_nmd.eq(1), "NMD, main-ORF stop",
                  np.where(per.main_ejc.eq(0), "control, no main-ORF stop",
                           "control, main-ORF stop")))
    A, B = "NMD, no main-ORF stop", "control, no main-ORF stop"

    for scope, sub in (("ALL DATA", per), ("test split only", per[per.split == "test"])):
        two = sub[sub.grp.isin([A, B])]
        print(f"\n=== {scope}: {len(sub):,} transcripts, {sub.gene.nunique():,} genes ===")
        print(sub.grp.value_counts().to_string())
        print(f"  {'group':<28}{'upstream':>10}{'capable':>10}{'overlap':>10}{'annot':>10}")
        for g_, s in sub.groupby("grp"):
            print(f"  {g_:<28}{s.upstream_mass.mean():>10.3f}{s.capable_mass.mean():>10.3f}"
                  f"{s.overlap_mass.mean():>10.3f}{s.annotated_mass.mean():>10.3f}")
        print(f"  contrast n = {int((two.grp==A).sum())} vs {int((two.grp==B).sum())}, "
              f"{two.gene.nunique()} genes")
        for col, lab in (("upstream_mass", "mass upstream of the annotated start"),
                         ("capable_mass", "mass on decay-capable upstream ORFs"),
                         ("overlap_mass", "mass on overlapping upstream ORFs"),
                         ("annotated_mass", "mass on the annotated start itself")):
            a_, b_ = two.loc[two.grp == A, col], two.loc[two.grp == B, col]
            lo, hi = gene_boot(two, col, (A, B))
            print(f"    {lab:<40}{a_.mean():>7.3f} vs{b_.mean():>7.3f}  "
                  f"diff {a_.mean()-b_.mean():>+7.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
            if scope == "ALL DATA":
                emit("5.90.10", lab, float(a_.mean() - b_.mean()), n=int(len(two)),
                     population="all splits; full-splice-match transcripts with the "
                                "GENCODE-annotated start admitted and no premature stop "
                                f"in the annotated ORF; '{A}' minus '{B}'",
                     sd_between=float((hi - lo) / 3.92))

    # ---------------------------------------------------------------- ATF4
    print(f"\n=== ATF4 — a case whose mechanism is established independently ===")
    idx = [i for i, nm in enumerate(iso) if nm.startswith("ENSG00000128272")
           or nm in ("ENST00000674920.3", "ENST00000337304.2")]
    if not idx:
        print("  no ATF4 isoform in this tensor")
    for i in idx:
        nm = iso[i]
        sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
        s0, e0, m_, ej, kz = o_s[sl], o_e[sl], sel[sl], ejc[sl], koz[sl]
        c = cds.get(nm)
        tot = m_.sum()
        lab = f"{nm}  split={split[i]}  candidates={int(cnt[i])}"
        print(f"\n  {lab}")
        if c is None:
            print("    no GENCODE-annotated start projected for this isoform")
        else:
            up = s0 < c
            print(f"    annotated start at tx position {c}; "
                  f"upstream candidates {int(up.sum())}")
            print(f"    mass upstream {m_[up].sum()/tot:.3f}   "
                  f"on decay-capable {m_[up&(ej>0)].sum()/tot:.3f}   "
                  f"on overlapping {m_[up&(e0>=c)].sum()/tot:.3f}   "
                  f"on annotated {m_[s0==c].sum()/tot:.3f}")
        top = np.argsort(-m_)[:6]
        print(f"    {'rank':>4} {'orf_start':>10} {'orf_end':>9} {'len':>6} "
              f"{'ejc':>4} {'kozak':>7} {'P(select)':>10} {'position':>12}")
        for r, k in enumerate(top, 1):
            posn = ("upstream" if c is not None and s0[k] < c else
                    "annotated" if c is not None and s0[k] == c else "downstream")
            if c is not None and s0[k] < c and e0[k] >= c:
                posn = "OVERLAPPING"
            print(f"    {r:>4} {s0[k]:>10,} {e0[k]:>9,} {e0[k]-s0[k]+1:>6,} "
                  f"{int(ej[k]):>4} {kz[k]:>7.2f} {m_[k]/tot:>10.3f} {posn:>12}")


if __name__ == "__main__":
    main()
