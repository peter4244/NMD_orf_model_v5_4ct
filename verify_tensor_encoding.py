#!/usr/bin/env python
"""
verify_tensor_encoding.py — does build_tensor.py encode what section 5 specifies?

Decodes sampled candidates back out of the HDF5 and checks each against the
transcript sequence, independently of the code that wrote them.

  atg_anchor_base    array index ATG_LEFT holds the A of the start codon
  stop_anchor_base   array index STOP_LEFT holds the middle base of the stop codon
  onehot_sum         channels 0-3 sum to exactly 1 at every fillable position and
                     to 0 at every padded one
  frame_at_anchor    channel 6 is 1 at the ATG anchor, so codon position is
                     measured from this candidate's own start codon
  junction_channel   channel 4 marks exactly the positions listed in JUNC that
                     fall inside the filled region
  windows_overlap    no transcript position is filled in both windows
  gc_range           channel 5 stays inside [0, 1]

Usage:
    python verify_tensor_encoding.py --tensor results_tensor_chr21 --n 400
"""
import argparse, os
import h5py, numpy as np, pandas as pd

T = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
FASTA = os.path.expanduser(
    "~/claude_projects/nmd_deposit_2026/source_data/sqanti/nmd_lungcells_corrected.fasta")
NUC = "ACGT"

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", default="results_tensor_chr21")
ap.add_argument("--pool", default="results_pool_v6")
ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()
H = os.path.join(a.tensor, "nmd_tensor.h5")

with h5py.File(H) as f:
    ids = set(s.decode() for s in f["isoform_id"][:])
    AL, SL = int(f.attrs["atg_left"]), int(f.attrs["stop_left"])
    W = int(f.attrs["window"])

pool = pd.read_csv(os.path.join(a.pool, "orf_pool.tsv"), sep="\t")
pool = pool[pool.isoform_id.isin(ids)].sort_values(
    ["isoform_id", "slot"], kind="stable").reset_index(drop=True)
jd = pd.read_csv(os.path.join(T, "junctions.tsv"), sep="\t", dtype=str,
                 keep_default_na=False)
J = {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
         if j not in ("", "NA") else np.empty(0, np.int64))
     for i, j in zip(jd.isoform_id, jd.junctions)}
seqs, name, buf = {}, None, []
with open(FASTA) as fh:
    for line in fh:
        if line[0] == ">":
            if name in ids: seqs[name] = "".join(buf)
            name = line[1:].split()[0]; buf = []
        else: buf.append(line.strip())
    if name in ids: seqs[name] = "".join(buf)

rng = np.random.default_rng(0)
pick = sorted(rng.choice(len(pool), min(a.n, len(pool)), replace=False))
bad = dict.fromkeys(["atg_anchor_base", "stop_anchor_base", "onehot_sum",
                     "frame_at_anchor", "junction_channel", "windows_overlap",
                     "gc_range"], 0)
with h5py.File(H) as f:
    A, S = f["atg_windows"], f["stop_windows"]
    for i in pick:
        r = pool.iloc[i]; iso = r.isoform_id
        s, e = int(r.orf_start), int(r.orf_end)
        seq = seqs[iso]; L = len(seq); mid = (s + e) // 2
        aw = np.asarray(A[i], np.float32); sw = np.asarray(S[i], np.float32)
        if not (aw[:4, AL].sum() == 1 and NUC[int(np.argmax(aw[:4, AL]))] == seq[s - 1]):
            bad["atg_anchor_base"] += 1
        if not (sw[:4, SL].sum() == 1 and NUC[int(np.argmax(sw[:4, SL]))] == seq[e - 2]):
            bad["stop_anchor_base"] += 1
        for win, anchor, left, lo, hi in ((aw, s, AL, 1, mid), (sw, e - 1, SL, mid + 1, L)):
            pos = anchor - left + np.arange(W)
            filled = (pos >= max(1, lo)) & (pos <= min(L, hi))
            oh = win[:4].sum(0)
            if not (np.allclose(oh[filled], 1.0) and np.allclose(oh[~filled], 0.0)):
                bad["onehot_sum"] += 1
            if win[5].min() < -1e-3 or win[5].max() > 1 + 1e-3:
                bad["gc_range"] += 1
        if aw[6, AL] != 1.0:
            bad["frame_at_anchor"] += 1
        pa = s - AL + np.arange(W); fa = (pa >= 1) & (pa <= min(L, mid))
        if not np.array_equal(aw[4] > 0.5, np.isin(pa, J.get(iso, np.empty(0, np.int64))) & fa):
            bad["junction_channel"] += 1
        ps = e - 1 - SL + np.arange(W); fs = (ps >= mid + 1) & (ps <= L)
        if set(pa[fa].tolist()) & set(ps[fs].tolist()):
            bad["windows_overlap"] += 1

print(f"{len(pick)} candidates from {len(ids)} transcripts, {H}\n")
for k, v in bad.items():
    print(f"  {'FAIL' if v else 'pass'}  {k:<20} {v}")
raise SystemExit(1 if any(bad.values()) else 0)
