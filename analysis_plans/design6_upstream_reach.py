#!/usr/bin/env python
"""
DESIGN 6 -- how far upstream the ATG window has to reach.

The plan's ATG window is asymmetric: 900 bases upstream of the start codon and
100 into the ORF. Two coverage figures justify that split, and they were
produced by a command that was never saved, so the plan cited a run log that
does not contain them. This script produces them.

  1. how much of the annotated 5'UTR fits within a given upstream reach
  2. how far a scanning ribosome has travelled to reach an admitted UPSTREAM
     candidate -- that is, bases from the transcript 5' end to that candidate's
     start codon

The second is the operative one: it is the region the initiation head would
have to see in full for an upstream candidate, and it is longer than the 5'UTR
figure because an upstream candidate can sit anywhere within the 5'UTR rather
than at its 3' end.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python design6_upstream_reach.py
"""

import os
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
CAL = os.path.expanduser(
    "~/claude_projects/Isopair/inst/extdata/kozak_mane_calibration.rds")

FR = np.array([[.23, .25, .25, .37, .27, .19, .23, .17],
               [.35, .22, .32, .19, .33, .39, .16, .31],
               [.23, .33, .25, .34, .17, .23, .46, .34],
               [.19, .20, .18, .10, .23, .19, .15, .18]])
PWM = np.log2(FR / .25)
OFF = np.array([-6, -5, -4, -3, -2, -1, 3, 4])
NTI = np.full(256, -1, np.int8)
for _i, _c in enumerate("ACGT"):
    NTI[ord(_c)] = _i
STOPS = {"TAA", "TAG", "TGA"}


def scan(s):
    n, o = len(s), []
    i = s.find("ATG")
    while i != -1:
        j = i + 3
        while j + 3 <= n:
            if s[j:j + 3] in STOPS:
                o.append((i, j + 3))
                break
            j += 3
        i = s.find("ATG", i + 1)
    return o


def kz(c, st):
    if not len(st):
        return np.empty(0)
    p = st[:, None] + OFF[None, :]
    ok = (p >= 0) & (p < len(c))
    b = c[np.clip(p, 0, len(c) - 1)]
    v = ok & (b >= 0)
    x = np.where(v, PWM[np.clip(b, 0, 3), np.arange(8)[None, :]], 0.)
    return np.where(v.sum(1) > 0, x.sum(1), np.nan)


def main():
    r = subprocess.run(["Rscript", "-e", f'x<-readRDS("{CAL}");cat(x$threshold_q05)'],
                       capture_output=True, text=True)
    FLOOR = float(r.stdout.strip())
    print("=" * 88)
    print("DESIGN 6 -- upstream reach required by the ATG window")
    print("=" * 88)
    print(f"\n  admission floor {FLOOR:.4f} (MANE q05), position rule: start in the "
          f"first half")

    z = np.load(os.path.join(HERE, "seq_store.npz"), allow_pickle=False)
    ids, blob, off = z["ids"], z["blob"], z["offsets"]
    ref = pd.read_csv(os.path.join(T, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "ref_utr5_length", "ref_atg_available"]
                      ).drop_duplicates("isoform_id")
    ref = ref[ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()]
    m5 = dict(zip(ref["isoform_id"], ref["ref_utr5_length"].astype(int)))

    u = ref["ref_utr5_length"].astype(int).values
    print("\n" + "=" * 88)
    print("1. ANNOTATED 5'UTR LENGTH")
    print("=" * 88)
    print(f"\n  transcripts with a projected reference start codon: {len(u):,}")
    print(f"  {'percentile':<14} {'bases':>8}")
    for q in (25, 50, 75, 90, 95, 99):
        print(f"  p{q:<13} {np.percentile(u, q):>8.0f}")
    print(f"\n  {'upstream reach':<16} {'5UTR fully covered':>20}")
    print(f"  {'-'*16} {'-'*20}")
    for w in (400, 500, 900, 1900):
        print(f"  {w:<16} {(u <= w).mean()*100:>19.1f}%")

    print("\n" + "=" * 88)
    print("2. DISTANCE FROM THE TRANSCRIPT 5' END TO AN ADMITTED UPSTREAM CANDIDATE")
    print("=" * 88)
    dist = []
    for k, iso in enumerate(ids):
        m = m5.get(iso)
        if m is None:
            continue
        s = blob[int(off[k]):int(off[k + 1])].tobytes().decode("ascii")
        L = len(s)
        o = scan(s)
        if not o:
            continue
        a = np.array(o, dtype=np.int64)
        sc = kz(NTI[np.frombuffer(s.encode("ascii"), np.uint8)], a[:, 0])
        keep = (np.nan_to_num(sc, nan=-1e9) >= FLOOR) & (a[:, 0] <= L // 2)
        h = np.flatnonzero(a[:, 0] == m)
        if len(h):
            keep[h[0]] = True
        st = a[keep, 0]
        dist.extend(st[st < m].tolist())
    d = np.array(dist)
    print(f"\n  admitted candidates starting upstream of the reference start codon: "
          f"{len(d):,}")
    print(f"  {'percentile':<14} {'bases':>8}")
    for q in (25, 50, 75, 90, 99):
        print(f"  p{q:<13} {np.percentile(d, q):>8.0f}")
    print(f"\n  {'upstream reach':<16} {'candidate fully covered':>25}")
    print(f"  {'-'*16} {'-'*25}")
    for w in (400, 500, 900, 1900):
        print(f"  {w:<16} {(d <= w).mean()*100:>24.1f}%")

    print("\n" + "=" * 88)
    print("THE TWO FIGURES THE PLAN CITES")
    print("=" * 88)
    print(f"\n  at 900 bases of upstream reach:")
    print(f"    whole annotated 5'UTR covered, transcripts        "
          f"{(u <= 900).mean()*100:.1f}%")
    print(f"    whole 5' region covered, upstream candidates      "
          f"{(d <= 900).mean()*100:.1f}%")

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)


if __name__ == "__main__":
    main()
