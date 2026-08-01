#!/usr/bin/env python
"""
DESIGN 5 -- the candidate pool under the settled admission rules.

design1 measured the enumeration, design2 the MANE floor, design4 the position
discount and the empty-pool fallback in isolation. This applies all four rules
together and produces the values §3.4 of the plan predicts.

  1  enumerate every ATG with an in-frame stop, no length floor
  2  score initiation context with the Cavener-Ray PWM
  3  admit at or above the MANE q05 floor (Isopair's threshold_q05)
  4  discount candidates starting past the transcript midpoint
  5  always admit the GENCODE CDS start, whatever its score or position
  6  where the pool is still empty, admit the five highest-scoring candidates
  7  order by start position, 5'->3'

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python design5_final_pool.py
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
    print("=" * 96)
    print("DESIGN 5 -- the pool under all four admission rules")
    print("=" * 96)
    print(f"\n  floor {FLOOR:.4f} (Isopair kozak_mane_calibration.rds, threshold_q05)")

    z = np.load(os.path.join(HERE, "seq_store.npz"), allow_pickle=False)
    ids, blob, off = z["ids"], z["blob"], z["offsets"]
    ref = pd.read_csv(os.path.join(T, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "ref_utr5_length", "ref_atg_available"]
                      ).drop_duplicates("isoform_id")
    ref = ref[ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()]
    m5 = dict(zip(ref["isoform_id"], ref["ref_utr5_length"].astype(int)))
    jd = pd.read_csv(os.path.join(T, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    J = {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
             if j not in ("", "NA") else np.empty(0, dtype=np.int64))
         for i, j in zip(jd["isoform_id"], jd["junctions"])}

    def beyond(j, p):
        return len(j) - int(np.searchsorted(j, p, side="right"))

    n_final, n_disc = [], []
    rel_tot = rel_keep = 0
    cds_has = cds_in = cds_rescued = 0
    fallback = 0
    print("\n  enumerating ...", flush=True)
    for k, iso in enumerate(ids):
        if k % 12000 == 0 and k:
            print(f"    {k:,}", flush=True)
        s = blob[int(off[k]):int(off[k + 1])].tobytes().decode("ascii")
        L = len(s)
        o = scan(s)
        if not o:
            n_final.append(0); n_disc.append(0)
            continue
        a = np.array(o, dtype=np.int64)
        sc = kz(NTI[np.frombuffer(s.encode("ascii"), np.uint8)], a[:, 0])
        m = m5.get(iso)

        above = np.nan_to_num(sc, nan=-1e9) >= FLOOR
        half = a[:, 0] <= L // 2
        keep = above & half
        n_disc.append(int((above & ~half).sum()))

        cds_i = None
        if m is not None:
            h = np.flatnonzero(a[:, 0] == m)
            if len(h):
                cds_i = int(h[0])
                cds_has += 1
                if not keep[cds_i]:
                    cds_rescued += 1
                keep[cds_i] = True            # rule 5: always in
                cds_in += 1

        if keep.sum() == 0:                   # rule 6
            fallback += 1
            order = np.argsort(-np.nan_to_num(sc, nan=-1e9))
            keep = np.zeros(len(a), bool)
            keep[order[:5]] = True

        n_final.append(int(keep.sum()))

        if m is None:
            continue
        j = J.get(iso, np.empty(0, dtype=np.int64))
        if not len(j):
            continue
        up = a[:, 0] < m
        if not up.any():
            continue
        rel = np.array([beyond(j, int(e) + 50) > 0 for e in a[up, 1]])
        if not rel.any():
            continue
        rel_tot += int(rel.sum())
        rel_keep += int((rel & keep[up]).sum())

    n_final = np.array(n_final)
    n_disc = np.array(n_disc)

    print("\n" + "=" * 96)
    print("THE POOL")
    print("=" * 96)
    print(f"\n  transcripts                                  {len(n_final):,}")
    print(f"  candidates per transcript   mean {n_final.mean():.1f}   "
          f"median {np.median(n_final):.0f}   p90 {np.percentile(n_final,90):.0f}   "
          f"p99 {np.percentile(n_final,99):.0f}   max {n_final.max():,}")
    print(f"  candidates total                             {n_final.sum():,}")
    print(f"  transcripts with an empty pool               "
          f"{int((n_final==0).sum()):,}")
    print(f"\n  discounted by the position rule              {n_disc.sum():,}")
    print(f"  fallback fired                               {fallback:,}")

    print("\n" + "=" * 96)
    print("WHAT IT CONTAINS")
    print("=" * 96)
    print(f"\n  GENCODE CDS start enumerable                 {cds_has:,}")
    print(f"    in the pool                                {cds_in:,} "
          f"({cds_in/max(cds_has,1)*100:.1f}%)")
    print(f"    of which admitted ONLY by rule 5           {cds_rescued:,} "
          f"({cds_rescued/max(cds_has,1)*100:.1f}%)")
    print(f"\n  triggering upstream ORFs                     {rel_tot:,}")
    print(f"    in the pool                                {rel_keep:,} "
          f"({rel_keep/max(rel_tot,1)*100:.1f}%)")

    print("\n" + "=" * 96)
    print("COST")
    print("=" * 96)
    print(f"\n  current tensor: 5 slots at 500-wide windows = 2.9 GB")
    print(f"  this pool:      {n_final.mean():.1f} slots at 1000-wide windows "
          f"= {2.9*(n_final.mean()/5)*2:.0f} GB")

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
