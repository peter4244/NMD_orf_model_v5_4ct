#!/usr/bin/env python
"""
DESIGN 4 -- the two admission rules Pete added.

  RULE 1  discount ORFs starting more than 50% into the transcript, and record
          that outcome.
  RULE 2  where NO AUG clears the Kozak floor, admit the GENCODE CDS start plus
          the top-scoring AUGs from the scan to a total of 5 -- the CDS start
          plus top 4 where the transcript has one, top 5 where it does not.

Read as: "more than 50% into a transcript" applies to the ORF's START position,
since that is the initiation site and the rule is about how far a scanning
ribosome plausibly reaches. Flagged rather than assumed.

MEASURED HERE
  1. what rule 1 removes, and what it costs in coverage of the ORFs the rebuild
     exists for
  2. how many transcripts need rule 2 at all
  3. the interaction: whether rule 1 would remove a GENCODE CDS start that
     rule 2 requires
  4. the pool after both rules

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python design4_position_and_fallback.py
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
    print("DESIGN 4 -- position discount and the empty-pool fallback")
    print("=" * 96)
    print(f"\n  MANE q05 floor {FLOOR:.4f}")

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

    n_floor, n_pos, n_final = [], [], []
    rel_tot = rel_floor = rel_pos = 0
    empty_pool = 0
    empty_with_cds = 0
    cds_past_half = 0
    cds_has = 0
    cds_below_floor = 0
    discounted_total = 0
    print(f"\n  enumerating ...", flush=True)
    for k, iso in enumerate(ids):
        if k % 12000 == 0 and k:
            print(f"    {k:,}", flush=True)
        s = blob[int(off[k]):int(off[k + 1])].tobytes().decode("ascii")
        L = len(s)
        o = scan(s)
        if not o:
            continue
        a = np.array(o, dtype=np.int64)
        sc = kz(NTI[np.frombuffer(s.encode("ascii"), np.uint8)], a[:, 0])
        m = m5.get(iso)

        above = np.nan_to_num(sc, nan=-1e9) >= FLOOR
        half = a[:, 0] <= L // 2                       # rule 1: keep first half
        n_floor.append(int(above.sum()))
        n_pos.append(int((above & half).sum()))
        discounted_total += int((above & ~half).sum())

        if m is not None:
            hit = np.flatnonzero(a[:, 0] == m)
            if len(hit):
                cds_has += 1
                if not above[hit[0]]:
                    cds_below_floor += 1
                if a[hit[0], 0] > L // 2:
                    cds_past_half += 1

        if above.sum() == 0:
            empty_pool += 1
            if m is not None and len(np.flatnonzero(a[:, 0] == m)):
                empty_with_cds += 1
            # rule 2: CDS start + top by score to a total of 5
            order = np.argsort(-np.nan_to_num(sc, nan=-1e9))
            keep = set(order[:5].tolist())
            if m is not None:
                h = np.flatnonzero(a[:, 0] == m)
                if len(h):
                    keep = set(order[:4].tolist()) | {int(h[0])}
            n_final.append(len(keep))
        else:
            n_final.append(int((above & half).sum()))

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
        rel_floor += int((rel & above[up]).sum())
        rel_pos += int((rel & above[up] & half[up]).sum())

    n_floor = np.array(n_floor); n_pos = np.array(n_pos); n_final = np.array(n_final)

    print("\n" + "=" * 96)
    print("1. RULE 1 -- discount ORFs starting past the transcript midpoint")
    print("=" * 96)
    print(f"\n  {'':<34} {'mean/tx':>9} {'median':>8} {'p99':>7} {'max':>7} {'total':>13}")
    print(f"  {'-'*34} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*13}")
    print(f"  {'above the floor':<34} {n_floor.mean():>9.1f} {np.median(n_floor):>8.0f} "
          f"{np.percentile(n_floor,99):>7.0f} {n_floor.max():>7,} {n_floor.sum():>13,}")
    print(f"  {'...and starting in the first half':<34} {n_pos.mean():>9.1f} "
          f"{np.median(n_pos):>8.0f} {np.percentile(n_pos,99):>7.0f} {n_pos.max():>7,} "
          f"{n_pos.sum():>13,}")
    print(f"\n  discounted: {discounted_total:,} candidates "
          f"({discounted_total/max(n_floor.sum(),1)*100:.1f}% of the above-floor pool)")
    print(f"\n  coverage of triggering upstream ORFs ({rel_tot:,} of them):")
    print(f"    above the floor                  {rel_floor/max(rel_tot,1)*100:>6.1f}%")
    print(f"    ...and in the first half         {rel_pos/max(rel_tot,1)*100:>6.1f}%")
    print(f"    cost of rule 1 to coverage       "
          f"{(rel_floor-rel_pos)/max(rel_tot,1)*100:>6.1f} points")

    print("\n" + "=" * 96)
    print("2. RULE 2 -- transcripts where nothing clears the floor")
    print("=" * 96)
    print(f"\n  transcripts with zero candidates above the floor: {empty_pool:,} "
          f"({empty_pool/len(n_floor)*100:.2f}%)")
    print(f"    of those, a GENCODE CDS start is enumerable: {empty_with_cds:,}")
    print(f"    so they get CDS + top 4; the other "
          f"{empty_pool-empty_with_cds:,} get top 5")

    print("\n" + "=" * 96)
    print("3. INTERACTION -- does rule 1 remove a CDS start rule 2 needs?")
    print("=" * 96)
    print(f"\n  transcripts with an enumerable GENCODE CDS start: {cds_has:,}")
    print(f"    its start lies past the transcript midpoint: {cds_past_half:,} "
          f"({cds_past_half/max(cds_has,1)*100:.1f}%)")
    print(f"    it falls below the floor:                    {cds_below_floor:,} "
          f"({cds_below_floor/max(cds_has,1)*100:.1f}%)")

    print("\n" + "=" * 96)
    print("4. THE POOL AFTER BOTH RULES")
    print("=" * 96)
    print(f"\n  mean {n_final.mean():.1f}   median {np.median(n_final):.0f}   "
          f"p90 {np.percentile(n_final,90):.0f}   p99 {np.percentile(n_final,99):.0f}   "
          f"max {n_final.max():,}   total {n_final.sum():,}")
    print(f"  transcripts with zero candidates after both rules: "
          f"{int((n_final==0).sum()):,}")
    print(f"  projected HDF5 at 1000-wide windows: "
          f"{2.9 * (n_final.mean()/5) * 2:.0f} GB")

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
