#!/usr/bin/env python
"""
DESIGN 3 -- checking Track A's three measurements on my own pool.

They sent three numbers that bear on the plan. Each is checked here rather than
accepted, on the no-floor enumeration under the MANE threshold.

  A  "84.4% of upstream ORFs have their entire ORF, stop included, inside the
     start window at W=1000, median upstream-ORF length 96 nt" -- which would
     make my start-window restriction on the initiation head vacuous for exactly
     the ORFs it was added to protect.

  B  "weak-initiation triggering uORFs carry essentially the signal strong ones
     do" -- 22.9% vs 25.0% NMD across their 3-level score. They flag it is a
     coarse proxy for the PWM floor. Redone here against the PWM itself, and
     against the MANE cuts specifically, because it decides q05 against q01.

  C  cap sizes. Theirs are on the current 33 nt-floor pool; mine is ~1.5x larger
     with the floor removed, so the numbers do not transfer.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python design3_check_track_a.py
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
STORE = os.path.join(HERE, "seq_store.npz")
CALIB = os.path.expanduser(
    "~/claude_projects/Isopair/inst/extdata/kozak_mane_calibration.rds")

FREQS = np.array([
    [0.23, 0.25, 0.25, 0.37, 0.27, 0.19, 0.23, 0.17],
    [0.35, 0.22, 0.32, 0.19, 0.33, 0.39, 0.16, 0.31],
    [0.23, 0.33, 0.25, 0.34, 0.17, 0.23, 0.46, 0.34],
    [0.19, 0.20, 0.18, 0.10, 0.23, 0.19, 0.15, 0.18],
])
PWM = np.log2(FREQS / 0.25)
OFFSETS = np.array([-6, -5, -4, -3, -2, -1, 3, 4])
NTI = np.full(256, -1, dtype=np.int8)
for _i, _c in enumerate("ACGT"):
    NTI[ord(_c)] = _i
STOPS = {"TAA", "TAG", "TGA"}


def mane():
    r = subprocess.run(["Rscript", "-e",
        f'x <- readRDS("{CALIB}"); cat(sprintf("{{\\"q01\\":%.6f,\\"q05\\":%.6f}}",'
        ' x$threshold_q01, x$threshold_q05))'], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr)
    return json.loads(r.stdout.strip())


def scan(s):
    n, out = len(s), []
    i = s.find("ATG")
    while i != -1:
        j = i + 3
        while j + 3 <= n:
            if s[j:j + 3] in STOPS:
                out.append((i, j + 3))
                break
            j += 3
        i = s.find("ATG", i + 1)
    return out


def kz(codes, starts):
    if not len(starts):
        return np.empty(0)
    pos = starts[:, None] + OFFSETS[None, :]
    ok = (pos >= 0) & (pos < len(codes))
    b = codes[np.clip(pos, 0, len(codes) - 1)]
    v = ok & (b >= 0)
    c = np.where(v, PWM[np.clip(b, 0, 3), np.arange(8)[None, :]], 0.0)
    return np.where(v.sum(axis=1) > 0, c.sum(axis=1), np.nan)


def main():
    thr = mane()
    print("=" * 96)
    print("DESIGN 3 -- checking Track A on my own pool")
    print("=" * 96)
    print(f"\n  MANE cuts: q05 {thr['q05']:.4f}   q01 {thr['q01']:.4f}")

    ids, blob, off = np.load(STORE, allow_pickle=False).values()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd"]]
    lab = dict(zip(tx["isoform_id"], tx["is_nmd"]))
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "ref_utr5_length", "ref_atg_available"]
                      ).drop_duplicates("isoform_id")
    ref = ref[ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()]
    matg = dict(zip(ref["isoform_id"], ref["ref_utr5_length"].astype(int)))
    jdf = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                      dtype=str, keep_default_na=False)
    junc = {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(jdf["isoform_id"], jdf["junctions"])}

    up_len, trig_len, trig_score, trig_lab = [], [], [], []
    slot_counts = []
    print(f"\n  enumerating ...", flush=True)
    for k, iso in enumerate(ids):
        if k % 12000 == 0 and k:
            print(f"    {k:,}", flush=True)
        s = blob[int(off[k]):int(off[k + 1])].tobytes().decode("ascii")
        o = scan(s)
        if not o:
            continue
        a = np.array(o, dtype=np.int64)
        codes = NTI[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        sc = kz(codes, a[:, 0])
        m = matg.get(iso)
        keep = (sc >= thr["q05"])
        if m is not None:
            keep = keep | (a[:, 0] == m)
        slot_counts.append(int(keep.sum()))
        if m is None:
            continue
        up = a[:, 0] < m
        if not up.any():
            continue
        L = a[up, 1] - a[up, 0]
        up_len.extend(L.tolist())
        j = junc.get(iso, np.empty(0, dtype=np.int64))
        if not len(j):
            continue
        ends = a[up, 1]
        rel = np.array([len(j) - int(np.searchsorted(j, int(e) + 50,
                                                     side="right")) > 0
                        for e in ends])
        if rel.any():
            trig_len.extend(L[rel].tolist())
            trig_score.extend(sc[up][rel].tolist())
            trig_lab.extend([lab.get(iso, 0)] * int(rel.sum()))

    up_len = np.array(up_len)
    trig_score = np.array(trig_score)
    trig_lab = np.array(trig_lab)
    slot_counts = np.array(slot_counts)

    print("\n" + "=" * 96)
    print("A. DOES THE START WINDOW ALREADY CONTAIN THE STOP?")
    print("=" * 96)
    print(f"\n  upstream ORFs enumerated: {len(up_len):,}")
    print(f"  length: median {np.median(up_len):.0f} nt   "
          f"p75 {np.percentile(up_len,75):.0f}   p90 {np.percentile(up_len,90):.0f}")
    print(f"\n  A start window of width W centred on the start codon reaches")
    print(f"  W/2 downstream, so an ORF of length <= W/2 has its stop inside it.\n")
    print(f"  {'window W':<12} {'reaches':>9} {'upstream ORFs with stop inside':>32}")
    print(f"  {'-'*12} {'-'*9} {'-'*32}")
    for W in (1000, 500, 200, 100, 50):
        print(f"  {W:<12} {W//2:>9} {(up_len <= W//2).mean()*100:>31.1f}%")
    print(f"\n  Track A reported 84.4% at W=1000 and median 96 nt.")
    print(f"  Here: {(up_len <= 500).mean()*100:.1f}% and {np.median(up_len):.0f} nt.")

    print("\n" + "=" * 96)
    print("B. DO WEAK-INITIATION TRIGGERING uORFs CARRY SIGNAL?")
    print("=" * 96)
    print(f"\n  triggering upstream ORFs (own stop has a junction >50 nt "
          f"downstream): {len(trig_score):,}")
    print(f"  background NMD+ over all transcripts: {tx['is_nmd'].mean()*100:.1f}%\n")
    print(f"  {'PWM band':<24} {'n uORFs':>10} {'NMD+ of their transcript':>26}")
    print(f"  {'-'*24} {'-'*10} {'-'*26}")
    bands = [(-99, thr["q01"], f"below q01 ({thr['q01']:.2f})"),
             (thr["q01"], thr["q05"], "q01 to q05"),
             (thr["q05"], 0.0, "q05 to 0"),
             (0.0, 99, "above 0")]
    for lo, hi, nm in bands:
        m = (trig_score > lo) & (trig_score <= hi)
        if m.sum():
            print(f"  {nm:<24} {m.sum():>10,} {trig_lab[m].mean()*100:>25.1f}%")
    below = trig_score <= thr["q05"]
    above = trig_score > thr["q05"]
    print(f"\n  pooled at the default cut:")
    print(f"    discarded by q05  {below.sum():>8,} uORFs   "
          f"NMD+ {trig_lab[below].mean()*100:.1f}%")
    print(f"    kept by q05       {above.sum():>8,} uORFs   "
          f"NMD+ {trig_lab[above].mean()*100:.1f}%")
    b2 = trig_score <= thr["q01"]
    print(f"    discarded by q01  {b2.sum():>8,} uORFs   "
          f"NMD+ {trig_lab[b2].mean()*100:.1f}%")

    print("\n" + "=" * 96)
    print("C. THE TAIL, ON THIS POOL")
    print("=" * 96)
    print(f"\n  candidates per transcript at MANE q05 with the main ORF forced in:")
    print(f"    p50 {np.median(slot_counts):.0f}   "
          f"p90 {np.percentile(slot_counts,90):.0f}   "
          f"p99 {np.percentile(slot_counts,99):.0f}   max {slot_counts.max():,}")
    print(f"\n  {'cap':<10} {'transcripts truncated':>24} {'share':>9} "
          f"{'candidates dropped':>20}")
    print(f"  {'-'*10} {'-'*24} {'-'*9} {'-'*20}")
    for cap in (50, 100, 200, 400, 800):
        t = slot_counts > cap
        print(f"  {cap:<10} {t.sum():>24,} {t.mean()*100:>8.2f}% "
              f"{np.clip(slot_counts - cap, 0, None).sum():>20,}")

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
