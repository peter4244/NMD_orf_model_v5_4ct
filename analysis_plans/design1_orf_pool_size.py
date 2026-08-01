#!/usr/bin/env python
"""
DESIGN 1 -- how big does the candidate ORF pool get with no length floor?

PETE'S INSTRUCTION, WHICH IS WHY THIS RUNS BEFORE ANYTHING IS BUILT
  "No minimum ORF length -- just go with the Kozak scoring threshold idea."
  And: "Measure before building: with no length floor, every start codon with an
  in-frame stop becomes a candidate, including one- and two-codon ones.
  Start-stop elements are real, so this is defensible, but the threshold then
  becomes the only thing controlling pool size and we should know how large it
  gets first."

  That measurement has not been done. Everything known about pool size was
  measured WITH the 33 nt floor still in place, so it does not answer the
  question.

WHAT THE FLOOR ACTUALLY IS -- corrected by Track A, verified here
  05s_orfik_scan.R:112 calls ORFik::findORFs(..., minimumLength = 9). ORFik
  measures that in CODONS EXCLUDING the start and stop, so 9 gives
  3 + 9*3 + 3 = 33 nt -- exactly the observed floor, with 59,425 ORFs sitting on
  it. Pete's ruling therefore means minimumLength = 0, i.e. START + STOP = 6 nt.
  Anyone who reads the field name as nucleotides sets 9 again and gets 33 again.

WHY THIS CANNOT BE ANSWERED FROM THE EXISTING TABLES
  orf_features.tsv was produced WITH the floor. Every ORF below 33 nt was never
  written. So the pool is rescanned here from the transcript sequences, in
  Python, with no floor at all -- the same operation findORFs performs.

WHAT IS REPORTED
  1. how many candidates per transcript at floor 0 against floor 33
  2. the Kozak PWM score distribution of the new short ones -- if start-stop
     elements admit mostly on weak context, the threshold handles them
  3. pool size at thresholds CALIBRATED ON REAL ANNOTATED START CODONS, which
     is the only principled way to set it
  4. what it costs: slots per transcript drives HDF5 size and training time
     linearly, and infer_uorf_attention.py silently truncates above K=5

The PWM is the Cavener & Ray 1991 matrix used by Isopair::scoreKozakPWM, copied
from kozak_pwm_rescore.py where it was validated against the R implementation to
4.99e-13.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python design1_orf_pool_size.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
STORE = os.path.join(HERE, "seq_store.npz")

FREQS = np.array([
    [0.23, 0.25, 0.25, 0.37, 0.27, 0.19, 0.23, 0.17],   # A
    [0.35, 0.22, 0.32, 0.19, 0.33, 0.39, 0.16, 0.31],   # C
    [0.23, 0.33, 0.25, 0.34, 0.17, 0.23, 0.46, 0.34],   # G
    [0.19, 0.20, 0.18, 0.10, 0.23, 0.19, 0.15, 0.18],   # T
])
PWM = np.log2(FREQS / 0.25)
OFFSETS = np.array([-6, -5, -4, -3, -2, -1, 3, 4])      # relative to A of ATG, 0-based
NTI = np.full(256, -1, dtype=np.int8)
for i, c in enumerate("ACGT"):
    NTI[ord(c)] = i

STOPS = {"TAA", "TAG", "TGA"}


def load_seqs():
    z = np.load(STORE, allow_pickle=False)
    return z["ids"], z["blob"], z["offsets"]


def scan_transcript(s):
    """Every ATG with an in-frame downstream stop, no length floor.
    Returns (start0, end0, length) with 0-based inclusive-exclusive starts."""
    n = len(s)
    out = []
    i = s.find("ATG")
    while i != -1:
        j = i + 3
        while j + 3 <= n:
            if s[j:j + 3] in STOPS:
                out.append((i, j + 3, j + 3 - i))
                break
            j += 3
        i = s.find("ATG", i + 1)
    return out


def kozak_scores(codes, starts):
    """Vectorised PWM score for many ATG positions in one transcript.
    codes is an int8 array of base indices, -1 for anything else."""
    if not len(starts):
        return np.empty(0)
    pos = starts[:, None] + OFFSETS[None, :]
    ok = (pos >= 0) & (pos < len(codes))
    idx = np.clip(pos, 0, len(codes) - 1)
    b = codes[idx]
    valid = ok & (b >= 0)
    contrib = np.where(valid, PWM[np.clip(b, 0, 3), np.arange(8)[None, :]], 0.0)
    nvalid = valid.sum(axis=1)
    s = contrib.sum(axis=1)
    return np.where(nvalid > 0, s, np.nan)


def main():
    print("=" * 100)
    print("DESIGN 1 -- candidate ORF pool size with no length floor")
    print("=" * 100)

    ids, blob, off = load_seqs()
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "ref_utr5_length", "ref_atg_available"]
                      ).drop_duplicates("isoform_id")
    ref = ref[ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()]
    main_atg = dict(zip(ref["isoform_id"], ref["ref_utr5_length"].astype(int)))
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd"]]
    lab = dict(zip(tx["isoform_id"], tx["is_nmd"]))

    print(f"\n  rescanning {len(ids):,} transcripts with NO length floor ...",
          flush=True)
    rows = []
    ann_scores = []
    for k, iso in enumerate(ids):
        if k % 8000 == 0 and k:
            print(f"    {k:,} ...", flush=True)
        s = blob[int(off[k]):int(off[k + 1])].tobytes().decode("ascii")
        codes = NTI[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        orfs = scan_transcript(s)
        if not orfs:
            continue
        a = np.array(orfs, dtype=np.int64)
        sc = kozak_scores(codes, a[:, 0])
        m = main_atg.get(iso)
        rows.append((iso, a, sc, m))
        if m is not None:
            hit = np.flatnonzero(a[:, 0] == m)          # ref AUG is at 0-based utr5len
            if len(hit):
                ann_scores.append(sc[hit[0]])
    print(f"  scanned; {len(rows):,} transcripts with at least one ORF")

    n_all = np.array([len(r[1]) for r in rows])
    n_33 = np.array([(r[1][:, 2] >= 33).sum() for r in rows])
    print("\n" + "=" * 100)
    print("1. HOW MANY CANDIDATES, FLOOR 0 AGAINST FLOOR 33")
    print("=" * 100)
    print(f"\n  {'':<22} {'mean':>8} {'median':>8} {'p90':>8} {'p99':>8} {'max':>8} "
          f"{'total':>14}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*14}")
    for nm, v in (("floor 33 nt (current)", n_33), ("floor 0 (Pete's ruling)", n_all)):
        print(f"  {nm:<22} {v.mean():>8.1f} {np.median(v):>8.0f} "
              f"{np.percentile(v,90):>8.0f} {np.percentile(v,99):>8.0f} "
              f"{v.max():>8,} {v.sum():>14,}")
    print(f"\n  dropping the floor multiplies the pool by "
          f"{n_all.sum()/max(n_33.sum(),1):.2f}x")
    lens = np.concatenate([r[1][:, 2] for r in rows])
    print(f"  ORFs shorter than 33 nt: {(lens < 33).sum():,} "
          f"({(lens < 33).mean()*100:.1f}% of the new pool)")
    for c in (6, 9, 12, 15, 21, 33):
        print(f"    exactly {c:>2} nt ({(c-6)//3} codon{'s' if c != 9 else ' '} "
              f"between start and stop): {(lens == c).sum():>8,}")

    # ------------------------------------------------------------------ 2
    print("\n" + "=" * 100)
    print("2. ARE THE NEW SHORT ORFs WEAK? -- Kozak score by length")
    print("=" * 100)
    allsc = np.concatenate([r[2] for r in rows])
    print(f"\n  {'ORF length':<18} {'n':>10} {'median PWM':>12} {'p75':>8} {'p90':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")
    for lo, hi, nm in ((0, 15, "6-15 nt"), (15, 33, "16-32 nt"),
                       (33, 100, "33-99 nt"), (100, 10**9, ">=100 nt")):
        m = (lens >= lo) & (lens < hi)
        if m.sum() == 0:
            continue
        v = allsc[m]
        v = v[~np.isnan(v)]
        print(f"  {nm:<18} {m.sum():>10,} {np.median(v):>12.3f} "
              f"{np.percentile(v,75):>8.3f} {np.percentile(v,90):>8.3f}")

    ann = np.array([x for x in ann_scores if not np.isnan(x)])
    print(f"\n  REAL ANNOTATED START CODONS (the calibration set), n = {len(ann):,}")
    for q in (1, 5, 10, 25, 50):
        print(f"    {q:>2}th percentile of annotated starts: "
              f"{np.percentile(ann, q):>7.3f}")

    # ------------------------------------------------------------------ 3
    print("\n" + "=" * 100)
    print("3. POOL SIZE AT THRESHOLDS CALIBRATED ON ANNOTATED START CODONS")
    print("=" * 100)
    print("\n  A threshold set at the qth percentile of real start codons admits")
    print("  candidates at least as good as (100-q)% of genuine initiation sites.\n")
    print(f"  {'threshold':<26} {'PWM cut':>9} {'mean slots':>11} {'median':>8} "
          f"{'p99':>8} {'max':>8} {'total':>14}")
    print(f"  {'-'*26} {'-'*9} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*14}")
    keep_by_thr = {}
    for q in (1, 5, 10, 25, 50):
        cut = np.percentile(ann, q)
        counts = np.array([np.nansum(r[2] >= cut) for r in rows])
        keep_by_thr[q] = (cut, counts)
        print(f"  {'annotated q'+str(q):<26} {cut:>9.3f} {counts.mean():>11.1f} "
              f"{np.median(counts):>8.0f} {np.percentile(counts,99):>8.0f} "
              f"{counts.max():>8,} {counts.sum():>14,}")
    print(f"\n  current model: exactly 5 slots per transcript, "
          f"{5*len(rows):,} total")

    # ------------------------------------------------------------------ 4
    print("\n" + "=" * 100)
    print("4. WHAT IT COSTS, AND THE LANDMINE")
    print("=" * 100)
    cur = 5
    print(f"\n  HDF5 size and training time scale linearly in slots per transcript.")
    print(f"  Current 5 slots at atg500/stop500 gives a 2.9 GB file "
          f"(results_4ct_dn/nmd_orf_data.h5).\n")
    print(f"  {'threshold':<26} {'mean slots':>11} {'vs current':>11} "
          f"{'projected HDF5':>16}")
    print(f"  {'-'*26} {'-'*11} {'-'*11} {'-'*16}")
    for q, (cut, counts) in keep_by_thr.items():
        r = counts.mean() / cur
        print(f"  {'annotated q'+str(q):<26} {counts.mean():>11.1f} "
              f"{r:>10.2f}x {2.9*r:>15.1f} GB")
    print("""
  infer_uorf_attention.py hardcodes attn_0..attn_4 and range(5) at lines 174 and
  207. At any K above 5 it does not fail -- it silently uses the first five and
  produces plausible numbers. That has to be fixed BEFORE the pool changes, not
  after, because nothing about its output would look wrong.""")

    # ------------------------------------------------------------------ 5
    print("\n" + "=" * 100)
    print("5. DOES A THRESHOLD ADMIT THE THING THE FLOOR WAS BLOCKING?")
    print("=" * 100)
    print("""
  The floor's cost was named specifically: ATF4's regulatory uORF1 is 3 codons
  and cannot be a candidate at 33 nt. Below, how many sub-33 nt ORFs survive
  each threshold -- i.e. whether dropping the floor actually buys anything once
  the Kozak cut is applied, or whether the threshold removes what the floor
  removed.""")
    short = lens < 33
    print(f"\n  {'threshold':<26} {'short ORFs kept':>17} {'% of short':>12} "
          f"{'% of kept pool':>16}")
    print(f"  {'-'*26} {'-'*17} {'-'*12} {'-'*16}")
    for q, (cut, counts) in keep_by_thr.items():
        kept = allsc >= cut
        ks = int((kept & short).sum())
        print(f"  {'annotated q'+str(q):<26} {ks:>17,} "
              f"{ks/max(short.sum(),1)*100:>11.1f}% "
              f"{ks/max(kept.sum(),1)*100:>15.1f}%")

    # ------------------------------------------------------------------ 6
    print("\n" + "=" * 100)
    print("6. THE OTHER HALF OF THE DECISION -- what does each pool size BUY?")
    print("=" * 100)
    print("""
  Cost without benefit is not a decision. The ORFs that matter for the uORF
  mechanism are those starting upstream of the main AUG whose OWN stop has a
  junction more than 50 nt downstream -- each one is a PTC in its own right and
  a candidate trigger. The prior window measured that HALF of this population is
  invisible to the model at K = 5. So: what fraction does each threshold admit,
  and how does that trade against slots?""")
    jdf = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                      dtype=str, keep_default_na=False)
    junc = {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(jdf["isoform_id"], jdf["junctions"])}

    tot_rel = 0
    admitted = {q: 0 for q in keep_by_thr}
    admitted_topk = {k: 0 for k in (5, 10, 20, 30, 50)}
    tx_with_rel = 0
    tx_covered = {q: 0 for q in keep_by_thr}
    for iso, a, sc, m in rows:
        if m is None:
            continue
        j = junc.get(iso, np.empty(0, dtype=np.int64))
        if not len(j):
            continue
        up = a[:, 0] < m                       # starts upstream of the main AUG
        if not up.any():
            continue
        # 0-based end -> 1-based last base of stop is a[:,1]; PTC if a junction
        # lies beyond that + 50
        ends = a[up, 1]
        rel = np.array([len(j) - int(np.searchsorted(j, int(e) + 50, side="right")) > 0
                        for e in ends])
        if not rel.any():
            continue
        tot_rel += int(rel.sum())
        tx_with_rel += 1
        s_up = sc[up]
        for q, (cut, _) in keep_by_thr.items():
            keep = rel & (s_up >= cut)
            admitted[q] += int(keep.sum())
            if keep.sum() == rel.sum():
                tx_covered[q] += 1
        order = np.argsort(-np.nan_to_num(sc, nan=-1e9))
        for k in admitted_topk:
            sel_idx = set(order[:k].tolist())
            up_idx = np.flatnonzero(up)
            admitted_topk[k] += int(sum(
                1 for t, gi in enumerate(up_idx) if rel[t] and gi in sel_idx))

    print(f"\n  PTC-bearing upstream ORFs in the full no-floor scan: {tot_rel:,}")
    print(f"  transcripts carrying at least one:                   {tx_with_rel:,}")
    print(f"\n  {'threshold':<20} {'mean slots':>11} {'ORFs admitted':>15} "
          f"{'coverage':>10} {'transcripts fully covered':>26}")
    print(f"  {'-'*20} {'-'*11} {'-'*15} {'-'*10} {'-'*26}")
    for q, (cut, counts) in keep_by_thr.items():
        print(f"  {'annotated q'+str(q):<20} {counts.mean():>11.1f} "
              f"{admitted[q]:>15,} {admitted[q]/max(tot_rel,1)*100:>9.1f}% "
              f"{tx_covered[q]/max(tx_with_rel,1)*100:>25.1f}%")
    print(f"\n  and for comparison, keeping the top-K by Kozak score with no")
    print(f"  threshold at all:")
    print(f"  {'top-K':<20} {'mean slots':>11} {'ORFs admitted':>15} {'coverage':>10}")
    print(f"  {'-'*20} {'-'*11} {'-'*15} {'-'*10}")
    for k, v in admitted_topk.items():
        print(f"  {'K = '+str(k):<20} {float(k):>11.1f} {v:>15,} "
              f"{v/max(tot_rel,1)*100:>9.1f}%")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
