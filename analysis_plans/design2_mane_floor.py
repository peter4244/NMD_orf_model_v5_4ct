#!/usr/bin/env python
"""
DESIGN 2 -- candidate ORF pool under the MANE-calibrated Kozak threshold.

Supersedes the threshold sweep in design1_orf_pool_size.py, which calibrated on
this dataset's own annotated start codons at percentiles I chose. Pete: use the
previous MANE calibration instead.

THE THRESHOLD, AND WHERE IT COMES FROM
  ~/claude_projects/Isopair/inst/extdata/kozak_mane_calibration.rds, produced by
  Isopair's data-raw/calibrate_mane_kozak.R from 19,226 of 19,276 MANE Select
  transcripts in GENCODE v49, computed 2026-04-19. The 5th-percentile value is
  Isopair's canonical default -- defaultKozakThreshold() returns it, and
  enumerateOrfs() and selectPrimaryOrf() use it.

  Read from the .rds at run time rather than copied, so the dependency is
  explicit and a recalibration propagates.

WHY THE SCALES ARE COMPARABLE
  The MANE calibration scores each annotated CDS start with
  Isopair::scoreKozakPWM under .defaultKozakPWM(). The Python PWM used here was
  validated against that same R function at max |diff| 4.994e-13 over 400 sites
  (kozak_pwm_rescore.py). Same matrix, same offsets, same units.

WHAT IS MEASURED
  1. candidates per transcript at each MANE quantile, with no ORF length floor
  2. coverage of the ORFs the rebuild exists for -- upstream ORFs whose own stop
     has a junction more than 50 nt downstream
  3. whether the annotated start codon itself survives admission. Admission by
     initiation score does not guarantee the main CDS a slot, and 51.5% of
     NMD-positive transcripts are degraded through the main ORF (measured
     elsewhere, build_mechanism_classes_runlog.txt, 41,765 transcripts). This
     quantity has no prior prediction.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python design2_mane_floor.py
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


def read_mane_thresholds():
    """Read the calibration from the Isopair .rds via Rscript."""
    r = subprocess.run(
        ["Rscript", "-e",
         f'x <- readRDS("{CALIB}"); '
         'cat(sprintf("{\\"q01\\":%.6f,\\"q05\\":%.6f,\\"q10\\":%.6f,'
         '\\"q25\\":%.6f,\\"q50\\":%.6f,\\"n_scored\\":%d,\\"n_input\\":%d,'
         '\\"gencode\\":\\"%s\\",\\"computed_at\\":\\"%s\\"}", '
         'x$threshold_q01, x$threshold_q05, x$threshold_q10, x$threshold_q25, '
         'x$threshold_q50, x$n_mane_scored, x$n_mane_input, x$gencode_version, '
         'x$computed_at))'],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"could not read {CALIB}\n{r.stderr}")
    return json.loads(r.stdout.strip())


def load_seqs():
    z = np.load(STORE, allow_pickle=False)
    return z["ids"], z["blob"], z["offsets"]


def scan_transcript(s):
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


def kozak_scores(codes, starts):
    if not len(starts):
        return np.empty(0)
    pos = starts[:, None] + OFFSETS[None, :]
    ok = (pos >= 0) & (pos < len(codes))
    b = codes[np.clip(pos, 0, len(codes) - 1)]
    valid = ok & (b >= 0)
    contrib = np.where(valid, PWM[np.clip(b, 0, 3), np.arange(8)[None, :]], 0.0)
    return np.where(valid.sum(axis=1) > 0, contrib.sum(axis=1), np.nan)


def main():
    thr = read_mane_thresholds()
    print("=" * 100)
    print("DESIGN 2 -- pool under the MANE-calibrated Kozak threshold")
    print("=" * 100)
    print(f"\n  calibration: {CALIB}")
    print(f"    {thr['n_scored']:,} of {thr['n_input']:,} MANE Select transcripts, "
          f"{thr['gencode']}, computed {thr['computed_at']}")
    print(f"    q01 {thr['q01']:.4f}   q05 {thr['q05']:.4f}  <- Isopair default   "
          f"q10 {thr['q10']:.4f}   q25 {thr['q25']:.4f}   q50 {thr['q50']:.4f}")

    ids, blob, off = load_seqs()
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "ref_utr5_length", "ref_atg_available"]
                      ).drop_duplicates("isoform_id")
    ref = ref[ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()]
    main_atg = dict(zip(ref["isoform_id"], ref["ref_utr5_length"].astype(int)))
    jdf = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                      dtype=str, keep_default_na=False)
    junc = {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(jdf["isoform_id"], jdf["junctions"])}

    print(f"\n  enumerating ORFs with no length floor over {len(ids):,} "
          f"transcripts ...", flush=True)
    rows = []
    for k, iso in enumerate(ids):
        if k % 10000 == 0 and k:
            print(f"    {k:,} ...", flush=True)
        s = blob[int(off[k]):int(off[k + 1])].tobytes().decode("ascii")
        orfs = scan_transcript(s)
        if not orfs:
            continue
        a = np.array(orfs, dtype=np.int64)
        codes = NTI[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        rows.append((iso, a, kozak_scores(codes, a[:, 0]), main_atg.get(iso)))
    print(f"  {len(rows):,} transcripts with at least one ORF")

    QS = ["q01", "q05", "q10", "q25", "q50"]
    n_all = np.array([len(r[1]) for r in rows])

    # relevant ORFs and main-ORF identity, computed once
    rel_tot, main_tot = 0, 0
    rel_keep = {q: 0 for q in QS}
    main_keep = {q: 0 for q in QS}
    counts = {q: [] for q in QS}
    for iso, a, sc, m in rows:
        for q in QS:
            counts[q].append(int(np.nansum(sc >= thr[q])))
        if m is None:
            continue
        hit = np.flatnonzero(a[:, 0] == m)
        if len(hit):
            main_tot += 1
            for q in QS:
                if sc[hit[0]] >= thr[q]:
                    main_keep[q] += 1
        j = junc.get(iso, np.empty(0, dtype=np.int64))
        if not len(j):
            continue
        up = a[:, 0] < m
        if not up.any():
            continue
        ends = a[up, 1]
        relm = np.array([len(j) - int(np.searchsorted(j, int(e) + 50, side="right")) > 0
                         for e in ends])
        if not relm.any():
            continue
        rel_tot += int(relm.sum())
        s_up = sc[up]
        for q in QS:
            rel_keep[q] += int((relm & (s_up >= thr[q])).sum())

    print("\n" + "=" * 100)
    print("POOL SIZE, COVERAGE, AND WHETHER THE MAIN ORF SURVIVES")
    print("=" * 100)
    print(f"\n  no floor at all: mean {n_all.mean():.1f} candidates per transcript, "
          f"{n_all.sum():,} total, max {n_all.max():,}")
    print(f"  relevant ORFs (upstream, own stop has a junction >50 nt "
          f"downstream): {rel_tot:,}")
    print(f"  transcripts whose annotated start codon is enumerable: {main_tot:,}")
    print(f"\n  {'MANE quantile':<16} {'cut':>8} {'mean':>7} {'median':>7} {'p99':>7} "
          f"{'max':>7} {'total':>12} {'coverage':>10} {'main ORF kept':>15} "
          f"{'HDF5':>9}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*12} {'-'*10} "
          f"{'-'*15} {'-'*9}")
    for q in QS:
        c = np.array(counts[q])
        tag = "  <- default" if q == "q05" else ""
        print(f"  {q:<16} {thr[q]:>8.4f} {c.mean():>7.1f} {np.median(c):>7.0f} "
              f"{np.percentile(c,99):>7.0f} {c.max():>7,} {c.sum():>12,} "
              f"{rel_keep[q]/max(rel_tot,1)*100:>9.1f}% "
              f"{main_keep[q]/max(main_tot,1)*100:>14.1f}% "
              f"{2.9*c.mean()/5:>8.1f}G{tag}")
    print(f"\n  current model: 5 slots, {5*len(rows):,} total, 2.9 GB")
    print(f"  HDF5 column scales the current 2.9 GB file linearly in slots.")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
