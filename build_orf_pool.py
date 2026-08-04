#!/usr/bin/env python
"""
build_orf_pool.py — construct the candidate ORF pool.

Implements section 3 of analysis_plans/RETRAIN_PLAN_2026-08-01.md. Replaces
`select_priority_orfs` in data_prep.py, which kept 5 candidates per transcript
chosen by annotation priority.

  1  enumerate every ATG with an in-frame stop, no minimum ORF length
  2  score initiation context with the Cavener-Ray PWM
  3  read the admission floor from Isopair's MANE calibration
  4  admit at or above the floor AND starting in the first half of the
     transcript; always admit the reference start codon; fall back to the five
     highest-scoring candidates where that leaves the pool empty
  5  attach n_downstream_ejc, the count of junctions more than 50 bases past
     the candidate's stop
  6  emit one row per admitted candidate, ordered 5'->3'

ALL POSITIONS ARE 1-BASED TRANSCRIPT COORDINATES. Position 1 is the first base
of the transcript and a range [a, b] includes both ends. `orf_start` is the A of
the ATG, `orf_end` the last base of the stop codon, so
`orf_length = orf_end - orf_start + 1`.

Every quantity the plan predicts in section 3.4 is printed, so the run log is
the evidence rather than a summary of it.

Usage:
    python build_orf_pool.py --out results_pool_v6
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent

# ---------------------------------------------------------------- inputs
TABLES = Path(os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30"))
DEPOSIT = Path(os.path.expanduser("~/claude_projects/nmd_deposit_2026/source_data"))
FASTA = DEPOSIT / "sqanti" / "nmd_lungcells_corrected.fasta"
SQANTI_CLASS = DEPOSIT / "sqanti" / "nmd_lungcells_classification.txt"
CALIBRATION = Path(os.path.expanduser(
    "~/claude_projects/Isopair/inst/extdata/kozak_mane_calibration.rds"))

# ------------------------------------------------- the initiation-context PWM
# Cavener & Ray 1991 observed base frequencies at vertebrate translation
# initiation sites, as carried by Isopair::.defaultKozakPWM(). Columns are
# LABELLED positions with the A of the ATG numbered +1; rows are A, C, G, T.
COL_LABELS = np.array([-6, -5, -4, -3, -2, -1, 4, 5])
CAVENER_RAY = np.array([
    [0.23, 0.25, 0.25, 0.37, 0.27, 0.19, 0.23, 0.17],   # A
    [0.35, 0.22, 0.32, 0.19, 0.33, 0.39, 0.16, 0.31],   # C
    [0.23, 0.33, 0.25, 0.34, 0.17, 0.23, 0.46, 0.34],   # G
    [0.19, 0.20, 0.18, 0.10, 0.23, 0.19, 0.15, 0.18],   # T
])
PWM = np.log2(CAVENER_RAY / 0.25)

# A label is not a displacement: the A of the ATG is labelled +1 and sits at
# displacement 0, so a positive label loses one and a negative label does not.
# Collapsing the two is how an off-by-one enters a score that then decides
# admission.
DISPLACEMENT = np.where(COL_LABELS < 0, COL_LABELS, COL_LABELS - 1)

NUC_INDEX = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    NUC_INDEX[ord(_b)] = _i

STOP_CODONS = ("TAA", "TAG", "TGA")
EJC_RULE_NT = 50            # a junction must lie more than this far past the stop


# ============================================================ step 3, the floor
def read_admission_floor(path=CALIBRATION, quantile="threshold_q05"):
    """Read the MANE-calibrated Kozak threshold from Isopair's calibration.

    Read at run time rather than copied so a recalibration propagates. The scale
    matches the PWM above: the calibration scores with Isopair::scoreKozakPWM
    under .defaultKozakPWM(), which is the matrix this file carries.
    """
    r = subprocess.run(
        ["Rscript", "-e",
         f'x <- readRDS("{path}"); '
         # %.17g round-trips a double exactly. At %.10f a candidate within 1e-10
         # of the floor lands on either side depending on the formatting, and one
         # does.
         f'cat(sprintf("%.17g|%d|%d|%s|%s", x${quantile}, x$n_mane_scored, '
         'x$n_mane_input, x$gencode_version, x$computed_at))'],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"could not read {path}\n{r.stderr}")
    v, n_scored, n_input, gencode, when = r.stdout.strip().split("|")
    return float(v), dict(n_scored=int(n_scored), n_input=int(n_input),
                          gencode=gencode, computed_at=when, quantile=quantile,
                          path=str(path))


# ============================================================ step 1, enumerate
def enumerate_orfs(seq):
    """Every ATG with an in-frame stop codon downstream.

    Returns an (n, 2) int64 array of 1-based (start, end): start is the A of the
    ATG, end the last base of the stop codon. No minimum length, so an ATG
    immediately followed by a stop is a candidate. Two ATGs in the same frame
    sharing one stop are separate candidates; ORFs in different frames are
    separate candidates.
    """
    n = len(seq)
    out = []
    i = seq.find("ATG")                       # 0-based while scanning the string
    while i != -1:
        j = i + 3
        while j + 3 <= n:
            if seq[j:j + 3] in STOP_CODONS:
                out.append((i + 1, j + 3))    # -> 1-based (A of ATG, last base of stop)
                break
            j += 3
        i = seq.find("ATG", i + 1)
    return np.array(out, dtype=np.int64) if out else np.empty((0, 2), dtype=np.int64)


# ================================================================ step 2, score
def score_initiation(codes, starts_1based):
    """Cavener-Ray PWM score at each start codon.

    `codes` is the transcript as base indices (A=0..T=3, -1 for anything else).
    Positions running off either end of the transcript are skipped and the
    remainder summed; a start codon with no scorable position scores NaN and is
    not admitted.
    """
    if not len(starts_1based):
        return np.empty(0, dtype=np.float64)
    pos0 = (starts_1based[:, None] - 1) + DISPLACEMENT[None, :]   # 0-based lookup
    in_range = (pos0 >= 0) & (pos0 < len(codes))
    base = codes[np.clip(pos0, 0, len(codes) - 1)]
    usable = in_range & (base >= 0)
    contrib = np.where(usable, PWM[np.clip(base, 0, 3), np.arange(8)[None, :]], 0.0)
    n_usable = usable.sum(axis=1)
    return np.where(n_usable > 0, contrib.sum(axis=1), np.nan)


# ================================================================== step 5, EJC
def count_downstream_ejc(junctions, orf_end):
    """Junctions lying more than EJC_RULE_NT bases past the last base of the stop."""
    return len(junctions) - int(np.searchsorted(junctions, orf_end + EJC_RULE_NT,
                                                side="right"))


# ================================================================= input readers
def read_sequences(wanted):
    """Stream the FASTA, keeping only the transcripts in `wanted`."""
    seqs, name, buf = {}, None, []
    with open(FASTA) as fh:
        for line in fh:
            if line[0] == ">":
                if name is not None and name in wanted:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name is not None and name in wanted:
        seqs[name] = "".join(buf)
    return seqs


def read_junctions():
    df = pd.read_csv(TABLES / "junctions.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    return {i: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for i, j in zip(df["isoform_id"], df["junctions"])}


def read_sqanti_cds():
    """1-based transcript position of the first base of the SQANTI CDS call.

    The C parser, not engine="python" as data_prep.py uses — on this 367 MB file
    that is 10 seconds against several minutes, and it parses the three columns
    needed here without complaint.
    """
    df = pd.read_csv(SQANTI_CLASS, sep="\t", dtype={"isoform": str},
                     usecols=["isoform", "coding", "CDS_start"], low_memory=False)
    df = df[df["coding"].eq("coding") & df["CDS_start"].notna()]
    df = df.drop_duplicates("isoform")
    return dict(zip(df["isoform"], df["CDS_start"].astype(np.int64)))


# ====================================================================== the build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_pool_v6")
    ap.add_argument("--limit", type=int, default=0,
                    help="build only the first N transcripts, for a smoke test")
    args = ap.parse_args()
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 92, flush=True)
    print("build_orf_pool.py — section 3 of RETRAIN_PLAN_2026-08-01.md")
    print("=" * 92)

    floor, calib = read_admission_floor()
    print(f"\nSTEP 3 — admission floor")
    print(f"  {calib['path']}")
    print(f"  {calib['quantile']} = {floor:.4f}")
    print(f"  {calib['n_scored']:,} of {calib['n_input']:,} MANE Select transcripts, "
          f"{calib['gencode']}, computed {calib['computed_at']}")

    print(f"\nINPUTS", flush=True)
    tx = pd.read_csv(TABLES / "tx_summary.tsv", sep="\t",
                     usecols=["isoform_id", "tx_length", "chr", "is_nmd"])
    if args.limit:
        tx = tx.head(args.limit)
    print(f"  TX      {len(tx):,} transcripts")

    ref = pd.read_csv(TABLES / "ref_cds_features.tsv", sep="\t",
                      usecols=["isoform_id", "gene_id", "ref_utr5_length",
                               "ref_atg_available"]).drop_duplicates("isoform_id")
    n_ref_rows = len(ref)
    ref = ref[ref["isoform_id"].isin(set(tx["isoform_id"]))]
    print(f"  REFCDS  {n_ref_rows:,} rows, {len(ref):,} after the join to TX "
          f"({n_ref_rows - len(ref):,} unlabelled and dropped)")
    has_ref = ref["ref_atg_available"].eq(1) & ref["ref_utr5_length"].notna()
    # 1-based position of the A of the reference start codon
    ref_atg = dict(zip(ref.loc[has_ref, "isoform_id"],
                       ref.loc[has_ref, "ref_utr5_length"].astype(np.int64) + 1))
    gene_of = dict(zip(ref["isoform_id"], ref["gene_id"]))
    print(f"          {len(ref_atg):,} with a projected reference start codon")

    junc = read_junctions()
    print(f"  JUNC    {len(junc):,} transcripts")
    sq_cds = read_sqanti_cds()
    print(f"  SQCDS   {len(sq_cds):,} transcripts SQANTI calls coding")
    print(f"  SEQ     reading {FASTA.name} ...", flush=True)
    seqs = read_sequences(set(tx["isoform_id"]))
    print(f"          {len(seqs):,} sequences kept")
    bad_len = [i for i, L in zip(tx["isoform_id"], tx["tx_length"])
               if i in seqs and len(seqs[i]) != L]
    print(f"          length disagreements with TX.tx_length: {len(bad_len):,}")

    # ------------------------------------------------------------- per transcript
    print(f"\nSTEPS 1, 2, 4, 5 — enumerate, score, admit, attach", flush=True)
    rows = []
    rec = []
    n_done = 0
    for iso, L in zip(tx["isoform_id"], tx["tx_length"]):
        seq = seqs.get(iso)
        if seq is None:
            rec.append((iso, 0, 0, 0, False, False, 0, 0))
            continue
        n_done += 1
        if n_done % 5000 == 0:
            print(f"  {n_done:,} / {len(tx):,}   ({time.time()-t0:.0f}s)", flush=True)

        orfs = enumerate_orfs(seq)
        if not len(orfs):
            rec.append((iso, 0, 0, 0, False, False, 0, 0))
            continue
        codes = NUC_INDEX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        score = score_initiation(codes, orfs[:, 0])

        above = np.nan_to_num(score, nan=-np.inf) >= floor
        first_half = orfs[:, 0] <= L // 2
        keep = above & first_half
        # A Python list, not np.where(...): a numpy string array takes its width
        # from the first literal it sees, so "reference" and "fallback" would be
        # silently truncated to five characters and every later comparison
        # against the full word would be False.
        admitted_by = ["floor" if k else "" for k in keep]

        cds_pos = ref_atg.get(iso)
        cds_idx = -1
        cds_rescued = False
        if cds_pos is not None:
            hit = np.flatnonzero(orfs[:, 0] == cds_pos)
            if len(hit):
                cds_idx = int(hit[0])
                if not keep[cds_idx]:
                    admitted_by[cds_idx] = "reference"
                    cds_rescued = True
                keep[cds_idx] = True

        fallback = False
        if not keep.any():
            fallback = True
            order = np.argsort(-np.nan_to_num(score, nan=-np.inf))
            keep = np.zeros(len(orfs), dtype=bool)
            keep[order[:5]] = True
            admitted_by = ["fallback" if k else "" for k in keep]

        # Classify every ORF, admitted or not, so section 3.4's coverage
        # denominators are produced here rather than assumed. "Upstream" is
        # strictly 5' of the reference start codon and is defined only where the
        # transcript has one; "triggering" is the ORF's own stop carrying a
        # junction more than EJC_RULE_NT bases past it.
        n_up_trig = n_up_trig_adm = 0
        j = junc.get(iso, np.empty(0, dtype=np.int64))
        if cds_pos is not None and len(j):
            upstream = orfs[:, 0] < cds_pos
            if upstream.any():
                ends = orfs[upstream, 1]
                trig = (len(j) - np.searchsorted(j, ends + EJC_RULE_NT,
                                                 side="right")) > 0
                n_up_trig = int(trig.sum())
                n_up_trig_adm = int((trig & keep[upstream]).sum())

        rec.append((iso, int(len(orfs)), int(above.sum()),
                    int((above & ~first_half).sum()), cds_rescued, fallback,
                    n_up_trig, n_up_trig_adm))

        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(orfs[idx, 0], kind="stable")]        # step 4, 5'->3'
        sq_pos = sq_cds.get(iso, -1)
        for slot, k in enumerate(idx):
            s, e = int(orfs[k, 0]), int(orfs[k, 1])
            rows.append((iso, slot, s, e, e - s + 1,
                         count_downstream_ejc(j, e),
                         float(score[k]) if not np.isnan(score[k]) else np.nan,
                         int(cds_pos is not None and s == cds_pos),
                         int(s == sq_pos),
                         s / L, e / L,
                         admitted_by[k] or "floor"))

    pool = pd.DataFrame(rows, columns=[
        "isoform_id", "slot", "orf_start", "orf_end", "orf_length",
        "n_downstream_ejc", "kozak_score", "is_ref_cds", "is_sqanti_cds",
        "frac_start", "frac_stop", "admitted_by"])
    record = pd.DataFrame(rec, columns=[
        "isoform_id", "n_enumerated", "n_above_floor", "n_discounted_by_position",
        "reference_rescued", "fallback_fired",
        "n_upstream_triggering", "n_upstream_triggering_admitted"])

    # ------------------------------------------------------------------- outputs
    pool.to_csv(outdir / "orf_pool.tsv", sep="\t", index=False,
                float_format="%.6f")
    record.to_csv(outdir / "orf_pool_record.tsv", sep="\t", index=False)
    meta = dict(built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                floor=floor, calibration=calib, ejc_rule_nt=EJC_RULE_NT,
                n_transcripts=int(len(tx)), n_candidates=int(len(pool)),
                script=str(Path(__file__).name))
    (outdir / "orf_pool_provenance.json").write_text(json.dumps(meta, indent=2))

    # ------------------------------------------- section 3.4, every predicted value
    per_tx = record.set_index("isoform_id")
    n_adm = pool.groupby("isoform_id").size().reindex(record["isoform_id"]).fillna(0)
    print("\n" + "=" * 92)
    print("SECTION 3.4 — quantities this step reports")
    print("=" * 92)
    print(f"\n  {'quantity':<52} {'measured':>14} {'predicted':>14}")
    print(f"  {'-'*52} {'-'*14} {'-'*14}")
    def line(name, got, want):
        print(f"  {name:<52} {got:>14} {want:>14}")
    line("candidates per transcript, no floor (mean)",
         f"{record['n_enumerated'].mean():.1f}", "54.6")
    line("...above FLOOR (mean)", f"{record['n_above_floor'].mean():.1f}", "36.4")
    line("...after all four rules (mean)", f"{n_adm.mean():.1f}", "19.1")
    line("...median", f"{n_adm.median():.0f}", "17")
    line("...p90", f"{n_adm.quantile(0.90):.0f}", "35")
    line("...p99", f"{n_adm.quantile(0.99):.0f}", "59")
    line("...max", f"{n_adm.max():,.0f}", "565")
    line("candidates total", f"{len(pool):,}", "802,430")
    line("transcripts with an empty pool", f"{int((n_adm == 0).sum()):,}", "0")
    line("candidates discounted by position",
         f"{record['n_discounted_by_position'].sum():,}", "731,937")
    line("transcripts where the fallback fires",
         f"{int(record['fallback_fired'].sum()):,}", "58")
    n_cds = int(pool["is_ref_cds"].sum())
    line("reference start codon in the pool", f"{n_cds:,}", "28,775")
    line("...reaching it only by the always-admit rule",
         f"{int(record['reference_rescued'].sum()):,}", "3,823")
    line("candidates admitted below FLOOR",
         f"{int((pool['kozak_score'] < floor).sum()):,}", "2,493")
    tot_trig = int(record["n_upstream_triggering"].sum())
    adm_trig = int(record["n_upstream_triggering_admitted"].sum())
    carrying = record["n_upstream_triggering"] > 0
    full = (record.loc[carrying, "n_upstream_triggering_admitted"]
            == record.loc[carrying, "n_upstream_triggering"])
    line("triggering upstream ORFs, total", f"{tot_trig:,}", "133,765")
    line("...admitted", f"{adm_trig:,} ({adm_trig/max(tot_trig,1)*100:.1f}%)",
         "89,073 (66.6%)")
    line("transcripts carrying at least one", f"{int(carrying.sum()):,}", "17,944")
    line("...with ALL of theirs admitted",
         f"{int(full.sum()):,} ({full.mean()*100:.1f}%)", "not yet measured")

    print(f"\n  admitted_by: {pool['admitted_by'].value_counts().to_dict()}")
    print(f"\nwrote {outdir/'orf_pool.tsv'}  ({len(pool):,} rows)")
    print(f"wrote {outdir/'orf_pool_record.tsv'}  ({len(record):,} rows)")
    print(f"wrote {outdir/'orf_pool_provenance.json'}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
