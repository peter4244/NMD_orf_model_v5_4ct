#!/usr/bin/env python
"""
build_tensor.py — encode the candidate pool as the model's input tensor.

Implements section 5 of analysis_plans/RETRAIN_PLAN_2026-08-01.md. Reads the
pool from build_orf_pool.py and writes an HDF5 the trainer consumes.

  1  two 1000-base windows per candidate, each anchored at a fixed array index
  2  nine channels per window position
  3  the structural block, normalised on the training split only
  4  ragged storage — a flat candidate array with per-transcript offsets
  5  splits, with each evaluation side screened against its own paralog list

ALL POSITIONS ARE 1-BASED TRANSCRIPT COORDINATES, as in the pool table.

Run one chromosome first; the full build does not fit on a laptop.

    python build_tensor.py --pool results_pool_v6 --out results_tensor_chr21 --chrom chr21
    python build_tensor.py --pool results_pool_v6 --out results_tensor_v6
"""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
# Defaults are the local layout; --tables and --fasta override them so the same
# script runs on the cluster, where the tables sit beside the repo and the FASTA
# is the transcript-filtered copy rather than the 1.6 GB original.
TABLES = Path(os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30"))
DEPOSIT = Path(os.path.expanduser("~/claude_projects/nmd_deposit_2026/source_data"))
FASTA = DEPOSIT / "sqanti" / "nmd_lungcells_corrected.fasta"

# ---------------------------------------------------------------- geometry
# Each window spans [anchor - LEFT, anchor + RIGHT - 1] and is LEFT + RIGHT wide,
# with the anchor at array index LEFT. A window that cannot be filled to its full
# extent is zero-padded WITHOUT the anchor moving, so the reading-frame channels
# mean the same thing for an ORF of any length.
ATG_LEFT, ATG_RIGHT = 900, 100       # asymmetric: initiation reads what was scanned
STOP_LEFT, STOP_RIGHT = 500, 500
WINDOW = ATG_LEFT + ATG_RIGHT
assert STOP_LEFT + STOP_RIGHT == WINDOW

N_CHANNELS = 9
GC_SPAN = 50                          # channel 5, centred rolling window

STRUCTURAL_COLS = ["n_downstream_ejc", "is_ref_cds", "is_sqanti_cds",
                   "frac_start", "frac_stop"]
INTERPRETABLE_COLS = ["n_downstream_ejc"]      # the rest go to the predictor only

TEST_CHRS = {"chr1", "chr3", "chr5", "chr7"}
VAL_CHRS = {"chr2", "chr4"}

NUC_INDEX = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    NUC_INDEX[ord(_b)] = _i
IS_GC = np.zeros(256, dtype=np.float32)
IS_GC[ord("G")] = IS_GC[ord("C")] = 1.0


def read_sequences(wanted):
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


# ==================================================================== step 2
def encode_window(codes, gc_flag, junc_pos, tx_len, anchor, left, right,
                  orf_start, fill_lo, fill_hi):
    """One 9-channel window.

    `anchor` is the 1-based transcript position that lands at array index `left`.
    Array index k holds transcript position `anchor - left + k`. A position is
    filled only when it lies inside the transcript AND inside [fill_lo, fill_hi];
    everything else stays zero across all nine channels.
    """
    out = np.zeros((N_CHANNELS, left + right), dtype=np.float16)
    lo = max(1, fill_lo)
    hi = min(tx_len, fill_hi)
    if lo > hi:
        return out
    # array indices holding a fillable position
    k0 = max(0, lo - anchor + left)
    k1 = min(left + right, hi - anchor + left + 1)
    if k0 >= k1:
        return out
    pos = np.arange(anchor - left + k0, anchor - left + k1, dtype=np.int64)  # 1-based
    idx0 = pos - 1                                                          # 0-based

    # channels 0-3, one-hot; a base that is not ACGT leaves all four at zero
    nuc = codes[idx0]
    ok = nuc >= 0
    out[nuc[ok], np.arange(k0, k1)[ok]] = 1.0

    # channel 4, exon-exon junction positions
    if len(junc_pos):
        hit = np.isin(pos, junc_pos)
        if hit.any():
            out[4, np.arange(k0, k1)[hit]] = 1.0

    # channel 5, rolling GC over GC_SPAN centred here, computed from THIS window's
    # filled content. Padding counts as absent, not as GC-free: the denominator is
    # the number of filled positions in range, so a window edge is not diluted
    # toward zero.
    g = gc_flag[idx0].astype(np.float32)
    cg = np.concatenate([[0.0], np.cumsum(g)])
    cn = np.arange(len(g) + 1, dtype=np.float32)
    half = GC_SPAN // 2
    a = np.clip(np.arange(len(g)) - half, 0, len(g))
    b = np.clip(np.arange(len(g)) + half + 1, 0, len(g))
    num = cg[b] - cg[a]
    den = cn[b] - cn[a]
    out[5, k0:k1] = np.where(den > 0, num / np.maximum(den, 1), 0.0)

    # channels 6-8, codon position relative to THIS candidate's start codon
    frame = np.mod(pos - orf_start, 3)
    out[6 + frame, np.arange(k0, k1)] = 1.0
    return out


def main():
    global TABLES, FASTA        # must precede any use of the names in this scope
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results_pool_v6")
    ap.add_argument("--out", default="results_tensor_v6")
    ap.add_argument("--chrom", default="",
                    help="build only this chromosome, for a local check")
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--fasta", default=str(FASTA))
    args = ap.parse_args()
    TABLES, FASTA = Path(args.tables), Path(args.fasta)
    pooldir, outdir = REPO / args.pool, REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 92, flush=True)
    print("build_tensor.py — section 5 of RETRAIN_PLAN_2026-08-01.md", flush=True)
    print("=" * 92, flush=True)

    # ------------------------------------------------------------- step 5, splits
    tx = pd.read_csv(TABLES / "tx_summary.tsv", sep="\t",
                     usecols=["isoform_id", "tx_length", "chr", "is_nmd"])
    ref = pd.read_csv(TABLES / "ref_cds_features.tsv", sep="\t",
                      usecols=["isoform_id", "gene_id"]).drop_duplicates("isoform_id")
    tx = tx.merge(ref, on="isoform_id", how="left")

    # Read-through composite loci are two genes transcribed as one unit, so the
    # split, the paralog screen and gene clustering are all ill-defined on them,
    # and the paralog screen cannot see inside one.
    composite = tx["gene_id"].astype(str).str.contains("::", na=False)
    print(f"\nSTEP 5 — splits")
    print(f"  composite read-through loci removed: "
          f"{int(composite.sum()):,} transcripts on "
          f"{tx.loc[composite, 'gene_id'].nunique():,} loci")
    tx = tx[~composite].copy()

    test_par = set(pd.read_csv(TABLES / "paralog_genes.tsv", sep="\t").iloc[:, 0])
    val_par = set(pd.read_csv(TABLES / "val_paralog_genes.tsv", sep="\t").iloc[:, 0])

    def assign(row):
        c, g = row["chr"], row["gene_id"]
        if c in TEST_CHRS:
            return "test_paralog" if g in test_par else "test"
        if c in VAL_CHRS:
            return "val_paralog" if g in val_par else "val"
        return "train"

    tx["split"] = tx.apply(assign, axis=1)
    for s in ["train", "val", "val_paralog", "test", "test_paralog"]:
        print(f"  {s:<14} {int(tx['split'].eq(s).sum()):>7,}")

    if args.chrom:
        tx = tx[tx["chr"].eq(args.chrom)].copy()
        print(f"\n  RESTRICTED to {args.chrom}: {len(tx):,} transcripts")

    # -------------------------------------------------------------- the pool
    pool = pd.read_csv(pooldir / "orf_pool.tsv", sep="\t")
    pool = pool[pool["isoform_id"].isin(set(tx["isoform_id"]))]
    pool = pool.sort_values(["isoform_id", "slot"], kind="stable").reset_index(drop=True)
    tx = tx[tx["isoform_id"].isin(set(pool["isoform_id"]))].copy()
    tx = tx.sort_values("isoform_id", kind="stable").reset_index(drop=True)
    print(f"\n  {len(tx):,} transcripts, {len(pool):,} candidates")

    counts = pool.groupby("isoform_id", sort=True).size()
    counts = counts.reindex(tx["isoform_id"]).fillna(0).astype(np.int64).values
    offsets = np.concatenate([[0], np.cumsum(counts)])[:-1]

    # ------------------------------------------------- step 3, normalisation
    struct = pool[STRUCTURAL_COLS].to_numpy(dtype=np.float32)
    train_tx = set(tx.loc[tx["split"].eq("train"), "isoform_id"])
    train_row = pool["isoform_id"].isin(train_tx).to_numpy()
    print(f"\nSTEP 3 — structural block normalised on the training split only")
    print(f"  {int(train_row.sum()):,} of {len(pool):,} candidates are training rows")
    mean = struct[train_row].mean(axis=0) if train_row.any() else struct.mean(axis=0)
    std = struct[train_row].std(axis=0) if train_row.any() else struct.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    for c, m, s in zip(STRUCTURAL_COLS, mean, std):
        print(f"    {c:<18} mean {m:>10.4f}   sd {s:>10.4f}")
    struct_norm = (struct - mean) / std

    # ------------------------------------------------- steps 1, 2, 4, encode
    seqs = read_sequences(set(tx["isoform_id"]))
    junc = read_junctions()
    print(f"\nSTEPS 1, 2, 4 — encode {len(pool):,} candidates", flush=True)
    n = len(pool)
    est_gb = n * 2 * N_CHANNELS * WINDOW * 2 / 1e9
    print(f"  window {WINDOW} wide, ATG anchored at index {ATG_LEFT}, "
          f"stop at index {STOP_LEFT}")
    print(f"  projected {est_gb:.1f} GB of float16 window data", flush=True)

    h5path = outdir / "nmd_tensor.h5"
    with h5py.File(h5path, "w") as f:
        atg_ds = f.create_dataset("atg_windows", (n, N_CHANNELS, WINDOW),
                                  dtype="float16", chunks=(1, N_CHANNELS, WINDOW))
        stop_ds = f.create_dataset("stop_windows", (n, N_CHANNELS, WINDOW),
                                   dtype="float16", chunks=(1, N_CHANNELS, WINDOW))
        iso_a = pool["isoform_id"].to_numpy()
        st_a = pool["orf_start"].to_numpy(np.int64)
        en_a = pool["orf_end"].to_numpy(np.int64)
        len_by_iso = dict(zip(tx["isoform_id"], tx["tx_length"]))
        codes_cache, gc_cache = {}, {}
        n_atg_clipped = n_stop_clipped = 0

        for i in range(n):
            if i and i % 20000 == 0:
                print(f"  {i:,} / {n:,}   ({time.time()-t0:.0f}s)", flush=True)
            iso = iso_a[i]
            if iso not in codes_cache:
                codes_cache.clear(); gc_cache.clear()
                raw = np.frombuffer(seqs[iso].encode("ascii"), dtype=np.uint8)
                codes_cache[iso] = NUC_INDEX[raw]
                gc_cache[iso] = IS_GC[raw]
            codes, gcf = codes_cache[iso], gc_cache[iso]
            L = int(len_by_iso[iso])
            s, e = int(st_a[i]), int(en_a[i])
            mid = (s + e) // 2                       # mid belongs to the ATG window
            jp = junc.get(iso, np.empty(0, dtype=np.int64))

            atg_ds[i] = encode_window(codes, gcf, jp, L, anchor=s,
                                      left=ATG_LEFT, right=ATG_RIGHT,
                                      orf_start=s, fill_lo=1, fill_hi=mid)
            stop_ds[i] = encode_window(codes, gcf, jp, L, anchor=e - 1,
                                       left=STOP_LEFT, right=STOP_RIGHT,
                                       orf_start=s, fill_lo=mid + 1, fill_hi=L)
            if s + ATG_RIGHT - 1 > mid:
                n_atg_clipped += 1
            if e - 1 - STOP_LEFT < mid:
                n_stop_clipped += 1

        f.create_dataset("structural", data=struct_norm.astype(np.float32))
        f.create_dataset("structural_raw", data=struct.astype(np.float32))
        f.create_dataset("isoform_id", data=np.array(tx["isoform_id"], dtype="S"))
        f.create_dataset("chr", data=np.array(tx["chr"], dtype="S"))
        f.create_dataset("split", data=np.array(tx["split"], dtype="S"))
        f.create_dataset("gene_id", data=np.array(tx["gene_id"].astype(str), dtype="S"))
        f.create_dataset("labels", data=tx["is_nmd"].to_numpy(np.int8))
        f.create_dataset("offset", data=offsets)
        f.create_dataset("count", data=counts)
        g = f.create_group("normalization")
        g.create_dataset("mean", data=mean)
        g.create_dataset("std", data=std)
        g.attrs["columns"] = json.dumps(STRUCTURAL_COLS)
        g.attrs["interpretable_columns"] = json.dumps(INTERPRETABLE_COLS)
        g.attrs["n_train_candidates"] = int(train_row.sum())
        f.attrs["window"] = WINDOW
        f.attrs["atg_left"], f.attrs["atg_right"] = ATG_LEFT, ATG_RIGHT
        f.attrs["stop_left"], f.attrs["stop_right"] = STOP_LEFT, STOP_RIGHT
        f.attrs["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f.attrs["pool"] = str(pooldir)

    print(f"\n  candidates whose ATG window is clipped by the midpoint : "
          f"{n_atg_clipped:,} ({n_atg_clipped/max(n,1)*100:.1f}%)")
    print(f"  candidates whose stop window is clipped by the midpoint: "
          f"{n_stop_clipped:,} ({n_stop_clipped/max(n,1)*100:.1f}%)")
    print(f"\nwrote {h5path}  ({os.path.getsize(h5path)/1e9:.2f} GB)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
