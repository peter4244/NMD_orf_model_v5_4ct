#!/usr/bin/env python
"""
10_export_stop_codon_freq_sf37.py — export per-class stop-codon frequencies
for SF37 (Stop-codon usage in NMD susceptible vs non-NMD isoforms).

Runs on the Explorer cluster where `selected_orfs.tsv` (per-transcript ORF
sequences + stop codon 3-mer) is the source of truth. Local Mac only has a
symlink to that file, so this script exists to be run on the cluster and
have its output committed back to GitHub for local re-rendering.

Population: held-out test split (chr 1/3/5/7), excluding `test_paralog`.
Priority ORF = ORF rank 0 (the one the model attends to, per the priority
described in SF36 and Methods).

Output: `results_4ct/stop_codon_freq_by_class_sf37.tsv`
Columns:
    class          — "NMD" or "Control"
    stop_codon     — one of AUG-notation stop codons: UGA / UAA / UAG
    n              — number of transcripts in that class with that stop
    pct            — n / class_total * 100 (rounded to 1 decimal)
    class_total    — total transcripts in the class

Usage (on Explorer):
    cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
    git pull
    python 10_export_stop_codon_freq_sf37.py
    git add results_4ct/stop_codon_freq_by_class_sf37.tsv
    git commit -m "SF37: export stop-codon frequencies for local re-render"
    git push
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_4ct"

SELECTED = RESULTS / "selected_orfs.tsv"
PREDS    = RESULTS / "predictions_atg500_stop500.tsv"
OUT_TSV  = RESULTS / "stop_codon_freq_by_class_sf37.tsv"

# Map DNA → RNA notation for display (T → U in the stop codon).
DNA2RNA = str.maketrans({"T": "U"})


def normalize_stop(seq: str) -> str | None:
    """Return the RNA-notation stop codon (UGA/UAA/UAG) or None if the input
    doesn't look like a valid stop codon."""
    if not isinstance(seq, str):
        return None
    seq = seq.strip().upper()
    if len(seq) != 3:
        return None
    rna = seq.translate(DNA2RNA)
    return rna if rna in {"UGA", "UAA", "UAG"} else None


def main():
    # --results-dir, matching the pattern applied to 03_train.py, evaluate.py,
    # 11_kernel_shap_branches.py and deepshap.py: the deposit-native rebuild writes to
    # results_4ct_dn, and a hardcoded results_4ct silently measures the PUBLISHED run instead.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_4ct")
    ap.add_argument("--tag", default="atg500_stop500")
    args = ap.parse_args()
    global SELECTED, PREDS, OUT_TSV
    RES = HERE / args.results_dir
    SELECTED = RES / "selected_orfs.tsv"
    PREDS    = RES / f"predictions_{args.tag}.tsv"
    OUT_TSV  = RES / "stop_codon_freq_by_class_sf37.tsv"
    print(f"[results-dir] {RES}")

    if not SELECTED.exists():
        sys.exit(f"[ERROR] {SELECTED} not found. Run this on the cluster.")
    if not PREDS.exists():
        sys.exit(f"[ERROR] {PREDS} not found.")

    print(f"[load] {SELECTED}")
    sel = pd.read_csv(SELECTED, sep="\t")
    print(f"       cols = {list(sel.columns)}")

    # Detect which column carries the stop-codon 3-mer.
    candidates_stop = [c for c in sel.columns
                       if c.lower() in {"stop_codon", "stop_seq", "stop"}
                       or c.lower().startswith("stop_codon")]
    if not candidates_stop:
        sys.exit(f"[ERROR] no stop-codon column found. Columns: {list(sel.columns)}")
    stop_col = candidates_stop[0]
    print(f"       stop-codon column = {stop_col!r}")

    # Detect the priority-ORF (rank 0) filter column.
    rank_col = None
    for c in ("orf_rank", "rank", "priority_rank"):
        if c in sel.columns:
            rank_col = c
            break
    if rank_col is None:
        # Fall back — assume selected_orfs already carries one row per priority ORF
        print("[warn] no rank column found; assuming one row per priority ORF.")
        priority = sel.copy()
    else:
        print(f"       rank column = {rank_col!r}; filtering to rank == 0")
        priority = sel[sel[rank_col] == 0].copy()

    # Join with predictions to get the class label. Filter test set by
    # chromosome (chr 1/3/5/7 = train-time held-out split); predictions TSV
    # doesn't carry a `split` column.
    print(f"[load] {PREDS}")
    preds = pd.read_csv(PREDS, sep="\t")
    TEST_CHRS = {"chr1", "chr3", "chr5", "chr7"}
    keep = preds[preds["chr"].isin(TEST_CHRS)][["isoform_id", "label"]].copy()
    print(f"       test-chr rows = {len(keep):,}")

    df = priority.merge(keep, on="isoform_id", how="inner")
    df["stop_rna"] = df[stop_col].apply(normalize_stop)
    df = df.dropna(subset=["stop_rna"])

    print(f"[join] {len(df):,} priority-ORF test transcripts with a valid stop codon")

    rows = []
    for cls_val, cls_label in [(1, "NMD"), (0, "Control")]:
        sub = df[df["label"] == cls_val]
        total = len(sub)
        for stop in ("UGA", "UAA", "UAG"):
            n = int((sub["stop_rna"] == stop).sum())
            rows.append({
                "class":       cls_label,
                "stop_codon":  stop,
                "n":           n,
                "pct":         round(100.0 * n / total, 1) if total else 0.0,
                "class_total": total,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"[wrote] {OUT_TSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
