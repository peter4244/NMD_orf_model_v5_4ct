#!/usr/bin/env python
"""
10_export_stop_codon_freq_sf37.py — export per-class stop-codon frequencies
for SF37 (Stop-codon usage in NMD susceptible vs non-NMD isoforms).

Runs on the Explorer cluster where `selected_orfs.tsv` (per-transcript ORF
sequences + stop codon 3-mer) is the source of truth. Local Mac only has a
symlink to that file, so this script exists to be run on the cluster and
have its output committed back to GitHub for local re-rendering.

POPULATION IS AN ARGUMENT, NOT A CONSTANT (2026-08-03). This used to hardcode the
held-out test split by filtering `chr` to 1/3/5/7, with a comment explaining that
the predictions TSV carried no `split` column. It does now — evaluate.py writes a
per-row `split` so a pooled file is self-describing — so the chromosome filter was
a workaround for a limitation that no longer exists, and it silently pinned this
export to one universe. Pass `--split`; `all` is the full cohort (D74/D77).

Priority ORF = ORF rank 0 (the one the model attends to, per the priority
described in SF36 and Methods).

Output: `<results-dir>/stop_codon_freq_by_class_sf37_<split>.tsv` — the split is IN
the filename so two universes cannot overwrite each other, which is how one of them
would otherwise be silently replaced by the other.
Columns:
    population     — the universe: "All", "NMD" or "Control". D78 requires all three
                     be recorded for every analysis, so a value can never be read
                     without the set it was computed over.
    split          — the split argument this run was given, carried into the file
                     so the table is self-describing when separated from the command
    stop_codon     — one of AUG-notation stop codons: UGA / UAA / UAG
    n              — number of transcripts in that population with that stop
    pct            — n / population_total * 100 (rounded to 1 decimal)
    population_total — total transcripts in that population

Second output: `<results-dir>/stop_codon_test_sf37_<split>.tsv` — the NMD-vs-Control
Fisher test per codon, with counts, both percentages, odds ratio and p. It exists
because the manuscript sentence asserts a p-value, and a p-value computed at a prompt
and typed into prose has no producer. ALL THREE codons are tested, so quoting UGA is
visibly a choice rather than the only thing that was measured.

Usage (on Explorer):
    cd /home/p.castaldi/cc/NMD_orf_model_v5_4ct
    python 10_export_stop_codon_freq_sf37.py \
        --config config_dn.yaml --results-dir results_deposit_h5_2026-08-04 \
        --member-seed 42 --split all

    The example named results_4ct_dn until 2026-08-15. That tree is DEPRECATED -- it holds an
    HDF5 built from Channing inputs and is segregated under deprecated_ -- so a reader copying
    the usage line ran the script against the wrong universe while passing every flag correctly.
    The deposit-native tree is results_deposit_h5_2026-08-04. The `cd` path was also the lowercase
    spelling, which resolves only through a symlink made 2026-08-09.

It reads `predictions_{tag}_seed{N}_{split}.tsv`, so evaluate.py must have been run
for that split first — `--split all` additionally requires `--full-cohort` there,
because pooling training and held-out data is legitimate for interpretation and
never for a performance number.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

from paths_config import load_config, selected_tag
# member_tag is THE definition of member naming (utils.py:79). Imported rather than
# reproduced: a second copy of a filename convention is how the published run got five
# 'members' that were one checkpoint written five times.
from utils import member_tag

HERE = Path(__file__).resolve().parent
# NO MODULE-LEVEL RESULTS TREE. This was `RESULTS = HERE / "results_4ct"` -- the PUBLISHED run --
# and SELECTED and OUT_TSV were built from it at import time. main() declares them global and
# rebuilds all three from --results-dir before any read, so the values never reached an output;
# but anything importing this module rather than running it got the published tree, and a reader
# scanning the header saw results_4ct as the default of a deposit-native script. PREDS was already
# None for exactly this reason (2026-08-04); the other two are now joined to it.

SELECTED = None
# Derived, not a literal: this path silently pinned the script to atg500_stop500 even when
# --tag named a different configuration (2026-07-29).
# MODULE-LEVEL DEFAULT REMOVED 2026-08-04. This read config.yaml (Channing) at import
# time regardless of --config, and is superseded by the PREDS built inside main() from
# the caller's config. Kept as None so a stale reference fails loudly instead of
# silently naming a Channing-derived file.
PREDS    = None
OUT_TSV  = None

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
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tag", default=None,
                                        help="Window-config tag. Default: the `selected:` block in --config. "
                                             "Never a hardcoded literal -- see utils.selected_tag.")
    ap.add_argument("--config", required=True,
                       help="Where the selected window configuration is read from")
    ap.add_argument("--split", required=True,
                       help="Which universe to compute over, naming the predictions file to read. "
                            "'all' is the full cohort (D74/D77). Required rather than defaulted: "
                            "the population is the thing this script most easily gets wrong.")
    ap.add_argument("--member-seed", type=int, default=None,
                       help="Ensemble member, by training seed. Names the predictions file; "
                            "omitted = the legacy un-seeded name.")
    args = ap.parse_args()
    # Resolve the tag from the ONE place that names the selected configuration.
    if args.tag is None:
        args.tag = selected_tag(load_config(args.config))
    global SELECTED, PREDS, OUT_TSV
    RES = HERE / args.results_dir
    SELECTED = RES / "selected_orfs.tsv"
    # SAME STEM evaluate.py WRITES (evaluate.py:134, :249), via the one function that defines
    # member naming. Built here rather than restated: this path was `predictions_{tag}.tsv`, which
    # evaluate has not written since members gained seeds, so the script could not find its input.
    PREDS    = RES / f"predictions_{member_tag(args.tag, args.member_seed)}_{args.split}.tsv"
    OUT_TSV  = RES / f"stop_codon_freq_by_class_sf37_{args.split}.tsv"
    print(f"[results-dir] {RES}")
    print(f"[population]  split={args.split}")

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

    # Join with predictions to get the class label. THE PREDICTIONS FILE IS ALREADY THE POPULATION:
    # evaluate.py wrote it for --split, so every row in it belongs to the requested universe and no
    # filtering is needed here. The chromosome filter this replaced was a workaround for a missing
    # `split` column and, worse, re-derived a population the file already carried -- two definitions
    # of one set, which is this project's standing failure.
    print(f"[load] {PREDS}")
    preds = pd.read_csv(PREDS, sep="\t")
    if "split" in preds.columns:
        seen = sorted(preds["split"].astype(str).unique())
        print(f"       splits present = {seen}")
    keep = preds[["isoform_id", "label"]].copy()
    print(f"       prediction rows = {len(keep):,}")

    df = priority.merge(keep, on="isoform_id", how="inner")
    df["stop_rna"] = df[stop_col].apply(normalize_stop)
    df = df.dropna(subset=["stop_rna"])

    print(f"[join] {len(df):,} priority-ORF transcripts in split={args.split} with a valid stop codon")

    # ALL THREE UNIVERSES, EVERY TIME (D78). The pooled row is not decoration: without it a reader
    # comparing this table to another has no way to tell whether a percentage is over everything or
    # over one class, and both are called "the UGA share".
    rows = []
    populations = [("All", df),
                   ("NMD", df[df["label"] == 1]),
                   ("Control", df[df["label"] == 0])]
    for pop_label, sub in populations:
        total = len(sub)
        for stop in ("UGA", "UAA", "UAG"):
            n = int((sub["stop_rna"] == stop).sum())
            rows.append({
                "population":       pop_label,
                "split":            args.split,
                "stop_codon":       stop,
                "n":                n,
                "pct":              round(100.0 * n / total, 1) if total else 0.0,
                "population_total": total,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"[wrote] {OUT_TSV}")
    print(out.to_string(index=False))

    # THE TEST GETS A PRODUCER TOO. The manuscript sentence this feeds is not only two
    # percentages -- it asserts a p-value -- and a number computed at a prompt and typed into prose
    # has no provenance. Every codon is tested, not just the one the sentence happens to quote, so
    # picking UGA afterwards is visibly a choice rather than the only thing measured.
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        sys.exit("[ERROR] scipy is required for the NMD-vs-Control test. Refusing to write a table "
                 "of percentages whose accompanying p-value would then have to be computed by hand.")

    nmd, ctl = df[df["label"] == 1], df[df["label"] == 0]
    trows = []
    for stop in ("UGA", "UAA", "UAG"):
        a = int((nmd["stop_rna"] == stop).sum()); b = len(nmd) - a
        c = int((ctl["stop_rna"] == stop).sum()); d = len(ctl) - c
        odds, p = fisher_exact([[a, b], [c, d]])
        trows.append({
            "split":      args.split,
            "stop_codon": stop,
            "nmd_n": a, "nmd_total": len(nmd),
            "nmd_pct": round(100.0 * a / len(nmd), 1) if len(nmd) else 0.0,
            "control_n": c, "control_total": len(ctl),
            "control_pct": round(100.0 * c / len(ctl), 1) if len(ctl) else 0.0,
            "odds_ratio": round(float(odds), 4),
            "p_value": float(p),
        })
    tout = pd.DataFrame(trows)
    test_path = OUT_TSV.parent / f"stop_codon_test_sf37_{args.split}.tsv"
    tout.to_csv(test_path, sep="\t", index=False)
    print(f"[wrote] {test_path}")
    print(tout.to_string(index=False))


if __name__ == "__main__":
    main()
