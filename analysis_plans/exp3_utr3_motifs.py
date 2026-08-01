#!/usr/bin/env python
"""
EXPERIMENT 3 -- are there sequence motifs after the stop codon that change the
outcome?

THREE MOTIFS, THREE DIFFERENT PREDICTED DIRECTIONS
  AATAAA      polyadenylation signal   -> LESS NMD (PABP close to the stop
                                          suppresses decay)
  TTATTTATT   AU-rich element          -> MORE NMD
  CU-rich run PTBP1 site               -> LESS NMD (protective)

The three directions ARE the test. Base composition would push all three the
same way; only a mechanism can push two down and one up.

THE CONTROL THAT DOES THE WORK
  Each motif is compared not against its absence but against its own ANAGRAMS
  -- strings with exactly the same base composition and length, differing only
  in order. AATAAA has five other arrangements of {A,A,A,A,A,T}; none is a
  polyadenylation signal. If AATAAA moves NMD and its anagrams do not, the
  effect is the ORDER of the bases, which is what "motif" means. If the
  anagrams move it too, the effect is composition and there is no motif.

  This replaces the dinucleotide-shuffle control, which died on this project
  because a shuffled 3-mer is the identity map (BRIEF section 6). At 6 and 9
  bases there are real alternatives, so the control is available again.

WINDOW
  A FIXED 200 nt window starting at the first base of the 3'UTR, on the
  isoforms that have at least that much 3'UTR. Fixed length matters: motif
  presence rises with scan length, so an unequal window would make this a
  3'UTR-length contrast wearing a motif's name.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python exp3_utr3_motifs.py
"""

import itertools
import os

import numpy as np
import pandas as pd

TABLES = os.path.expanduser("~/claude_projects/nmd_w69_tables_2026-07-30")
DN = os.path.expanduser("~/claude_projects/NMD_orf_model_v5_4ct/results_4ct_dn")
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "seq_store.npz")
RNG = np.random.default_rng(20260801)

WINDOW = 200
PAS = "AATAAA"
ARE = "TTATTTATT"


def load_seqs():
    z = np.load(STORE, allow_pickle=False)
    ids, blob, off = z["ids"], z["blob"], z["offsets"]
    return blob, off, {s: i for i, s in enumerate(ids)}


def sub(blob, off, i, a, b):
    lo, hi = int(off[i]), int(off[i + 1])
    a0, b0 = lo + a - 1, lo + b - 1
    if a0 < lo or b0 > hi or b0 <= a0:
        return ""
    return blob[a0:b0].tobytes().decode("ascii")


def load_junctions():
    df = pd.read_csv(os.path.join(TABLES, "junctions.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    return {iso: (np.sort(np.fromstring(j, sep=",", dtype=np.int64))
                  if j not in ("", "NA") else np.empty(0, dtype=np.int64))
            for iso, j in zip(df["isoform_id"], df["junctions"])}


def anagrams(motif, k=None, rng=RNG):
    """Distinct rearrangements of the motif's own bases, excluding the motif."""
    uniq = sorted(set("".join(p) for p in itertools.permutations(motif)))
    uniq = [u for u in uniq if u != motif]
    if k is not None and len(uniq) > k:
        uniq = list(rng.choice(uniq, size=k, replace=False))
    return uniq


def standardised(d, flag, strata, label="is_nmd", min_cell=20):
    d = d.dropna(subset=strata + [label])
    key = d[strata].astype(str).agg("|".join, axis=1)
    d = d.assign(_k=key)
    w = d["_k"].value_counts(normalize=True)
    out = {}
    for v, gg in d.groupby(flag):
        num = den = 0.0
        for k, h in gg.groupby("_k"):
            if len(h) < min_cell:
                continue
            num += w[k] * h[label].mean()
            den += w[k]
        out[bool(v)] = (len(gg), gg[label].mean() * 100,
                        num / den * 100 if den > 0 else np.nan)
    return out


def effect(d, win_col, motif, strata):
    """Standardised NMD+ difference, present minus absent, for one string."""
    f = d[win_col].str.contains(motif, regex=False)
    r = standardised(d.assign(_f=f), "_f", strata)
    if True not in r or False not in r:
        return np.nan, int(f.sum())
    return r[True][2] - r[False][2], int(f.sum())


def main():
    print("=" * 96)
    print(f"EXPERIMENT 3 -- 3'UTR motifs in a fixed {WINDOW} nt window after the stop")
    print("=" * 96)

    blob, off, idx = load_seqs()
    junc = load_junctions()
    tx = pd.read_csv(os.path.join(TABLES, "tx_summary.tsv"), sep="\t")[
        ["isoform_id", "is_nmd", "tx_length"]]
    ref = pd.read_csv(os.path.join(TABLES, "ref_cds_features.tsv"), sep="\t",
                      usecols=["isoform_id", "gene_id"])
    sel = pd.read_csv(os.path.join(DN, "selected_orfs.tsv"), sep="\t",
                      usecols=["isoform_id", "orf_end", "is_ref_cds", "stop_codon"])

    d = sel[sel["is_ref_cds"].astype(bool)].drop_duplicates("isoform_id").copy()
    d = d.merge(tx, on="isoform_id").merge(ref.drop_duplicates("isoform_id"),
                                           on="isoform_id", how="left")
    d["_i"] = d["isoform_id"].map(idx)
    d = d[d["_i"].notna()].copy()
    d["_i"] = d["_i"].astype(int)
    d["utr3"] = d["tx_length"] - d["orf_end"]
    d = d[d["utr3"] >= WINDOW].copy()
    print(f"\n  isoforms with a reference-CDS slot and >= {WINDOW} nt of 3'UTR: "
          f"{len(d):,}")

    d["win"] = [sub(blob, off, i, e + 1, e + 1 + WINDOW)
                for i, e in zip(d["_i"], d["orf_end"])]
    bad = d["win"].str.len().ne(WINDOW)
    print(f"  windows of the wrong length (dropped): {int(bad.sum()):,}")
    d = d[~bad].copy()

    d["ptc"] = [len(junc.get(i, np.empty(0, dtype=np.int64)))
                - int(np.searchsorted(junc.get(i, np.empty(0, dtype=np.int64)),
                                      int(e) + 50, side="right")) > 0
                for i, e in zip(d["isoform_id"], d["orf_end"])]
    d["ptc"] = d["ptc"].astype(int)
    d["gc"] = d["win"].str.count("[GC]") / WINDOW
    d["at"] = 1 - d["gc"]
    d["utr3_q"] = pd.qcut(d["utr3"], 4, labels=False, duplicates="drop").astype(int)
    d["gc_q"] = pd.qcut(d["gc"], 5, labels=False, duplicates="drop").astype(int)
    STRATA = ["ptc", "utr3_q", "gc_q"]
    print(f"  PTC+ {d['ptc'].mean()*100:.1f}%   NMD+ {d['is_nmd'].mean()*100:.1f}%   "
          f"median window GC {d['gc'].median()*100:.1f}%")
    print(f"  strata: {STRATA} "
          f"({d[STRATA].astype(str).agg('|'.join, axis=1).nunique()} cells)")

    # -------------------------------------------------------------- PAS
    print("\n" + "=" * 96)
    print(f"A. {PAS} (polyadenylation signal) AGAINST ITS OWN ANAGRAMS")
    print("   predicted direction: NEGATIVE (less NMD)")
    print("=" * 96)
    ctrls = anagrams(PAS)
    print(f"\n  {len(ctrls)} composition-matched controls: {', '.join(ctrls)}")
    print(f"\n  {'string':<12} {'n present':>10} {'crude +':>9} {'crude -':>9} "
          f"{'standardised diff':>19}")
    print(f"  {'-'*12} {'-'*10} {'-'*9} {'-'*9} {'-'*19}")
    rows = []
    for m in [PAS] + ctrls:
        f = d["win"].str.contains(m, regex=False)
        r = standardised(d.assign(_f=f), "_f", STRATA)
        diff = (r[True][2] - r[False][2]) if (True in r and False in r) else np.nan
        cp = r[True][1] if True in r else np.nan
        cm = r[False][1] if False in r else np.nan
        tag = "  <-- PAS" if m == PAS else ""
        print(f"  {m:<12} {int(f.sum()):>10,} {cp:>8.1f}% {cm:>8.1f}% "
              f"{diff:>+18.2f}pp{tag}")
        rows.append((m, diff))
    ctrl_vals = np.array([v for m, v in rows if m != PAS and not np.isnan(v)])
    pas_val = [v for m, v in rows if m == PAS][0]
    print(f"\n  control anagrams: mean {ctrl_vals.mean():+.2f}pp, "
          f"sd {ctrl_vals.std(ddof=1):+.2f}, range "
          f"[{ctrl_vals.min():+.2f}, {ctrl_vals.max():+.2f}]")
    z = (pas_val - ctrl_vals.mean()) / ctrl_vals.std(ddof=1)
    print(f"  {PAS} sits {z:+.2f} sd from its own composition controls")

    # -------------------------------------------------------------- ARE
    print("\n" + "=" * 96)
    print(f"B. {ARE} (AU-rich element) AGAINST 20 SAMPLED ANAGRAMS")
    print("   predicted direction: POSITIVE (more NMD)")
    print("=" * 96)
    ctrls = anagrams(ARE, k=20)
    print(f"\n  {'string':<12} {'n present':>10} {'standardised diff':>19}")
    print(f"  {'-'*12} {'-'*10} {'-'*19}")
    v, n = effect(d, "win", ARE, STRATA)
    print(f"  {ARE:<12} {n:>10,} {v:>+18.2f}pp  <-- ARE")
    cv = []
    for m in ctrls:
        vv, nn = effect(d, "win", m, STRATA)
        if nn >= 50 and not np.isnan(vv):
            cv.append(vv)
            print(f"  {m:<12} {nn:>10,} {vv:>+18.2f}pp")
    cv = np.array(cv)
    if len(cv) > 1:
        print(f"\n  control anagrams: mean {cv.mean():+.2f}pp, sd "
              f"{cv.std(ddof=1):.2f}, range [{cv.min():+.2f}, {cv.max():+.2f}]")
        print(f"  {ARE} sits {(v - cv.mean()) / cv.std(ddof=1):+.2f} sd from them")

    # ---------------------------------------------------------- CU-rich
    print("\n" + "=" * 96)
    print("C. CU-RICH RUN (PTBP1 site) -- predicted direction: NEGATIVE")
    print("=" * 96)
    print("\n  NOTE: unlike A and B this one has NO composition-matched control,")
    print("  because the motif IS a composition. It is reported as the")
    print("  COMPOSITION SENTINEL: whatever a pure pyrimidine-content contrast")
    print("  returns in this window, it returns here.")
    for k, thr in ((10, 9), (10, 10), (15, 13)):
        pat = f"[CT]{{{thr}}}" if thr == k else None
        f = d["win"].str.contains(f"(?=([CT]{{{thr}}}))", regex=True)
        r = standardised(d.assign(_f=f), "_f", STRATA)
        diff = (r[True][2] - r[False][2]) if (True in r and False in r) else np.nan
        print(f"  run of >= {thr} pyrimidines: n present {int(f.sum()):>6,}   "
              f"standardised diff {diff:>+7.2f}pp")

    # ------------------------------------------------------ the joint test
    print("\n" + "=" * 96)
    print("D. THE THREE-DIRECTION TEST")
    print("=" * 96)
    print("\n  If these are mechanisms, the signs are  PAS -, ARE +, CU-rich -.")
    print("  If this is composition, all three carry the sign of the AT-content")
    print("  effect in this window. That effect is measured directly:\n")
    for q in range(5):
        g = d[d["gc_q"].eq(q)]
        print(f"    window GC quintile {q+1}: median GC {g['gc'].median()*100:>5.1f}%   "
              f"n {len(g):>6,}   NMD+ {g['is_nmd'].mean()*100:>5.1f}%")
    print("\n  So the composition gradient in this window runs in the direction")
    print("  shown above, and any motif effect has to be read against it.")

    print("\n" + "=" * 96)
    print("E. RESTRICTED TO PTC+ TRANSCRIPTS -- where a 3'UTR-borne modifier of")
    print("   decay has something to modify")
    print("=" * 96)
    g = d[d["ptc"].eq(1)]
    print(f"\n  n = {len(g):,}, NMD+ {g['is_nmd'].mean()*100:.1f}%")
    for nm, m in (("PAS", PAS), ("ARE", ARE)):
        v, n = effect(g, "win", m, ["utr3_q", "gc_q"])
        print(f"    {nm:<5} {m:<12} n present {n:>6,}   "
              f"standardised diff {v:>+7.2f}pp")

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
