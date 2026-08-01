#!/usr/bin/env python
"""
analysis3c_codon_composition_test.py — is the period-3 preference a coding-composition detector?

THE PREDICTION. The capture head prefers G at codon position 1, A at position 2 and
C at position 3, in a frame anchored on each candidate's own AUG, strengthening
downstream. The proposed explanation is that it has learned the period-3
compositional signature of coding sequence — the same signal ab initio gene
finders use.

That explanation makes a falsifiable prediction: **the model's per-phase base
preference should track the ACTUAL base composition of real coding sequence by
codon position, measured in these very transcripts.** If the ranking matches,
the interpretation stands. If the model prefers bases that real coding sequence
does not, the periodicity is frame-locked but is something else.

MEASURED HERE, from the annotated CDS regions of the pooled transcripts: base
frequency at codon positions 1, 2 and 3, computed over the reading frame that
starts at each isoform's GENCODE-annotated AUG. Compared against the model's
signed capture preference at matching phases.

CONTROLS THAT MATTER.

  The comparison is made in the CODING region only (downstream of the annotated
  start), because that is where coding composition exists. The 5'UTR is included
  as a negative control: if the model's preference tracks composition, it should
  track it where composition is real and fail to where it is not.

  Composition is reported against the same window's OVERALL base frequency, not
  against 0.25, so a global GC bias cannot masquerade as a positional preference.

  The model's preference is the five-seed mean already measured; nothing is
  re-run.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRACK_A = Path.home() / "claude_projects" / "nmd_lung_longread_2026"
sys.path.insert(0, str(TRACK_A / "tools"))
sys.path.insert(0, str(REPO))
from claim_emit import emit                                  # noqa: E402
from tensor_io import decode_windows                         # noqa: E402

TENSOR = REPO / "results_tensor_v6" / "nmd_tensor.h5"
FLAGS = REPO / "results_ism_v6" / "gencode_candidate_flags.tsv"
PROBE = REPO / "results_interp_all" / "period3_probe.npz"
BASES = "ACGT"
N_TX = 3000


def main():
    sys.stdout.reconfigure(line_buffering=True)
    with h5py.File(TENSOR, "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        codes = f["codes"][:]
        L, W = int(f.attrs["atg_left"]), int(f.attrs["window"])

    fl = pd.read_csv(FLAGS, sep="\t",
                     usecols=["isoform_id", "slot", "has_gencode_cds", "is_gencode_start"])
    fl = fl[(fl.has_gencode_cds == 1) & (fl.is_gencode_start == 1)]
    key = {s: i for i, s in enumerate(iso)}
    fl = fl[fl.isoform_id.isin(key)].sample(min(N_TX, len(fl)), random_state=20260801)
    print(f"measuring codon-position base composition over {len(fl):,} "
          f"annotated coding frames")

    # ---- empirical base composition by codon position -----------------------
    # fill state 1..4 == A,C,G,T; the frame is anchored on the annotated AUG,
    # which sits at window index L, so window index k has phase (k - L) % 3.
    cds = np.zeros((3, 4), dtype=np.int64)      # downstream of the AUG: coding
    utr = np.zeros((3, 4), dtype=np.int64)      # upstream of it: negative control
    for r in fl.itertuples():
        i = key[r.isoform_id]
        row = int(off[i]) + int(r.slot)
        bc = codes[row, 0]
        fill = (bc & 7)
        ok = (fill >= 1) & (fill <= 4)
        idx = np.flatnonzero(ok)
        ph = (idx - L) % 3
        b = fill[idx] - 1
        down = idx >= L
        np.add.at(cds, (ph[down], b[down]), 1)
        np.add.at(utr, (ph[~down], b[~down]), 1)

    def freq(tab):
        return tab / tab.sum(axis=1, keepdims=True)

    fc, fu = freq(cds), freq(utr)
    overall_c = cds.sum(0) / cds.sum()
    overall_u = utr.sum(0) / utr.sum()

    print(f"\n=== observed base frequency by codon position ===")
    print(f"  CODING region (downstream of the annotated AUG), "
          f"{cds.sum():,} positions")
    print(f"  {'codon pos':<11}" + "".join(f"{b:>9}" for b in BASES) + "   most enriched")
    for ph in range(3):
        enr = fc[ph] / overall_c
        print(f"  {ph+1:<11}" + "".join(f"{x:>9.4f}" for x in fc[ph])
              + f"   {BASES[int(np.argmax(enr))]}  (x{enr.max():.3f} vs window mean)")
    print(f"  {'window mean':<11}" + "".join(f"{x:>9.4f}" for x in overall_c))

    print(f"\n  5'UTR region (upstream), {utr.sum():,} positions — negative control")
    print(f"  {'codon pos':<11}" + "".join(f"{b:>9}" for b in BASES) + "   most enriched")
    for ph in range(3):
        enr = fu[ph] / overall_u
        print(f"  {ph+1:<11}" + "".join(f"{x:>9.4f}" for x in fu[ph])
              + f"   {BASES[int(np.argmax(enr))]}  (x{enr.max():.3f} vs window mean)")
    print(f"  {'window mean':<11}" + "".join(f"{x:>9.4f}" for x in overall_u))

    # ---- the model's preference, from the probe already run -----------------
    z = np.load(PROBE)
    S, N = z["S_annotated"].sum(0) / z["S_annotated"].shape[0], z["N_annotated"]
    print(f"\n=== the model's signed capture preference, same phases ===")
    print(f"  measured downstream of the AUG (offsets 3..99), five-seed mean")
    mod = np.zeros((3, 4))
    for ph in range(3):
        sel = [L + o for o in range(3, 100) if o % 3 == ph]
        mod[ph] = S[sel].sum(0) / np.maximum(N[sel].sum(0), 1)
        print(f"  {ph+1:<11}" + "".join(f"{x:>9.4f}" for x in mod[ph])
              + f"   {BASES[int(np.argmax(mod[ph]))]}")

    print(f"\n=== does the model's preference track real coding composition? ===")
    hits = 0
    for ph in range(3):
        enr = fc[ph] / overall_c
        obs, pred = BASES[int(np.argmax(enr))], BASES[int(np.argmax(mod[ph]))]
        r = np.corrcoef(enr, mod[ph])[0, 1]
        ok = obs == pred
        hits += ok
        print(f"  codon position {ph+1}: coding-enriched base {obs}, "
              f"model prefers {pred}   {'MATCH' if ok else 'differ'}   "
              f"rank correlation of the four bases r = {r:+.3f}")
    print(f"\n  {hits}/3 codon positions agree on the most-preferred base")
    allr = np.corrcoef((fc / overall_c).ravel(), mod.ravel())[0, 1]
    print(f"  correlation over all twelve position-by-base cells: r = {allr:+.3f}")
    emit("5.90.20", "correlation between model per-phase base preference and "
         "observed coding-sequence composition", float(allr), n=12,
         population="annotated coding frames of sampled pooled transcripts; "
                    "base enrichment relative to window mean at codon positions "
                    "1-3 versus five-seed mean signed delta capture at the same "
                    "phases, offsets +3 to +99")

    ur = np.corrcoef((fu / overall_u).ravel(), mod.ravel())[0, 1]
    print(f"  same against the 5'UTR composition (negative control): r = {ur:+.3f}")


if __name__ == "__main__":
    main()
