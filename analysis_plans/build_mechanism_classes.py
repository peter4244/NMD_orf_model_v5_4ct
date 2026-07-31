#!/usr/bin/env python3
"""
build_mechanism_classes.py — partition the model's isoform universe by NMD mechanism, in the
vocabulary of manuscript section 4.

WHY THIS EXISTS. Section 5's interpretation work needs to know WHICH NMD mechanism an isoform
carries, because the model's response to a sequence perturbation has a different meaning in each.
Section 4 already defines a mechanism classification, but it is defined on ISOPAIRS (an NMD
isoform and its gene-matched comparator, n=1,385 at ENST-only scope) and cannot stratify the
model's 41,765 isoforms. This builds the isoform-level analogue and keeps every definition it
shares with section 4 identical, so the two cannot drift.

TWO DEFINITIONS ARE TAKEN FROM SECTION 4 RATHER THAN CHOSEN HERE
(figures/multipanel/figure4_ptcneg_and_model/RATIONALE.md section 3):

  1. A PTC is a stop with an exon junction MORE THAN 50 nt downstream. Not "at least one
     downstream junction" -- measured here, 7.9% of slots with n_downstream_ejc >= 1 have every
     downstream junction within 50 nt, and counting those as PTCs costs class PTC+ 8.2 points of
     NMD+ rate.

  2. The CDS anchor is the GENCODE-projected reference AUG (`is_ref_cds`), NEVER the TD2 call
     (`is_sqanti_cds`). RATIONALE section 3: the section 4 classification is "100% ref-AUG-derived
     and does NOT use Isopair's original_ptc field ... TD2 has the documented PTC-avoidance bias".
     Measured here in the model universe: anchoring on TD2 moves 1,556 isoforms out of PTC+
     (-23%) and inflates the uORF class by 135%, because TD2's ATG sits downstream of the
     reference AUG in 99.1% of occult-PTC pairs, so a main-ORF PTC is re-read as an upstream ORF.

THE PTC CALL IS WINDOW-WIDTH INDEPENDENT, deliberately. A junction beyond the stop window would
be invisible, so "no junction more than 50 nt downstream within the window" is not sufficient.
`n_downstream_ejc` counts junctions over the whole transcript, so comparing it against the number
visible in the window detects a junction beyond the window, which is necessarily more than 50 nt
downstream. Without this the same isoform can be classified differently at the two configurations.

WHAT THIS IS NOT. Section 4's groups are named NMD+/PTC+ and so on because section 4 classifies
NMD+ pairs. These classes partition ALL isoforms without reference to the label, so the NMD+ rate
of a class is a result rather than part of its definition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct")
PTC_MIN_NT = 50          # section 4: a junction MORE than 50 nt downstream of the stop
REF_I, SQ_I = 2, 3       # orf_features columns is_ref_cds / is_sqanti_cds
FS_I, FE_I, EJC_I = 0, 1, 4

CLASSES = {
    "PTC+":             "the reference-AUG CDS stop has a junction >50 nt downstream",
    "PTC- uORF-PTC":    "reference-AUG CDS stop is clean; an ORF starting upstream of it is a PTC",
    "PTC- no trigger":  "reference-AUG CDS stop is clean; no upstream ORF is a PTC",
    "Ref AUG absent":   "no slot matches the GENCODE-projected reference AUG",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(REPO / "results_4ct_dn/nmd_orf_data.h5"))
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO / "results_interp_all/mechanism_classes"))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    with h5py.File(args.h5, "r") as f:
        F = f["orf_features"][:]
        M = f["orf_mask"][:]
        L = f["labels"][:]
        n = len(L)
        ds = f[f"w{args.width}"]["stop_windows"]
        c = ds.shape[-1] // 2
        beyond50 = np.zeros(M.shape, bool)
        n_vis = np.zeros(M.shape, np.int32)
        for a in range(0, n, 4000):
            b = min(a + 4000, n)
            down = ds[a:b, :, 4, c + 1:]          # down[...,k] is a junction at distance k+1
            beyond50[a:b] = down[:, :, PTC_MIN_NT:].max(-1) > 0
            n_vis[a:b] = (down > 0).sum(-1)
    print(f"{args.h5}: {n:,} isoforms, NMD+ {L.mean():.1%}, stop window w{args.width}")

    ejc = F[:, :, EJC_I]
    fs, fe = F[:, :, FS_I], F[:, :, FE_I]
    ref, sq = F[:, :, REF_I] > 0.5, F[:, :, SQ_I] > 0.5

    # A slot is a PTC when a junction lies MORE than 50 nt downstream of its stop. Either one is
    # visible in the window, or the transcript-wide count exceeds what the window shows, which
    # places a junction beyond the window and therefore beyond 50 nt.
    beyond_window = ejc > n_vis
    ptc = M & (ejc >= 1) & (beyond50 | beyond_window)
    print(f"  slots masked in {M.sum():,}; n_downstream_ejc>=1 {(M & (ejc >= 1)).sum():,}; "
          f"PTC by the >50nt rule {ptc.sum():,} "
          f"({1 - ptc.sum() / (M & (ejc >= 1)).sum():.1%} of them demoted)")
    print(f"  demotions rescued by the beyond-window count: "
          f"{(M & (ejc >= 1) & ~beyond50 & beyond_window).sum():,} slots")

    r = np.arange(n)
    has_ref = (ref & M).any(1)
    assert (ref & M).sum(1).max() <= 1, "more than one slot flagged is_ref_cds"
    mi = (ref & M).argmax(1)
    ref_fs = np.where(has_ref, fs[r, mi], np.nan)
    main_ptc = has_ref & ptc[r, mi]
    up = ptc & (fs < ref_fs[:, None] - 1e-9)
    up[r, mi] = False

    cls = np.full(n, "Ref AUG absent", dtype=object)
    cls[has_ref & ~main_ptc & ~up.any(1)] = "PTC- no trigger"
    cls[has_ref & ~main_ptc & up.any(1)] = "PTC- uORF-PTC"
    cls[main_ptc] = "PTC+"

    # Within the uORF class, whether the upstream PTC ORF also STOPS before the reference AUG
    # (a proper uORF) or reads through it (overlapping uORF / N-terminal extension).
    first_up = up.argmax(1)
    proper = np.zeros(n, bool)
    b = cls == "PTC- uORF-PTC"
    proper[b] = fe[b, first_up[b]] < ref_fs[b] - 1e-9

    print(f"\n{'class':<18} {'definition':<62} {'n':>7} {'share':>7} {'NMD+':>7}")
    for k, v in CLASSES.items():
        s = cls == k
        print(f"{k:<18} {v:<62} {s.sum():>7,} {s.mean():>6.1%} {L[s].mean():>6.1%}")
    print(f"\nshare of all NMD+ isoforms: " + "  ".join(
        f"{k} {L[cls == k].sum() / L.sum():.1%}" for k in CLASSES))
    print(f"PTC- uORF-PTC composition: proper uORF (stops before the reference AUG) "
          f"{proper[b].mean():.1%}, overlapping or N-terminal extension {1 - proper[b].mean():.1%}")

    anchor = np.where(ref[:, 0] & sq[:, 0], "ref+TD2 agree",
             np.where(ref[:, 0], "ref only (TD2 disagrees)",
             np.where(sq[:, 0], "TD2 only", "neither (Kozak fill)")))
    print(f"\nslot-0 anchor, and why the TD2 flag is not neutral:")
    for k in ("ref only (TD2 disagrees)", "ref+TD2 agree", "TD2 only", "neither (Kozak fill)"):
        s = anchor == k
        print(f"  {k:<26} n={s.sum():>6,} ({s.mean():>5.1%})  NMD+ {L[s].mean():>5.1%}  "
              f"slot 0 is a PTC {ptc[s, 0].mean():>5.1%}")

    np.savez_compressed(args.out + ".npz", cls=cls.astype("U16"), ptc=ptc, mi=mi,
                        has_ref=has_ref, proper_uorf=proper, anchor=anchor.astype("U26"),
                        labels=L, width=args.width)
    with open(args.out + ".tsv", "w") as fh:
        fh.write("row\tmechanism_class\tproper_uorf\tslot0_anchor\tref_slot\tlabel\n")
        for i in range(n):
            fh.write(f"{i}\t{cls[i]}\t{int(proper[i])}\t{anchor[i]}\t"
                     f"{mi[i] if has_ref[i] else -1}\t{int(L[i])}\n")
    print(f"\nwrote {args.out}.npz and {args.out}.tsv")


if __name__ == "__main__":
    main()
