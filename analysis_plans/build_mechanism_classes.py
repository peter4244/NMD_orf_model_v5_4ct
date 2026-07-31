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
     downstream junction" -- the demoted fraction and its cost in NMD+ rate are printed by this
     script rather than quoted here, because an earlier version of this docstring carried a
     number that its own run log contradicted.

  2. The CDS anchor is the REFERENCE ISOFORM'S PROJECTED AUG (`is_ref_cds`), never the isoform's
     own TD2 call (`is_sqanti_cds`). RATIONALE section 3: the section 4 classification is "100%
     ref-AUG-derived and does NOT use Isopair's original_ptc field ... TD2 has the documented
     PTC-avoidance bias". That is the part which matters -- the anchor does not come from the
     transcript under test -- and it is what avoids the bias.

     TWO DIVERGENCES FROM SECTION 4, DISCLOSED RATHER THAN IMPLIED. (a) The reference AUG is
     read from the reference isoform's CDS in `cds.rds`, which is built from the SQANTI corrected
     CDS GFF3, so for a reference isoform that is itself a novel PB transcript the anchor is a
     TD2 call one step removed. (b) Section 4 restricts primary analyses to GENCODE-annotated
     references via `ref_source()` in `figures/lib/mechanism_class.R`; this does not, and 32.8%
     of the reference isoforms here are novel PB transcripts rather than ENST. The dilution is
     class-dependent, so it is not a uniform offset. Neither divergence is described by the word
     "GENCODE", which an earlier version of this docstring used.

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
import pandas as pd

REPO = Path("/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct")
PTC_MIN_NT = 50          # section 4: a junction MORE than 50 nt downstream of the stop
REF_I, SQ_I = 2, 3       # orf_features columns is_ref_cds / is_sqanti_cds
FS_I, FE_I, EJC_I = 0, 1, 4

CLASSES = {
    "PTC+":             "the reference-AUG CDS stop has a junction >50 nt downstream",
    "PTC- uORF-PTC":    "reference-AUG CDS stop is clean; an ORF starting upstream of it is a PTC",
    "PTC- no trigger":  "reference-AUG CDS stop is clean; no upstream ORF is a PTC",
    "Ref AUG absent":   "no slot matches the reference isoform's projected AUG",
}


def full_scan_upstream_ptc(data_dir, ids, has_ref):
    """Does ANY ORF in the full ORFik scan, starting upstream of the reference AUG, carry a PTC?

    The 5 model slots are a Kozak-priority subsample of the scan -- mean 36.6 ORFs per isoform --
    so a uORF can be true and absent from the slots. Same PTC rule as everywhere else: a junction
    more than 50 nt past the ORF's stop. The furthest junction is sufficient to decide it.
    """
    orf = pd.read_csv(data_dir / "orf_features.tsv", sep="\t",
                      usecols=["isoform_id", "orf_start", "orf_end"])
    jn = pd.read_csv(data_dir / "junctions.tsv", sep="\t")
    jmax = {i: (max(int(x) for x in str(s).split(",")) if isinstance(s, str) and s.strip() else -1)
            for i, s in zip(jn["isoform_id"], jn["junctions"])}
    sel = pd.read_csv(data_dir / "selected_orfs.tsv", sep="\t",
                      usecols=["isoform_id", "orf_start", "is_ref_cds"])
    ref_start = dict(zip(sel.loc[sel["is_ref_cds"] == 1, "isoform_id"],
                         sel.loc[sel["is_ref_cds"] == 1, "orf_start"]))
    orf["jmax"] = orf["isoform_id"].map(jmax).fillna(-1)
    orf["ref_start"] = orf["isoform_id"].map(ref_start)
    hit = orf[(orf["jmax"] > orf["orf_end"] + PTC_MIN_NT - 1)
              & (orf["orf_start"] < orf["ref_start"])]["isoform_id"].unique()
    n_orfs = orf.groupby("isoform_id").size()
    print(f"  full ORFik scan: {len(orf):,} ORFs over {orf['isoform_id'].nunique():,} isoforms "
          f"(mean {n_orfs.mean():.1f} per isoform, max {n_orfs.max():,}); "
          f"the model keeps at most 5")
    return np.isin(ids, hit) & has_ref


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
        ids = np.array([x.decode() if isinstance(x, bytes) else x
                        for x in f["isoform_id"][:]])
        n = len(L)
        ds = f[f"w{args.width}"]["stop_windows"]
        c = ds.shape[-1] // 2
        beyond50 = np.zeros(M.shape, bool)
        n_vis = np.zeros(M.shape, np.int32)
        for a in range(0, n, 4000):
            b = min(a + 4000, n)
            down = ds[a:b, :, 4, c + 1:]          # down[...,k] is a junction at distance k+1
            beyond50[a:b] = down[:, :, PTC_MIN_NT:].max(-1) > 0
            # Index 0 is the LAST BASE of the stop codon (the window is centered on the middle
            # base), and `n_downstream_ejc` is sum(junctions > orf_end), which excludes it.
            # Counting it here inflates n_vis and can cancel a genuine beyond-window junction,
            # which also destroys the width-independence this rule is built for.
            n_vis[a:b] = (down[:, :, 1:] > 0).sum(-1)
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
    # (a proper uORF) or reads through it (overlapping uORF / N-terminal extension). Take the
    # MOST UPSTREAM qualifying slot, not the first in slot order -- slot order is the priority
    # rule (annotation, then Kozak) and carries no positional meaning, and 40% of this class has
    # more than one qualifying slot.
    # SUBGROUP BY WHETHER THE UPSTREAM ORF READS INTO THE CDS, and it is a mechanism split rather
    # than a purity check (Pete, 2026-07-31): an out-of-frame overlapping ORF carries the ribosome
    # past the main AUG in the wrong frame, so there is no reinitiation rescue, which is the more
    # NMD-disposing arrangement. An IN-FRAME upstream ORF cannot appear here at all -- it
    # terminates at the main ORF's own stop and so inherits its PTC status, and this class is
    # conditioned on that stop being clean. Measured: 0 of the class.
    up_fs = np.where(up, fs, np.inf)
    most_up = up_fs.argmin(1)
    b = cls == "PTC- uORF-PTC"
    reads_in = np.zeros(n, bool)
    reads_in[b] = fe[b, most_up[b]] >= ref_fs[b] - 1e-9
    proper = b & ~reads_in                      # terminates in the 5'UTR
    uorf_kind = np.where(b & reads_in, "oORF", np.where(proper, "uORF", ""))

    # --- the same uORF question asked of the FULL ORFik scan, not just the 5 model slots -----
    # The model sees at most 5 candidate ORFs, chosen by annotation then Kozak. Asking "is there
    # an upstream PTC-bearing ORF" of that set answers what the MODEL can see; asking it of the
    # full scan answers what is TRUE. Both are reported because the gap between them is the uORF
    # evidence the model is structurally unable to use.
    fs_up = full_scan_upstream_ptc(Path(args.h5).parent, ids, has_ref)
    slot_up = up.any(1)
    both = has_ref & ~main_ptc
    print(f"\nUPSTREAM PTC-BEARING ORF: model's 5 slots vs the full ORFik scan")
    print(f"  isoforms with a clean reference-AUG stop (the population at issue): {both.sum():,}")
    print(f"  found in the 5 model slots : {(both & slot_up).sum():,}  "
          f"NMD+ {L[both & slot_up].mean():.1%}")
    print(f"  found in the full scan     : {(both & fs_up).sum():,}  "
          f"NMD+ {L[both & fs_up].mean():.1%}")
    miss = both & fs_up & ~slot_up
    print(f"  TRUE but INVISIBLE to the model: {miss.sum():,} isoforms "
          f"({miss.sum() / max((both & fs_up).sum(), 1):.1%} of the true set), NMD+ "
          f"{L[miss].mean():.1%}")

    print(f"\n{'class':<18} {'definition':<62} {'n':>7} {'share':>7} {'NMD+':>7}")
    for k, v in CLASSES.items():
        s = cls == k
        print(f"{k:<18} {v:<62} {s.sum():>7,} {s.mean():>6.1%} {L[s].mean():>6.1%}")
    print(f"\nshare of all NMD+ isoforms: " + "  ".join(
        f"{k} {L[cls == k].sum() / L.sum():.1%}" for k in CLASSES))
    print(f"PTC- uORF-PTC subgroups (labelled by the MOST UPSTREAM qualifying ORF):")
    for k in ("uORF", "oORF"):
        s_ = uorf_kind == k
        lab = ("terminates in the 5'UTR" if k == "uORF"
               else "reads into the CDS, out of frame -- no reinitiation rescue")
        print(f"  {k:<5} {lab:<56} n={s_.sum():>6,} ({s_.sum()/max(b.sum(),1):>5.1%})  "
              f"NMD+ {L[s_].mean():>5.1%}")

    anchor = np.where(ref[:, 0] & sq[:, 0], "ref+TD2 agree",
             np.where(ref[:, 0], "ref only (TD2 disagrees)",
             np.where(sq[:, 0], "TD2 only", "neither (Kozak fill)")))
    print(f"\nslot-0 anchor, and why the TD2 flag is not neutral:")
    for k in ("ref only (TD2 disagrees)", "ref+TD2 agree", "TD2 only", "neither (Kozak fill)"):
        s = anchor == k
        print(f"  {k:<26} n={s.sum():>6,} ({s.mean():>5.1%})  NMD+ {L[s].mean():>5.1%}  "
              f"slot 0 is a PTC {ptc[s, 0].mean():>5.1%}")

    np.savez_compressed(args.out + ".npz", cls=cls.astype("U16"), ptc=ptc, mi=mi,
                        has_ref=has_ref, proper_uorf=proper,
                        uorf_kind=uorf_kind.astype("U6"), fullscan_upstream_ptc=fs_up, anchor=anchor.astype("U26"),
                        labels=L, width=args.width)
    with open(args.out + ".tsv", "w") as fh:
        fh.write("row\tmechanism_class\tuorf_kind\tfullscan_upstream_ptc\tslot0_anchor\tref_slot\tlabel\n")
        for i in range(n):
            fh.write(f"{i}\t{cls[i]}\t{uorf_kind[i]}\t{int(fs_up[i])}\t{anchor[i]}\t"
                     f"{mi[i] if has_ref[i] else -1}\t{int(L[i])}\n")
    print(f"\nwrote {args.out}.npz and {args.out}.tsv")


if __name__ == "__main__":
    main()
