"""
model_a2_exclusion_check.py — is the >=100 floor differential on the mechanism cell?

The one unmeasured risk in the SEQ-A2 banding. At 8 global live-mass bands with a
>=100 live floor, 46.0% of (transcript x band) cells qualify (job 8896445).
Qualifying cells are spread evenly ACROSS bands -- 2,100 to 2,653 -- which rules
out a starved band. It does not rule out a starved transcript CLASS, and section
5.5 says exactly which class to suspect: unreachable positions concentrate in
transcripts with long 5'UTRs, which is the mechanism cell.

If the floor removes NMD transcripts, or long-5'UTR transcripts, at a different
rate from their comparators, then the gate is computed on a population that
differs from the one the claim is about -- and the exclusion would be invisible in
the result, because a cell that does not qualify never appears in it.

--------------------------------------------------------------------------------
THE ROW, recorded before the run. Thirteen fields per the template.

 1 HYPOTHESIS      The >=100 live floor at 8 bands retains the same fraction of a
                   transcript's live positions regardless of NMD label, 5'UTR
                   length, or dead fraction. (Stated as the null it would be
                   comforting to keep, so that finding against it is the
                   informative outcome.)
 2 SELECTION RULE  Every transcript in the bank with >=1 live position. Cells are
                   (transcript x band); a cell qualifies at >=100 LIVE positions.
                   Bands are global log-mass quantiles over live positions -- the
                   same construction the gate uses, not a reimplementation.
 3 BACKGROUND      Each transcript is its own denominator. Retention is that
                   transcript's live positions inside qualifying cells over its
                   total live positions. No pooled background: pooling would let
                   large transcripts mask small ones, which is the effect being
                   tested.
 4 HELD FIXED      Nothing, by stratification or otherwise. This is a descriptive
                   comparison of retention across groups, not an effect estimate.
 5 NOT HELD        Transcript size, deliberately. Size is the obvious driver of
                   retention and is ALSO how the mechanism cell differs, so
                   adjusting it away would remove the exposure. It is reported as
                   a stratifier instead (field 8) so a size-driven answer is
                   visible rather than hidden.
 6 NULL            None. No test statistic and no p-value: this is enumeration.
                   A null here would invite a non-significant result to be read as
                   "not differential", which at this n it would not support.
 7 REFERENCE PTS   Retention is bounded [0, 1] by construction, both ends
                   attainable and both observed in-sample. Reported as a
                   distribution, not a mean -- the 8896445 dead-fraction result is
                   the worked example of a mean hiding a concentrated tail.
 8 AGGREGATION     Per transcript, then reported by group as deciles. Groups:
                   NMD label; 5'UTR-length quintile; dead-fraction stratum.
 9 SWEEP           The floor, over 50 / 100 / 200, and band count 4 / 8 / 16. The
                   floor is the parameter under suspicion, so reporting one value
                   of it would be reporting a property of the number we chose.
10 DECISION RULE   Fixed before the run, three outcomes.
                   NOT DIFFERENTIAL -- median retention differs by <5 percentage
                     points across every group comparison. Freeze 8 / 10% / >=100
                     and proceed.
                   DIFFERENTIAL -- any comparison differs by >15 points. The floor
                     is selecting a population; report the gate stratified by the
                     offending variable, or lower the floor until it is not.
                   AMBIGUOUS -- between 5 and 15. Not resolved by looking at it:
                     resolved by the floor sweep in field 9, decided before it is
                     examined.
11 LICENSED        If not differential: "the floor removes cells, not classes."
                   It does NOT license "the floor is harmless" -- 54% of cells
                   still leave the analysis and that bounds power everywhere.
                   If differential: nothing about WHY. Retention is a geometric
                   property of a transcript, and a mechanism-cell association
                   could be size, mass spread, or 5'UTR length; this design cannot
                   separate them and is not asked to.
12 OWNER           Model window. No second implementation: this is a diagnostic
                   that sets a parameter, not a load-bearing result. Replication
                   is reserved for survivors.
13 ENUMERATION     Reported beside every number: n transcripts per group, n live
                   positions, n cells, n qualifying, seed, dead handling, and the
                   finite-mask expression.

--------------------------------------------------------------------------------
Run from the repo root.
"""

import argparse
import numpy as np
import h5py

NT = "ACGT"
DEAD_CUT = 1e-8


def transcript_rows(f, N):
    """Per-transcript live log-mass, plus the covariates the groups are cut on."""
    spans = f["spans"][:]
    cand_off = f["cand_offset"][:]
    cand_cnt = f["cand_count"][:]
    labels = f["labels"][:]
    p_select = f["p_select"][:]
    orf_start = f["cand_orf_start"][:]
    up_ref = f["cand_upstream_of_ref"][:]

    out = []
    for i in range(N):
        lo, nk = int(cand_off[i]), int(cand_cnt[i])
        b = spans[lo:lo + nk]
        P = int(max(b[:, 3].max(), b[:, 5].max()))
        if P < 50:
            continue
        v = f["vals_decay"][i, :P].astype(np.float64)
        o = f["obs"][i, :P]
        m = f["mass"][i, :P].astype(np.float64)
        with np.errstate(invalid="ignore"):
            e = np.nanmax(np.abs(v), axis=1)
        ok = f["valid"][i, :P].astype(bool) & np.isfinite(e) & (o >= 0)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        mm = m[idx]
        live = mm >= DEAD_CUT
        if not live.any():
            continue

        # the model's own choice of reading frame, which is what "5'UTR length"
        # means for a transcript whose annotation may disagree with it
        ps = p_select[lo:lo + nk]
        sel_k = int(np.argmax(ps))
        out.append(dict(
            i=i,
            lm=np.log10(mm[live]).astype(np.float32),
            n_valid=len(idx),
            n_live=int(live.sum()),
            dead_frac=float((~live).sum()) / len(idx),
            label=int(labels[i]),
            utr5=int(orf_start[lo + sel_k]),
            has_up_ref=bool(np.any(up_ref[lo:lo + nk])),
        ))
    return out


def qualifying_retention(rows, nb, floor):
    """Fraction of each transcript's live positions inside qualifying cells."""
    allm = np.concatenate([r["lm"] for r in rows])
    edges = np.quantile(allm, np.linspace(0, 1, nb + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ret = np.zeros(len(rows))
    nq = np.zeros(len(rows), dtype=int)
    for j, r in enumerate(rows):
        sizes = np.bincount(np.digitize(r["lm"], edges[1:-1]), minlength=nb)
        keep = sizes >= floor
        nq[j] = int(keep.sum())
        ret[j] = sizes[keep].sum() / sizes.sum()
    return ret, nq


def decile_line(x):
    return " ".join(f"{np.percentile(x, d):.3f}" for d in range(0, 101, 25))


def group_report(name, ret, nq, groups, labels_txt, n_live):
    print(f"\n  BY {name}")
    print(f"    {'group':<22} {'n_tx':>6} {'n_live':>12} {'median ret':>11}"
          f" {'ret 0/25/50/75/100':>28} {'zero-cell tx':>13}")
    meds = []
    for g, txt in zip(groups, labels_txt):
        if not g.any():
            continue
        r = ret[g]
        meds.append(np.median(r))
        print(f"    {txt:<22} {g.sum():>6} {n_live[g].sum():>12,}"
              f" {np.median(r):>11.3f} {decile_line(r):>28}"
              f" {(nq[g] == 0).sum():>13}")
    if len(meds) >= 2:
        spread = (max(meds) - min(meds)) * 100
        verdict = ("NOT DIFFERENTIAL" if spread < 5 else
                   "DIFFERENTIAL" if spread > 15 else "AMBIGUOUS")
        print(f"    -> median retention spread {spread:.1f} points : {verdict}")
    return meds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
    args = ap.parse_args()

    with h5py.File(args.bank, "r") as f:
        N = len(f["transcript_id"])
        print(f"BANK {args.bank}   transcripts {N}   dead cut {DEAD_CUT:g}")
        print("finite mask: (np.isfinite(vals_decay).sum(1) == 3)\n")
        rows = transcript_rows(f, N)

    n_live = np.array([r["n_live"] for r in rows])
    lab = np.array([r["label"] for r in rows])
    utr5 = np.array([r["utr5"] for r in rows])
    dead = np.array([r["dead_frac"] for r in rows])
    upref = np.array([r["has_up_ref"] for r in rows])

    print(f"transcripts contributing {len(rows)}   live positions {n_live.sum():,}")
    print(f"NMD {int((lab == 1).sum())}   control {int((lab == 0).sum())}")
    print(f"5'UTR length (model-selected ORF start), quintile edges: "
          + " ".join(f"{np.percentile(utr5, q):.0f}" for q in (0, 20, 40, 60, 80, 100)))

    for nb in (4, 8, 16):
        for floor in (50, 100, 200):
            ret, nq = qualifying_retention(rows, nb, floor)
            tag = f"{nb} bands, floor >={floor}"
            print("\n" + "=" * 78)
            print(f"{tag}   cells {len(rows)*nb:,}"
                  f"   qualifying {int((nq).sum()):,}"
                  f"   live retained {(ret*n_live).sum()/n_live.sum():.1%}")
            print("=" * 78)

            group_report("NMD LABEL", ret, nq,
                         [lab == 1, lab == 0], ["NMD", "control"], n_live)

            qs = np.percentile(utr5, [20, 40, 60, 80])
            gs = [utr5 <= qs[0],
                  (utr5 > qs[0]) & (utr5 <= qs[1]),
                  (utr5 > qs[1]) & (utr5 <= qs[2]),
                  (utr5 > qs[2]) & (utr5 <= qs[3]),
                  utr5 > qs[3]]
            group_report("5'UTR LENGTH QUINTILE", ret, nq, gs,
                         [f"Q{k+1}" for k in range(5)], n_live)

            group_report("DEAD FRACTION", ret, nq,
                         [dead == 0, (dead > 0) & (dead <= 0.1), dead > 0.1],
                         ["none", "0-10%", ">10%"], n_live)

            group_report("UPSTREAM-OF-REF CANDIDATE", ret, nq,
                         [upref, ~upref], ["has upstream", "none"], n_live)


if __name__ == "__main__":
    main()
