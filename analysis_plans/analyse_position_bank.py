#!/usr/bin/env python3
"""
analyse_position_bank.py — contrasts derived from the cached sweep, each against its own
position-matched reference.

FIVE OPERATORS, all re-cuts of the same four per-position values, chosen so that composition
and base identity can be separated:

  GC_vs_AT   {G,C} - {A,T}   pure composition. Channel 5 is recomputed on every substitution,
                             so this operator moves the GC channel and nothing else
                             systematically. It ISOLATES the confound rather than avoiding it.
  purine     {A,G} - {C,T}   the Kozak -3 operator. GC-balanced in the MEAN (one GC base per
                             side) but not within a side.
  A_vs_T      A - T          purine vs pyrimidine at matched GC, both AT bases.
  G_vs_C      G - C          purine vs pyrimidine at matched GC, both GC bases.
  G_vs_notG   G - {A,C,T}    the Kozak +4 operator. Maximally GC-confounded by construction.

A -3 purine preference that is real as base identity must show in BOTH A_vs_T and G_vs_C, which
are individually GC-neutral. One that is composition shows in GC_vs_AT and disappears from both.

THE REFERENCE IS THE CONTROL POSITIONS, not zero. For every statistic reported for a target,
the same statistic is computed at every control position and the target is placed as a
percentile within it. This is what makes across-member agreement interpretable: members share
training data and architecture, so they agree at control positions far more often than the 6.25%
that independent signs would give, and an agreement rate is only evidence when read against that.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

A, C, G, T = 0, 1, 2, 3
OPERATORS = {
    "GC_vs_AT":  ([G, C], [A, T]),
    "purine":    ([A, G], [C, T]),
    "A_vs_T":    ([A], [T]),
    "G_vs_C":    ([G], [C]),
    "G_vs_notG": ([G], [A, C, T]),
}


def contrast(vals, ok, excl, hi_b, lo_b, iso_sel=None, partial_sides=False):
    """Per-(member, isoform, position) contrast; NaN unless BOTH sides are complete.

    COMPLETE-CASE IS THE DEFAULT, and the reason is that a partially-excluded side is not the
    operator it claims to be. Averaging {A,G} against {C} because T was excluded gives a
    contrast whose low side is 100% GC instead of 50%, which is the exact confound `purine` and
    `A_vs_T` exist to avoid. Exclusion is strongly base-asymmetric -- at -3 the rates are A
    0.044, C 0.000, G 0.034, T 0.288 -- so the surviving side is systematically GC-enriched and
    the contrast reads low. Roughly a third of otherwise-usable isoforms are affected at the
    hypothesis positions. `partial_sides=True` reproduces the earlier behavior for comparison.
    """
    keep_hi = np.stack([~excl[:, :, b] for b in hi_b], -1)      # (n_i, n_p, |hi|)
    keep_lo = np.stack([~excl[:, :, b] for b in lo_b], -1)
    hi = np.where(keep_hi[None], vals[:, :, :, hi_b], np.nan)
    lo = np.where(keep_lo[None], vals[:, :, :, lo_b], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN slices are expected
        out = np.nanmean(hi, -1) - np.nanmean(lo, -1)
    side = (lambda m: m.any(-1)) if partial_sides else (lambda m: m.all(-1))
    good = ok[None] & side(keep_hi)[None] & side(keep_lo)[None]
    out[~np.broadcast_to(good, out.shape)] = np.nan
    if iso_sel is not None:
        out[:, ~iso_sel, :] = np.nan
    return out


def summarise(per_iso, n_seeds):
    """Collapse to per-position across-member statistics."""
    with np.errstate(invalid="ignore"):
        pm = np.nanmean(per_iso, axis=1)                        # (n_seeds, n_pos)
    mean, sd = pm.mean(0), pm.std(0, ddof=1)
    t = np.abs(mean) / (sd / np.sqrt(n_seeds))
    agree = (np.sign(pm) == np.sign(pm[0])).all(0)
    n = (~np.isnan(per_iso[0])).sum(0)
    return pm, mean, sd, t, agree, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="position_bank.npz")
    ap.add_argument("--strata", default="all",
                    choices=["all", "attention", "observed_base", "flags"])
    ap.add_argument("--h5", default="/Users/petecastaldi/claude_projects/"
                                    "NMD_orf_model_v5_4ct/results_4ct_dn/nmd_orf_data.h5")
    ap.add_argument("--all-offsets", action="store_true",
                    help="use every control position. Default restricts the reference to controls "
                         "sharing the hypothesis positions' codon frame offset, because the "
                         "created-motif exclusion rate depends on it (23.6%% of T-cells at offset "
                         "0 against 8.7%% at offset 2).")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    vals, ok, excl, obs = d["vals"], d["ok"], d["excl"], d["obs"]
    names = [str(x) for x in d["names"]]
    n_seeds = len(d["seeds"])
    ctrl = np.array([n.startswith("c") for n in names])
    tgt = np.where(~ctrl)[0]
    offset = (d["idxs"] - (int(d["W"]) // 2 - 1)) % 3
    if not args.all_offsets:
        assert len(set(offset[tgt])) == 1, "hypothesis positions differ in frame offset"
        ctrl &= offset == offset[tgt[0]]
    print(f"reference: {int(ctrl.sum())} control positions"
          + ("" if args.all_offsets else f" at frame offset {offset[tgt[0]]}, matching the "
                                         f"hypothesis positions"))
    attn_slot = np.nanmean(d["attn"], 0)[:, int(d["slot"])]

    print(f"{args.npz}: {vals.shape[1]} isoforms, {vals.shape[2]} positions "
          f"({int(ctrl.sum())} controls), {n_seeds} members, tag {d['tag']}")
    print(f"method floor (no-op |effect|, per member max): "
          f"{np.array2string(d['floor'], precision=1)}  "
          f"-- every effect below ~{d['floor'].max():.0e} is arithmetic noise")
    flag_strata = {}
    if args.strata == "flags":
        import h5py
        # Raw (un-normalised) flags for the perturbed slot. NMDDataset with split="all" and
        # restrict_to=arange(n) keeps file rows 0..n-1 in order, so these align by construction.
        with h5py.File(args.h5, "r") as f:
            fl = f["orf_features"][0:vals.shape[1], int(d["slot"]), 2:4]
        ref, sq = fl[:, 0] > 0.5, fl[:, 1] > 0.5
        flag_strata = {"neither flag": ~ref & ~sq, "is_sqanti_cds only": ~ref & sq,
                       "is_ref_cds (any)": ref, "either flag": ref | sq}
        print(f"slot-{int(d['slot'])} annotation flags: is_ref_cds {ref.mean():.1%}, "
              f"is_sqanti_cds {sq.mean():.1%}, neither {(~ref & ~sq).mean():.1%}")
        print(f"  mean slot attention by stratum: " + "  ".join(
            f"{k} {attn_slot[v].mean():.3f}" for k, v in flag_strata.items()))

    print(f"observed-base purine fraction: " + "  ".join(
        f"{names[k]} {np.isin(obs[ok[:,k],k],[A,G]).mean():.3f}" for k in tgt) +
        f"   controls median {np.median([np.isin(obs[ok[:,k],k],[A,G]).mean() for k in np.where(ctrl)[0]]):.3f}")

    for opname, (hi_b, lo_b) in OPERATORS.items():
        per_iso = contrast(vals, ok, excl, hi_b, lo_b)
        pm, mean, sd, t, agree, n = summarise(per_iso, n_seeds)
        print(f"\n{'='*92}\n### {opname}")
        print(f"  control reference (n={int(ctrl.sum())} positions): "
              f"|mean| median {np.median(np.abs(mean[ctrl])):.5f}   "
              f"|t| median {np.median(t[ctrl]):.2f}, 90th {np.percentile(t[ctrl],90):.2f}, "
              f"max {t[ctrl].max():.2f}   all-{n_seeds}-signs-agree {agree[ctrl].mean():.0%} "
              f"(chance {2*0.5**n_seeds:.1%})")
        signed = np.mean(np.sign(mean[ctrl]) == np.sign(np.median(mean[ctrl])))
        print(f"  control mean effects are {signed:.0%} same-signed -- a value near 50% means the "
              f"operator has no systematic direction; well above means it does")
        for k in tgt:
            print(f"  TARGET {names[k]:>3}  n={n[k]:<4} mean {mean[k]:>+9.5f}  sd {sd[k]:.5f}  "
                  f"|t| {t[k]:>6.2f}  signs {'agree' if agree[k] else 'DISAGREE'}")
            print(f"           vs controls: |t| exceeds {(t[ctrl]<t[k]).mean():>4.0%}   "
                  f"|mean| exceeds {(np.abs(mean[ctrl])<abs(mean[k])).mean():>4.0%}   "
                  f"magnitude {abs(mean[k])/np.median(np.abs(mean[ctrl])):.2f}x median control")
            print(f"           members {np.array2string(pm[:,k], precision=5, floatmode='fixed')}")

        if args.strata == "attention":
            q = np.percentile(attn_slot, [33.33, 66.67])
            for g in range(3):
                sel = np.digitize(attn_slot, q) == g
                _, mn, _, tt, ag, _ = summarise(contrast(vals, ok, excl, hi_b, lo_b, sel), n_seeds)
                head = f"  attn tercile {g} (mean {attn_slot[sel].mean():.3f}, n={sel.sum()})"
                print(f"{head}  controls: |t| median {np.median(tt[ctrl]):.2f}, "
                      f"agree {ag[ctrl].mean():.0%}")
                for k in tgt:
                    print(f"{'':>4}{names[k]:>5}: mean {mn[k]:>+9.5f}  |t| {tt[k]:>6.2f}  "
                          f"signs {'agree' if ag[k] else 'no':>5}  "
                          f"exceeds {(tt[ctrl]<tt[k]).mean():.0%} of controls")

        if args.strata == "flags":
            # Pete's override hypothesis, tested WITHOUT an ablation. is_ref_cds / is_sqanti_cds
            # tell the model where the start is without reading sequence. If they override
            # initiation context, the Kozak contrast should be weaker where they are PRESENT --
            # a between-isoform comparison in the native model, so it is not the global
            # attention re-weighting that an inference-time ablation produces.
            for lab, sel in flag_strata.items():
                _, mn, _, tt, ag, _ = summarise(contrast(vals, ok, excl, hi_b, lo_b, sel), n_seeds)
                print(f"  {lab:<26} n={sel.sum():<4} controls |t| median {np.median(tt[ctrl]):.2f}, "
                      f"agree {ag[ctrl].mean():>3.0%}   " + "   ".join(
                          f"{names[k]} mean {mn[k]:+.5f} |t| {tt[k]:5.2f} "
                          f"exceeds {(tt[ctrl]<tt[k]).mean():>4.0%}" for k in tgt))

        if args.strata == "observed_base":
            # The target and control positions differ in which base is actually present, and the
            # observed base's own cell is the no-op. Conditioning on it removes that difference.
            for lab, keep in (("obs purine", [A, G]), ("obs pyrimidine", [C, T])):
                mn_l, tt_l, ag_l = [], [], []
                for k in range(len(names)):
                    sel = np.isin(obs[:, k], keep) & ok[:, k]
                    ci = contrast(vals, ok, excl, hi_b, lo_b, sel)[:, :, k]
                    with np.errstate(invalid="ignore"):
                        pmk = np.nanmean(ci, 1)
                    mn_l.append(pmk.mean())
                    tt_l.append(abs(pmk.mean()) / (pmk.std(ddof=1) / np.sqrt(n_seeds)))
                    ag_l.append(bool((np.sign(pmk) == np.sign(pmk[0])).all()))
                mn_l, tt_l, ag_l = np.array(mn_l), np.array(tt_l), np.array(ag_l)
                print(f"  {lab:>14}: controls |t| median {np.median(tt_l[ctrl]):.2f}, "
                      f"agree {ag_l[ctrl].mean():.0%}   " + "   ".join(
                          f"{names[k]} mean {mn_l[k]:+.5f} |t| {tt_l[k]:.2f} "
                          f"exceeds {(tt_l[ctrl]<tt_l[k]).mean():.0%}" for k in tgt))


if __name__ == "__main__":
    main()
