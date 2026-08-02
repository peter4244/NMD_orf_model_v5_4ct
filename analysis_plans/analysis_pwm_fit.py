#!/usr/bin/env python
"""
analysis_pwm_fit.py — fit a PWM directly to the ISM scores.

Pete's call, 2026-08-02: PWM first, MoDISco second.

WHY THIS BEFORE SEQLET CLUSTERING. It needs neither a background set nor a
clustering step, which between them produced every error of the last day: regional
composition, the GC control that was itself GC-biased, the truncated depleted tail.
It estimates 4w numbers rather than grouping ~110,000 seqlets.

THE FORMULATION, AND WHY IT HAS A CLOSED FORM. A PWM match score is

    s(p) = sum_j W[j, seq[p+j]]

which is LINEAR in W once the window is one-hot encoded. So "find the PWM whose
match score best tracks ISM importance" is ordinary least squares of importance on
the one-hot window, and the fitted coefficients ARE the PSSM. No optimiser, no
initialisation, no local minima, and 4w+1 parameters for w=9 is 37 numbers.

NO FOREGROUND/BACKGROUND SPLIT ANYWHERE. The regression runs over EVERY valid
position, not over an elevated set against a comparison population. That is the
property being bought: there is no set to enumerate wrongly.

THE HELD-OUT TEST IS THE EVIDENCE. Fit on discovery genes, evaluate on
confirmation genes. The arms are disjoint by gene and no gene appears in both, so
a correlation on confirmation is a statement about genes the fit never saw. An
in-sample r would be guaranteed positive and would mean nothing.

TWO TARGETS, BECAUSE THE CRITERIA SELECT DIFFERENT POSITIONS. Measured on s100:
the unsigned top-1% and the signed top-1% overlap at Jaccard 0.524, so a third of
each is criterion-specific.

    unsigned   max_b |vals[b]|        sensitive in any direction
    signed     -mean_b vals[b]        MoDISco's actual contribution, hyp[obs]

Directionality |mean|/max runs 0.466 at elevated positions against a CEILING OF
0.75 (all three substitutions agreeing), so both kinds of feature are present.

WHAT A POOR FIT DOES NOT LICENSE. A single PWM assumes ONE motif; several motifs
give a blend indistinguishable from a bad fit. Pre-registered, before running:
**a poor fit licenses going to MoDISco, and does NOT license "no sequence
preference".** That is the underpowered negative that demoted MoDISco the first
time and it is not to be repeated in a new costume.

THE COMPOSITION FLOOR. The elevated set is keto-skewed -- G+T at 1.16x, A+C at
0.85x, GC flat at 1.00x, measured over 4,999 transcripts and 11,062,149 valid
positions (model_a2_enumeration.py) -- so a fitted column can be entirely
composition. Each
column's KL against background composition is reported beside it; the
composition-only bar is 0.016-0.018 bits, measured independently at 600 and 800
transcripts. Reported per column rather than as a verdict, because at the weak end
(a 35/25/25/15 column is only 3x the bar) multiplicity eats it.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NT = "ACGT"
KL_BAR = (0.0159, 0.0183)          # measured, 800 and 600 transcripts


def importance(vals, mode):
    """Per-position importance under the two seqlet criteria."""
    with np.errstate(invalid="ignore"):
        if mode == "unsigned":
            return np.nanmax(np.abs(vals), axis=1)
        # signed: hyp[obs] = -mean_b vals[b] with vals[obs] := 0
        return -np.where(np.isfinite(vals), vals, 0.0).mean(axis=1)


def accumulate(path, width, mode, arms, cap=None):
    """Stream X'X and X'y over every valid position. X is the one-hot window.

    Streamed rather than materialised: 11M positions x 4w features would be a
    dense matrix in the tens of gigabytes, and the normal equations are 4w+1
    square -- 37x37 at w=9 -- so nothing large ever exists.
    """
    half = width // 2
    d = 4 * width + 1
    XtX = np.zeros((d, d)); Xty = np.zeros(d); yty = 0.0; n = 0
    comp = np.zeros(4)
    with h5py.File(path, "r") as f:
        co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
        arm = np.array([s.decode() for s in f["arm"][:]])
        tx = [s.decode() for s in f["transcript_id"][:]]
        for i in range(len(tx)):
            if arm[i] not in arms:
                continue
            if cap and n > cap:
                break
            lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo + nk]
            P = int(max(b[:, 3].max(), b[:, 5].max()))
            if P < 50:
                continue
            v = f["vals_decay"][i, :P].astype(np.float64)
            o = f["obs"][i, :P]
            y = importance(v, mode)
            ok = f["valid"][i, :P] & np.isfinite(y) & (o >= 0)
            idx = np.flatnonzero(ok)
            idx = idx[(idx >= half) & (idx < P - half)]
            if not len(idx):
                continue
            win = np.stack([o[idx + j - half] for j in range(width)], 1)
            if (win < 0).any():
                keep = (win >= 0).all(1)
                idx, win = idx[keep], win[keep]
                if not len(idx):
                    continue
            X = np.zeros((len(idx), d))
            for j in range(width):
                X[np.arange(len(idx)), j * 4 + win[:, j]] = 1.0
            X[:, -1] = 1.0                       # intercept
            yy = y[idx]
            XtX += X.T @ X; Xty += X.T @ yy; yty += float(yy @ yy)
            n += len(idx)
            for base in range(4):
                comp[base] += (o[idx] == base).sum()
    return XtX, Xty, yty, n, comp


def kl_bits(col, bg):
    p = np.clip(col, 1e-9, None); p = p / p.sum()
    q = np.clip(bg, 1e-9, None); q = q / q.sum()
    return float((p * np.log2(p / q)).sum())


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--width", type=int, default=9)
    ap.add_argument("--mode", default="unsigned", choices=["unsigned", "signed"])
    ap.add_argument("--ridge", type=float, default=1e-6)
    args = ap.parse_args()

    print(f"bank   {args.bank}")
    print(f"width  {args.width}   target {args.mode}\n")

    XtX, Xty, yty, n_fit, comp = accumulate(args.bank, args.width,
                                            args.mode, {"discovery"})
    if n_fit == 0:
        sys.exit("no discovery positions")
    d = XtX.shape[0]
    # Ridge only to break the exact collinearity of a full one-hot per column
    # (each block sums to the intercept); it is not regularisation for its own
    # sake and is set far below any scale that would shrink a real coefficient.
    W = np.linalg.solve(XtX + args.ridge * np.eye(d), Xty)
    bg = comp / comp.sum()
    print(f"fit on DISCOVERY genes: {n_fit:,} positions")
    print(f"  background composition  " +
          "  ".join(f"{NT[j]} {bg[j]:.3f}" for j in range(4)))

    print(f"\n  the fitted PSSM, and each column's KL against that background")
    print(f"  (composition-only bar {KL_BAR[0]:.4f}-{KL_BAR[1]:.4f} bits, "
          f"measured at 800 and 600 tx)\n")
    print(f"    {'offset':>7} " + "".join(f"{NT[j]:>10}" for j in range(4))
          + f"{'KL bits':>10} {'x bar':>7}")
    half = args.width // 2
    for j in range(args.width):
        w = W[j * 4:(j + 1) * 4]
        # a PSSM column read as a preference: softmax to a distribution so KL is
        # defined, which is a monotone re-expression and changes no ordering
        e = np.exp(w - w.max()); p = e / e.sum()
        kl = kl_bits(p, bg)
        print(f"    {j - half:>+7} " + "".join(f"{w[b]:>10.3e}" for b in range(4))
              + f"{kl:>10.4f} {kl / KL_BAR[0]:>6.1f}x")

    # ---- HELD OUT. The only number that is evidence. ----------------------
    XtX2, Xty2, yty2, n_ho, _ = accumulate(args.bank, args.width, args.mode,
                                           {"confirmation"})
    # Pearson r on held-out genes from accumulated moments alone -- the design
    # matrix is never held. E[pred*y] = W'X'y/n, E[pred^2] = W'X'X W/n,
    # E[y^2] = y'y/n, and the intercept column gives both means.
    n2 = max(n_ho, 1)
    mean_pred = float(W @ XtX2[:, -1]) / n2
    mean_y = float(Xty2[-1]) / n2
    cov = float(W @ Xty2) / n2 - mean_pred * mean_y
    var_p = float(W @ XtX2 @ W) / n2 - mean_pred ** 2
    var_y = yty2 / n2 - mean_y ** 2
    r_ho = cov / np.sqrt(max(var_p, 1e-300) * max(var_y, 1e-300))
    # in-sample, for contrast only
    n1 = max(n_fit, 1)
    mp1 = float(W @ XtX[:, -1]) / n1; my1 = float(Xty[-1]) / n1
    r_in = (float(W @ Xty) / n1 - mp1 * my1) / np.sqrt(
        max(float(W @ XtX @ W) / n1 - mp1 ** 2, 1e-300)
        * max(yty / n1 - my1 ** 2, 1e-300))
    print(f"\n  HELD OUT on CONFIRMATION genes: {n_ho:,} positions, disjoint by gene")
    print(f"    r  in-sample (discovery)   {r_in:+.4f}")
    print(f"    r  HELD OUT (confirmation) {r_ho:+.4f}   <- the only number that is evidence")
    print(f"    variance of ISM importance explained on held-out genes: "
          f"{100*r_ho**2:.2f}%")
    print(f"\n  A POOR FIT LICENSES GOING TO MoDISco. It does NOT license "
          f"'no sequence\n  preference' -- a single PWM assumes one motif and "
          f"several give a blend\n  indistinguishable from a bad fit. "
          f"Pre-registered before running.")


if __name__ == "__main__":
    main()
