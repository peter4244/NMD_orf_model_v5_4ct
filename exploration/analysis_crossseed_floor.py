#!/usr/bin/env python
"""
analysis_crossseed_floor.py — does cross-seed agreement survive conditioning on the floor?

Five seeds agree on the sign of a `vals` entry 26.8% of the time, against a chance
floor of 6.25% (two of 2^5). That is 4.3x chance and it is not high, and the
question it leaves open is whether the disagreement is concentrated in the entries
too small to be read.

If agreement rises sharply once every seed clears its own batch-shape offset, the
sub-floor bulk was doing the disagreeing and above-floor positional claims are
usable. If it does not rise, positional claims from these banks need the
discovery/confirmation arm before they mean anything, because the seeds are not
agreeing even where they can resolve.

THE CONDITION IS PER SEED, ON THAT SEED'S OWN FLOOR. The offsets differ across
banks -- 1.258e-06 to 1.873e-06 -- and using one seed's floor for another would be
the same error as scoring the capture arm over stop-only positions: a threshold
that means something different depending on which set it is applied to.

An entry is kept when EVERY seed clears its own floor there. That is the
conservative choice and it is stated because the alternatives are not equivalent:
requiring the mean to clear would keep entries where one seed is at noise, and
requiring any seed to clear would keep entries where four are.

CAPTURE IS SCORED ONLY WHERE CAPTURE CAN RESPOND. A substitution in a stop window
cannot reach z_p, so vals_capture is exactly zero there by construction and such an
entry can never agree on a sign. The ATG mask is geometry and geometry does not
depend on the seed, so all banks must agree on it -- asserted.

    python analysis_crossseed_floor.py results_ism_v6/bank_interp_s*.h5
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def atg_mask(f):
    """Positions inside at least one ATG window. Geometry, not model output."""
    valid = f["valid"][:]
    spans = f["spans"][:]
    c_off, c_cnt = f["cand_offset"][:], f["cand_count"][:]
    out = np.zeros_like(valid)
    for i in range(valid.shape[0]):
        for row in spans[int(c_off[i]):int(c_off[i]) + int(c_cnt[i])]:
            a_lo, a_hi = int(row[2]), int(row[3])
            if a_hi >= a_lo:
                out[i, max(0, a_lo - 1):a_hi] = True
    return out & valid


def agreement(stk, keep):
    """Unanimous-sign fraction and seed-1-vs-rest correlation over `keep`."""
    if not keep.any():
        return dict(n=0, unan=np.nan, r=np.nan)
    x = stk[:, keep]
    sgn = np.sign(x)
    unan = float((np.abs(sgn.sum(0)) == stk.shape[0]).mean())
    r = float(np.corrcoef(x[0], x[1:].mean(0))[0, 1]) if x.shape[1] > 1 else np.nan
    return dict(n=int(keep.sum()), unan=unan, r=r)


def main():
    paths = sorted(Path(p) for p in sys.argv[1:])
    if len(paths) < 2:
        print(__doc__)
        return 2
    S = len(paths)
    chance = 2 * (0.5 ** S)
    print(f"cross-seed agreement over {S} seeds; chance unanimity {100*chance:.2f}%")
    for p in paths:
        print(f"  {p.name}")

    floors, masks = [], None
    fs = [h5py.File(p, "r") for p in paths]
    for f in fs:
        floors.append(float(f.attrs["batch_shape_offset"]))
    print("\nper-seed batch-shape offset: " + "  ".join(f"{x:.3e}" for x in floors))

    m0 = atg_mask(fs[0])
    for f in fs[1:]:
        assert np.array_equal(m0, atg_mask(f)), \
            "banks disagree on ATG coverage — different candidate geometry"
    print(f"ATG coverage identical across seeds: {int(m0.sum()):,} positions\n")

    print(f"  {'arm':<14} {'condition':<26} {'n':>13} {'unanimous':>11} "
          f"{'x chance':>9} {'r':>7}")
    for arm in ("vals", "vals_decay", "vals_capture"):
        stk = np.stack([f[arm][:] for f in fs])
        finite = np.isfinite(stk).all(0)
        base = finite & (m0[..., None] if arm == "vals_capture" else True)
        # every seed clears ITS OWN floor at this entry
        clears = np.ones_like(base)
        for s in range(S):
            clears &= np.abs(stk[s]) > floors[s]
        for label, keep in (("all finite", base), ("all seeds clear floor", base & clears)):
            a = agreement(stk, keep)
            if not a["n"]:
                continue
            print(f"  {arm:<14} {label:<26} {a['n']:>13,} {100*a['unan']:>10.1f}% "
                  f"{a['unan']/chance:>8.1f}x {a['r']:>7.3f}")
        del stk, finite, base, clears
    for f in fs:
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
