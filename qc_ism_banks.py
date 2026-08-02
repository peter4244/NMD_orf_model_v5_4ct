#!/usr/bin/env python
"""
qc_ism_banks.py — is the bank complete, and is any of it readable above its floor?

A finished SLURM job and a written file are not evidence that a bank is usable.
This asks the questions the interpretation will ask first, before anyone builds an
analysis on top of an answer that was never there.

  COMPLETENESS   every transcript present, nothing all-NaN, one chunk shape
  THE FLOOR      what fraction of each response clears the offset the pipeline
                 itself reports, per arm
  EXPRESSIBILITY what fraction of positions sit under candidates carrying enough
                 selection mass to move the output at all
  SEEDS          do the five agree, position by position

THE FLOOR IS THE POINT. `vals_capture` has a far smaller dynamic range than
`vals` or `vals_decay` -- measured elsewhere at a median near 9e-05 against 1e-02
-- while the batch-shape offset is the same size for all three. So the capture arm
can be almost entirely floor while the decay arm is almost entirely signal, and a
positional profile computed over both without conditioning would be a profile of
the decay arm wearing the capture arm's name.

CROSS-SEED AGREEMENT IS THE OTHER POINT. Seeds differ in initialisation only, so a
feature in one and not the others is a property of that initialisation. The §8.5
pair statistic ran 0.528 to 0.595 across these same five seeds -- a spread of 0.067
on an effect of 0.080 -- so capture-arm findings should be assumed seed-dependent
until shown otherwise.

    python qc_ism_banks.py results_ism_v6/bank_interp_s*.h5
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def arm_stats(v, floor):
    """Median |effect| and the share clearing the floor and ten times it."""
    a = np.abs(v[np.isfinite(v)])
    if not a.size:
        return dict(n=0, median=np.nan, above=np.nan, above10=np.nan)
    return dict(n=int(a.size), median=float(np.median(a)),
                above=float((a > floor).mean()),
                above10=float((a > 10 * floor).mean()))


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    banks, problems = {}, []
    for p in paths:
        if not p.exists():
            problems.append(f"{p.name}: MISSING")
            continue
        with h5py.File(p, "r") as f:
            n = f["vals"].shape[0]
            floor = float(f.attrs.get("batch_shape_offset", np.nan))
            valid = f["valid"][:]
            vals = f["vals"][:]
            cap = f["vals_capture"][:] if "vals_capture" in f else None
            dec = f["vals_decay"][:] if "vals_decay" in f else None
            mass = f["mass"][:] if "mass" in f else None
            crows = f["chunk_rows"][:] if "chunk_rows" in f else None
            print(f"\n=== {p.name} ===")
            print(f"  transcripts {n:,}   valid positions {int(valid.sum()):,}   "
                  f"reported floor {floor:.3e}")

            # completeness: a transcript with no finite response is a build that
            # produced a row rather than a result
            per_tx = np.isfinite(vals).reshape(n, -1).sum(1)
            dead = int((per_tx == 0).sum())
            print(f"  transcripts with no finite vals: {dead}"
                  f"{'  <-- PROBLEM' if dead else ''}")
            if dead:
                problems.append(f"{p.name}: {dead} transcripts have no finite vals")

            # one chunk shape, or the floor varies by build order
            if crows is not None:
                u = np.unique(crows)
                print(f"  chunk_rows across transcripts: {u.tolist()}"
                      f"{'  <-- MIXED' if len(u) > 1 else ''}")
                if len(u) > 1:
                    problems.append(f"{p.name}: built at mixed chunk shapes {u.tolist()}")
            else:
                print("  chunk_rows: not recorded (shard predates the field)")

            print(f"  {'arm':<14} {'n finite':>12} {'median |eff|':>14} "
                  f"{'>floor':>8} {'>10x':>8}")
            for name, arr in (("vals", vals), ("vals_capture", cap),
                              ("vals_decay", dec)):
                if arr is None:
                    continue
                s = arm_stats(arr, floor)
                print(f"  {name:<14} {s['n']:>12,} {s['median']:>14.3e} "
                      f"{100*s['above']:>7.1f}% {100*s['above10']:>7.1f}%")
                if name == "vals_capture" and s["above"] < 0.5:
                    problems.append(
                        f"{p.name}: only {100*s['above']:.1f}% of vals_capture "
                        f"clears the floor — the capture arm is mostly floor")

            # expressibility: a position whose covering candidates carry no
            # selection mass cannot move the output however its base changes
            if mass is not None:
                m = mass[valid]
                for thr in (1e-4, 1e-2):
                    print(f"  positions with selection mass < {thr:g}: "
                          f"{100*float((m < thr).mean()):.1f}%")
            banks[p.name] = dict(vals=vals, cap=cap, valid=valid)

    # cross-seed agreement, on the entries every seed calls finite
    names = sorted(banks)
    if len(names) > 1:
        print(f"\n=== cross-seed agreement, {len(names)} seeds ===")
        shp = {banks[k]["vals"].shape for k in names}
        if len(shp) > 1:
            problems.append(f"seeds have different shapes: {shp}")
            print(f"  shapes differ: {shp}  <-- cannot compare")
        else:
            for tag in ("vals", "cap"):
                stk = np.stack([banks[k][tag] for k in names])
                ok = np.isfinite(stk).all(0)
                if not ok.any():
                    continue
                x = stk[:, ok]
                # sign agreement: how often all five seeds move the same way
                sgn = np.sign(x)
                unanimous = float((np.abs(sgn.sum(0)) == len(names)).mean())
                # correlation of seed 1 against the mean of the rest
                r = float(np.corrcoef(x[0], x[1:].mean(0))[0, 1])
                print(f"  {tag:<6} entries compared {int(ok.sum()):,}   "
                      f"all seeds same sign {100*unanimous:>5.1f}%   "
                      f"seed1 vs mean of rest r={r:.3f}")

    print(f"\n{'=' * 60}")
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for q in problems:
            print(f"  - {q}")
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
