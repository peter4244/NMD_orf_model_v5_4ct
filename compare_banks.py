#!/usr/bin/env python
"""
compare_banks.py — step 3 of the verification bar: two banks, entry by entry.

Compares every dataset and every attribute of two bank files. Used to show that a
change to how the model's input is built changed nothing about the bank: the
inputs were proved bitwise equal upstream, so the outputs must be too, and a
difference here is a defect rather than a tolerance to widen.

NaN is compared as a value, not skipped. `vals` is NaN wherever a position is
invalid or holds the observed base, and those NaNs are load-bearing -- a build
that silently turned one into a number, or a number into a NaN, would pass any
comparison that dropped them.

    python compare_banks.py a.h5 b.h5
"""

import sys

import h5py
import numpy as np


def same(a, b):
    """Bitwise equality, with NaN equal to NaN and dtype/shape part of the test."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
    if a.dtype.kind == "f":
        eq = (a == b) | (np.isnan(a) & np.isnan(b))
        if eq.all():
            return True, ""
        bad = int((~eq).sum())
        d = np.abs(a[~eq].astype(np.float64) - b[~eq].astype(np.float64))
        d = d[np.isfinite(d)]
        return False, (f"{bad:,} of {a.size:,} entries differ, "
                       f"max |diff| {d.max() if len(d) else float('nan'):.6g}, "
                       f"NaN pattern {'differs' if (np.isnan(a) != np.isnan(b)).any() else 'matches'}")
    if np.array_equal(a, b):
        return True, ""
    return False, f"{int((a != b).sum()):,} of {a.size:,} entries differ"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    n_ok = n_bad = 0
    with h5py.File(sys.argv[1], "r") as fa, h5py.File(sys.argv[2], "r") as fb:
        keys = sorted(set(fa.keys()) | set(fb.keys()))
        missing = [k for k in keys if k not in fa or k not in fb]
        for k in missing:
            print(f"  MISSING  {k}: only in {sys.argv[1] if k in fa else sys.argv[2]}")
            n_bad += 1
        for k in [k for k in keys if k not in missing]:
            ok, why = same(fa[k][:], fb[k][:])
            if ok:
                n_ok += 1
            else:
                n_bad += 1
                print(f"  DIFFERS  {k}: {why}")
        for k in sorted(set(fa.attrs) | set(fb.attrs)):
            va, vb = fa.attrs.get(k), fb.attrs.get(k)
            # the checkpoint path, the run's own timing and the output path differ
            # by construction between two builds and say nothing about the arrays
            if k in ("built_at", "out", "seconds", "rows_per_s", "peak_gb", "host",
                     "device", "chunk_rows"):
                continue
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                eq = np.array_equal(va, vb)
            else:
                eq = va == vb
            if not eq:
                n_bad += 1
                print(f"  DIFFERS  attr {k}: {va!r} vs {vb!r}")
    print(f"\ndatasets identical: {n_ok}")
    print(f"datasets or attributes differing: {n_bad}")
    if n_bad:
        print("FAIL — the two banks are not the same bank", file=sys.stderr)
        return 1
    print("PASS — every array and attribute is bitwise identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
