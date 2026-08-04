#!/usr/bin/env python
"""
verify_window_cache.py — step 1 of the bank's verification bar.

`WindowCache.windows` must return EXACTLY what `decode_windows` returns for the
same perturbed codes. Not close: the two compute the same divisions of the same
exact integers, so any difference at all is a defect, and the whole bank is
computed from this one function.

Run it on the hardware the bank will be built on. The patched path runs on the
device and the reference path runs in numpy, so device float behaviour is part of
what is under test, and a laptop cannot answer for a V100.

    python verify_window_cache.py --tensor results_tensor_v6 --device cuda --n-cand 4000
"""

import argparse
import sys

import h5py
import numpy as np
import torch

from tensor_io import decode_windows
from window_cache import WindowCache

ATG_LEFT, STOP_LEFT, WINDOW = 900, 500, 1000


def probe_indices(fill_row, rng, n_random):
    """Window indices to substitute at: every awkward one, then a random sample.

    The span of channel 5 clips against the filled run and against the array
    bounds, so the first and last filled positions and the positions one span in
    from each are where a patch that got the arithmetic right can still get the
    bounds wrong. They are tested every time rather than sampled.
    """
    ok = np.flatnonzero((fill_row >= 1) & (fill_row <= 4))
    if not len(ok):
        return ok
    first, last = ok[0], ok[-1]
    edges = [first, last, first + 25, last - 25, first + 24, last - 24,
             first + 26, last - 26, 0, WINDOW - 1, 25, WINDOW - 26]
    edges = [i for i in edges if i in set(ok.tolist())]
    pool = ok if len(ok) <= n_random else rng.choice(ok, n_random, replace=False)
    return np.unique(np.concatenate([np.array(edges, dtype=np.int64), pool]))


def check(tag, codes, anchor, left, orf_start, device, rng, n_random, batch):
    """Compare patched against decoded over every (index, base) of these windows."""
    K = len(orf_start)
    cache = WindowCache(codes, anchor, left, orf_start, device)
    fill = (codes & 7).astype(np.int16)

    rc, rw, rb = [], [], []
    for k in range(K):
        idx = probe_indices(fill[k], rng, n_random)
        for b in range(1, 5):
            rc.append(np.full(len(idx), k)); rw.append(idx); rb.append(np.full(len(idx), b))
    if not rc:
        return 0, 0.0, 0, cache.n_gc_patched, cache.n_gc_skipped
    rc = np.concatenate(rc); rw = np.concatenate(rw); rb = np.concatenate(rb)

    n_bad, worst, n_rows = 0, 0.0, len(rc)
    for s in range(0, n_rows, batch):
        e = min(s + batch, n_rows)
        c, w, b = rc[s:e], rw[s:e], rb[s:e]

        # the reference: build the perturbed codes and decode them whole, exactly
        # as build_ism_bank did before the cache existed
        pc = codes[c].copy()
        pc[np.arange(len(pc)), w] = (pc[np.arange(len(pc)), w] & 8) | b
        ref = decode_windows(pc, anchor[c], left, orf_start[c])

        got = cache.windows(torch.as_tensor(c, dtype=torch.long, device=device),
                            torch.as_tensor(w, dtype=torch.long, device=device),
                            torch.as_tensor(b, dtype=torch.long, device=device))
        got = got.cpu().numpy()

        diff = got != ref
        if diff.any():
            n_bad += int(diff.any(axis=(1, 2)).sum())
            worst = max(worst, float(np.abs(got[diff] - ref[diff]).max()))
            r0, c0, p0 = np.argwhere(diff)[0]
            print(f"    {tag}: MISMATCH row {s + r0} channel {c0} position {p0}: "
                  f"got {got[r0, c0, p0]!r} ref {ref[r0, c0, p0]!r}", file=sys.stderr)
    return n_rows, worst, n_bad, cache.n_gc_patched, cache.n_gc_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="results_tensor_v6")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-cand", type=int, default=4000, help="candidates to test")
    ap.add_argument("--n-random", type=int, default=6, help="random indices per window")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()

    device = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    with h5py.File(f"{a.tensor}/nmd_tensor.h5", "r") as f:
        n_tx = len(f["offset"])
        offset, count = f["offset"][:], f["count"][:]
        # whole transcripts, so candidate sets are real ones rather than a mixture
        order = rng.permutation(n_tx)
        take, total = [], 0
        for t in order:
            take.append(t); total += int(count[t])
            if total >= a.n_cand:
                break
        take = np.array(sorted(take))
        print(f"testing {total} candidates over {len(take)} transcripts, device {device}")

        n_rows = n_bad = n_gc = n_nogc = 0
        worst = 0.0
        for t in take:
            lo, hi = int(offset[t]), int(offset[t]) + int(count[t])
            codes = f["codes"][lo:hi]
            orf_start = f["orf_start"][lo:hi].astype(np.int64)
            orf_end = f["orf_end"][lo:hi].astype(np.int64)
            for tag, cd, anchor, left in (
                    ("atg", codes[:, 0], orf_start, ATG_LEFT),
                    ("stop", codes[:, 1], orf_end - 1, STOP_LEFT)):
                r, w, bad, gc, nogc = check(tag, cd, anchor, left, orf_start,
                                            device, rng, a.n_random, a.batch)
                n_rows += r; n_bad += bad; worst = max(worst, w)
                n_gc += gc; n_nogc += nogc

    print(f"substitutions compared: {n_rows:,}")
    print(f"  ...that moved the local GC count, taking the span patch: {n_gc:,}")
    print(f"  ...that did not, skipping it:                            {n_nogc:,}")
    print(f"rows differing from decode_windows: {n_bad}")
    print(f"largest absolute difference: {worst!r}")
    if n_bad or worst != 0.0:
        print("FAIL — the patched path is not the decoded path", file=sys.stderr)
        return 1
    # Both branches must have run. A pass drawn entirely from substitutions that
    # leave GC unchanged would never touch channel 5 and would say nothing about
    # the part of the patch that is not a single index.
    if not n_gc or not n_nogc:
        print(f"FAIL — vacuous: gc-patched {n_gc}, gc-skipped {n_nogc}", file=sys.stderr)
        return 1
    print("PASS — bitwise equal on every substitution tested, both branches exercised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
