#!/usr/bin/env python3
"""verify_clip_in_h5.py — did the ORF-midpoint clip actually take effect in this HDF5?

WHY THIS EXISTS, and why it is not the obvious check. The tempting way to confirm the clip is to
re-run the branch decomposition and see whether 60.0/29.1/10.9 moved. That inference is backwards:
Track A's point, 2026-07-30, is that every section-5 deposit-native measurement in the ledger was
taken BEFORE the three data_prep fixes landed (2658732, 553d4c0, cf19dd6, all 2026-07-29), so those
numbers are PRE-CLIP BASELINES, not targets. Reproducing them exactly would be evidence the clip
did NOT take effect. Unexpected STASIS is the alarm, not movement.

So the clip should be assayed where it happens -- in the encoding -- not inferred from a downstream
percentage that also depends on training, seeding and the universe change.

WHAT IT MEASURES, and why it needs no anchor arrays. data_prep.py does not write atg_centers /
stop_centers into the HDF5, so transcript coordinates are not recoverable from the file. They are
not needed. Channels 6-8 (reading frame) are an EXACT validity mask -- non-zero on precisely the
filled range and zero on padding, including where channels 0-3 are all-zero because of an ambiguous
base (verified by perturbation across 24 window/ORF-length combinations, 2026-07-30). So the filled
WIDTH of each window is readable directly, and the clip has an arithmetic signature in it:

  anchor separation is L-3 for an ORF of length L, so
    both windows clip      (L-3 <  W):  atg_filled + stop_filled == W + (L-3)   -- STRICTLY < 2W
    neither window clips   (L-3 >= W):  atg_filled + stop_filled == 2W

  Pre-clip, every ORF with enough flanking sequence fills BOTH windows completely: 2W.

Hence: post-clip, a large mass of ORFs must sit strictly below 2W, and the mass grows with W because
more ORFs are shorter than the window. Pre-clip that mass is ~0 except at transcript edges. At
W=2000 the separation is stark, since most ORFs are far shorter than 2000 nt.

Two further necessary conditions, both anchor-free:
  * a CLIPPED atg window must not reach its own right edge, and a clipped stop window must not
    reach its own left edge -- the clip restricts the filled range and pads inward.
  * the filled range of each window must be CONTIGUOUS. A gap would mean something other than the
    clip is blanking positions.

    python3 verify_clip_in_h5.py results_4ct_dn/nmd_orf_data.h5
    python3 verify_clip_in_h5.py <h5> --window 2000 --max-tx 4000

Exit 0 = the clip is present in this HDF5. Exit 1 = it is not, or the file is internally
inconsistent. Run this BEFORE reading anything downstream of the encoding.
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

FRAME_CHANNELS = slice(6, 9)


def filled_mask(win):
    """(9, W) -> (W,) bool. Channels 6-8 are one-hot exactly on the filled range."""
    return np.asarray(win[FRAME_CHANNELS], dtype=np.float32).sum(axis=0) > 0


def contiguous(mask):
    idx = np.flatnonzero(mask)
    return idx.size == 0 or idx.size == (idx[-1] - idx[0] + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--window", type=int, default=2000,
                    help="window size group to read (default 2000: the clip's signature is "
                         "strongest where most ORFs are shorter than the window)")
    ap.add_argument("--max-tx", type=int, default=3000,
                    help="transcripts to sample (reads two W-wide arrays per ORF; keep modest)")
    a = ap.parse_args()

    W = a.window
    with h5py.File(a.h5, "r") as f:
        grp = f"w{W}"
        if grp not in f:
            sys.exit(f"no group {grp} in {a.h5}; present: {[k for k in f.keys()]}")
        n_tx = f["orf_mask"].shape[0]
        take = min(a.max_tx, n_tx)
        print(f"file      : {a.h5}")
        print(f"window    : {W}   transcripts: {take:,} of {n_tx:,}")

        mask = f["orf_mask"][:take]
        atg = f[grp]["atg_windows"][:take]
        stop = f[grp]["stop_windows"][:take]

    sums, atg_w, stop_w = [], [], []
    bad_contig = bad_edge = 0
    for i in range(mask.shape[0]):
        for k in range(mask.shape[1]):
            if not mask[i, k]:
                continue
            fa, fs = filled_mask(atg[i, k]), filled_mask(stop[i, k])
            na, ns = int(fa.sum()), int(fs.sum())
            if na == 0 and ns == 0:
                continue
            atg_w.append(na); stop_w.append(ns); sums.append(na + ns)
            if not contiguous(fa) or not contiguous(fs):
                bad_contig += 1
            # A window that is clipped must be padded on its INNER side. If the atg window is
            # short of full AND still reaches its right edge, something other than the clip
            # truncated it (transcript edge on the outer side is the legitimate alternative,
            # which shows up as the LEFT edge being unfilled -- so only flag the both-edges case).
            if na < W and fa[-1] and fa[0]:
                bad_edge += 1
            if ns < W and fs[0] and fs[-1]:
                bad_edge += 1

    if not sums:
        sys.exit("no valid ORFs found in the sample -- cannot assay")

    sums = np.array(sums)
    atg_w, stop_w = np.array(atg_w), np.array(stop_w)
    n = sums.size
    below = int((sums < 2 * W).sum())
    at_full = int((sums == 2 * W).sum())
    over = int((sums > 2 * W).sum())

    print(f"\nORFs assayed        : {n:,}")
    print(f"atg filled  (median): {int(np.median(atg_w)):,}  range {atg_w.min():,}-{atg_w.max():,}")
    print(f"stop filled (median): {int(np.median(stop_w)):,}  range {stop_w.min():,}-{stop_w.max():,}")
    print(f"atg+stop filled     : median {int(np.median(sums)):,}  (2W = {2*W:,})")
    print(f"  strictly below 2W : {below:,} ({100*below/n:.1f}%)   <- the clip's signature")
    print(f"  exactly 2W        : {at_full:,} ({100*at_full/n:.1f}%)  (ORFs with L-3 >= W)")
    print(f"  above 2W          : {over:,}   (must be 0)")

    fail = []
    if over:
        fail.append(f"{over} ORFs have more filled positions than 2W -- impossible; "
                    f"the validity mask or the reader is wrong")
    if bad_contig:
        fail.append(f"{bad_contig} windows have a NON-CONTIGUOUS filled range -- something other "
                    f"than the clip is blanking interior positions")
    if bad_edge:
        fail.append(f"{bad_edge} partially-filled windows still reach BOTH edges -- a clipped "
                    f"window must be padded on its inner side")
    # The decisive one. At W=2000 essentially every real ORF is shorter than the window, so a
    # pre-clip file sits at 2W almost everywhere.
    if below == 0:
        fail.append(f"NO ORF has fewer than 2W filled positions. This is a PRE-CLIP encoding: "
                    f"both windows are filling completely, i.e. they are reading shared bases. "
                    f"Do not read anything downstream of the encoding from this file.")
    elif below / n < 0.5 and W >= 1000:
        fail.append(f"only {100*below/n:.1f}% of ORFs are below 2W at W={W}. Expected a large "
                    f"majority, since most ORFs are far shorter than {W} nt. Check whether the "
                    f"clip is being applied to every ORF rather than some.")

    print("\n=== RESULT ===")
    if fail:
        for msg in fail:
            print(f"FAIL  {msg}")
        return 1
    print("PASS — the ORF-midpoint clip is present in this HDF5.")
    print("       Note what this does and does not establish: the two heads no longer receive "
          "overlapping\n       bases. It says nothing about whether any downstream number is "
          "correct, and the ledger's\n       section-5 deposit-native values are PRE-CLIP "
          "baselines -- reproducing them would be the alarm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
