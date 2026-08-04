#!/usr/bin/env python
"""
window_cache.py — decode each candidate's window once, then substitute by patching.

`decode_windows` rebuilds all 1000 positions of a window from scratch: cumulative
sums for channel 5, a one-hot scatter for channels 0-3, a frame gather for 6-8. In
the mutagenesis bank it is called once per (substitution, covering window), and on a
V100 it was 88.2% of the run -- the encoders, on the GPU, cost 4.4%.

A single base substitution changes about 51 of those 1000 positions:

    channels 0-3   one index -- the substituted base itself
    channel  4     NOTHING. A junction is annotation; the substitution does not move it.
    channel  5     the +/-25 span over which the rolling GC mean sees the changed base
    channels 6-8   NOTHING. The frame grid is anchored on the candidate's own start,
                   which the substitution does not move, and the fill mask is
                   unchanged because a base is replaced by a base.

So the window is decoded once per candidate and a substitution is a copy plus a
patch of that span, which converts the dominant cost from arithmetic to a memory
copy -- and, done on the device the encoders already run on, removes the
host-to-device transfer of the perturbed windows as well.

WHY THE PATCH IS BITWISE EQUAL, not merely close.

Channel 5 is `num / den` where both are counts over at most 1000 positions, so both
are exact integers in float32 (`tensor_io.gc_terms`). The substitution moves the GC
count over the span by exactly -1, 0 or +1 and leaves the denominator alone, because
the set of filled positions does not change. `(num + delta) / den` is therefore the
same division of the same two exact integers that a full recompute would perform,
and IEEE-754 rounds it identically. The three possible numerators are precomputed
per candidate, so the hot path is a gather and a scatter with no arithmetic at all.

This is checked rather than argued: `verify_window_cache.py` compares the patched
output against `decode_windows` on real windows, and requires equality.
"""

import numpy as np
import torch

from tensor_io import GC_SPAN, decode_windows, gc_terms

HALF = GC_SPAN // 2


class WindowCache:
    """The decoded windows of one transcript's candidates, patchable in place.

    Built once per (transcript, window kind). `base` is exactly what
    `decode_windows` returns for the unperturbed codes, so a caller that needs the
    unperturbed pass uses it directly rather than decoding twice.
    """

    def __init__(self, codes, anchor, left, orf_start, device):
        codes = np.asarray(codes)
        K, W = codes.shape
        fill = (codes & 7).astype(np.int16)

        base = decode_windows(codes, anchor, left, orf_start)      # (K, 9, W)
        num, den, ok = gc_terms(fill)

        # channel 5 under a GC count of num-1, num, num+1 -- the only three values
        # a single substitution can produce at any position of the span.
        gc = np.zeros((3, K, W), dtype=np.float32)
        d = np.maximum(den, 1)
        for t, delta in enumerate((-1.0, 0.0, 1.0)):
            gc[t] = np.where(ok, (num + delta) / d, 0.0)

        # The delta-0 table IS channel 5 of the unperturbed decode. If this ever
        # fails, the two are computing channel 5 differently and every patched
        # window is wrong in a way no downstream check would attribute here.
        assert np.array_equal(gc[1], base[:, 5, :]), \
            "GC table disagrees with decode_windows on the unperturbed window"

        self.W = W
        self.device = device
        self.base = torch.as_tensor(base, device=device)
        self.gc = torch.as_tensor(gc, device=device)
        self.state = torch.as_tensor(fill.astype(np.int64), device=device)
        self.offsets = torch.arange(-HALF, HALF + 1, device=device)
        # How many rows took the channel-5 span patch, and how many did not. A
        # substitution that does not change GC status skips that branch entirely,
        # so a check that happened to draw only such rows would pass without ever
        # testing the hard part. Counted rather than assumed; verify_window_cache
        # reports both and fails if either is zero.
        self.n_gc_patched = 0
        self.n_gc_skipped = 0

    def windows(self, cand, widx, new_state):
        """The (n, 9, W) input for n substitutions, on `self.device`.

        cand       (n,) long, which candidate's window each row perturbs
        widx       (n,) long, the index within that window
        new_state  (n,) long, the substituted base as a fill state, 1-4 = ACGT

        A row whose new base equals the observed one returns the cached window
        unchanged, which is what makes the same-chunk no-op baseline exact.
        """
        n = cand.numel()
        rows = torch.arange(n, device=self.device)
        old = self.state[cand, widx]

        # The one-hot write below indexes channel `state - 1`, so a state outside
        # 1-4 would silently write into channel 4, the junction channel. Callers
        # restrict substitutions to positions whose observed base is one of ACGT;
        # this is that precondition made to fail loudly instead of corrupting a
        # channel nothing downstream would trace back here.
        assert bool(((old >= 1) & (old <= 4)).all()), "observed base is not ACGT"
        assert bool(((new_state >= 1) & (new_state <= 4)).all()), "substituted base is not ACGT"

        x = self.base.index_select(0, cand)               # (n, 9, W), a copy
        x[rows, old - 1, widx] = 0.0
        x[rows, new_state - 1, widx] = 1.0                # order matters if old == new

        is_gc_old = ((old == 2) | (old == 3)).long()
        is_gc_new = ((new_state == 2) | (new_state == 3)).long()
        delta = is_gc_new - is_gc_old                     # -1, 0, +1
        moved = delta != 0
        n_moved = int(moved.sum())
        self.n_gc_patched += n_moved
        self.n_gc_skipped += n - n_moved
        if n_moved:
            jm, cm = rows[moved], cand[moved]
            tm = delta[moved] + 1                         # index into self.gc
            cols = widx[moved][:, None] + self.offsets[None, :]        # (m, 51)
            inside = (cols >= 0) & (cols < self.W)
            colc = cols.clamp(0, self.W - 1)
            val = self.gc[tm[:, None].expand_as(cols),
                          cm[:, None].expand_as(cols), colc]
            jj = jm[:, None].expand_as(cols)
            x[jj[inside], 5, colc[inside]] = val[inside]
        return x
