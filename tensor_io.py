#!/usr/bin/env python
"""
tensor_io.py — the one definition of how a window is stored and read back.

Section 5 of the plan specifies nine channels per window position. Eight of the
nine are binary and one is derivable from the others, so storing all nine as
float16 costs 18 bytes per position to carry about 3 bits of information. At
802,035 candidates that is 28.8 GB, which exceeded the cluster quota.

STORED INSTEAD: one uint8 per position per window.

    bits 0-2   fill state   0 unfilled, 1 A, 2 C, 3 G, 4 T, 5 filled but not ACGT
    bit  3     a junction lies at this transcript position

1.6 GB for the same tensor. The nine channels are reconstructed in the data
loader from this byte plus the candidate's own start and end coordinates, which
are already in the pool table.

"Filled but not ACGT" is a distinct state, not the same as unfilled: an N inside
the window counts toward the GC denominator and carries a reading frame, while a
position outside the window or past the midpoint clip carries neither. Collapsing
the two would shift channel 5 at every window holding an N.

encode() and decode() are the same function in two directions and are checked
against each other rather than against a description of them.
"""

import numpy as np

N_CHANNELS = 9
GC_SPAN = 50                       # channel 5, centred; half-width 25 either side
UNFILLED, OTHER_BASE = 0, 5
JUNCTION_BIT = 8

# fill state 1..4 -> channel 0..3
BASE_TO_STATE = np.zeros(256, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    BASE_TO_STATE[ord(_b)] = _i + 1


def encode_window_codes(seq_bytes, junc_pos, tx_len, anchor, left, right,
                        fill_lo, fill_hi):
    """One window as uint8 codes.

    `anchor` is the 1-based transcript position landing at array index `left`.
    Array index k holds transcript position `anchor - left + k`. A position is
    filled only when it lies inside the transcript AND inside [fill_lo, fill_hi].
    """
    W = left + right
    out = np.zeros(W, dtype=np.uint8)
    lo, hi = max(1, fill_lo), min(tx_len, fill_hi)
    if lo > hi:
        return out
    k0 = max(0, lo - anchor + left)
    k1 = min(W, hi - anchor + left + 1)
    if k0 >= k1:
        return out
    pos = np.arange(anchor - left + k0, anchor - left + k1, dtype=np.int64)
    state = BASE_TO_STATE[seq_bytes[pos - 1]]
    state[state == 0] = OTHER_BASE          # filled, but not one of ACGT
    out[k0:k1] = state
    if len(junc_pos):
        out[k0:k1] |= np.isin(pos, junc_pos).astype(np.uint8) * JUNCTION_BIT
    return out


def gc_terms(fill):
    """Channel 5's numerator, denominator and validity, before the division.

    `fill` is (n, W) integer fill state, 0-5. Returns (num, den, ok), each (n, W).

    Both sums count 0/1 over at most W = 1000 positions, so every entry of `num`
    and `den` is an exact integer in float32 -- well inside the 2^24 where
    float32 holds integers exactly. That is what lets a caller that knows how a
    substitution moved the GC count patch `num` by +/-1 and divide, and get the
    SAME float32 result as recomputing the cumulative sums from scratch: both
    forms divide one exact integer by another, so both round identically.

    The bounds are clipped to each row's own filled run, so an edge is not
    diluted by positions the encoder never averaged over.
    """
    filled = fill > 0
    n, W = fill.shape
    is_gc = ((fill == 2) | (fill == 3)).astype(np.float32)   # C or G
    fmask = filled.astype(np.float32)
    cg = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(is_gc, axis=1)], 1)
    cn = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(fmask, axis=1)], 1)
    half = GC_SPAN // 2
    first = np.argmax(filled, axis=1)
    cnt = filled.sum(axis=1)
    last = first + cnt                                        # exclusive
    k = np.arange(W)[None, :]
    a = np.clip(k - half, first[:, None], last[:, None])
    b = np.clip(k + half + 1, first[:, None], last[:, None])
    ridx = np.arange(n)[:, None]
    num = cg[ridx, b] - cg[ridx, a]
    den = cn[ridx, b] - cn[ridx, a]
    return num, den, (den > 0) & filled


def decode_windows(codes, anchor, left, orf_start):
    """Reconstruct the nine channels from stored codes.

    codes:      (n, W) uint8
    anchor:     (n,)   1-based transcript position sitting at index `left`
    orf_start:  (n,)   1-based start of the candidate, for the reading frame

    Returns (n, 9, W) float32, identical to what the nine-channel encoder wrote.
    """
    codes = np.asarray(codes)
    n, W = codes.shape
    fill = (codes & 7).astype(np.int16)
    junc = ((codes >> 3) & 1).astype(np.float32)
    filled = fill > 0

    out = np.zeros((n, N_CHANNELS, W), dtype=np.float32)

    # channels 0-3, one-hot; state 5 leaves all four at zero, as an N should
    r, c = np.nonzero((fill >= 1) & (fill <= 4))
    out[r, fill[r, c] - 1, c] = 1.0

    # channel 4
    out[:, 4, :] = junc

    # channel 5: rolling GC across the FILLED range only, with padding counted as
    # absent rather than as GC-free. The filled region is one contiguous run, so
    # the running sums are taken over it and written back in place.
    num, den, ok = gc_terms(fill)
    out[:, 5, :] = np.where(ok, num / np.maximum(den, 1), 0.0)
    k = np.arange(W)[None, :]

    # channels 6-8, codon position relative to this candidate's own start codon.
    #
    # frame[r, c] = (anchor[r] - left + c - orf_start[r]) mod 3
    #             = (off[r] + c) mod 3        with off[r] = (anchor[r]-left-orf_start[r]) mod 3
    #
    # so there are only THREE distinct row patterns, not n. The previous form
    # built an (n, W) int64 `pos` and an (n, W) `frame` for every chunk and was
    # half the decode time -- and for the ATG window, where anchor IS orf_start,
    # every row's pattern is identical.
    off = np.mod(anchor - left - orf_start, 3)
    base3 = np.mod(np.arange(3)[:, None] + k, 3)          # (3, W)
    frame = base3[off]                                     # (n, W), a gather
    rr, cc = np.nonzero(filled)
    out[rr, 6 + frame[rr, cc], cc] = 1.0
    return out
