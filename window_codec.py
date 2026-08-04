#!/usr/bin/env python
"""window_codec.py — the ONE definition of how a 9-channel window is stored and read back.

Nine channels per position stored as float16 costs 18 bytes to carry about 3 bits. Eight of
the nine are binary and the ninth is derivable, so the window is stored as ONE uint8 PER
POSITION and the channels are rebuilt on read. Measured on this repo's own data, a real
2,048-transcript slice of w500/atg_windows, 92.2 MB dense:

    9-channel float16 + lzf            13.05 MB    1.00x
    9-channel float16 + gzip4+shuffle   4.57 MB    2.86x
    packed uint8      + gzip4+shuffle   0.69 MB   19.02x

The scheme is tensor_io.py's, applied to this layout; the encoding idea is that file's, not
this one's.

    bits 0-2   fill state: 0 unfilled, 1 A, 2 C, 3 G, 4 T, 5 filled but not ACGT
    bit  3     a junction lies at this position

plus one int8 per window: the codon phase at the first filled position. A byte per position
cannot carry channels 6-8, because the frame grid is anchored on the candidate's own start,
which is not a property of any single position. The phase array is (n_tx, MAX_ORFS) int8 --
~418 KB against gigabytes, so it costs nothing to store what cannot be derived.

WHY THIS IS LOSSLESS HERE, WHICH IS NOT TRUE IN GENERAL. Channel 5 is a rolling GC. Had it
been computed over the whole transcript it would depend on bases outside the window and could
not be recovered from a byte-per-position. It is not: data_prep.encode_window_v5 computes it
from `seq_uint8[w_start:w_end]`, the filled region only, deliberately, so that the channel
means "GC of the sequence this head can see". Every channel is therefore a function of the
filled bases plus the candidate's own coordinates.

STATE 5 IS NOT DECORATION. A padded position and an N base both lack a one-hot, but the N sits
inside the filled span, so it counts in the GC denominator and carries a frame. Collapsing the
two would corrupt channel 5 and channels 6-8 together, and would do it silently.

VERIFIED, NOT ASSERTED. verify_packed_roundtrip.py packs and unpacks real windows and compares
against the dense arrays element for element: 10,240 windows over all four window groups and
both anchors, 0 mismatches. That test is the licence for replacing the dense arrays at all --
section 5's numbers were computed from them, so "decodes identically" had to be shown on THIS
data rather than inherited from build_tensor.py's claim about its own tensor.
"""
import numpy as np

GC_ROLLING_WINDOW = 50
N_SEQ_CHANNELS = 9
BASES = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)


def compute_rolling_gc(seq_uint8, window=GC_ROLLING_WINDOW):
    """Identical in form to data_prep.compute_rolling_gc, and it must stay that way: the
    decoded channel 5 is only bit-identical because the same arithmetic runs on the same
    bases. If one changes, both change."""
    seq_len = len(seq_uint8)
    if seq_len == 0:
        return np.zeros(0, dtype=np.float32)
    is_gc = ((seq_uint8 == ord("G")) | (seq_uint8 == ord("C"))).astype(np.float32)
    cumsum = np.concatenate([[0.0], np.cumsum(is_gc)])
    half = window // 2
    positions = np.arange(seq_len)
    lo = np.maximum(0, positions - half)
    hi = np.minimum(seq_len, positions + half + 1)
    return ((cumsum[hi] - cumsum[lo]) / (hi - lo)).astype(np.float32)


def pack_window(dense):
    """(9, W) float16 dense window -> (codes (W,) uint8, phase int8).

    The filled span is READ FROM THE CHANNELS rather than assumed: a position is filled if it
    carries a one-hot base OR a frame bit. The second clause is what catches an N, which has a
    frame but no one-hot.
    """
    onehot = dense[0:4]
    has_base = onehot.sum(axis=0) > 0
    filled = has_base | (dense[6:9].sum(axis=0) > 0)

    codes = np.zeros(dense.shape[1], dtype=np.uint8)
    codes[filled] = 5
    idx = onehot.argmax(axis=0)
    codes[has_base] = (idx[has_base] + 1).astype(np.uint8)
    codes |= (dense[4] > 0).astype(np.uint8) << 3

    if not filled.any():
        return codes, np.int8(-1)          # -1 = no filled positions
    lo = int(np.argmax(filled))
    return codes, np.int8(np.argmax(dense[6:9, lo]))


def unpack_window(codes, phase, n_ch=N_SEQ_CHANNELS, dtype=np.float16):
    """The exact inverse. phase < 0 means the window is entirely padding."""
    W = len(codes)
    out = np.zeros((n_ch, W), dtype=dtype)
    if phase < 0:
        return out
    state = codes & 0x07
    filled = state > 0
    if not filled.any():
        return out
    lo = int(np.argmax(filled))
    hi = int(W - np.argmax(filled[::-1]))

    is_base = (state >= 1) & (state <= 4)
    out[(state[is_base] - 1).astype(np.int64), np.nonzero(is_base)[0]] = 1.0
    out[4] = ((codes >> 3) & 0x01).astype(dtype)

    seq = np.full(hi - lo, ord("N"), dtype=np.uint8)
    span = state[lo:hi]
    sb = (span >= 1) & (span <= 4)
    seq[sb] = BASES[span[sb] - 1]
    out[5, lo:hi] = compute_rolling_gc(seq).astype(dtype)

    frames = (np.arange(hi - lo) + int(phase)) % 3
    out[6 + frames, np.arange(lo, hi)] = 1.0
    return out


def unpack_batch(codes, phases, n_ch=N_SEQ_CHANNELS, dtype=np.float32):
    """(n, K, W) uint8 + (n, K) int8 -> (n, K, 9, W). The read-side entry point."""
    n, K, W = codes.shape
    out = np.zeros((n, K, n_ch, W), dtype=dtype)
    for i in range(n):
        for k in range(K):
            if phases[i, k] >= 0:
                out[i, k] = unpack_window(codes[i, k], phases[i, k], n_ch, dtype)
    return out
