#!/usr/bin/env python
"""verify_packed_roundtrip.py — does the packed uint8 representation decode BIT-IDENTICALLY
back to the nine dense float16 channels this repo currently writes?

WHY THIS TEST AND NOT verify_tensor_encoding.py. That one decodes candidates and checks them
against the TRANSCRIPT SEQUENCE -- seven good checks, none of which is this question. The
author's note "verified to decode identically" is a claim about build_tensor.py's own tensor,
not about ours. Section 5's numbers were computed from the dense arrays, so the only claim that
licenses replacing them is: decoded == dense, element for element, on OUR data.

WHAT MAKES IT POSSIBLE AT ALL. Channel 5 is a rolling GC, and if it were computed over the
whole transcript it would depend on bases outside the window and could NOT be recovered from a
byte-per-position. It is not: data_prep.encode_window_v5 computes it from
`seq_uint8[w_start:w_end]` -- the filled region only, deliberately, so the channel means "GC of
the sequence this head can see". So every channel is a function of the filled bases plus the
candidate's own coordinates.

THE ENCODING (tensor_io.py's scheme, applied to this layout):
    bits 0-2  fill state: 0 unfilled, 1 A, 2 C, 3 G, 4 T, 5 filled but not ACGT
    bit  3    a junction lies at this position
Plus one int8 per (transcript, ORF, window): the codon phase at the first filled position,
which is what channels 6-8 need and what a byte-per-position cannot carry.

State 5 matters and is not decoration: a padded position and an N base are both "no one-hot",
but the N is inside the filled span, so it contributes to the GC denominator and carries a
frame. Collapsing them would corrupt both channel 5 and channels 6-8.
"""
import sys

import h5py
import numpy as np

from window_codec import pack_window, unpack_window

GC_ROLLING_WINDOW = 50


def compute_rolling_gc(seq_uint8, window=GC_ROLLING_WINDOW):
    """Byte-for-byte the same computation as data_prep.compute_rolling_gc."""
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


BASES = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)


def pack(dense):
    """dense (9, W) float16  ->  (codes (W,) uint8, phase int, filled_lo, filled_hi).

    The filled span is read from the channels rather than assumed: a position is filled if it
    carries a one-hot base OR a frame bit. That second clause is what catches an N.
    """
    onehot = dense[0:4]
    has_base = onehot.sum(axis=0) > 0
    has_frame = dense[6:9].sum(axis=0) > 0
    filled = has_base | has_frame

    codes = np.zeros(dense.shape[1], dtype=np.uint8)
    idx = onehot.argmax(axis=0)
    codes[filled] = 5                       # filled but not ACGT
    codes[has_base] = (idx[has_base] + 1)   # 1..4 = A,C,G,T
    codes |= (dense[4] > 0).astype(np.uint8) << 3

    if not filled.any():
        return codes, 0, 0, 0
    lo = int(np.argmax(filled))
    hi = int(len(filled) - np.argmax(filled[::-1]))
    phase = int(np.argmax(dense[6:9, lo]))   # codon position at the first filled base
    return codes, phase, lo, hi


def unpack(codes, phase, lo, hi, n_ch=9):
    """The inverse. Must reproduce all nine channels exactly."""
    W = len(codes)
    out = np.zeros((n_ch, W), dtype=np.float16)
    if hi <= lo:
        return out
    state = codes & 0x07
    junc = (codes >> 3) & 0x01

    # channels 0-3
    is_base = (state >= 1) & (state <= 4)
    out[(state[is_base] - 1).astype(np.int64), np.nonzero(is_base)[0]] = 1.0
    # channel 4
    out[4] = junc.astype(np.float16)
    # channel 5 -- recomputed over the filled span, from the bases the byte carries
    seq = np.zeros(hi - lo, dtype=np.uint8)
    span_state = state[lo:hi]
    sb = (span_state >= 1) & (span_state <= 4)
    seq[sb] = BASES[span_state[sb] - 1]
    seq[~sb] = ord("N")
    out[5, lo:hi] = compute_rolling_gc(seq).astype(np.float16)
    # channels 6-8 -- frame anchored on the candidate's own start
    frames = (np.arange(hi - lo) + phase) % 3
    out[6 + frames, np.arange(lo, hi)] = 1.0
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/p.castaldi/cc/nmd_orf_model_v5_4ct/deprecated_2026-08-04/"
        "results_4ct_dn/nmd_orf_data.h5")
    n_tx = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    total = mismatch = 0
    worst = {}
    with h5py.File(src, "r") as f:
        groups = [k for k in f.keys() if k.startswith("w")]
        for g in sorted(groups):
            for which in ("atg_windows", "stop_windows"):
                arr = f[g][which][:n_tx]          # (n, K, 9, W) float16
                n, K = arr.shape[0], arr.shape[1]
                bad = 0
                for i in range(n):
                    for k in range(K):
                        d = arr[i, k]
                        if not d.any():
                            continue
                        codes, phase = pack_window(d)
                        r = unpack_window(codes, phase)
                        total += 1
                        if not np.array_equal(r, d):
                            bad += 1
                            if len(worst) < 3:
                                ch = [c for c in range(9)
                                      if not np.array_equal(r[c], d[c])]
                                worst[f"{g}/{which}[{i},{k}]"] = ch
                mismatch += bad
                print("  %-6s %-13s windows=%-6d mismatched=%d" % (g, which, n * K, bad))

    print()
    print("windows compared : %d" % total)
    print("bit-identical    : %d" % (total - mismatch))
    print("MISMATCHED       : %d" % mismatch)
    if worst:
        print("first mismatches (window -> differing channels):")
        for kk, ch in worst.items():
            print("   %s -> channels %s" % (kk, ch))
    print()
    print("RESULT: %s" % ("PASS - packing is lossless on this data"
                          if mismatch == 0 else "FAIL - do NOT adopt packing"))
    sys.exit(0 if mismatch == 0 else 1)


if __name__ == "__main__":
    main()
