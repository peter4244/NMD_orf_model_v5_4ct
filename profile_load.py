#!/usr/bin/env python
"""Where does the batch time go? Prefetching only helps if load overlaps compute;
it does not help if load IS the step. Split the load into its parts first."""
import sys, time, numpy as np
sys.path.insert(0, ".")
from train_v6 import TensorSource, make_batches, INTERPRETABLE
from tensor_io import decode_windows

src = TensorSource("results_tensor_v6/nmd_tensor.h5", INTERPRETABLE)
tr = src.indices("train")
b = make_batches(src.count[tr], 2048)
print(f"{len(b):,} batches/epoch, {src.count[tr].sum():,} candidates\n")

n = 15
t_read = t_dec = t_pad = 0.0
for bb in b[:n]:
    idxs = tr[bb]
    counts = src.count[idxs]
    rows = np.concatenate([np.arange(src.offset[i], src.offset[i] + src.count[i])
                           for i in idxs])
    srt = np.argsort(rows, kind="stable"); inv = np.argsort(srt, kind="stable")
    r = rows[srt]
    t0 = time.time(); codes = src.f["codes"][r][inv]; t_read += time.time() - t0
    s0, e0 = src.orf_start[rows], src.orf_end[rows]
    t0 = time.time()
    atg = decode_windows(codes[:, 0], s0, src.atg_left, s0)
    stop = decode_windows(codes[:, 1], e0 - 1, src.stop_left, s0)
    t_dec += time.time() - t0
    t0 = time.time()
    K = int(counts.max()); m = len(idxs)
    A = np.zeros((m, K, 9, src.window), np.float32); S = np.zeros_like(A)
    at = 0
    for j, c in enumerate(counts):
        c = int(c); A[j, :c] = atg[at:at+c]; S[j, :c] = stop[at:at+c]; at += c
    t_pad += time.time() - t0
for nm, v in (("h5 read", t_read), ("decode", t_dec), ("pad into batch", t_pad)):
    print(f"  {nm:<16} {v/n:.3f} s/batch  {v/(t_read+t_dec+t_pad)*100:>5.1f}%"
          f"   epoch {len(b)*v/n/60:>5.1f} min")
print(f"\n  total load {(t_read+t_dec+t_pad)/n:.3f} s/batch")
