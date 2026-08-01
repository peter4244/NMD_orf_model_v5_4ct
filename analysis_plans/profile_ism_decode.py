#!/usr/bin/env python
"""
profile_ism_decode.py — where do the 260 substitutions per second go?

A twelve-fold change in batch size moved throughput from 257 to 260 to 258 while
GPU memory went 1.2 -> 4.8 -> 12.6 GB. The card is idle and waiting. This finds
out what it is waiting for, per chunk, on the real inner loop rather than on a
guess about it.

The candidates, in the order the chunk touches them:

  1  building the perturbed codes      numpy, one uint8 row per (pair, base)
  2  decode_windows                    numpy, uint8 -> (n, 9, 1000) float32
  3  host-to-device transfer           the decoded array
  4  the three encoder passes          GPU
  5  stick-breaking + aggregation      GPU, over K numbers
  6  writeback                         python loop over the chunk's positions

decode_windows is the suspect: it allocates an (n, 9, 1000) float32 for every
chunk, which at 4096 rows is 147 MB, and it computes a rolling GC by cumulative
sum over the full 1000-wide axis for every row -- including the ~26% of positions
that are padding and the six channels that are one-hot writes.

Run on CPU: the point is the CPU-side cost, and the GPU share is measured
separately by the chunk sweep already done.
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model_v6 import ScanningNMDModel          # noqa: E402
from tensor_io import decode_windows           # noqa: E402

ATG_LEFT, STOP_LEFT = 900, 500


def main():
    sys.stdout.reconfigure(line_buffering=True)
    tensor = REPO / "results_tensor_v6" / "nmd_tensor.h5"
    if not tensor.exists():
        tensor = REPO / "results_tensor_chr21" / "nmd_tensor.h5"
    with h5py.File(tensor, "r") as f:
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:], f["orf_end"][:]
        # a transcript of typical candidate count
        i = int(np.argmin(np.abs(cnt - 19)))
        sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
        codes = f["codes"][sl]
        s0 = o_s[sl].astype(np.int64)
    K = len(s0)
    print(f"tensor {tensor.parent.name}   transcript with K={K}\n")

    ck = torch.load(REPO / "results_interp_all" / "v6_checkpoints" / "b8_s100.pt",
                    map_location="cpu", weights_only=False)
    m = ScanningNMDModel(conv_channels=32, n_bins=8, n_structural=1)
    m.load_state_dict(ck["model"]); m.eval()

    for n_rows in (4096, 16384):
        print(f"=== chunk of {n_rows:,} rows ===")
        rc = np.random.default_rng(0).integers(0, K, n_rows)
        pc = codes[rc, 0].copy()
        idx = np.random.default_rng(1).integers(50, 950, n_rows)
        anchor = s0[rc]

        t = time.perf_counter()
        for _ in range(3):
            pc2 = codes[rc, 0].copy()
            pc2[np.arange(n_rows), idx] = (pc2[np.arange(n_rows), idx] & 8) | 2
        t_build = (time.perf_counter() - t) / 3

        t = time.perf_counter()
        for _ in range(3):
            dec = decode_windows(pc, anchor, ATG_LEFT, anchor)
        t_dec = (time.perf_counter() - t) / 3

        t = time.perf_counter()
        for _ in range(3):
            x = torch.as_tensor(dec)
        t_tens = (time.perf_counter() - t) / 3

        with torch.no_grad():
            t = time.perf_counter()
            for _ in range(3):
                _ = m.enc_init(x)
                _ = m.enc_atg(x)
            t_enc = (time.perf_counter() - t) / 3

        tot = t_build + t_dec + t_tens + t_enc
        for name, v in (("build perturbed codes", t_build),
                        ("decode_windows", t_dec),
                        ("to torch tensor", t_tens),
                        ("2 encoder passes (CPU)", t_enc)):
            print(f"  {name:<26} {v:>7.3f} s   {100*v/tot:>5.1f}%")
        print(f"  {'TOTAL':<26} {tot:>7.3f} s   "
              f"{n_rows/tot:>8,.0f} rows/s\n")

    # ---- where inside decode_windows? -------------------------------------
    print("=== inside decode_windows, 16,384 rows ===")
    n = 16384
    rc = np.random.default_rng(0).integers(0, K, n)
    cd = codes[rc, 0].copy()
    anchor = s0[rc]
    W = cd.shape[1]

    t = time.perf_counter()
    fill = (cd & 7).astype(np.int16); junc = ((cd >> 3) & 1).astype(np.float32)
    filled = fill > 0
    t_masks = time.perf_counter() - t

    t = time.perf_counter()
    out = np.zeros((n, 9, W), dtype=np.float32)
    t_alloc = time.perf_counter() - t

    t = time.perf_counter()
    r, c = np.nonzero((fill >= 1) & (fill <= 4))
    out[r, fill[r, c] - 1, c] = 1.0
    t_onehot = time.perf_counter() - t

    t = time.perf_counter()
    is_gc = ((fill == 2) | (fill == 3)).astype(np.float32)
    fmask = filled.astype(np.float32)
    cg = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(is_gc, axis=1)], 1)
    cn = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(fmask, axis=1)], 1)
    t_gc = time.perf_counter() - t

    t = time.perf_counter()
    k = np.arange(W)[None, :]
    pos = (anchor[:, None] - ATG_LEFT) + k
    frame = np.mod(pos - anchor[:, None], 3)
    rr, cc = np.nonzero(filled)
    out[rr, 6 + frame[rr, cc], cc] = 1.0
    t_frame = time.perf_counter() - t

    tot = t_masks + t_alloc + t_onehot + t_gc + t_frame
    for name, v in (("masks from the codes", t_masks),
                    ("allocate (n,9,1000) f32", t_alloc),
                    ("channels 0-3 one-hot", t_onehot),
                    ("channel 5 rolling GC", t_gc),
                    ("channels 6-8 frame grid", t_frame)):
        print(f"  {name:<26} {v:>7.3f} s   {100*v/tot:>5.1f}%")
    print(f"  {'TOTAL':<26} {tot:>7.3f} s")
    print(f"\n  allocation alone is {n*9*W*4/1e6:,.0f} MB per chunk")


if __name__ == "__main__":
    main()
