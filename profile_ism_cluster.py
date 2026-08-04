#!/usr/bin/env python
"""
profile_ism_cluster.py — where does the time go ON THE CLUSTER?

Written because profiling on the laptop misled me. There, decode was 13-16% of a
CPU-bound loop and the encoders were 85%; the two fixes that followed gave 2.9x
locally and 10% on a V100. The laptop's timings also varied twofold on identical
work, which I should have caught before quoting a speedup from one pair.

So this runs where the code actually runs, times the real inner loop rather than
a reconstruction of it, and repeats each measurement so variance is visible
instead of assumed away.

THE SHAPE TO EXPLAIN. Throughput is flat at 426-427 substitutions/s across an
eightfold change in chunk size, while peak GPU memory moves 1.21 -> 9.56 GB. A
cost that ignores batch size is a cost paid per CALL or per POSITION, not per
row. The candidates:

  per chunk    building perturbed codes, decode, host->device, 3 encoders,
               3 aggregations, the writeback loop
  per position the python writeback: 4 dict lookups and up to 10 float() casts
  per pass     GPU kernel launch overhead, which is per-call and would explain
               flatness exactly -- three encoders plus three aggregations is six
               launches per chunk however large the chunk is

torch.cuda.synchronize() before every timer, or GPU work is attributed to
whatever CPU line happens to touch the result.
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

ATG_LEFT = 900


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}")
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    with h5py.File(REPO / "results_tensor_v6" / "nmd_tensor.h5", "r") as f:
        off, cnt = f["offset"][:], f["count"][:]
        o_s = f["orf_start"][:]
        i = int(np.argmin(np.abs(cnt - 19)))
        sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
        codes = f["codes"][sl]
        s0 = o_s[sl].astype(np.int64)
    K = len(s0)
    print(f"  transcript with K={K}\n")

    ck = torch.load(REPO / "runs" / "interp_c32_b8_s100" / "best.pt",
                    map_location="cpu", weights_only=False)
    m = ScanningNMDModel(conv_channels=32, n_bins=8, n_structural=1)
    m.load_state_dict(ck["model"]); m.to(dev).eval()

    rng = np.random.default_rng(0)
    REP = 5
    for n_rows in (4096, 16384, 49152):
        rc = rng.integers(0, K, n_rows)
        idx = rng.integers(50, 950, n_rows)
        anchor = s0[rc]
        t = {}

        def clock(name, fn):
            vals = []
            for _ in range(REP):
                sync(dev); a = time.perf_counter()
                out = fn()
                sync(dev); vals.append(time.perf_counter() - a)
            t[name] = (float(np.median(vals)), float(np.std(vals)))
            return out

        pc = clock("build codes", lambda: (
            lambda z: (z.__setitem__((np.arange(n_rows), idx),
                                     (z[np.arange(n_rows), idx] & 8) | 2), z)[1]
        )(codes[rc, 0].copy()))
        dec = clock("decode_windows", lambda: decode_windows(pc, anchor, ATG_LEFT, anchor))
        x = clock("host->device", lambda: torch.as_tensor(dec, device=dev))
        with torch.no_grad():
            clock("enc_init", lambda: m.enc_init(x))
            clock("enc_atg", lambda: m.enc_atg(x))
            e = m.enc_init(x)
            zp = clock("init_head", lambda: m.init_head(e).squeeze(-1))
            npert = max(1, n_rows // 4)
            zpv = zp[:npert].reshape(npert, 1).expand(npert, K).contiguous()
            msk = torch.ones(npert, K, dtype=torch.bool, device=dev)
            clock("aggregate x1", lambda: m.aggregate(zpv.double(), zpv.double(),
                                                     msk, stable=True))
            clock("aggregate x3", lambda: [m.aggregate(zpv.double(), zpv.double(),
                                                       msk, stable=True)
                                           for _ in range(3)])

        tot = sum(v[0] for k, v in t.items() if k != "aggregate x1")
        print(f"=== chunk {n_rows:,} rows ===")
        for k, (v, sd) in t.items():
            flag = "   (x3 replaces x1 in the total)" if k == "aggregate x3" else ""
            print(f"  {k:<16} {v:>8.4f} s  +/-{sd:>7.4f}  {100*v/tot:>5.1f}%{flag}")
        print(f"  {'TOTAL':<16} {tot:>8.4f} s   {n_rows/tot:>9,.0f} rows/s\n")

    # kernel-launch overhead, the only cost that would be flat in chunk size
    if dev == "cuda":
        print("=== is it per-call overhead? same total work, split into more calls ===")
        x = torch.randn(49152, 9, 1000, device=dev)
        with torch.no_grad():
            for parts in (1, 4, 16, 64):
                n = 49152 // parts
                sync(dev); a = time.perf_counter()
                for _ in range(REP):
                    for p_ in range(parts):
                        m.enc_init(x[p_ * n:(p_ + 1) * n])
                sync(dev); dt = (time.perf_counter() - a) / REP
                print(f"  {parts:>3} call(s) of {n:>6,} rows: {dt:>7.4f} s   "
                      f"{49152/dt:>9,.0f} rows/s")


if __name__ == "__main__":
    main()
