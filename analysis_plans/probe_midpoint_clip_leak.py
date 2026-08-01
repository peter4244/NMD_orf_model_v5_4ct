#!/usr/bin/env python
"""
probe_midpoint_clip_leak.py — checking Maude's second geometric leak, and
re-cutting my below-floor result against it.

THE LEAK, confirmed in code before measuring anything. build_tensor.py:271-273
encodes the ATG window with fill_hi = mid, where mid = (orf_start + orf_end)//2.
So the window's right-hand fill is min(ATG_RIGHT, mid - orf_start + 1) and it
REPORTS THE ORF'S LENGTH. Reference coding sequences are long and fill all 100;
upstream ORFs are short and truncate.

WHY IT DEFEATS THE MATCHED-PAIR TEST. |delta orf_start| <= 50 matches where the
ORF STARTS. This leaks from where the ORF ENDS. The two are independent, and the
pair test was built to control the first. My 5'-padding analysis looked at
upstream fill and could not have seen it either.

WHAT THIS SCRIPT DECIDES. Three things, in order of how much they cost us:

  1. Reproduce her decomposition -- the pair test restricted to pairs where BOTH
     windows are untruncated, so no length information is available to either arm.
  2. RE-CUT MY BELOW-FLOOR RESULT the same way. I reported 73.8% at a mean Kozak
     deficit of -2.308 and called it the strongest anti-PWM evidence in the
     analysis. Below-floor references are still long coding sequences, so those
     pairs carry the length leak too and the number is probably inflated. Maude
     flagged this against her own interest and she is likely right.
  3. Ask whether ANY geometry is left after both leaks are controlled -- match on
     upstream fill as well as right-hand fill, and see what survives.

A residual at n in the hundreds needs its uncertainty stated, not implied, so
every rate here carries an exact binomial interval and the count it rests on.

chr2 is VALIDATION. Held out from gradients, not from model selection.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tensor_io import decode_windows           # noqa: E402

FLOOR = -1.2507921188400943


def ci(k, n):
    """Exact-ish binomial interval, Wilson. Returns (lo, hi) in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.959964
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def rate(won, label, indent=4):
    n, k = len(won), int(won.sum())
    if n == 0:
        print(" " * indent + f"{label:<34} n 0"); return
    lo, hi = ci(k, n)
    z = (k - n / 2) / np.sqrt(n / 4) if n else 0.0
    print(" " * indent + f"{label:<34} {100*k/n:>5.1f}%   n {n:>5,}   "
          f"z {z:>5.2f}   95% CI [{lo:.1f}, {hi:.1f}]")


def main():
    sys.stdout.reconfigure(line_buffering=True)

    with h5py.File(REPO / "results_tensor_chr2" / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        cnt = f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        o_e = f["orf_end"][:].astype(np.int64)
        codes = f["codes"][:, 0]
        L = int(f.attrs["atg_left"]); R = int(f.attrs["atg_right"])

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))

    ref = pool["is_ref_cds"].to_numpy() == 1
    kz = pool["kozak_score"].to_numpy()
    tx = pool["tx"].to_numpy()
    cap = np.load(REPO / "results_interp_all" / "capture_chr2.npz")["cap"]
    assert len(cap) == len(pool)

    # fill geometry, measured from the stored window rather than from the formula
    right_fill = np.empty(len(pool), dtype=np.int32)
    up_fill = np.empty(len(pool), dtype=np.float64)
    B = 4096
    for i in range(0, len(pool), B):
        w = decode_windows(codes[i:i + B], o_s[i:i + B], L, o_s[i:i + B])
        filled = w[:, 6:9].sum(1) > 0
        right_fill[i:i + B] = filled[:, L:].sum(1)
        up_fill[i:i + B] = filled[:, :L].mean(1)

    orf_len = (o_e - o_s + 1).astype(np.int64)
    predicted = np.minimum(R, (o_s + o_e) // 2 - o_s + 1)
    agree = float(np.mean(right_fill == np.clip(predicted, 0, R)))
    print(f"chr2: {len(pool):,} candidates, {int(ref.sum()):,} reference starts")
    print(f"right-hand fill == min({R}, mid - orf_start + 1) in "
          f"{100*agree:.1f}% of candidates  <- the clip rule, verified on data\n")

    full = right_fill >= R
    print("=== 1. does the clip separate the arms? ===")
    print(f"  full {R} bases of right fill, reference     {100*full[ref].mean():>5.1f}%")
    print(f"  full {R} bases of right fill, non-reference {100*full[~ref].mean():>5.1f}%")
    print(f"  median ORF length  reference {np.median(orf_len[ref]):>7,.0f} nt"
          f"   non-reference {np.median(orf_len[~ref]):>7,.0f} nt\n")

    # ---- pairs ------------------------------------------------------------
    def pairs_for(keep, D=50):
        out = []
        for t in np.unique(tx):
            idx = np.where((tx == t) & keep)[0]
            r_i, o_i = idx[ref[idx]], idx[~ref[idx]]
            if not len(r_i) or not len(o_i):
                continue
            for ri in r_i:
                for oi in o_i[np.abs(o_s[o_i] - o_s[ri]) <= D]:
                    out.append((ri, oi))
        return np.array(out) if out else np.empty((0, 2), int)

    P = pairs_for(kz >= FLOOR)                      # Maude's floor-restricted set
    print(f"=== 2. baselines on the {len(P):,} floor-restricted matched pairs ===")
    won_cap = cap[P[:, 0]] > cap[P[:, 1]]
    for nm, v in (("orf_length", orf_len.astype(float)),
                  ("p_capture (the model)", cap),
                  ("right-hand window fill", right_fill.astype(float)),
                  ("5'-proximity (-orf_start)", -o_s.astype(float)),
                  ("kozak_score", kz)):
        w = v[P[:, 0]] > v[P[:, 1]]
        print(f"    {nm:<28} {100*w.mean():>5.1f}%")

    both_full = full[P[:, 0]] & full[P[:, 1]]
    print(f"\n=== 3. pairs where BOTH windows are untruncated ===")
    print("    no ORF-length information available to either arm")
    rate(won_cap[both_full], "capture")
    up = o_s[P[:, 0]] < o_s[P[:, 1]]
    rate(won_cap[both_full & up], "  reference upstream")
    rate(won_cap[both_full & ~up], "  reference downstream")
    rate(won_cap[~both_full], "at least one truncated")

    print(f"\n=== 4. RE-CUT: my below-floor result, both arms untruncated ===")
    PA = pairs_for(np.ones(len(ref), bool))
    won_a = cap[PA[:, 0]] > cap[PA[:, 1]]
    bf = (kz[PA[:, 0]] < FLOOR) & (kz[PA[:, 1]] >= FLOOR)
    bf_full = bf & full[PA[:, 0]] & full[PA[:, 1]]
    rate(won_a[bf], "below-floor, as I reported it")
    rate(won_a[bf_full], "below-floor, both untruncated")
    if bf_full.sum():
        dk = kz[PA[bf_full][:, 0]] - kz[PA[bf_full][:, 1]]
        print(f"        mean Kozak deficit of the reference: {dk.mean():+.3f}")

    print(f"\n=== 5. is any geometry left after BOTH leaks are controlled? ===")
    print("    both untruncated AND upstream fill matched to within 0.05")
    du = np.abs(up_fill[P[:, 0]] - up_fill[P[:, 1]])
    tight = both_full & (du <= 0.05)
    rate(won_cap[tight], "capture")
    for nm, v in (("right-hand fill", right_fill.astype(float)),
                  ("upstream fill", up_fill),
                  ("orf_length", orf_len.astype(float))):
        w = v[P[tight][:, 0]] > v[P[tight][:, 1]]
        rate(w, f"  control: {nm}")


if __name__ == "__main__":
    main()
