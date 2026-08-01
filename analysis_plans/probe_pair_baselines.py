#!/usr/bin/env python
"""
probe_pair_baselines.py — trivial baselines on the matched-pair test, and one
confound neither window has checked.

The interpretability window asked for the §8.3 tabular baseline to be run against
the PAIR test and not only against AUC, so both halves of §5 have a trivial
comparator in one table. That is right and it is done here.

THE CONFOUND, which came out of writing those baselines. The ATG window is
anchored at the start codon with 900 to its left and 100 to its right, but the
fill is clipped at the ORF MIDPOINT (plan §5.3 step 1), so

    right-hand fill = min(100, mid - orf_start + 1, tx_length - orf_start + 1)

An ORF shorter than about 200 bases has its window truncated ON THE RIGHT, and
the amount of truncation is a direct readout of ORF LENGTH. Reference coding
sequences are long; upstream ORFs are often very short. So "the window is not
right-truncated" is available to the capture head and correlates with being the
reference — a second geometric leak, distinct from the 5'-end padding the
interpretability window found, and one that the |orf_start| matching does NOT
control because it is about the ORF's end rather than its start.

The test: restrict the matched pairs to those where BOTH candidates have the full
100 bases of right-hand fill, so neither window carries any length information at
all. If the preference survives there, it is not ORF length.

Everything is on chr2, which is validation — held out from gradients, chosen on
for early stopping. Label it that way.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model_v6 import ScanningNMDModel          # noqa: E402
from tensor_io import decode_windows           # noqa: E402

ATG_LEFT, ATG_RIGHT = 900, 100
FLOOR = -1.2507921188400943


def winrate(pairs, score, higher_is_reference=True):
    """Fraction of (reference, competitor) pairs the score orders correctly."""
    if not len(pairs):
        return float("nan"), 0
    a = score[pairs[:, 0]]
    b = score[pairs[:, 1]]
    if not higher_is_reference:
        a, b = -a, -b
    return float(np.mean(np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0)))), len(pairs)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="results_tensor_chr2")
    ap.add_argument("--ckpt-dir", default="results_interp_all/v6_checkpoints")
    ap.add_argument("--caliper", type=int, default=50)
    args = ap.parse_args()

    with h5py.File(REPO / args.tensor / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:], f["orf_end"][:]
        codes = f["codes"][:]
        raw = f["structural_raw"][:] if "structural_raw" in f else f["structural"][:]

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score",
                                "orf_length", "frac_start"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    key = {s: i for i, s in enumerate(iso)}
    pool["tx"] = pool.isoform_id.map(key)
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)

    caps = []
    for cp in sorted((REPO / args.ckpt_dir).glob("b8_s*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False); a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1)
        m.load_state_dict(ck["model"]); m.eval()
        acc = []
        with torch.no_grad():
            for i in range(len(iso)):
                sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
                s0 = o_s[sl].astype(np.int64)
                x = torch.as_tensor(decode_windows(codes[sl][:, 0], s0, ATG_LEFT, s0))
                acc.append(torch.sigmoid(m.init_head(m.enc_init(x)).squeeze(-1)).numpy())
        caps.append(np.concatenate(acc))
        print(f"  {cp.name}: done", flush=True)
    cap = np.mean(caps, axis=0)

    tx = pool.tx.to_numpy()
    ref = pool.is_ref_cds.to_numpy() == 1
    kz = pool.kozak_score.to_numpy()
    olen = pool.orf_length.to_numpy()
    start = np.concatenate([o_s[int(off[i]):int(off[i]) + int(cnt[i])]
                            for i in range(len(iso))]).astype(np.int64)
    end = np.concatenate([o_e[int(off[i]):int(off[i]) + int(cnt[i])]
                          for i in range(len(iso))]).astype(np.int64)
    txlen = np.concatenate([np.full(int(cnt[i]), 0) for i in range(len(iso))])
    mid = (start + end) // 2
    # right-hand fill: how far into the ORF the ATG window actually reaches
    right_fill = np.minimum(ATG_RIGHT, np.maximum(0, mid - start + 1))
    ejc = raw[:, 0]

    above = kz >= FLOOR
    idx = np.flatnonzero(above)
    r_i = idx[ref[idx]]
    pairs = []
    for i in r_i:
        sib = idx[(tx[idx] == tx[i]) & (~ref[idx]) &
                  (np.abs(start[idx] - start[i]) <= args.caliper)]
        for j in sib:
            pairs.append((i, j))
    P = np.array(pairs)
    print(f"\nchr2 (VALIDATION -- held out from gradients, chosen on for early "
          f"stopping)")
    print(f"  {len(P):,} position-matched pairs, both arms at or above the Kozak floor")

    print(f"\n=== 1. what wins these pairs? ===")
    print(f"  {'score':<34} {'win rate':>10} {'n':>8}")
    for name, sc, hi in (("p_capture (the model)", cap, True),
                         ("kozak_score (PWM)", kz, True),
                         ("orf_length", olen.astype(float), True),
                         ("n_downstream_ejc", ejc, True),
                         ("5'-proximity (-orf_start)", -start.astype(float), True),
                         ("right-hand window fill", right_fill.astype(float), True)):
        w, n = winrate(P, np.asarray(sc, dtype=float), hi)
        print(f"  {name:<34} {100*w:>9.1f}% {n:>8,}")
    print(f"  50% is chance. The pairs are matched on start position, so a score")
    print(f"  that is a function of start position alone must sit near 50%.")

    print(f"\n=== 2. THE ORF-LENGTH LEAK, and whether it explains the result ===")
    print(f"  The ATG window fills 100 bases past the start codon BUT IS CLIPPED AT")
    print(f"  THE ORF MIDPOINT, so an ORF under ~200 bases has a right-truncated")
    print(f"  window and the truncation reports its length. Reference CDSs are long.")
    full = right_fill >= ATG_RIGHT
    print(f"  candidates with the full 100 bases of right fill: "
          f"{100*full.mean():.1f}%")
    print(f"    among reference starts     : {100*full[ref].mean():.1f}%")
    print(f"    among non-reference        : {100*full[~ref].mean():.1f}%")
    both = full[P[:, 0]] & full[P[:, 1]]
    print(f"  pairs where BOTH windows are untruncated: {int(both.sum()):,} of {len(P):,}")
    for lab, m in (("both untruncated (no length leak)", both),
                   ("at least one truncated", ~both)):
        w, n = winrate(P[m], cap, True)
        wl, _ = winrate(P[m], olen.astype(float), True)
        print(f"  {lab:<36} capture {100*w:>5.1f}%   orf_length {100*wl:>5.1f}%   n {n:,}")

    print(f"\n=== 3. direction split, on the untruncated pairs only ===")
    print(f"  The interpretability window's control, restricted to pairs carrying")
    print(f"  neither the 5'-padding leak nor the ORF-length leak.")
    up = start[P[:, 0]] < start[P[:, 1]]
    for lab, m in (("reference UPSTREAM", both & up),
                   ("reference DOWNSTREAM", both & ~up)):
        w, n = winrate(P[m], cap, True)
        print(f"  {lab:<36} capture {100*w:>5.1f}%   n {n:,}")
    print(f"  A monotone positional preference must put one of these BELOW 50%.")


if __name__ == "__main__":
    main()
