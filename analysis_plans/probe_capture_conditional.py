#!/usr/bin/env python
"""
probe_capture_conditional.py — is the capture preference position, or sequence?

The interpretability window's challenge, and it is a good one: the ATG window is
900 upstream and 100 into the ORF, so a candidate starting near the 5' end has a
window that is mostly UNFILLED, and the index where filling begins is
max(901 - orf_start, 0) -- distance to the 5' end, to the nucleotide, handed to
the model for free. Reference start codons sit near the 5' end. So "the model
prefers the annotated start" may be "the model reads how much 5' context the
window has".

They showed that unfitted positional features discriminate reference from
non-reference about as well as the trained capture head does. That is necessary
for their reading but not sufficient for it: two features can both discriminate
without either explaining the other.

THE TEST THAT SEPARATES THEM IS CONDITIONAL, NOT MARGINAL. Ask whether capture
still separates reference from non-reference AT MATCHED POSITION. Inside a narrow
band of orf_start the positional features are nearly constant, so they carry
almost no information, and any discrimination that survives is not positional.

Three strata, weakest to strongest:

  1  within bands of orf_start, pooled across transcripts
  2  within bands of orf_start AND of window fill fraction
  3  WITHIN TRANSCRIPT and within a caliper on orf_start -- same 5' end, same
     exon structure, same amount of 5' context, two start codons a few dozen
     bases apart. Anything left here is the sequence at the start codon.

The positional features are scored through the identical stratification, where
they must fall to chance by construction. That is the check that the
stratification is doing what it claims.
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

ATG_LEFT = 900
KOZAK_FLOOR = -1.2507921188400943


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / c)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def stratified_auc(y, s, strata, min_per_arm=5):
    """Pooled within-stratum AUC: concordant pairs over comparable pairs, counted
    only WITHIN a stratum. A stratum with one arm empty contributes nothing."""
    conc = comp = 0.0
    used = 0
    for g in np.unique(strata):
        m = strata == g
        yy, ss = y[m], s[m]
        if yy.sum() < min_per_arm or (1 - yy).sum() < min_per_arm:
            continue
        a = auc(yy, ss)
        n = yy.sum() * (1 - yy).sum()
        conc += a * n; comp += n; used += 1
    return (conc / comp if comp else np.nan), int(comp), used


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--caliper", type=int, default=50)
    args = ap.parse_args()

    with h5py.File(Path(args.tensor) / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:], f["orf_end"][:]
        codes = f["codes"][:]

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    key = {s: i for i, s in enumerate(iso)}
    pool["tx"] = pool.isoform_id.map(key)
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)

    caps = []
    for cp in sorted(Path(args.ckpt_dir).glob("b8_s*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False); a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1); m.load_state_dict(ck["model"]); m.eval()
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

    ref = (pool.is_ref_cds.to_numpy() == 1)
    kz = pool.kozak_score.to_numpy()
    tx = pool.tx.to_numpy()
    start = np.concatenate([o_s[int(off[i]):int(off[i]) + int(cnt[i])]
                            for i in range(len(iso))]).astype(np.int64)
    end = np.concatenate([o_e[int(off[i]):int(off[i]) + int(cnt[i])]
                          for i in range(len(iso))]).astype(np.int64)
    txlen = np.concatenate([np.full(int(cnt[i]), 0) for i in range(len(iso))])
    # the two positional quantities the encoding hands the model
    first_filled = np.maximum(ATG_LEFT + 1 - start, 0)          # index where fill begins
    mid = (start + end) // 2
    a_hi = np.minimum(mid, start + 99)
    fill = np.maximum(0, a_hi - np.maximum(1, start - ATG_LEFT) + 1) / 1000.0

    above = kz >= KOZAK_FLOOR
    y = ref[above].astype(np.int8)
    print(f"\n{len(pool):,} candidates, {int(above.sum()):,} at or above the Kozak floor")
    print(f"  reference {int(y.sum()):,}   other {int((1-y).sum()):,}")

    print(f"\n=== marginal: what discriminates reference from non-reference? ===")
    for name, s in (("p_capture (trained)", cap), ("first filled index", -first_filled),
                    ("window fill fraction", -fill), ("slot", -pool.slot.to_numpy()),
                    ("orf_start", -start), ("kozak_score (PWM)", kz)):
        print(f"  {name:<26} AUC {auc(y, np.asarray(s)[above]):.4f}")
    print(f"  (positional features negated so that 5'-proximal scores high)")

    print(f"\n=== 1. within bands of orf_start ===")
    print(f"  Inside a band the positional features are nearly constant, so they")
    print(f"  must fall to chance. Anything capture retains is not positional.")
    for nb in (10, 25, 50):
        band = pd.qcut(start[above], nb, labels=False, duplicates="drop")
        ac, np_, ns = stratified_auc(y, cap[above], band)
        af, _, _ = stratified_auc(y, -first_filled[above], band)
        ak, _, _ = stratified_auc(y, kz[above], band)
        print(f"  {nb:>3} bands: capture {ac:.4f}   first-filled {af:.4f}   "
              f"kozak {ak:.4f}   ({np_:,.0f} pairs, {ns} bands)")

    print(f"\n=== 2. within bands of orf_start AND fill fraction ===")
    b1 = pd.qcut(start[above], 20, labels=False, duplicates="drop")
    b2 = pd.qcut(fill[above], 5, labels=False, duplicates="drop")
    joint = b1 * 10 + b2
    ac, np_, ns = stratified_auc(y, cap[above], joint)
    af, _, _ = stratified_auc(y, -first_filled[above], joint)
    ak, _, _ = stratified_auc(y, kz[above], joint)
    print(f"  capture {ac:.4f}   first-filled {af:.4f}   kozak {ak:.4f}   "
          f"({np_:,.0f} pairs, {ns} strata)")

    print(f"\n=== 3. WITHIN TRANSCRIPT, matched on start position (+/- {args.caliper} nt) ===")
    print(f"  Same 5' end, same exon structure, same amount of 5' context. Two start")
    print(f"  codons a few dozen bases apart, one annotated and one not.")
    idx = np.flatnonzero(above)
    r_i = idx[ref[idx]]
    won = tied = lost = 0
    dk = []
    for i in r_i:
        sib = idx[(tx[idx] == tx[i]) & (~ref[idx]) &
                  (np.abs(start[idx] - start[i]) <= args.caliper)]
        for j in sib:
            if cap[i] > cap[j]:
                won += 1
            elif cap[i] == cap[j]:
                tied += 1
            else:
                lost += 1
            dk.append(kz[i] - kz[j])
    n = won + tied + lost
    if n:
        print(f"  matched pairs: {n:,} over {len(r_i):,} reference candidates")
        print(f"  reference has the HIGHER p_capture in {100*(won+0.5*tied)/n:.1f}% "
              f"({won:,} won, {tied:,} tied, {lost:,} lost)")
        print(f"  mean Kozak difference within these pairs: {np.mean(dk):+.4f}")
        print(f"  -> 50% is what pure position predicts; the pairs are matched on it.")


if __name__ == "__main__":
    main()
