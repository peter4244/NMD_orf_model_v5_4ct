#!/usr/bin/env python
"""
probe_position_matched_pairs.py — checking Maude's conditional test on chr2.

THE DISAGREEMENT. I measured that unfitted positional features match the trained
capture head marginally (chr21: capture 0.8382, downstream-fill 0.8384, slot
0.8573) and concluded the reference-start preference is explained by position.
Maude's reply: that is a MARGINAL comparison, two correlated predictors both
discriminating shows neither explains the other, and conditionally the answer
flips. She is right about the logic. This checks the measurement.

HER TEST, which I am reproducing rather than accepting: within a transcript,
reference start against a non-reference candidate whose orf_start is within 50 nt
of it. Same 5' end, same exon structure, position held fixed by construction
rather than modelled.

TWO CONTROLS SHE DID NOT RUN, AND ONE OF THEM IS THE ONE THAT MATTERS.

  1. HER STRATIFICATION CHECK IS CIRCULAR, THOUGH NOT WRONG. She reports that
     first-filled falls to chance inside orf_start bands and reads that as
     evidence the stratification works. But first_filled = 900 - min(900,
     orf_start - 1) is a DETERMINISTIC FUNCTION of orf_start, so it must go to
     chance inside a narrow orf_start band whatever else is or is not
     controlled. It confirms the band is narrow; it is not independent evidence
     that geometry is removed.

  2. DIRECTION IS THE REAL RESIDUAL, AND IT IS TESTABLE. Matching |delta
     orf_start| <= 50 still leaves one member of each pair upstream of the
     other, and the upstream member has strictly more 5' context and a window
     boundary up to 50 positions further left. If the model carries any residual
     preference for upstream candidates, and the reference is usually the
     upstream member, the pair test recovers position under a different name.
     THE CHECK: split the win rate by whether the reference is the upstream or
     the downstream member of its pair. Position controlled => both rates alike.
     Residual position => the reference wins when upstream and loses when not.

Also reported: the same pairs scored by a pure geometry feature, which must sit
at chance if the matching did its job, and the Kozak-advantage split, which is
the part of her argument that most needs to be independently seen.

chr2 is VALIDATION. Held out from gradient updates, but early stopping chose the
epoch on it, so it is model-selection data, not a clean test set.
"""

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


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=1, permute_bins=False)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def capture(model, codes, anchor, L, bs=256):
    out = np.empty(len(codes), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(codes), bs):
            w = decode_windows(codes[i:i + bs], anchor[i:i + bs], L,
                               anchor[i:i + bs])
            z = model.init_head(model.enc_init(torch.as_tensor(w)))
            out[i:i + bs] = torch.sigmoid(z.squeeze(-1)).double().numpy()
    return out


def auc(score, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(np.asarray(score, float)).rank().to_numpy()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ckdir = Path(sys.argv[1])
    ckpts = sorted(ckdir.glob("b8_s*.pt"))

    with h5py.File(REPO / "results_tensor_chr2" / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        cnt = f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        codes = f["codes"][:, 0]
        attrs = dict(f.attrs)
    L = int(attrs["atg_left"])

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))

    ref = pool["is_ref_cds"].to_numpy() == 1
    slot = pool["slot"].to_numpy()
    kz = pool["kozak_score"].to_numpy()
    tx = pool["tx"].to_numpy()
    first = np.maximum(0, L - np.minimum(L, o_s - 1)).astype(float)

    print(f"chr2 (VALIDATION): {len(iso):,} transcripts, {len(pool):,} candidates, "
          f"{int(ref.sum()):,} reference starts, {len(ckpts)} seeds")

    per_seed = np.stack([capture(load(cp), codes, o_s, L) for cp in ckpts])
    cap = per_seed.mean(0)

    print("\n=== 1. marginal, which is the comparison I over-read ===")
    for nm, v in (("capture", cap), ("slot", -slot.astype(float)),
                  ("first filled", -first), ("orf_start", -o_s.astype(float)),
                  ("kozak_score", kz)):
        a = auc(v, ref)
        print(f"  {nm:<14} AUC {max(a, 1-a):.4f}")

    print("\n=== 2. within-transcript pairs, matched on orf_start ===")
    print("  reference vs a non-reference candidate of the SAME transcript whose")
    print("  orf_start is within D nt. Position held fixed by construction.")
    order = np.argsort(tx, kind="stable")
    bounds = {}
    for t in np.unique(tx):
        idx = np.where(tx == t)[0]
        bounds[t] = idx

    for D in (10, 25, 50, 100):
        pairs = []
        for t, idx in bounds.items():
            r_i = idx[ref[idx]]
            o_i = idx[~ref[idx]]
            if len(r_i) == 0 or len(o_i) == 0:
                continue
            for ri in r_i:
                d = np.abs(o_s[o_i] - o_s[ri])
                for oi in o_i[d <= D]:
                    pairs.append((ri, oi))
        if not pairs:
            print(f"  D={D}: no pairs"); continue
        P = np.array(pairs)
        rc, oc = cap[P[:, 0]], cap[P[:, 1]]
        win = float((rc > oc).mean() + 0.5 * (rc == oc).mean())
        nref = len(np.unique(P[:, 0]))
        # geometry-only control on the SAME pairs
        gr, go = -first[P[:, 0]], -first[P[:, 1]]
        gwin = float((gr > go).mean() + 0.5 * (gr == go).mean())
        print(f"  D={D:>3}nt  {len(P):>6,} pairs over {nref:>5,} reference candidates"
              f"   capture wins {100*win:>5.1f}%   geometry-only {100*gwin:>5.1f}%")

    # ---- the direction control, at D=50 -----------------------------------
    D = 50
    pairs = []
    for t, idx in bounds.items():
        r_i, o_i = idx[ref[idx]], idx[~ref[idx]]
        if len(r_i) == 0 or len(o_i) == 0:
            continue
        for ri in r_i:
            d = np.abs(o_s[o_i] - o_s[ri])
            for oi in o_i[d <= D]:
                pairs.append((ri, oi))
    P = np.array(pairs)
    rc, oc = cap[P[:, 0]], cap[P[:, 1]]
    won = rc > oc
    ref_is_upstream = o_s[P[:, 0]] < o_s[P[:, 1]]

    print(f"\n=== 3. THE DIRECTION CONTROL (D=50) ===")
    print("  If position were still doing the work, the reference would win when it")
    print("  is the upstream member and lose when it is the downstream one.")
    for lab, m in (("reference UPSTREAM of competitor", ref_is_upstream),
                   ("reference DOWNSTREAM of competitor", ~ref_is_upstream)):
        if m.sum():
            print(f"    {lab:<38} {100*won[m].mean():>5.1f}%   n {int(m.sum()):,}")
    print(f"    {'difference':<38} "
          f"{100*(won[ref_is_upstream].mean() - won[~ref_is_upstream].mean()):>+5.1f} pp")

    print(f"\n=== 4. by Kozak advantage (D=50) ===")
    dk = kz[P[:, 0]] - kz[P[:, 1]]
    for lab, m in (("reference WORSE Kozak (< -0.25)", dk < -0.25),
                   ("roughly equal", np.abs(dk) <= 0.25),
                   ("reference BETTER Kozak (> +0.25)", dk > 0.25)):
        if m.sum():
            print(f"    {lab:<34} {100*won[m].mean():>5.1f}%   n {int(m.sum()):,}")

    print(f"\n=== 5. per seed (D=50) ===")
    for j, cp in enumerate(ckpts):
        rc_, oc_ = per_seed[j][P[:, 0]], per_seed[j][P[:, 1]]
        w = float((rc_ > oc_).mean() + 0.5 * (rc_ == oc_).mean())
        print(f"    {cp.name:<12} {100*w:>5.1f}%")

    print(f"\n=== 6. is first_filled a deterministic function of orf_start? ===")
    u = pd.DataFrame({"o": o_s, "f": first}).groupby("o")["f"].nunique()
    print(f"    distinct first_filled values per orf_start value: "
          f"max {int(u.max())}, mean {u.mean():.3f}  "
          f"({'deterministic' if u.max() == 1 else 'NOT deterministic'})")
    print("    -> if deterministic, 'first_filled falls to chance inside orf_start")
    print("       bands' is guaranteed by construction and is not an independent")
    print("       check that geometry has been removed.")


if __name__ == "__main__":
    main()
