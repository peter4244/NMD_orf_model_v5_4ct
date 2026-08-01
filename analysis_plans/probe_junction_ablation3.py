#!/usr/bin/env python
"""
probe_junction_ablation3.py — sufficiency, because ablation alone cannot answer it.

WHAT THE FIRST TWO PASSES ESTABLISHED. On AUC, which is the only scale comparable
across these conditions, EVERY single-group ablation leaves the reference-start
discrimination nearly intact:

    full 0.8382 | zero_junc 0.8369 | zero_gc 0.8666 | zero_phase 0.8205 |
    zero_seq 0.7934

Removing the junction channel costs 0.4% of the distance above 0.5. Removing all
four sequence-identity channels costs 13%. Removing GC IMPROVES it.

THAT PATTERN IS THE SIGNATURE OF REDUNDANCY, NOT OF AN UNUSED CHANNEL. When two
channels carry overlapping information, deleting either one alone costs almost
nothing, and a suite of single-channel ablations reports that nothing matters.
So "the gap survived zeroing channel 4" does NOT license "therefore the model is
reading start-codon sequence" -- the identical argument applied to channels 0-3
would license "therefore it is not reading sequence at all", and both cannot hold.

Ablation measures a channel's UNIQUE contribution. Under redundancy that is the
wrong quantity. This script measures the other one:

    SUFFICIENCY -- zero everything EXCEPT one group, and see what discrimination
    survives on that group alone.

Ablate-one and keep-one bracket the question. A channel group can be unnecessary
(ablation cheap) and sufficient (keep-one strong) at the same time; that is what
redundancy MEANS, and it is a finding rather than a failure.

CAVEATS, STATED NOT BURIED. Keep-one inputs are further off the data manifold
than ablate-one, so a keep-one AUC is a bound and not an effect size. Two
groups also cannot be fully separated: channels 0-3 are zero outside the filled
region, so keep_seq still carries the fill mask implicitly. Where a comparison
depends on that, it is named.

`perm_junc` is the on-manifold counterpart: candidate i receives another
candidate's channel-4 track, drawn from its own upstream-fill decile so the
substitute is plausible. That destroys the candidate-specific junction signal
while keeping the channel's marginal distribution real.

IN-SAMPLE, chr21, a training chromosome.
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

GROUPS = {"seq": [0, 1, 2, 3], "junc": [4], "gc": [5], "phase": [6, 7, 8]}


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=1, permute_bins=False)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def capture(model, win, bs=512):
    out = np.empty(len(win), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(win), bs):
            z = model.init_head(model.enc_init(torch.as_tensor(win[i:i + bs])))
            out[i:i + bs] = torch.sigmoid(z.squeeze(-1)).double().numpy()
    return out


def auc(score, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_auc(score, pos, strat, n_min=8):
    num = den = 0.0
    for s in np.unique(strat):
        m = strat == s
        a, b = m & pos, m & ~pos
        if a.sum() >= n_min and b.sum() >= n_min:
            num += a.sum() * auc(score[m], a[m]); den += a.sum()
    return (num / den if den else float("nan")), int(den)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ckdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    ckpts = sorted(ckdir.glob("b8_s*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {ckdir}")

    with h5py.File(REPO / "results_tensor_chr21" / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        cnt, o_s, codes = f["count"][:], f["orf_start"][:], f["codes"][:]
        attrs = dict(f.attrs)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))

    ref = pool["is_ref_cds"].to_numpy() == 1
    slot, kz = pool["slot"].to_numpy(), pool["kozak_score"].to_numpy()

    L = int(attrs["atg_left"])
    anchor = o_s.astype(np.int64)
    win = decode_windows(codes[:, 0], anchor, L, anchor)
    filled = win[:, 6:9].sum(1) > 0
    up_fill = filled[:, :L].mean(1)
    fdec = np.asarray(pd.qcut(up_fill, 10, labels=False, duplicates="drop"))
    dec = np.asarray(pd.qcut(kz, 10, labels=False, duplicates="drop"))

    conds = {"full": win}
    for g, ch in GROUPS.items():
        w = np.zeros_like(win)
        w[:, ch, :] = win[:, ch, :]
        conds[f"keep_{g}"] = w
    w = np.zeros_like(win)                       # everything positional, no sequence
    w[:, [4, 6, 7, 8], :] = win[:, [4, 6, 7, 8], :]
    conds["keep_geom"] = w
    w = np.zeros_like(win)                       # sequence plus the fill mask
    w[:, [0, 1, 2, 3, 6, 7, 8], :] = win[:, [0, 1, 2, 3, 6, 7, 8], :]
    conds["keep_seq_phase"] = w

    # on-manifold junction control: swap in another candidate's channel-4 track,
    # drawn from the same upstream-fill decile. Seeded, so it is reproducible.
    rng = np.random.default_rng(20260801)
    perm = np.arange(len(win))
    for d in np.unique(fdec):
        idx = np.where(fdec == d)[0]
        perm[idx] = rng.permutation(idx)
    w = win.copy()
    w[:, 4, :] = win[perm, 4, :]
    conds["perm_junc"] = w
    print(f"perm_junc: {100*np.mean(perm != np.arange(len(win))):.1f}% of candidates "
          f"received a different candidate's junction track, matched on fill decile")

    print(f"\nchr21: {len(iso):,} transcripts, {len(pool):,} candidates, "
          f"{len(ckpts)} seeds, {int(ref.sum())} reference starts. IN-SAMPLE.\n")

    rows = {}
    for cname, w in conds.items():
        per_seed = np.stack([capture(load(cp), w) for cp in ckpts])
        cap = per_seed.mean(0)
        a_raw = auc(cap, ref)
        a_slot, _ = strat_auc(cap, ref, slot)
        a_kz, _ = strat_auc(cap, ref, dec)
        a_fill, _ = strat_auc(cap, ref, fdec)
        rows[cname] = (a_raw, a_slot, a_kz, a_fill, cap.mean(),
                       [auc(per_seed[j], ref) for j in range(len(ckpts))])
        print(f"  {cname:<15} AUC {a_raw:.4f}  |slot {a_slot:.4f}  "
              f"|kozak {a_kz:.4f}  |fill {a_fill:.4f}   mean_p {cap.mean():.4f}")

    base = rows["full"][0]
    print(f"\n=== SUFFICIENCY: what one group alone can do ===")
    print(f"  full model AUC {base:.4f}. Share of its distance above 0.5 that each")
    print(f"  group reproduces ON ITS OWN:")
    for c in ("keep_seq", "keep_junc", "keep_gc", "keep_phase", "keep_geom",
              "keep_seq_phase"):
        a = rows[c][0]
        print(f"    {c:<15} AUC {a:.4f}   {(a-0.5)/(base-0.5):>7.1%}")

    print(f"\n=== the on-manifold junction control ===")
    a = rows["perm_junc"][0]
    print(f"    perm_junc       AUC {a:.4f}   {(a-0.5)/(base-0.5):>7.1%} retained")
    print(f"    (zero_junc, from pass 2, retained 99.6% -- two different operators)")

    print(f"\n=== per-seed, every condition ===")
    print("  {:<15}".format("condition")
          + "".join(f"{c.name.replace('.pt',''):>10}" for c in ckpts))
    for c in conds:
        print(f"  {c:<15}" + "".join(f"{v:>10.4f}" for v in rows[c][5]))


if __name__ == "__main__":
    main()
