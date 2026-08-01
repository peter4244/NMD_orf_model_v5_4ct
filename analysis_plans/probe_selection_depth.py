#!/usr/bin/env python
"""
probe_selection_depth.py — did the initiation head learn initiation?

RECONNAISSANCE, NOT A RESULT. Run on chr21, which is a TRAINING chromosome for
these checkpoints, so every number here is in-sample. It is here to decide
whether the mutagenesis bank is worth building, not to support a claim.

FOUR QUESTIONS, in order of how badly a bad answer would hurt.

1. IS p_capture NEAR 0.5? Stick-breaking gives P_select(k) = p_k * prod(1-p_j),
   so at p_k = 0.5 the mass halves every slot and dies by slot ~10, while 73.3%
   of transcripts hold more than 10 candidates. A model whose p_capture sits at
   0.5 has learned no initiation preference, and a substitution in a deep
   candidate cannot move its output however the sequence changes.

2. HOW DEEP DOES THE MASS ACTUALLY REACH? The consequence of 1, measured rather
   than predicted from it.

3. DOES p_capture PREFER THE ANNOTATED START CODON? `is_ref_cds` is supplied to
   the PREDICTOR variant only (plan §3.6); the interpretable model never sees it.
   So a preference is learned from sequence.

4. DOES p_capture TRACK THE KOZAK SCORE? The PWM score decides pool admission and
   is supplied to NEITHER variant (plan §3.6 step 5). A correlation is the
   initiation head having recovered initiation context from the window alone,
   which is the whole premise of reading this model.

3 and 4 are the two that say something positive rather than only ruling something
out, and both are free: the quantities are already in the pool table.
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
                         n_structural=1,
                         permute_bins=bool(a.get("permute_bins", False)))
    m.load_state_dict(ck["model"]); m.eval()
    return m, a


def main():
    sys.stdout.reconfigure(line_buffering=True)
    tensor = REPO / "results_tensor_chr21" / "nmd_tensor.h5"
    ckpts = sorted((Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")).glob("b8_s*.pt"))
    if not ckpts:
        raise SystemExit("no checkpoints found")

    with h5py.File(tensor, "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:], f["orf_end"][:]
        struct = f["structural"][:]
        codes = f["codes"][:]
        labels = f["labels"][:]

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score",
                                "admitted_by"])
    pool = pool[pool.isoform_id.isin(set(iso))].sort_values(["isoform_id", "slot"])
    key = {s: i for i, s in enumerate(iso)}
    pool["tx"] = pool.isoform_id.map(key)
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)

    print(f"chr21: {len(iso):,} transcripts, {len(pool):,} candidates, "
          f"{len(ckpts)} checkpoints")
    print("chr21 is a TRAINING chromosome for these checkpoints. In-sample "
          "reconnaissance.\n")

    per_seed = []
    for cp in ckpts:
        model, a = load(cp)
        caps, sels = [], []
        with torch.no_grad():
            for i in range(len(iso)):
                sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
                s0 = o_s[sl].astype(np.int64); e0 = o_e[sl].astype(np.int64)
                K = len(s0)
                atg = torch.as_tensor(decode_windows(codes[sl][:, 0], s0, 900, s0))
                stp = torch.as_tensor(decode_windows(codes[sl][:, 1], e0 - 1, 500, s0))
                u = torch.as_tensor(struct[sl][:, [0]], dtype=torch.float32)
                _, parts = model(atg[None], stp[None], u[None],
                                 torch.ones(1, K, dtype=torch.bool), return_parts=True)
                caps.append(parts["p"][0].numpy())
                sels.append(parts["p_select"][0].numpy())
        per_seed.append((np.concatenate(caps), np.concatenate(sels)))
        print(f"  {cp.name}: done", flush=True)

    cap = np.mean([x[0] for x in per_seed], axis=0)
    sel = np.mean([x[1] for x in per_seed], axis=0)
    slot = pool["slot"].to_numpy()
    assert len(cap) == len(pool), f"{len(cap)} vs {len(pool)}"

    print(f"\n=== 1. p_capture: has the initiation head learned a preference? ===")
    print(f"  0.5 means no preference at all. Averaged over {len(ckpts)} seeds.")
    q = np.percentile(cap, [1, 10, 25, 50, 75, 90, 99])
    print(f"  percentiles  1% {q[0]:.4f}  10% {q[1]:.4f}  25% {q[2]:.4f}  "
          f"50% {q[3]:.4f}  75% {q[4]:.4f}  90% {q[5]:.4f}  99% {q[6]:.4f}")
    print(f"  mean {cap.mean():.4f}   sd {cap.std():.4f}")
    print(f"  fraction within 0.05 of 0.5: {100*np.mean(np.abs(cap-0.5) < 0.05):.1f}%")
    for s, (c, _) in zip(ckpts, per_seed):
        print(f"    {s.name:<12} median {np.median(c):.4f}  sd {c.std():.4f}")

    print(f"\n=== 2. how deep does selection mass reach? ===")
    tot = sel.sum()
    for lo, hi in ((0, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)):
        m = (slot >= lo) & (slot <= hi)
        if m.any():
            lab = f"slots {lo}-{hi}" if hi < 10**9 else f"slots {lo}+"
            print(f"  {lab:<14} {100*sel[m].sum()/tot:>6.2f}% of mass   "
                  f"{int(m.sum()):>7,} candidates")
    print(f"  candidates with P_select < 1e-4: "
          f"{100*np.mean(sel < 1e-4):.1f}%  (a substitution there cannot move the output)")
    print(f"  candidates with P_select < 1e-2: {100*np.mean(sel < 1e-2):.1f}%")

    print(f"\n=== 3. does p_capture prefer the annotated start codon? ===")
    print(f"  is_ref_cds is supplied to the PREDICTOR only; this model never sees it.")
    ref = pool["is_ref_cds"].to_numpy() == 1
    print(f"  reference start codons: {int(ref.sum()):,} of {len(pool):,}")
    print(f"    p_capture at the reference start : mean {cap[ref].mean():.4f}  "
          f"median {np.median(cap[ref]):.4f}")
    print(f"    p_capture elsewhere              : mean {cap[~ref].mean():.4f}  "
          f"median {np.median(cap[~ref]):.4f}")
    # matched on slot, because slot and is_ref_cds are correlated (median ref slot 1)
    print(f"  matched within slot (slot is confounded with being the reference):")
    for k in range(0, 6):
        m = slot == k
        if (m & ref).sum() >= 20 and (m & ~ref).sum() >= 20:
            print(f"    slot {k}: ref {cap[m & ref].mean():.4f} (n={int((m&ref).sum()):,})"
                  f"   other {cap[m & ~ref].mean():.4f} (n={int((m&~ref).sum()):,})"
                  f"   diff {cap[m & ref].mean()-cap[m & ~ref].mean():+.4f}")

    print(f"\n=== 4. does p_capture track the Kozak score? ===")
    print(f"  kozak_score decides pool admission and is supplied to NEITHER variant.")
    kz = pool["kozak_score"].to_numpy()
    ok = np.isfinite(kz)
    r = np.corrcoef(kz[ok], cap[ok])[0, 1]
    rs = pd.Series(kz[ok]).corr(pd.Series(cap[ok]), method="spearman")
    print(f"  Pearson r  {r:+.4f}     Spearman rho {rs:+.4f}   over {int(ok.sum()):,}")
    print(f"  by Kozak decile:")
    d = pd.qcut(kz[ok], 10, labels=False, duplicates="drop")
    for i in sorted(set(d)):
        m = d == i
        print(f"    decile {i}: kozak {kz[ok][m].mean():>7.3f}   "
              f"p_capture {cap[ok][m].mean():.4f}   n {int(m.sum()):,}")
    # within slot 0 only, where selection order cannot confound it
    m0 = ok & (slot == 0)
    if m0.sum() > 100:
        r0 = np.corrcoef(kz[m0], cap[m0])[0, 1]
        print(f"  within slot 0 only (order cannot confound): r {r0:+.4f}, "
              f"n {int(m0.sum()):,}")


if __name__ == "__main__":
    main()
