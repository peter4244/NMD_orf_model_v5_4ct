#!/usr/bin/env python3
"""
probe_head_swap.py — how much of this model's prediction depends on mixing ORF REPRESENTATIONS
rather than mixing ORF VERDICTS?

THE QUESTION. The model computes  logit = g(sum_r a_r e_r)  -- attention-mix the per-ORF
embeddings, then form one verdict. The mechanistic factorization we have been describing is
  P(NMD) = sum_r P(select ORF r) * P(decay | translate r),
which is  logit = sum_r a_r g(e_r)  -- form a verdict per ORF, then mix. These are different
functions of the same parameters.

THE ALGEBRA, which is why this costs no training. With g = cls_head . head =
  Linear(64->32) -> ReLU -> Dropout -> Linear(32->1)
and sum_r a_r = 1, write u_r = W1 e_r + b1. Then

    current    logit = W2 . ReLU( sum_r a_r u_r ) + b2
    factorized logit = W2 . ( sum_r a_r ReLU(u_r) ) + b2

Everything outside the ReLU is identical, so THE ENTIRE DIFFERENCE IS ReLU(sum a u) vs
sum a ReLU(u). Two predictions follow and both are tested here:

  1. The forms are EXACTLY equal when, for each hidden unit, every attended ORF puts u_r on the
     same side of zero. So they coincide on isoforms whose candidate ORFs agree and diverge only
     on those where they disagree.
  2. ReLU is convex, so by Jensen ReLU(sum a u) <= sum a ReLU(u) elementwise in the hidden layer.
     The sign of the logit difference is therefore governed by the sign of W2, not random.

Parameter count is identical, so the existing checkpoints are used unchanged.

THE GUARD THAT MAKES THE REST MEAN ANYTHING. The manual reconstruction of the CURRENT form is
checked against `model(...)` itself. If that does not match to floating-point tolerance, every
other number here is meaningless and the script says so and stops.

Dropout is identity in eval mode, so the "K independent dropout masks" question does not arise
here; it would only matter if the factorized form were TRAINED.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path("/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct")
sys.path.insert(0, str(REPO))
from model import build_model                                    # noqa: E402
from utils import NMDDataset, resolve_checkpoint                 # noqa: E402


def both_forms(m, batch):
    """Return (logit_current, logit_factorized, per_slot_logits, attn, mixed_sign_frac).

    Replicates NMDOrfModel.forward exactly for the current form, then re-associates the sum.
    """
    atg, stop, feat, mask = batch
    B, K = mask.shape
    D = m.aggregator.attn_score.in_features

    flat = mask.reshape(-1)
    e = torch.zeros(B * K, D, dtype=atg.dtype)
    if flat.any():
        e[flat] = m.orf_encoder(atg.reshape(B * K, *atg.shape[2:])[flat],
                                stop.reshape(B * K, *stop.shape[2:])[flat],
                                feat.reshape(B * K, -1)[flat]).to(e.dtype)
    e = e.reshape(B, K, D)

    scores = m.aggregator.attn_score(e).squeeze(-1).masked_fill(~mask, float("-inf"))
    a = torch.softmax(scores, dim=-1).nan_to_num(nan=0.0)                    # (B, K)

    lin1, lin2 = m.head[0], m.cls_head
    u = lin1(e)                                                             # (B, K, 32)

    # current: mix in embedding space, then one ReLU
    cur = lin2(torch.relu(torch.einsum("bk,bkh->bh", a, u))).squeeze(-1)
    # factorized: ReLU per ORF, then mix. sum_r a_r = 1 carries b2 through unchanged.
    fac = lin2(torch.einsum("bk,bkh->bh", a, torch.relu(u))).squeeze(-1)
    # per-ORF verdict, the quantity the factorized form mixes
    per_slot = lin2(torch.relu(u)).squeeze(-1)                              # (B, K)

    # prediction 1: equality iff every attended ORF shares a sign per hidden unit
    att = (a > 1e-6).unsqueeze(-1)
    big = torch.where(att, u, torch.full_like(u, float("inf"))).amin(1)
    small = torch.where(att, u, torch.full_like(u, float("-inf"))).amax(1)
    mixed = ((big < 0) & (small > 0)).float().mean(-1)                      # (B,)
    return cur, fac, per_slot, a, mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="atg1000_stop1000")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    ap.add_argument("--chunk", type=int, default=1500)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cfg = yaml.safe_load((REPO / "config_dn.yaml").read_text())
    h5 = str(REPO / cfg["data"]["hdf5_path"])
    W = args.width
    with h5py.File(h5, "r") as f:
        sp = np.array([s.decode() if isinstance(s, bytes) else s for s in f["split"][:]])
    n_split = int((sp == args.split).sum())
    print(f"split '{args.split}': {n_split:,} isoforms; members {args.seeds}; tag {args.tag}")

    models = {}
    for s in args.seeds:
        m = build_model({**cfg["model"], "window_size_atg": W, "window_size_stop": W})
        m.load_state_dict(torch.load(resolve_checkpoint(REPO / "results_4ct_sweep", args.tag, s),
                                     map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval()
        models[s] = m

    cur = {s: [] for s in args.seeds}
    fac = {s: [] for s in args.seeds}
    spread = {s: [] for s in args.seeds}
    mixed = {s: [] for s in args.seeds}
    lab, nslot = [], []
    guard_max = 0.0
    for lo in range(0, n_split, args.chunk):
        idx = np.arange(lo, min(lo + args.chunk, n_split))
        ds = NMDDataset(h5, W, W, split=args.split, restrict_to=idx)
        batch = (torch.from_numpy(ds.atg_windows), torch.from_numpy(ds.stop_windows),
                 torch.from_numpy(ds.orf_features), torch.from_numpy(ds.orf_mask))
        lab.append(ds.labels)
        nslot.append(ds.orf_mask.sum(1))
        for si, s in enumerate(args.seeds):
            with torch.no_grad():
                c, fv, ps, a, mx = both_forms(models[s], batch)
                ref = models[s](*batch).squeeze(-1)
            guard_max = max(guard_max, float((c - ref).abs().max()))
            cur[s].append(c.numpy())
            fac[s].append(fv.numpy())
            # how much do the per-ORF verdicts disagree, among attended ORFs
            w = a.numpy()
            v = np.where(w > 1e-6, ps.numpy(), np.nan)
            spread[s].append(np.nanmax(v, 1) - np.nanmin(v, 1))
            # PER MEMBER. The exact-equality prediction is a statement about ONE model's own
            # hidden units, so testing a member's mixed-sign fraction against the ENSEMBLE gap
            # compares two different objects and cannot confirm or refute the algebra.
            mixed[s].append(mx.numpy())

    if guard_max > 1e-4:
        print(f"\nGUARD FAILED: manual current-form differs from model(...) by {guard_max:.2e}. "
              f"Every number below would be meaningless. Stopping.")
        sys.exit(1)
    print(f"guard: manual current-form reproduces model(...) to {guard_max:.2e}  OK")

    y = np.concatenate(lab)
    mixed = {s: np.concatenate(v) for s, v in mixed.items()}
    nslot = np.concatenate(nslot)
    print(f"isoforms {len(y):,}; NMD+ {y.mean():.1%}; masked-in slots per isoform "
          f"min {nslot.min()} max {nslot.max()} mean {nslot.mean():.2f}")

    print(f"\n{'member':>8} {'AUC current':>12} {'AUC factor':>12} {'dAUC':>8} "
          f"{'AUPRC cur':>10} {'AUPRC fac':>10} {'dAUPRC':>8}")
    C, F = {}, {}
    for s in args.seeds:
        C[s], F[s] = np.concatenate(cur[s]), np.concatenate(fac[s])
        ac, af = roc_auc_score(y, C[s]), roc_auc_score(y, F[s])
        pc, pf = average_precision_score(y, C[s]), average_precision_score(y, F[s])
        print(f"{s:>8} {ac:>12.4f} {af:>12.4f} {af-ac:>+8.4f} "
              f"{pc:>10.4f} {pf:>10.4f} {pf-pc:>+8.4f}")
    ec = np.mean([C[s] for s in args.seeds], 0)
    ef = np.mean([F[s] for s in args.seeds], 0)
    print(f"{'ensemble':>8} {roc_auc_score(y,ec):>12.4f} {roc_auc_score(y,ef):>12.4f} "
          f"{roc_auc_score(y,ef)-roc_auc_score(y,ec):>+8.4f} "
          f"{average_precision_score(y,ec):>10.4f} {average_precision_score(y,ef):>10.4f} "
          f"{average_precision_score(y,ef)-average_precision_score(y,ec):>+8.4f}")

    d = ef - ec
    print(f"\nLOGIT GAP (factorized - current), ensemble")
    print(f"  mean {d.mean():+.4f}   median {np.median(d):+.4f}   "
          f"sd {d.std():.4f}   |gap| 90th pct {np.percentile(np.abs(d),90):.4f}")
    print(f"  sign: {(d > 0).mean():.1%} positive   "
          f"(Jensen predicts one-signed if W2 were one-signed; W2 has "
          f"{(models[args.seeds[0]].cls_head.weight.detach().numpy() > 0).mean():.0%} positive weights)")
    print(f"  rank correlation of predictions: "
          f"{np.corrcoef(np.argsort(np.argsort(ec)), np.argsort(np.argsort(ef)))[0,1]:.4f}")

    print(f"\nPREDICTION 1 — the forms coincide exactly where every attended ORF shares a sign")
    print(f"  tested PER MEMBER, since it is a statement about that member's own hidden units")
    print(f"{'member':>8} {'mixed-unit frac':>16} {'n with none mixed':>18} "
          f"{'mean |gap| there':>17} {'mean |gap| elsewhere':>21}")
    for s in args.seeds:
        gs = np.abs(np.concatenate(fac[s]) - np.concatenate(cur[s]))
        z = mixed[s] == 0
        print(f"{s:>8} {mixed[s].mean():>16.3f} {z.sum():>18,} "
              f"{(gs[z].mean() if z.any() else float('nan')):>17.2e} {gs[~z].mean():>21.4f}")
    s0 = args.seeds[0]
    g0 = np.abs(np.concatenate(fac[s0]) - np.concatenate(cur[s0]))
    q = np.quantile(mixed[s0], np.linspace(0, 1, 6))
    print(f"\n  member {s0}: {'mixed-unit frac':>18} {'n':>6} {'mean |gap|':>11}")
    for i in range(5):
        sel = (mixed[s0] >= q[i]) & (mixed[s0] <= q[i + 1] if i == 4 else mixed[s0] < q[i + 1])
        if sel.sum():
            print(f"  {'':>10}{q[i]:.3f}-{q[i+1]:<10.3f} {sel.sum():>6} {g0[sel].mean():>11.4f}")
    for s in args.seeds[:2]:
        gs = np.abs(np.concatenate(fac[s]) - np.concatenate(cur[s]))
        spv = np.concatenate(spread[s])
        ok = ~np.isnan(spv)
        print(f"  member {s}: corr(|gap|, mixed-unit fraction) = "
              f"{np.corrcoef(gs, mixed[s])[0,1]:+.3f}   "
              f"corr(|gap|, per-ORF verdict spread) = {np.corrcoef(gs[ok], spv[ok])[0,1]:+.3f}")


if __name__ == "__main__":
    main()
