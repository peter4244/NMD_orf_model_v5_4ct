#!/usr/bin/env python
"""
baseline_tabular.py — what do the five tabular numbers alone get you?

Section 8.3 of analysis_plans/RETRAIN_PLAN_2026-08-01.md requires this and
nothing implements it: "the tabular-only GBM is refitted on the same split."

WHY IT IS ON THE CRITICAL PATH. If a gradient-boosted model over five numbers
already in the pool table matches the sequence model, the sequence model has not
earned its place in the manuscript, and no amount of interpretability makes it
earn it. This is the first thing a reviewer asks and the answer should not be
discovered late.

WHAT IT IS GIVEN. Per transcript, aggregated over that transcript's candidates:
n_downstream_ejc, is_ref_cds, is_sqanti_cds, frac_start, frac_stop -- the same
five the PREDICTOR variant receives (§6.3), and four more than the interpretable
variant does. No sequence. Aggregation is max, mean and the slot-0 value, so the
baseline is not handicapped by having to pick one.

THE SPLIT IS THE MODEL'S OWN. Trained on `train`, tuned on `val_clean`, read once
on `test_clean`. The published §5 number this replaces was selected on the test
split -- twelve window combinations ranked by test AUC -- so it is optimistically
biased, and reproducing that defect while replacing it would be worse than
leaving it alone.

Usage:
    python baseline_tabular.py --tensor results_tensor_v6 --split test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
COLS = ["n_downstream_ejc", "is_ref_cds", "is_sqanti_cds", "frac_start", "frac_stop"]


def auc_roc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / c)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(y, s):
    """Average precision, with TIES GROUPED.

    The step-wise form breaks ties by array order, which is wrong: no threshold
    separates two identical scores, so they must be pooled. It matters here and
    not elsewhere -- a boosted tree's probabilities tie far more often than a
    network's logits do. sklearn's implementation groups correctly and is
    already a dependency of this script.
    """
    from sklearn.metrics import average_precision_score
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def build_features(tensor):
    """One row per transcript: max, mean and slot-0 of each of the five columns."""
    with h5py.File(Path(tensor) / "nmd_tensor.h5", "r") as f:
        off, cnt = f["offset"][:], f["count"][:]
        raw = f["structural_raw"][:] if "structural_raw" in f else f["structural"][:]
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        split = np.array([s.decode() for s in f["split"][:]])
        gene = np.array([s.decode() for s in f["gene_id"][:]])
        y = f["labels"][:].astype(np.int8)
    n = len(iso)
    X = np.zeros((n, len(COLS) * 3 + 1), dtype=np.float32)
    for i in range(n):
        s = slice(int(off[i]), int(off[i]) + int(cnt[i]))
        b = raw[s]
        X[i, :5] = b.max(axis=0)
        X[i, 5:10] = b.mean(axis=0)
        X[i, 10:15] = b[0]
        X[i, 15] = cnt[i]                      # candidate count, free and informative
    names = ([f"max_{c}" for c in COLS] + [f"mean_{c}" for c in COLS]
             + [f"slot0_{c}" for c in COLS] + ["n_candidates"])
    return X, y, split, gene, iso, names


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--seeds", default="100,200,300,400,500",
                    help="the model is a five-seed ensemble and the baseline was "
                         "a single fit; that asymmetry is not defensible, so the "
                         "baseline is fitted at the SAME five seeds and reported "
                         "as a mean with its range")
    ap.add_argument("--drop-cols", default="",
                    help="comma-separated column names to withhold. The baseline "
                         "exists to be a STRONG comparator, so weakening it "
                         "quietly would flatter the model for the wrong reason -- "
                         "it is run both ways and both are reported.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    t0 = time.time()

    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y, split, gene, iso, names = build_features(args.tensor)
    if args.drop_cols:
        drop = {c.strip() for c in args.drop_cols.split(",") if c.strip()}
        keep = [i for i, n in enumerate(names)
                if not any(n.endswith(d) for d in drop)]
        dropped = [n for i, n in enumerate(names) if i not in set(keep)]
        print(f"WITHHELD: {', '.join(sorted(drop))} -> dropped {len(dropped)} "
              f"of {len(names)} features")
        X, names = X[:, keep], [names[i] for i in keep]
    tr = split == "train"
    va = split == "val"
    ev = split == args.split
    print(f"tabular baseline over {len(names)} features from {len(COLS)} columns")
    print(f"  train {int(tr.sum()):,}   val {int(va.sum()):,}   "
          f"{args.split} {int(ev.sum()):,}")
    print(f"  features: {', '.join(names)}")
    for nm, m_ in (("train", tr), ("val", va), (args.split, ev)):
        if m_.sum() == 0:
            raise SystemExit(
                f"split '{nm}' is empty in {args.tensor}. This baseline fits on "
                f"train and tunes on val, so a single-chromosome tensor holding "
                f"only one split cannot run it. Splits present: "
                + ", ".join(f"{s}={int((split == s).sum()):,}"
                            for s in sorted(set(split.tolist()))))

    # Tuned on val_clean only, exactly as the sequence model's configuration is,
    # and fitted at the same five seeds the model uses.
    seeds = [int(x) for x in args.seeds.split(",")]
    per_seed, preds = [], []
    for sd in seeds:
        best, best_auc = None, -np.inf
        for lr in (0.03, 0.06, 0.1):
            for leaves in (15, 31, 63):
                m = HistGradientBoostingClassifier(
                    learning_rate=lr, max_leaf_nodes=leaves, max_iter=500,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=sd)
                m.fit(X[tr], y[tr])
                a = auc_roc(y[va], m.predict_proba(X[va])[:, 1])
                if a > best_auc:
                    best, best_auc = m, a
        pp = best.predict_proba(X[ev])[:, 1]
        preds.append(pp)
        per_seed.append((auc_roc(y[ev], pp), auprc(y[ev], pp), best_auc))
        print(f"  seed {sd}: val AUC {best_auc:.4f}   "
              f"{args.split} AUC {per_seed[-1][0]:.4f}  AUPRC {per_seed[-1][1]:.4f}")

    aucs = [x[0] for x in per_seed]; prs = [x[1] for x in per_seed]
    a, pr = float(np.mean(aucs)), float(np.mean(prs))
    p = np.mean(preds, axis=0)                     # the ensemble, matched to the model
    ea, epr = auc_roc(y[ev], p), auprc(y[ev], p)
    print(f"\n=== tabular baseline on {args.split}_clean ===")
    print(f"  seed mean  AUC {a:.4f} [{min(aucs):.4f}, {max(aucs):.4f}]   "
          f"AUPRC {pr:.4f} [{min(prs):.4f}, {max(prs):.4f}]")
    print(f"  ensemble   AUC {ea:.4f}   AUPRC {epr:.4f}   "
          f"<- the aggregation the model's headline uses")
    print(f"  n {int(ev.sum()):,}   n_nmd {int(y[ev].sum()):,}   "
          f"prevalence {y[ev].mean():.4f}")

    # gene- and transcript-resampled intervals, the same pair evaluate_v6 reports
    rng = np.random.default_rng(args.seed)
    g = gene[ev]
    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    groups = np.split(order, starts[1:])
    ng, nt = len(groups), int(ev.sum())
    bg, bt = np.empty(args.bootstrap), np.empty(args.bootstrap)
    yy = y[ev]
    for b in range(args.bootstrap):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        bg[b] = auc_roc(yy[idx], p[idx])
        it = rng.integers(0, nt, nt)
        bt[b] = auc_roc(yy[it], p[it])
    lo, hi = np.percentile(bg, [2.5, 97.5])
    print(f"  gene-resampled 95%       [{lo:.4f}, {hi:.4f}]  sd {bg.std():.5f}")
    lo2, hi2 = np.percentile(bt, [2.5, 97.5])
    print(f"  transcript-resampled 95% [{lo2:.4f}, {hi2:.4f}]  sd {bt.std():.5f}")
    print(f"  design effect on the metric (variance ratio): "
          f"{(bg.std()/bt.std())**2:.2f}")

    imp = sorted(zip(names, best.feature_importances_ if hasattr(
        best, "feature_importances_") else [0] * len(names)),
        key=lambda t: -t[1])[:8] if hasattr(best, "feature_importances_") else []
    if imp:
        print(f"\n  top features: " + ", ".join(f"{n} {v:.3f}" for n, v in imp))

    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(split=args.split, auc=a, auprc=pr, n=int(ev.sum()),
                 val_auc=best_auc, gene_ci=[lo, hi], tx_ci=[lo2, hi2],
                 features=names), indent=2))
        print(f"\nwrote {args.out}")
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
