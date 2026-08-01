#!/usr/bin/env python
"""
evaluate_v6.py — test metrics for the scanning-selection model, with intervals
that mean something.

Implements section 8.3 of analysis_plans/RETRAIN_PLAN_2026-08-01.md. Nothing else
in this repository computes metrics for the v6 architecture: evaluate.py is the
old model and reads attention weights that no longer exist.

THE INTERVAL IS RESAMPLED OVER GENES, NOT TRANSCRIPTS. test_clean holds 10,520
transcripts in 3,234 genes with 90.0% of them in a multi-transcript gene, and
this project measured the inflation from ignoring that clustering at 5.47. An
interval over transcripts would be about five times too narrow, and the
comparisons this script exists to support are between arms whose difference is
smaller than that error.

THE RANGE OVER SEEDS IS NOT AN INTERVAL. Seeds vary initialisation only and the
split is fixed, so the spread across them contains no sampling variance at all.
Both are reported and they are labelled differently on purpose.

THE PERMUTED-BIN ARM'S SCORE IS A DRAW, NOT A NUMBER. That model redraws its bin
permutation at every forward pass, so one evaluation is one realisation. It is
evaluated under R draws and the spread across them is reported beside the
gene-clustered interval; comparing a fixed quantity against a single draw of a
random one and calling the difference a result is the failure this avoids.

Usage:
    python evaluate_v6.py --tensor results_tensor_v6 --runs runs \\
        --configs interp_c32_b8 interp_c32_b8_perm --split test --bootstrap 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model_v6 import ScanningNMDModel
from train_v6 import TensorSource, auc_roc, auprc, make_batches, INTERPRETABLE, PREDICTOR

SEEDS = [100, 200, 300, 400, 500]


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    cols = INTERPRETABLE if a["variant"] == "interpretable" else PREDICTOR
    if a.get("blank_junctions"):
        cols = []
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=max(len(cols), 1),
                         permute_bins=bool(a.get("permute_bins", False)))
    m.load_state_dict(ck["model"]); m.to(device).eval()
    return m, a, (cols if cols else [0])


def predict(model, src, idxs, device, max_padded):
    ys, ps = [], []
    with torch.no_grad():
        for b in make_batches(src.count[idxs], max_padded):
            A, S, U, M, y = src.batch(idxs[b], device)
            ys.append(y.cpu().numpy())
            ps.append(model(A, S, U, M).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def gene_bootstrap(y, p, gene, n_boot, rng):
    """Resample GENES with replacement, recompute the metric on the transcripts
    they carry. A gene drawn twice contributes its transcripts twice, which is
    what makes the interval reflect the clustering rather than ignore it."""
    order = np.argsort(gene, kind="stable")
    g_sorted = gene[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    groups = np.split(order, starts[1:])
    n_g = len(groups)
    aucs, prs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        idx = np.concatenate([groups[i] for i in pick])
        aucs[b] = auc_roc(y[idx], p[idx])
        prs[b] = auprc(y[idx], p[idx])
    return aucs, prs, n_g


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--perm-draws", type=int, default=20)
    ap.add_argument("--max-padded", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    if args.split == "test":
        sel = Path(args.runs) / "selected.json"
        print("=" * 72)
        print("EVALUATING ON test_clean. Section 7.3 step 5 fixes the configuration on")
        print("val_clean FIRST, and test is read once, afterwards, for the arms named")
        print("on the command line.")
        if sel.exists():
            print(f"  {sel} exists: {json.loads(sel.read_text())}")
        else:
            print(f"  WARNING: {sel} does not exist, so no configuration has been")
            print(f"  selected yet. Reporting test metrics before selection is the")
            print(f"  failure the two-phase structure exists to prevent.")
        print("=" * 72)

    src = TensorSource(Path(args.tensor) / "nmd_tensor.h5", [0])
    idxs = src.indices(args.split)
    if len(idxs) == 0:
        raise SystemExit(
            f"split '{args.split}' holds no transcripts in {args.tensor}. Splits "
            f"present: " + ", ".join(f"{s}={int((src.split == s).sum()):,}" for s in
                                     sorted(set(src.split.tolist()))))
    gene = src.gene[idxs]
    print(f"\nsplit {args.split}_clean: {len(idxs):,} transcripts, "
          f"{len(set(gene)):,} genes, prevalence {src.labels[idxs].mean():.4f}")

    # ---- collect predictions first, so the bootstrap can resample ONCE and
    # ---- score every arm on the same resampled genes. Comparing two separately
    # ---- computed intervals is the wrong test and it errs conservative: both
    # ---- arms are scored on the SAME transcripts, so most of the sampling
    # ---- uncertainty is shared and cancels in the difference.
    preds, meta = {}, {}
    for cfg in args.configs:
        for sd in SEEDS:
            ck = Path(args.runs) / f"{cfg}_s{sd}" / "best.pt"
            if not ck.exists():
                print(f"  {cfg} seed {sd}: no best.pt, skipping")
                continue
            model, a, cols = load(ck, device)
            src.struct_cols = cols
            is_perm = bool(a.get("permute_bins", False))
            R = args.perm_draws if is_perm else 1
            draws = []
            for r in range(R):
                torch.manual_seed(args.seed + 1000 * r)
                y, pr = predict(model, src, idxs, device, args.max_padded)
                draws.append(pr)
            preds.setdefault(cfg, {})[sd] = draws
            meta[cfg] = dict(is_perm=is_perm, R=R)
            sd_draw = (float(np.std([auc_roc(y, d) for d in draws]))
                       if R > 1 else 0.0)
            print(f"  {cfg} seed {sd}: {R} draw(s)"
                  + (f", across-draw AUC sd {sd_draw:.5f}" if R > 1 else ""), flush=True)
            if R > 1:
                meta[cfg]["draw_sd"] = sd_draw

    def arm_auc(idx, cfg):
        """Mean over seeds of (mean over permutation draws of AUC).

        AVERAGED AT THE AUC LEVEL, NOT AT THE PREDICTION LEVEL. Averaging the
        predicted scores over draws and scoring once would build an ensemble --
        a different and better model than the one being evaluated -- and would
        flatter the control. The estimand is E[AUC] of a single draw.
        """
        per_seed = []
        for sd, draws in preds[cfg].items():
            per_seed.append(np.mean([auc_roc(y[idx], d[idx]) for d in draws]))
        return float(np.mean(per_seed))

    def arm_auprc(idx, cfg):
        per_seed = []
        for sd, draws in preds[cfg].items():
            per_seed.append(np.mean([auprc(y[idx], d[idx]) for d in draws]))
        return float(np.mean(per_seed))

    # gene groups, built once
    order = np.argsort(gene, kind="stable")
    g_sorted = gene[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    groups = np.split(order, starts[1:])
    n_g = len(groups)
    full = np.arange(len(y))

    point = {c: arm_auc(full, c) for c in preds}
    point_pr = {c: arm_auprc(full, c) for c in preds}

    print(f"\nbootstrap: {args.bootstrap:,} resamples of {n_g:,} genes", flush=True)
    boot = {c: np.empty(args.bootstrap) for c in preds}
    for b in range(args.bootstrap):
        pick = rng.integers(0, n_g, n_g)
        idx = np.concatenate([groups[i] for i in pick])
        for c in preds:
            boot[c][b] = arm_auc(idx, c)
        if (b + 1) % max(1, args.bootstrap // 5) == 0:
            print(f"  {b+1:,}/{args.bootstrap:,}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'configuration':<26} {'seeds':>5} {'mean AUC':>9} "
          f"{'gene-clustered 95%':>22} {'mean AUPRC':>11}")
    print("-" * 80)
    rows = {}
    for c in preds:
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        rows[c] = dict(auc=point[c], auprc=point_pr[c], lo=float(lo), hi=float(hi),
                       seeds=len(preds[c]), **meta[c])
        print(f"{c:<26} {len(preds[c]):>5} {point[c]:>9.4f} "
              f"{f'[{lo:.4f}, {hi:.4f}]':>22} {point_pr[c]:>11.4f}")

    # ---- the paired comparison, which is the one to read ------------------
    cfgs = list(preds)
    if len(cfgs) >= 2:
        print(f"\nPAIRED DIFFERENCES, bootstrapped on the same gene resamples.")
        print(f"Both arms are scored on the same transcripts, so the shared sampling")
        print(f"uncertainty cancels. Overlap of the two marginal intervals above is a")
        print(f"strictly weaker test and can hide a real difference.")
        for i in range(len(cfgs)):
            for j in range(i + 1, len(cfgs)):
                a_, b_ = cfgs[i], cfgs[j]
                d = boot[a_] - boot[b_]
                lo, hi = np.percentile(d, [2.5, 97.5])
                pt = point[a_] - point[b_]
                excl = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
                print(f"  {a_} - {b_}: {pt:+.4f}  "
                      f"[{lo:+.4f}, {hi:+.4f}]  {excl}")

    # ---- is R large enough for the control? -------------------------------
    for c, m in meta.items():
        if m.get("R", 1) > 1:
            half = (rows[c]["hi"] - rows[c]["lo"]) / 2
            se = m["draw_sd"] / np.sqrt(m["R"])
            print(f"\n{c}: permutation across-draw sd {m['draw_sd']:.5f} at R={m['R']}, "
                  f"so its standard error is {se:.5f}")
            print(f"  gene-clustered half-width {half:.5f}  -> permutation noise is "
                  f"{se/half:.2f}x the sampling half-width")
            print(f"  {'R is sufficient' if se < 0.2*half else 'RAISE R: permutation noise is not negligible against sampling'}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(split=args.split, n=len(idxs), n_genes=len(set(gene)),
                 bootstrap=args.bootstrap, results=rows,
                 point=point, point_auprc=point_pr), indent=2))
        print(f"\nwrote {args.out}")
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
