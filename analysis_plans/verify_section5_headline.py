#!/usr/bin/env python
"""
verify_section5_headline.py — trace §5's headline numbers first-hand.

WHY. The claim list marks A1 (ensemble AUC/AUPRC) and A5 (tabular baseline) as
PENDING VERIFICATION. Both were reported by the model window and neither was
reproduced here. They are the two numbers §5 leads on, and the standing rule on
this thread is that a number is not repeated until it has been traced or
recomputed. The legacy comparator has been verified in detail; our own side has
not, which is the wrong way round.

INDEPENDENCE. This does not import the model window's metric code. AUC and AUPRC
are computed twice — once by a rank-based implementation written here, once by
sklearn — and both are printed. A disagreement between them is a bug in one of
them; agreement with the reported figure is the verification.

Both ensemble rules are computed. Mean logit is what the legacy comparator
declares; mean probability is what the model window chose as the headline, before
seeing the comparator's rule. Reporting both is what makes the choice not a
selection effect.

The tabular baseline is re-fitted from `baseline_tabular.py`'s own specification
rather than by calling it: sixteen features (max, mean and slot-0 of the five
structural columns, plus candidate count), the same hyperparameter grid, selected
on validation, test read once.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model_v6 import ScanningNMDModel                        # noqa: E402
from tensor_io import decode_windows                         # noqa: E402

TENSOR = REPO / "results_tensor_v6" / "nmd_tensor.h5"
CKPT = REPO / "results_interp_all" / "ckpt_interp_c32_b8"
COLS = ["n_downstream_ejc", "is_ref_cds", "is_sqanti_cds", "frac_start", "frac_stop"]

REPORTED = {"ens_logit": (0.9327, 0.8684), "ens_prob": (0.9322, 0.8679),
            "gbm": (0.8638, 0.7013)}


def auc_mine(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap_mine(y, s):
    o = np.argsort(-np.asarray(s, float), kind="stable")
    y = np.asarray(y)[o]
    tp = np.cumsum(y)
    return float(((tp / np.arange(1, len(y) + 1)) * y).sum() / y.sum())


def both(y, s, tag, reported=None):
    """AUC from my rank implementation and sklearn's; AUPRC from both, with
    sklearn's taken as authoritative.

    The two average-precision implementations agree exactly on distinct scores
    and diverge on TIES: the step-wise form breaks ties by array order, while
    sklearn groups tied scores because no threshold can separate them. Ties are
    expected here rather than incidental — model_v6 clamps the probability to
    [1e-6, 1-1e-6], so every saturated prediction carries the identical logit.
    So AUC is asserted equal and AUPRC is reported both ways with the gap shown.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    a1, a2 = auc_mine(y, s), roc_auc_score(y, s)
    p1, p2 = ap_mine(y, s), average_precision_score(y, s)
    flag = ""
    if reported:
        da, dp = abs(a1 - reported[0]), abs(p2 - reported[1])
        flag = ("  <- MATCHES reported" if da < 5e-4 and dp < 5e-4
                else f"  <- DIFFERS from reported {reported[0]}/{reported[1]} "
                     f"by {da:.4f}/{dp:.4f}")
    print(f"  {tag:<26} AUC {a1:.4f}   AUPRC {p2:.4f} "
          f"(step-wise {p1:.4f}, gap {abs(p1-p2):.2e}){flag}")
    assert abs(a1 - a2) < 1e-9, f"AUC implementations disagree by {abs(a1-a2):.2e}"
    return a1, p2


def batches(counts, max_padded=2048):
    order = np.argsort(counts, kind="stable")
    out, cur, cmax = [], [], 0
    for i in order:
        k = int(counts[i]); nm = max(cmax, k)
        if cur and (len(cur) + 1) * nm > max_padded:
            out.append(np.array(cur)); cur, cmax = [i], k
        else:
            cur.append(i); cmax = nm
    if cur:
        out.append(np.array(cur))
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    with h5py.File(TENSOR, "r") as f:
        split = np.array([s.decode() for s in f["split"][:]])
        labels = f["labels"][:].astype(int)
        off, cnt = f["offset"][:], f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        o_e = f["orf_end"][:].astype(np.int64)
        struct = f["structural"][:]
        raw = f["structural_raw"][:] if "structural_raw" in f else struct
        codes = f["codes"][:]
        L, SL = int(f.attrs["atg_left"]), int(f.attrs["stop_left"])

    te = np.flatnonzero(split == "test")
    print(f"test split: {len(te):,} transcripts, {int(labels[te].sum()):,} NMD "
          f"({100*labels[te].mean():.1f}%)")
    print("legacy comparator declares n 10520, n_nmd 2405\n")

    # ---- A1: the ensemble ---------------------------------------------------
    ckpts = sorted(CKPT.glob("b8_s*.pt"))
    LCACHE = REPO / "results_interp_all" / "test_logits_interp_c32_b8.npz"
    if LCACHE.exists():
        logits = np.load(LCACHE)["logits"]
        print(f"loaded cached test logits from {LCACHE.name}")
        ckpts_done = True
    else:
        ckpts_done = False
        logits = np.zeros((len(ckpts), len(te)))
    for si, cp in enumerate([] if ckpts_done else ckpts):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1, permute_bins=False)
        m.load_state_dict(ck["model"]); m.eval()
        pos = {int(t): j for j, t in enumerate(te)}
        with torch.no_grad():
            for b in batches(cnt[te]):
                sel = te[b]
                c = cnt[sel].astype(int); K = int(c.max())
                rows = np.concatenate([np.arange(off[i], off[i] + cnt[i]) for i in sel])
                s0, e0 = o_s[rows], o_e[rows]
                atg = decode_windows(codes[rows][:, 0], s0, L, s0)
                stp = decode_windows(codes[rows][:, 1], e0 - 1, SL, s0)
                st = struct[rows][:, [0]]
                W = atg.shape[2]
                A = np.zeros((len(sel), K, 9, W), np.float32); S = np.zeros_like(A)
                U = np.zeros((len(sel), K, 1), np.float32)
                M = np.zeros((len(sel), K), bool)
                at = 0
                for j, cc in enumerate(c):
                    A[j, :cc] = atg[at:at+cc]; S[j, :cc] = stp[at:at+cc]
                    U[j, :cc] = st[at:at+cc]; M[j, :cc] = True; at += cc
                lg = m(torch.as_tensor(A), torch.as_tensor(S),
                       torch.as_tensor(U), torch.as_tensor(M)).numpy()
                for j, i in enumerate(sel):
                    logits[si, pos[int(i)]] = lg[j]
        print(f"  {cp.name}: done", flush=True)

    if not ckpts_done:
        np.savez_compressed(LCACHE, logits=logits)
        print(f"cached test logits -> {LCACHE.name}")
    y = labels[te]
    nt = len(y) - len(np.unique(logits.mean(0)))
    print(f"\nexact ties in the mean-logit ensemble score: {nt:,} of {len(y):,}")
    print("\n=== A1: per seed ===")
    for si, cp in enumerate(ckpts):
        both(y, logits[si], cp.name.replace(".pt", ""))
    per = [auc_mine(y, logits[si]) for si in range(len(ckpts))]
    print(f"  seed mean AUC {np.mean(per):.4f}   range "
          f"[{min(per):.4f}, {max(per):.4f}]")

    print("\n=== A1: ensemble, both rules ===")
    both(y, logits.mean(0), "mean logit", REPORTED["ens_logit"])
    p = 1 / (1 + np.exp(-logits))
    both(y, p.mean(0), "mean probability", REPORTED["ens_prob"])

    # ---- A5: the tabular baseline ------------------------------------------
    print("\n=== A5: tabular baseline, refitted from its own specification ===")
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(split)
    X = np.zeros((n, len(COLS) * 3 + 1), np.float32)
    for i in range(n):
        b = raw[off[i]:off[i] + cnt[i]]
        X[i, :5] = b.max(0); X[i, 5:10] = b.mean(0); X[i, 10:15] = b[0]
        X[i, 15] = cnt[i]
    tr, va = split == "train", split == "val"
    print(f"  train {int(tr.sum()):,}  val {int(va.sum()):,}  test {len(te):,}  "
          f"features {X.shape[1]}")
    best, best_auc = None, -np.inf
    for lr in (0.03, 0.06, 0.1):
        for lv in (15, 31, 63):
            g = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=lv,
                                               max_iter=500, early_stopping=True,
                                               validation_fraction=0.15,
                                               random_state=100)
            g.fit(X[tr], labels[tr])
            a = auc_mine(labels[va], g.predict_proba(X[va])[:, 1])
            if a > best_auc:
                best, best_auc = g, a
    print(f"  selected on val: AUC {best_auc:.4f}")
    both(y, best.predict_proba(X[te])[:, 1], "tabular GBM (test)", REPORTED["gbm"])


if __name__ == "__main__":
    main()
