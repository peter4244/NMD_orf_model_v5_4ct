#!/usr/bin/env python
"""Independent recomputation of the §5 headline (claim A1) from cached test logits.

Does not import the model window's or the interpretability window's metric code.
AUC computed three ways (Mann-Whitney U on ranks, trapezoid over the ROC, sklearn);
AUPRC two ways (step-wise, sklearn). Labels read straight from the tensor.
"""
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

REPO = Path.home() / "claude_projects" / "NMD_orf_model_v5_4ct"
TENSOR = REPO / "results_tensor_v6" / "nmd_tensor.h5"
LCACHE = REPO / "results_interp_all" / "test_logits_interp_c32_b8.npz"

with h5py.File(TENSOR, "r") as f:
    split = np.array([s.decode() for s in f["split"][:]])
    labels = f["labels"][:].astype(int)

te = np.flatnonzero(split == "test")
y = labels[te]
print(f"test split from the tensor: n={len(te):,}  n_nmd={int(y.sum()):,}  "
      f"prevalence={y.mean():.4f}")
print(f"legacy comparator declares  n=10520     n_nmd=2405")
print(f"match: n {'OK' if len(te)==10520 else 'MISMATCH'}, "
      f"n_nmd {'OK' if int(y.sum())==2405 else 'MISMATCH'}\n")

logits = np.load(LCACHE)["logits"]
assert logits.shape[1] == len(te), f"logit columns {logits.shape[1]} != test n {len(te)}"
print(f"cached logits {logits.shape} from {LCACHE.name}\n")


def auc_mw(y, s):
    """Mann-Whitney U over midranks."""
    r = rankdata(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_trapz(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    # np.trapezoid on numpy>=2, np.trapz on 1.x; this runs under both envs.
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


def ap_stepwise(y, s):
    o = np.argsort(-np.asarray(s, float), kind="stable")
    yy = np.asarray(y)[o]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, len(yy) + 1)) * yy).sum() / yy.sum())


def report(tag, s, reported):
    a_mw, a_tz, a_sk = auc_mw(y, s), auc_trapz(y, s), roc_auc_score(y, s)
    p_sw, p_sk = ap_stepwise(y, s), average_precision_score(y, s)
    spread = max(abs(a_mw - a_tz), abs(a_mw - a_sk))
    ok = abs(a_mw - reported[0]) < 5e-4 and abs(p_sk - reported[1]) < 5e-4
    print(f"{tag}")
    print(f"    AUC    mannwhitney {a_mw:.6f}   trapezoid {a_tz:.6f}   "
          f"sklearn {a_sk:.6f}   (max spread {spread:.2e})")
    print(f"    AUPRC  stepwise    {p_sw:.6f}   sklearn   {p_sk:.6f}   "
          f"(gap {abs(p_sw - p_sk):.2e})")
    print(f"    reported {reported[0]} / {reported[1]}  ->  "
          f"{'MATCHES' if ok else 'DIFFERS'}\n")
    return a_mw, p_sk


print("=== per member ===")
for i in range(logits.shape[0]):
    a = auc_mw(y, logits[i]); p = average_precision_score(y, logits[i])
    print(f"    member {i}    AUC {a:.4f}   AUPRC {p:.4f}")
print()

print("=== ensemble ===")
report("mean logit        (the rule the legacy comparator declares)",
       logits.mean(axis=0), (0.9327, 0.8684))

probs = 1.0 / (1.0 + np.exp(-logits))
report("mean probability  (the rule the model window chose first)",
       probs.mean(axis=0), (0.9322, 0.8679))

ties = len(logits.mean(axis=0)) - len(np.unique(logits.mean(axis=0)))
print(f"exact ties in the mean-logit score: {ties} of {len(te):,}")
