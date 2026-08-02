"""
model_one_track.py — is the model a single junction-detector wearing two names?

SCOPE: a claim ABOUT THE MODEL.

PETE, 2026-08-02: "It's basically got a one track mind for PTCs, even though it may
not look like that."

THE EVIDENCE THAT MOTIVATES IT. Among short candidates the INITIATION head predicts
decay at +0.362 and its top-scoring short candidate carries a downstream junction
76.3% of the time -- while being the 5'-most only 22.8% of the time. So the head is
not applying a positional prior; it has learned to detect junction-relevance from
the start window. Both heads are junction detectors reading different windows.

THE PREDICTION THAT FOLLOWS. If the model has ONE detector, it should have no
channel at all for NMD that is not junction-dependent. Transcripts labelled NMD
whose candidates carry NO downstream junction should be scored like controls,
because nothing in either head can see why they decay.

REGISTERED BEFORE THE RUN:

  ONE-TRACK   among NMD transcripts, those with no junction-bearing candidate are
              scored close to CONTROL level -- the NMD/control separation collapses
              in that stratum. The model cannot represent non-junction NMD.
  BROADER     the separation survives in that stratum, so the model has some other
              channel and "one track" is too strong.
  UNDERPOWERED fewer than 50 NMD transcripts lack a junction-bearing candidate;
              report and draw nothing.

Two predictions of P(NMD) are computed and cross-checked: sum_k p_select_k * d_k
from the per-candidate arrays, and sigmoid(base_logit) from the unperturbed forward
pass. If they disagree the arrays are being misread and nothing below is
trustworthy.

Run from the repo root.
"""
import numpy as np, h5py

f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
off, cnt = f["cand_offset"][:], f["cand_count"][:]
ps_, pd_, ejc_ = f["p_select"][:], f["p_decay"][:], f["cand_n_downstream_ejc"][:]
lab, bl = f["labels"][:], f["base_logit"][:]
N = len(cnt)

pnmd, has_j, keep = [], [], []
for i in range(N):
    lo, k = int(off[i]), int(cnt[i])
    if k < 1:
        continue
    ps, pd = ps_[lo:lo + k], pd_[lo:lo + k]
    pnmd.append(float((ps * pd).sum()))
    has_j.append(bool((ejc_[lo:lo + k] > 0).any()))
    keep.append(i)

pnmd = np.array(pnmd); has_j = np.array(has_j); keep = np.array(keep)
y = lab[keep].astype(int)
sig = 1.0 / (1.0 + np.exp(-bl[keep].astype(float)))

print("CROSS-CHECK -- two routes to P(NMD) must agree")
print(f"  sum p*d vs sigmoid(base_logit):  r = {np.corrcoef(pnmd, sig)[0,1]:.4f}"
      f"   max|diff| = {np.abs(pnmd - sig).max():.2e}")
print(f"  medians  {np.median(pnmd):.4f}  vs  {np.median(sig):.4f}\n")

print("=" * 72)
print("IS THERE A CHANNEL FOR NON-JUNCTION NMD?")
print("=" * 72)
print(f"  {'stratum':<34} {'n':>6} {'median P(NMD)':>14}")
for hj, hn in ((True, "has junction-bearing cand"), (False, "NO junction-bearing cand")):
    for lv, ln in ((1, "NMD"), (0, "control")):
        m = (has_j == hj) & (y == lv)
        if m.sum():
            print(f"  {hn + ', ' + ln:<34} {int(m.sum()):>6} {np.median(pnmd[m]):>14.4f}")

sep = {}
for hj in (True, False):
    a = pnmd[(has_j == hj) & (y == 1)]
    b = pnmd[(has_j == hj) & (y == 0)]
    if len(a) and len(b):
        # AUC as the separation measure -- scale-free, no threshold
        allv = np.concatenate([a, b]); r = np.argsort(np.argsort(allv)) + 1
        auc = (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        sep[hj] = (auc, len(a), len(b))
        print(f"\n  NMD-vs-control AUC, {'WITH' if hj else 'WITHOUT'} a "
              f"junction-bearing candidate: {auc:.3f}   (n {len(a):,} / {len(b):,})")

n_nojunc_nmd = int(((~has_j) & (y == 1)).sum())
if n_nojunc_nmd < 50:
    v = f"UNDERPOWERED -- only {n_nojunc_nmd} NMD transcripts lack one"
elif False in sep and True in sep:
    v = ("ONE-TRACK -- separation collapses without a junction-bearing candidate"
         if sep[False][0] < 0.60 else
         "BROADER -- the model retains a channel for non-junction NMD")
else:
    v = "INCONCLUSIVE"
print(f"\n  -> {v}")
