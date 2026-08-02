"""
model_c16_retraction.py — the producer for C16's retraction.

SCOPE: a claim ABOUT THE MODEL.

WHY THIS FILE EXISTS. The C16 retraction was run from an uncommitted temp file.
The conclusion was committed; the producer, the runlog and the job id were not.
The interpretability window caught it and put it exactly right: **the retraction
was less well-evidenced than the claim it retracted** -- C16 had a committed
script, a runlog, a job id and predictions registered before the run, and its
retraction had prose. That is the failure this project exists to prevent,
committed inside a retraction performed for that same failure.

AND IT CAUSED A REAL NUMBER DISCREPANCY. The narrative's table reported +0.452
while job 8900685 reported +0.442 for what reads as the same quantity. Cause:
the two runs used different candidate-count filters, k>=6 against k>=4. One name,
two sets, in the table that answers Pete's founding question. **Both filters are
reported here so the difference is visible rather than resolved by preference.**

THE QUESTION. C16 claimed that holding ORF length, routing goes toward
junction-bearing candidates at +0.442. The retraction: holding length UNMASKS
POSITION -- at equal length an earlier start has more sequence behind it, and the
5'->3' queue favours upstream -- so the partial measures position, not junction
structure.

  p_select ~ ejc, conditioning on nothing / length / position / both

Second-order partial by rank-residual regression: rank-transform all four
variables, regress the two of interest on both covariates, correlate the
residuals. The one-covariate partial uses the closed form, and the two agree at
one covariate, which is asserted below.

REGISTERED (as originally, before the first run):
  SURVIVES (>0.20)  junction structure is a real second channel; C16 stands
  COLLAPSES (<0.10) C16 is position re-entering and must be retracted
  BETWEEN            unresolved

Run from the repo root.
"""
import numpy as np, h5py

f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
off, cnt = f["cand_offset"][:], f["cand_count"][:]
ps_, ej_ = f["p_select"][:], f["cand_n_downstream_ejc"][:]
os_, oe_ = f["cand_orf_start"][:], f["cand_orf_end"][:]

def rk(x):
    r = np.argsort(np.argsort(x)).astype(float)
    return (r - r.mean()) / (r.std() + 1e-12)

def sp(x, y):
    if len(x) < 4: return np.nan
    a, b = rk(x), rk(y)
    if a.std() == 0 or b.std() == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def par(a, b, cs):
    """partial of a~b holding 1 or 2 covariates, by rank-residual regression"""
    R = [rk(v) for v in (a, b, *cs)]
    X = np.column_stack([*R[2:], np.ones(len(a))])
    try: B = np.linalg.lstsq(X, np.column_stack([R[0], R[1]]), rcond=None)[0]
    except Exception: return np.nan
    E = np.column_stack([R[0], R[1]]) - X @ B
    if E[:, 0].std() < 1e-9 or E[:, 1].std() < 1e-9: return np.nan
    return float(np.corrcoef(E[:, 0], E[:, 1])[0, 1])

def closed(a, b, c):
    rab, rac, rbc = sp(a, b), sp(a, c), sp(b, c)
    if not all(np.isfinite([rab, rac, rbc])): return np.nan
    d = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
    return float((rab - rac * rbc) / d) if d > 1e-9 else np.nan

for KMIN in (4, 6):
    R = {k: [] for k in ("raw", "len", "pos", "both", "chk")}
    for i in range(len(cnt)):
        lo, k = int(off[i]), int(cnt[i])
        if k < KMIN: continue
        ps = ps_[lo:lo + k]; ej = ej_[lo:lo + k].astype(float)
        ln = (oe_[lo:lo + k] - os_[lo:lo + k]).astype(float); st = os_[lo:lo + k].astype(float)
        R["raw"].append(sp(ps, ej)); R["len"].append(par(ps, ej, (ln,)))
        R["pos"].append(par(ps, ej, (st,))); R["both"].append(par(ps, ej, (ln, st)))
        R["chk"].append(closed(ps, ej, ln))
    def m(k):
        x = np.array(R[k], float); x = x[np.isfinite(x)]; return np.median(x), len(x)
    print(f"\n{'='*70}\nCANDIDATE FILTER  k >= {KMIN}\n{'='*70}")
    for lab, key in (("conditioning on nothing", "raw"), ("holding ORF LENGTH", "len"),
                     ("holding POSITION", "pos"), ("holding LENGTH AND POSITION", "both")):
        v, n = m(key); print(f"  p_select ~ ejc, {lab:<30} {v:+.3f}   n {n:,}")
    a, _ = m("len"); b, _ = m("chk")
    print(f"  [check] one-covariate partial, regression {a:+.3f} vs closed form {b:+.3f}"
          f"  -> {'AGREE' if abs(a-b) < 0.01 else 'DISAGREE'}")
    v, _ = m("both")
    print(f"  -> {'SURVIVES' if v > 0.20 else 'COLLAPSES — C16 retracted' if v < 0.10 else 'BETWEEN'}")
print("\nThe k>=4 row is the one the narrative should quote: it matches job 8900685's")
print("population. The k>=6 row is what the uncommitted temp file used, and is the")
print("source of the +0.452 that appeared in the table against 8900685's +0.442.")
