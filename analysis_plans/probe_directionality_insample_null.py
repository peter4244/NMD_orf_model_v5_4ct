"""
probe_directionality_null.py — the in-sample null for |mean_b vals| / max_b|vals|.

A RATIO STATISTIC HAS TWO REFERENCE POINTS AND BOTH WERE WRONG ONCE. The ceiling
is 0.75, not 1.0 -- three substitutions agreeing give mean 3v/4 against max v. The
floor is not 0: random signs give a MEAN of 0.375 and a MEDIAN of 0.25, and
neither describes this data, whose sub-floor band measures 0.387 median / 0.370
mean. Sub-floor positions are not sign-random; their substitutions agree far more
often than chance, so a shared per-position component dominates at that scale.

The null is therefore measured in-sample from positions below the bank's own
batch-shape floor, per statistic, and never taken from the analytic model. Reports
mean, median and median-of-medians, because the first two differ by 0.02 and that
difference is what made two windows disagree for an hour.
"""

import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
FLOOR = 1.66e-06          # the bank's own batch-shape floor, s100
pool_all, pool_sub, pool_hi = [], [], []
med_all, med_sub, med_hi = [], [], []
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 50: continue
    v = f["vals_decay"][i,:P].astype(np.float64)
    ok = f["valid"][i,:P] & np.isfinite(v).any(1)
    if ok.sum() < 100: continue
    vv = np.where(np.isfinite(v), v, 0.0)
    with np.errstate(invalid="ignore"):
        uns = np.nanmax(np.abs(np.where(np.isfinite(v), v, np.nan)), axis=1)
    good = ok & np.isfinite(uns) & (uns > 0)
    r = np.abs(vv.mean(1))[good] / uns[good]
    u = uns[good]
    sub = u < FLOOR
    k = max(1, int(round(0.01*good.sum())))
    hi = u >= np.partition(u, -k)[-k]
    pool_all.append(r); med_all.append(np.median(r))
    if sub.sum() > 20: pool_sub.append(r[sub]); med_sub.append(np.median(r[sub]))
    if hi.sum() > 5:   pool_hi.append(r[hi]);  med_hi.append(np.median(r[hi]))
def rep(name, pool, meds):
    p = np.concatenate(pool)
    print(f"  {name:<26} pooled median {np.median(p):.4f}   pooled mean {p.mean():.4f}"
          f"   median-of-medians {np.median(meds):.4f}   n {len(p):,}")
rep("all positions", pool_all, med_all)
rep("SUB-FLOOR (|eff|<1.66e-6)", pool_sub, med_sub)
rep("top 1%", pool_hi, med_hi)
