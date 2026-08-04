# HYPOTHESIS (interpretability window, untested when offered): the inside-ATG
# dead-perturbation rate RISES with upstream fill extent, because with more filled
# positions competing, any single one is less likely to be the argmax of its bin.
#
# Restricted to positions covered by EXACTLY ONE ATG window. A dead capture
# perturbation at a multiply-covered position needs every covering window to be
# unchanged, so the covariate is ill-defined there; one window makes the
# attribution clean at the cost of a subset.
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
spans = f["spans"][:]; off = f["cand_offset"][:]; cnt = f["cand_count"][:]
c_start = f["cand_orf_start"][:]
n = f["vals"].shape[0]
rng = np.random.default_rng(1)
take = rng.choice(n, 400, replace=False)
ext_all, dead_all = [], []
for i in take:
    v = f["valid"][i]; cap = f["vals_capture"][i]
    sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
    sp = spans[sl]; st = c_start[sl]
    W = len(v)
    cov = np.zeros(W, np.int16); who = np.full(W, -1, np.int32)
    for k, row in enumerate(sp):
        a_lo, a_hi = int(row[2]), int(row[3])
        if a_hi >= a_lo:
            cov[max(0, a_lo-1):a_hi] += 1
            who[max(0, a_lo-1):a_hi] = k
    fin = np.isfinite(cap).any(1) & v
    with np.errstate(all="ignore"):
        mx = np.nanmax(np.abs(np.where(np.isfinite(cap), cap, np.nan)), 1)
    dead = fin & (mx == 0.0)
    sel = fin & (cov == 1)
    if not sel.any():
        continue
    k = who[sel]
    # upstream fill extent of that candidate: filled positions 5' of its start
    up = (st[k] - sp[k][:, 2]).astype(np.int32)      # orf_start - a_lo
    ext_all.append(up); dead_all.append(dead[sel])
ext = np.concatenate(ext_all); dead = np.concatenate(dead_all)
print(f"  singly-covered ATG positions: {len(ext):,}   dead: {int(dead.sum()):,} "
      f"({100*dead.mean():.2f}%)")
edges = [0, 50, 100, 200, 400, 600, 900, 10**9]
print(f"  {'upstream fill extent':<24} {'n':>10} {'dead rate':>10}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (ext >= lo) & (ext < hi)
    if m.sum() < 50: continue
    print(f"  {f'{lo}-{hi if hi<10**9 else 900}':<24} {int(m.sum()):>10,} "
          f"{100*dead[m].mean():>9.2f}%")
if dead.sum() and dead.sum() < len(dead):
    print(f"  point-biserial r(extent, dead) = {np.corrcoef(ext, dead.astype(float))[0,1]:+.4f}")
