# HYPOTHESIS 3 (interpretability window, untested when offered): upstream fill
# extent is a PROXY for candidate ordinal position, and the real driver of dead
# capture perturbations is SELECTION MASS. A candidate deep in the 5'->3' order
# carries stick-breaking mass ~0.5^k; a Delta z_p multiplied by 1e-5 rounds to
# exactly zero in the aggregate. Predicts a step (fill extent saturates at 900,
# ordinal position keeps climbing past it) rather than a slope.
#
# DISCRIMINATING TEST: if mass is the driver, the dead rate is strongly monotone in
# p_select, and fill extent's association VANISHES once p_select is held.
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
spans = f["spans"][:]; off = f["cand_offset"][:]; cnt = f["cand_count"][:]
c_start = f["cand_orf_start"][:]; p_sel = f["p_select"][:]
n = f["vals"].shape[0]
rng = np.random.default_rng(1)
take = rng.choice(n, 400, replace=False)
E, D, M = [], [], []
for i in take:
    v = f["valid"][i]; cap = f["vals_capture"][i]
    sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
    sp = spans[sl]; st = c_start[sl]; ps = p_sel[sl]
    W = len(v); cov = np.zeros(W, np.int16); who = np.full(W, -1, np.int32)
    for k, row in enumerate(sp):
        a_lo, a_hi = int(row[2]), int(row[3])
        if a_hi >= a_lo:
            cov[max(0, a_lo-1):a_hi] += 1; who[max(0, a_lo-1):a_hi] = k
    fin = np.isfinite(cap).any(1) & v
    with np.errstate(all="ignore"):
        mx = np.nanmax(np.abs(np.where(np.isfinite(cap), cap, np.nan)), 1)
    dead = fin & (mx == 0.0); sel = fin & (cov == 1)
    if not sel.any(): continue
    k = who[sel]
    E.append((st[k] - sp[k][:, 2]).astype(np.int32)); D.append(dead[sel]); M.append(ps[k])
E = np.concatenate(E); D = np.concatenate(D); M = np.concatenate(M)
print(f"  singly-covered ATG positions {len(E):,}   dead {100*D.mean():.2f}%\n")
print("  === dead rate by SELECTION MASS ===")
me = [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.01]
for lo, hi in zip(me[:-1], me[1:]):
    m = (M >= lo) & (M < hi)
    if m.sum() < 50: continue
    print(f"    p_select [{lo:g}, {hi:g}){'':<4} n {int(m.sum()):>8,}   dead {100*D[m].mean():>7.2f}%")
print("\n  === dead rate by FILL EXTENT, HOLDING selection mass ===")
for lo, hi in zip(me[:-1], me[1:]):
    st_m = (M >= lo) & (M < hi)
    if st_m.sum() < 200: continue
    sat = st_m & (E >= 900); uns = st_m & (E < 900)
    if sat.sum() < 30 or uns.sum() < 30: continue
    print(f"    p_select [{lo:g}, {hi:g}): saturated {100*D[sat].mean():>6.2f}% "
          f"(n {int(sat.sum()):,})   truncated {100*D[uns].mean():>6.2f}% (n {int(uns.sum()):,})")
print(f"\n  r(extent, dead)          = {np.corrcoef(E, D.astype(float))[0,1]:+.4f}")
print(f"  r(log10 p_select, dead)  = {np.corrcoef(np.log10(np.maximum(M,1e-30)), D.astype(float))[0,1]:+.4f}")
