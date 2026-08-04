import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
rng = np.random.default_rng(0)
take = rng.choice(f["vals_decay"].shape[0], 500, replace=False)
num, den, rat = [], [], []
for i in take:
    v = f["valid"][i]; x = f["vals_decay"][i].astype(np.float64)
    fin = (np.isfinite(x).sum(1) == 3) & v
    if fin.sum() < 200: continue
    s = np.where(np.isfinite(x[fin]), x[fin], 0.0)
    m = np.abs(s).max(1); ok = m > 0
    n_ = np.abs(s[ok].mean(1)); d_ = m[ok]
    num.append(n_); den.append(d_); rat.append(n_/d_)
N = np.concatenate(num); D = np.concatenate(den); R = np.concatenate(rat)
print(f"  n = {len(R):,}\n")
print(f"  {'aggregation':<46} {'value':>8}")
print(f"  {'mean of ratios  (mine)':<46} {R.mean():>8.4f}")
print(f"  {'ratio of means  sum|mean| / sum max':<46} {N.sum()/D.sum():>8.4f}")
print(f"  {'median of ratios':<46} {np.median(R):>8.4f}")
print(f"  {'mean of ratios, magnitude-weighted by max':<46} {(R*D).sum()/D.sum():>8.4f}")
print(f"\n  the other window reports 0.391 for 'all positions'")
for nm, val in (("mean of ratios", R.mean()), ("ratio of means", N.sum()/D.sum()),
                ("median of ratios", np.median(R)),
                ("magnitude-weighted", (R*D).sum()/D.sum())):
    print(f"    {nm:<28} {val:.4f}   gap {0.391-val:+.4f}")
