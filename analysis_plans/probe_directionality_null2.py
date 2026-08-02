import h5py, numpy as np
rng = np.random.default_rng(0)
# analytic, equal magnitudes: ratio is 0.75 w.p. 2/8, 0.25 w.p. 6/8
print("  RANDOM-SIGN NULL depends on the summary statistic:")
N = 2_000_000
for nm, mag in (("equal magnitudes", np.ones((N,3))),
                ("exponential", rng.exponential(1,(N,3))),
                ("half-normal", np.abs(rng.normal(0,1,(N,3))))):
    v = rng.choice([-1.,1.],(N,3))*mag
    xx = np.concatenate([v, np.zeros((N,1))],1)
    m = np.abs(xx).max(1); ok = m>0
    r = np.abs(xx[ok].mean(1))/m[ok]
    print(f"    {nm:<22} mean {r.mean():.4f}   median {np.median(r):.4f}")
print("    analytic, equal mag:   mean 0.3750   median 0.2500")
print()
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
floor = float(f.attrs["batch_shape_offset"])
take = rng.choice(f["vals_decay"].shape[0], 500, replace=False)
R, M = [], []
for i in take:
    v=f["valid"][i]; x=f["vals_decay"][i].astype(np.float64)
    fin=(np.isfinite(x).sum(1)==3)&v
    if fin.sum()<200: continue
    s=np.where(np.isfinite(x[fin]),x[fin],0.0); mm=np.abs(s).max(1); ok=mm>0
    R.append(np.abs(s[ok].mean(1))/mm[ok]); M.append(mm[ok])
R=np.concatenate(R); M=np.concatenate(M)
sub = M < floor          # pure noise: below the batch-shape floor
top = M >= np.quantile(M,0.99)
print("  MEASURED, s100, 500 tx:")
print(f"    {'set':<28} {'mean':>8} {'median':>8} {'n':>12}")
for nm, msk in (("all positions", np.ones(len(R),bool)),
                ("sub-floor (pure noise)", sub),
                ("top 1% by |effect|", top)):
    print(f"    {nm:<28} {R[msk].mean():>8.4f} {np.median(R[msk]):>8.4f} {int(msk.sum()):>12,}")
