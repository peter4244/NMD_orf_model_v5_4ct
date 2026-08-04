# VERIFY THE RETRACTION. Claim: the |vals_decay| track is autocorrelated at every
# scale and never decays, so a random-placement null is wrong -- it destroys
# structure the data has ARCHITECTURALLY, and the run-length excess then measures
# smoothness rather than a sequence feature. And the smoothness is selection mass.
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
rng = np.random.default_rng(0)
take = rng.choice(f["vals_decay"].shape[0], 400, replace=False)
LAGS = [1,10,25,50,80]
raw = {L:[] for L in LAGS}; res = {L:[] for L in LAGS}
r_fill, r_mass = [], []
def ac(x, L):
    if len(x) <= L+10: return np.nan
    a, b = x[:-L], x[L:]
    if a.std() == 0 or b.std() == 0: return np.nan
    return float(np.corrcoef(a,b)[0,1])
for i in take:
    v=f["valid"][i]; fc=f["fill_count"][i].astype(float); ms=f["mass"][i].astype(float)
    x=np.abs(f["vals_decay"][i])
    with np.errstate(all="ignore"):
        eff=np.nanmax(np.where(np.isfinite(x),x,np.nan),1)
    ok=v&np.isfinite(eff)&(fc>0)
    if ok.sum()<300: continue
    e=eff[ok]; c=fc[ok]; m=np.log10(np.maximum(ms[ok],1e-30))
    le=np.log10(np.maximum(e,1e-30))
    if e.std()==0: continue
    r_fill.append(float(np.corrcoef(le,c)[0,1]))
    r_mass.append(float(np.corrcoef(le,m)[0,1]))
    # residualise log effect on coverage + log mass
    X=np.column_stack([np.ones(len(e)), c, m])
    beta,*_=np.linalg.lstsq(X, le, rcond=None)
    rsd = le - X@beta
    for L in LAGS:
        raw[L].append(ac(le,L)); res[L].append(ac(rsd,L))
print(f"  {len(r_fill)} transcripts\n")
print(f"  correlation of the log effect track with     median r")
print(f"    fill_count                                 {np.nanmedian(r_fill):+.3f}")
print(f"    log selection mass                         {np.nanmedian(r_mass):+.3f}")
print(f"\n  {'lag':>5} {'raw autocorr':>14} {'after coverage+mass':>21}")
for L in LAGS:
    print(f"  {L:>5} {np.nanmedian(raw[L]):>14.3f} {np.nanmedian(res[L]):>21.3f}")
