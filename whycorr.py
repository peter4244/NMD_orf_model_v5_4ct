import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
LAGS=[1,5,10,25,50,80]
raw={L:[0.0,0] for L in LAGS}; res={L:[0.0,0] for L in LAGS}
cor_fill=[]; cor_mass=[]
for i in range(400):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 400: continue
    v=f["vals_decay"][i,:P].astype(np.float64)
    fc=f["fill_count"][i,:P].astype(np.float64); ms=f["mass"][i,:P].astype(np.float64)
    ok=f["valid"][i,:P]&np.isfinite(v).any(1)
    with np.errstate(invalid="ignore"): e=np.nanmax(np.abs(v),axis=1)
    g=ok&np.isfinite(e)
    if g.sum()<400: continue
    idx=np.flatnonzero(g)
    r=np.full(P,np.nan); r[idx]=np.argsort(np.argsort(e[idx]))/max(len(idx)-1,1)
    # how much of the track is explained by coverage and mass?
    X=np.column_stack([fc[idx], np.log1p(np.maximum(ms[idx],0)), np.ones(len(idx))])
    y=r[idx]
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    pred=X@beta
    cor_fill.append(np.corrcoef(fc[idx],y)[0,1] if fc[idx].std()>0 else np.nan)
    cor_mass.append(np.corrcoef(np.log1p(np.maximum(ms[idx],0)),y)[0,1] if ms[idx].std()>0 else np.nan)
    rr=np.full(P,np.nan); rr[idx]=y-pred            # residual after coverage+mass
    for nm,arr in (("raw",r),("res",rr)):
        a0=arr-np.nanmean(arr); sd=np.nanstd(a0)
        if not np.isfinite(sd) or sd==0: continue
        for L in LAGS:
            x1,x2=a0[:-L],a0[L:]; m=np.isfinite(x1)&np.isfinite(x2)
            if m.sum()>50:
                d=(raw if nm=="raw" else res)[L]
                d[0]+=float((x1[m]*x2[m]).sum())/(sd*sd); d[1]+=int(m.sum())
print(f"  correlation of the effect track with coverage / mass, per transcript")
print(f"    fill_count  median r {np.nanmedian(cor_fill):+.3f}")
print(f"    log mass    median r {np.nanmedian(cor_mass):+.3f}")
print(f"\n  autocorrelation, before and after removing coverage + mass")
print(f"    {'lag':>5} {'raw':>9} {'residual':>10}")
for L in LAGS:
    a=raw[L][0]/max(raw[L][1],1); b2=res[L][0]/max(res[L][1],1)
    print(f"    {L:>5} {a:>9.4f} {b2:>10.4f}")
