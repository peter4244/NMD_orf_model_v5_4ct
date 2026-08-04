import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
ps, pd_ = f["p_select"][:], f["p_decay"][:]
bl = f["base_logit"][:]
def lg(p): p=np.clip(p,1e-300,1-1e-300); return np.log(p)-np.log1p(-p)
keep_v, keep_d, keep_m = [], [], []
for i in range(300):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 200: continue
    vd = f["vals_decay"][i,:P].astype(np.float64)
    fc = f["fill_count"][i,:P]
    ok = f["valid"][i,:P] & (fc == 1)           # singly covered -> candidate is determined
    if ok.sum() < 50: continue
    # which candidate covers each singly-covered position
    owner = np.full(P, -1)
    for k in range(nk):
        a_lo,a_hi,s_lo,s_hi = b[k,2],b[k,3],b[k,4],b[k,5]
        if a_hi>=a_lo: owner[max(a_lo,1)-1:a_hi] = k
        if s_hi>=s_lo: owner[max(s_lo,1)-1:s_hi] = k
    P0 = 1/(1+np.exp(-bl[i]))
    for pos in np.flatnonzero(ok):
        k = owner[pos]
        if k < 0: continue
        sel = float(ps[lo+k]); d0 = float(pd_[lo+k])
        if sel <= 0 or not (0 < d0 < 1): continue
        for bidx in range(4):
            v = vd[pos, bidx]
            if not np.isfinite(v): continue
            P1 = 1/(1+np.exp(-(bl[i] + v)))
            arg = d0 + (P1 - P0)/sel
            if not (0 < arg < 1): continue
            keep_v.append(abs(v)); keep_d.append(abs(lg(arg) - lg(d0))); keep_m.append(sel)
v,d,m = map(np.array,(keep_v,keep_d,keep_m))
lm = np.log10(np.maximum(m,1e-30))
print(f"  {len(v):,} singly-covered substitutions recovered")
print(f"\n  correlation with log10(selection mass):")
print(f"    |vals_decay|   (the aggregated quantity)   r = {np.corrcoef(np.log10(np.maximum(v,1e-30)), lm)[0,1]:+.4f}")
print(f"    |delta z_d|    (the decay head's own)      r = {np.corrcoef(np.log10(np.maximum(d,1e-30)), lm)[0,1]:+.4f}")
print(f"\n  If the second is near zero, the mass factor is removed exactly rather")
print(f"  than adjusted for -- and no stratification is needed at all.")
