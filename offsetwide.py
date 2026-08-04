import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
NT="ACGT"; W=60
hi = np.zeros((2*W+1,4)); bg = np.zeros(4)
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 2*W+50: continue
    v = f["vals_decay"][i,:P].astype(np.float64); o = f["obs"][i,:P]
    with np.errstate(invalid="ignore"):
        e = np.nanmax(np.abs(v), axis=1)
    ok = f["valid"][i,:P] & np.isfinite(e) & (o>=0)
    idx = np.flatnonzero(ok); idx = idx[(idx>=W)&(idx<P-W)]
    if len(idx) < 200: continue
    k = max(1,int(round(0.01*len(idx)))); cut = np.partition(e[idx],-k)[-k]
    sel = idx[e[idx] >= cut]
    for base in range(4): bg[base] += (o[idx]==base).sum()
    for d in range(-W, W+1):
        w = o[sel+d]
        for base in range(4): hi[d+W, base] += (w==base).sum()
q = bg/bg.sum()
print(f"  background  " + "  ".join(f"{NT[j]} {q[j]:.3f}" for j in range(4)))
print(f"\n  U enrichment and C depletion by distance from the elevated position")
print(f"    {'offset':>7} {'U ratio':>9} {'C ratio':>9} {'A+T':>8} {'G+C':>8}")
for d in list(range(-60,-19,10)) + list(range(-8,9,2)) + list(range(20,61,10)):
    p = hi[d+W]/hi[d+W].sum()
    at=(p[0]+p[3])/(q[0]+q[3]); gc=(p[1]+p[2])/(q[1]+q[2])
    print(f"    {d:>+7} {p[3]/q[3]:>9.3f} {p[1]/q[1]:>9.3f} {at:>8.3f} {gc:>8.3f}")
u = hi[:, 3]/hi.sum(1)/q[3]
far = np.r_[u[:20], u[-20:]].mean()
print(f"\n  U ratio at |offset|>40 : {far:.3f}   at offset 0 : {u[W]:.3f}   at |offset|<=4 : {u[W-4:W+5].mean():.3f}")
print(f"  half-decay: the GC channel averages over +/-25, so a decay length near 25")
print(f"  is the encoding; a decay length of a few bases is local sequence context.")
