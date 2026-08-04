import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
NT="ACGT"; W=4
hi = np.zeros((2*W+1,4)); bg = np.zeros(4)
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 50: continue
    v = f["vals_decay"][i,:P].astype(np.float64); o = f["obs"][i,:P]
    with np.errstate(invalid="ignore"):
        e = np.nanmax(np.abs(v), axis=1)
    ok = f["valid"][i,:P] & np.isfinite(e) & (o>=0)
    idx = np.flatnonzero(ok); idx = idx[(idx>=W)&(idx<P-W)]
    if len(idx) < 100: continue
    k = max(1,int(round(0.01*len(idx)))); cut = np.partition(e[idx],-k)[-k]
    sel = idx[e[idx] >= cut]
    for base in range(4): bg[base] += (o[idx]==base).sum()
    for d in range(-W, W+1):
        w = o[sel+d]
        for base in range(4): hi[d+W, base] += ((w==base)).sum()
q = bg/bg.sum()
print(f"  background  " + "  ".join(f"{NT[j]} {q[j]:.3f}" for j in range(4)))
print()
print(f"  base composition BY OFFSET from the elevated position (ratio to background)")
print(f"    {'offset':>7} " + "".join(f"{NT[j]:>8}" for j in range(4)) + f"{'keto G+T':>10}")
for d in range(-W, W+1):
    p = hi[d+W]/hi[d+W].sum()
    print(f"    {d:>+7} " + "".join(f"{p[j]/q[j]:>8.3f}" for j in range(4))
          + f"{(p[2]+p[3])/(q[2]+q[3]):>10.3f}")
print()
print("  If the skew is confined to offset 0, the k-mer table is a CENTRE-base effect")
print("  smeared by the window. If it extends, neighbouring positions share it.")
