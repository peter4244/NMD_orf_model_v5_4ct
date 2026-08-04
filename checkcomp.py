import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
NT = "ACGT"
bg = np.zeros(4); hi = np.zeros(4); hn = np.zeros(4)
for i in range(800):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 50: continue
    v = f["vals_decay"][i,:P]; o = f["obs"][i,:P]; dg = f["dgc"][i,:P]
    with np.errstate(invalid="ignore"):
        e = np.nanmax(np.abs(v), axis=1)
    ok = f["valid"][i,:P] & np.isfinite(e) & (o >= 0)
    if ok.sum() < 50: continue
    k = max(1, int(round(0.01*ok.sum()))); cut = np.partition(e[ok], -k)[-k]
    en = np.full(P, np.nan)
    m = (dg == 0) & np.isfinite(v); r_, c_ = np.nonzero(m); en[r_] = np.abs(v[r_, c_])
    okn = ok & np.isfinite(en)
    kn = max(1, int(round(0.01*okn.sum()))); cutn = np.partition(en[okn], -kn)[-kn]
    for base in range(4):
        bg[base] += ((o == base) & ok).sum()
        hi[base] += ((o == base) & ok & (e >= cut)).sum()
        hn[base] += ((o == base) & okn & (en >= cutn)).sum()
q = bg/bg.sum()
for nm, a in (("background", bg), ("elevated, all scoring", hi), ("elevated, GC-preserving", hn)):
    p = a/a.sum()
    print("  %-26s" % nm + "  ".join("%s %.3f" % (NT[j], p[j]) for j in range(4)) + "   G+C %.3f" % (p[1]+p[2]))
    if nm != "background":
        print("  %-26s" % "  ratio to background" + "  ".join("%s %.2f" % (NT[j], p[j]/q[j]) for j in range(4)))
p = hi/hi.sum()
print()
print("  keto (G+T)  elevated %.3f  background %.3f  ratio %.3f" % (p[2]+p[3], q[2]+q[3], (p[2]+p[3])/(q[2]+q[3])))
print("  amino (A+C) elevated %.3f  background %.3f  ratio %.3f" % (p[0]+p[1], q[0]+q[1], (p[0]+p[1])/(q[0]+q[1])))
