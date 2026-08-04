import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
R, U = [], []
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 50: continue
    v = f["vals_decay"][i,:P].astype(np.float64)
    ok = f["valid"][i,:P] & np.isfinite(v).any(1)
    if ok.sum() < 100: continue
    vv = np.where(np.isfinite(v), v, 0.0)
    with np.errstate(invalid="ignore"):
        uns = np.nanmax(np.abs(np.where(np.isfinite(v), v, np.nan)), axis=1)
    g = ok & np.isfinite(uns) & (uns > 0)
    R.append(np.abs(vv.mean(1))[g] / uns[g]); U.append(uns[g])
r = np.concatenate(R); u = np.concatenate(U)
tx = np.concatenate([np.full(len(x), i) for i, x in enumerate(R)])
FLOOR = 1.66e-06
sub = u < FLOOR
print(f"  {len(r):,} positions, {len(R)} transcripts")
print(f"  sub-floor null      median {np.median(r[sub]):.4f}   mean {r[sub].mean():.4f}")
# GLOBAL top 1%
gk = int(round(0.01*len(u))); gsel = u >= np.partition(u, -gk)[-gk]
# PER-TRANSCRIPT top 1%
psel = np.zeros(len(u), bool)
for i in range(len(R)):
    m = np.flatnonzero(tx == i); k = max(1, int(round(0.01*len(m))))
    psel[m[np.argsort(u[m])[-k:]]] = True
for nm, s in (("GLOBAL top 1%", gsel), ("PER-TRANSCRIPT top 1%", psel)):
    med, mn = np.median(r[s]), r[s].mean()
    print(f"  {nm:<24} n {int(s.sum()):>7,}  median {med:.4f} ({100*(med/np.median(r[sub])-1):+.1f}%)"
          f"   mean {mn:.4f} ({100*(mn/r[sub].mean()-1):+.1f}%)")
