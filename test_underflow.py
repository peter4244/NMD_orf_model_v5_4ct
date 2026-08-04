# Harold: an exact zero at p_select < 1e-8 cannot be the mass annihilating a real
# change, because float64 resolves it by ~10^5 ulps -- so it requires dz_p bitwise
# zero, which relocates the question to enc_init.
#
# TRUE ABOVE A BOUNDARY THEY DID NOT STATE. Unresolvable needs
# mass * dz_p < ~2.2e-16. With dz_p ~1e-3 that is mass < 2.2e-13; with dz_p ~1e-5,
# mass < 2.2e-11. The stratum "p_select < 1e-8" spans many decades below that, and
# stick-breaking mass ~0.5^k reaches 1e-15 by slot 50. So: do the deaths
# concentrate below the underflow boundary (aggregation) or above it (encoder)?
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
spans=f["spans"][:]; off=f["cand_offset"][:]; cnt=f["cand_count"][:]; p_sel=f["p_select"][:]
n=f["vals"].shape[0]; rng=np.random.default_rng(1); take=rng.choice(n,400,replace=False)
D,M=[],[]
for i in take:
    v=f["valid"][i]; cap=f["vals_capture"][i]
    sl=slice(int(off[i]), int(off[i])+int(cnt[i])); sp=spans[sl]; ps=p_sel[sl]
    W=len(v); cov=np.zeros(W,np.int16); who=np.full(W,-1,np.int32)
    for k,row in enumerate(sp):
        a_lo,a_hi=int(row[2]),int(row[3])
        if a_hi>=a_lo: cov[max(0,a_lo-1):a_hi]+=1; who[max(0,a_lo-1):a_hi]=k
    fin=np.isfinite(cap).any(1)&v
    with np.errstate(all="ignore"):
        mx=np.nanmax(np.abs(np.where(np.isfinite(cap),cap,np.nan)),1)
    dead=fin&(mx==0.0); sel=fin&(cov==1)
    if not sel.any(): continue
    D.append(dead[sel]); M.append(ps[who[sel]])
D=np.concatenate(D); M=np.concatenate(M)
low = M < 1e-8
print(f"  dead-mass stratum (p_select < 1e-8): n {int(low.sum()):,}, "
      f"dead {100*D[low].mean():.2f}%\n")
print(f"  {'p_select band':<26} {'n':>9} {'dead rate':>10}   regime")
edges=[0,1e-30,1e-20,1e-16,1e-13,1e-11,1e-8]
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(M>=lo)&(M<hi)
    if m.sum()<20: continue
    reg = "underflow possible" if hi<=2.2e-13 else "encoder must be bitwise-equal"
    print(f"  [{lo:g}, {hi:g}){'':<6} {int(m.sum()):>9,} {100*D[m].mean():>9.2f}%   {reg}")
print(f"\n  exactly zero p_select: n {int((M==0).sum()):,}, dead "
      f"{100*D[M==0].mean() if (M==0).any() else float('nan'):.2f}%")
