# THE COMPOSITION NULL AT THREE LOCALITIES.
# The null says: how much information can a CWM column carry from base composition
# alone? It is KL(elevated composition || background composition). The question is
# WHICH background -- and the answer changes the bar by a lot.
#   GLOBAL       one composition for the whole dataset
#   REGIONAL     within upstream / downstream of the operative stop
#   PER-TRANSCRIPT  each transcript against its own composition
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
n = f["vals_decay"].shape[0]
c_off,c_cnt,c_end,p_sel = f["cand_offset"][:],f["cand_count"][:],f["cand_orf_end"][:],f["p_select"][:]
rng = np.random.default_rng(0); take = rng.choice(n, 600, replace=False)
def H(p): p=np.asarray(p,float); p=p/p.sum(); return -(p*np.log2(p)).sum()
def KL(p,q):
    p=np.asarray(p,float); q=np.asarray(q,float); p=p/p.sum(); q=q/q.sum()
    return float((p*np.log2(p/q)).sum())
G={"fg":np.zeros(4),"bg":np.zeros(4)}
R={r:{"fg":np.zeros(4),"bg":np.zeros(4)} for r in ("upstream","downstream")}
per_tx=[]
for i in take:
    v=f["valid"][i]; obs=f["obs"][i]; ok=v&(obs>=0)
    if ok.sum()<400: continue
    x=np.abs(f["vals_decay"][i])
    with np.errstate(all="ignore"):
        eff=np.nanmax(np.where(np.isfinite(x),x,np.nan),1)
    good=ok&np.isfinite(eff)
    if good.sum()<400: continue
    sl=slice(int(c_off[i]),int(c_off[i])+int(c_cnt[i]))
    stop=int(c_end[sl][int(np.argmax(p_sel[sl]))])
    pos=np.arange(1,len(v)+1)
    for nm,rm in (("upstream",pos<=stop),("downstream",pos>stop)):
        sel=good&rm; m=int(sel.sum())
        if m<200: continue
        idx=np.flatnonzero(sel); k=max(1,int(round(0.01*m)))
        hi=idx[np.argsort(-eff[idx])[:k]]
        R[nm]["fg"]+=np.bincount(obs[hi],minlength=4); R[nm]["bg"]+=np.bincount(obs[idx],minlength=4)
    idx=np.flatnonzero(good); k=max(1,int(round(0.01*len(idx))))
    hi=idx[np.argsort(-eff[idx])[:k]]
    G["fg"]+=np.bincount(obs[hi],minlength=4); G["bg"]+=np.bincount(obs[idx],minlength=4)
    fgt=np.bincount(obs[hi],minlength=4); bgt=np.bincount(obs[idx],minlength=4)
    if fgt.sum()>=20 and (fgt>0).all(): per_tx.append(KL(fgt,bgt))
print(f"  {'null':<34} {'A':>6} {'C':>6} {'G':>6} {'T':>6}   {'KL bits':>9}")
print(f"  {'GLOBAL  background':<34} " + " ".join(f"{x:6.3f}" for x in G['bg']/G['bg'].sum()))
print(f"  {'GLOBAL  elevated':<34} " + " ".join(f"{x:6.3f}" for x in G['fg']/G['fg'].sum())
      + f"   {KL(G['fg'],G['bg']):9.4f}")
for nm in ("upstream","downstream"):
    d=R[nm]
    print(f"  {'REGIONAL '+nm+' background':<34} " + " ".join(f"{x:6.3f}" for x in d['bg']/d['bg'].sum()))
    print(f"  {'REGIONAL '+nm+' elevated':<34} " + " ".join(f"{x:6.3f}" for x in d['fg']/d['fg'].sum())
          + f"   {KL(d['fg'],d['bg']):9.4f}")
pt=np.array(per_tx)
print(f"\n  PER-TRANSCRIPT KL, n={len(pt):,}   mean {pt.mean():.4f}   median {np.median(pt):.4f}"
      f"   p90 {np.quantile(pt,0.9):.4f}")
print(f"\n  regional background differs from global:")
for nm in ("upstream","downstream"):
    print(f"    {nm:<12} KL(region bg || global bg) = {KL(R[nm]['bg'],G['bg']):.4f} bits")
