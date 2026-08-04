import h5py, numpy as np
want = ['ENSG00000000457.15.novel6','ENSG00000000457.15.novel8','ENSG00000001497.19.novel1']
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
tid = np.array([t.decode() if isinstance(t,bytes) else t for t in f["transcript_id"][:]])
spans=f["spans"][:]; off=f["cand_offset"][:]; cnt=f["cand_count"][:]
print("GPU bank, same transcripts, seed 100:")
for t in want:
    ix = np.flatnonzero(tid==t)
    if not len(ix):
        print(f"  {t}: NOT IN BANK"); continue
    i=int(ix[0])
    v=f["valid"][i]; cap=f["vals_capture"][i]
    sl=slice(int(off[i]), int(off[i])+int(cnt[i]))
    W=len(v); atg=np.zeros(W,bool)
    for row in spans[sl]:
        a_lo,a_hi=int(row[2]),int(row[3])
        if a_hi>=a_lo: atg[max(0,a_lo-1):a_hi]=True
    fin=np.isfinite(cap).any(1)&v
    with np.errstate(all="ignore"):
        mx=np.nanmax(np.abs(np.where(np.isfinite(cap),cap,np.nan)),1)
    z=fin&(mx==0.0)
    print(f"  {t}: valid {int(fin.sum()):>6,}  ATG-cov {int((fin&atg).sum()):>6,}  "
          f"exact-zero {int(z.sum()):>5,}  inside-ATG zeros {int((z&atg).sum()):>5,}  "
          f"base_logit {float(f['base_logit'][i]):.10f}")
