import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
ov, n_tx = [], 0
dir_ratio_hi, dir_ratio_all = [], []
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 50: continue
    v = f["vals_decay"][i,:P].astype(np.float64)
    ok = f["valid"][i,:P] & np.isfinite(v).any(1)
    if ok.sum() < 100: continue
    vv = np.where(np.isfinite(v), v, 0.0)          # vals[obs] := 0
    uns = np.nanmax(np.abs(np.where(np.isfinite(v), v, np.nan)), axis=1)   # ours
    sig = np.abs(vv.mean(1))                                              # |hyp[obs]|
    good = ok & np.isfinite(uns) & (uns > 0)
    k = max(1, int(round(0.01*good.sum())))
    a = set(np.flatnonzero(good)[np.argsort(uns[good])[-k:]].tolist())
    c = set(np.flatnonzero(good)[np.argsort(sig[good])[-k:]].tolist())
    ov.append(len(a & c)/len(a | c)); n_tx += 1
    r = sig[good]/np.maximum(uns[good], 1e-30)
    dir_ratio_all.append(np.median(r))
    idx = np.flatnonzero(good)[np.argsort(uns[good])[-k:]]
    dir_ratio_hi.append(np.median(sig[idx]/np.maximum(uns[idx],1e-30)))
print(f"  {n_tx} transcripts, top-1% each way")
print(f"    Jaccard(unsigned top1%, signed top1%)   median {np.median(ov):.4f}   "
      f"quartiles {np.percentile(ov,25):.4f}-{np.percentile(ov,75):.4f}")
print()
print(f"  DIRECTIONALITY  |mean_b vals| / max_b|vals|   (1.0 = all three substitutions agree in sign")
print(f"                                                 0.25 = perfectly opposing, cancels)")
print(f"    all positions        median {np.median(dir_ratio_all):.3f}")
print(f"    unsigned-elevated    median {np.median(dir_ratio_hi):.3f}")
