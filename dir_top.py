# Their top-1% is 0.4664, mine 0.4481, both medians on the same bank. Candidate:
# GLOBAL top 1% (quantile over pooled positions) vs PER-TRANSCRIPT top 1%. They
# select different sets -- global favours transcripts whose effects are large
# overall, per-transcript takes the same share from every transcript.
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
floor = float(f.attrs["batch_shape_offset"])
rng = np.random.default_rng(0)
take = rng.choice(f["vals_decay"].shape[0], 500, replace=False)
R, M, PT = [], [], []
for i in take:
    v=f["valid"][i]; x=f["vals_decay"][i].astype(np.float64)
    fin=(np.isfinite(x).sum(1)==3)&v
    if fin.sum()<200: continue
    s=np.where(np.isfinite(x[fin]),x[fin],0.0); mm=np.abs(s).max(1); ok=mm>0
    r=np.abs(s[ok].mean(1))/mm[ok]; m=mm[ok]
    k=max(1,int(round(0.01*len(m))))
    pt=np.zeros(len(m),bool); pt[np.argsort(-m)[:k]]=True    # per-transcript top 1%
    R.append(r); M.append(m); PT.append(pt)
R=np.concatenate(R); M=np.concatenate(M); PT=np.concatenate(PT)
glob = M >= np.quantile(M,0.99)
sub  = M < floor
print(f"  {'set':<34} {'median':>8} {'mean':>8} {'n':>10}")
for nm,msk in (("sub-floor null (measured)",sub),
               ("top 1% GLOBAL quantile (mine)",glob),
               ("top 1% PER-TRANSCRIPT (theirs?)",PT)):
    print(f"  {nm:<34} {np.median(R[msk]):>8.4f} {R[msk].mean():>8.4f} {int(msk.sum()):>10,}")
nullm, nullmean = np.median(R[sub]), R[sub].mean()
print(f"\n  against the measured sub-floor null ({nullm:.4f} median):")
for nm,msk in (("global top 1%",glob),("per-transcript top 1%",PT)):
    print(f"    {nm:<26} {100*(np.median(R[msk])/nullm-1):>6.1f}% above null   "
          f"{100*(np.median(R[msk])-nullm)/(0.75-nullm):>5.1f}% of range {nullm:.3f}-0.75")
print(f"\n  overlap of the two definitions: Jaccard "
      f"{np.logical_and(glob,PT).sum()/np.logical_or(glob,PT).sum():.3f}")
