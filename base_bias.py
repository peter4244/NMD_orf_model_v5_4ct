# Does restricting to GC-preserving substitutions SELECT FOR C/G positions?
# If it does, the "GC control" swaps one compositional confound for its mirror
# image, and the k-mer shift it produces is an artifact of the control itself.
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
rng = np.random.default_rng(0)
take = rng.choice(f["vals_decay"].shape[0], 600, replace=False)
cnt = {s: np.zeros(4) for s in ("all", "neutral")}
bgc = np.zeros(4)
for i in take:
    v = f["valid"][i]; obs = f["obs"][i]
    ok = v & (obs >= 0)
    if ok.sum() < 200: continue
    x = np.abs(f["vals_decay"][i]); dgc = f["dgc"][i]
    bgc += np.bincount(obs[ok], minlength=4)
    for s in ("all", "neutral"):
        xx = x if s == "all" else np.where(dgc == 0, x, np.nan)
        with np.errstate(all="ignore"):
            eff = np.nanmax(np.where(np.isfinite(xx), xx, np.nan), axis=1)
        good = ok & np.isfinite(eff)
        m = int(good.sum())
        if m < 200: continue
        idx = np.flatnonzero(good)
        k = max(1, int(round(0.01 * m)))
        cut = np.partition(eff[idx], -k)[-k]
        hi = idx[eff[idx] >= cut]
        cnt[s] += np.bincount(obs[hi], minlength=4)
bgc /= bgc.sum()
print(f"  {'set':<26} {'A':>7} {'C':>7} {'G':>7} {'T':>7}   {'G+C':>7}")
print(f"  {'background (all valid)':<26} " + " ".join(f"{x:>7.3f}" for x in bgc)
      + f"   {bgc[1]+bgc[2]:>7.3f}")
for s in ("all", "neutral"):
    p = cnt[s]/cnt[s].sum()
    print(f"  {'elevated, '+s+' scoring':<26} " + " ".join(f"{x:>7.3f}" for x in p)
          + f"   {p[1]+p[2]:>7.3f}")
    print(f"  {'  ratio to background':<26} " + " ".join(f"{x:>7.2f}" for x in p/bgc))
