# Why does my directionality read 0.373 where the other window reads 0.391?
# Compute the variants that plausibly differ, on ONE sample, so the gap is
# attributed rather than guessed at.
import h5py, numpy as np
import sys
COL = sys.argv[1] if len(sys.argv) > 1 else "vals_decay"
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
floor = float(f.attrs["batch_shape_offset"])
rng = np.random.default_rng(0)
take = rng.choice(f["vals_decay"].shape[0], 500, replace=False)

pool4, pool3, per_tx4, above_floor, nonzero_only = [], [], [], [], []
for i in take:
    v = f["valid"][i]
    x = f[COL][i].astype(np.float64)
    fin = (np.isfinite(x).sum(1) == 3) & v
    if fin.sum() < 200: continue
    sub = np.where(np.isfinite(x[fin]), x[fin], 0.0)      # (m,4) with obs = 0
    mx = np.abs(sub).max(1)
    ok = mx > 0
    if ok.sum() < 50: continue
    s, m = sub[ok], mx[ok]
    d4 = np.abs(s.mean(1)) / m                            # mean over 4 (obs=0)
    d3 = np.abs(s.sum(1) / 3.0) / m                       # mean over the 3 substitutions
    pool4.append(d4); pool3.append(d3)
    per_tx4.append(d4.mean())
    af = m > floor
    if af.sum(): above_floor.append(d4[af])
    # the observed base is 0 by construction, so .all(1) over four is never true --
    # third time this exact shape has bitten today. Check the three substitutions.
    nz = (np.abs(s) > 0).sum(1) == 3                       # no dead substitution
    if nz.sum(): nonzero_only.append(d4[nz])

P4 = np.concatenate(pool4); P3 = np.concatenate(pool3)
print(f"  COLUMN = {COL}")
print(f"  {'variant':<44} {'value':>8} {'n':>12}")
print(f"  {'pooled over positions, mean over 4 (MINE)':<44} {P4.mean():>8.4f} {len(P4):>12,}")
print(f"  {'pooled over positions, mean over 3':<44} {P3.mean():>8.4f} {len(P3):>12,}")
print(f"  {'per-transcript mean, then mean over tx':<44} {np.mean(per_tx4):>8.4f} {len(per_tx4):>12,}")
A = np.concatenate(above_floor); print(f"  {'above the batch-shape floor only':<44} {A.mean():>8.4f} {len(A):>12,}")
Z = np.concatenate(nonzero_only); print(f"  {'excluding any dead substitution':<44} {Z.mean():>8.4f} {len(Z):>12,}")
print()
print(f"  ratio, mean-over-3 to mean-over-4: {P3.mean()/P4.mean():.4f}  (4/3 = 1.3333)")
print(f"  the other window reports 0.391 for 'all positions'")
for nm, val in (("pooled/4", P4.mean()), ("pooled/3", P3.mean()),
                ("per-transcript", np.mean(per_tx4)), ("above floor", A.mean()),
                ("no dead sub", Z.mean())):
    print(f"    {nm:<16} {val:.4f}   gap to 0.391: {0.391-val:+.4f}")
