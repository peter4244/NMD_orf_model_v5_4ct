# IS DIRECTIONALITY JUST MAGNITUDE?
# |mean_b vals| / max_b |vals| is higher at elevated positions (0.466) than at all
# positions (0.391). But elevated positions are DEFINED as the largest effects, and
# a small effect is closer to the noise floor, where the three substitutions have
# near-random signs and the signed mean cancels. So higher directionality among
# elevated positions may be entirely a signal-to-noise effect and not evidence of
# learned directional structure.
#
# TEST: does directionality rise monotonically with |effect| across ALL positions?
# If it does, the elevated-vs-background gap is explained by magnitude alone and
# says nothing extra. Ceiling is 0.75: with vals[obs]=0 and three substitutions
# agreeing, mean = 3v/4 against max = v.
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
rng = np.random.default_rng(0)
take = rng.choice(f["vals_decay"].shape[0], 500, replace=False)
D, M, floor = [], [], float(f.attrs["batch_shape_offset"])
for i in take:
    v = f["valid"][i]
    x = f["vals_decay"][i].astype(np.float64)
    # vals is NaN AT THE OBSERVED BASE by construction -- you cannot substitute a
    # base for itself -- so .all(1) is never true. Require the three substitutions
    # finite and set the observed base to 0, which is the mean-centring convention.
    fin = (np.isfinite(x).sum(1) == 3) & v
    if fin.sum() < 200: continue
    xx = np.where(np.isfinite(x[fin]), x[fin], 0.0)   # vals[obs] := 0
    mx = np.abs(xx).max(1)
    ok = mx > 0
    D.append(np.abs(xx[ok].mean(1)) / mx[ok]); M.append(mx[ok])
D = np.concatenate(D); M = np.concatenate(M)
print(f"  {len(D):,} positions   ceiling 0.75   floor {floor:.3e}\n")
q = np.quantile(M, [0, .2, .4, .6, .8, .95, .99, 1.0])
print(f"  {'|effect| decile band':<26} {'n':>10} {'median |eff|':>13} {'directionality':>15} {'% of ceiling':>13}")
for lo, hi in zip(q[:-1], q[1:]):
    m = (M >= lo) & (M < hi) if hi < q[-1] else (M >= lo)
    if m.sum() < 100: continue
    print(f"  {f'{lo:.2e} - {hi:.2e}':<26} {int(m.sum()):>10,} "
          f"{np.median(M[m]):>13.3e} {np.mean(D[m]):>15.3f} {100*np.mean(D[m])/0.75:>12.1f}%")
top = M >= np.quantile(M, 0.99)
print(f"\n  all positions          directionality {D.mean():.3f}  ({100*D.mean()/0.75:.0f}% of ceiling)")
print(f"  top 1% by |effect|     directionality {D[top].mean():.3f}  ({100*D[top].mean()/0.75:.0f}% of ceiling)")
print(f"  above the floor only   directionality {D[M>floor].mean():.3f}  "
      f"({100*D[M>floor].mean()/0.75:.0f}% of ceiling)   n {int((M>floor).sum()):,}")
