# Is vals_capture "mostly floor", or mostly STRUCTURALLY ZERO?
# A substitution in a STOP window changes e_stop -> z_d only. It cannot touch z_p,
# so vals_capture is EXACTLY 0 there by construction, not by being small. If most
# sub-floor capture entries are stop-only positions, then "42% clears the floor"
# is measured against the wrong denominator and says nothing about readability.
import h5py, numpy as np
p = "results_ism_v6/bank_interp_s100.h5"
with h5py.File(p, "r") as f:
    spans = f["spans"][:]; off = f["cand_offset"][:]; cnt = f["cand_count"][:]
    n = f["vals"].shape[0]
    rng = np.random.default_rng(0)
    take = rng.choice(n, 300, replace=False)
    tot_valid = atg_cov = cap_zero = cap_zero_in_atg = 0
    for i in take:
        v = f["valid"][i]; cap = f["vals_capture"][i]
        sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
        s = spans[sl]                      # a_lo, a_hi, s_lo, s_hi
        W = len(v)
        atg = np.zeros(W, bool)
        for _r, _sl, a_lo, a_hi, _s1, _s2 in s:
            if a_hi >= a_lo:
                atg[max(0, a_lo - 1):a_hi] = True
        fin = np.isfinite(cap).any(1) & v
        z = fin & (np.nanmax(np.abs(cap), 1) == 0.0)
        tot_valid += int(fin.sum()); atg_cov += int((fin & atg).sum())
        cap_zero += int(z.sum());   cap_zero_in_atg += int((z & atg).sum())
print(f"  sampled 300 transcripts, {tot_valid:,} valid positions")
print(f"  covered by >=1 ATG window            : {100*atg_cov/tot_valid:5.1f}%")
print(f"  vals_capture EXACTLY zero            : {100*cap_zero/tot_valid:5.1f}%")
print(f"  ...of those, inside an ATG window    : {100*cap_zero_in_atg/max(cap_zero,1):5.1f}%")
print(f"  exact zeros OUTSIDE any ATG window   : {100*(cap_zero-cap_zero_in_atg)/tot_valid:5.1f}% of valid")
