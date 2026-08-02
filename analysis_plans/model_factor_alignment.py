"""
model_factor_alignment.py — how aligned are the two factors of the product?

SCOPE -- A CLAIM ABOUT THE MODEL, NOT ABOUT BIOLOGY.

Pete's question: a direct correlation between p_select and d.

WHY IT IS NOT MERELY A DIAGNOSTIC. P(NMD) = sum_k p_select_k * d_k. If the two
factors were independent across candidates, that sum would be about
(sum_k p_select_k) * mean(d). ANY CORRELATION BETWEEN THEM IS THE DIFFERENCE
BETWEEN THE MODEL'S ACTUAL OUTPUT AND THAT INDEPENDENT BASELINE. So the
correlation is not describing the model, it is part of what the model computes.

  positive  the model routes TOWARD decay-prone candidates and the product is
            amplified above the independent baseline -- Pete's hypothesis, at the
            level of the product rather than in the head
  negative  it routes AWAY from them and the product is suppressed
  zero      routing and decay are orthogonal and the mixture is uninformative
            about their interaction

RAW IS THE PRIMARY, deliberately. Both factors relate to ORF length, but length is
not a nuisance here: the product is what the model computes, confounds included.
Within-mass-band figures are reported as DESCRIPTION with no verdict attached,
because stratifying answers a different and narrower question.

Also reported: p_capture against d, which is the same question with the
stick-breaking queue removed, so the difference between the two is the queue's
contribution to alignment.

Run from the repo root.
"""
import argparse
import numpy as np
import h5py

DEAD_CUT = 1e-8


def sp(a, b):
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
a = ap.parse_args()

with h5py.File(a.bank, "r") as f:
    off, cnt = f["cand_offset"][:], f["cand_count"][:]
    ps_, pc_, pd_ = f["p_select"][:], f["p_capture"][:], f["p_decay"][:]
    lab = f["labels"][:]
    N = len(cnt)
    ck = f.attrs.get("checkpoint", "?")

r_sel, r_cap, amp = [], [], []
by_lab = {0: [], 1: []}
for i in range(N):
    lo, k = int(off[i]), int(cnt[i])
    if k < 3:
        continue
    p, c, d = ps_[lo:lo + k], pc_[lo:lo + k], pd_[lo:lo + k]
    rs = sp(p, d)
    r_sel.append(rs)
    r_cap.append(sp(c, d))
    # alignment gain: actual mixture against the independent-factor baseline
    act = float((p * d).sum())
    ind = float(p.sum() * d.mean())
    if ind > 1e-12:
        amp.append(act / ind)
    if int(lab[i]) in by_lab and np.isfinite(rs):
        by_lab[int(lab[i])].append(rs)


def rep(name, x):
    x = np.array(x, float); x = x[np.isfinite(x)]
    print(f"  {name:<44} median {np.median(x):+.3f}   mean {x.mean():+.3f}"
          f"   n {len(x):,}")
    return x


print(f"BANK {a.bank}\ncheckpoint {ck}\n")
print("=" * 74)
print("FACTOR ALIGNMENT -- within transcript, across candidates")
print("=" * 74)
rs = rep("p_select  ~  d        [THE ANSWER]", r_sel)
rep("p_capture ~  d        (queue removed)", r_cap)
print(f"  positive share {float((rs > 0).mean()):.3f}"
      f"   negative share {float((rs < 0).mean()):.3f}")

print("\n" + "=" * 74)
print("WHAT THE ALIGNMENT DOES TO THE OUTPUT")
print("=" * 74)
am = np.array(amp); am = am[np.isfinite(am)]
print(f"  actual mixture / independent-factor baseline")
print(f"    median {np.median(am):.3f}   mean {am.mean():.3f}   n {len(am):,}")
print(f"    deciles " + " ".join(f"{np.percentile(am, q):.3f}"
                                 for q in (10, 25, 50, 75, 90)))
print("  1.0 = the factors are independent and the mixture is the product of")
print("  the marginals. Above 1 = routing and decay ALIGN and the model's")
print("  prediction is amplified by that alignment.")

print("\n" + "=" * 74)
print("BY LABEL")
print("=" * 74)
for lv, nm in ((1, "NMD"), (0, "control")):
    x = np.array(by_lab[lv], float); x = x[np.isfinite(x)]
    if len(x):
        print(f"  p_select ~ d, {nm:<8} median {np.median(x):+.3f}   n {len(x):,}")
