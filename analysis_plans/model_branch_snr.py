"""
model_branch_snr.py — does vals_capture clear its own floor as well as vals_decay
clears its own?

SCOPE -- A CLAIM ABOUT THE MODEL / OUR INSTRUMENT, NOT ABOUT BIOLOGY.

WHY. I argued the capture branch's exclusion from the programme was untested,
citing the two in-sample noise floors: capture 2.25e-11, decay 1.35e-10, a factor
of six rather than the "roughly a thousandfold" the exclusion rested on.

THE INTERPRETABILITY WINDOW POINTED OUT THAT THIS IS THE WRONG COMPARISON AND
MIGHT VINDICATE THE EXCLUSION RATHER THAN RETIRE IT. The rationale was about the
VALUES being ~1000x smaller. If the values are 1000x smaller and the floor only
6x smaller, capture's signal-to-noise is ~160x WORSE, and comparable floors with
incomparable magnitudes is the worst case rather than the reassuring one.

The deciding statistic is therefore NOT the floors side by side. It is the ratio
of typical |vals| to ITS OWN floor, per branch. That is what says whether a
branch's signal is readable at all.

Reported over LIVE positions only (mass >= 1e-8), because a dead position cannot
respond in either branch and would drag both ratios toward 1 equally.

DECISION, registered before the run:
  capture ratio within ~2x of decay's   -> the exclusion is dead, the branch is
                                           as readable as the one we have been
                                           using all week
  capture ratio 2-10x worse             -> usable with care, at reduced
                                           resolution, and every capture claim
                                           carries that as a bound
  capture ratio >10x worse              -> the exclusion was RIGHT and its real
                                           reason was never written down. I
                                           withdraw the re-plan.

Run from the repo root.
"""
import argparse
import numpy as np
import h5py

DEAD_CUT = 1e-8
ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
a = ap.parse_args()

f = h5py.File(a.bank, "r")
spans, off, cnt = f["spans"][:], f["cand_offset"][:], f["cand_count"][:]
N = len(cnt)

cap_dead, dec_dead, cap_live, dec_live = [], [], [], []
for i in range(0, N, 5):
    lo, k = int(off[i]), int(cnt[i])
    b = spans[lo:lo + k]
    P = int(max(b[:, 3].max(), b[:, 5].max()))
    if P < 50:
        continue
    m = f["mass"][i, :P].astype(np.float64)
    ok = f["valid"][i, :P].astype(bool)
    with np.errstate(invalid="ignore"):
        ec = np.nanmax(np.abs(f["vals_capture"][i, :P]), 1)
        ed = np.nanmax(np.abs(f["vals_decay"][i, :P]), 1)
    d_ = ok & (m < DEAD_CUT) & np.isfinite(ec) & np.isfinite(ed)
    l_ = ok & (m >= DEAD_CUT) & np.isfinite(ec) & np.isfinite(ed)
    if d_.any():
        cap_dead.append(ec[d_]); dec_dead.append(ed[d_])
    if l_.any():
        cap_live.append(ec[l_]); dec_live.append(ed[l_])
f.close()

cd, dd = np.concatenate(cap_dead), np.concatenate(dec_dead)
cl, dl = np.concatenate(cap_live), np.concatenate(dec_live)
fc, fd = np.percentile(cd, 95), np.percentile(dd, 95)

print(f"BANK {a.bank}")
print(f"live positions sampled {len(cl):,}   dead {len(cd):,}\n")
print("=" * 72)
print("MAGNITUDES -- is capture ~1000x smaller, as the exclusion said?")
print("=" * 72)
print(f"  median |vals| among LIVE positions   capture {np.median(cl):.3e}"
      f"   decay {np.median(dl):.3e}   ratio {np.median(dl)/np.median(cl):.0f}x")
print(f"  95th pct floor from DEAD positions   capture {fc:.3e}"
      f"   decay {fd:.3e}   ratio {fd/fc:.1f}x")

print("\n" + "=" * 72)
print("THE DECIDING STATISTIC -- each branch against ITS OWN floor")
print("=" * 72)
rc, rd = np.median(cl) / fc, np.median(dl) / fd
print(f"  capture  median |vals| / own floor   {rc:>10.1f}")
print(f"  decay    median |vals| / own floor   {rd:>10.1f}")
print(f"  capture is {rd/rc:.1f}x worse in signal-to-noise")
for q in (10, 25, 50, 75, 90):
    print(f"    {q:>3}th pct   capture {np.percentile(cl,q)/fc:>10.1f}"
          f"   decay {np.percentile(dl,q)/fd:>10.1f}")
print(f"\n  live positions clearing their own floor:"
      f"   capture {(cl>fc).mean():.3f}   decay {(dl>fd).mean():.3f}")

r = rd / rc
v = ("EXCLUSION DEAD -- capture is as readable as decay" if r <= 2
     else "USABLE WITH CARE -- reduced resolution, carried as a bound" if r <= 10
     else "EXCLUSION WAS RIGHT -- and its real reason was never written down")
print(f"\n  -> {v}")
