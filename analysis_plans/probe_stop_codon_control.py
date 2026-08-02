"""
probe_stop_codon_control.py — does the decay branch read the stop codon it is
anchored on?

THE POSITIVE CONTROL THIS WAS BUILT TO BE, AND IT FAILS. The decay branch is
anchored on a stop codon by construction, so if any sequence feature should move
`vals_decay` it is that one. If the method cannot recover it the method is
suspect; if the method is sound and the codon is still not elevated, the MODEL is
not reading it. The composition profile settles which -- it recovers UAA/UAG/UGA
exactly, so the coordinate system is right and the model is what is silent.

TWO THINGS THIS GETS RIGHT THAT THE FIRST VERSION DID NOT.
  - The codon is at (s-2, s-1, s) where s = cand_orf_end - 1. Sampling
    (s-3, s-2, s-1) put one base outside the codon and moved the answer from the
    75th percentile to the 24th.
  - Matched controls at +/-25 and 30 from the same anchor, so "elevated" is judged
    against the stop window's own level rather than against the transcript.

Verified before scoring: the first codon base must be T, or the transcript is
skipped rather than silently mis-anchored.
"""

import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
ce, isref, psel = f["cand_orf_end"][:], f["cand_is_ref_cds"][:], f["p_select"][:]
NT = "ACGT"
rk = {d: [] for d in (-2,-1,0)}; rk_ctrl = []
codon_ok = 0; n = 0
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 200: continue
    r = np.flatnonzero(isref[lo:lo+nk] == 1)
    k = int(r[0]) if len(r) else int(np.argmax(psel[lo:lo+nk]))
    s = int(ce[lo+k]) - 1
    v = f["vals_decay"][i,:P].astype(np.float64); o = f["obs"][i,:P]
    with np.errstate(invalid="ignore"):
        e = np.nanmax(np.abs(v), axis=1)
    ok = f["valid"][i,:P] & np.isfinite(e) & (o >= 0)
    idx = np.flatnonzero(ok)
    if len(idx) < 200 or not (20 < s < P-20): continue
    # VERIFY the codon is where we think before scoring it
    tri = [o[s-2], o[s-1], o[s]]
    if tri[0] != 3: continue                     # must be T
    codon_ok += 1
    pct = {p: 1.0 - j/max(len(idx)-1,1) for p, j in
           zip(idx, np.argsort(np.argsort(-e[idx])))}
    for d in (-2,-1,0):
        if s+d in pct: rk[d].append(pct[s+d])
    for d in (-30,-25,25,30):                    # matched controls, same window
        if s+d in pct: rk_ctrl.append(pct[s+d])
    n += 1
print(f"  {n} transcripts scored, stop codon verified T-first in all of them")
print(f"\n  percentile rank of each stop-codon base within its own transcript")
print(f"    {'base':>8} {'median pct':>12} {'mean pct':>10} {'in top 1%':>11}  n")
for d, nm in ((-2,'stop[1] T'), (-1,'stop[2]'), (0,'stop[3]')):
    a = np.array(rk[d])
    print(f"    {nm:>8} {100*np.median(a):>12.1f} {100*a.mean():>10.1f} "
          f"{100*(a>=0.99).mean():>10.1f}%  {len(a):,}")
c = np.array(rk_ctrl)
print(f"    {'control':>8} {100*np.median(c):>12.1f} {100*c.mean():>10.1f} "
      f"{100*(c>=0.99).mean():>10.1f}%  {len(c):,}   (+/-25,30 from the stop)")
print(f"\n  If the stop bases are not above the control, the decay branch is not")
print(f"  reading the stop codon it is anchored on.")
