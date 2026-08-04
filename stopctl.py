import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
ce, isref = f["cand_orf_end"][:], f["cand_is_ref_cds"][:]
psel = f["p_select"][:]
NT = "ACGT"; W = 12
prof = np.zeros((2*W+1, 4)); bg = np.zeros(4)
rank_at_stop = []; n = 0
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 200: continue
    r = np.flatnonzero(isref[lo:lo+nk] == 1)
    k = int(r[0]) if len(r) else int(np.argmax(psel[lo:lo+nk]))
    stop_last = int(ce[lo+k]) - 1                 # last base of the stop codon
    v = f["vals_decay"][i,:P].astype(np.float64); o = f["obs"][i,:P]
    with np.errstate(invalid="ignore"):
        e = np.nanmax(np.abs(v), axis=1)
    ok = f["valid"][i,:P] & np.isfinite(e) & (o >= 0)
    idx = np.flatnonzero(ok)
    if len(idx) < 200 or not (W < stop_last < P - W): continue
    # where do the three stop-codon bases rank within this transcript?
    order = np.argsort(np.argsort(e[idx]))        # 0 = smallest
    pos_of = {p: j for j, p in enumerate(idx)}
    for d in (-3, -2, -1):                        # 0-based indices of the codon
        p = stop_last - 1 + d + 1
        if p in pos_of:
            rank_at_stop.append(1.0 - order[pos_of[p]]/max(len(idx)-1, 1))
    for base in range(4): bg[base] += (o[idx] == base).sum()
    for d in range(-W, W+1):
        p = stop_last + d
        if 0 <= p < P and o[p] >= 0:
            prof[d+W, o[p]] += 1
    n += 1
q = bg/bg.sum()
rs = np.array(rank_at_stop)
print(f"  {n} transcripts, anchor = reference candidate where present")
print(f"\n  KNOWN ANSWER 1 — are the stop-codon bases elevated?")
print(f"    percentile rank of the 3 stop bases within their own transcript")
print(f"      median {100*np.median(rs):.1f}   mean {100*rs.mean():.1f}   "
      f"fraction in top 1% {100*(rs>=0.99).mean():.1f}%   n {len(rs):,}")
print(f"\n  KNOWN ANSWER 2 — does the composition profile show the stop codon?")
print(f"    {'offset':>7} " + "".join(f"{NT[j]:>8}" for j in range(4)) + "   (offset 0 = last base of the stop codon)")
for d in range(-6, 7):
    p = prof[d+W]/max(prof[d+W].sum(), 1)
    star = "  <-- stop codon" if d in (-2,-1,0) else ""
    print(f"    {d:>+7} " + "".join(f"{p[j]:>8.3f}" for j in range(4)) + star)
