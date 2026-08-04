import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
ce, isref, psel = f["cand_orf_end"][:], f["cand_is_ref_cds"][:], f["p_select"][:]
NT="ACGT"; W=8
COD = {(3,0,0):"TAA",(3,0,2):"TAG",(3,2,0):"TGA"}
rk = {c: {d: [] for d in range(-W,W+1)} for c in COD.values()}
comp = {c: np.zeros((2*W+1,4)) for c in COD.values()}
for i in range(600):
    lo, nk = int(co[i]), int(cc[i]); b = sp[lo:lo+nk]
    P = int(max(b[:,3].max(), b[:,5].max()))
    if P < 200: continue
    r = np.flatnonzero(isref[lo:lo+nk]==1)
    k = int(r[0]) if len(r) else int(np.argmax(psel[lo:lo+nk]))
    s = int(ce[lo+k]) - 1
    if not (W < s < P-W): continue
    v = f["vals_decay"][i,:P].astype(np.float64); o = f["obs"][i,:P]
    with np.errstate(invalid="ignore"): e = np.nanmax(np.abs(v), axis=1)
    ok = f["valid"][i,:P] & np.isfinite(e) & (o>=0)
    idx = np.flatnonzero(ok)
    if len(idx) < 200: continue
    c = COD.get((o[s-2], o[s-1], o[s]))
    if c is None: continue
    # PERCENTILE RANK, not fold over median: 2.8% of transcripts have a median
    # importance below 1e-6 and dividing by it produces 1e8-fold artefacts.
    pct = {p: 1.0 - j/max(len(idx)-1,1) for p, j in
           zip(idx, np.argsort(np.argsort(-e[idx])))}
    for d in range(-W, W+1):
        if s+d in pct: rk[c][d].append(pct[s+d])
        if 0 <= s+d < P and o[s+d] >= 0: comp[c][d+W, o[s+d]] += 1
print("  stop usage: " + "  ".join(f"{c} {len(rk[c][0])}" for c in COD.values()))
print(f"\n  MEDIAN PERCENTILE RANK of importance, by offset and stop codon")
print(f"    {'offset':>7} " + "".join(f"{c:>9}" for c in COD.values()) + "     note")
for d in range(-2, 7):
    row = "".join(f"{100*np.median(rk[c][d]):>9.1f}" for c in COD.values())
    tag = "  stop" if d <= 0 else ("  +4 position" if d == 1 else "")
    print(f"    {d:>+7} " + row + tag)
print(f"\n  BASE COMPOSITION at +4, by stop identity (data property, not model)")
print(f"    {'stop':>6} " + "".join(f"{NT[j]:>8}" for j in range(4)))
for c in COD.values():
    p = comp[c][1+W]/max(comp[c][1+W].sum(),1)
    print(f"    {c:>6} " + "".join(f"{p[j]:>8.3f}" for j in range(4)))
print(f"    {'bg':>6} " + "   0.251   0.248   0.259   0.242")
