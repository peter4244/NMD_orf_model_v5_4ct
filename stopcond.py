import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
co, cc, sp = f["cand_offset"][:], f["cand_count"][:], f["spans"][:]
ce, isref, psel = f["cand_orf_end"][:], f["cand_is_ref_cds"][:], f["p_select"][:]
NT="ACGT"; W=8
COD = {(3,0,0):"TAA",(3,0,2):"TAG",(3,2,0):"TGA"}
comp = {c: np.zeros((2*W+1,4)) for c in COD.values()}
imp  = {c: np.zeros(2*W+1) for c in COD.values()}
impn = {c: np.zeros(2*W+1) for c in COD.values()}
base = {c: np.zeros(4) for c in COD.values()}
nrm  = {c: [] for c in COD.values()}
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
    if ok.sum() < 200: continue
    tri = (o[s-2], o[s-1], o[s])
    c = COD.get(tri)
    if c is None: continue
    med = np.median(e[ok])
    if med <= 0: continue
    nrm[c].append(med)
    for base_ in range(4): base[c][base_] += (o[ok]==base_).sum()
    for d in range(-W, W+1):
        p = s+d
        if 0 <= p < P and o[p] >= 0:
            comp[c][d+W, o[p]] += 1
            if ok[p]: imp[c][d+W] += e[p]/med; impn[c][d+W] += 1
print(f"  stop codon usage among reference candidates:")
for c in COD.values(): print(f"    {c}  {len(nrm[c]):>4} transcripts")
print(f"\n  IMPORTANCE by offset, as a fold over each transcript's own median")
print(f"  (offset +1 = the base immediately 3' of the stop; the classic readthrough position)")
print(f"    {'offset':>7} " + "".join(f"{c:>9}" for c in COD.values()))
for d in range(-2, 7):
    row = "".join(f"{imp[c][d+W]/max(impn[c][d+W],1):>9.3f}" for c in COD.values())
    tag = "  <- stop" if d <= 0 else ("  <- +4 position" if d == 1 else "")
    print(f"    {d:>+7} " + row + tag)
print(f"\n  BASE COMPOSITION at +1 (the +4 position), by stop codon identity")
print(f"    {'stop':>6} " + "".join(f"{NT[j]:>8}" for j in range(4)) + f"{'n':>8}")
for c in COD.values():
    p = comp[c][1+W]/max(comp[c][1+W].sum(),1)
    print(f"    {c:>6} " + "".join(f"{p[j]:>8.3f}" for j in range(4)) + f"{int(comp[c][1+W].sum()):>8}")
print(f"\n    transcriptome background at these positions, for reference")
tot = sum(base.values()); tot = tot/tot.sum()
print(f"    {'bg':>6} " + "".join(f"{tot[j]:>8.3f}" for j in range(4)))
