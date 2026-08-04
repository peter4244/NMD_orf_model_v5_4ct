import h5py, numpy as np, pandas as pd
sub = pd.read_csv("results_ism_v6/ism_subset.tsv", sep="\t")
f = h5py.File("results_ism_v6/bank_interp_s100.h5","r")
tx = [s.decode() for s in f["transcript_id"][:]]
lab = f["labels"][:]
co, cc = f["cand_offset"][:], f["cand_count"][:]
ce, isref, psel = f["cand_orf_end"][:], f["cand_is_ref_cds"][:], f["p_select"][:]
m = dict(zip(sub.isoform_id, zip(sub.main_orf_stop, sub.is_nmd, sub.cell)))
rows=[]
for i,t in enumerate(tx):
    lo,nk=int(co[i]),int(cc[i])
    r=np.flatnonzero(isref[lo:lo+nk]==1)
    if not len(r): continue
    ref_stop=int(ce[lo+int(r[0])])
    k=int(np.argmax(psel[lo:lo+nk])); sel_stop=int(ce[lo+k])
    mos, isnmd, cell = m.get(t,(None,None,None))
    rows.append((t, sel_stop < ref_stop, bool(isnmd), mos, cell))
d=pd.DataFrame(rows, columns=["tx","model_stop_early","is_nmd","main_orf_stop","cell"])
print(f"  {len(d):,} transcripts with a reference candidate\n")
print("  CELL AS CURRENTLY DEFINED — model-selected stop before the annotated stop")
sel=d[d.model_stop_early]
print(f"    n = {len(sel):,}   NMD {int(sel.is_nmd.sum()):,}   control {int((~sel.is_nmd).sum()):,}"
      f"   -> {100*sel.is_nmd.mean():.1f}% NMD")
print(f"    background prevalence in the same set: {100*d.is_nmd.mean():.1f}% NMD\n")
print("  ANNOTATION-DERIVED ALTERNATIVE — subset table's main_orf_stop flag")
print(f"    main_orf_stop values: {d.main_orf_stop.value_counts(dropna=False).to_dict()}")
for v in d.main_orf_stop.dropna().unique():
    g=d[d.main_orf_stop==v]
    print(f"      main_orf_stop={v}:  n {len(g):,}  {100*g.is_nmd.mean():.1f}% NMD")
print(f"\n  CROSSED — does the annotation flag separate the model-stop-early cell?")
for mse in (True, False):
    g=d[d.model_stop_early==mse]
    if len(g)==0: continue
    xt=pd.crosstab(g.main_orf_stop, g.is_nmd)
    print(f"    model_stop_early={mse}:")
    print("      " + xt.to_string().replace("\n","\n      "))
