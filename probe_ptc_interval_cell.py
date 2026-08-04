# SAMPLE SIZE FOR THE PTC-INTERVAL CELL.
# The cell is: transcript positions that are DOWNSTREAM of the operative stop but
# UPSTREAM of the annotated stop -- coding sequence sitting in a post-termination
# position. It exists only where the ORF the model commits to terminates before the
# annotated one, and it is the cell no artificial background can construct.
import h5py, numpy as np
f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
n = f["vals_decay"].shape[0]
c_off, c_cnt = f["cand_offset"][:], f["cand_count"][:]
c_end, c_ref = f["cand_orf_end"][:], f["cand_is_ref_cds"][:]
p_sel = f["p_select"][:]
lab = f["labels"][:]
w = f["sampling_weight"][:] if "sampling_weight" in f else np.ones(n)

has_ref = np.zeros(n, bool); premature = np.zeros(n, bool)
n_int = np.zeros(n, np.int64)      # positions in the PTC interval
n_dn  = np.zeros(n, np.int64)      # positions downstream of the ANNOTATED stop
n_val = np.zeros(n, np.int64)
for i in range(n):
    v = f["valid"][i]; obs = f["obs"][i]
    ok = v & (obs >= 0)
    n_val[i] = ok.sum()
    sl = slice(int(c_off[i]), int(c_off[i]) + int(c_cnt[i]))
    r = np.flatnonzero(c_ref[sl] == 1)
    if not len(r): continue
    has_ref[i] = True
    ref_end = int(c_end[sl][int(r[0])])
    op_end  = int(c_end[sl][int(np.argmax(p_sel[sl]))])
    pos = np.arange(1, len(v) + 1)
    n_dn[i] = int((ok & (pos > ref_end)).sum())
    if op_end < ref_end:
        premature[i] = True
        n_int[i] = int((ok & (pos > op_end) & (pos <= ref_end)).sum())

def row(name, m):
    if not m.any():
        print(f"  {name:<44} {0:>7}"); return
    print(f"  {name:<44} {int(m.sum()):>7,}   weighted {int(w[m].sum()):>9,}")
print(f"  transcripts in bank                          {n:>7,}\n")
row("with an annotated (reference) candidate", has_ref)
row("  ...NMD", has_ref & (lab == 1))
row("  ...control", has_ref & (lab == 0))
print()
row("operative stop BEFORE annotated stop (PTC-like)", premature)
row("  ...NMD", premature & (lab == 1))
row("  ...control", premature & (lab == 0))
print()
sel = premature & (n_int > 0)
print(f"  transcripts with >=1 position in the interval  {int(sel.sum()):>7,}")
if sel.any():
    print(f"    interval positions: total {int(n_int[sel].sum()):>10,}   "
          f"median/tx {int(np.median(n_int[sel])):>6,}   max {int(n_int[sel].max()):>6,}")
    print(f"    NMD    tx {int((sel&(lab==1)).sum()):>6,}   positions "
          f"{int(n_int[sel&(lab==1)].sum()):>9,}")
    print(f"    control tx {int((sel&(lab==0)).sum()):>5,}   positions "
          f"{int(n_int[sel&(lab==0)].sum()):>9,}")
print()
print("  COMPARISON CELL -- positions downstream of the ANNOTATED stop (true 3'UTR)")
m = has_ref & (n_dn > 0)
print(f"    transcripts {int(m.sum()):>6,}   positions {int(n_dn[m].sum()):>10,}   "
      f"median/tx {int(np.median(n_dn[m])):>6,}")
print(f"\n  at top 1% elevation, expected elevated positions in the PTC interval: "
      f"~{int(0.01*n_int[sel].sum()):,}")
