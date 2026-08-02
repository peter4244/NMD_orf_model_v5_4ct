"""
model_recovery_gencode_nmd.py — the recovery benchmark against the REAL gold standard.

SCOPE: a claim ABOUT THE MODEL.

WHY THIS EXISTS. The earlier rebuild (job 8900473) defined "the annotated frame is
the decay-causing one" STRUCTURALLY, as `n_downstream_ejc > 0` on the annotated
candidate. Pete asked whether we had used GENCODE's own NMD biotype, and we had
not. The structural proxy is CIRCULAR in a way the biotype is not: it uses the EJC
rule to build a benchmark that partly tests whether the model uses the EJC rule.

GENCODE's `transcript_type == "nonsense_mediated_decay"` is a curated annotation
call made independently of anything we compute. For those transcripts the annotated
CDS IS the PTC-bearing frame by GENCODE's own judgement, so RECOVERY AND
DECAY-CAUSATION COINCIDE WITH NO CIRCULARITY.

  bank transcripts                       4,999
  ENST-named (mappable to GENCODE)       2,645   all 2,645 matched the GTF
  GENCODE nonsense_mediated_decay        1,125
  GENCODE protein_coding                 1,332   the contrast

ANCHOR: `cand_is_gencode_start == 1`. NEVER astype(bool) -- that column is int8
with a -1 sentinel meaning "no GENCODE CDS", and -1 is truthy. That defect
produced a retracted result earlier today.

REGISTERED BEFORE THE RUN:

  A  recovery on GENCODE NMD-biotype transcripts is comparable to the structural
     proxy's 0.821 prior / 0.885 posterior -> the proxy was measuring the right
     thing and the circularity did not bite.
  B  materially different -> the proxy was measuring something else, and the
     structural result should be replaced rather than supplemented.

  And the non-circular version of the earlier finding: the POSTERIOR should beat
  the PRIOR on NMD-biotype transcripts (where the annotated frame is the
  decay-causing one) and should LOSE to it on protein_coding (where it is not).
  That is the decay-seeking-correction claim, tested without the EJC rule in the
  target definition.

Also reported: agreement between GENCODE biotype and the structural proxy, which
says directly how good the proxy was.

Run from the repo root.
"""
import numpy as np, h5py, pandas as pd

bio = pd.read_csv("analysis_plans/gencode_biotype_bank.tsv", sep="\t")
bmap = dict(zip(bio.isoform_id, bio.gencode_biotype))

f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
tx = np.array([s.decode() for s in f["transcript_id"][:]])
off, cnt = f["cand_offset"][:], f["cand_count"][:]
ps_, pd_, ej_ = f["p_select"][:], f["p_decay"][:], f["cand_n_downstream_ejc"][:]
gen_ = f["cand_is_gencode_start"][:]
lab = f["labels"][:]

rows, agree = {}, []
for i in range(len(cnt)):
    bt = bmap.get(tx[i])
    if bt is None:
        continue
    lo, k = int(off[i]), int(cnt[i])
    if k < 2:
        continue
    a = gen_[lo:lo + k] == 1          # NEVER astype(bool)
    if not a.any():
        continue
    ai = int(np.flatnonzero(a)[0])
    ps, pdc = ps_[lo:lo + k], pd_[lo:lo + k]
    grp = ("NMD_biotype" if bt == "nonsense_mediated_decay"
           else "protein_coding" if bt == "protein_coding" else None)
    if grp is None:
        continue
    r = rows.setdefault(grp, [0, 0, 0])
    r[0] += bool(a[int(np.argmax(ps))])
    r[1] += bool(a[int(np.argmax(ps * pdc))])
    r[2] += 1
    agree.append((grp == "NMD_biotype", int(ej_[lo + ai]) > 0))

print("=" * 74)
print("RECOVERY AGAINST GENCODE BIOTYPE -- no EJC rule in the target definition")
print("=" * 74)
print(f"  {'GENCODE biotype':<20} {'n':>6} {'prior':>8} {'posterior':>10} {'post-prior':>11}")
for g in ("NMD_biotype", "protein_coding"):
    r = rows.get(g)
    if r:
        print(f"  {g:<20} {r[2]:>6} {r[0]/r[2]:>8.3f} {r[1]/r[2]:>10.3f}"
              f" {(r[1]-r[0])/r[2]:>+11.3f}")
a, b = rows.get("NMD_biotype"), rows.get("protein_coding")
if a and b:
    print(f"\n  posterior helps on NMD-biotype: {(a[1]-a[0])/a[2]:+.3f}")
    print(f"  posterior helps on protein_coding: {(b[1]-b[0])/b[2]:+.3f}")
    print("  the decay-seeking-correction claim predicts + on the first, - on the second")

ag = np.array(agree)
print("\n" + "=" * 74)
print("HOW GOOD WAS THE STRUCTURAL PROXY?")
print("=" * 74)
tp = int((ag[:, 0] & ag[:, 1]).sum()); fp = int((~ag[:, 0] & ag[:, 1]).sum())
fn = int((ag[:, 0] & ~ag[:, 1]).sum()); tn = int((~ag[:, 0] & ~ag[:, 1]).sum())
print(f"                        proxy says PTC-bearing   proxy says not")
print(f"  GENCODE NMD biotype   {tp:>20,} {fn:>17,}")
print(f"  GENCODE protein_cod   {fp:>20,} {tn:>17,}")
print(f"  agreement {(tp+tn)/max(1,tp+tn+fp+fn):.3f}"
      f"   proxy precision {tp/max(1,tp+fp):.3f}   proxy recall {tp/max(1,tp+fn):.3f}")
