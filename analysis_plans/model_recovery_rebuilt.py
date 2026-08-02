"""
model_recovery_rebuilt.py — the recovery benchmark, on transcripts where it means
what we thought it meant.

SCOPE: a claim ABOUT THE MODEL.

WHY C2 / C3 / C11 NEEDED REBUILDING (Pete, 2026-08-02). Those three score the model
on recovering the ANNOTATED reference CDS. For an NMD substrate that is close to a
DISJOINT concept from finding the decay-causing ORF, because the mechanism is that
a non-canonical frame carries the stop.

OUR OWN ATF4 RESULT IS THE PROOF. The posterior flips 18x onto a 179-nt uORF and
puts 93% of the NMD signal there -- the textbook mechanism, recovered -- and on the
old benchmark that scores as a MISS. A benchmark on which the right answer is
wrong is not a benchmark.

THE FIX, identified structurally rather than by biotype label because no biotype
column exists in our tables. **A transcript's annotated CDS is itself PTC-bearing
when its own stop has a junction downstream** -- `cand_n_downstream_ejc > 0` on the
reference candidate. That is the GENCODE NMD-biotype case by construction, and in
those transcripts RECOVERING THE REFERENCE IS FINDING THE DECAY-CAUSING ORF, so
recovery and decay-causation coincide and the benchmark is meaningful.

Reported against the complement -- transcripts whose reference CDS is NOT
PTC-bearing -- where the two concepts come apart and the old benchmark was
measuring main-ORF recovery.

Both anchors are used and compared: `cand_is_ref_cds` (the pool's reference) and
`cand_is_gencode_start` (the GENCODE annotation directly), because the two need not
agree and an unexamined choice between them is the error class this project has.

REGISTERED BEFORE THE RUN:

  MEANINGFUL   recovery on PTC-bearing-reference transcripts is comparable to the
               0.697 headline -> the model finds decay-causing ORFs about as well
               as it finds main ORFs, and C2/C3/C11 transfer with relabelling.
  WORSE        recovery is materially lower there -> the model is good at main-ORF
               recovery and BAD at the thing we care about, and the 0.697 headline
               was measuring the easy case.
  BETTER       recovery is higher -> the model is tuned to decay-causing frames,
               which would be the backwards-reasoning account showing up in the
               benchmark itself.

Posterior recovery (argmax p_select*d) is reported beside prior recovery
(argmax p_select) throughout, since the posterior is what the prediction is about.

Run from the repo root.
"""
import numpy as np, h5py

f = h5py.File("results_ism_v6/bank_interp_s100.h5", "r")
off, cnt = f["cand_offset"][:], f["cand_count"][:]
ps_, pd_, ej_ = f["p_select"][:], f["p_decay"][:], f["cand_n_downstream_ejc"][:]
ref_ = f["cand_is_ref_cds"][:]
gen_ = f["cand_is_gencode_start"][:] if "cand_is_gencode_start" in f else None
lab = f["labels"][:]
N = len(cnt)

def run(anchor, name):
    rows = {}
    for i in range(N):
        lo, k = int(off[i]), int(cnt[i])
        if k < 2:
            continue
        a = anchor[lo:lo + k].astype(bool)
        if not a.any():
            continue
        ai = int(np.flatnonzero(a)[0])
        ps, pd = ps_[lo:lo + k], pd_[lo:lo + k]
        ptc = int(ej_[lo + ai]) > 0                     # is the ANCHOR PTC-bearing?
        key = (ptc, int(lab[i]))
        r = rows.setdefault(key, [0, 0, 0])
        r[0] += bool(a[int(np.argmax(ps))])             # prior recovery
        r[1] += bool(a[int(np.argmax(ps * pd))])        # posterior recovery
        r[2] += 1
    print(f"\n{'='*74}\nANCHOR = {name}\n{'='*74}")
    print(f"  {'stratum':<44} {'n':>6} {'prior':>7} {'post':>7}")
    for ptc in (True, False):
        for lv in (1, 0):
            r = rows.get((ptc, lv))
            if not r:
                continue
            tag = ("anchor IS PTC-bearing" if ptc else "anchor NOT PTC-bearing")
            tag += ", " + ("NMD" if lv else "control")
            print(f"  {tag:<44} {r[2]:>6} {r[0]/r[2]:>7.3f} {r[1]/r[2]:>7.3f}")
    # the headline comparison, NMD transcripts only
    a = rows.get((True, 1)); b = rows.get((False, 1))
    if a and b:
        pa, pb = a[0]/a[2], b[0]/b[2]
        qa, qb = a[1]/a[2], b[1]/b[2]
        print(f"\n  NMD transcripts, prior recovery:      PTC-bearing anchor {pa:.3f}"
              f"   non-PTC {pb:.3f}   gap {pa-pb:+.3f}")
        print(f"  NMD transcripts, posterior recovery:  PTC-bearing anchor {qa:.3f}"
              f"   non-PTC {qb:.3f}   gap {qa-qb:+.3f}")
        v = ("MEANINGFUL -- comparable to the 0.697 headline" if pa >= 0.60
             else "WORSE -- the headline was measuring the easy case" if pa < 0.50
             else "INTERMEDIATE")
        print(f"  -> {v}")

run(ref_, "cand_is_ref_cds (pool reference)")
if gen_ is not None:
    run(gen_, "cand_is_gencode_start (GENCODE annotation)")
else:
    print("\ncand_is_gencode_start NOT PRESENT in this bank -- anchor comparison skipped")
