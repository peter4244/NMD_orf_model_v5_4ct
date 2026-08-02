"""
model_gate_vs_ranker_gencode.py — re-score the gate-vs-ranker comparison against the
right target.

SCOPE: a claim ABOUT THE MODEL.

WHY THIS EXISTS. Pete, 2026-08-02: section 1 of the narrative argues the picker is a
GATE rather than a RANKER, and it argues it with three numbers scored on recovering
the ANNOTATED MAIN ORF -- 0.305 head argmax, 0.424 most-5', 0.697 queue. That is the
target section 4 of the same document spends a section establishing is WRONG. For an
NMD substrate the main ORF is not necessarily the frame being picked, and it is not
what the model is trying to find. The document contradicted itself two sections
apart.

  "This should not be about the head picking the annotated main ORF -- that is not
   necessarily the ORF that is actually being picked under the NMD case, and that is
   also not what the model is trying to do."

THE FIX, WHICH IS PETE'S. On GENCODE `nonsense_mediated_decay` transcripts the
annotated CDS IS the PTC-bearing frame by GENCODE's own curation, so recovery and
decay-causation coincide with no circularity and no EJC rule in the target. That is
already the section 4 benchmark (job 8900631, 0.793 prior / 0.883 posterior). It has
only ever been run with two arms. This runs the SAME FIVE ARMS on it, so the gate
argument is made on the target the document says is correct.

  1  head's own argmax       argmax p_capture    the head ALONE, no queue
  2  most 5' candidate       argmin orf_start    pure position, no model
  3  longest ORF             argmax orf_length   pure length, no model
  4  the queue (prior)       argmax p_select     the head THROUGH stick-breaking
  5  posterior               argmax p_select*d   what the prediction is about

Arms 1/2/4 are section 1's three numbers on the right target. Arm 3 is carried
because it costs one line in the same loop and it is the `[unclaimed]` heuristic
FLOOR UNDER 0.883 that both windows flagged as the largest open gap -- the 0.678
longest-ORF figure belongs to main-ORF recovery, a different target, and cannot be
quoted here.

ANCHOR: `cand_is_gencode_start == 1`. NEVER astype(bool) -- int8 with a -1 sentinel
meaning "no GENCODE CDS", and -1 is truthy. That defect produced a retracted result
on 2026-08-02 (0.941 -> 0.885 -> 0.883).

`protein_coding` is carried as the contrast throughout, because an arm that wins on
both groups is finding annotated CDSs, not decay-causing frames.

REGISTERED BEFORE THE RUN. The gate claim is two separable things and only one is at
risk here:

  THE MECHANISM is read off the code -- p_select_k = p_capture_k * prod(1-p_capture_j)
  is stick-breaking in model_v6.py and is true whatever we score against. Not at risk.

  THE DEMONSTRATION is the three-way comparison, and it is entirely target-dependent.

  G1  queue > head's own argmax SURVIVES on NMD-biotype transcripts
        -> section 1 stands, with these numbers replacing the main-ORF ones.
  G2  queue <= head's own argmax
        -> the gate DEMONSTRATION was an artifact of the main-ORF target and section
           1's table is RETRACTED. The mechanism sentence survives on code alone and
           must be labelled as derived-not-measured.

  MY PREDICTIONS, recorded so they can be wrong:

  - G1 holds. Stick-breaking is architectural and the queue cannot help but differ
    from the head's raw argmax; the ordering should be robust to the target.
  - POSITION FALLS. Most-5' scored 0.424 on main ORFs because the main ORF is often
    the first substantial frame. On NMD-biotype transcripts the most 5' candidate is
    frequently a uORF and NOT the PTC-bearing CDS, so this arm should drop
    materially. This is the prediction with teeth: if position holds up near 0.424
    the two targets are not as distinct as section 4 claims.
  - LENGTH STAYS HIGH, near or above the 0.678 it got on main ORFs, because an
    NMD-biotype CDS is still typically the longest frame in its transcript. If it
    lands near 0.883 then the model's headline number has almost no room above a
    one-line heuristic, and that belongs in the narrative as a bound.
  - The posterior reproduces 0.883 exactly. It is the same computation as job
    8900631 on the same bank; any drift means one of the two is filtering
    differently and BOTH are suspect until reconciled.

Run from the repo root.
"""
import numpy as np, h5py, pandas as pd

BANK = "results_ism_v6/bank_interp_s100.h5"
ARMS = ["head argmax (p_capture)", "most 5' candidate", "longest ORF",
        "queue / prior (p_select)", "posterior (p_select*d)"]

bio = pd.read_csv("analysis_plans/gencode_biotype_bank.tsv", sep="\t")
bmap = dict(zip(bio.isoform_id, bio.gencode_biotype))

f = h5py.File(BANK, "r")
tx = np.array([s.decode() for s in f["transcript_id"][:]])
off, cnt = f["cand_offset"][:], f["cand_count"][:]
pcap_, ps_, pd_ = f["p_capture"][:], f["p_select"][:], f["p_decay"][:]
st_, en_ = f["cand_orf_start"][:], f["cand_orf_end"][:]
gen_ = f["cand_is_gencode_start"][:]

rows, skipped = {}, {"no_biotype": 0, "k<2": 0, "no_anchor": 0, "other_biotype": 0}
for i in range(len(cnt)):
    bt = bmap.get(tx[i])
    if bt is None:
        skipped["no_biotype"] += 1
        continue
    grp = ("NMD_biotype" if bt == "nonsense_mediated_decay"
           else "protein_coding" if bt == "protein_coding" else None)
    if grp is None:
        skipped["other_biotype"] += 1
        continue
    lo, k = int(off[i]), int(cnt[i])
    if k < 2:
        skipped["k<2"] += 1
        continue
    a = gen_[lo:lo + k] == 1                      # NEVER astype(bool)
    if not a.any():
        skipped["no_anchor"] += 1
        continue

    pcap, ps, pdc = pcap_[lo:lo + k], ps_[lo:lo + k], pd_[lo:lo + k]
    st, en = st_[lo:lo + k].astype(np.int64), en_[lo:lo + k].astype(np.int64)

    picks = [int(np.argmax(pcap)),          # 1 the head alone
             int(np.argmin(st)),            # 2 pure position, not index order
             int(np.argmax(en - st)),       # 3 pure length
             int(np.argmax(ps)),            # 4 the queue
             int(np.argmax(ps * pdc))]      # 5 the posterior

    r = rows.setdefault(grp, [0] * len(ARMS) + [0])
    for j, p in enumerate(picks):
        r[j] += bool(a[p])
    r[-1] += 1
f.close()

print("=" * 78)
print("GATE VS RANKER, SCORED AGAINST GENCODE's OWN NMD CALL")
print("=" * 78)
print(f"bank {BANK}")
print(f"target: cand_is_gencode_start == 1  (the annotated CDS, which on NMD-biotype")
print(f"        transcripts IS the PTC-bearing frame by GENCODE's curation)")
print("skipped: " + "  ".join(f"{k}={v:,}" for k, v in skipped.items()))

hdr = f"\n  {'arm':<28}" + "".join(f"{g:>16}" for g in ("NMD_biotype", "protein_coding"))
print(hdr)
print("  " + "-" * (28 + 32))
for j, name in enumerate(ARMS):
    cells = ""
    for g in ("NMD_biotype", "protein_coding"):
        r = rows.get(g)
        cells += f"{r[j]/r[-1]:>16.3f}" if r else f"{'-':>16}"
    print(f"  {name:<28}{cells}")
r = rows.get("NMD_biotype"); rc = rows.get("protein_coding")
print(f"  {'n':<28}{r[-1] if r else 0:>16,}{rc[-1] if rc else 0:>16,}")

if r:
    head, pos, lng, queue, post = (r[j] / r[-1] for j in range(5))
    print("\n" + "=" * 78)
    print("THE REGISTERED DECISION")
    print("=" * 78)
    print(f"  queue {queue:.3f}  vs  head's own argmax {head:.3f}"
          f"   -> {'G1 SURVIVES' if queue > head else 'G2 -- SECTION 1 TABLE RETRACTED'}")
    print(f"  position {pos:.3f} against 0.424 on main ORFs"
          f"   -> {'fell as predicted' if pos < 0.424 - 0.05 else 'DID NOT FALL -- the two targets are less distinct than section 4 claims'}")
    print(f"  longest-ORF floor under the 0.883 headline: {lng:.3f}"
          f"   -> headroom {post - lng:+.3f}")
    print(f"  posterior {post:.3f} against job 8900631's 0.883"
          f"   -> {'reconciles' if abs(post - 0.883) < 0.005 else 'DRIFT -- both numbers suspect until reconciled'}")
    print("\n  NOTE: arms 1/2/4 are the section 1 comparison; arm 3 is the previously")
    print("  [unclaimed] heuristic floor. Arm 5 is the reconciliation check, not a result.")
