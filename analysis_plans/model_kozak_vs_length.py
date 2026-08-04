"""
model_kozak_vs_length.py — does the head read initiation context where biology says
it must?

SCOPE. The measurements are ABOUT THE MODEL. The motivating reasoning is background
biology and is labelled as such; the bridge between them is an inference, not a
result.

THE BIOLOGY THAT MOTIVATES IT (Pete, 2026-08-02; background, not measured here).
The candidate pool is 69.1% ORFs under 200 nt, median 81 nt. By the EJC rule a
short upstream ORF terminating early in a multi-exon transcript has junctions
behind it and should trigger decay. Most do not -- because RIBOSOMES MOSTLY DO NOT
USE THEM. Scanning leaks past weak start codons, and after terminating at a short
uORF the 40S can resume scanning and initiate downstream; short uORFs are
especially permissive of reinitiation. So the reason most short candidates produce
no decay is that INITIATION never happens there.

That is exactly the job p_capture exists to do. Suppressing the short-ORF mass is
the design, not a side effect.

WHAT WE MEASURED THAT SITS AWKWARDLY WITH IT. capture ~ ORF length is +0.760 and
capture ~ kozak_score is +0.124 (job 8898926) -- a factor of six the wrong way
round. The head suppresses short ORFs because they are SHORT, not because they
initiate poorly. And the encoding marks short candidates with a 47x discriminative
fill signature before any sequence is read.

THE CONSEQUENCE, WHICH IS WHAT THIS TESTS. A length-driven head should fail
specifically on SHORT ORFs WITH STRONG INITIATION CONTEXT -- biologically those ARE
used, and they are the uORF-mediated NMD substrates.

--------------------------------------------------------------------------------
REGISTERED PREDICTIONS, fixed before the run.

  A -- LENGTH-DRIVEN. Selection accuracy on transcripts whose reference ORF is
       SHORT (<200 nt) is much lower than on long-reference transcripts, and
       capture ~ kozak among short candidates is near zero. The head is not
       reading initiation context where biology says it decides.

  B -- INITIATION-CONTEXT. Accuracy is comparable across the two, and kozak
       carries real discrimination among short candidates. The +0.124 aggregate
       understated it because length dominates the pooled figure.

  SPLIT -- accuracy drops but kozak still discriminates: the head reads context
       AND is overridden by length. Reported as its own outcome, not rounded to
       either.

RESTRICTION RATHER THAN PARTIALLING, deliberately, per Pete's correction: length
is upstream of window content rather than a confound beside it, so conditioning on
it blocks the path being measured. Restricting to short candidates limits length
range naturally and needs no adjustment.

Run from the repo root.
"""
import argparse
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
a = ap.parse_args()


def sp(x, y):
    if len(x) < 4:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


with h5py.File(a.bank, "r") as f:
    off, cnt = f["cand_offset"][:], f["cand_count"][:]
    ps_, pc_ = f["p_select"][:], f["p_capture"][:]
    ref_, koz_ = f["cand_is_ref_cds"][:], f["cand_kozak_score"][:]
    os_, oe_ = f["cand_orf_start"][:], f["cand_orf_end"][:]
    lab = f["labels"][:]
    N = len(cnt)
    ck = f.attrs.get("checkpoint", "?")

hit = {"short": [0, 0], "long": [0, 0]}
hit_nmd = {"short": [0, 0], "long": [0, 0]}
r_koz_short, r_koz_long, r_len_short = [], [], []
koz_rescue = {"hi": [0, 0], "lo": [0, 0]}

for i in range(N):
    lo, k = int(off[i]), int(cnt[i])
    if k < 3:
        continue
    ref = ref_[lo:lo + k].astype(bool)
    ln = (oe_[lo:lo + k] - os_[lo:lo + k]).astype(float)
    pc, ps, kz = pc_[lo:lo + k], ps_[lo:lo + k], koz_[lo:lo + k].astype(float)

    short = ln < 200
    if short.sum() >= 4:
        r_koz_short.append(sp(pc[short], kz[short]))
        r_len_short.append(sp(pc[short], ln[short]))
    if (~short).sum() >= 4:
        r_koz_long.append(sp(pc[~short], kz[~short]))

    if not ref.any():
        continue
    ri = int(np.flatnonzero(ref)[0])
    grp = "short" if ln[ri] < 200 else "long"
    ok = bool(ref[int(np.argmax(ps))])
    hit[grp][0] += ok; hit[grp][1] += 1
    if int(lab[i]) == 1:
        hit_nmd[grp][0] += ok; hit_nmd[grp][1] += 1
    # does a strong Kozak rescue a short reference ORF?
    if grp == "short":
        b = "hi" if kz[ri] >= np.median(kz) else "lo"
        koz_rescue[b][0] += ok; koz_rescue[b][1] += 1

print(f"BANK {a.bank}\ncheckpoint {ck}\n")
print("=" * 72)
print("DOES THE MODEL FIND THE REFERENCE ORF WHEN IT IS SHORT?")
print("=" * 72)
for g in ("long", "short"):
    h, n = hit[g]
    if n:
        print(f"  reference ORF {g:<6}  accuracy {h/n:.3f}   n {n:,}")
hl, nl = hit["long"]; hs, ns = hit["short"]
if nl and ns:
    print(f"  gap long - short = {hl/nl - hs/ns:+.3f}")
print("  NMD transcripts only:")
for g in ("long", "short"):
    h, n = hit_nmd[g]
    if n:
        print(f"    reference ORF {g:<6}  accuracy {h/n:.3f}   n {n:,}")

print("\n" + "=" * 72)
print("DOES A STRONG KOZAK RESCUE A SHORT REFERENCE ORF?")
print("=" * 72)
for b, nm in (("hi", "kozak >= median"), ("lo", "kozak <  median")):
    h, n = koz_rescue[b]
    if n:
        print(f"  short reference, {nm:<16} accuracy {h/n:.3f}   n {n:,}")

print("\n" + "=" * 72)
print("WHAT DOES CAPTURE READ AMONG SHORT CANDIDATES?")
print("=" * 72)


def rep(nm, x):
    x = np.array(x, float); x = x[np.isfinite(x)]
    print(f"  {nm:<44} median {np.median(x):+.3f}   n {len(x):,}")
    return float(np.median(x))


ks = rep("capture ~ kozak, ORFs < 200 nt", r_koz_short)
rep("capture ~ kozak, ORFs >= 200 nt", r_koz_long)
ls = rep("capture ~ length, ORFs < 200 nt", r_len_short)
print(f"\n  among short candidates, length beats kozak by "
      f"{abs(ls)/max(abs(ks), 1e-9):.1f}x")
v = ("A -- LENGTH-DRIVEN" if abs(ks) < 0.10 and (nl and ns and hl/nl - hs/ns > 0.15)
     else "B -- INITIATION-CONTEXT" if abs(ks) >= 0.10 and (nl and ns and hl/nl - hs/ns <= 0.15)
     else "SPLIT -- reads context AND is overridden by length")
print(f"\n  -> {v}")
