#!/usr/bin/env python3
"""
kozak_pwm_rescore.py — replace the pipeline's 3-level Kozak score with the continuous PWM, and
test the pre-registered prediction that it sharpens the composition effect.

WHY. The ORF scan scores initiation context as `as.integer(has_m3) + as.integer(has_p4)`
(05s_orfik_scan.R) -- THREE possible values. Isopair exports `scoreKozakPWM()`, an 8-position
log-odds PWM over -6..-1 and +4,+5 (Cavener & Ray 1991, 3,012 vertebrate mRNAs), and the pipeline
does not use it. Consequences measured elsewhere: 6.3 ORFs tie at each isoform's maximum and
60.8% of isoforms have more ties than candidate slots, so the slot fill is decided by transcript
order among tied candidates.

THE PRE-REGISTERED PREDICTION, stated before this script was run. Under a scanning model what
matters is initiation PROBABILITY, not the presence of an upstream AUG. Measured with the 3-level
score, on isoforms with a clean reference-AUG stop, splitting at the median strong-AUG fraction
within fixed upstream-AUG-count strata (which matches 5'UTR length by construction):

    AUG count      low vs high strong fraction        ratio
        2            2.35%  vs  4.87%                 2.07x
        3-4          5.29%  vs  6.55%                 1.24x
        5-9         10.72%  vs 15.01%                 1.40x
        10+         17.36%  vs 18.09%                 1.04x

If initiation strength is the operative variable, a continuous score should separate these strata
BETTER than a binary one. The prediction is that each ratio RISES. If they do not rise, Kozak
strength is not the operative variable and a scanning architecture built on it is premature --
which is the point of running this before any architecture work.

THE VALIDATION GATE. The PWM is reimplemented in Python for speed over ~1.5M ORFs. That
reimplementation is checked against `Isopair::scoreKozakPWM()` itself on a random sample; if the
maximum absolute difference exceeds 1e-9 the script stops and reports, because every number below
would otherwise be a reimplementation artifact rather than a measurement.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path("/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct")
DATA = REPO / "results_4ct_dn"

# Cavener & Ray 1991 observed frequencies at vertebrate TIS, Isopair R/uorf.R:189-208.
# Columns are -6 -5 -4 -3 -2 -1 +4 +5; rows A C G T.
FREQS = np.array([
    [0.23, 0.25, 0.25, 0.37, 0.27, 0.19, 0.23, 0.17],   # A
    [0.35, 0.22, 0.32, 0.19, 0.33, 0.39, 0.16, 0.31],   # C
    [0.23, 0.33, 0.25, 0.34, 0.17, 0.23, 0.46, 0.34],   # G
    [0.19, 0.20, 0.18, 0.10, 0.23, 0.19, 0.15, 0.18],   # T
])
PWM = np.log2(FREQS / 0.25)
POS = [-6, -5, -4, -3, -2, -1, 4, 5]
# uorf.R:290 -- offsets are relative to the A of ATG at +1, so a negative position is used as-is
# and a positive one is decremented by 1 (since +1=A, +2=T, +3=G, +4=atg+3).
OFFSETS = [p if p < 0 else p - 1 for p in POS]
NT = {"A": 0, "C": 1, "G": 2, "T": 3}


def read_fasta(path, keep=None):
    """Stream, keeping only ids in `keep`. The full transcriptome is 1.6 GB and the scan needs
    ~7% of it; loading all of it is the difference between fitting in memory and not."""
    seqs, name, buf = {}, None, []
    def flush():
        if name is not None and (keep is None or name in keep):
            seqs[name] = "".join(buf)
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    flush()
    return seqs


def score_pwm(seq, atg_pos):
    """Port of Isopair::scoreKozakPWM for one site. atg_pos is 1-based at the A of ATG."""
    n = len(seq)
    s, ns = 0.0, 0
    for j, off in enumerate(OFFSETS):
        p = atg_pos + off            # 1-based
        if p < 1 or p > n:
            continue
        b = NT.get(seq[p - 1])
        if b is None:
            continue
        s += PWM[b, j]
        ns += 1
    return (s if ns else np.nan), ns


def validate_against_r(seqs, sites, n=400, tol=1e-9):
    """Hard gate: the Python port must reproduce Isopair::scoreKozakPWM exactly."""
    rng = np.random.default_rng(0)
    pick = sites.sample(min(n, len(sites)), random_state=0)
    rows = [(seqs[i], int(p)) for i, p in zip(pick.isoform_id, pick.orf_start) if i in seqs]
    if not rows:
        sys.exit("VALIDATION IMPOSSIBLE: no sampled isoform_id found in the FASTA")
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        fh.write("seq\tpos\n")
        for s, p in rows:
            fh.write(f"{s}\t{p}\n")
        tsv = fh.name
    r = subprocess.run(
        ["Rscript", "-e",
         f'd <- read.delim("{tsv}", stringsAsFactors=FALSE); '
         'x <- Isopair::scoreKozakPWM(d$seq, d$pos); '
         'cat(paste(sprintf("%.12f", x$score), collapse="\\n"))'],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"VALIDATION FAILED: Rscript errored\n{r.stderr[:2000]}")
    r_scores = np.array([float(x) if x != "NA" else np.nan for x in r.stdout.split("\n") if x])
    py = np.array([score_pwm(s, p)[0] for s, p in rows])
    ok = ~(np.isnan(r_scores) | np.isnan(py))
    if len(r_scores) != len(py):
        sys.exit(f"VALIDATION FAILED: length mismatch R={len(r_scores)} py={len(py)}")
    d = np.abs(r_scores[ok] - py[ok]).max() if ok.any() else np.inf
    print(f"  validated against Isopair::scoreKozakPWM on {ok.sum()} sites: max |diff| {d:.3e}")
    print(f"  NA agreement: R {np.isnan(r_scores).sum()} vs python {np.isnan(py).sum()}")
    if not (d <= tol):
        sys.exit(f"VALIDATION FAILED: max |diff| {d:.3e} exceeds {tol:.0e}. Stopping — every "
                 f"number below would be a reimplementation artifact.")
    return d


def contrast_table(P, score_col, label):
    """Composition-at-fixed-count: does a median split on initiation strength predict NMD?"""
    print(f"\n  {label}")
    print(f"  {'AUG count':<10} {'n':>7}   {'low':>26}   {'high':>26}   {'ratio':>6}")
    out = {}
    for lo, hi, nm in ((2, 3, "2"), (3, 5, "3-4"), (5, 10, "5-9"), (10, 10**9, "10+")):
        s = P[(P.n_aug >= lo) & (P.n_aug < hi)]
        if len(s) < 300:
            continue
        med = s[score_col].median()
        a, b = s[s[score_col] <= med], s[s[score_col] > med]
        if len(a) < 50 or len(b) < 50:
            continue
        ra, rb = a.lab.mean(), b.lab.mean()
        out[nm] = rb / max(ra, 1e-9)
        print(f"  {nm:<10} {len(s):>7}   {100*ra:>6.2f}% (n={len(a):>5}, utr {a.utr.median():>4.0f})"
              f"   {100*rb:>6.2f}% (n={len(b):>5}, utr {b.utr.median():>4.0f})   {out[nm]:>5.2f}x")
    return out


def main():
    ap = argparse.ArgumentParser()
    # Deposit-native source, matching results_4ct_dn. Verified to cover 100% of the h5 isoforms.
    # NOT the out_scan_staging copy, which holds 474 sequences (1.1% coverage) and silently
    # produced an all-NaN test on the first run.
    ap.add_argument("--fasta", default=str(Path.home() / "claude_projects/nmd_deposit_2026/"
                    "source_data/sqanti/nmd_lungcells_corrected.fasta"))
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    orf = pd.read_csv(DATA / "orf_features.tsv", sep="\t",
                      usecols=["isoform_id", "orf_start", "kozak_score"])
    sel = pd.read_csv(DATA / "selected_orfs.tsv", sep="\t",
                      usecols=["isoform_id", "orf_start", "is_ref_cds"])
    refs = dict(zip(sel.loc[sel.is_ref_cds == 1, "isoform_id"],
                    sel.loc[sel.is_ref_cds == 1, "orf_start"]))
    print(f"orf_features: {len(orf):,} ORFs over {orf.isoform_id.nunique():,} isoforms")

    print(f"\nreading {args.fasta}")
    seqs = read_fasta(args.fasta, keep=set(orf.isoform_id.unique()))
    print(f"  {len(seqs):,} sequences; overlap with scan isoforms: "
          f"{orf.isoform_id.isin(seqs).groupby(orf.isoform_id).first().mean():.1%}")

    # The scan's orf_start must be the 1-based position of the A of ATG, or every score is shifted.
    chk = orf.sample(3000, random_state=1)
    hits = [(seqs[i][int(p) - 1:int(p) + 2] == "ATG") for i, p in zip(chk.isoform_id, chk.orf_start)
            if i in seqs]
    print(f"  orf_start points at ATG in {np.mean(hits):.2%} of {len(hits)} sampled ORFs")
    if np.mean(hits) < 0.99:
        sys.exit("STOPPING: orf_start does not reliably index the A of ATG; every score would shift.")

    if not args.skip_validation:
        print("\nVALIDATION GATE")
        validate_against_r(seqs, orf[orf.isoform_id.isin(seqs)][["isoform_id", "orf_start"]])

    print("\nscoring all ORFs with the continuous PWM ...")
    sc = np.full(len(orf), np.nan)
    for k, (i, p) in enumerate(zip(orf.isoform_id.values, orf.orf_start.values)):
        s = seqs.get(i)
        if s is not None:
            sc[k] = score_pwm(s, int(p))[0]
    orf["pwm"] = sc
    print(f"  scored {np.isfinite(sc).sum():,} of {len(orf):,}")
    print(f"  PWM score: min {np.nanmin(sc):.3f}  median {np.nanmedian(sc):.3f}  "
          f"max {np.nanmax(sc):.3f}  distinct values {len(np.unique(np.round(sc[np.isfinite(sc)],6))):,}")
    print(f"  3-level score distinct values: {orf.kozak_score.nunique()}")
    print(f"  Spearman(PWM, 3-level) = "
          f"{orf[['pwm','kozak_score']].dropna().corr(method='spearman').iloc[0,1]:.3f}")

    # ties at each isoform's maximum, the thing the 3-level score causes
    for col, nm in (("kozak_score", "3-level"), ("pwm", "continuous PWM")):
        g = orf.dropna(subset=[col]).groupby("isoform_id")[col]
        ties = (g.transform("max") == orf.loc[g.obj.index, col]).groupby(orf.isoform_id).sum()
        print(f"  ties at the isoform maximum, {nm}: mean {ties.mean():.2f}, "
              f"isoforms with >4 tied {100*(ties > 4).mean():.1f}%")

    # --- the pre-registered test -----------------------------------------------------------
    with h5py.File(DATA / "nmd_orf_data.h5", "r") as f:
        ids = np.array([x.decode() for x in f["isoform_id"][:]])
        lab = f["labels"][:]
    cls = np.array([str(x) for x in np.load(REPO / "results_interp_all/mechanism_classes.npz")["cls"]])

    orf["ref"] = orf.isoform_id.map(refs)
    u = orf.dropna(subset=["ref"])
    u = u[u.orf_start < u.ref]
    g = u.groupby("isoform_id")
    D = pd.DataFrame({"n_aug": g.size(),
                      "n_strong3": g.kozak_score.apply(lambda x: (x == 2).sum()),
                      "mean_pwm": g.pwm.mean(),
                      "max_pwm": g.pwm.max()}).reindex(ids)
    D["utr"] = pd.Series(refs).reindex(ids).values - 1
    D["lab"] = lab
    D["cls"] = cls
    P = D[(D.cls != "PTC+") & (D.cls != "Ref AUG absent") & D.utr.notna() & D.n_aug.notna()].copy()
    P["frac3"] = P.n_strong3 / P.n_aug
    print(f"\nPRE-REGISTERED TEST — population n={len(P):,}, NMD {100*P.lab.mean():.2f}%")
    a = contrast_table(P, "frac3", "BASELINE: 3-level strong-AUG fraction (the published cut)")
    b = contrast_table(P, "mean_pwm", "CONTINUOUS: mean PWM score over upstream AUGs")
    c = contrast_table(P, "max_pwm", "CONTINUOUS: max PWM score over upstream AUGs")

    print(f"\n  {'stratum':<10} {'3-level':>9} {'mean PWM':>9} {'max PWM':>9}   prediction was that "
          f"the continuous scores RISE")
    for k in a:
        print(f"  {k:<10} {a[k]:>8.2f}x {b.get(k, float('nan')):>8.2f}x "
              f"{c.get(k, float('nan')):>8.2f}x")
    rose = sum(1 for k in a if b.get(k, 0) > a[k])
    print(f"\n  mean-PWM beats the 3-level score in {rose} of {len(a)} strata")


if __name__ == "__main__":
    main()
