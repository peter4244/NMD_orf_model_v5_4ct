#!/usr/bin/env python
"""
model_scanning_geometry.py — the two geometric predictions the scanning synthesis makes.

SCOPE: facts about the CANDIDATE GEOMETRY. No checkpoint is loaded, no model runs.
Terms are in Stories/ORF_selection/GLOSSARY.md.

THE SYNTHESIS BEING TESTED. scan_select_k = scan_p_k x PROD(1 - scan_p_j) over
upstream candidates, so the LEVEL of scan_p decides whether the queue preserves the
scanner's ranking or overrides it. Where scan_p is small the survival term stays
near 1 and scan_select tracks scan_p; where scan_p is large, survival collapses and
whichever candidate comes FIRST takes nearly everything. Measured levels are 0.015
among candidates under 200 nt and 0.850 among longer ones (R-sm-md-0008), and the
concordance between scan_p and scan_select is +0.713 and +0.125 in those two
strata (R-sm-md-0009). That is leaky scanning past short upstream frames and
commitment at the first long one.

TWO PREDICTIONS FOLLOW, AND BOTH ARE PURE GEOMETRY. They are stated here with the
values that would REFUTE them, before the numbers are printed.

  P1. scan_select ~ ORF length is +0.900 among long candidates (R-sm-md-0009). Under
      this account that is POSITION dominance, not a length preference. It can only
      appear as length if longer candidates systematically start earlier WITHIN the
      long stratum. Pooled, orf_start ~ length is only -0.151, far too weak.
      PREDICTION: strongly negative within the long stratum, near -0.9, since a
      quantity driven by -position would then correlate with length at about its
      negation. REFUTED IF: weak, or positive.

  P2. scan_p ~ downstream junction count is +0.204 among short candidates and -0.433
      among long ones (R-sm-md-0009). The scanner cannot see junction counts -- it
      reads the start window only -- so under a length account those signs must be
      inherited from length.
      PREDICTION: ORF length ~ junction count carries OPPOSITE signs in the two
      strata, positive among short and negative among long. REFUTED IF: same sign in
      both.

WHY THIS FILE IS MODEL-FREE ON PURPOSE. Both predictions are about the arrangement
of candidates within a transcript, not about anything the network computes. Running
them without a checkpoint means a confirmation cannot be an artifact of the model,
and it makes them cheap enough to be rerun whenever the pool changes.

Usage:
    python model_scanning_geometry.py --out runlog.txt
"""
import argparse, hashlib, os, sys, pathlib
from pathlib import Path
import numpy as np, h5py
from scipy.stats import spearmanr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(os.environ.get(
    "NMD_TOOLS", Path.home() / "claude_projects/nmd_lung_longread_2026/tools"))))
from claim_emit import emit                               # noqa: E402

BOUNDARY, MIN_K = 200, 4


def in_repo(rel):
    p = (REPO_ROOT / rel).resolve()
    if not str(p).startswith(str(REPO_ROOT.resolve()) + "/"):
        raise SystemExit(f"REFUSED: {rel} resolves outside the repo, to {p}")
    if not p.exists():
        raise SystemExit(f"REFUSED: {rel} does not exist under {REPO_ROOT}")
    return p


TENSOR = in_repo("results_tensor_v6/nmd_tensor.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = open(a.out, "w")

    def P(*s):
        print(*s); print(*s, file=log); log.flush()

    # W288. One key per line, fixed order, labelled (observed): the block records what
    # the producer reports about ITSELF and nothing verifies it. Printed before any
    # compute. A library this producer does not import is reported `absent` rather than
    # omitted, so the key set is stable across producers and the backfill can parse one
    # shape rather than several.
    import platform as _pl, socket as _sk, sys as _sys
    P("=== environment (observed) ===")
    P(f"  python {_pl.python_version()}")
    for _m in ("numpy", "torch"):
        try:
            P(f"  {_m} {__import__(_m).__version__}")
        except Exception:
            P(f"  {_m} absent")
    P(f"  node {_sk.gethostname()}")
    P(f"  platform {_sys.platform}")

    P("=== code and input provenance (sha256) ===")
    P(f"  {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}  {Path(__file__).name}")
    P(f"  tensor {TENSOR}  {TENSOR.stat().st_size:,} bytes")
    P("  NO CHECKPOINT IS LOADED — both predictions are about candidate geometry")

    f = h5py.File(TENSOR, "r")
    off, cnt = f["offset"][:], f["count"][:]
    os_, oe_ = f["orf_start"][:], f["orf_end"][:]
    ejc = f["structural_raw"][:, 0]
    f.close()
    L = (oe_ - os_ + 1).astype(float)

    R = {k: [] for k in ("p1_all", "p1_s", "p1_l", "p2_all", "p2_s", "p2_l")}
    for t in range(len(off)):
        lo, hi = int(off[t]), int(off[t]) + int(cnt[t])
        if hi - lo < MIN_K:
            continue
        st, ln, ej = os_[lo:hi].astype(float), L[lo:hi], ejc[lo:hi].astype(float)
        for tag, m in (("all", np.ones(len(ln), bool)),
                       ("s", ln < BOUNDARY), ("l", ln >= BOUNDARY)):
            if m.sum() < MIN_K:
                continue
            x, y, z = st[m], ln[m], ej[m]
            if x.std() > 0 and y.std() > 0:
                R[f"p1_{tag}"].append(spearmanr(x, y).statistic)
            if y.std() > 0 and z.std() > 0:
                R[f"p2_{tag}"].append(spearmanr(y, z).statistic)

    def med(k):
        v = np.array([x for x in R[k] if np.isfinite(x)])
        return float(np.median(v)), len(v)

    # D92/D94. Transcript-unit, k>=MIN_K within the stratum -- the parent x class sets,
    # NOT the candidate-unit U-TENSOR-ORF-* pair which partitions the tensor.
    UNIV = {"all candidates": "U-TENSOR-K4", "short < 200 nt": "U-TENSOR-K4-SHORT",
            "long >= 200 nt": "U-TENSOR-K4-LONG"}
    # Measurement-dependent, so restriction (D94): med() keeps only finite values, and
    # which transcripts those are depends on the variable pair correlated.
    RESTR = ("transcripts of the universe dropped where the within-transcript Spearman "
             "is not finite for the variable pair measured")

    def POP_(lab, est):
        return (f"universe={UNIV[lab]}; restriction={RESTR}; estimator={est}; "
                f"params=min_k={MIN_K},boundary_nt=200,no_model=true")

    P(f"\n=== P1  orf_start ~ ORF length — do longer candidates start earlier? ===")
    for tag, lab in (("all", "all candidates"), ("s", "short < 200 nt"), ("l", "long >= 200 nt")):
        m, n = med(f"p1_{tag}")
        P(f"    {lab:18s} {m:+.3f}   n {n:,}")
        emit("geometry", f"orf_start ~ ORF length, {lab}", m, n=n,
             population=POP_(lab, "within-transcript tie-corrected Spearman, median across transcripts"))
    ml, _ = med("p1_l")
    P(f"  PREDICTION: strongly negative in the long stratum.  measured {ml:+.3f}  -> "
      f"{'CONFIRMED' if ml < -0.5 else 'REFUTED'}")
    P(f"  If scan_select there is driven by -position, scan_select ~ length should be")
    P(f"  about {-ml:+.3f}; R-sm-md-0009 measured +0.900.")

    P(f"\n=== P2  ORF length ~ downstream junction count ===")
    for tag, lab in (("all", "all candidates"), ("s", "short < 200 nt"), ("l", "long >= 200 nt")):
        m, n = med(f"p2_{tag}")
        P(f"    {lab:18s} {m:+.3f}   n {n:,}")
        emit("geometry", f"ORF length ~ downstream junction count, {lab}", m, n=n,
             population=POP_(lab, "within-transcript tie-corrected Spearman, median across transcripts"))
    ms, mlq = med("p2_s")[0], med("p2_l")[0]
    P(f"  PREDICTION: opposite signs, positive short and negative long.")
    P(f"  measured {ms:+.3f} and {mlq:+.3f}  -> "
      f"{'CONFIRMED' if ms > 0 > mlq else 'REFUTED'}")
    if not (ms > 0 > mlq):
        P(f"  So the scanner's POSITIVE association with junction count among short")
        P(f"  candidates (+0.204) is NOT inherited from length: routing it through")
        P(f"  length predicts about {0.415*ms:+.3f}. A residual preference remains,")
        P(f"  and the scanner cannot see junction counts, so it is reading something")
        P(f"  in the start window that tracks them. Position is the obvious candidate")
        P(f"  and is NOT tested here — it needs length and position held jointly.")

    P("\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
