#!/usr/bin/env python
"""
model_tie_estimator_check.py — does the published routing estimator handle ties?

SCOPE: a claim about an INSTRUMENT, measured on model-free data. No checkpoint.

THE FINDING THIS TESTS. model_route_to_ptc.py (job 8900643, claim C14) and
model_queue_null_inbank.py both rank with

    rx = np.argsort(np.argsort(x))

which assigns ORDINAL ranks. That is not Spearman's rank when ties are present:
Spearman requires tied values to share their AVERAGE rank. argsort breaks ties by
ARRAY INDEX, and the candidate array is ordered 5'->3', so tied values are ranked
by candidate POSITION.

n_downstream_ejc is a small non-negative integer, so ties are the normal case, not
the exception. If the tie share is high, every published ejc correlation carries a
position component that nobody put there deliberately.

WHY THIS IS TESTED ON orf_start ~ ejc. Both quantities come from the data alone,
so the two estimators can be compared with no model and no checkpoint, and any
difference is the estimator and nothing else. It is also the exact quantity job
8900643 prints as its premise check ("start position ~ ejc"), so the published
number is directly comparable.

POPULATION: the tensor's candidates, restricted to the ISM subset's transcripts,
k>=4 — the closest reachable match to job 8900643's stated population. It is NOT
identical: 8900643 reads results_ism_v6/bank_interp_s100.h5, whose candidate set
is the ISM bank's, and that file is not local. So a residual difference between
this file's ordinal-rank number and the published one is POPULATION, not estimator,
and is reported as such rather than folded in.

Usage:
    python model_tie_estimator_check.py --out runlog_model_tie_estimator_check.txt
"""
import argparse, hashlib, os, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(os.environ.get(
    "NMD_TOOLS", Path.home() / "claude_projects/nmd_lung_longread_2026/tools"))))
from claim_emit import emit                               # noqa: E402

import pathlib

# ---------------------------------------------------------------- D77 / W279
# Data resolves from THIS repo, not from the parent manuscript's tree. Every
# result filed on 2026-08-02 recorded its inputs as
# NMD_orf_model_v5_4ct/... because these paths were hard-coded and the APFS
# clone (D77) landed after those runs. The numbers were unaffected -- the trees
# are byte-identical -- but a descent chain that terminates in the other
# manuscript's working tree is not a chain this repo can be cited on, which is
# what D68 exists to prevent.
#
# Derived from __file__ rather than from $HOME so it survives a move, and
# GUARDED rather than merely documented: byte-identical trees mean nothing in
# the output would reveal the regression if it recurred.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def in_repo(rel):
    """Resolve rel under this repo and refuse anything outside it."""
    p = (REPO_ROOT / rel).resolve()
    if not str(p).startswith(str(REPO_ROOT.resolve()) + "/"):
        raise SystemExit(f"REFUSED: {rel} resolves outside the repo, to {p}")
    if not p.exists():
        raise SystemExit(f"REFUSED: {rel} does not exist under {REPO_ROOT}")
    return p

TENSOR = in_repo("results_tensor_v6/nmd_tensor.h5")
SUBSET = in_repo("results_ism_v6/ism_subset.tsv")
PUBLISHED_ORDINAL = -0.334      # job 8900643, "start position ~ ejc"
MIN_K = 4


def sp_ordinal(x, y):
    """The published estimator: argsort(argsort(.)), ties broken by array index."""
    if len(x) < MIN_K:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def sp_tie_corrected(x, y):
    """Spearman proper: tied values share their average rank."""
    if len(x) < MIN_K or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(spearmanr(x, y).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = open(a.out, "w")

    def P(*s):
        print(*s); print(*s, file=log); log.flush()

    # W288. One key per line, fixed order, labelled (observed). A library this producer
    # does not import reports `absent` so the key set is stable across producers.
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

    P("\n=== self-test: the two estimators must AGREE when there are no ties ===")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(MIN_K, 30))
        x = rng.permutation(n).astype(float)          # distinct by construction
        y = rng.permutation(n).astype(float)
        worst = max(worst, abs(sp_ordinal(x, y) - sp_tie_corrected(x, y)))
    P(f"  200 tie-free pairs, max |ordinal - tie-corrected| = {worst:.2e}")
    assert worst < 1e-9, "estimators disagree without ties — the comparison below is meaningless"
    P("  SELF-TEST PASSED — any difference below is attributable to ties alone")

    f = h5py.File(TENSOR, "r")
    iso = np.array([s.decode() for s in f["isoform_id"][:]])
    off, cnt = f["offset"][:], f["count"][:]
    ejc_all = f["structural_raw"][:, 0]
    os_all = f["orf_start"][:]
    want = set(pd.read_csv(SUBSET, sep="\t")["isoform_id"].astype(str))
    keep = np.flatnonzero(np.isin(iso, list(want)))
    P(f"\n=== POPULATION ===")
    P(f"  tensor candidates, restricted to the ISM subset's transcripts, k>={MIN_K}")
    P(f"  {len(keep):,} transcripts requested; tensor holds {len(iso):,} / {f['codes'].shape[0]:,} candidates")

    A, B, tie = [], [], []
    for t in keep:
        lo, k = int(off[t]), int(cnt[t])
        if k < MIN_K:
            continue
        ej = ejc_all[lo:lo+k].astype(float)
        st = os_all[lo:lo+k].astype(float)
        A.append(sp_ordinal(st, ej))
        B.append(sp_tie_corrected(st, ej))
        tie.append(1.0 - len(np.unique(ej)) / len(ej))
    f.close()
    A = np.array(A, float); B = np.array(B, float); tie = np.array(tie, float)
    Af, Bf = A[np.isfinite(A)], B[np.isfinite(B)]

    P(f"\n=== HOW TIED IS n_downstream_ejc WITHIN A TRANSCRIPT? ===")
    P(f"  median share of candidates whose ejc value is not unique: {100*np.median(tie):.1f}%")
    P(f"  transcripts where >50% of candidates share an ejc value:  {100*(tie>0.5).mean():.1f}%")

    # D92/D94. Transcript-unit over the ISM subset at k>=MIN_K.
    def POP_(est):
        return (f"universe=U-TENSOR-ISM4999-K4; restriction=none; estimator={est}; "
                f"params=min_k={MIN_K},n_transcripts={len(keep)}")
    emit("estimator", "share of candidates with a non-unique ejc value, median within transcript",
         float(np.median(tie)), n=len(tie),
         population=POP_("median within transcript of the non-unique-ejc share"))
    emit("estimator", "orf_start ~ ejc, ORDINAL ranks (published estimator)",
         float(np.median(Af)), published=PUBLISHED_ORDINAL, n=len(Af),
         population=POP_("ORDINAL-rank Spearman within transcript, median across transcripts"))
    emit("estimator", "orf_start ~ ejc, TIE-CORRECTED Spearman",
         float(np.median(Bf)), n=len(Bf),
         population=POP_("TIE-CORRECTED Spearman within transcript, median across transcripts"))

    P(f"\n=== orf_start ~ n_downstream_ejc, SAME DATA, TWO ESTIMATORS ===")
    P(f"  ORDINAL ranks   (the published estimator)  median {np.median(Af):+.3f}   n {len(Af):,}")
    P(f"  TIE-CORRECTED Spearman                     median {np.median(Bf):+.3f}   n {len(Bf):,}")
    P(f"  published job 8900643, same quantity       median {PUBLISHED_ORDINAL:+.3f}   n 4,815")
    P(f"\n  estimator accounts for   {np.median(Bf):+.3f} -> {np.median(Af):+.3f}   "
      f"(shift {np.median(Af)-np.median(Bf):+.3f})")
    P(f"  UNEXPLAINED residual     {np.median(Af):+.3f} -> {PUBLISHED_ORDINAL:+.3f}   "
      f"(shift {PUBLISHED_ORDINAL-np.median(Af):+.3f})")
    P(f"  -> the residual is POPULATION: 8900643 reads the ISM bank's candidate set,")
    P(f"     which is not local. It is NOT attributed to the estimator here.")
    P(f"\n  DIRECTION OF THE BIAS: ordinal ranking pulls the correlation TOWARD ZERO,")
    P(f"  because tied ejc values are assigned ranks in 5'->3' order, injecting a")
    P(f"  positive position component into a genuinely negative association.")

    P("\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
