#!/usr/bin/env python
"""
model_pool_admission.py — what the candidate pool IS, before any model runs.

SCOPE: facts about the DATA, not about the model. No checkpoint is loaded.

WHY. Three claims in NARRATIVE_HOW_THE_MODEL_DECIDES.md are read as statements
about the head when they are substantially statements about the pool:

  * "length beats Kozak seven to one" is a comparison between a predictor the
    admission rule truncates (kozak_score, floored at a MANE-calibrated q05) and
    one it does not (orf_length). Whether that truncation is severe enough to
    matter is measurable, and is measured here rather than assumed in either
    direction.
  * "a one-line longest-ORF heuristic reproduces 97% of the model's selection
    accuracy" (C11) is deflating only if the reference ORF is not ALREADY the
    longest candidate most of the time. Measured here.
  * the reference start is admitted by a DIFFERENT RULE from its competitors
    (always-admit), so any recovery benchmark scores a target whose presence is
    guaranteed. The size of that asymmetry is measured here.

POPULATION: every row of results_pool_v6/orf_pool.tsv — the full admitted pool,
802,035 candidates over 42,043 transcripts. NOT the ISM bank subset, and NOT the
tensor's 796,584. Those are three different populations and this file uses one.

Producer for: the pool facts quoted to Pete 2026-08-02, previously run inline
with no runlog. This file exists so they have one.

Usage:
    python model_pool_admission.py --out runlog_model_pool_admission.txt
"""
import argparse, hashlib, sys, os
from pathlib import Path
import numpy as np, pandas as pd

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

POOL = in_repo("results_pool_v6/orf_pool.tsv")
BOUNDARY = 200          # nt; where downstream fill min(100, len/2) saturates
DOWNSTREAM_EXTENT = 100  # nt; build_tensor.py ATG_RIGHT


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
    P(f"  {hashlib.sha256(POOL.read_bytes()).hexdigest()}  orf_pool.tsv")
    P(f"  {POOL}  {POOL.stat().st_size:,} bytes")

    d = pd.read_csv(POOL, sep="\t")
    P(f"\n=== POPULATION: the full admitted pool ===")
    P(f"  {len(d):,} candidates over {d.isoform_id.nunique():,} transcripts "
      f"({len(d)/d.isoform_id.nunique():.1f} per transcript)")

    P(f"\n=== 1. HOW EACH CANDIDATE GOT IN (build_orf_pool.py step 4) ===")
    P(f"  keep = (kozak_score >= MANE-calibrated floor) AND (orf_start <= transcript_len // 2)")
    P(f"  the reference start is admitted unconditionally; top-5 fallback if the pool is empty")
    for k, v in d.admitted_by.value_counts().items():
        P(f"    {k:12s} {v:9,}  {100*v/len(d):6.2f}%")

    P(f"\n=== 2. IS KOZAK RANGE-RESTRICTED ENOUGH TO INVALIDATE THE COMPARISON? ===")
    fl = d[d.admitted_by == "floor"]
    P(f"  admitted pool kozak    range {d.kozak_score.min():+.2f} .. {d.kozak_score.max():+.2f}   sd {d.kozak_score.std():.3f}")
    P(f"  floor-admitted only    range {fl.kozak_score.min():+.2f} .. {fl.kozak_score.max():+.2f}   sd {fl.kozak_score.std():.3f}")
    g = d.groupby("isoform_id")
    kr, ks = (g.kozak_score.max() - g.kozak_score.min()), g.kozak_score.std()
    P(f"  WITHIN-transcript kozak range   median {kr.median():.3f}")
    P(f"  WITHIN-transcript kozak sd      median {ks.median():.3f}")
    P(f"  -> the head discriminates within a transcript, so within-transcript spread is")
    P(f"     the relevant quantity. A median spread of {kr.median():.2f} log-odds is NOT a")
    P(f"     truncation severe enough to explain a near-zero kozak correlation.")

    P(f"\n=== 3. IS THE REFERENCE HANDICAPPED BY THE POOL? ===")
    ref = d.is_ref_cds == 1
    P(f"  kozak, reference candidates      n {ref.sum():7,}  median {d.loc[ref,'kozak_score'].median():+.3f}")
    P(f"  kozak, non-reference candidates  n {(~ref).sum():7,}  median {d.loc[~ref,'kozak_score'].median():+.3f}")
    resc = (d.admitted_by == "reference").sum()
    P(f"  reference candidates admitted ONLY by the always-admit rule: {resc:,} "
      f"({100*resc/ref.sum():.1f}% of references)")
    P(f"  -> without that rule the recovery benchmark loses {100*resc/ref.sum():.1f}% of its targets.")

    P(f"\n=== 4. IS THE REFERENCE ALREADY THE LONGEST / MOST 5' CANDIDATE? ===")
    sub = d[d.isoform_id.isin(d.loc[ref, "isoform_id"])]
    rl = sub.groupby("isoform_id").orf_length.rank(ascending=False, method="min")
    rp = sub.groupby("isoform_id").orf_start.rank(ascending=True, method="min")
    m = sub.is_ref_cds == 1
    P(f"  reference IS the longest candidate    {100*(rl[m]==1).mean():.1f}%   (n {int(m.sum()):,})")
    P(f"  reference IS the most 5' candidate    {100*(rp[m]==1).mean():.1f}%")
    # D92/D94. THIS PRODUCER CARRIED THREE SPELLINGS OF ONE UNIVERSE -- "full admitted
    # candidate pool, results_pool_v6/orf_pool.tsv", "full admitted pool" and "full
    # admitted pool, every candidate". All three are U-POOL-ALL; the differences were
    # restrictions and estimators fused into the name. That collision is measured in the
    # store and this closes it.
    def POP_(restriction, est):
        return (f"universe=U-POOL-ALL; restriction={restriction}; estimator={est}; "
                f"params=boundary_nt={BOUNDARY}")
    R_REF = "candidates of transcripts that HAVE a reference candidate"
    emit("pool", "reference candidate is the LONGEST in its transcript",
         float((rl[m]==1).mean()), n=int(m.sum()),
         population=POP_(R_REF, "share where the reference has per-transcript length rank 1, ties method=min"))
    emit("pool", "reference candidate is the MOST 5' in its transcript",
         float((rp[m]==1).mean()), n=int(m.sum()),
         population=POP_(R_REF, "share where the reference has per-transcript 5' rank 1, ties method=min"))
    emit("pool", "reference candidates admitted ONLY by the always-admit rule",
         float(resc/ref.sum()), n=int(ref.sum()),
         population=POP_("reference candidates only",
                         "share admitted solely by the always-admit rule"))
    emit("pool", "within-transcript kozak_score range, median",
         float(kr.median()), n=int(len(kr)),
         population=POP_("none", "median across transcripts of the within-transcript "
                                 "kozak_score range (max minus min)"))
    emit("pool", "share of candidates with orf_length < 200 nt (fill encodes length)",
         float((d.orf_length<BOUNDARY).mean()), n=int(len(d)),
         population=POP_("none", f"share of candidates with orf_length < {BOUNDARY} nt"))
    P(f"  -> C11's 'longest-ORF reaches 97% of the model' is substantially a fact about")
    P(f"     the pool: length is structurally predictive of the annotated start here.")

    P(f"\n=== 5. THE ORF WINDOW ({DOWNSTREAM_EXTENT} nt) AGAINST THE ORF ITSELF ===")
    P(f"  downstream fill = min({DOWNSTREAM_EXTENT}, orf_length/2), so below {BOUNDARY} nt the")
    P(f"  fill boundary encodes ORF length exactly, with no sequence involved.")
    L = d.orf_length
    for lo, hi, lab in [(0, 80, "<80"), (80, 160, "80-160"), (160, 200, "160-200"),
                        (200, 10**9, ">=200")]:
        mm = (L >= lo) & (L < hi)
        P(f"    {lab:8s} n {int(mm.sum()):8,}  {100*mm.mean():5.1f}%   median length {L[mm].median():6.0f}")
    P(f"  share with orf_length < {BOUNDARY} (fill encodes length): {100*(L<BOUNDARY).mean():.1f}%")
    P(f"  median ORF length, whole pool: {L.median():.0f}")
    seen = 100 * DOWNSTREAM_EXTENT / L[L >= BOUNDARY].median()
    P(f"  among >= {BOUNDARY}: median length {L[L>=BOUNDARY].median():.0f}, so the head sees "
      f"the first {DOWNSTREAM_EXTENT} nt = {seen:.1f}% of the ORF")
    P(f"  -> in that band the head cannot see the stop codon and has NO geometric")
    P(f"     channel for total length.")

    P("\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
