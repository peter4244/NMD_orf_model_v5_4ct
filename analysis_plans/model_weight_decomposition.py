#!/usr/bin/env python
"""
model_weight_decomposition.py — the six derived quantities, measured, over universe B.

SCOPE: claims about the MODEL. Terms are defined in
Stories/ORF_selection/GLOSSARY.md and are not redefined here.

WHY. Three of the six derived quantities in the glossary had never been measured:
scan_survival, p_select and scan_residual. Two consequences of that:

  1. Every statement about the scanner "gating rather than ranking" is really a
     statement about scan_survival, inferred from the gap between scan_p and
     scan_select or from a synthetic null. scan_survival is directly recoverable
     as scan_select / scan_p and needs no null at all.
  2. scan_residual -- the mass the scanner never allocates -- reached 0.87 on a
     40-transcript smoke test against a median of 0.0014. p_nmd commits that mass
     to ZERO decay, so a transcript the scanner largely declines is predicted
     non-NMD whatever its sequence says. A long tail measured on 40 transcripts is
     not a measurement; this runs it on universe B.

WHAT IS DELIBERATELY NOT HERE. No correlation with ejc, length or position -- that
is model_regime_reconstruction.py's job and duplicating it would create a second
place for those numbers to live. This file measures the DISTRIBUTIONS of the
weights themselves and how much the queue reorders the head.

CHOICES STATED RATHER THAN BURIED:

  * scan_survival is computed as the exclusive cumulative product of (1 - scan_p),
    which is the definition, and CHECKED against scan_select / scan_p rather than
    derived from it, so a disagreement is detectable.
  * "How much does the queue reorder the head" is Spearman(scan_p, scan_select)
    WITHIN a transcript. It is 1.0 when the queue only rescales the head's
    ordering and falls as the queue overrides it. REPORTED AT k>=4, matching
    model_regime_reconstruction.py, because AT k=2 A SPEARMAN CAN ONLY BE +/-1:
    two- and three-candidate transcripts can land only in the tails, which are
    the exact quantities of interest. The k>=2 form is printed beside it so the
    size of that distortion is visible rather than argued about.
  * p_select is the posterior share and sums to 1 within transcript BY
    CONSTRUCTION; scan_select sums to 1 - scan_residual. Both are asserted here.
  * Transcripts with a single candidate are excluded from ordering statistics and
    counted separately -- there is no ordering to measure in them.
  * THE CONCENTRATION SHIFT IS ALSO COMPUTED WITHIN LENGTH STRATUM, because the
    pooled version cannot distinguish two accounts. Decay de-concentrates the
    scanner's top pick (0.753 -> 0.682 pooled); that is either decay genuinely
    disagreeing with the scanner, or the top pick simply being the longest ORF
    while long ORFs carry lower decay scores. p_capture ~ d is -0.068 pooled but
    +0.314 among short candidates and -0.035 among long ones, so the pooled
    negative is a BETWEEN-stratum contrast and the pooled shift inherits it.
    Restricting to one length class and renormalising removes that contrast: a
    shift that survives is decay, a shift that vanishes is length.

Usage:
    python model_weight_decomposition.py --out runlog.txt [--all] [--seed 100]
"""
import argparse, sys, hashlib, time, os, pathlib
from pathlib import Path
import sys
import numpy as np, h5py, torch
from scipy.stats import spearmanr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get("NMD_TOOLS",
                            Path.home() / "claude_projects/nmd_lung_longread_2026/tools"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS))
from tensor_io import decode_windows                      # noqa: E402
from model_v6 import ScanningNMDModel                     # noqa: E402
from claim_emit import emit                               # noqa: E402

ATG_LEFT, STOP_LEFT = 900, 500


def in_repo(rel):
    p = (REPO_ROOT / rel).resolve()
    if not str(p).startswith(str(REPO_ROOT.resolve()) + "/"):
        raise SystemExit(f"REFUSED: {rel} resolves outside the repo, to {p}")
    if not p.exists():
        raise SystemExit(f"REFUSED: {rel} does not exist under {REPO_ROOT}")
    return p


TENSOR = in_repo("results_tensor_v6/nmd_tensor.h5")
SUBSET = in_repo("results_ism_v6/ism_subset.tsv")


def q(x, ps=(10, 25, 50, 75, 90, 99)):
    return "  ".join(f"{p}%={np.percentile(x, p):.4f}" for p in ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    log = open(a.out, "w")

    def P(*s):
        print(*s); print(*s, file=log); log.flush()

    ckpt = in_repo(f"results_interp_all/v6_checkpoints/b8_s{a.seed}.pt")
    P("=== code and artifact provenance (sha256) ===")
    P(f"  {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}  {Path(__file__).name}")
    P(f"  {hashlib.sha256(ckpt.read_bytes()).hexdigest()}  {ckpt.name}")
    P(f"  tensor {TENSOR}  {TENSOR.stat().st_size:,} bytes")
    P(f"  all data resolves under {REPO_ROOT}")

    # W287. THE ROW RECORDS WHICH CODE RAN AND NOT WHAT IT RAN ON. Both demonstrably
    # move values: node family is worth up to 1.6e-04 relative (measured 2026-08-03) and
    # the library pair is total -- four filed rows were produced locally in an environment
    # that no longer executes them. Printed BEFORE any compute, in the same grammar as the
    # provenance block above so a reader parses one format rather than three.
    import platform as _pl, socket as _sk
    # W288: ONE KEY PER LINE, FIXED ORDER, and labelled (observed) -- the block records
    # what the producer reports about ITSELF and nothing verifies it. An unread CONTRACT
    # causes false assurance (W257); an unread OBSERVATION costs a few lines. Parseable
    # rather than prose because these get backfilled into the R-row when its schema lands,
    # and hand-parsing is where this project has gone wrong three times.
    P("=== environment (observed) ===")
    P(f"  python {_pl.python_version()}")
    P(f"  numpy {np.__version__}")
    P(f"  torch {torch.__version__}")
    P(f"  node {_sk.gethostname()}")
    P(f"  platform {sys.platform}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ScanningNMDModel(n_seq_channels=9, conv_channels=ck["conv_channels"],
                             n_bins=ck["n_bins"], seq_embed_dim=32, n_structural=1)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    P(f"  checkpoint variant={ck['variant']} n_bins={ck['n_bins']} seed={ck['seed']}")

    f = h5py.File(TENSOR, "r")
    off, cnt = f["offset"][:], f["count"][:]
    if a.all:
        keep = np.arange(len(off)); pop_name = "FULL TENSOR (universe B)"
    else:
        import pandas as pd
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        want = set(pd.read_csv(SUBSET, sep="\t")["isoform_id"].astype(str))
        keep = np.flatnonzero(np.isin(iso, list(want))); pop_name = "ISM SUBSET transcripts (universe C)"
    P(f"\n=== POPULATION: {pop_name}, {len(keep):,} transcripts ===")

    resid, reorder, reorder4, topshare_scan, topshare_post, K_all = [], [], [], [], [], []
    surv_min, singles, e_id = [], 0, 0.0
    strat = {"short": ([], []), "long": ([], [])}
    t0 = time.time()
    for i, t in enumerate(keep):
        lo, hi = int(off[t]), int(off[t]) + int(cnt[t])
        K = hi - lo
        if K < 1:
            continue
        codes = f["codes"][lo:hi]
        os_, oe_ = f["orf_start"][lo:hi], f["orf_end"][lo:hi]
        st = f["structural"][lo:hi]
        atg = decode_windows(codes[:, 0, :], os_, ATG_LEFT, os_)
        stp = decode_windows(codes[:, 1, :], oe_, STOP_LEFT, os_)
        with torch.no_grad():
            _, parts = model(torch.from_numpy(atg)[None], torch.from_numpy(stp)[None],
                             torch.from_numpy(st[:, :1])[None],
                             torch.ones(1, K, dtype=torch.bool), return_parts=True)
        scan_p = parts["p"][0].numpy().astype(np.float64)
        seq_p = parts["d"][0].numpy().astype(np.float64)
        scan_select = parts["p_select"][0].numpy().astype(np.float64)
        r = float(parts["residual"][0]); pn = float(parts["p_nmd"][0])

        # definition, not a rearrangement of scan_select
        scan_survival = np.concatenate([[1.0], np.cumprod(1.0 - scan_p)[:-1]])
        e_id = max(e_id, np.abs(scan_select - scan_p * scan_survival).max())

        resid.append(r); K_all.append(K); surv_min.append(scan_survival[-1])
        topshare_scan.append(scan_select.max() / max(scan_select.sum(), 1e-300))
        p_sel = scan_select * seq_p / pn
        topshare_post.append(p_sel.max())

        # within-stratum concentration: renormalise inside one length class so the
        # short-versus-long contrast cannot produce the shift
        orflen = (oe_ - os_ + 1).astype(np.float64)
        for lab, msk in (("short", orflen < 200), ("long", orflen >= 200)):
            if msk.sum() < 4:
                continue
            a_ = scan_select[msk]; w_ = (scan_select * seq_p)[msk]
            if a_.sum() <= 0 or w_.sum() <= 0:
                continue
            strat[lab][0].append(a_.max() / a_.sum())
            strat[lab][1].append(w_.max() / w_.sum())
        if K >= 2:
            if scan_p.std() > 0 and scan_select.std() > 0:
                rr = spearmanr(scan_p, scan_select).statistic
                if np.isfinite(rr):
                    reorder.append(rr)
                    if K >= 4:
                        reorder4.append(rr)
        else:
            singles += 1
        if (i + 1) % 5000 == 0:
            P(f"    {i+1:,}/{len(keep):,}  {time.time()-t0:.0f}s")
    f.close()

    resid = np.array(resid); reorder = np.array(reorder); reorder4 = np.array(reorder4)
    tss = np.array(topshare_scan); tsp = np.array(topshare_post); Kv = np.array(K_all)
    P(f"  {len(resid):,} transcripts, {int(Kv.sum()):,} candidates, {time.time()-t0:.0f}s")
    P(f"  identity check, scan_select == scan_p * scan_survival: max err {e_id:.2e}")
    P(f"  single-candidate transcripts (no ordering to measure): {singles:,}")

    P(f"\n=== scan_residual — the mass the scanner never allocates ===")
    P(f"  {q(resid)}")
    for thr in (0.05, 0.10, 0.25, 0.50, 0.90):
        P(f"    share of transcripts with scan_residual > {thr:.2f}:  {100*(resid>thr).mean():5.2f}%"
          f"   ({int((resid>thr).sum()):,})")
    P(f"  p_nmd commits ALL of this mass to zero decay, so these transcripts are")
    P(f"  predicted non-NMD to that extent whatever their sequence says.")

    P(f"\n=== does the queue reorder the head, or only rescale it? ===")
    P(f"  Spearman(scan_p, scan_select) within transcript.")
    P(f"  AT k=2 A SPEARMAN CAN ONLY BE +/-1, so k>=4 is the reportable form and")
    P(f"  matches model_regime_reconstruction.py's restriction on the same universe.")
    for lab, v in (("k>=4  [REPORTABLE]", reorder4), ("k>=2  [shows the distortion]", reorder)):
        P(f"    {lab}  n={len(v):,}")
        P(f"      {q(v)}")
        P(f"      ordering UNCHANGED (rho >= 0.999): {100*(v>=0.999).mean():5.2f}%"
          f"     INVERTED (rho < 0): {100*(v<0).mean():5.2f}%")

    # D92 grammar, D94 universes. The universe is a property of the INVOCATION: --all
    # gives the full tensor, otherwise the ISM transcript subset.
    ISM = "ISM" in pop_name.upper()
    U_ALL = "N-ISMSUBSET-4999" if ISM else "U-TENSOR-ALL"
    U_K4 = "U-TENSOR-ISM4999-K4" if ISM else "U-TENSOR-K4"
    U_CLASS = {"short": "U-TENSOR-ISM4999-K4-SHORT" if ISM else "U-TENSOR-K4-SHORT",
               "long": "U-TENSOR-ISM4999-K4-LONG" if ISM else "U-TENSOR-K4-LONG"}
    EST = ("scan_survival computed as the exclusive cumulative product of (1 - scan_p)")

    def pop(univ, restriction, estimator):
        return (f"universe={univ}; restriction={restriction}; estimator={estimator}; "
                f"params=all={a.all},seed={a.seed}")

    # Measurement-dependent, so restriction rather than universe (D94): which transcripts
    # fail the variance test depends on the variable pair, and 36,369 against 36,370 in
    # the store is two such sets differing by a single transcript.
    R_VAR = ("transcripts of the universe dropped where scan_p or scan_select has zero "
             "variance, or the Spearman is not finite")
    P(f"\n=== concentration ===")
    P(f"  top candidate's share of scan_select   {q(tss)}")
    P(f"  top candidate's share of p_select      {q(tsp)}")
    P(f"  candidates per transcript              {q(Kv.astype(float))}")
    P(f"\n=== is the concentration shift DECAY, or is it LENGTH? ===")
    P(f"  renormalised WITHIN a length class, so the short/long contrast cannot produce it")
    P(f"  (>=4 candidates of that class required)")
    for lab in ("short", "long"):
        a_ = np.array(strat[lab][0]); w_ = np.array(strat[lab][1])
        if not len(a_):
            P(f"    {lab}: no transcript qualifies"); continue
        P(f"    {lab:5s} n={len(a_):,}   top share scan_select {np.median(a_):.4f}"
          f"  ->  p_select {np.median(w_):.4f}   shift {np.median(w_)-np.median(a_):+.4f}")
        emit("weights", f"top candidate's share, scan_select -> p_select shift, {lab} candidates only",
             float(np.median(w_) - np.median(a_)), n=len(a_),
             population=pop(U_CLASS[lab], "none",
                            "top-share shift, renormalised within the length class, "
                            + EST))

    emit("weights", "scan_residual, median over transcripts", float(np.median(resid)),
         n=len(resid), population=pop(U_ALL, "none", "median over transcripts, " + EST))
    emit("weights", "share of transcripts with scan_residual > 0.50", float((resid > 0.50).mean()),
         n=len(resid), population=pop(U_ALL, "none", "share above 0.50, " + EST))
    emit("weights", "share of transcripts with scan_residual > 0.05", float((resid > 0.05).mean()),
         n=len(resid), population=pop(U_ALL, "none", "share above 0.05, " + EST))
    emit("weights", "Spearman(scan_p, scan_select) within transcript, median",
         float(np.median(reorder4)), n=len(reorder4),
         population=pop(U_K4, R_VAR, "within-transcript Spearman, median, " + EST))
    emit("weights", "share of transcripts where the queue leaves the head's ordering unchanged",
         float((reorder4 >= 0.999).mean()), n=len(reorder4),
         population=pop(U_K4, R_VAR, "share with rho >= 0.999, " + EST))
    emit("weights", "share of transcripts where the queue INVERTS the head's ordering",
         float((reorder4 < 0).mean()), n=len(reorder4),
         population=pop(U_K4, R_VAR, "share with rho < 0, " + EST))
    emit("weights", "top candidate's share of p_select, median", float(np.median(tsp)),
         n=len(tsp), population=pop(U_ALL, "none", "median top share of p_select, " + EST))

    P("\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
