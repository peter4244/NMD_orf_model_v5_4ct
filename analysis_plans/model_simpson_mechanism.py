#!/usr/bin/env python
"""
model_simpson_mechanism.py — why the scanner/sequence correlation reverses sign
when candidates are stratified by ORF length.

SCOPE: claims about the MODEL. Terms are in Stories/ORF_selection/GLOSSARY.md.

THE THING BEING EXPLAINED, already measured elsewhere and NOT recomputed here:
scan_p against seq_p_total is negative pooled and positive among short candidates
(R-sm-md-0005), and the sequence head's effect on concentration reverses the same
way (R-sm-md-0007). Both are Simpson reversals.

THE EXPLANATION HAS NEVER BEEN MEASURED. The account offered in conversation was
"long candidates draw high scanner scores while carrying low decay scores, so
pooling manufactures a negative". That is a claim about LEVELS in the two strata,
and about seq_p_total against length, and neither exists. This file measures them,
so the mechanism sentence in the story is supported rather than asserted.

WHAT IS MEASURED, and why each is needed for the sentence:

  1. LEVELS. Median scan_p and seq_p_total among short and long candidates. A
     Simpson reversal driven by composition requires the two strata to sit at
     different levels on BOTH axes, and in opposing directions. If they do not,
     the composition account is wrong whatever the correlations do.
  2. seq_p_total ~ ORF length, within transcript. The partner of the known
     scan_p ~ ORF length = +0.750. The composition account predicts it is
     NEGATIVE; if it is positive the account fails.
  3. The pooled correlation recomputed on candidates of ONE class at a time is
     already in R-sm-md-0005 and is not repeated.

TWO STATISTICS, DELIBERATELY OF DIFFERENT KINDS, and the file keeps them apart.
A LEVEL is a median over candidates, pooled across transcripts — it answers "are
long candidates scored higher". A within-transcript CORRELATION answers "does the
model rank them higher inside one transcript". Only the first can carry the
composition account, because composition is a between-transcript, between-class
fact. Reporting them in one table would invite exactly the conflation this file
exists to resolve.

Usage:
    python model_simpson_mechanism.py --out runlog.txt [--all] [--seed 100]
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
BOUNDARY = 200
MIN_K = 4


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

    f = h5py.File(TENSOR, "r")
    off, cnt = f["offset"][:], f["count"][:]
    keep = np.arange(len(off)) if a.all else np.arange(min(5000, len(off)))
    pop_name = "FULL TENSOR (universe B)" if a.all else "first 5,000 transcripts of B"
    P(f"\n=== POPULATION: {pop_name}, {len(keep):,} transcripts ===")

    SP, SQ, LN = [], [], []
    corr_len = []
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
        sp = parts["p"][0].numpy().astype(np.float64)
        sq = parts["d"][0].numpy().astype(np.float64)
        ln = (oe_ - os_ + 1).astype(np.float64)
        SP.append(sp); SQ.append(sq); LN.append(ln)
        if K >= MIN_K and sq.std() > 0 and ln.std() > 0:
            r = spearmanr(sq, ln).statistic
            if np.isfinite(r):
                corr_len.append(r)
        if (i + 1) % 10000 == 0:
            P(f"    {i+1:,}/{len(keep):,}  {time.time()-t0:.0f}s")
    f.close()

    sp = np.concatenate(SP); sq = np.concatenate(SQ); ln = np.concatenate(LN)
    short, long_ = ln < BOUNDARY, ln >= BOUNDARY
    P(f"  {len(sp):,} candidates, {time.time()-t0:.0f}s")

    P(f"\n=== 1. LEVELS — medians over CANDIDATES, pooled across transcripts ===")
    P(f"  This is the statistic the composition account needs. It is NOT a")
    P(f"  within-transcript correlation and must not be read as one.")
    P(f"  {'':22s}{'short <200':>14s}{'long >=200':>14s}{'difference':>14s}")
    out = {}
    for nm, v in (("scan_p", sp), ("seq_p_total", sq)):
        ms, ml = float(np.median(v[short])), float(np.median(v[long_]))
        out[nm] = (ms, ml)
        P(f"  {nm:22s}{ms:14.4f}{ml:14.4f}{ml-ms:+14.4f}")
    P(f"  {'n candidates':22s}{int(short.sum()):14,}{int(long_.sum()):14,}")

    dp = out["scan_p"][1] - out["scan_p"][0]
    dq = out["seq_p_total"][1] - out["seq_p_total"][0]
    P(f"\n  The composition account requires these two differences to have OPPOSING")
    P(f"  signs. scan_p {dp:+.4f}, seq_p_total {dq:+.4f}  ->  "
      f"{'OPPOSING, account holds' if dp*dq < 0 else 'SAME SIGN, account FAILS'}")

    P(f"\n=== 2. seq_p_total ~ ORF length, within transcript, k>={MIN_K} ===")
    cl = np.array(corr_len)
    P(f"  median {np.median(cl):+.4f}   n {len(cl):,}")
    P(f"  the partner of scan_p ~ ORF length = +0.750 (R-sm-md-0005, same universe)")
    P(f"  the composition account predicts NEGATIVE here; measured "
      f"{'NEGATIVE, consistent' if np.median(cl) < 0 else 'POSITIVE, account fails'}")

    # D92 grammar, D94 universes. THE DEFECT THIS REPLACES: short and long both emitted
    # POP_L, which says "split at 200 nt" and never says WHICH SIDE -- two disjoint
    # populations under one identical string, distinguished only by the quantity name and
    # by n. A bit-identical re-run would have carried that forward faithfully.
    #
    # The two level universes are CANDIDATE-unit and pooled: 550,350 + 246,234 = 796,584,
    # partitioning U-TENSOR-ALL exactly. The correlation universe is TRANSCRIPT-unit and
    # k>=4, which is a different unit over a different parent -- U-TENSOR-K4 at 40,245.
    # Same words, different unit; see the `unit` column in the registry.
    if not a.all:
        # first-5,000-of-B is a positional slice nothing has filed over, so it has no id.
        # Say so rather than borrow the full-tensor ids, which would file this run's
        # numbers against a universe it was not computed over.
        UNIV_S = UNIV_L = UNIV_C = "UNREGISTERED:first-5000-of-B, positional slice"
    else:
        UNIV_S, UNIV_L, UNIV_C = "U-TENSOR-ORF-SHORT", "U-TENSOR-ORF-LONG", "U-TENSOR-K4"

    def pop(univ, restriction, estimator):
        return (f"universe={univ}; restriction={restriction}; estimator={estimator}; "
                f"params=boundary_nt={BOUNDARY},min_k={MIN_K},all={a.all},seed={a.seed}")

    LEVEL = "median over candidates pooled across transcripts"
    # The guard is measurement-dependent and therefore restriction, not universe (D94):
    # a transcript flat in either variable never reaches the statistic, and WHICH
    # transcripts those are depends on the variable pair being correlated.
    CORR_R = ("transcripts of the universe dropped where sq.std()==0 or ln.std()==0 or "
              "the Spearman is not finite; the dropped set depends on the variable pair")
    for nm in ("scan_p", "seq_p_total"):
        emit("simpson", f"{nm}, median over short (<200 nt) candidates", out[nm][0],
             n=int(short.sum()), population=pop(UNIV_S, "none", LEVEL))
        emit("simpson", f"{nm}, median over long (>=200 nt) candidates", out[nm][1],
             n=int(long_.sum()), population=pop(UNIV_L, "none", LEVEL))
    emit("simpson", "seq_p_total ~ ORF length, within transcript", float(np.median(cl)),
         n=len(cl),
         population=pop(UNIV_C, CORR_R,
                        "within-transcript tie-corrected Spearman, median across transcripts"))

    P("\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
