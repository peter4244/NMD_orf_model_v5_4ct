#!/usr/bin/env python
"""
model_regime_reconstruction.py — rerun the routing measurements locally, stratified
by the 200 nt ORF-length boundary, against the v6 checkpoint.

WHY THIS EXISTS. Two things the record leaves open and one it gets wrong:

  1. Every regime-split number in the narrative is measured on p_capture, the HEAD.
     The step to p_select, the PICKER, was never run. The narrative flags this twice
     as the cheapest outstanding thing.
  2. The queue-geometry null (+0.334 raw) has no runlog anywhere. It exists in a
     commit message and a worklog row. It is the reference the whole "zero is not
     the null" correction rests on.
  3. The narrative quotes the LENGTH-HELD null comparison, which the null's own
     author withdrew: holding length does not describe a regime where within-
     transcript candidates span ~83x in length.

CHOICES MADE HERE, STATED RATHER THAN BURIED IN THE CODE:

  * POPULATION. Default is the ISM bank's transcript subset, because that is the
     population every published routing number uses. --all runs the full tensor.
  * p_select IS COMPUTED OVER THE FULL CANDIDATE LIST of each transcript and THEN
     subset by the length stratum. The queue is the model; re-running it on a
     subset would measure a different model. This is the one choice in this file
     that could reasonably have gone the other way.
  * k>=4 candidates REQUIRED WITHIN THE STRATUM for that stratum's correlation.
  * NULL: every z_p set to the same constant, so p_select depends only on slot.
     Computed on the same rows, with the same filters, in every stratum.
  * Quantities come from the TENSOR's own structural_raw and orf coordinates, not
     from a join against the pool, so a mismatched join cannot silently reorder.
  * THE GRID IS COMPLETED, not extended for its own sake. seq_p_total against ORF
     length existed only POOLED (R-sm-md-0008) while every other pair was
     stratified, so the three per-candidate quantities could not be presented
     symmetrically. p_nmd is per TRANSCRIPT and cannot enter a within-transcript
     correlation at all; that asymmetry is scope, not a gap, and is not filled.

Usage:
    python model_regime_reconstruction.py --out runlog.txt [--all] [--seed 100]
"""
import argparse, sys, json, hashlib, time, os
from pathlib import Path
import sys
import numpy as np, h5py, torch
from scipy.stats import spearmanr

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

REPO = REPO_ROOT
TOOLS = Path(os.environ.get("NMD_TOOLS",
                            Path.home() / "claude_projects/nmd_lung_longread_2026/tools"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TOOLS))
from tensor_io import decode_windows                      # noqa: E402
from model_v6 import ScanningNMDModel                     # noqa: E402
from claim_emit import emit                               # noqa: E402

TENSOR = in_repo("results_tensor_v6/nmd_tensor.h5")
SUBSET = in_repo("results_ism_v6/ism_subset.tsv")
ATG_LEFT, STOP_LEFT = 900, 500


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def within_tx_spearman(vals, ref, tx_index, min_k):
    """Median within-transcript Spearman of vals against ref. Returns (median, n)."""
    out = []
    for lo, hi in tx_index:
        if hi - lo < min_k:
            continue
        a, b = vals[lo:hi], ref[lo:hi]
        if np.all(a == a[0]) or np.all(b == b[0]):
            continue
        r = spearmanr(a, b).statistic
        if np.isfinite(r):
            out.append(r)
    return (float(np.median(out)) if out else float("nan")), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--all", action="store_true", help="full tensor, not the bank subset")
    ap.add_argument("--min-k", type=int, default=4)
    ap.add_argument("--boundary", type=int, default=200)
    a = ap.parse_args()

    log = open(a.out, "w")
    def P(*s):
        print(*s); print(*s, file=log); log.flush()

    ckpt = in_repo(f"results_interp_all/v6_checkpoints/b8_s{a.seed}.pt")
    P("=== code and artifact provenance (sha256, what actually ran) ===")
    P(f"  {sha(__file__)}  {Path(__file__).name}")
    P(f"  {sha(ckpt)}  {ckpt.name}")
    P(f"  tensor {TENSOR}  {TENSOR.stat().st_size:,} bytes")

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
    P(f"  checkpoint: variant={ck['variant']} conv_channels={ck['conv_channels']} "
      f"n_bins={ck['n_bins']} seed={ck['seed']} epoch={ck['epoch']} val_auc={ck['val_auc']:.4f}")
    P(f"  structural_columns={ck['structural_columns']}  trained on {ck['args']['tensor']}")

    model = ScanningNMDModel(n_seq_channels=9, conv_channels=ck["conv_channels"],
                             n_bins=ck["n_bins"], seq_embed_dim=32, n_structural=1)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    P(f"  state_dict loaded STRICT, {sum(p.numel() for p in model.parameters()):,} params")

    f = h5py.File(TENSOR, "r")
    iso = np.array([s.decode() for s in f["isoform_id"][:]])
    off, cnt = f["offset"][:], f["count"][:]
    P(f"\n=== population ===")
    P(f"  tensor: {len(iso):,} transcripts, {f['codes'].shape[0]:,} candidates")

    if a.all:
        keep_tx = np.arange(len(iso)); pop_name = "FULL TENSOR"
    else:
        import pandas as pd
        want = set(pd.read_csv(SUBSET, sep="\t")["isoform_id"].astype(str))
        keep_tx = np.flatnonzero(np.isin(iso, list(want))); pop_name = "ISM BANK SUBSET"
        P(f"  ism_subset.tsv: {len(want):,} transcripts requested")
    P(f"  POPULATION = {pop_name}: {len(keep_tx):,} transcripts")

    # ---- forward pass, one transcript at a time (K is ragged) ----------------
    P(f"\n=== forward pass ===")
    t0 = time.time()
    P_cap, P_sel, D, EJC, LEN, START, SLOT, TXI = [], [], [], [], [], [], [], []
    tx_index, cur = [], 0
    for n_done, t in enumerate(keep_tx):
        lo, hi = int(off[t]), int(off[t]) + int(cnt[t])
        K = hi - lo
        if K < 2:
            continue
        codes = f["codes"][lo:hi]                       # (K, 2, 1000)
        os_, oe_ = f["orf_start"][lo:hi], f["orf_end"][lo:hi]
        struct = f["structural"][lo:hi]                 # z-scored
        raw = f["structural_raw"][lo:hi]
        atg = decode_windows(codes[:, 0, :], os_, ATG_LEFT, os_)
        stop = decode_windows(codes[:, 1, :], oe_, STOP_LEFT, os_)
        with torch.no_grad():
            _, parts = model(torch.from_numpy(atg)[None], torch.from_numpy(stop)[None],
                             torch.from_numpy(struct[:, :1])[None],
                             torch.ones(1, K, dtype=torch.bool), return_parts=True)
        P_cap.append(parts["p"][0].numpy()); P_sel.append(parts["p_select"][0].numpy())
        D.append(parts["d"][0].numpy())
        EJC.append(raw[:, 0]); LEN.append((oe_ - os_ + 1).astype(np.float64))
        START.append(os_.astype(np.float64)); SLOT.append(np.arange(K, dtype=np.float64))
        TXI.append(np.full(K, len(tx_index)))
        tx_index.append((cur, cur + K)); cur += K
        if (n_done + 1) % 500 == 0:
            P(f"    {n_done+1:,}/{len(keep_tx):,} transcripts  {time.time()-t0:.0f}s")
    f.close()

    p_cap = np.concatenate(P_cap); p_sel = np.concatenate(P_sel); d = np.concatenate(D)
    ejc = np.concatenate(EJC); orflen = np.concatenate(LEN)
    start = np.concatenate(START); slot = np.concatenate(SLOT)
    P(f"  {len(tx_index):,} transcripts with >=2 candidates, {len(p_cap):,} candidates, "
      f"{time.time()-t0:.0f}s")

    # ---- the null: every z_p equal => p_select is a function of slot alone ----
    # log_sel_k = k*log(1-c) + log(c), strictly decreasing in k for any c, so the
    # within-transcript RANK of p_select equals the rank of -slot. Rank statistics
    # are all this file reports, so -slot IS the null exactly, for every c.
    null_sel = -slot

    # Universe per stratum. ISM-subset parent when the run is subset-restricted, full
    # tensor otherwise -- the universe is a property of the INVOCATION, not the producer.
    UNIVERSE = ({"ALL": "U-TENSOR-ISM4999-K4", "SHORT": "U-TENSOR-ISM4999-K4-SHORT",
                 "LONG": "U-TENSOR-ISM4999-K4-LONG"} if "ISM" in pop_name.upper() else
                {"ALL": "U-TENSOR-K4", "SHORT": "U-TENSOR-K4-SHORT",
                 "LONG": "U-TENSOR-K4-LONG"})
    # Measurement-dependent and therefore restriction, not universe (D94): which
    # transcripts fail the variance/finiteness test depends on the variable pair.
    RESTRICTION = ("transcripts of the universe dropped where the within-transcript "
                   "Spearman is undefined or not finite for the variable pair measured")

    strata = [("ALL candidates", np.ones(len(p_cap), bool)),
              (f"SHORT  ORF < {a.boundary} nt", orflen < a.boundary),
              (f"LONG   ORF >= {a.boundary} nt", orflen >= a.boundary)]

    P(f"\n{'='*78}\nWITHIN-TRANSCRIPT SPEARMAN, median across transcripts")
    P(f"population {pop_name};  k>={a.min_k} WITHIN the stratum;  seed {a.seed}")
    P(f"p_select computed over the FULL candidate list, then subset by stratum")
    P("="*78)

    for label, m in strata:
        idx = [(lo, hi) for lo, hi in tx_index]
        # restrict each transcript's slice to the stratum, keeping order
        sub_tx, sv_cap, sv_sel, sv_d, sv_ejc, sv_len, sv_null, sv_start = [], [], [], [], [], [], [], []
        c = 0
        for lo, hi in idx:
            sel = np.flatnonzero(m[lo:hi]) + lo
            if len(sel) < a.min_k:
                continue
            sv_cap.append(p_cap[sel]); sv_sel.append(p_sel[sel]); sv_d.append(d[sel])
            sv_ejc.append(ejc[sel]); sv_len.append(orflen[sel]); sv_null.append(null_sel[sel])
            sv_start.append(start[sel])
            sub_tx.append((c, c + len(sel))); c += len(sel)
        if not sub_tx:
            P(f"\n--- {label}: no transcript reaches k>={a.min_k}"); continue
        cap = np.concatenate(sv_cap); sl = np.concatenate(sv_sel); dd = np.concatenate(sv_d)
        ej = np.concatenate(sv_ejc); ln = np.concatenate(sv_len); nl = np.concatenate(sv_null)
        st = np.concatenate(sv_start)

        P(f"\n--- {label}   ({len(sub_tx):,} transcripts, {len(cap):,} candidates)")
        stratum = label.split()[0]
        # D92 grammar, D94 universes. The stratum selects TRANSCRIPTS (k>=min_k within
        # the stratum), so the universe is parent x class at transcript unit -- not the
        # candidate-unit U-TENSOR-ORF-* pair, which partitions the tensor and is a
        # different unit over a different parent.
        pop = (f"universe={UNIVERSE[stratum]}; restriction={RESTRICTION}; "
               f"estimator=within-transcript tie-corrected Spearman, median across "
               f"transcripts, p_select computed over the FULL candidate list then subset; "
               f"params=boundary_nt={a.boundary},min_k={a.min_k},seed={a.seed}")
        for name, v, r, published in [
            ("p_capture ~ ejc      [the head]", cap, ej, None),
            ("p_select  ~ ejc      [THE PICKER]", sl, ej, -0.050 if stratum == "ALL" else None),
            ("NULL: queue only, no model ~ ejc", nl, ej, None),
            ("posterior ~ ejc      [p_select*d]", sl * dd, ej, None),
            ("p_capture ~ ORF length", cap, ln, 0.761 if stratum == "ALL" else None),
            ("p_select  ~ ORF length", sl, ln, None),
            ("p_capture ~ d", cap, dd, None),
            ("p_select  ~ d", sl, dd, None),
            ("d         ~ ORF length", dd, ln, None),
            ("p_capture ~ p_select", cap, sl, None),
            ("orf_start ~ ejc      [premise]", st, ej, None),
        ]:
            med, n = within_tx_spearman(v, r, sub_tx, a.min_k)
            P(f"    {name:38s} {med:+.3f}   n {n:,}")
            emit("routing", f"{name.split('[')[0].strip()} | {stratum}", med,
                 published=published, n=n, population=pop)

    P(f"\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
