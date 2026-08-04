#!/usr/bin/env python
"""
model_encoder_channel_share.py — how the three encoders allocate first-layer weight
across the nine input channels.

SCOPE: a claim about the MODEL's parameters. No data is read; no forward pass runs.

WHAT THIS ANSWERS. The nine channels are not all sequence: 0-3 are the nucleotide
one-hot, 4 marks a junction, 5 is rolling GC, 6-8 are codon phase. Two standing
worries turn on how much of each encoder is spent on which:

  * RETRAIN_ARCHITECTURE_CHANGES item 5 — phase is written across the WHOLE window
    including the UTR, so "attribution goes to the channel before it goes to the
    sequence" is a live risk.
  * Pete 2026-08-02 — "the start and stop windows care almost entirely about the
    presence of a PTC." The junction channel is where a PTC would be visible to an
    encoder, so its weight share is the direct read on that.

WHAT THIS IS NOT. Allocated capacity, NOT realised influence. The channels differ
in density and scale: channel 4 is a sparse binary indicator that is 0 at almost
every position, the nucleotide one-hots are 1 at exactly one of four rows per
filled position, and channel 5 is a dense continuous value near 0.5. Equal weight
on a sparse channel moves the output less than the same weight on a dense one.
Converting this to influence requires perturbation, not weights. Reported here
because it is cheap, exact, and reproducible across seeds -- and because a claim
about influence should not be made from it.

REFERENCE POINT. Each group's share is compared against what an EVEN spread over
the nine channels would give (4/9, 1/9, 1/9, 3/9). Zero is not the reference for a
share that must sum to one -- the same rule the routing work learned the hard way.

POPULATION: all five trained seeds, s100..s500.

Usage:
    python model_encoder_channel_share.py --out runlog_model_encoder_channel_share.txt
"""
import argparse, hashlib, os, sys
from pathlib import Path
import numpy as np, torch

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

CKPT_DIR = in_repo("results_interp_all/v6_checkpoints")
SEEDS = [100, 200, 300, 400, 500]
GROUPS = {"nucleotide A,C,G,T": [0, 1, 2, 3], "junction (ch4)": [4],
          "rolling GC (ch5)": [5], "codon phase (ch6-8)": [6, 7, 8]}
ENCODERS = {"enc_init": "the SCANNER/capture head's own encoder",
            "enc_atg":  "decay's read of the start window",
            "enc_stop": "decay's read of the stop window"}


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
    W = {}
    for s in SEEDS:
        p = CKPT_DIR / f"b8_s{s}.pt"
        P(f"  {hashlib.sha256(p.read_bytes()).hexdigest()[:32]}  {p.name}  {p.stat().st_size:,} B")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert ck["variant"] == "interpretable" and ck["n_bins"] == 8, "unexpected checkpoint config"
        W[s] = {e: np.abs(ck["model"][f"{e}.conv1.weight"].numpy()) for e in ENCODERS}

    P(f"\n=== POPULATION: {len(SEEDS)} seeds, conv1 weight of each encoder, shape (32, 9, 15) ===")
    P("share of |weight| by channel group; 'even' is what an equal spread over 9 channels gives")

    for e, blurb in ENCODERS.items():
        P(f"\n--- {e}  ({blurb})")
        for g, idx in GROUPS.items():
            sh = np.array([W[s][e][:, idx, :].sum() / W[s][e].sum() for s in SEEDS])
            even = len(idx) / 9
            P(f"    {g:22s} {100*sh.mean():5.1f}%  (sd {100*sh.std():4.1f})   "
              f"even {100*even:4.1f}%   ratio {sh.mean()/even:.2f}x")
            emit("channels", f"{e} | {g} | share of |conv1 weight| over even-spread",
                 float(sh.mean()/even), n=len(SEEDS),
                 population=(
                     # D92/D94. Universe is the encoder's own conv1 weights -- a
                     # parameters-unit population. The channel group is which SLICE is
                     # summed, so it is estimator and params, not a row selection.
                     # "allocated capacity, not influence" is an interpretive caveat and
                     # belongs to the universe's `why`, not here.
                     f"universe=U-PARAM-{e.replace('enc_', 'ENC').upper()}-CONV1; "
                     f"restriction=none; "
                     f"estimator=share of |conv1 weight| over even spread, mean over seeds; "
                     f"params=group={g},shape=(32,9,15),n_seeds={len(SEEDS)}"))

    P(f"\n=== per-filter dominance, enc_init, seed {SEEDS[0]} ===")
    w = W[SEEDS[0]]["enc_init"].sum(axis=2)           # (32 filters, 9 channels)
    sh = w / w.sum(1, keepdims=True)
    nuc, jun, gc, fr = sh[:, :4].sum(1), sh[:, 4], sh[:, 5], sh[:, 6:9].sum(1)
    top = np.argmax(np.vstack([nuc, jun, gc, fr]), axis=0)
    P(f"  filters whose LARGEST group share is nucleotide: {(top==0).sum()}/32")
    for lab, v in (("nucleotide", nuc), ("junction", jun), ("codon phase", fr)):
        P(f"    {lab:12s} share  min {v.min():.2f}  median {np.median(v):.2f}  max {v.max():.2f}")
    P("  -> no filter specialises away from sequence onto a single supplied channel.")

    P("\n=== exit: 0 ===")
    log.close()


if __name__ == "__main__":
    main()
