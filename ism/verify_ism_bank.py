#!/usr/bin/env python
"""
verify_ism_bank.py — check the bank against a deliberately slow reference.

The bank of build_ism_bank.py is fast because it recomputes only the candidate
windows a substitution touches and reuses every other candidate's embeddings. That
is an identity in eval mode, but it is an identity about THIS model wired THIS
way, and the plan's whole point is that a claim like that gets checked against the
producing code rather than against a description of it.

So this script computes the same quantity a second time, the slow way: for every
perturbation it rebuilds the full (K, 9, 1000) input of BOTH windows from the
stored codes and calls the model's own forward() end to end. No caching, no
partial recomputation, no shared code path with the bank beyond tensor_io and the
model itself.

FIVE CHECKS.

  1  obs against the FASTA. The observed base the bank reports at each transcript
     position must be the base the transcript actually has there. This checks the
     whole window geometry — anchors, offsets, midpoint clip — against something
     outside the tensor.
  2  window spans. Every position the bank calls valid must be inside some span it
     shipped, and every position inside a span must be valid or non-ACGT.
  3  propagation. A perturbation must reach EVERY window holding that coordinate.
     Verified by mutating one window only and showing the answer differs wherever
     a coordinate is covered more than once — the failure mode the interpretation
     window flagged in their own version.
  4  vals against the reference, entry by entry.
  5  the no-op floor, on the reference path.

Usage:
    python verify_ism_bank.py --tensor results_tensor_chr21 \\
        --checkpoint /tmp/ckpt.pt --bank /tmp/bank.h5 --transcripts 3
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from build_ism_bank import (ATG_LEFT, STOP_LEFT, WINDOW, load_model,
                            window_spans, covering_index)
from tensor_io import decode_windows

FASTA = Path.home() / "claude_projects" / "nmd_deposit_2026" / "source_data" / \
        "sqanti" / "nmd_lungcells_corrected.fasta"


def read_fasta(wanted):
    seqs, name, buf = {}, None, []
    with open(FASTA) as fh:
        for line in fh:
            if line[0] == ">":
                if name in wanted:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if name in wanted:
        seqs[name] = "".join(buf)
    return seqs


def reference_logit(model, codes, orf_start, orf_end, struct, device):
    """The model's own forward(), on the full candidate set, from raw codes."""
    K = len(orf_start)
    atg = decode_windows(codes[:, 0], orf_start, ATG_LEFT, orf_start)
    stop = decode_windows(codes[:, 1], orf_end - 1, STOP_LEFT, orf_start)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        # stable=True: the same quantity the bank computes. The training path
        # clamps P(NMD) and round-trips through it, and both destroy the response
        # in the tails -- comparing against it would compare two different
        # quantities and call the disagreement a bug in the cache.
        return float(model(t(atg)[None], t(stop)[None], t(struct)[None],
                           torch.ones(1, K, dtype=torch.bool, device=device),
                           stable=True).item())


def perturb_codes(codes, orf_start, orf_end, p, b, spans, one_window_only=None):
    """Substitute base b at transcript position p, in every covering window.

    `one_window_only` restricts the substitution to a single (candidate, window)
    pair, which is the off-manifold variant check 3 needs.
    """
    a_lo, a_hi, s_lo, s_hi = spans
    out = codes.copy()
    touched = []
    for k in range(len(orf_start)):
        for w, lo, hi, anchor, left in ((0, a_lo[k], a_hi[k], orf_start[k], ATG_LEFT),
                                        (1, s_lo[k], s_hi[k], orf_end[k] - 1, STOP_LEFT)):
            if not (lo <= p <= hi):
                continue
            touched.append((k, w))
            if one_window_only is not None and (k, w) != one_window_only:
                continue
            i = p - anchor + left
            out[k, w, i] = (out[k, w, i] & 8) | (b + 1)
    return out, touched


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--transcripts", type=int, default=3)
    ap.add_argument("--positions", type=int, default=60,
                    help="positions per transcript checked against the reference")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = args.device
    t0 = time.time()

    model, ck, cols, ckargs = load_model(args.checkpoint, device)
    assert not ckargs.get("permute_bins"), \
        "this reference calls forward(), which redraws the permutation; the control " \
        "arm is verified by its own paired-draw check, not here"

    b = h5py.File(args.bank, "r")
    bid = np.array([s.decode() for s in b["transcript_id"][:]])
    spans_all = b["spans"][:]
    coff, ccnt = b["cand_offset"][:], b["cand_count"][:]

    with h5py.File(str(Path(args.tensor) / "nmd_tensor.h5"), "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        row = {s: i for i, s in enumerate(iso)}
        offset, count = f["offset"][:], f["count"][:]
        o_start_all, o_end_all = f["orf_start"][:], f["orf_end"][:]
        struct_all, codes_all = f["structural"][:], f["codes"][:]

    pick = list(range(min(args.transcripts, len(bid))))
    seqs = read_fasta(set(bid[pick]))

    n_obs_bad = n_span_bad = n_val_bad = 0
    worst_val, worst_base, n_cmp = 0.0, 0.0, 0
    prop_tested = prop_differed = 0
    prop_delta, prop_rel, prop_cover = [], [], []
    by_k = {}
    noop_max = 0.0

    for n in pick:
        iso_id = bid[n]
        i = row[iso_id]
        sl = slice(int(offset[i]), int(offset[i]) + int(count[i]))
        codes = codes_all[sl]
        os_ = o_start_all[sl].astype(np.int64)
        oe_ = o_end_all[sl].astype(np.int64)
        st = struct_all[sl][:, cols]
        seq = seqs[iso_id]
        L = len(seq)
        spans = window_spans(os_, oe_, L)
        valid = b["valid"][n]
        obs = b["obs"][n]
        vals = b["vals"][n]
        K = len(os_)

        # ---- 1. obs against the FASTA -------------------------------------
        vp = np.flatnonzero(valid)
        got = np.array([obs[p] for p in vp])
        want = np.array(["ACGT".find(seq[p]) for p in vp])
        bad = int((got != want).sum())
        n_obs_bad += bad

        # ---- 2. spans -----------------------------------------------------
        srow = spans_all[coff[n]:coff[n] + ccnt[n]]
        assert len(srow) == K and (srow[:, 0] == n).all()
        inside = np.zeros(len(valid), bool)
        for _, k, alo, ahi, slo, shi in srow:
            if ahi >= alo:
                inside[alo - 1:ahi] = True
            if shi >= slo:
                inside[slo - 1:shi] = True
        n_span_bad += int((valid & ~inside).sum())
        # a position inside a span but not valid must be a non-ACGT base
        odd = np.flatnonzero(inside & ~valid)
        n_val_bad += int(sum(1 for p in odd if p < L and seq[p] in "ACGT"))

        # ---- 4/5. vals against the reference ------------------------------
        base_ref = reference_logit(model, codes, os_, oe_, st, device)
        assert abs(base_ref - float(b["base_logit"][n])) < 1e-4, \
            f"{iso_id}: base logit {base_ref} vs bank {float(b['base_logit'][n])}"
        worst_base = max(worst_base, abs(base_ref - float(b["base_logit"][n])))

        probe = rng.choice(vp, size=min(args.positions, len(vp)), replace=False)
        for p0 in probe:
            p = int(p0) + 1
            o = int(obs[p0])
            for bb in range(4):
                pc, touched = perturb_codes(codes, os_, oe_, p, bb, spans)
                ref = reference_logit(model, pc, os_, oe_, st, device) - base_ref
                if bb == o:
                    noop_max = max(noop_max, abs(ref))
                    continue
                d = abs(ref - float(vals[p0, bb]))
                worst_val = max(worst_val, d)
                by_k.setdefault(K, ([], [], []))[0].append(d)
                by_k[K][1].append(abs(ref))
                by_k[K][2].append(iso_id)
                n_cmp += 1

            # ---- 3. propagation --------------------------------------------
            # Not a pass/fail. Propagating to every covering window CANNOT be
            # expected to change every answer: the encoder reduces the length axis
            # by a maximum within each bin, so a single-base change that does not
            # move any bin's maximum leaves the embedding identical. What matters
            # is how often, and by how much, the one-window shortcut differs — the
            # shortcut is what the interpretation window's local version computes.
            if len(touched) > 1:
                bb = (o + 1) % 4
                pc1, _ = perturb_codes(codes, os_, oe_, p, bb, spans,
                                       one_window_only=touched[0])
                one = reference_logit(model, pc1, os_, oe_, st, device) - base_ref
                full = float(vals[p0, bb])
                prop_tested += 1
                d = abs(one - full)
                if d > 1e-6:
                    prop_differed += 1
                    prop_delta.append(d)
                    prop_rel.append(d / max(abs(full), 1e-12))
                prop_cover.append(len(touched))

        print(f"  [{n+1}/{len(pick)}] {iso_id:<34} K={K:>3} L={L:>6,} "
              f"valid={int(valid.sum()):>6,}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== 1. observed base vs the FASTA ===")
    print(f"  mismatches                                   : {n_obs_bad:,}  "
          f"-> {'PASS' if n_obs_bad == 0 else 'FAIL'}")
    print(f"\n=== 2. window spans ===")
    print(f"  valid positions outside every shipped span   : {n_span_bad:,}  "
          f"-> {'PASS' if n_span_bad == 0 else 'FAIL'}")
    print(f"  ACGT positions inside a span but not valid   : {n_val_bad:,}  "
          f"-> {'PASS' if n_val_bad == 0 else 'FAIL'}")
    print(f"\n=== 3. propagation to every covering window ===")
    print(f"  multiply-covered positions tested            : {prop_tested:,}")
    print(f"  windows covering them, mean                  : "
          f"{np.mean(prop_cover) if prop_cover else float('nan'):.2f}")
    print(f"  where the ONE-WINDOW shortcut differs        : {prop_differed:,} "
          f"({100*prop_differed/max(prop_tested,1):.1f}%)")
    if prop_delta:
        print(f"  |shortcut - propagated|, median / max        : "
              f"{np.median(prop_delta):.3e} / {np.max(prop_delta):.3e}")
        print(f"  as a fraction of the propagated effect       : "
              f"median {np.median(prop_rel):.2f}, p90 {np.percentile(prop_rel,90):.2f}, "
              f"max {np.max(prop_rel):.2f}")
    print(f"  -> the bank is the propagated quantity; check 4 is what verifies it.")
    print(f"     A shortcut that differs on {100*prop_differed/max(prop_tested,1):.0f}% of "
          f"multiply-covered positions is not a rounding difference.")
    print(f"\n=== 4. vals vs the uncached reference ===")
    print(f"  comparisons                                  : {n_cmp:,}")
    print(f"  max |bank - reference|                       : {worst_val:.3e}")
    print(f"  max |base logit difference|                  : {worst_base:.3e}")
    print(f"\n=== 5. no-op on the reference path, and the batch-shape offset ===")
    print(f"  max |effect| substituting the observed base  : {noop_max:.3e}")
    print(f"  batch-shape offset the bank removed          : "
          f"{float(b.attrs['batch_shape_offset']):.3e}")
    print(f"\n=== 6. bank-vs-reference disagreement, BY CANDIDATE COUNT ===")
    print(f"  The encoder has three batch-shape regimes (1, 2-7, 8+). The bank now")
    print(f"  differences against a same-chunk baseline and the reference against a")
    print(f"  batch-K pass, so any residual should NOT track K.")
    print(f"  {'K':>5} {'regime':>7} {'n':>5} {'max resid':>12} {'median |eff|':>13} "
          f"{'resid/|eff|':>12}")
    for k in sorted(by_k):
        d, e, iso = by_k[k]
        reg = 1 if k == 1 else (2 if k <= 7 else 3)
        med = float(np.median(e)) if e else float("nan")
        # pair each residual with its own effect rather than dividing maxima
        rel = float(np.median([x / y for x, y in zip(d, e) if y > 0])) if e else float("nan")
        print(f"  {k:>5} {reg:>7} {len(d):>5} {max(d):>12.3e} {med:>13.3e} {rel:>12.3e}")
    print(f"\n  Chunks sit in regime 3. If the residual were only regime DISTANCE,")
    print(f"  K=15 and K=146 (both regime 3, distance 0) would match. If it were only")
    print(f"  relative float precision, resid/|eff| would be flat across every K.")
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
