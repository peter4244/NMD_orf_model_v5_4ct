#!/usr/bin/env python
"""
probe_capture_ablation.py — is the capture head reading initiation context, or
junction geometry, or the admission rule?

Follows probe_selection_depth.py, which found on chr21 (in-sample) that p_capture
is 0.676 at reference start codons against 0.234 elsewhere, with the Cavener-Ray
PWM explaining 9% of the gap. Three things that result needed and did not have.

  HELD-OUT DATA. chr21 is a training chromosome. This runs on chr2, which is
  validation, and on chr21 for comparison.

  THE ADMISSION RULE CONFOUNDS THE KOZAK MATCHING. Non-reference candidates are
  admitted only at or above the MANE floor of -1.2508; the reference start codon
  is admitted whatever its score (plan §3.3 step 4). So the two arms are drawn
  from different ranges of the very variable being matched on, and the lowest
  decile is the one that straddles the floor. Restricting BOTH arms to at or
  above the floor removes the truncation rather than bounding it.

  WHAT THE CAPTURE HEAD ACTUALLY READS. Capture sees the start-codon window and
  nothing else -- model_v6.py builds z_p from init_head(enc_init(atg)) alone, and
  the structural block reaches only decay. Within that window the discriminating
  channels are the bases (0-3), the junction mark (4) and the rolling GC (5);
  channels 6-8 are a phase grid fixed by the anchor and carry nothing. Zeroing
  channel 4 asks whether the preference is initiation context or junction
  geometry. Zeroing channel 5 asks whether it is composition, which accounted for
  every 3'UTR motif finding this project has tested.

THE NORMALIZATION COMES FROM THE CHECKPOINT, not from the tensor. A single
chromosome's build computes its own constants over its own rows, and the model
was fit with the full training split's. It cannot touch p_capture -- the
structural block never reaches the capture head -- but it is wrong for anything
that reads the logit, so it is fixed here rather than carried.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model_v6 import ScanningNMDModel          # noqa: E402
from tensor_io import decode_windows           # noqa: E402

KOZAK_FLOOR = -1.2507921188400943               # plan §3.3 step 3


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=1, permute_bins=bool(a.get("permute_bins", False)))
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


def capture(model, atg, zero_channels=()):
    x = atg.clone()
    for c in zero_channels:
        x[:, c] = 0.0
    with torch.no_grad():
        return torch.sigmoid(model.init_head(model.enc_init(x)).squeeze(-1)).numpy()


def gap(cap, ref, mask=None):
    m = np.ones(len(cap), bool) if mask is None else mask
    a, b = m & ref, m & ~ref
    if a.sum() < 5 or b.sum() < 5:
        return np.nan, int(a.sum()), int(b.sum())
    return float(cap[a].mean() - cap[b].mean()), int(a.sum()), int(b.sum())


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    args = ap.parse_args()

    with h5py.File(Path(args.tensor) / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s, o_e = f["orf_start"][:], f["orf_end"][:]
        codes = f["codes"][:]
        split = np.array([s.decode() for s in f["split"][:]])

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score",
                                "admitted_by"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    key = {s: i for i, s in enumerate(iso)}
    pool["tx"] = pool.isoform_id.map(key)
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)

    print(f"tensor {args.tensor}")
    print(f"  {len(iso):,} transcripts, {len(pool):,} candidates")
    print(f"  splits: " + ", ".join(f"{s}={int((split==s).sum()):,}"
                                    for s in sorted(set(split))))
    print(f"  admitted_by: " + ", ".join(
        f"{k}={v:,}" for k, v in pool.admitted_by.value_counts().items()))

    ckpts = sorted(Path(args.ckpt_dir).glob("b8_s*.pt"))
    caps = {k: [] for k in ("base", "no_junc", "no_gc", "seq_only")}
    for cp in ckpts:
        model, ck = load(cp)
        acc = {k: [] for k in caps}
        for i in range(len(iso)):
            sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
            s0 = o_s[sl].astype(np.int64)
            atg = torch.as_tensor(decode_windows(codes[sl][:, 0], s0, 900, s0))
            acc["base"].append(capture(model, atg))
            acc["no_junc"].append(capture(model, atg, (4,)))
            acc["no_gc"].append(capture(model, atg, (5,)))
            acc["seq_only"].append(capture(model, atg, (4, 5)))
        for k in caps:
            caps[k].append(np.concatenate(acc[k]))
        print(f"  {cp.name}: done", flush=True)
    cap = {k: np.mean(v, axis=0) for k, v in caps.items()}

    ref = pool.is_ref_cds.to_numpy() == 1
    kz = pool.kozak_score.to_numpy()
    slot = pool.slot.to_numpy()
    above = kz >= KOZAK_FLOOR

    print(f"\n=== the admission-rule confound, sized ===")
    print(f"  reference start codons below the Kozak floor : "
          f"{int((ref & ~above).sum()):,} of {int(ref.sum()):,} "
          f"({100*(ref & ~above).mean()/max(ref.mean(),1e-9):.1f}%)")
    print(f"  non-reference candidates below the floor     : "
          f"{int((~ref & ~above).sum()):,}")
    print(f"  -> below the floor the non-reference arm is nearly empty, so a decile")
    print(f"     spanning it is not matched. Restricting both arms to >= the floor")
    print(f"     removes the truncation instead of bounding it.")

    print(f"\n=== p_capture: reference start vs everything else ===")
    print(f"  {'condition':<34} {'gap':>9} {'n ref':>7} {'n other':>9}")
    for name, mask in (("all candidates", None),
                       ("both arms at or above the floor", above),
                       ("...and slot 0 only", above & (slot == 0)),
                       ("...and slot 1-4", above & (slot >= 1) & (slot <= 4)),
                       ("...and slot 5+", above & (slot >= 5))):
        g, na, nb = gap(cap["base"], ref, mask)
        print(f"  {name:<34} {g:>+9.4f} {na:>7,} {nb:>9,}")

    print(f"\n  Kozak-decile matched, restricted to at or above the floor:")
    d = pd.qcut(kz[above], 10, labels=False, duplicates="drop")
    num = den = 0.0
    print(f"  {'decile':>7} {'kozak':>8} {'n ref':>6} {'n other':>8} {'gap':>9}")
    for i in sorted(set(d)):
        m = np.zeros(len(cap["base"]), bool); m[np.flatnonzero(above)[d == i]] = True
        g, na, nb = gap(cap["base"], ref, m)
        if not np.isnan(g):
            num += na * g; den += na
            print(f"  {i:>7} {kz[m].mean():>8.3f} {na:>6,} {nb:>8,} {g:>+9.4f}")
    print(f"  n-weighted mean gap, floor-restricted and Kozak-matched: {num/den:+.4f}")

    print(f"\n=== what the capture head is reading (Track A's ablation) ===")
    print(f"  Capture sees the start-codon window alone. Zeroing a channel of that")
    print(f"  window at inference asks what the preference survives on.")
    print(f"  {'channels zeroed':<34} {'gap':>9} {'vs base':>9} {'median p':>10}")
    base_g, _, _ = gap(cap["base"], ref, above)
    for name, k in (("none (baseline)", "base"),
                    ("4  junction mark", "no_junc"),
                    ("5  rolling GC", "no_gc"),
                    ("4 and 5  bases only", "seq_only")):
        g, _, _ = gap(cap[k], ref, above)
        print(f"  {name:<34} {g:>+9.4f} {100*g/base_g:>8.0f}% {np.median(cap[k]):>10.4f}")
    print(f"\n  A gap that survives with 4 and 5 zeroed is carried by base identity")
    print(f"  in the start-codon window -- initiation context. One that collapses when")
    print(f"  channel 4 goes was junction geometry.")

    print(f"\n=== selection depth, out of sample ===")
    p = cap["base"]
    # P_select from the capture probabilities alone: stick-breaking needs nothing else
    sel = np.empty_like(p)
    at = 0
    for i in range(len(iso)):
        k = int(cnt[i]); q = p[at:at + k]
        sel[at:at + k] = q * np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])
        at += k
    tot = sel.sum()
    for lo, hi in ((0, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)):
        m = (slot >= lo) & (slot <= hi)
        if m.any():
            lab = f"slots {lo}-{hi}" if hi < 10**9 else f"slots {lo}+"
            print(f"  {lab:<14} {100*sel[m].sum()/tot:>6.2f}% of mass  "
                  f"{int(m.sum()):>7,} candidates")
    print(f"  candidates with P_select < 1e-4: {100*np.mean(sel < 1e-4):.1f}%")
    print(f"  p_capture median {np.median(p):.4f}  mean {p.mean():.4f}  "
          f"sd {p.std():.4f}  within 0.05 of 0.5: {100*np.mean(np.abs(p-0.5)<0.05):.1f}%")


if __name__ == "__main__":
    main()
