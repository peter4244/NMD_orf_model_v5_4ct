#!/usr/bin/env python3
"""
probe_ism_cost.py — what does in silico mutagenesis cost on this model?

ANALYSIS4_PLAN §7 requires this before anything is submitted, and fixes the §4 subsample size
from it. The discipline has already caught a 35.3 GiB peak against a 32 GB request and a
per-isoform rate that was wrong by 6x.

It also measures the two things §6 lists as acceptance criteria, because they are cheap here
and expensive to discover later:

  - THE GC RECOMPUTATION IS EXACT. Channel 5 is a rolling GC fraction computed from THIS
    WINDOW'S OWN SEQUENCE (data_prep.py:178-205, changed 2026-07-29), not from the transcript.
    That is what makes ISM tractable: a substitution's effect on channel 5 is fully determined
    by the window, so no transcript sequence is needed. The probe verifies this by recomputing
    channel 5 from the UNPERTURBED one-hot and comparing against the stored channel.
  - A NO-OP PERTURBATION GIVES EXACTLY ZERO. Substituting the observed base for itself must
    return an effect of 0.0. Free, and it catches an indexing error that a plausible-looking
    profile would not.

No shap, no background distribution, no attribution: ISM is a difference between two forward
passes of the model, so this runs locally.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model import build_model                                       # noqa: E402
from utils import NMDDataset, load_config, resolve_checkpoint       # noqa: E402

CONFIG   = REPO / "config_dn.yaml"
CKPT_DIR = REPO / "results_4ct_sweep"
TAG      = "atg1000_stop1000"
MEMBER   = 100
N_ISO    = 8            # cost probe only
BATCH    = 512
GC_WIN   = 50           # data_prep.GC_ROLLING_WINDOW
NT       = "ACGT"


def rolling_gc(is_gc, window=GC_WIN):
    """data_prep.compute_rolling_gc, over an is_gc indicator. Same support, same edge shrink."""
    n = len(is_gc)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    cs = np.concatenate([[0.0], np.cumsum(is_gc)])
    half = window // 2
    p = np.arange(n)
    lo = np.maximum(0, p - half)
    hi = np.minimum(n, p + half + 1)
    return ((cs[hi] - cs[lo]) / (hi - lo)).astype(np.float32)


def filled_span(win):
    """The unpadded span of a window: positions where any frame channel (6-8) is set."""
    on = win[6:9].sum(axis=0) > 0
    idx = np.flatnonzero(on)
    return (int(idx[0]), int(idx[-1] + 1)) if len(idx) else (0, 0)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    wa = int(TAG.split("_")[0].replace("atg", ""))
    ws = int(TAG.split("_")[1].replace("stop", ""))
    dev = torch.device("cpu")

    model = build_model({**cfg["model"], "window_size_atg": wa, "window_size_stop": ws}).to(dev)
    ck = resolve_checkpoint(CKPT_DIR, TAG, MEMBER)
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model_state_dict"])
    model.eval()
    print(f"model {ck.name}   window {wa}")

    h5 = str(REPO / cfg["data"]["hdf5_path"])
    ds = NMDDataset(h5, wa, ws, split="all", restrict_to=np.arange(N_ISO))

    # ---- acceptance check 1: is the GC channel reproducible from the window alone? ----
    print("\n=== GC channel recomputed from the window's own one-hot, vs the stored channel ===")
    worst = 0.0
    for i in range(N_ISO):
        w = ds[i]["atg_windows"][0].numpy()
        lo, hi = filled_span(w)
        onehot = w[0:4, lo:hi]
        is_gc = (onehot[1] + onehot[2])                     # C or G
        got = rolling_gc(is_gc)
        worst = max(worst, float(np.abs(got - w[5, lo:hi]).max()))
    print(f"  max |recomputed - stored| over {N_ISO} isoforms: {worst:.3e}")
    print(f"  -> {'EXACT (float16 storage precision)' if worst < 2e-3 else 'MISMATCH -- the channel is NOT window-local'}")

    # ---- the ISM inner loop ----
    print(f"\n=== ISM cost, start branch, {N_ISO} isoforms ===")
    per_iso_positions, per_iso_time, noop_max = [], [], 0.0
    t_all = time.perf_counter()
    for i in range(N_ISO):
        s = ds[i]
        aw = s["atg_windows"].numpy().copy()
        w0 = aw[0]
        lo, hi = filled_span(w0)
        npos = hi - lo
        base_idx = w0[0:4, lo:hi].argmax(axis=0)

        # Build every perturbation for this isoform: (npos x 3) alternatives, plus one no-op.
        perts, meta = [], []
        for k, p in enumerate(range(lo, hi)):
            obs = int(base_idx[k])
            for b in range(4):
                if b == obs and k != 0:
                    continue                                 # no-op only once, at the first pos
                w = w0.copy()
                w[0:4, p] = 0.0
                w[b, p] = 1.0
                seg = w[0:4, lo:hi]
                w[5, lo:hi] = rolling_gc(seg[1] + seg[2])    # RECOMPUTE derived channel
                perts.append(w)
                meta.append((p, b, b == obs))
        perts = np.stack(perts)

        t0 = time.perf_counter()
        with torch.no_grad():
            base_logit = model(torch.from_numpy(aw).unsqueeze(0),
                               s["stop_windows"].unsqueeze(0),
                               s["orf_features"].unsqueeze(0),
                               s["orf_mask"].unsqueeze(0)).item()
            outs = []
            for st in range(0, len(perts), BATCH):
                chunk = perts[st:st + BATCH]
                n = len(chunk)
                full = np.repeat(aw[None], n, axis=0)
                full[:, 0] = chunk
                outs.append(model(torch.from_numpy(full),
                                  s["stop_windows"].unsqueeze(0).expand(n, -1, -1, -1),
                                  s["orf_features"].unsqueeze(0).expand(n, -1, -1),
                                  s["orf_mask"].unsqueeze(0).expand(n, -1)).squeeze(-1).numpy())
        dt = time.perf_counter() - t0
        eff = np.concatenate(outs) - base_logit
        for (p, b, is_noop), e in zip(meta, eff):
            if is_noop:
                noop_max = max(noop_max, abs(float(e)))
        per_iso_positions.append(npos)
        per_iso_time.append(dt)
        print(f"  isoform {i}: {npos:5,} unpadded positions, {len(perts):6,} forward rows, {dt:6.2f} s")
    total = time.perf_counter() - t_all

    # ---- acceptance check 2 ----
    print(f"\n=== no-op perturbation (observed base substituted for itself) ===")
    print(f"  max |effect| = {noop_max:.3e}   -> {'PASS (exactly zero)' if noop_max == 0.0 else 'NON-ZERO -- indexing error'}")

    npos = float(np.mean(per_iso_positions))
    t_iso = float(np.mean(per_iso_time))
    print(f"\n=== COST ===")
    print(f"  mean unpadded positions per window : {npos:,.0f} of {wa:,}  ({100*npos/wa:.1f}%)")
    print(f"  mean wall time per isoform, 1 branch: {t_iso:.2f} s")
    print(f"  -> per isoform, BOTH branches       : {2*t_iso:.2f} s")
    print(f"  -> per isoform, both x 5 members    : {10*t_iso:.1f} s")
    for n in (500, 2000, 9321):
        h = 10 * t_iso * n / 3600
        print(f"     {n:5,} isoforms, both branches x 5 members: {h:7.1f} CPU-hours "
              f"({h/20:5.1f} h wall at 20-way parallel)")
    print(f"\n  NOTE: measured on this laptop's CPU at W={wa}. The wider configuration doubles the")
    print(f"        window and therefore roughly doubles both position count and per-row cost.")


if __name__ == "__main__":
    main()
