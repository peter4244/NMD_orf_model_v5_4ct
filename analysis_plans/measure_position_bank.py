#!/usr/bin/env python3
"""
measure_position_bank.py — one forward-pass sweep, cached; all contrasts derived afterwards.

WHY SPLIT IT. Every substitution at a position is measured for all four bases, so every
single-position contrast anyone might want -- purine vs pyrimidine, G vs notG, A vs T, G vs C,
GC vs AT -- is a re-cut of the SAME numbers. Recomputing forward passes to change an operator
invites the two runs to differ in something other than the operator. This writes the raw
per-(member, isoform, position, base) change in log-odds once; analyse_position_bank.py reads
it. Nothing here forms a contrast.

WHAT IS MEASURED. For slot SLOT's start window: at each position in the bank, substitute each
of A/C/G/T, recompute channel 5 over the window's filled extent, score, and record the change
against a baseline built through the identical recomputation path. Also recorded per isoform:
the attention weight the model puts on each ORF slot, the observed base, the created-motif
exclusions, and eligibility.

THE POSITION BANK is two hypothesis positions (Kozak -3, +4) and N_CTRL control positions
spread over +/-CTRL_HALFSPAN. Controls carry the identical operator, exclusions, recomputation
and population test, so they are a reference for everything systematic -- including the
window-local (unpropagated) substitution's off-manifold contamination.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path("/Users/petecastaldi/claude_projects/NMD_orf_model_v5_4ct")
sys.path.insert(0, str(REPO))
from model import build_model                                    # noqa: E402
from utils import NMDDataset, resolve_checkpoint                 # noqa: E402

NT = "ACGT"
GC_WIN = 50
STOPS = {("T", "A", "A"), ("T", "A", "G"), ("T", "G", "A")}


def pos_to_index(p, W):
    """Codon convention (ANALYSIS4_PLAN 3.1): A of the start codon is +1, no zero."""
    if p >= 1:
        return p + W // 2 - 2
    if p <= -1:
        return p + W // 2 - 1
    raise ValueError("position 0 does not exist in the codon convention")


def rolling_gc(is_gc, window=GC_WIN):
    """Exact reimplementation of data_prep.compute_rolling_gc over a one-hot G+C row."""
    n = len(is_gc)
    cs = np.concatenate([[0.0], np.cumsum(is_gc)])
    half = window // 2
    p = np.arange(n)
    lo = np.maximum(0, p - half)
    hi = np.minimum(n, p + half + 1)
    return ((cs[hi] - cs[lo]) / (hi - lo)).astype(np.float32)


def frame_span(win):
    """Filled extent. Channels 6-8 fill exactly arr_start:arr_end (data_prep.py:207-211)."""
    on = win[6:9].sum(axis=0) > 0
    i = np.flatnonzero(on)
    return (int(i[0]), int(i[-1] + 1)) if len(i) else (0, 0)


def recompute_gc(w, lo, hi):
    seg = w[0:4, lo:hi]
    w[5, lo:hi] = rolling_gc(seg[1] + seg[2])
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iso", type=int, default=300)
    ap.add_argument("--n-ctrl", type=int, default=0,
                    help="optional cap on controls, split evenly between the upstream and "
                         "downstream arms. 0 keeps every frame-matched position.")
    ap.add_argument("--ctrl-halfspan", type=int, default=60)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--tag", default="atg1000_stop1000")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    ap.add_argument("--out", default="position_bank.npz")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    W, SLOT = args.width, args.slot
    cfg = yaml.safe_load((REPO / "config_dn.yaml").read_text())
    h5 = str(REPO / cfg["data"]["hdf5_path"])

    targets = [("-3", -3), ("+4", 4)]
    codon = {W // 2 - 1, W // 2, W // 2 + 1}
    tgt_off = {(pos_to_index(p, W) - (W // 2 - 1)) % 3 for _, p in targets}
    assert len(tgt_off) == 1, f"hypothesis positions differ in frame offset: {tgt_off}"
    off0 = tgt_off.pop()

    # CONTROLS ARE SELECTED BY FRAME OFFSET, NOT BY STRIDE. The created-motif exclusion rate
    # depends on where a position sits within its in-frame codon, so a mixed-offset control set
    # is not a matched reference. A uniform stride of 3 does NOT achieve matching: the codon
    # convention skips zero and the codon, so stepping by 3 from -60 lands on the targets'
    # offset upstream and a DIFFERENT offset downstream. The earlier bank was built that way and
    # its 19 frame-matched controls were 100% upstream, leaving the downstream hypothesis
    # position (+4) with no region-matched reference.
    cand = [p for p in range(-args.ctrl_halfspan, args.ctrl_halfspan + 1)
            if p != 0 and pos_to_index(p, W) not in codon
            and p not in [q for _, q in targets]
            and (pos_to_index(p, W) - (W // 2 - 1)) % 3 == off0]
    up = [p for p in cand if p < 0]
    dn = [p for p in cand if p > 0]
    if args.n_ctrl:                       # optional cap, applied evenly to both arms
        k = max(1, args.n_ctrl // 2)
        up = up[::max(1, len(up) // k)][:k]
        dn = dn[::max(1, len(dn) // k)][:k]
    ctrl = [(f"c{p:+d}", p) for p in up + dn]
    bank = targets + ctrl
    idxs = np.array([pos_to_index(p, W) for _, p in bank])
    names = np.array([n for n, _ in bank])
    codon_pos = np.array([p for _, p in bank])
    print(f"bank: {len(targets)} targets at frame offset {off0} + {len(ctrl)} controls, "
          f"all at the same offset, within +/-{args.ctrl_halfspan}")
    print(f"  upstream controls   ({len(up)}): {up}")
    print(f"  downstream controls ({len(dn)}): {dn}")

    ds = NMDDataset(h5, W, W, split="all", restrict_to=np.arange(args.n_iso))
    n_iso, n_pos, n_orf = len(ds), len(idxs), ds.orf_mask.shape[1]

    # --- geometry, exclusions, eligibility: computed once, shared by every member ----------
    ok = np.zeros((n_iso, n_pos), dtype=bool)
    excl = np.zeros((n_iso, n_pos, 4), dtype=bool)
    obs = np.full((n_iso, n_pos), -1, dtype=np.int8)
    spans, w0s = [], []
    for i in range(n_iso):
        w0 = ds[i]["atg_windows"].numpy()[SLOT]
        lo, hi = frame_span(w0)
        nuc = w0[0:4].sum(axis=0) > 0.5              # a real ACGT base, so N is excluded
        for k, j in enumerate(idxs):
            if not (lo <= j - 2 and j + 2 < hi and nuc[j - 2:j + 3].all()):
                continue
            ok[i, k] = True
            obs[i, k] = int(w0[0:4, j].argmax())
            base = [NT[int(w0[0:4, j + d].argmax())] for d in (-2, -1, 0, 1, 2)]
            c = (j - (W // 2 - 1)) % 3               # offset of j within its in-frame codon
            for b in range(4):
                trio = list(base)
                trio[2] = NT[b]
                cod = tuple(trio[2 - c:5 - c])       # the in-frame codon containing j
                excl[i, k, b] = (cod in STOPS) or any(
                    tuple(trio[m:m + 3]) == ("A", "T", "G") for m in range(3))
        spans.append((lo, hi))
        w0s.append(w0)
    print(f"\nisoforms {n_iso}; eligible at all {n_pos} positions: {int(ok.all(1).sum())} "
          f"({ok.all(1).mean():.1%}); per-position n {ok.sum(0).min()}-{ok.sum(0).max()}")
    print(f"observed-base composition, purine fraction: " + ", ".join(
        f"{names[k]} {np.isin(obs[ok[:,k],k],[0,2]).mean():.3f}" for k in range(min(n_pos, 4))))

    # --- forward passes -------------------------------------------------------------------
    vals = np.full((len(args.seeds), n_iso, n_pos, 4), np.nan, dtype=np.float64)
    attn = np.full((len(args.seeds), n_iso, n_orf), np.nan, dtype=np.float64)
    floor = np.zeros(len(args.seeds))
    for si, seed in enumerate(args.seeds):
        t0 = time.time()
        m = build_model({**cfg["model"], "window_size_atg": W, "window_size_stop": W})
        m.load_state_dict(torch.load(resolve_checkpoint(REPO / "results_4ct_sweep",
                                                        args.tag, seed),
                                     map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval()
        for i in range(n_iso):
            s = ds[i]
            aw = s["atg_windows"].numpy()
            lo, hi = spans[i]
            wins, keys = [recompute_gc(w0s[i].copy(), lo, hi)], []
            for k, j in enumerate(idxs):
                if not ok[i, k]:
                    continue
                for b in range(4):
                    w = w0s[i].copy()
                    w[0:4, j] = 0
                    w[b, j] = 1
                    wins.append(recompute_gc(w, lo, hi))
                    keys.append((k, b))
            batch = np.repeat(aw[None], len(wins), axis=0)
            batch[:, SLOT] = np.stack(wins)
            with torch.no_grad():
                out, at = m(torch.from_numpy(batch),
                            s["stop_windows"].unsqueeze(0).expand(len(wins), -1, -1, -1),
                            s["orf_features"].unsqueeze(0).expand(len(wins), -1, -1),
                            s["orf_mask"].unsqueeze(0).expand(len(wins), -1),
                            return_attention=True)
            attn[si, i] = at[0].numpy()
            lg = out.squeeze(-1).numpy().astype(np.float64)
            for (k, b), v in zip(keys, lg[1:]):
                vals[si, i, k, b] = v - lg[0]
        # method floor: putting the observed base back is a no-op, so its effect is the
        # arithmetic noise of the measurement and bounds every effect from below.
        noop = np.array([vals[si, i, k, obs[i, k]] for i in range(n_iso)
                         for k in range(n_pos) if ok[i, k]])
        floor[si] = np.abs(noop).max()
        print(f"  seed {seed}: {time.time()-t0:6.1f}s   no-op |effect| max {floor[si]:.3e}")

    np.savez_compressed(args.out, vals=vals, attn=attn, ok=ok, excl=excl, obs=obs,
                        names=names, idxs=idxs, codon_pos=codon_pos, floor=floor,
                        labels=ds.labels, seeds=np.array(args.seeds),
                        orf_mask=ds.orf_mask, W=W, slot=SLOT, tag=args.tag)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
