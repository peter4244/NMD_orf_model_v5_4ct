#!/usr/bin/env python
"""
interp_tiled_perturbation.py — does the initiation head read sequence, or read where
our window's fill stops?

ROW: ROW_TILED_PERTURBATION_2026-08-02.md. Read it first; the two predictions are
registered there and were fixed before this file existed.

  A  the head reads initiation context -> sensitivity peaks at a FIXED offset from
     the AUG anchor, unmoved as ORF length changes
  B  the head reads the fill boundary  -> sensitivity peaks at min(100, length/2),
     which MOVES with length below 200 nt and pins at +100 above it

THE DISCRIMINATING AXIS IS WHETHER THE PEAK MOVES, NOT WHERE IT IS. At 60 nt the two
predictions point at nearly the same positions; at 190 nt they sit 85 bases apart.
ORF length supplies the variation for free -- which is the intervention we could not
run for window size, because no model varies that.

TILE SCHEME, AND WHY IT IS NOT UNIFORM. The encoder's bins are 1000/n_bins wide -- at
n_bins=8 that is 125 -- and the last bin spans [875,1000), which contains BOTH the AUG
anchor at index 900 AND the whole downstream fill region. A uniform bin-width tiling
therefore cannot separate the two predictions: they land in the same tile. So the
window is tiled COARSE upstream (where the question is only "does the UTR matter") and
FINE downstream of the anchor (where the two predictions diverge).

Resolution is bounded at ~42 positions by the receptive field -- conv1 k=15, then
MaxPool1d(4), then conv2 k=7 -- so downstream tiles finer than that measure blur, not
location. The length bands below are chosen so the PREDICTED boundary differs between
bands by more than 42, which is what makes the movement detectable at all.

THE PERTURBATION, AND THE PART THAT MUST NOT BE GOT WRONG. Codes pack the base state
in the low three bits (0 unfilled, 1-4 ACGT, 5 N) and the junction flag in bit 3. A
shuffle permutes base identity AMONG FILLED, NON-N POSITIONS ONLY, in place:

  - the FILL MASK is preserved exactly. Moving it would recreate the very leak this
    analysis exists to test, and section 5.3 records two earlier leaks of that class
    that were invisible to ablation
  - the JUNCTION BIT stays where it is. A junction is annotation, not a property of
    the base sitting on it
  - the FRAME CHANNELS are positional, so they are untouched by construction -- which
    means this tests sequence content with reading frame HELD FIXED, for free
  - composition within the tile is preserved, so the arm isolates ARRANGEMENT

The substitution arm redraws filled bases at random instead, destroying composition as
well; the difference between the two arms is the composition contribution.

--self-test RUNS WITHOUT THE MODEL and asserts the four properties above on real codes.
It is in this file rather than beside it because a validator in another script is one
that stops being run, and this project has the scar.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ATG_LEFT = 900                      # AUG sits at this index of the 1000-wide window
WINDOW = 1000
RECEPTIVE = 42                      # conv1 k15 -> pool4 -> conv2 k7, in input positions

# Length bands chosen so the PREDICTED fill boundary differs by more than the
# receptive field between neighbouring bands. Boundary = min(100, L/2).
LENGTH_BANDS = [(0, 80, "<80      boundary <40"),
                (80, 160, "80-160   boundary 40-80"),
                (160, 200, "160-200  boundary 80-100"),
                (200, 10 ** 9, ">=200    boundary pinned at 100")]


def tile_scheme(coarse=125, fine=25):
    """Coarse upstream, fine downstream. Returns [(lo, hi, offset_from_anchor)]."""
    tiles = []
    for lo in range(0, ATG_LEFT, coarse):
        hi = min(lo + coarse, ATG_LEFT)
        tiles.append((lo, hi, (lo + hi) // 2 - ATG_LEFT))
    for lo in range(ATG_LEFT, WINDOW, fine):
        hi = min(lo + fine, WINDOW)
        tiles.append((lo, hi, (lo + hi) // 2 - ATG_LEFT))
    return tiles


def perturb_tile(codes, lo, hi, mode, rng):
    """Return a copy of `codes` (n, W) with tile [lo,hi) perturbed.

    Preserves the fill mask, the junction bits, and (for mode="shuffle") the base
    composition within the tile. See the module docstring for why each matters.
    """
    out = codes.copy()
    state = out[:, lo:hi] & 7
    junc = out[:, lo:hi] & 8
    movable = (state >= 1) & (state <= 4)          # filled, not N, not unfilled
    for i in range(out.shape[0]):
        m = movable[i]
        k = int(m.sum())
        if k < 2:
            continue                                # nothing to permute
        vals = state[i, m]
        if mode == "shuffle":
            new = rng.permutation(vals)
        elif mode == "substitute":
            new = rng.integers(1, 5, size=k, dtype=vals.dtype)
        else:
            raise ValueError(mode)
        state[i, m] = new
    out[:, lo:hi] = state | junc                    # junction bit put back unchanged
    return out


# ------------------------------------------------------------------ the self-test

def self_test(bank, n=400, seed=0):
    """Assert the four preservation properties on real codes, with no model."""
    import h5py
    rng = np.random.default_rng(seed)
    with h5py.File(bank, "r") as f:
        codes = f["codes"][:n, 0, :].astype(np.uint8)
    tiles = tile_scheme()
    print(f"SELF-TEST on {n} real ATG windows, {len(tiles)} tiles\n")
    ok = True
    for mode in ("shuffle", "substitute"):
        worst_fill = worst_junc = 0
        comp_bad = 0
        for lo, hi, _ in tiles:
            p = perturb_tile(codes, lo, hi, mode, rng)
            # 1. fill mask preserved everywhere
            worst_fill = max(worst_fill,
                             int((((p & 7) > 0) != ((codes & 7) > 0)).sum()))
            # 2. junction bits preserved everywhere
            worst_junc = max(worst_junc, int(((p & 8) != (codes & 8)).sum()))
            # 3. nothing outside the tile changed
            outside = np.concatenate([p[:, :lo] != codes[:, :lo],
                                      p[:, hi:] != codes[:, hi:]], axis=1)
            worst_fill = max(worst_fill, int(outside.sum()))
            # 4. composition within tile preserved, for shuffle only
            if mode == "shuffle":
                a = np.sort(p[:, lo:hi] & 7, axis=1)
                b = np.sort(codes[:, lo:hi] & 7, axis=1)
                comp_bad += int((a != b).any(axis=1).sum())
        print(f"  {mode:<11} fill-mask/outside violations {worst_fill:>5}   "
              f"junction violations {worst_junc:>5}   composition changed {comp_bad:>5}")
        if worst_fill or worst_junc:
            ok = False
        if mode == "shuffle" and comp_bad:
            ok = False
    # 5. AN ENTIRELY UNFILLED TILE MUST BE A NO-OP -- checked PER WINDOW, not per
    # column. Requiring a column unfilled across every window is far too strict and
    # silently skipped this assertion on the first run; the real condition is that
    # wherever a given window's tile holds no filled positions, nothing changes.
    n_checked = n_bad = 0
    for lo, hi, _ in tiles:
        p = perturb_tile(codes, lo, hi, "shuffle", rng)
        empty = (((codes[:, lo:hi] & 7) > 0).sum(axis=1) == 0)
        n_checked += int(empty.sum())
        if empty.any():
            n_bad += int((p[empty] != codes[empty]).sum())
    if n_checked:
        print(f"  unfilled (window, tile) pairs checked {n_checked:,}   "
              f"changed by perturbation {n_bad}")
        ok = ok and n_bad == 0
    else:
        print("  *** no unfilled (window, tile) pair in the sample -- assertion untested")
        ok = False
    print("\n  SELF-TEST PASSED" if ok else "\n  *** SELF-TEST FAILED")
    return ok


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="results_tensor_v6/nmd_tensor.h5")
    ap.add_argument("--ckpt")
    ap.add_argument("--sample", type=int, default=12000)
    ap.add_argument("--coarse", type=int, default=125)
    ap.add_argument("--fine", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(0 if self_test(a.bank, seed=a.seed) else 1)
    if not a.ckpt:
        ap.error("--ckpt is required unless --self-test")

    import h5py, torch
    from build_ism_bank import load_model, Encoders
    from window_cache import WindowCache

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, cols, args = load_model(a.ckpt, dev)

    # ---- THE VETO. Registered in the row as prerequisite 1.
    print(f"checkpoint n_bins={args['n_bins']}  permute_bins="
          f"{bool(args.get('permute_bins', False))}  "
          f"conv_channels={args['conv_channels']}  variant={args['variant']}")
    if bool(args.get("permute_bins", False)):
        sys.exit("VETO: permute_bins is true. Bins are shuffled per pass, position is "
                 "destroyed by design, and this analysis measures nothing. The row "
                 "names this as prerequisite 1.")
    print(f"bin width {WINDOW // args['n_bins']} positions; receptive field ~{RECEPTIVE}\n")

    rng = np.random.default_rng(a.seed)
    with h5py.File(a.bank, "r") as f:
        n_cand = f["codes"].shape[0]
        orf_start = f["orf_start"][:]
        orf_end = f["orf_end"][:]
        L = orf_end - orf_start + 1
        take = np.sort(rng.choice(n_cand, min(a.sample, n_cand), replace=False))
        codes = f["codes"][take, 0, :].astype(np.uint8)
    Ls, starts = L[take], orf_start[take]
    tiles = tile_scheme(a.coarse, a.fine)
    print(f"sampled {len(take):,} candidates of {n_cand:,};  {len(tiles)} tiles "
          f"(coarse {a.coarse} upstream, fine {a.fine} downstream)\n")

    enc = Encoders(model, None)

    def logits(c):
        wc = WindowCache(c, starts, ATG_LEFT, starts, dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(c), 512):
                x = wc.base[i:i + 512]
                idx = torch.arange(i, min(i + 512, len(c)), device=dev)
                out.append(model.init_head(enc.init(x, idx)).squeeze(-1).cpu().numpy())
        return np.concatenate(out)

    base_z = logits(codes)
    print(f"  {'tile':>12} {'offset':>7} {'fill%':>7} " +
          "".join(f"{n.split()[0]:>12}" for _, _, n in LENGTH_BANDS))
    for lo, hi, off in tiles:
        fill_frac = float((((codes[:, lo:hi] & 7) > 0)).mean())
        row = []
        for mode in ("shuffle",):
            z = logits(perturb_tile(codes, lo, hi, mode, rng))
            d = np.abs(z - base_z)
            for blo, bhi, _ in LENGTH_BANDS:
                m = (Ls >= blo) & (Ls < bhi)
                row.append(float(np.median(d[m])) if m.any() else float("nan"))
        print(f"  {f'{lo}-{hi}':>12} {off:>+7} {fill_frac:>6.1%} " +
              "".join(f"{v:>12.4g}" for v in row))
    print("\n  Read DOWN each length-band column for the peak, then ACROSS bands:")
    print("  a peak at a fixed offset is prediction A; a peak whose offset tracks")
    print("  min(100, length/2) is prediction B. Both registered before this ran.")


if __name__ == "__main__":
    main()
