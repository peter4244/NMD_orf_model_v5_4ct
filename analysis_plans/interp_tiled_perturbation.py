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
import os
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
# D92: each band IS a universe under D81, so `universe=` names it and `restriction`
# is none. Mapping lives beside the bands so the two cannot drift apart.
BAND_UNIVERSE = {(0, 80): "U-TILE12000-ORF-0-80", (80, 160): "U-TILE12000-ORF-80-160",
                 (160, 200): "U-TILE12000-ORF-160-200",
                 (200, 10 ** 9): "U-TILE12000-ORF-200-INF"}

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


# ------------------------------------- initiation context: position versus content
#
# PLAN: analysis_plans/PLAN_INITIATION_POSITION_VS_CONTENT.md. Predictions registered at
# 165369d; AMENDED at fdc847d, before any model was loaded, because the fill-matched
# control it registered cannot be built. PIN fdc847d WITH --predicted-at -- the earlier
# sha names a plan whose control section no longer describes what this code does.
#
# The -13 finding says arrangement matters immediately 5' of the start codon. It does
# NOT say the head reads initiation-context SEQUENCE, for two reasons that are in the
# producer rather than in the result: the statistic is median |delta z_p|, which is
# unsigned, and `shuffle` preserves composition, so what was measured is arrangement
# sensitivity. A content reader should move z_p UP for a more initiation-like context
# and DOWN for a less initiation-like one, and an absolute value collapses those.
#
# BASE ENCODING IS READ FROM tensor_io.py:12, NOT ASSUMED -- bits 0-2 are
# 0 unfilled, 1 A, 2 C, 3 G, 4 T, 5 filled-but-not-ACGT.
#
# THE ANCHOR IS VERIFIED, NOT ASSUMED. Indices 900/901/902 decode to A/T/G in
# 2,000 of 2,000 sampled windows of results_tensor_v6/nmd_tensor.h5. The Kozak
# signal is present in the candidate set at both critical positions: purine at -3
# in 72.3% of windows, G at +4 in 37.6%.
#
# THE MANIPULATION IS MINIMAL, AND THAT IS A CHOICE THIS FILE MUST NOT HIDE. `strong`
# and `weak` differ ONLY at -3 and +4, the two positions Kozak strength actually turns
# on. Writing a full gccRccAUGG consensus instead would change ~10 bases and confound
# the contrast with a large composition change, which the scramble arm could no longer
# control -- it is composition-matched to the OBSERVED context, not to a consensus.
# The cost of the minimal form is sensitivity: two bases inside a ~42-position
# receptive field is a small intervention and a null is correspondingly weak evidence.
#
# THE TWO POSITIONS ARE NOT EQUALLY FILLED, AND THE ARM IS A NO-OP WHERE THEY ARE NOT.
# Measured over 5,000 windows: -3 is filled in 99.9%, +4 in 97.0%. A base is written
# only where one already sits, so `strong`/`weak` manipulate ONE position rather than
# two in ~3% of windows. State that with any result -- an effective n that differs
# between the two positions of the same arm is an unstated population, which is this
# project's signature defect. For reference at greater distance, fill falls to 93.4% at
# -200 and 60.9% at -800, which is what makes the distal control's fill-matching a
# measurement rather than a guess.
#
# THE START CODON IS NEVER TOUCHED in any arm. Destroying it is already known to RAISE
# capture (+0.0022 mean, lowered in only 39% of windows), so an arm that moved it would
# confound the manipulation with that effect.

A_, C_, G_, T_ = 1, 2, 3, 4
AUG = (900, 903)                    # [start, stop) -- A T G, verified above
KOZAK_M3 = 897                      # -3 by Kozak numbering (A of AUG is +1)
KOZAK_P4 = 903                      # +4, the base after the G
CONTEXT = (894, 904)                # -6 .. +4, the span the scramble arm permutes

CONTEXT_MODES = {
    # mode      (-3 base, +4 base)   Kozak reading
    "strong":   (A_, G_),            # purine at -3 AND G at +4 -- both strong
    "weak":     (C_, T_),            # pyrimidine at -3, non-G at +4 -- both weak
}


def perturb_context(codes, mode, rng, m3=KOZAK_M3, p4=KOZAK_P4, span=CONTEXT):
    """Return a copy of `codes` with the initiation context manipulated.

    `strong` / `weak` write the two Kozak-critical bases; `scramble` permutes base
    identity within `span` while holding composition. Every arm preserves the fill
    mask (a base is written only where one already sits), the junction bits, and the
    start codon itself.
    """
    out = codes.copy()
    a_lo, a_hi = AUG

    if mode in CONTEXT_MODES:
        for idx, base in zip((m3, p4), CONTEXT_MODES[mode]):
            if a_lo <= idx < a_hi:
                raise ValueError(f"index {idx} is inside the start codon")
            col = out[:, idx]
            state = col & 7
            # ONLY where a real base already sits. Writing into an unfilled position
            # would move the fill mask, which is the leak this whole analysis tests.
            w = (state >= 1) & (state <= 4)
            out[w, idx] = (col[w] & 8) | base
        return out

    if mode != "scramble":
        raise ValueError(mode)

    lo, hi = span
    state = out[:, lo:hi] & 7
    junc = out[:, lo:hi] & 8
    movable = (state >= 1) & (state <= 4)
    # hold the start codon fixed inside the span
    keep = np.zeros(hi - lo, dtype=bool)
    keep[max(a_lo - lo, 0):max(a_hi - lo, 0)] = True
    movable[:, keep] = False
    for i in range(out.shape[0]):
        m = movable[i]
        if int(m.sum()) < 2:
            continue
        state[i, m] = rng.permutation(state[i, m])
    out[:, lo:hi] = state | junc
    return out


def resolve_tools():
    """Where claim_emit lives, and WHICH COPY it is.

    The cluster copy is a DEPLOYMENT ARTIFACT, not a tracked file -- it is staged at
    $NMD_TOOLS and can drift from the repo it was copied from. A copy that cannot say
    which version it is has exactly the property this project has already paid for
    twice, so the runlog carries the checksum of the file that actually ran rather
    than a path that merely looks right. The sha is computed from the resolved file,
    never declared, because a declared checksum is a claim about an artifact instead
    of a measurement of it.
    """
    import hashlib
    d = Path(os.environ.get(
        "NMD_TOOLS", str(Path.home() / "claude_projects/nmd_lung_longread_2026/tools")))
    p = d / "claim_emit.py"
    if not p.exists():
        sys.exit(f"claim_emit.py not found at {p}. Set NMD_TOOLS to the staged tools "
                 f"directory (on Explorer: export NMD_TOOLS=$HOME/cc/tools).")
    return d, p, hashlib.sha256(p.read_bytes()).hexdigest()


def pick_distal_control(codes, min_gap=2 * RECEPTIVE):
    """The control span: same width, same spacing, as close to the anchor as legal.

    NO SPAN IS FILL-MATCHED TO THE ANCHOR AND THAT IS A PROPERTY OF THE ENCODING, NOT A
    TOLERANCE TO LOOSEN. Measured over 5,000 windows: the anchor span is 99.63% filled
    and fill falls monotonically with distance -- 96.9% at 84 upstream, 93.1% at 200,
    60.1% at 800. Nothing outside the anchor's reach comes within two points.

    Matching MEAN fill was the original design and was dropped on Pete's ruling of
    2026-08-02: a population average is coarser than the per-window property it would be
    controlling, which is the shape of error this project has logged repeatedly. The
    control is instead applied WITHIN windows filled at both spans -- see
    `measure_context`, which does the conditioning.

    So this returns the CLOSEST legal span, because every extra base of distance costs
    fill and buys nothing: `min_gap` of two receptive fields already puts the control
    outside anything the anchor's own convolutions reach.
    """
    lo, hi = CONTEXT
    width = hi - lo
    gap = KOZAK_P4 - KOZAK_M3
    start = lo - min_gap - width
    if start < 0:
        return None, None, None, float("nan")
    off = KOZAK_M3 - lo                           # same offset within the span
    fill = float((((codes[:, start:start + width] & 7) > 0)).mean())
    return (start, start + width), start + off, start + off + gap, fill


def measure_context(codes, base_z, logits, rng, emit=None, seed=0):
    """Signed response of z_p to each context arm, at the anchor and at the control.

    SIGNED IS THE POINT. The existing tiling statistic is median |delta z_p|, and an
    absolute value cannot tell "reads the motif" from "is disturbed by any change
    here". A content reader moves z_p UP for `strong` and DOWN for `weak`; a merely
    position-sensitive head moves it by a similar magnitude either way with no ordering.
    Both are reported so the magnitude remains comparable to the tiling run.
    """
    lo, hi = CONTEXT
    anchor_fill = float((((codes[:, lo:hi] & 7) > 0)).mean())
    ctl_span, ctl_m3, ctl_p4, ctl_fill = pick_distal_control(codes)
    if ctl_span is None:
        print("\n  *** no legal control span in a window this size — positional comparison "
              "not attempted")
        return []

    # THE CONFOUND IS REMOVED PER WINDOW, NOT ON AVERAGE. Pete's ruling, 2026-08-02.
    # No span outside the anchor's reach is fill-matched to it, so both arms are
    # measured on the SAME windows -- those fully filled at both spans. A window
    # unfilled at either would make its control arm a partial no-op, and comparing a
    # no-op against a real manipulation is what would produce a spurious positional
    # effect. The excluded fraction is reported because a conditioned population is
    # still a population and has to be stated.
    keep = ((((codes[:, lo:hi] & 7) > 0).all(1))
            & (((codes[:, ctl_span[0]:ctl_span[1]] & 7) > 0).all(1)))
    n_all = len(codes)
    codes, base_z = codes[keep], base_z[keep]

    print(f"\n  CONTEXT ARMS — anchor span {CONTEXT} writing {KOZAK_M3} and {KOZAK_P4}, "
          f"span fill {anchor_fill:.1%}")
    print(f"  control span {ctl_span} writing {ctl_m3} and {ctl_p4}, span fill "
          f"{ctl_fill:.1%} ({KOZAK_M3 - ctl_m3} positions upstream, "
          f"{2 * RECEPTIVE} = two receptive fields)")
    print(f"  CONDITIONED on windows fully filled at BOTH spans: {int(keep.sum()):,} of "
          f"{n_all:,} = {100 * float(keep.mean()):.1f}%  ({n_all - int(keep.sum()):,} excluded)")

    print(f"\n  {'arm':<10}{'position':<10}{'median signed':>15}{'median |d|':>13}"
          f"{'frac up':>10}{'n':>9}")
    rows = []
    for mode in ("strong", "weak", "scramble"):
        for label, span, m3, p4 in (("anchor", CONTEXT, KOZAK_M3, KOZAK_P4),
                                    ("control", ctl_span, ctl_m3, ctl_p4)):
            if span is None:
                continue
            z = logits(perturb_context(codes, mode, rng, m3=m3, p4=p4, span=span))
            d = z - base_z
            med, mad = float(np.median(d)), float(np.median(np.abs(d)))
            up = float((d > 0).mean())
            n = int(len(d))
            print(f"  {mode:<10}{label:<10}{med:>15.6g}{mad:>13.6g}{up:>10.1%}{n:>9,}")
            rows.append((mode, label, med, mad, up, n))
            if emit is not None:
                # D92 grammar. Four fields, nothing else. Interpretive caveats belong to
                # the universe and live in the registry's `why` -- the conditioning, the
                # 6.2% excluded and the held-fixed start codon are all properties of
                # U-TILE12000-BOTHSPANS, not of this measurement.
                def pop(est):
                    return (f"universe=U-TILE12000-BOTHSPANS; restriction=none; "
                            f"estimator={est}; "
                            f"params=arm={mode},position={label},"
                            f"span={span[0]}-{span[1]},written={m3}+{p4},seed={seed}")
                emit("initiation_context", f"median signed delta z_p, {mode} at {label}",
                     med, n=n,
                     population=pop("median signed delta z_p under context substitution"))
                emit("initiation_context", f"median abs delta z_p, {mode} at {label}",
                     mad, n=n,
                     population=pop("median absolute delta z_p under context substitution"))
    print("\n  REGISTERED READING (165369d, control amended fdc847d): content => strong and")
    print("  weak SEPARATE IN SIGN at")
    print("  the anchor, ordered strong > weak, and that ordering is materially weaker at")
    print("  the control. Position-only => similar magnitudes, no consistent sign ordering.")
    print("  Neither => signed responses inseparable from scramble at both positions.")
    print("  The scramble arm is the composition control; strong/weak minus scramble is the")
    print("  part attributable to WHICH bases rather than how many.")
    return rows


def self_test_context(codes, rng):
    """Assert what every context arm must satisfy, with no model. Returns ok."""
    ok = True
    a_lo, a_hi = AUG
    lo, hi = CONTEXT
    print("\nCONTEXT SELF-TEST")
    for mode in ("strong", "weak", "scramble"):
        p = perturb_context(codes, mode, rng)
        fill_bad = int((((p & 7) > 0) != ((codes & 7) > 0)).sum())
        junc_bad = int(((p & 8) != (codes & 8)).sum())
        aug_bad = int((p[:, a_lo:a_hi] != codes[:, a_lo:a_hi]).sum())
        outside = np.concatenate([p[:, :lo] != codes[:, :lo],
                                  p[:, hi:] != codes[:, hi:]], axis=1)
        out_bad = int(outside.sum())
        comp_bad = wrote = 0
        if mode == "scramble":
            # composition held over the span, EXCLUDING the start codon
            cols = [c for c in range(lo, hi) if not (a_lo <= c < a_hi)]
            comp_bad = int((np.sort(p[:, cols] & 7, axis=1)
                            != np.sort(codes[:, cols] & 7, axis=1)).any(axis=1).sum())
        else:
            # the arm must actually write the context it claims to, wherever filled
            for idx, base in zip((KOZAK_M3, KOZAK_P4), CONTEXT_MODES[mode]):
                filled = ((codes[:, idx] & 7) >= 1) & ((codes[:, idx] & 7) <= 4)
                wrote += int(((p[filled, idx] & 7) == base).sum())
                if not ((p[filled, idx] & 7) == base).all():
                    comp_bad += 1
        bad = fill_bad or junc_bad or aug_bad or out_bad or comp_bad
        ok = ok and not bad
        print(f"  {mode:<9} fill {fill_bad:>4}  junction {junc_bad:>4}  "
              f"START CODON {aug_bad:>4}  outside span {out_bad:>4}  "
              + (f"composition changed {comp_bad:>4}" if mode == "scramble"
                 else f"context written {wrote:>6}"))
    return ok


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
    ok = self_test_context(codes, rng) and ok
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
    ap.add_argument("--context", action="store_true",
                    help="run the initiation position-versus-content arms as well as the "
                         "tiling, and emit their values through claim_emit")
    # PRE-REGISTRATION, per Pete's ruling of 2026-08-02: fingerprint PLUS witnessed
    # order. A commit sha proves what the plan SAID, not that it said it FIRST -- the
    # plan could be amended after the result is seen and the new sha recorded at
    # filing, because file_result.py runs after the result exists and cannot witness
    # the order. Printing the sha into the runlog at START puts the ordering in the
    # artifact the JOB writes and timestamps. file_result.py refuses a filing whose
    # --predicted-at commit does not appear in the runlog.
    ap.add_argument("--predicted-at", default="", metavar="SHA:PATH",
                    help="commit and path of the plan registering this run's "
                         "predictions. Printed to the runlog before anything is loaded.")
    a = ap.parse_args()

    # FIRST, before the bank is opened or the checkpoint is loaded, so the runlog
    # witnesses that the prediction existed before the measurement did.
    if a.predicted_at:
        if ":" not in a.predicted_at:
            sys.exit(f"--predicted-at must be <commit>:<path>, got {a.predicted_at!r}")
        print(f"PREDICTED-AT {a.predicted_at}", flush=True)

    # Which claim_emit actually ran. Anchored at column 0 like the line above, and
    # printed BEFORE the bank is opened so a job that dies later still says which
    # emitter it was going to use.
    if a.context:
        _tools_dir, _emit_path, _emit_sha = resolve_tools()
        print(f"CLAIM-EMIT   {_emit_sha} {_emit_path}", flush=True)
        sys.path.insert(0, str(_tools_dir))
        from claim_emit import emit as _emit                          # noqa: E402
    else:
        _emit = None

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
            for blo, bhi, bname in LENGTH_BANDS:
                m = (Ls >= blo) & (Ls < bhi)
                v = float(np.median(d[m])) if m.any() else float("nan")
                row.append(v)
                # EMIT THE TILE TABLE. It was printed and never emitted, so the table
                # two windows rely on had a runlog and no row -- talk, not a record.
                # Band fill is reported per band, not overall: an unfilled tile is a
                # no-op, which is why three cells in the table read exactly 0.
                if _emit is not None and m.any():
                    bfill = float((((codes[m, lo:hi] & 7) > 0)).mean())
                    _emit("tiled_perturbation",
                          f"median abs delta z_p, tile {lo}-{hi} (offset {off:+d}), "
                          f"ORF length band {bname.split()[0]}",
                          v, n=int(m.sum()),
                          population=(
                              f"universe={BAND_UNIVERSE[(blo, bhi)]}; restriction=none; "
                              f"estimator=median absolute delta z_p under within-tile "
                              f"shuffle; params=tile={lo}-{hi},offset={off:+d},"
                              f"tile_fill={bfill:.3f},seed={a.seed}"))
        print(f"  {f'{lo}-{hi}':>12} {off:>+7} {fill_frac:>6.1%} " +
              "".join(f"{v:>12.4g}" for v in row))
    if a.context:
        # claim_emit lives in the analysis repo and is the ONLY producer of a values
        # file file_result.py will accept -- it refuses anything it did not write, and
        # `population` is mandatory at the call site because that is the only place it
        # is known. Imported rather than copied, the route CLAUDE.md documents.
        sys.path.insert(0, str(_tools_dir))
        measure_context(codes, base_z, logits, rng, emit=_emit, seed=a.seed)

    print("\n  Read DOWN each length-band column for the peak, then ACROSS bands:")
    print("  a peak at a fixed offset is prediction A; a peak whose offset tracks")
    print("  min(100, length/2) is prediction B. Both registered before this ran.")


if __name__ == "__main__":
    main()
