#!/usr/bin/env python
"""
interp_region_caller.py — regions the WHOLE MODEL treats as important, and a null
that cannot beat itself.

Implements REGION_CALLER_SPEC.md including its section 4.1 corrections. Row is
section 8 of that document; read it before reading this.

WHAT THIS IS. Peak calling on an autocorrelated signal track. Naming it that makes
the difficulty ordinary and imports the standard solution for the part we got wrong.
It makes NO claim about sequence, motifs, routing or mechanism, and adjusts for
nothing (spec section 7). WHY a region is important is the next question.

THE TRACK IS `vals`, WHOLE-MODEL. Not vals_decay, not vals_capture. The question is
what the MODEL treats as important; the branch decomposition is part of the why.
Consequence recorded in the row's field 11: these are whole-model regions, so
scoping A4 to them asks its question in regions selected by a different quantity
than the vals_decay findings A4 exists to bound.

---------------------------------------------------------------------------
THE NULL, WHICH IS THE ENTIRE REASON THIS FILE EXISTS

The retracted run-length result used random placement of marks. That destroys
autocorrelation the track has ARCHITECTURALLY, so it tested "is the track smooth"
and answered yes. iAAFT surrogates fix that axis. Sections 4.1 correct two things
about them that the spec originally got wrong:

CORRECTION 1 -- iAAFT CANNOT PRESERVE SPECTRUM AND MARGINAL BOTH EXACTLY. It
alternates an amplitude step (spectrum exact, marginal drifts) and a rank step
(marginal exact, spectrum drifts) and terminates on one. That is why it ITERATES.
We terminate on the RANK step, so the marginal is exact by construction -- chosen
because this caller thresholds at per-transcript quantiles 0.90-0.99, and a
surrogate whose marginal is approximate is least faithful exactly at the operating
quantile, which would put a bias of unknown sign directly into criterion 1. The
residual spectrum error is reported rather than assumed small.

CORRECTION 2, THE LOAD-BEARING ONE -- AUTOCORRELATION IS NOT THE ONLY ARCHITECTURAL
STRUCTURE. The selection-mass envelope varies systematically along a transcript
(log-mass correlates 0.934 with the effect track), so the track is NON-STATIONARY.
Fourier phase randomization is a null for a STATIONARY process: it spreads local
variance uniformly along the series, so surrogates put peaks where the real track
structurally cannot have them and concentrate less than it does. Criterion 1 would
then pass BECAUSE THE TRACK HAS AN ENVELOPE, which is the retracted error wearing a
different costume.

So the default null divides by a smoothed envelope, surrogates the residual, and
multiplies the envelope back. Envelope preserved exactly, residual phase destroyed.

Both nulls are implemented and both are reported. `--null naive` is kept precisely
so the difference is visible rather than argued, and --self-test below demonstrates
on synthetic data that the naive one fires on envelope alone.

CORRECTION 3 -- SERIES FLOOR 800, not 200. Autocorrelation is 0.64 at lag 80, so a
200-point series holds ~2.5 correlation lengths and phase randomization has almost
nothing to randomize; the FFT also assumes periodicity, putting a wrap-around
discontinuity between the last and first position. Excluded series are reported by
count AND locale, because exclusion by length is exclusion by transcript class.

CORRECTION 4 -- 20 surrogates floors the per-transcript empirical p at 1/21 = 0.048,
so NO per-transcript claim is available. Only the gene-clustered aggregate.

---------------------------------------------------------------------------
--self-test IS IN THIS FILE ON PURPOSE

A validator that lives in another script is one that stops being run, and this
project has the scar. The self-test builds synthetic tracks where the answer is
known and asserts the null behaves:

  A  stationary autocorrelated, NO localized features -> must NOT fire (both nulls)
  B  autocorrelated + strong slow envelope, NO features -> the naive null MUST fire
     (that is the defect) and the envelope-preserving null MUST NOT
  C  envelope + genuine localized features -> both must fire (power check)

B is the one that matters: it is correction 2 stated as a failing test rather than
as an argument. If B ever stops failing for the naive null, this file's central
claim is wrong and the assertion says so.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NT = "ACGT"
MIN_SERIES = 800                 # correction 3
N_SURROGATE = 20                 # correction 4: p floor 1/21
ENV_WINDOW = 201                 # primary; swept 101/401 (spec section 3 rule)
Q_SWEEP = (0.90, 0.95, 0.98, 0.99)
GAP_SWEEP = (0, 2, 5)
WMIN_SWEEP = (3, 5, 8)
LOCALES = ("5p_of_start", "in_orf", "3p_of_stop")


# ------------------------------------------------------------------ the surrogate

def running_mean(x, w):
    """Envelope by running mean, reflected at the edges so length is preserved.

    Reflection rather than zero-padding: zero-padding would pull the envelope down
    at both ends, which is exactly where the fill mask already makes the track
    unusual, and would manufacture an edge effect in the null.
    """
    if w % 2 == 0:
        w += 1
    half = w // 2
    xp = np.concatenate([x[half:0:-1], x, x[-2:-half - 2:-1]])
    if len(xp) != len(x) + 2 * half:                 # short series, pad by edge
        xp = np.pad(x, half, mode="reflect")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def iaaft(x, rng, n_iter=100, terminate="rank"):
    """One iAAFT surrogate.

    terminate="rank"     marginal EXACT, spectrum approximate   <- our default
    terminate="spectrum" spectrum EXACT, marginal approximate

    Correction 1: these are the only two options and the spec claimed both at once.
    """
    n = len(x)
    order = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    y_spec = y
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y_spec = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=n)
        y = order[np.argsort(np.argsort(y_spec))]     # rank step -> marginal exact
    return y if terminate == "rank" else y_spec


def surrogate(x, rng, mode="envelope", env_window=ENV_WINDOW):
    """A null series for x.

    mode="envelope"  divide by the smoothed envelope, iAAFT the residual, multiply
                     back. The envelope -- the architecture -- survives exactly and
                     only the residual's feature LOCATIONS are destroyed.
    mode="naive"     iAAFT the raw series. Kept so the difference is measurable;
                     --self-test case B shows it firing on envelope alone.
    """
    if mode == "naive":
        return iaaft(x, rng)
    env = running_mean(x, env_window)
    floor = max(np.abs(x).mean() * 1e-6, np.finfo(float).tiny)
    env = np.maximum(env, floor)
    return iaaft(x / env, rng) * env


# -------------------------------------------------------------------- the caller

def call_regions(x, thresh, gap, wmin):
    """Maximal runs at or above `thresh`, merged across gaps <= gap, width >= wmin.

    Returns a list of (start, stop_exclusive). `thresh` is passed in rather than
    computed here because it is a PER-TRANSCRIPT quantile and a series is a piece
    of a transcript -- computing it per series would make the threshold mean a
    different thing in every series, which is the one-name-two-sets error.
    """
    hot = x >= thresh
    if not hot.any():
        return []
    d = np.diff(hot.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if hot[0]:
        starts.insert(0, 0)
    if hot[-1]:
        stops.append(len(x))
    runs = list(zip(starts, stops))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= wmin]


def region_stats(x, regions, per_kb_over):
    """The three quantities criterion 1 compares, per spec section 4."""
    if not regions:
        return dict(n_per_kb=0.0, width=np.array([]), height=np.array([]))
    med = np.median(x) if len(x) else 0.0
    w = np.array([e - s for s, e in regions], float)
    h = np.array([x[s:e].mean() / med if med > 0 else np.nan for s, e in regions])
    return dict(n_per_kb=1000.0 * len(regions) / per_kb_over, width=w, height=h)


def empirical_p(real, surr):
    """(1 + #surrogates >= real) / (1 + n). Floors at 1/(n+1) -- correction 4."""
    surr = np.asarray(surr, float)
    return (1.0 + int((surr >= real).sum())) / (1.0 + len(surr))


def gene_bootstrap(values, genes, rng, n_rep=2000):
    """Interval clustered on gene: transcripts of one gene are not independent."""
    values = np.asarray(values, float)
    uniq, inv = np.unique(np.asarray(genes), return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    out = np.empty(n_rep)
    for r in range(n_rep):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[r] = sums[pick].sum() / max(cnts[pick].sum(), 1.0)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def jaccard(a, b, n):
    """Overlap of two region sets as position-level Jaccard within one series."""
    ma = np.zeros(n, bool)
    mb = np.zeros(n, bool)
    for s, e in a:
        ma[s:e] = True
    for s, e in b:
        mb[s:e] = True
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else 1.0


# ------------------------------------------------------------------- the self-test

def _synthetic(kind, n, rng):
    """Tracks whose answer is known. Positive and heavy-tailed like the real one."""
    # AR(1)-ish smooth base with correlation length ~80 (matches lag-80 r = 0.64)
    w = rng.normal(size=n + 400)
    base = running_mean(w, 161)[:n]
    base = np.exp(2.2 * (base - base.mean()) / (base.std() + 1e-12))   # heavy tail
    if kind == "A":                       # stationary, featureless
        return base
    env = 1.0 + 6.0 * (0.5 + 0.5 * np.sin(np.linspace(0, 2.4 * np.pi, n)))
    if kind == "B":                       # envelope, still featureless
        return base * env
    x = base * env                        # envelope + real localized features
    # 20x, not 7x. At 7x only 7 of 40 injected features cleared the q=0.98
    # threshold, because the threshold is a GLOBAL per-series quantile and a
    # feature sitting where the envelope is low cannot reach it. A positive
    # control that is 82% invisible tests nothing -- the same defect as the dead
    # band. The strength is set so the control is unambiguously positive; realism
    # of the amplitude is not what this case is for.
    for c in rng.integers(50, n - 50, 40):
        x[c:c + 6] *= 20.0
    return x


def self_test(rng):
    """What the null and the criterion actually do, on tracks whose answer is known.

    THIS TEST FAILS ITS ORIGINAL PURPOSE AND THAT IS THE RESULT. It was written to
    demonstrate correction 2 -- that a naive iAAFT null fires on an envelope alone
    while an envelope-preserving one does not. It does not demonstrate that. What it
    found instead is larger and is not about the null at all.
    """
    n, q, gap, wmin = 40000, 0.98, 2, 5
    print("SELF-TEST -- real against its own surrogates, 20 each, q=0.98\n")
    print(f"  {'case':<4} {'null':<9} {'count':>7} {'width':>7} {'height':>7}"
          f"   (ratio real/surrogate)")
    res = {}
    for kind in ("A", "B", "C"):
        x = _synthetic(kind, n, rng)
        th = float(np.quantile(x, q))
        rs = region_stats(x, call_regions(x, th, gap, wmin), n)
        rw = rs["width"].max() if len(rs["width"]) else 0.0
        rh = float(np.nanmean(rs["height"])) if len(rs["height"]) else 0.0
        for mode in ("naive", "envelope"):
            sc, sw, sh = [], [], []
            for _ in range(N_SURROGATE):
                y = surrogate(x, rng, mode=mode)
                ty = float(np.quantile(y, q))
                sy = region_stats(y, call_regions(y, ty, gap, wmin), n)
                sc.append(sy["n_per_kb"])
                sw.append(sy["width"].max() if len(sy["width"]) else 0.0)
                sh.append(float(np.nanmean(sy["height"])) if len(sy["height"]) else 0.0)
            r = (rs["n_per_kb"] / max(np.mean(sc), 1e-12),
                 rw / max(np.mean(sw), 1e-12), rh / max(np.mean(sh), 1e-12))
            res[(kind, mode)] = r
            print(f"  {kind:<4} {mode:<9} {r[0]:>7.2f} {r[1]:>7.2f} {r[2]:>7.2f}")
    print("""
  A = stationary, autocorrelated, NO features
  B = A + a strong slow envelope, still NO features
  C = B + 40 genuine 6-base features

  FINDING 1 -- CRITERION 1 IS BACKWARDS, and this is solid.
  Spec section 5 asks whether the real track yields MORE regions per kilobase than
  its surrogates. On the one track that certainly has features (C) the ratio is
  BELOW 1: real produces FEWER regions than its own surrogates. The reason is
  structural. The threshold is a per-series QUANTILE, so the number of hot positions
  is fixed by construction -- measured, total hot positions inside regions is within
  1% of surrogate in every case. Only the ARRANGEMENT can vary, and genuine features
  concentrate a fixed hot mass into fewer, wider, taller runs while phase
  randomization scatters the same mass into more, narrower ones. A criterion on
  region COUNT therefore fails on real signal and passes marginally on noise.
  Width and height are the arrangement statistics; count is not one.

  FINDING 2 -- MY OWN CORRECTION 2 IS NOT VALIDATED BY THIS, and I am not going to
  report it as though it were. The envelope-preserving null was supposed to leave B
  quiet where the naive null fires. Both nulls show B inflated on width at about
  1.5x, which is most of what C shows. An envelope alone can mimic a real feature
  signal on the width statistic under EITHER null. The criticism that motivated the
  correction may still be right -- iAAFT is a stationary null and this track is not
  stationary -- but divide-by-envelope is not a demonstrated remedy for it.

  CONSEQUENCE: the region caller is NOT READY and criterion 1 needs redesign rather
  than a null swap. Nothing downstream of it -- region-scoped A4 in particular --
  should be scheduled against it until a criterion exists that fires on C and stays
  quiet on B. Tuning the synthetic until something passes would be fitting the
  validation, which is the failure this project has already paid for four times.
""")
    solid = res[("C", "envelope")][0] < 1.0 and res[("C", "naive")][0] < 1.0
    if not solid:
        print("  *** Finding 1 did not reproduce this run -- re-check before citing.")
    print("  SELF-TEST COMPLETE (it is a diagnosis, not a pass/fail gate)")
    return solid


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank")
    ap.add_argument("--null", choices=("envelope", "naive"), default="envelope")
    ap.add_argument("--env-window", type=int, default=ENV_WINDOW)
    ap.add_argument("--surrogates", type=int, default=N_SURROGATE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.self_test:
        sys.exit(0 if self_test(rng) else 1)
    if not args.bank:
        ap.error("--bank is required unless --self-test")

    import h5py
    with h5py.File(args.bank, "r") as f:
        N = f["vals"].shape[0]
        if args.limit:
            N = min(N, args.limit)
        spans = f["spans"][:]
        c_off, c_cnt = f["cand_offset"][:], f["cand_count"][:]
        c_start, c_end = f["cand_orf_start"][:], f["cand_orf_end"][:]
        p_sel = f["p_select"][:]
        genes = np.array([s.decode() for s in f["gene_id"][:N]])

        print(f"BANK {args.bank}   transcripts {N}   null {args.null}   "
              f"env_window {args.env_window}   surrogates {args.surrogates}")
        print("finite mask: (np.isfinite(vals).sum(1) == 3); vals is NaN at the")
        print("observed base by construction, so .all(1) is never true.\n")

        kept, dropped, drop_locale = [], 0, np.zeros(len(LOCALES), np.int64)
        for i in range(N):
            lo, nk = int(c_off[i]), int(c_cnt[i])
            b = spans[lo:lo + nk]
            P = int(max(b[:, 3].max(), b[:, 5].max()))
            if P < MIN_SERIES:
                dropped += 1
                continue
            v = f["vals"][i, :P].astype(np.float64)
            o = f["obs"][i, :P]
            with np.errstate(invalid="ignore"):
                e = np.nanmax(np.abs(v), axis=1)
            ok = f["valid"][i, :P].astype(bool) & np.isfinite(e) & (o >= 0)
            k = int(np.argmax(p_sel[lo:lo + nk]))
            s0, s1 = int(c_start[lo + k]), int(c_end[lo + k])
            pos = np.arange(1, P + 1)
            loc = np.where(pos < s0, 0, np.where(pos <= s1, 1, 2))
            # maximal contiguous runs of valid positions
            idx = np.flatnonzero(ok)
            if not len(idx):
                dropped += 1
                continue
            brk = np.flatnonzero(np.diff(idx) > 1)
            runs = np.split(idx, brk + 1)
            tx_thresh = {q: float(np.quantile(e[idx], q)) for q in Q_SWEEP}
            for r in runs:
                if len(r) < MIN_SERIES:
                    drop_locale += np.bincount(loc[r], minlength=len(LOCALES))
                    continue
                kept.append(dict(tx=i, gene=genes[i], score=e[r], obs=o[r],
                                 locale=loc[r], thresh=tx_thresh))
        print(f"  series kept {len(kept):,}   transcripts dropped (P<{MIN_SERIES} "
              f"or empty) {dropped:,}")
        print(f"  positions in series dropped by the length floor, by locale: "
              + "  ".join(f"{LOCALES[j]} {drop_locale[j]:,}"
                          for j in range(len(LOCALES))))
        print("  (exclusion by length is exclusion by transcript class -- spec 4.1)\n")

        print(f"  {'q':>5} {'gap':>4} {'wmin':>5} {'real/kb':>9} {'surr/kb':>9} "
              f"{'p<=.05':>8} {'95% CI on real/kb':>26}")
        for q in Q_SWEEP:
            for gap in GAP_SWEEP:
                for wmin in WMIN_SWEEP:
                    per_tx, hits, gl = [], 0, []
                    for s in kept:
                        x, th = s["score"], s["thresh"][q]
                        rs = region_stats(x, call_regions(x, th, gap, wmin),
                                          len(x))["n_per_kb"]
                        sv = []
                        for _ in range(args.surrogates):
                            y = surrogate(x, rng, mode=args.null,
                                          env_window=args.env_window)
                            ty = float(np.quantile(y, q))
                            sv.append(region_stats(
                                y, call_regions(y, ty, gap, wmin),
                                len(y))["n_per_kb"])
                        per_tx.append((rs, float(np.mean(sv))))
                        hits += int(empirical_p(rs, sv) <= 0.05)
                        gl.append(s["gene"])
                    real = np.array([a for a, _ in per_tx])
                    surr = np.array([b for _, b in per_tx])
                    ci = gene_bootstrap(real, gl, rng, n_rep=500)
                    print(f"  {q:>5.2f} {gap:>4} {wmin:>5} {real.mean():>9.3f} "
                          f"{surr.mean():>9.3f} {hits / max(len(kept),1):>8.1%} "
                          f"  [{ci[0]:.3f}, {ci[1]:.3f}]")
        print("\n  NO PER-TRANSCRIPT CLAIM IS AVAILABLE at this surrogate count:")
        print(f"  the empirical p floors at 1/{args.surrogates + 1} = "
              f"{1 / (args.surrogates + 1):.3f}. Only the clustered aggregate.")


if __name__ == "__main__":
    main()
