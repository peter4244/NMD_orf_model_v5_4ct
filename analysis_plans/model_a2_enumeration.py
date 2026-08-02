"""
model_a2_enumeration.py — the enumeration SEQ-A2 is blocked on.

Namespaced `model_*` per ANALYSIS_SEQUENCING_PROPOSAL.md: two windows once wrote
`autocorr.py` to the same cluster directory and the second silently replaced the
first.

Answers two questions and nothing else. Both are descriptive counting on a bank
that already exists. No new forward passes, no adjustment, no null.

--------------------------------------------------------------------------------
QUESTION 1 — the dead fraction, so three provisional parameters can be frozen.

The A2 specification fixes 8 mass bands, top 10% within cell, and a >=100 valid
floor. All three were derived from "mean 2,213 VALID positions per transcript, so
~277 per cell at 8 bands, ~28 elevated". But the same specification puts dead
positions (mass < DEAD_CUT) in their own band, and builds the eight quantile bands
over LIVE positions only -- a smaller set. So all three numbers were chosen against
a denominator the design removes. That is the enumerate-what-you-divided-by error
in the specification's own arithmetic.

Reported as a DISTRIBUTION, not a mean. Section 5.5 puts the dead-heavy tail in
transcripts with long 5'UTRs, which is the mechanism cell, so a >=100 floor that
bites in the tail bites differentially. A mean cannot show that.

--------------------------------------------------------------------------------
QUESTION 2 — which set produced the unstratified keto ratio.

Three values are live in the repository for one statistic:

  1.16x   HANDOFF_2026-08-01_night_banks.md:751  (elevated .581, background .501)
  1.148x  FINDINGS_DECAY_SEQUENCE_2026-08-02.md:29 and the A2 row
  1.15x   analysis_pwm_fit.py:46

Both reconstructable ones are internally consistent -- the backgrounds sum to
1.000 in each -- so this is not rounding and not a transcription slip. Same
statistic, different set.

Reading the two producers narrows it to three knobs, and this script varies
exactly those three and nothing else:

  transcripts   probe_elevated_composition_profile.py iterates range(600).
                analysis_pwm_fit.py iterates every transcript in the arm filter.
  edge trim     the profile probe restricts to W <= idx < P-W at W=60, dropping
                the first and last 60 positions of EVERY transcript. The FINDINGS
                text describes its background as "all valid positions of the same
                transcripts", which is not what the code computes.
  dead handling dead positions cannot be elevated -- max_b|vals_decay| is at the
                floor -- but they are valid, so including them moves the
                BACKGROUND without touching the elevated set.

A fourth is reported rather than varied: pooled-by-count against
per-transcript-then-averaged (axis 8). Both producers pool by count.

--------------------------------------------------------------------------------
Every table carries its enumeration -- n transcripts, n valid, n live, n elevated,
seed, dead handling, edge trim, and the finite-mask expression. Never the ratio
alone. That is field 13 of the row template, and its absence is why three values
for one statistic survived two windows and two days.

Run from the repo root: paths are relative to it.
"""

import argparse
import numpy as np
import h5py

NT = "ACGT"
DEAD_CUT = 1e-8            # section 5.5: every dead perturbation sits below this


def transcript_length(spans, lo, nk):
    """Positions covered by this transcript's candidate windows.

    spans columns are (n, k, atg_lo, atg_hi, stop_lo, stop_hi); the covered
    extent is the larger of the two window highs. Both existing producers
    compute P this way and it is reproduced rather than reinvented.
    """
    b = spans[lo:lo + nk]
    return int(max(b[:, 3].max(), b[:, 5].max()))


def load_transcript(f, i, spans, cand_off, cand_cnt):
    """Per-transcript slice. h5py is lazy and [:] is what defeats it."""
    lo, nk = int(cand_off[i]), int(cand_cnt[i])
    P = transcript_length(spans, lo, nk)
    if P < 50:
        return None
    v = f["vals_decay"][i, :P].astype(np.float64)
    o = f["obs"][i, :P]
    m = f["mass"][i, :P].astype(np.float64)
    valid = f["valid"][i, :P].astype(bool)
    with np.errstate(invalid="ignore"):
        e = np.nanmax(np.abs(v), axis=1)
    # vals is NaN at the observed base BY CONSTRUCTION, so isfinite().all(1) is
    # never true. Three scripts across two windows hit this. n_finite is carried
    # so the enumeration can report how many rows are not exactly 3.
    n_finite = np.isfinite(v).sum(1)
    ok = valid & np.isfinite(e) & (o >= 0)
    return dict(P=P, e=e, o=o, mass=m, ok=ok, n_finite=n_finite)


# ---------------------------------------------------------------- question 1

def dead_fraction_distribution(f, N, spans, cand_off, cand_cnt):
    """Per-transcript dead fraction, and the live counts the bands will use."""
    rows = []
    for i in range(N):
        t = load_transcript(f, i, spans, cand_off, cand_cnt)
        if t is None:
            continue
        idx = np.flatnonzero(t["ok"])
        if not len(idx):
            continue
        live = t["mass"][idx] >= DEAD_CUT
        rows.append((len(idx), int(live.sum()), int((~live).sum()),
                     int((t["n_finite"][idx] != 3).sum())))
    return np.array(rows, dtype=np.int64)


def band_cell_report(f, N, spans, cand_off, cand_cnt, band_counts, floor):
    """Global log-mass quantile bands over LIVE positions; per-cell live counts.

    Global rather than per-transcript so a band means the same thing in every
    transcript, per the specification. Two passes: the edges cannot be known
    until every live mass is seen.
    """
    pool = []
    per_tx = []
    for i in range(N):
        t = load_transcript(f, i, spans, cand_off, cand_cnt)
        if t is None:
            continue
        idx = np.flatnonzero(t["ok"])
        if not len(idx):
            continue
        m = t["mass"][idx]
        lm = np.log10(m[m >= DEAD_CUT])       # log(0) is why the dead cut is a
        pool.append(lm.astype(np.float32))    # hard threshold and not a quantile
        per_tx.append(lm.astype(np.float32))
    allm = np.concatenate(pool)

    out = {}
    for nb in band_counts:
        edges = np.quantile(allm, np.linspace(0, 1, nb + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        sizes = []
        for lm in per_tx:
            b = np.digitize(lm, edges[1:-1])
            sizes.append(np.bincount(b, minlength=nb))
        sizes = np.array(sizes)               # (transcripts, bands)
        out[nb] = dict(
            edges=edges,
            sizes=sizes,
            qualifying=int((sizes >= floor).sum()),
            total=int(sizes.size),
            per_band_qualifying=(sizes >= floor).sum(0),
        )
    return out


# ---------------------------------------------------------------- question 2

def composition(f, N, spans, cand_off, cand_cnt, *, n_tx, trim, drop_dead_bg,
                top_frac=0.01, min_valid=200):
    """Elevated-versus-background base composition under one set definition.

    Only three things vary across calls -- transcript cap, edge trim, and whether
    dead positions are in the background. Everything else is held identical so a
    difference between two rows is attributable to the knob that moved.
    """
    hi = np.zeros(4)
    bg = np.zeros(4)
    per_tx_hi = []
    n_used = n_valid = n_elev = n_live_bg = 0
    for i in range(min(N, n_tx)):
        t = load_transcript(f, i, spans, cand_off, cand_cnt)
        if t is None or t["P"] < 2 * trim + 50:
            continue
        idx = np.flatnonzero(t["ok"])
        if trim:
            idx = idx[(idx >= trim) & (idx < t["P"] - trim)]
        if len(idx) < min_valid:
            continue
        bg_idx = idx
        if drop_dead_bg:
            bg_idx = idx[t["mass"][idx] >= DEAD_CUT]
            if not len(bg_idx):
                continue
        k = max(1, int(round(top_frac * len(idx))))
        cut = np.partition(t["e"][idx], -k)[-k]
        sel = idx[t["e"][idx] >= cut]

        counts = np.array([(t["o"][sel] == b).sum() for b in range(4)], float)
        hi += counts
        per_tx_hi.append(counts / counts.sum())
        for b in range(4):
            bg[b] += (t["o"][bg_idx] == b).sum()
        n_used += 1
        n_valid += len(idx)
        n_live_bg += len(bg_idx)
        n_elev += len(sel)

    p = hi / hi.sum()
    q = bg / bg.sum()
    p_tx = np.mean(per_tx_hi, axis=0)          # axis 8, reported beside pooled
    return dict(
        p=p, q=q, p_tx=p_tx,
        keto=(p[2] + p[3]) / (q[2] + q[3]),
        amino=(p[0] + p[1]) / (q[0] + q[1]),
        gc=(p[1] + p[2]) / (q[1] + q[2]),
        keto_tx=(p_tx[2] + p_tx[3]) / (q[2] + q[3]),
        n_tx=n_used, n_valid=n_valid, n_live_bg=n_live_bg, n_elev=n_elev,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
    ap.add_argument("--floor", type=int, default=100)
    args = ap.parse_args()

    with h5py.File(args.bank, "r") as f:
        spans = f["spans"][:]
        cand_off = f["cand_offset"][:]
        cand_cnt = f["cand_count"][:]
        N = len(f["transcript_id"])

        print(f"BANK {args.bank}   transcripts {N}   dead cut {DEAD_CUT:g}")
        print("finite mask: (np.isfinite(vals_decay).sum(1) == 3); vals is NaN at")
        print("the observed base by construction, so .all(1) is never true.\n")

        # ---- question 1a
        rows = dead_fraction_distribution(f, N, spans, cand_off, cand_cnt)
        valid_n, live_n, dead_n, odd_finite = rows.T
        frac = dead_n / valid_n
        print("=" * 78)
        print("Q1a  DEAD FRACTION PER TRANSCRIPT -- distribution, not a mean")
        print("=" * 78)
        print(f"  transcripts contributing        {len(rows)}")
        print(f"  valid positions, total          {valid_n.sum():,}")
        print(f"  live  positions, total          {live_n.sum():,}")
        print(f"  dead  positions, total          {dead_n.sum():,}"
              f"   ({dead_n.sum()/valid_n.sum():.1%} of valid)")
        print(f"  rows whose finite count != 3    {odd_finite.sum():,}")
        print()
        print(f"  {'decile':>8} {'dead frac':>10} {'valid/tx':>10} {'live/tx':>10}")
        for d in range(0, 101, 10):
            print(f"  {d:>7}% {np.percentile(frac, d):>10.3f}"
                  f" {np.percentile(valid_n, d):>10.0f}"
                  f" {np.percentile(live_n, d):>10.0f}")
        print()
        print(f"  MEAN valid per transcript  {valid_n.mean():>8.0f}"
              f"   (the specification's 2,213)")
        print(f"  MEAN live  per transcript  {live_n.mean():>8.0f}"
              f"   <- the denominator the bands actually use")

        # ---- question 1b
        bands = band_cell_report(f, N, spans, cand_off, cand_cnt,
                                 (4, 8, 16), args.floor)
        print()
        print("=" * 78)
        print(f"Q1b  LIVE-BANDED CELL SIZES -- qualifying at >={args.floor} live")
        print("=" * 78)
        for nb in (4, 8, 16):
            r = bands[nb]
            s = r["sizes"]
            print(f"\n  {nb} bands   cells {r['total']:,}"
                  f"   qualifying {r['qualifying']:,}"
                  f"  ({r['qualifying']/r['total']:.1%})")
            print(f"    per-cell live count, deciles: "
                  + " ".join(f"{np.percentile(s, d):.0f}" for d in range(0, 101, 20)))
            print("    qualifying cells by band: "
                  + " ".join(str(int(x)) for x in r["per_band_qualifying"]))
            for tf in (0.05, 0.10, 0.20):
                med = np.median(s[s >= args.floor]) * tf
                print(f"    at top {tf:>4.0%}: median elevated per qualifying cell"
                      f" ~{med:.0f}")

        # ---- question 2
        print()
        print("=" * 78)
        print("Q2  COMPOSITION BY SET DEFINITION -- reconciling 1.16 / 1.148 / 1.15")
        print("=" * 78)
        grid = [
            ("profile probe as written", dict(n_tx=600, trim=60, drop_dead_bg=False)),
            ("  ...all transcripts",     dict(n_tx=N,   trim=60, drop_dead_bg=False)),
            ("  ...no edge trim",        dict(n_tx=600, trim=0,  drop_dead_bg=False)),
            ("  ...live background",     dict(n_tx=600, trim=60, drop_dead_bg=True)),
            ("all tx, no trim",          dict(n_tx=N,   trim=0,  drop_dead_bg=False)),
            ("all tx, no trim, live bg", dict(n_tx=N,   trim=0,  drop_dead_bg=True)),
        ]
        print(f"\n  {'set':<26} {'keto':>7} {'amino':>7} {'GC':>7}"
              f" {'keto/tx':>8} {'n_tx':>6} {'n_elev':>9} {'n_valid':>11}")
        for label, kw in grid:
            r = composition(f, N, spans, cand_off, cand_cnt, **kw)
            print(f"  {label:<26} {r['keto']:>7.3f} {r['amino']:>7.3f}"
                  f" {r['gc']:>7.3f} {r['keto_tx']:>8.3f} {r['n_tx']:>6}"
                  f" {r['n_elev']:>9,} {r['n_valid']:>11,}")
            print(f"    {'':<24} elevated "
                  + " ".join(f"{NT[j]} {r['p'][j]:.3f}" for j in range(4))
                  + "   background "
                  + " ".join(f"{NT[j]} {r['q'][j]:.3f}" for j in range(4)))

        print()
        print("  NOTE: keto background and GC background are the SAME NUMBER")
        print("  whenever T == C, since G+T == G+C. Two quantities, one value --")
        print("  reconciling by eye matches the wrong pair.")


if __name__ == "__main__":
    main()
