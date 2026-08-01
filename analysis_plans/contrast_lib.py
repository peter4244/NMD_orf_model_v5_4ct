#!/usr/bin/env python
"""
Shared machinery for stratified two-arm contrasts. DIAGNOSTIC, NOT PRESCRIPTIVE.

An earlier version of this file enforced a rule: it raised if no stratum had
enough members in both arms, and it silently imposed common-stratum weighting.
Pete's objection was that this substitutes a hard rule for contextual judgement,
and checking it settles the matter --

  THE GUARD WOULD NOT HAVE CAUGHT THE ERROR THAT MOTIVATED IT. At 3'UTR
  position +25, the arms were 541 and 214, and 7 strata still cleared 10 in
  both. `both.any()` is True, so it would have returned a number and raised
  nothing. Meanwhile it WOULD have fired on legitimate small-n contrasts, where
  the person hitting it lowers min_cell until it stops complaining.

What actually caught the error was noticing that n = 214 against n = 5,818 is
not a like-for-like comparison, and then power-matching to check. That is
judgement, and no assertion produces it.

So this file's job is to make the facts that judgement needs IMPOSSIBLE TO
MISS, and then get out of the way:

  * `std_diff` returns BOTH weightings side by side -- per-arm and
    common-stratum -- so a divergence between them is visible rather than
    latent. It never raises and never picks for you.
  * every result carries n per arm, the arm ratio, and how many strata each arm
    could support alone versus how many they share.
  * `sweep_null` exists to make the right thing cheap: building an empirical
    null from many mechanism-free contrasts rather than from three.
  * `power_match` subsamples a large contrast down to a small one's size, which
    is the check that decided the 2026-08-01 retraction.

When the two weightings agree, the choice does not matter. When they diverge,
that is information about the contrast, and the analyst should look at why --
not be handed a verdict.
"""

import numpy as np
import pandas as pd

__all__ = ["code_strata", "std_diff", "boot_diff", "describe", "power_match",
           "sweep_null"]


def code_strata(df, cols):
    """Integer-code the joint strata defined by `cols`. Returns (codes, n_strata)."""
    key = df[cols].astype(str).agg("|".join, axis=1)
    codes, uniq = pd.factorize(key)
    if (codes < 0).any():
        raise ValueError("missing values in stratifying columns; drop them first")
    return codes.astype(np.int64), len(uniq)


def std_diff(kcode, arm, y, n_strata, min_cell=10):
    """Direct-standardised difference in mean(y), arm 1 minus arm 0.

    Computes the contrast BOTH ways and returns both, plus the diagnostics that
    tell you whether the choice matters here:

      diff_per_arm    each arm averaged over whatever strata it can support
      diff_common     both arms averaged over the strata they share

    They coincide when both arms are large enough to populate the same strata.
    They diverge when one arm is thin -- and the per-arm version is biased
    upward there, by 12 points at n = 500/250 on this data. Neither is "the"
    answer; which one is right depends on what you are estimating.
    """
    kcode = np.asarray(kcode)
    arm = np.asarray(arm).astype(np.int64)
    y = np.asarray(y).astype(np.float64)
    c1 = np.bincount(kcode[arm == 1], minlength=n_strata).astype(np.float64)
    c0 = np.bincount(kcode[arm == 0], minlength=n_strata).astype(np.float64)
    n_k = np.bincount(kcode, minlength=n_strata).astype(np.float64)
    g1, g0 = c1 >= min_cell, c0 >= min_cell
    both = g1 & g0
    rate1 = np.divide(np.bincount(kcode[arm == 1], weights=y[arm == 1],
                                  minlength=n_strata), c1,
                      out=np.zeros(n_strata), where=c1 > 0)
    rate0 = np.divide(np.bincount(kcode[arm == 0], weights=y[arm == 0],
                                  minlength=n_strata), c0,
                      out=np.zeros(n_strata), where=c0 > 0)

    def avg(rate, good):
        w = n_k * good
        return float((w * rate).sum() / w.sum()) if w.sum() > 0 else np.nan

    n1, n0 = int((arm == 1).sum()), int((arm == 0).sum())
    per_arm = (avg(rate1, g1) - avg(rate0, g0)) * 100
    common = (avg(rate1, both) - avg(rate0, both)) * 100

    # Mantel-Haenszel odds ratio over the shared strata, reported ALONGSIDE the
    # percentage-point difference rather than instead of it. Track A's point,
    # 2026-08-01: pp differences compress as the base rate approaches 0 or 1, so
    # a constant multiplicative effect produces a large pp gradient across
    # strata with different base rates. Two groups at 72% and 6% baseline with
    # the SAME odds ratio give +4.5pp and +1.3pp. Quoting the pp ratio as
    # evidence of a mechanism, without naming the scale, is therefore unsafe --
    # which is exactly what I did with PTC+ vs PTC-.
    a = np.bincount(kcode[arm == 1], weights=y[arm == 1], minlength=n_strata)
    b = c1 - a
    c = np.bincount(kcode[arm == 0], weights=y[arm == 0], minlength=n_strata)
    dd = c0 - c
    tot = c1 + c0
    ok = both & (tot > 0)
    num = float((a[ok] * dd[ok] / tot[ok]).sum())
    den = float((b[ok] * c[ok] / tot[ok]).sum())
    mh_or = num / den if den > 0 else np.nan

    return dict(diff_per_arm=per_arm, diff_common=common,
                rate1=avg(rate1, both) * 100, rate0=avg(rate0, both) * 100,
                base_rate=float(y[np.isin(kcode, np.flatnonzero(both))].mean() * 100)
                if both.any() else np.nan,
                mh_or=mh_or,
                n1=n1, n0=n0, arm_ratio=max(n1, n0) / max(min(n1, n0), 1),
                strata_arm1=int(g1.sum()), strata_arm0=int(g0.sum()),
                strata_shared=int(both.sum()), n_strata=n_strata)


def describe(res, label="", indent="    "):
    """Print a contrast with the facts judgement needs, and no verdict."""
    if label:
        print(f"{indent}{label}")
    d1, d0 = res["diff_per_arm"], res["diff_common"]
    print(f"{indent}  n {res['n1']:,} vs {res['n0']:,}  "
          f"(ratio {res['arm_ratio']:.1f}x)   strata "
          f"{res['strata_arm1']}/{res['strata_arm0']} alone, "
          f"{res['strata_shared']} shared of {res['n_strata']}")
    gap = abs(d1 - d0) if not (np.isnan(d1) or np.isnan(d0)) else np.nan
    note = ""
    if not np.isnan(gap):
        note = ("  <- weightings agree" if gap < 0.25
                else f"  <- weightings differ by {gap:.2f}pp; look at why")
    print(f"{indent}  per-arm {d1:+7.2f}pp    common-stratum {d0:+7.2f}pp{note}")
    # base rate and odds ratio always travel with the pp difference, because a
    # pp difference cannot be compared ACROSS groups whose base rates differ
    print(f"{indent}  rates {res['rate1']:.1f}% vs {res['rate0']:.1f}%   "
          f"base rate {res['base_rate']:.1f}%   MH odds ratio {res['mh_or']:.2f}")
    if "lo" in res and not np.isnan(res.get("lo", np.nan)):
        print(f"{indent}  95% CI [{res['lo']:+.2f}, {res['hi']:+.2f}]  "
              f"p = {res['p']:.3f}  ({res['n_draws']} gene-clustered draws)")
    if "or_lo" in res and not np.isnan(res.get("or_lo", np.nan)):
        print(f"{indent}  odds ratio 95% CI [{res['or_lo']:.2f}, {res['or_hi']:.2f}]")


def boot_diff(df, arm_col, a, b, strata_cols, label, gene_col,
              n=2000, min_cell=10, use="common", rng=None):
    """Gene-clustered bootstrap. `use` picks which weighting gets the interval;
    both point estimates come back either way."""
    rng = rng or np.random.default_rng(20260801)
    key = f"diff_{use}"
    d = df[df[arm_col].isin([a, b])].dropna(
        subset=list(strata_cols) + [label, gene_col, arm_col]).copy()
    if not len(d):
        raise ValueError("no rows after dropping missing values")
    kcode, nk = code_strata(d, list(strata_cols))
    gcode, guniq = pd.factorize(d[gene_col])
    arm = d[arm_col].eq(a).to_numpy().astype(np.int64)
    y = d[label].to_numpy().astype(np.float64)
    order = np.argsort(gcode, kind="stable")
    gs = gcode[order]
    st = np.searchsorted(gs, np.arange(len(guniq)), side="left")
    en = np.searchsorted(gs, np.arange(len(guniq)), side="right")
    rows_by_gene = [order[s:e] for s, e in zip(st, en)]

    res = std_diff(kcode, arm, y, nk, min_cell=min_cell)
    draws = np.empty(n)
    ors = np.empty(n)
    ng = len(guniq)
    for t in range(n):
        r = np.concatenate([rows_by_gene[g] for g in rng.integers(0, ng, ng)])
        s = std_diff(kcode[r], arm[r], y[r], nk, min_cell=min_cell)
        draws[t], ors[t] = s[key], s["mh_or"]
    ok_or = ors[np.isfinite(ors) & (ors > 0)]
    draws = draws[~np.isnan(draws)]
    if len(draws) < 20:
        res.update(lo=np.nan, hi=np.nan, p=np.nan, n_draws=len(draws))
        return res
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    res.update(lo=lo, hi=hi, p=max(p, 1.0 / len(draws)),
               n_draws=len(draws), interval_on=use, or_draws=ok_or)
    if len(ok_or) >= 20:
        res["or_lo"], res["or_hi"] = np.percentile(ok_or, [2.5, 97.5])
    return res


def power_match(df, arm_col, a, b, strata_cols, label, n1, n0,
                reps=500, min_cell=10, use="per_arm", rng=None):
    """Subsample a contrast to (n1, n0) and return the distribution of estimates.

    This is the check that mattered: run the TARGET contrast at the CONTROL's
    sample size and see what the estimator returns when the answer is known.
    """
    rng = rng or np.random.default_rng(20260801)
    key = f"diff_{use}"
    d = df[df[arm_col].isin([a, b])].dropna(
        subset=list(strata_cols) + [label, arm_col]).copy()
    kcode, nk = code_strata(d, list(strata_cols))
    arm = d[arm_col].eq(a).to_numpy().astype(np.int64)
    y = d[label].to_numpy().astype(np.float64)
    i1, i0 = np.flatnonzero(arm == 1), np.flatnonzero(arm == 0)
    if len(i1) < n1 or len(i0) < n0:
        raise ValueError(f"cannot subsample to {n1}/{n0} from {len(i1)}/{len(i0)}")
    out = np.empty(reps)
    for t in range(reps):
        pick = np.concatenate([rng.choice(i1, n1, replace=False),
                               rng.choice(i0, n0, replace=False)])
        out[t] = std_diff(kcode[pick], arm[pick], y[pick], nk,
                          min_cell=min_cell)[key]
    return out[~np.isnan(out)]


def sweep_null(make_arm, positions, df, strata_cols, label,
               min_cell=10, min_per_arm=30):
    """Empirical null from many mechanism-free contrasts.

    Three estimates are not a distribution. This is here so that building a
    proper one costs a single call.

    `make_arm(pos)` returns a Series aligned to `df`: True for arm 1, False for
    arm 0, NaN to exclude.
    """
    kcode_all, nk = code_strata(df, list(strata_cols))
    y_all = df[label].to_numpy().astype(np.float64)
    out = []
    for pos in positions:
        s = make_arm(pos)
        m = s.notna().to_numpy()
        if m.sum() == 0:
            continue
        arm = s[m].astype(bool).to_numpy().astype(np.int64)
        if min(int((arm == 1).sum()), int((arm == 0).sum())) < min_per_arm:
            continue
        r = std_diff(kcode_all[m], arm, y_all[m], nk, min_cell=min_cell)
        r["pos"] = pos
        out.append(r)
    return pd.DataFrame(out)
