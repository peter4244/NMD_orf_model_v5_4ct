#!/usr/bin/env python3
"""collect_sweep.py — turn the sweep's cells into the numbers the selection decision needs.

WHAT THIS IS FOR. D34: the configuration is selected AFTER the sweep runs, on validation
performance, because the configurations are expected to be close enough that no winner is
resolvable in advance. That ruling makes this script load-bearing rather than descriptive -- it is
the thing the selection reads -- so its statistics have to be right.

THE QUESTION IT ANSWERS is not "which config has the highest mean" but "can this sweep tell its
top configs apart at all". Historically it could not: the published grid's rank 1 and rank 2
differed by 0.00139 AUC while seed-to-seed sd is ~0.00248, i.e. the gap is smaller than the noise
nobody had measured. When the answer is "no", the honest output is a SET, and whatever tie-break
is applied is what selects -- which is a thing to know before quoting a winner, not after.

TWO DEFECTS IN THE PREVIOUS VERSION. The first unambiguously manufactured winners; the second's
DIRECTION depends on what you compare it against, which the earlier write-up got half right --
see the note under (2).

  1. THE LEADER WAS CHOSEN WITHOUT REQUIRING COMPLETE SEEDS. `best = rows[0]` after sorting, with
     no n filter. Demonstrated 2026-07-30 on synthetic cells: one configuration with a single
     lucky seed became the leader, all eleven fully-sampled configurations then reported "CANNOT
     ASSESS", and the metric-choice block named the 1-seed config as winner under both metrics
     while printing "Both metrics agree". The sweep is 60 INDIVIDUAL jobs by design (arrays are
     penalised on this cluster), so a missing cell is the expected case, not an edge case.

  2. THE INDISTINGUISHABILITY TEST WAS A HEURISTIC, NOT A TEST. It called two configs
     indistinguishable when `gap <= max(sd_a, sd_b)`, comparing a difference of MEANS to the sd of
     SINGLE RUNS -- which is neither a paired nor an unpaired standard error, and is insensitive
     to n entirely.

     ITS BIAS HAS NO SINGLE DIRECTION, and the handoff's "the reported tie sets are too small"
     holds only against the UNPAIRED comparison. Measured on synthetic null data with realistic
     paired structure (12 configs x 5 shared seeds, across-config seed correlation +0.953):

         old heuristic     gap <= max single-run sd   = 0.00312
         correct PAIRED    gap  > t*sigma_d/sqrt(n)   = 0.00168   <- the right test here
         correct UNPAIRED  gap  > t*sd*sqrt(2/n)      = 0.00538

     So the old threshold sits BETWEEN the two. Against an unpaired test it excludes too readily
     (sets too small, as recorded). Against the PAIRED test -- which is the correct analysis,
     because every configuration is trained at the same seed set -- it is too permissive and
     declares ties that a proper test excludes, making sets too LARGE. Both readings are of the
     same number; the heuristic is simply not a test, and quoting a direction for its bias without
     naming the comparison is itself the error.

WHAT REPLACES THEM. Seeds are paired across configurations -- every config is trained at the same
seed set -- so the comparison against the leader is a PAIRED one, and pairing removes the shared
seed effect that dominates the raw spread. Per-seed differences against the leader are formed, the
difference variance is POOLED across all contrasts (df = n_contrasts x (n-1), e.g. 11 x 4 = 44
rather than 4), and a configuration is excluded only when its gap exceeds the Bonferroni-corrected
paired critical value. The surviving set is a simultaneous 95% set of configurations that cannot be
excluded as best.

    python3 collect_sweep.py
    python3 collect_sweep.py --results-dir results_4ct_sweep --split val_clean
    python3 collect_sweep.py --alpha 0.05 --metric auc
"""
import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats


def load_cells(rd, split):
    pat = re.compile(rf"^metrics_(atg\d+_stop\d+)_seed(\d+)_{re.escape(split)}\.json$")
    cells = {}
    for f in sorted(Path(rd).glob(f"metrics_*_seed*_{split}.json")):
        m = pat.match(f.name)
        if not m:
            continue
        cells.setdefault(m.group(1), {})[int(m.group(2))] = json.loads(f.read_text())
    return cells


def paired_analysis(cells, tags, seeds, key, alpha):
    """Simultaneous set of configurations that cannot be excluded as best.

    Returns (leader, per-tag dict of {gap, excluded}, diagnostics).
    """
    n = len(seeds)
    vals = {t: np.array([cells[t][s][key] for s in seeds], dtype=float) for t in tags}
    means = {t: vals[t].mean() for t in tags}
    leader = max(tags, key=lambda t: means[t])

    contrasts = [t for t in tags if t != leader]
    if not contrasts or n < 2:
        return leader, {t: {"gap": means[leader] - means[t], "excluded": False} for t in contrasts}, {}

    # Per-seed paired differences against the leader, then POOL their variance across contrasts.
    per_sd, ss, naive_df = {}, 0.0, 0
    for t in contrasts:
        d = vals[leader] - vals[t]
        per_sd[t] = float(d.std(ddof=1))
        ss += float(((d - d.mean()) ** 2).sum())
        naive_df += n - 1
    k = len(contrasts)
    sigma_d = math.sqrt(ss / naive_df) if naive_df > 0 else float("nan")

    # THE POOLED SS IS UNBIASED BUT IS NOT CHI-SQUARE ON k(n-1) (corrected 2026-07-30, review).
    # Every difference vector d_t = X_leader - X_t contains the SAME leader residuals, so the k
    # contrasts are correlated: with equal variances Var(d)=2s^2 and Cov(d_t,d_u)=s^2, i.e. rho=1/2
    # between difference vectors and rho^2=1/4 between their squared deviations. Moment-matching
    # the sum of k correlated chi-square(n-1) terms:
    #
    #   E[SS]  = k(n-1)Var(d)
    #   Var[SS]= 2(n-1)Var(d)^2 [ k + k(k-1)/4 ]
    #   nu     = 2E[SS]^2/Var[SS] = k(n-1) / (1 + (k-1)/4) = 4k(n-1)/(k+3)
    #
    # At k=15, n=5 that is 13.3, not 60. Verified by simulation of this exact procedure under a
    # global null (20,000 reps): E[SS/Var(d)] = 59.8 -- unbiased, as derived -- but Var = 535
    # against 120 for chi-square_60, giving an effective df of 13.4. Pooling buys at most ~4x the
    # single-contrast df, never k times it.
    #
    # Consequence of having used 60: t_crit was 2.81 instead of 3.21, so the AUC threshold was
    # 0.00421 instead of 0.00480 and at least one configuration was reported excluded when it is
    # not. Coverage of the overall procedure happened to look right at ~0.95 only because this
    # error and the looseness of the Bonferroni union bound ran in opposite directions.
    df = 4 * k * (n - 1) / (k + 3) if k > 0 else float("nan")
    t_crit = float(stats.t.ppf(1 - alpha / k, df))          # Bonferroni over the k contrasts
    thresh = t_crit * sigma_d / math.sqrt(n)

    res = {}
    for t in contrasts:
        gap = means[leader] - means[t]
        res[t] = {"gap": gap, "excluded": bool(gap > thresh)}

    # Diagnostics that make the result interpretable rather than just reported.
    M = np.array([vals[t] for t in tags])                    # (n_configs, n_seeds)
    cc = np.corrcoef(M)
    off = cc[np.triu_indices_from(cc, k=1)]
    mdd80 = (t_crit + float(stats.t.ppf(0.80, df))) * sigma_d / math.sqrt(n)
    srt = sorted(means.values(), reverse=True)
    # mdd80 WAS COMPUTED AND THEN NEVER REPORTED. It was absent from `diag`, and the print guarded
    # itself with `if "mdd80" in diag` -- so the line emitted an empty string on every run since it
    # was written, and "this design's resolution" was quoted from the ALPHA threshold instead. The
    # two differ by ~30%: the honest statement of what a non-excluded configuration might cost is
    # the MDD, not the exclusion threshold.
    # HETEROSCEDASTICITY IS REPORTED, NOT ASSUMED AWAY. Pooling one sigma_d across contrasts
    # assumes Var(X_leader - X_t) is constant in t. When it is not, the configurations with the
    # TIGHTEST contrasts are over-retained -- they get a threshold wider than their own variance
    # warrants -- and those are exactly the configurations a cost-based tie-break then favours.
    sds = sorted(per_sd.values())
    diag = {
        "n": n, "k": k, "df": df, "naive_df_would_be": naive_df,
        "sigma_d": sigma_d, "t_crit": t_crit, "thresh": thresh,
        "mdd80": mdd80,
        "per_contrast_sd_min": sds[0] if sds else float("nan"),
        "per_contrast_sd_max": sds[-1] if sds else float("nan"),
        "per_contrast_sd_ratio": (sds[-1] / sds[0]) if sds and sds[0] > 0 else float("nan"),
        "mean_corr": float(np.nanmean(off)) if off.size else float("nan"),
        "top2_spacing": (srt[0] - srt[1]) if len(srt) > 1 else float("nan"),
        "single_run_sd_median": st.median([float(v.std(ddof=1)) for v in vals.values()]),
    }
    return leader, res, diag


def report(metric, key, cells, tags, seeds, alpha):
    leader, res, diag = paired_analysis(cells, tags, seeds, key, alpha)
    means = {t: st.mean([cells[t][s][key] for s in seeds]) for t in tags}
    sds = {t: st.stdev([cells[t][s][key] for s in seeds]) for t in tags}

    print(f"=== {metric}: paired, pooled-variance, simultaneous comparison ===")
    print(f"  {'config':22}{'mean':>10}{'sd':>10}{'gap':>10}   verdict")
    for t in sorted(tags, key=lambda x: -means[x]):
        if t == leader:
            print(f"  {t:22}{means[t]:>10.5f}{sds[t]:>10.5f}{'':>10}   leader")
            continue
        r = res[t]
        # "excluded" invited reading each row as a valid pairwise test that t is WORSE than the
        # leader. It is not: the procedure controls only the simultaneous coverage of the best, so
        # the only defensible per-row statement is set membership.
        verdict = "not in set" if r["excluded"] else "IN SET (cannot exclude as best)"
        print(f"  {t:22}{means[t]:>10.5f}{sds[t]:>10.5f}{r['gap']:>10.5f}   {verdict}")

    survivors = [leader] + [t for t in tags if t != leader and not res[t]["excluded"]]
    if diag:
        print(f"\n  paired sd of differences  {diag['sigma_d']:.5f}  (df {diag['df']}, pooled over "
              f"{diag['k']} contrasts)")
        print(f"  median single-run sd      {diag['single_run_sd_median']:.5f}  <- what the OLD "
              f"heuristic compared gaps against")
        print(f"  across-config seed corr   {diag['mean_corr']:+.3f}  (high = pairing is buying a lot)")
        print(f"  exclusion threshold       {diag['thresh']:.5f}  "
              f"(t={diag['t_crit']:.3f}, alpha={alpha}/{diag['k']}, n={diag['n']})")
        print(f"  min detectable diff @80%  {diag['mdd80']:.5f}  <- what a non-excluded config "
              f"could actually be worse by")
        print(f"  per-contrast sd spread    {diag['per_contrast_sd_min']:.5f}-"
              f"{diag['per_contrast_sd_max']:.5f}  ratio {diag['per_contrast_sd_ratio']:.2f}"
              + ("   *** pooling favours the tightest contrasts"
                 if diag['per_contrast_sd_ratio'] > 2 else ""))
        print(f"  effective df              {diag['df']:.1f}  (k(n-1) would be "
              f"{diag['naive_df_would_be']}; contrasts share the leader)")
        print(f"  observed top-two spacing  {diag['top2_spacing']:.5f}")

    print()
    if len(survivors) == 1:
        print(f"  RESOLVED by {metric}: {leader} excludes every other configuration.")
    else:
        print(f"  *** NOT RESOLVED by {metric}. {len(survivors)} configurations form a "
              f"simultaneous {int((1-alpha)*100)}% set")
        print(f"      that cannot be excluded as best: {', '.join(sorted(survivors))}")
        print(f"      Whatever tie-break is applied is what selects the configuration. Say so.")
    print()
    return leader, survivors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--split", default="val_clean")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    cells = load_cells(a.results_dir, a.split)
    if not cells:
        print(f"no cells found in {a.results_dir} for split={a.split}")
        print(f"  expected: metrics_atg<A>_stop<B>_seed<S>_{a.split}.json")
        return 1

    # ---- COMPLETENESS, BEFORE ANY RANKING -----------------------------------------------------
    # Both checks refuse rather than warn. A ranking computed over configurations trained at
    # DIFFERENT seed sets is not paired, and the whole analysis below assumes pairing.
    seed_sets = {t: frozenset(v) for t, v in cells.items()}
    common = frozenset.intersection(*seed_sets.values())
    counts = {t: len(v) for t, v in cells.items()}
    total = sum(counts.values())

    print(f"split={a.split}   configs={len(cells)}   cells={total}")
    ragged = {t: sorted(s) for t, s in seed_sets.items() if s != common}
    if ragged:
        print(f"\n  SEED SETS DIFFER ACROSS CONFIGURATIONS. Common seeds: {sorted(common)}")
        for t, s in sorted(ragged.items())[:8]:
            print(f"    {t:22} has {s}")
        print(f"  The comparison below is PAIRED, so it is restricted to the {len(common)} common "
              f"seed(s).")
    if len(common) < 2:
        print(f"\n  REFUSING TO RANK: only {len(common)} seed(s) common to all configurations. "
              f"A single seed cannot bound run-to-run variation -- that is exactly what made the "
              f"published ranking unreproducible (D11 fixed one seed). Finish the sweep first.")
        return 1

    incomplete = {t: c for t, c in counts.items() if c < max(counts.values())}
    if incomplete:
        print(f"  configs with fewer than {max(counts.values())} seeds: "
              f"{', '.join(f'{t}({c})' for t, c in sorted(incomplete.items()))}")

    seeds = sorted(common)
    tags = sorted(cells)
    print(f"  ranking over {len(tags)} configurations x {len(seeds)} paired seeds {seeds}\n")

    leaders = {}
    for metric, key in (("AUC", "auc"), ("AUPRC", "auprc")):
        leaders[metric], _ = report(metric, key, cells, tags, seeds, a.alpha)

    print("=== the metric choice (D-B3.2) ===")
    print(f"  AUC   leader: {leaders['AUC']}")
    print(f"  AUPRC leader: {leaders['AUPRC']}")
    if leaders["AUC"] != leaders["AUPRC"]:
        print("  THE METRICS DISAGREE, so the metric choice IS the winner choice. D34 defers the")
        print("  selection to after the sweep; the manuscript must therefore record that the")
        print("  configuration was selected post hoc (W114).")
    else:
        print("  Both metrics agree, so the metric choice does not change the leader here.")
        print("  That is not the same as the leader being resolvable -- read the sets above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
