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
    # Pooling is what buys the degrees of freedom: 11 contrasts x (5-1) = 44 rather than 4.
    ss, df = 0.0, 0
    for t in contrasts:
        d = vals[leader] - vals[t]
        ss += float(((d - d.mean()) ** 2).sum())
        df += n - 1
    sigma_d = math.sqrt(ss / df) if df > 0 else float("nan")

    k = len(contrasts)
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
    diag = {
        "n": n, "k": k, "df": df, "sigma_d": sigma_d, "t_crit": t_crit, "thresh": thresh,
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
        verdict = "excluded" if r["excluded"] else "CANNOT EXCLUDE as best"
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
        print(f"  min detectable diff @80%  {diag['mdd80']:.5f}" if "mdd80" in diag else "")
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
    ap.add_argument("--results-dir", default="results_4ct_sweep")
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
