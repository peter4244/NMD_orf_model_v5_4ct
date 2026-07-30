#!/usr/bin/env python3
"""collect_sweep.py — turn 60 sweep cells into the numbers the selection decision needs.

WHAT THIS IS FOR. D34 defers all six B3 decisions until the results exist, and two of them --
the selection metric and the tie-break -- get applied to these results. The question that
decides whether the tie-break matters is not "which config wins" but "can the sweep tell its
top configs apart at all". Historically it could not: the published grid's rank 1 and rank 2
differed by 0.00139 AUC on models that early-stopped at epoch 3-5, and run-to-run variation
was never measured, because D11 fixed a single seed and (before 77b4018) five array tasks
would have written one checkpoint anyway.

So this reports the SPREAD alongside the mean, and states explicitly which configurations are
statistically indistinguishable from the leader. If the leaders overlap within 1 sd there is
no winner in the data, and whatever tie-break rule was written down is what produces the
answer -- which is a thing to know before quoting a winner, not after.

It does NOT choose a winner. The metric and tie-break are Pete's, pre-registered or recorded
as post hoc; this prints what both choices would give so the difference is visible.

    python3 collect_sweep.py
    python3 collect_sweep.py --results-dir results_4ct_sweep --split val_clean
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_4ct_sweep")
    ap.add_argument("--split", default="val_clean")
    a = ap.parse_args()

    rd = Path(a.results_dir)
    pat = re.compile(rf"^metrics_(atg\d+_stop\d+)_seed(\d+)_{re.escape(a.split)}\.json$")
    cells = {}
    for f in sorted(rd.glob(f"metrics_*_seed*_{a.split}.json")):
        m = pat.match(f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        cells.setdefault(m.group(1), {})[int(m.group(2))] = d

    if not cells:
        print(f"no cells found in {rd} for split={a.split}")
        print("  expected: metrics_atg<A>_stop<B>_seed<S>_" + a.split + ".json")
        return 1

    rows = []
    for tag, seeds in cells.items():
        aucs = [v["auc"] for v in seeds.values()]
        aps = [v["auprc"] for v in seeds.values()]
        rows.append({
            "tag": tag, "n_seeds": len(seeds),
            "auc_mean": st.mean(aucs), "auc_sd": st.stdev(aucs) if len(aucs) > 1 else float("nan"),
            "ap_mean": st.mean(aps), "ap_sd": st.stdev(aps) if len(aps) > 1 else float("nan"),
            "epochs": sorted(v.get("best_epoch") for v in seeds.values()),
        })

    n_expected = 60
    got = sum(r["n_seeds"] for r in rows)
    print(f"split={a.split}   configs={len(rows)}   cells={got}/{n_expected}")
    if got < n_expected:
        print(f"  INCOMPLETE -- {n_expected - got} cells missing. Ranking below is provisional.")
    incomplete = [r["tag"] for r in rows if r["n_seeds"] < 5]
    if incomplete:
        print(f"  configs with <5 seeds: {', '.join(incomplete)}")
    print()

    for metric, mkey, skey in (("AUC", "auc_mean", "auc_sd"), ("AUPRC", "ap_mean", "ap_sd")):
        rows.sort(key=lambda r: -r[mkey])
        best = rows[0]
        print(f"=== ranked by {metric} (mean +/- sd across seeds) ===")
        print(f"  {'config':22}{'mean':>10}{'sd':>10}{'n':>4}   indistinguishable from leader?")
        # A MISSING sd MEANS "CANNOT ASSESS", NEVER "RESOLVED" (fixed 2026-07-29).
        # The first version compared a 1-seed config (sd = nan) against a 5-seed one and, because
        # every nan comparison is False, fell through to "leads by more than 1 sd; the ranking is
        # resolved". That is the published sweep's exact error rebuilt in the analysis: declaring
        # a ranking resolved from single seeds, when a single seed cannot bound run-to-run
        # variation at all. It would have fired on every config at the end of the seed-100 pass.
        def assessable(r):
            return r["n_seeds"] >= 2 and r[skey] == r[skey]
        for r in rows:
            gap = best[mkey] - r[mkey]
            if r is best:
                tie = ""
            elif not (assessable(best) and assessable(r)):
                tie = "  ?    CANNOT ASSESS (needs >=2 seeds on both)"
            else:
                pooled = max(best[skey], r[skey])
                tie = ("  YES  (gap %.5f <= 1 sd %.5f)" % (gap, pooled) if gap <= pooled
                       else "  no   (gap %.5f)" % gap)
            sd_txt = f"{r[skey]:>10.5f}" if r[skey] == r[skey] else f"{'--':>10}"
            print(f"  {r['tag']:22}{r[mkey]:>10.5f}{sd_txt}{r['n_seeds']:>4}{tie}")
        unassessable = [r["tag"] for r in rows if not assessable(r)]
        ties = [r["tag"] for r in rows[1:]
                if assessable(best) and assessable(r)
                and (best[mkey] - r[mkey]) <= max(best[skey], r[skey])]
        print(f"  leader: {best['tag']}")
        if unassessable:
            print(f"  *** {len(unassessable)} config(s) have <2 seeds, so NO ranking claim can be")
            print(f"      made about them yet: {', '.join(unassessable[:6])}"
                  + (" ..." if len(unassessable) > 6 else ""))
            print(f"  *** A single seed cannot bound run-to-run variation. This is exactly what")
            print(f"      made the published ranking unreproducible (D11 fixed one seed).")
        if ties:
            print(f"  *** {len(ties)} config(s) within 1 sd of it: {', '.join(ties)}")
            print(f"  *** The sweep does NOT resolve a winner by {metric}. Whatever tie-break")
            print(f"      rule is recorded is what selects the configuration -- say so.")
        elif not unassessable:
            print(f"  {best['tag']} leads by more than 1 sd; the ranking is resolved by {metric}.")
        print()

    by_auc = max(rows, key=lambda r: r["auc_mean"])["tag"]
    by_ap = max(rows, key=lambda r: r["ap_mean"])["tag"]
    print("=== the metric choice (D-B3.2) ===")
    print(f"  AUC   would select: {by_auc}")
    print(f"  AUPRC would select: {by_ap}")
    if by_auc != by_ap:
        print("  THE METRICS DISAGREE. The metric choice IS the winner choice; it cannot be")
        print("  made by looking at these numbers without that being selection on the outcome.")
    else:
        print("  Both metrics agree, so the metric choice does not change the winner here.")

    # Seed variability is a deliverable in its own right (spec section 9.2: it is what
    # justifies the ensemble), not just an input to the tie-break.
    sds = [r["auc_sd"] for r in rows if r["auc_sd"] == r["auc_sd"]]
    if sds:
        print(f"\n=== seed variability (the ensemble's justification) ===")
        print(f"  AUC sd across seeds: median {st.median(sds):.5f}, "
              f"range {min(sds):.5f}-{max(sds):.5f} over {len(sds)} configs")
        print(f"  For scale, the PUBLISHED grid's rank1-rank2 AUC gap was 0.00139.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
