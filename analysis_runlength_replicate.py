#!/usr/bin/env python
"""
analysis_runlength_replicate.py — an INDEPENDENT second computation of the
run-length result, on `vals_decay`.

The claim being replicated: elevated positions form runs of four or more bases far
more often than random placement of the same number of elevated positions would
give. It is §5's load-bearing positive and it was found by the interpretability
window's `analysis_ism_regions.py`. This shares no code with that script.

WHY DUPLICATE AT ALL. In one evening the two windows produced roughly ten wrong
claims between them and **not one was caught by whoever made it**. For the single
result the section rests on, two implementations agreeing is worth the cost, and
disagreeing is worth much more.

ON `vals_decay` AND NOTHING ELSE. The original was measured on the decay branch,
with capture held at its unperturbed value. Replicating on `vals` would mix both
branches and their interaction — it would be a third analysis wearing the word
"replication", and agreement or disagreement would be uninterpretable.

THE ELEVATION RULE, AND WHY THE OBVIOUS ONE IS BROKEN. The original used
`|effect| > K x that transcript's median`, K = 10. That self-normalises for
MAGNITUDE and not for TAIL SHAPE, and tail shape is exactly what differs between a
short-transcript pilot and the real subset: the same rule selects 1.7% of positions
on a 50-transcript short draw and 10.7% on the real banks, reaching 43% on the 2.8%
of transcripts whose median falls below 1e-6. The interpretability window measured
that and retracted the pilot's "34 runs against 0" on the strength of it -- on the
real banks the random-placement null does not merely become non-zero, it BEATS the
data (15,304 observed against 17,454 null, seed 100).

DEFAULT RULE HERE: elevated = the top FRACTION of each transcript's own valid
positions. The count is then fixed by construction, so a null that redraws the same
count is matched exactly rather than approximately, and the comparison is purely
about WHERE the elevated positions fall. The fraction is swept, because resting on
one value would hide that the effect size is a function of a number we chose.

The fold rule is kept behind `--rule fold` so this script can CONFIRM the inversion
independently rather than accept it on report. A retraction taken on trust is not
verified.

THE GC CONTROL, AND WHY IT IS THE DECISIVE ONE. Channel 5 averages GC over +/-25
bases, so two adjacent positions share 49 of the 51 positions in their GC window. If
the effect of a substitution is driven by the GC shift it causes, adjacent positions
would have correlated effects BY CONSTRUCTION and the clustering would be an
encoding artifact containing no biology.

Exactly one of the three substitutions at any position leaves GC status unchanged --
A<->T and C<->G -- and `dgc` records the change per entry. So:

    --subset neutral    the single GC-preserving substitution per position
    --subset changing   the two that move GC by +/-1
    --subset all        max over all three, as the original does

If the clustering survives on `neutral`, GC smoothing is not the driver. If it
appears only in `changing`, it is. This uses a column already in the bank and needs
no new forward passes.

THE NULL PRESERVES THE COUNT, NOT THE PLACEMENT. For each transcript the same
number of elevated positions is redrawn uniformly among that transcript's valid
positions and the runs recounted. That asks exactly "would this many elevated
positions clump this much by chance", which is the question. It does NOT preserve
any spatial structure of the transcript, so it is a permissive null: beating it is
necessary and not sufficient.

    python analysis_runlength_replicate.py results_ism_v6/bank_interp_s100.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def runs_of(mask):
    """Lengths of maximal runs of True in a 1-D boolean array."""
    if not mask.any():
        return np.zeros(0, dtype=np.int64)
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("--column", default="vals_decay",
                    choices=["vals_decay", "vals", "vals_capture"])
    ap.add_argument("--rule", default="fraction", choices=["fraction", "fold"],
                    help="fraction: top f of each transcript's valid positions "
                         "(count fixed, null matched exactly). fold: |eff| > K x "
                         "median, the retracted rule, kept to confirm the inversion")
    ap.add_argument("--k", type=float, default=0.01,
                    help="top fraction (rule=fraction) or median multiple (rule=fold)")
    ap.add_argument("--k-sweep", default="0.005,0.01,0.02,0.05")
    ap.add_argument("--subset", default="all", choices=["all", "neutral", "changing"],
                    help="which substitutions contribute: all three, the single "
                         "GC-preserving one, or the two that shift GC")
    ap.add_argument("--min-run", type=int, default=4)
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    sys.stdout.reconfigure(line_buffering=True)
    print(f"run-length replication on {Path(a.bank).name}, column {a.column}")
    rule_txt = ("top K fraction of each transcript's valid positions"
                if a.rule == "fraction" else
                "|effect| > K x transcript median  [RETRACTED RULE, for confirmation]")
    sub_txt = {"all": "all three substitutions per position",
               "neutral": "ONLY the GC-preserving substitution (A<->T, C<->G)",
               "changing": "ONLY substitutions that shift GC by +/-1"}[a.subset]
    print(f"  substitutions: {sub_txt}")
    print(f"  elevation: {rule_txt}; K = {a.k:g}; runs of >= {a.min_run}; "
          f"null redraws the same COUNT, {a.null_draws} draws")

    with h5py.File(a.bank, "r") as f:
        n = f[a.column].shape[0]
        arm = np.array([x.decode() if isinstance(x, bytes) else x for x in f["arm"][:]])
        gene = np.array([x.decode() if isinstance(x, bytes) else x for x in f["gene_id"][:]])
        take = range(n if not a.limit else min(a.limit, n))
        if a.limit:
            print(f"  --limit {a.limit}: a PREFIX of the stratified order, not a sample")

        ks = [float(x) for x in a.k_sweep.split(",")]
        acc = {k: dict(obs=0, null=0.0, elev=0, runs=0, lens=[]) for k in ks}
        per_arm = {}
        n_used = 0
        for i in take:
            v = f["valid"][i]
            if not v.any():
                continue
            # max over the contributing substitutions at each position
            x = np.abs(f[a.column][i])
            if a.subset != "all":
                # dgc is the GC-status change: 0 for the preserving substitution
                # (and for the observed base, whose vals are NaN and drop out).
                d = f["dgc"][i]
                want = (d == 0) if a.subset == "neutral" else (d != 0)
                x = np.where(want, x, np.nan)
            with np.errstate(all="ignore"):
                eff = np.nanmax(np.where(np.isfinite(x), x, np.nan), axis=1)
            ok = v & np.isfinite(eff)
            if ok.sum() < 50:
                continue
            e = eff[ok]
            med = float(np.median(e))
            if not med > 0:
                continue
            n_used += 1
            for k in ks:
                if a.rule == "fold":
                    hi = e > k * med
                else:
                    # top k-fraction by |effect|; count fixed by construction
                    m = max(1, int(round(k * len(e))))
                    hi = np.zeros(len(e), bool)
                    hi[np.argsort(-e, kind="stable")[:m]] = True
                r = runs_of(hi)
                long_obs = int((r >= a.min_run).sum())
                # null: same count, uniform placement among this transcript's valid
                cnt = int(hi.sum())
                long_null = 0.0
                if cnt:
                    for _ in range(a.null_draws):
                        p = np.zeros(len(e), bool)
                        p[rng.choice(len(e), cnt, replace=False)] = True
                        long_null += int((runs_of(p) >= a.min_run).sum())
                    long_null /= a.null_draws
                acc[k]["obs"] += long_obs
                acc[k]["null"] += long_null
                acc[k]["elev"] += cnt
                acc[k]["runs"] += len(r)
                acc[k]["lens"].append(r)
                if k == a.k:
                    d = per_arm.setdefault(arm[i], dict(tx=0, elev=0, runs=0,
                                                        obs=0, null=0.0, lens=[],
                                                        genes=set()))
                    d["tx"] += 1; d["elev"] += cnt; d["runs"] += len(r)
                    d["obs"] += long_obs; d["null"] += long_null
                    d["lens"].append(r); d["genes"].add(gene[i])

    print(f"\n  transcripts used {n_used:,}")
    print(f"\n  {'K':>5} {'elevated':>11} {'runs':>9} {'mean len':>9} "
          f"{'runs>={}'.format(a.min_run):>10} {'null':>9} {'ratio':>8}")
    for k in ks:
        d = acc[k]
        L = np.concatenate(d["lens"]) if d["lens"] else np.zeros(0)
        ml = float(L.mean()) if len(L) else float("nan")
        ratio = d["obs"] / d["null"] if d["null"] > 0 else float("inf")
        flag = "  <- null BEATS observed" if ratio < 1.0 else ""
        print(f"  {k:>5g} {d['elev']:>11,} {d['runs']:>9,} {ml:>9.2f} "
              f"{d['obs']:>10,} {d['null']:>9.1f} {ratio:>8.2f}x{flag}")

    if len(per_arm) > 1:
        print(f"\n  === by arm, K = {a.k:g} — genes are DISJOINT between arms ===")
        print(f"  {'arm':<14} {'tx':>6} {'genes':>7} {'elevated':>10} {'runs':>8} "
              f"{'mean len':>9} {'runs>=%d' % a.min_run:>10} {'null':>8}")
        seen = []
        for name in sorted(per_arm):
            d = per_arm[name]
            L = np.concatenate(d["lens"]) if d["lens"] else np.zeros(0)
            print(f"  {name:<14} {d['tx']:>6,} {len(d['genes']):>7,} {d['elev']:>10,} "
                  f"{d['runs']:>8,} {float(L.mean()) if len(L) else float('nan'):>9.2f} "
                  f"{d['obs']:>10,} {d['null']:>8.1f}")
            seen.append(d["genes"])
        if len(seen) == 2:
            print(f"  genes in BOTH arms: {len(seen[0] & seen[1])}  "
                  f"(must be 0 — §9 step 1 assigns each gene once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
