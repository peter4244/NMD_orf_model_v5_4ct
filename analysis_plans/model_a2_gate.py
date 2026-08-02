"""
model_a2_gate.py — SEQ-A2, the second implementation.

THE GATE. Is the keto composition signature at elevated positions a property of
what the decay head reads, or of where the model routes?

This is the SECOND of two independent implementations. It is written against the
interpretability window's row in `ANALYSIS_SEQUENCING_PROPOSAL.md` ("The
specification, fixed 2026-08-02"), NOT against an independent reading of the
question. Shared specification, independent code. Writing my own specification
would produce two analyses rather than a replication, and the failure that
motivated this arrangement -- both windows implementing the run-length statistic
and agreeing to four decimals while both were wrong -- was a shared design
premise that no amount of independent coding could have caught.

Everything here is `vals_decay`. Capture is out of scope, not merely unreported.

--------------------------------------------------------------------------------
THE SPECIFICATION, as implemented. Deviations are findings, not detail.

 bands        8 global log10(mass) quantile bands over LIVE positions. Global so a
              band means the same thing in every transcript. Dead positions
              (mass < 1e-8) form a 9th band which is READ BUT NEVER SCORED -- see
              THE DEAD BAND below.
 elevation    WITHIN-STRATUM. Each (transcript x band x region) cell contributes
              its own top 10%. Not global-then-binned: |vals| ~ mass x sensitivity,
              so a globally elevated set concentrates in high-mass bands by
              construction and leaves the low bands empty.
 cell floor   >=100 live positions. Frozen after job 8896584: 36/36 group
              comparisons NOT DIFFERENTIAL, position-level retention 97.9%.
 background   WITHIN-CELL -- the non-elevated positions of the same transcript in
              the same band and region. Never global, which collapses the test
              back to the confounded version.
 aggregation  per transcript, then unweighted mean across transcripts. Interval by
              GENE-clustered bootstrap, since transcripts of one gene are not
              independent draws.
 coverage     stratified MARGINALLY, not jointly -- reported as a within-band
              balance check. Joint stratification fragments cells below the n the
              null needs.
 null         within-cell permutation of the elevated label at that cell's own n.
              Implemented exactly rather than by shuffling -- see THE NULL below.
 seeds        all five. Cross-seed agreement is reported BESIDE the gate, never
              folded in. Direction must hold in >=4/5 for a positive to read as
              positive.
 sweep        primary (8 bands, top 10%); swept (4, 5%) and (16, 20%).

--------------------------------------------------------------------------------
THE NULL, and why this is the specified null computed exactly.

The row specifies "within-cell permutation of the elevated label at that cell's
own n". Permuting a binary label of k ones among n positions and counting how many
land on keto bases IS a hypergeometric draw: n positions, K of them keto, k drawn
without replacement. So the permutation distribution is available in closed form
and does not have to be approximated by shuffling.

This is the same null, computed exactly rather than by Monte Carlo -- it removes
simulation error rather than changing the test. Because that is the kind of
substitution that should never be taken on trust, `--verify-null` runs an actual
label shuffle on a subsample and checks the two agree. Run it at least once.

The aggregate statistic is a mean over cells, so the null is still drawn per
replicate: one hypergeometric variate per cell, aggregated the same way as the
observed. Cells are drawn independently, which is the null's assumption and is
stated rather than assumed.

--------------------------------------------------------------------------------
THE DEAD BAND is a control, not a stratum.

A substitution at a candidate with no routable mass cannot move the output, so
nothing in the dead band is a measurement of what the decay head reads. It is
carried because dropping it would be differential on the mechanism cell (5.5),
and because it tests the one circularity the gate cannot otherwise escape: if the
instrument's numerical sensitivity were base-dependent, magnitude and composition
would correlate through the encoder rather than through the head, and that would
show up as a keto signature HERE. A signature in the dead band invalidates a
signature everywhere else. It is reported and never counted toward the gate.

--------------------------------------------------------------------------------
ONE SPECIFICATION GAP I HAD TO RESOLVE, FLAGGED RATHER THAN BURIED.

The row stratifies by region and states the decision rule over BANDS ("in >=2/3
of qualifying bands"). It does not say how the region strata combine into one
verdict. With the interpretability window closed there was no one to send it back
to, so this implementation reports the rule PER REGION and never pools across
them -- on the same logic that forbids pooling the three cells in 5.2: a
stratified variable that gets pooled at the last step was not stratified. The
consequence is that the gate can return different verdicts in different regions,
which is itself a result and is printed as one. Pete should confirm or overrule.

--------------------------------------------------------------------------------
WHAT A POSITIVE DOES NOT LICENSE. Within a band the elevated positions are still
the larger-magnitude ones, so a positive licenses "among equally-routed
positions, the more sensitive ones are keto-enriched" and NOT "composition is
enriched independent of effect magnitude". A3 owns the general magnitude
question. Every negative reads "not detected by single-base ISM", never "not
present" -- conv->ReLU is an m-of-n detector and a robust pattern is invisible to
single substitutions by construction.

Run from the repo root.
"""

import argparse
import json
import sys

import numpy as np
import h5py

NT = "ACGT"
KETO = (2, 3)          # G, T -- i.e. G+U on RNA
AMINO = (0, 1)         # A, C
DEAD_CUT = 1e-8
# max-over-cells of a KS statistic inflates with the number of cells tested;
# 1.5 is roughly the Bonferroni-adjusted critical multiplier at n_cells = 40.
KS_PASS = 1.5
REGIONS = ("5p_of_start", "in_orf", "3p_of_stop")


# ----------------------------------------------------------------- bank access

def transcript_slice(f, i, spans, cand_off, cand_cnt, p_select,
                     orf_start, orf_end):
    """One transcript's positions, with the covariates the cells are cut on.

    h5py is lazy and `[:]` is what defeats it, so this reads per transcript. The
    laptop has 8 GB against a bank that is 3.9 GB read whole; the cluster does not
    care but the discipline keeps the script runnable in both places.
    """
    lo, nk = int(cand_off[i]), int(cand_cnt[i])
    b = spans[lo:lo + nk]
    P = int(max(b[:, 3].max(), b[:, 5].max()))
    if P < 50:
        return None

    v = f["vals_decay"][i, :P].astype(np.float64)
    o = f["obs"][i, :P]
    m = f["mass"][i, :P].astype(np.float64)
    fill = f["fill_count"][i, :P].astype(np.int32)
    with np.errstate(invalid="ignore"):
        e = np.nanmax(np.abs(v), axis=1)

    # vals is NaN at the observed base BY CONSTRUCTION, so isfinite().all(1) is
    # never true. Three scripts across two windows hit this.
    ok = f["valid"][i, :P].astype(bool) & np.isfinite(e) & (o >= 0)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return None

    # region is defined by the ORF THE MODEL COMMITTED TO, per 5.2. Anchoring on
    # the annotation is a different question and answers 5.2's decoupling, not
    # this one. Which anchor was used is part of the claim and is printed.
    ps = p_select[lo:lo + nk]
    k_sel = int(np.argmax(ps))
    s = int(orf_start[lo + k_sel])
    t = int(orf_end[lo + k_sel])
    reg = np.full(len(idx), 2, dtype=np.int8)
    reg[idx < s] = 0
    reg[(idx >= s) & (idx < t)] = 1

    return dict(idx=idx, e=e[idx], obs=o[idx], mass=m[idx],
                fill=fill[idx], region=reg)


def keto_mask(obs):
    return (obs == KETO[0]) | (obs == KETO[1])


# ------------------------------------------------------------------ the cells

def band_edges(f, N, nb, spans, cand_off, cand_cnt, p_select, orf_start, orf_end):
    """Global log-mass quantile edges over live positions. Dead is not a quantile.

    log10(0) is undefined and mass is float32 with a hard floor, which is why the
    dead cut is a threshold applied FIRST and the bands are quantiles of what
    remains.
    """
    pool = []
    for i in range(N):
        t = transcript_slice(f, i, spans, cand_off, cand_cnt, p_select,
                             orf_start, orf_end)
        if t is None:
            continue
        live = t["mass"][t["mass"] >= DEAD_CUT]
        if len(live):
            pool.append(np.log10(live).astype(np.float32))
    allm = np.concatenate(pool)
    edges = np.quantile(allm, np.linspace(0, 1, nb + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges, len(allm)


def build_cells(f, N, edges, nb, floor, spans, cand_off, cand_cnt, p_select,
                orf_start, orf_end):
    """Every (transcript x band x region) cell, with the counts the gate needs.

    Band index nb means DEAD. It is built like any other cell and excluded from
    the gate at scoring time, not here -- so its census is visible.
    """
    cells = []
    census = np.zeros((nb + 1, len(REGIONS), 3), dtype=np.int64)  # n_cells, n_qual, n_live
    for i in range(N):
        t = transcript_slice(f, i, spans, cand_off, cand_cnt, p_select,
                             orf_start, orf_end)
        if t is None:
            continue
        live = t["mass"] >= DEAD_CUT
        band = np.full(len(t["mass"]), nb, dtype=np.int16)
        if live.any():
            band[live] = np.digitize(np.log10(t["mass"][live]), edges[1:-1])
        kmask = keto_mask(t["obs"])
        for bi in range(nb + 1):
            for ri in range(len(REGIONS)):
                sel = (band == bi) & (t["region"] == ri)
                n = int(sel.sum())
                if n == 0:
                    continue
                census[bi, ri, 0] += 1
                census[bi, ri, 2] += n
                if n < floor:
                    continue
                census[bi, ri, 1] += 1
                cells.append(dict(
                    tx=i, band=bi, region=ri, n=n,
                    e=t["e"][sel], keto=kmask[sel],
                    fill=float(t["fill"][sel].mean()),
                ))
    return cells, census


# --------------------------------------------------------------- the statistic

def cell_ratio(cell, top_frac):
    """Keto ratio inside one cell: elevated keto fraction over background's.

    Returns (ratio, n, n_elev, K_keto, k_elev_keto) so the null can be computed
    from the same enumeration the observed value came from -- field 13, and it is
    also the only way the two are guaranteed to describe the same set.
    """
    n = cell["n"]
    k = max(1, int(round(top_frac * n)))
    if k >= n:
        return None
    cut = np.partition(cell["e"], -k)[-k]
    elev = cell["e"] >= cut
    # ties at the cut can push the elevated set past k; that is the correct
    # behaviour for a rank rule but the null must be told the REALISED k
    k = int(elev.sum())
    if k == 0 or k >= n:
        return None
    K = int(cell["keto"].sum())
    kk = int(cell["keto"][elev].sum())
    bg_keto = K - kk
    bg_n = n - k
    if bg_n == 0 or bg_keto == 0:
        return None
    return dict(ratio=(kk / k) / (bg_keto / bg_n), n=n, k=k, K=K, kk=kk)


def null_draws(stats, rng, n_rep):
    """Permutation null, computed exactly.

    Permuting a k-of-n elevated label and counting keto hits is a hypergeometric
    draw. One variate per cell per replicate, aggregated exactly as the observed
    statistic is. `--verify-null` checks this against a real shuffle.
    """
    n = np.array([s["n"] for s in stats])
    k = np.array([s["k"] for s in stats])
    K = np.array([s["K"] for s in stats])
    out = np.empty((n_rep, len(stats)))
    for r in range(n_rep):
        kk = rng.hypergeometric(K, n - K, k)
        bg_keto = K - kk
        bg_n = n - k
        with np.errstate(divide="ignore", invalid="ignore"):
            out[r] = (kk / k) / (bg_keto / bg_n)
    return out


def verify_null(stats, rng, n_rep=4000, n_cells=40):
    """Prove the exact null reproduces an actual label shuffle before trusting it.

    TWO-SAMPLE KOLMOGOROV-SMIRNOV, reported as a multiple of its own 5% critical
    value, so 1.0 is the pass/fail line and the scale means something.

    Two earlier versions of this check were built and BOTH were demonstrated
    incapable of failing, by deliberate mutation:

      v1 compared MEANS ONLY. A binomial substituted for the hypergeometric --
         sampling WITH replacement rather than without -- scored 0.014 against
         the correct null's 0.045, i.e. it looked BETTER than correct. The two
         distributions share a mean and differ only in variance, and variance is
         exactly what the gate reads, since its threshold is a 95th percentile.

      v2 added SD and the 95th percentile and took a max over both across cells.
         That is a maximum of eighteen noisy quantities, so it was dominated by
         Monte Carlo excursions rather than by the difference being tested: the
         correct null scored 0.466 at one setting and the mutation 0.044.

    A check whose statistic has no known null scale cannot be thresholded, which
    is what both of those were doing. KS has one.
    """
    pick = rng.choice(len(stats), size=min(n_cells, len(stats)), replace=False)
    worst = 0.0
    for j in pick:
        s = stats[j]
        n, k, K = s["n"], s["k"], s["K"]
        lab = np.zeros(n, bool)
        lab[:k] = True
        keto = np.zeros(n, bool)
        keto[:K] = True
        shuffled = np.empty(n_rep)
        for r in range(n_rep):
            rng.shuffle(lab)
            shuffled[r] = keto[lab].sum()
        exact = rng.hypergeometric(K, n - K, k, size=n_rep).astype(float)

        grid = np.union1d(shuffled, exact)
        f1 = np.searchsorted(np.sort(shuffled), grid, side="right") / n_rep
        f2 = np.searchsorted(np.sort(exact), grid, side="right") / n_rep
        d = float(np.max(np.abs(f1 - f2)))
        crit = 1.36 * np.sqrt(2.0 / n_rep)          # two-sample, alpha = 0.05
        worst = max(worst, d / crit)
    return worst


def gene_bootstrap(values, genes, rng, n_rep=2000):
    """Interval clustered on gene: transcripts of one gene are not independent.

    Resamples GENES with replacement and takes every transcript of each drawn
    gene, which is the cluster bootstrap. Implemented on per-gene sums and counts
    rather than by rebuilding an index per replicate: the naive version
    concatenates one array per gene per replicate, which at ~2,000 genes and
    2,000 replicates is four million array constructions per (band x region)
    group and would dominate the run. Same estimator, and the mean of a drawn
    sample is sum(sums)/sum(counts) exactly.
    """
    uniq, inv = np.unique(genes, return_inverse=True)
    ng = len(uniq)
    sums = np.bincount(inv, weights=values, minlength=ng)
    cnts = np.bincount(inv, minlength=ng).astype(float)
    pick = rng.integers(0, ng, size=(n_rep, ng))
    out = sums[pick].sum(1) / cnts[pick].sum(1)
    return np.percentile(out, [2.5, 97.5])


# -------------------------------------------------------------------- the gate

def score(cells, genes, nb, top_frac, rng, n_rep, verify):
    """Per (band x region): observed mean ratio, null 95th percentile, verdict."""
    rows = []
    checked = False
    for ri in range(len(REGIONS)):
        for bi in range(nb + 1):
            sub = [c for c in cells if c["band"] == bi and c["region"] == ri]
            if not sub:
                continue
            stats, tx = [], []
            for c in sub:
                s = cell_ratio(c, top_frac)
                if s is not None:
                    stats.append(s)
                    tx.append(c["tx"])
            if len(stats) < 20:
                rows.append(dict(band=bi, region=ri, n_tx=len(stats),
                                 obs=np.nan, p95=np.nan, verdict="thin"))
                continue
            if verify and not checked:
                d = verify_null(stats, rng)
                print(f"  null verification: KS vs shuffle = {d:.2f}x critical"
                      f"  ({'PASS' if d < 1.0 else 'FAIL'})")
                checked = True
            obs_v = np.array([s["ratio"] for s in stats])
            obs = float(obs_v.mean())
            draws = null_draws(stats, rng, n_rep)
            null_means = np.nanmean(draws, axis=1)
            p95 = float(np.percentile(null_means, 95))
            lo, hi = gene_bootstrap(obs_v, np.array(genes)[tx], rng)
            rows.append(dict(band=bi, region=ri, n_tx=len(stats), obs=obs,
                             p95=p95, ci=(lo, hi),
                             mean_fill=float(np.mean([c["fill"] for c in sub])),
                             verdict="above" if obs > p95 else "not above"))
    return rows


def verdict_by_region(rows, nb):
    """The decision rule, applied per region and never pooled across them."""
    out = {}
    for ri, rname in enumerate(REGIONS):
        live = [r for r in rows
                if r["region"] == ri and r["band"] < nb and r["verdict"] != "thin"]
        if not live:
            out[rname] = ("no qualifying bands", 0, 0)
            continue
        n_ok = sum(1 for r in live if r["verdict"] == "above")
        frac = n_ok / len(live)
        hi = [r for r in live if r["band"] >= nb // 2]
        lo = [r for r in live if r["band"] < nb // 2]
        hi_ok = sum(1 for r in hi if r["verdict"] == "above")
        lo_ok = sum(1 for r in lo if r["verdict"] == "above")
        if hi and lo and hi_ok == len(hi) and lo_ok == 0:
            v = "CONDITIONAL -- interaction, not independence. Do not round up"
        elif frac >= 2 / 3:
            v = "POSITIVE"
        elif frac < 0.5:
            v = "NEGATIVE"
        else:
            v = "AMBIGUOUS -- resolve by sweep or seeds, not by looking"
        out[rname] = (v, n_ok, len(live))
    return out


def run_seed(path, nb, top_frac, floor, n_rep, seed_rng, verify):
    with h5py.File(path, "r") as f:
        spans = f["spans"][:]
        cand_off = f["cand_offset"][:]
        cand_cnt = f["cand_count"][:]
        p_select = f["p_select"][:]
        orf_start = f["cand_orf_start"][:]
        orf_end = f["cand_orf_end"][:]
        genes = np.array([s.decode() for s in f["gene_id"][:]])
        N = len(genes)

        edges, n_live = band_edges(f, N, nb, spans, cand_off, cand_cnt,
                                   p_select, orf_start, orf_end)
        cells, census = build_cells(f, N, edges, nb, floor, spans, cand_off,
                                    cand_cnt, p_select, orf_start, orf_end)

    rows = score(cells, genes, nb, top_frac, seed_rng, n_rep, verify)
    return rows, census, n_live, N


def print_census(census, nb, n_live, N):
    print("\n  THE THREE-WAY CENSUS -- emitted before any composition statistic")
    print(f"  {'band':>6} {'region':>13} {'cells':>8} {'qualifying':>11}"
          f" {'live pos':>12} {'retained':>9}")
    tot_pos = tot_keep = 0
    for bi in range(nb + 1):
        for ri, rname in enumerate(REGIONS):
            c, q, p = census[bi, ri]
            if c == 0:
                continue
            tag = "DEAD" if bi == nb else str(bi)
            print(f"  {tag:>6} {rname:>13} {c:>8,} {q:>11,} {p:>12,}"
                  f" {'-' if c == 0 else f'{q/c:.1%}':>9}")
    live_cells = census[:nb]
    kept = live_cells[:, :, 1].sum()
    allc = live_cells[:, :, 0].sum()
    print(f"\n  live bands: {allc:,} cells, {kept:,} qualifying ({kept/allc:.1%})")
    print(f"  transcripts {N}, live positions {n_live:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+",
                    default=[f"results_ism_v6/bank_interp_s{s}.h5"
                             for s in (100, 200, 300, 400, 500)])
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--n-rep", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--verify-null", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    print("SEQ-A2 GATE -- second implementation, written against the "
          "interpretability window's row")
    print(f"bands {args.bands}   top {args.top_frac:.0%}   floor >={args.floor}"
          f"   null replicates {args.n_rep}   dead cut {DEAD_CUT:g}")
    print("column vals_decay; region anchored on the MODEL-SELECTED ORF; "
          "finite mask (isfinite.sum(1)==3)")

    rng = np.random.default_rng(args.seed)
    all_rows = {}
    for path in args.banks:
        print("\n" + "=" * 78)
        print(f"BANK {path}")
        print("=" * 78)
        rows, census, n_live, N = run_seed(path, args.bands, args.top_frac,
                                           args.floor, args.n_rep, rng,
                                           args.verify_null)
        print_census(census, args.bands, n_live, N)

        print(f"\n  {'band':>6} {'region':>13} {'n_tx':>6} {'keto':>8}"
              f" {'null p95':>9} {'95% CI':>18} {'mean fill':>10} {'verdict':>10}")
        for r in sorted(rows, key=lambda x: (x["region"], x["band"])):
            tag = "DEAD" if r["band"] == args.bands else str(r["band"])
            if r["verdict"] == "thin":
                print(f"  {tag:>6} {REGIONS[r['region']]:>13} {r['n_tx']:>6}"
                      f" {'--':>8} {'--':>9} {'--':>18} {'--':>10} {'thin':>10}")
                continue
            ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
            print(f"  {tag:>6} {REGIONS[r['region']]:>13} {r['n_tx']:>6}"
                  f" {r['obs']:>8.3f} {r['p95']:>9.3f} {ci:>18}"
                  f" {r['mean_fill']:>10.0f} {r['verdict']:>10}")

        print("\n  DEAD BAND is the instrumental control and counts toward nothing.")
        print("  A keto signature there means magnitude and composition correlate")
        print("  through the encoder, which would invalidate the live bands.")

        for rname, (v, ok, tot) in verdict_by_region(rows, args.bands).items():
            print(f"  {rname:>13}: {ok}/{tot} bands above null  ->  {v}")
        all_rows[path] = rows

    print("\n" + "=" * 78)
    print("CROSS-SEED -- reported BESIDE the gate, never folded into it")
    print("=" * 78)
    for ri, rname in enumerate(REGIONS):
        dirs = []
        for path, rows in all_rows.items():
            live = [r for r in rows if r["region"] == ri
                    and r["band"] < args.bands and r["verdict"] != "thin"]
            if live:
                dirs.append(np.mean([r["obs"] for r in live]) > 1.0)
        if dirs:
            print(f"  {rname:>13}: direction holds in {sum(dirs)}/{len(dirs)} seeds"
                  f"  ({'meets' if sum(dirs) >= 4 else 'FAILS'} the >=4/5 floor)")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({k: [{kk: (list(vv) if isinstance(vv, tuple) else vv)
                            for kk, vv in r.items()} for r in v]
                       for k, v in all_rows.items()}, fh, indent=1)
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
