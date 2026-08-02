#!/usr/bin/env python
"""
probe_decay_head_beyond_ejc.py — does the decay head read anything from sequence
beyond the premature-stop information it is handed?

PETE'S QUESTION, 2026-08-01: "the decay head may have learned PTC-driving features
directly, so ask whether there is anything beyond 'a premature stop is present'."

THE EXACT FORM OF IT. In the interpretable variant the decay head is

    z_d = decay_head(decay_body([ enc_atg(start window),
                                  enc_stop(stop window),
                                  relu(struct_fc(n_downstream_ejc)) ]))

and `n_downstream_ejc` is its ONLY structural input (model_v6.py:133-139,
build_ism_bank.py:56-58 with INTERPRETABLE = [0]). Under the 50-nt rule "a
premature stop is present" is close to `n_downstream_ejc > 0`. So the question is
what the two windows add over the count.

WHY THIS IS NOT AN ABLATION. Channel 4 IS the junction mark, so the stop window
can see downstream junctions itself; and every single-group channel deletion on
this encoding retained >=86% of discrimination, INCLUDING deleting all four base
channels. Ablation cannot separate redundant inputs and this pair is redundant by
construction. The share question is not answerable. The SUFFICIENCY question is:

    condition z_d on the EJC count and on the two measured geometry leaks, then
    ask whether what is left AGREES ACROSS THE FIVE INDEPENDENTLY TRAINED MEMBERS.

Agreement across members cannot come from the covariates -- they have been
conditioned out -- and cannot come from one member's private solution. It can only
come from a shared readable function of the windows. That is the thing ISM would
then be asked to localise, and if it is absent there is nothing there to localise.

THE COVARIATES, and why these. Conditioning runs as a LADDER, because one number
cannot say which control did the work:

  rung 1  the EJC count alone            -- "a premature stop is present"
  rung 2  + the two geometry leaks       -- upstream extent reports distance to
                                            the 5' end; downstream extent reports
                                            ORF length through the midpoint clip
                                            (build_tensor.py:270-271, mid =
                                            (s + e) // 2 passed as fill_hi)
  rung 3  + junction geometry            -- distance from the stop codon to the
                                            next junction, and how many follow

RUNG 3 IS THE ONE THAT DECIDES IT AND IT WAS ALMOST LEFT OUT. The 50-nt rule is
not a count, it is a DISTANCE, and the junction mark is channel 4 of the stop
window -- so a decay head that reads how far its stop sits from the next junction
is doing premature-stop logic more finely than the count it was handed, not
biology beyond it. Conditioning only on the count would score that as "something
new". What survives rung 3 is what the question actually asks for.

Fill counts and junction positions are read off the STORED CODES rather than
recomputed from the clip rule, so this cannot disagree with the encoder about what
was filled or where a junction is.

THE COUNT IS NOT IN THE COLUMN IT LOOKS LIKE IT IS IN. `structural` in the tensor
is NORMALISED on the training split (build_tensor.py:224, 291) -- its
n_downstream_ejc runs -0.672 to 10.06 over 52 non-integer values, so `min(ejc, 4)`
on it is 26 groups rather than 5 and a stratification built from it is not the one
described. The raw count is a separate dataset, `structural_raw`, and that is what
"a premature stop is present" is about. Caught here by an R2 that fell when
covariates were ADDED, which cannot happen to nested strata.

THE NULL. Permute candidates within stratum and recompute the same correlation.
An across-member agreement rate means nothing without the rate at matched
positions -- members share training data and architecture and agree far above
chance on meaningless quantities.

THE TRAP THIS DESIGN MUST NOT FALL INTO. A candidate carrying negligible selection
mass cannot influence the loss, so its z_d is unconstrained and will disagree
across members for reasons that have nothing to do with sequence. Pooling those in
would push the correlation toward zero and produce a confident negative. Every
statistic here is therefore also reported by selection-mass quartile.

POPULATION. The ISM subset (results_ism_v6/ism_subset.tsv), which is STRATIFIED
and not random -- it takes scarce mechanism cells whole. Correlations are reported
unweighted and describe this subset, not the pool. That is the right population
here because it is the one the mutagenesis bank will be built on, so the answer
transfers directly to whether the bank has a decay-side job.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from build_ism_bank import ATG_LEFT, STOP_LEFT, load_model      # noqa: E402
from tensor_io import decode_windows                            # noqa: E402

MEMBERS = [100, 200, 300, 400, 500]


def fill_counts(codes, left):
    """Filled positions on each side of the anchor, read off the stored codes.

    Not recomputed from the clip rule. build_tensor.py decides what is filled and
    a second implementation of that decision is exactly the thing that drifts --
    and both geometry leaks live in WHICH positions are filled, so a probe that
    disagreed with the encoder here would be measuring its own arithmetic.
    """
    filled = (codes & 7) > 0
    return filled[:, :left].sum(1), filled[:, left:].sum(1)


def forward_parts(model, atg9, stop9, u, device, batch=256):
    """z_p and z_d for every candidate, in batches. No aggregation: these are the
    per-candidate head outputs, which is what the question is about."""
    zp, zd = [], []
    with torch.no_grad():
        for i in range(0, len(atg9), batch):
            a = torch.as_tensor(atg9[i:i + batch], device=device)
            s = torch.as_tensor(stop9[i:i + batch], device=device)
            uu = torch.as_tensor(u[i:i + batch], device=device)
            zp.append(model.init_head(model.enc_init(a)).squeeze(-1).cpu().numpy())
            fused = torch.cat([model.enc_atg(a), model.enc_stop(s),
                               torch.relu(model.struct_fc(uu))], dim=-1)
            zd.append(model.decay_head(model.decay_body(fused)).squeeze(-1).cpu().numpy())
    return np.concatenate(zp), np.concatenate(zd)


def selection_mass(zp, tx_index):
    """P(select k) by stick-breaking within each transcript, from z_p.

    The same arithmetic as model_v6.aggregate, in log space for the same reason:
    p_k saturates and the naive product loses the small end entirely.
    """
    out = np.zeros_like(zp, dtype=np.float64)
    for t in np.unique(tx_index):
        m = tx_index == t
        z = zp[m].astype(np.float64)
        log_q = -np.logaddexp(0.0, z)            # log(1 - sigmoid(z))
        log_p = -np.logaddexp(0.0, -z)           # log sigmoid(z)
        out[m] = np.exp(np.cumsum(log_q) - log_q + log_p)
    return out


def junction_geometry(codes_stop, left):
    """Distance from the stop codon to the next junction, and how many follow.

    Read off the stored codes: bit 3 of the byte marks a junction at that
    transcript position (tensor_io.py:12-13), and the anchor of the stop window is
    the last base of the stop codon, at index `left`. Distance is in window
    indices, which are transcript positions here because the window is contiguous.

    A candidate with no downstream junction in the window gets the window width,
    not a sentinel: it is genuinely "further than we can see", it orders correctly
    against the others, and a NaN would silently drop exactly the candidates the
    50-nt rule calls SAFE.
    """
    j = ((codes_stop >> 3) & 1).astype(bool)
    W = codes_stop.shape[1]
    dn = j[:, left:]
    any_dn = dn.any(1)
    first = np.where(any_dn, dn.argmax(1), W - left)
    return first.astype(np.float64), dn.sum(1).astype(np.float64)


def strata(cols, n_q=4):
    """Joint stratification, one factor per column of `cols`.

    Quartiles, not deciles: the cell count is n_q ** len(cols) and a stratum with
    one member contributes nothing to a within-stratum residual except a zero,
    which would then read as perfect agreement.
    """
    def q(x):
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_q + 1)[1:-1]))
        return np.searchsorted(edges, x)
    key = np.zeros(len(cols[0]), dtype=np.int64)
    for v in cols:
        key = key * (n_q + 1) + q(v)
    return key


def residualise(z, key):
    """z minus its own stratum mean. Strata with one member are dropped rather
    than contributing an identically-zero residual, which would inflate every
    correlation that follows by adding a mass of exactly-agreeing zeros."""
    out = np.full_like(z, np.nan, dtype=np.float64)
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(k_sorted)) + 1, len(k_sorted)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if len(idx) > 1:
            out[idx] = z[idx] - z[idx].mean()
    return out


def pair_corr(mat, label, mask=None, rank=False):
    """Mean correlation over the 10 member pairs, on rows finite in every member.

    `rank=True` gives Spearman. Both are reported everywhere because Pearson on a
    residual is carried by its tails, and a handful of extreme candidates agreeing
    across members would produce a large r with nothing behind it. If the two
    disagree, the Pearson number is the one to distrust.
    """
    ok = np.isfinite(mat).all(0)
    if mask is not None:
        ok &= mask
    if ok.sum() < 20:
        return float("nan"), int(ok.sum())
    x = mat[:, ok]
    if rank:
        x = np.stack([np.argsort(np.argsort(r)).astype(np.float64) for r in x])
    rs = [np.corrcoef(x[i], x[j])[0, 1]
          for i in range(len(x)) for j in range(i + 1, len(x))]
    return float(np.mean(rs)), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="results_tensor_v6")
    ap.add_argument("--ckpt-dir", default="results_interp_all/v6_checkpoints")
    ap.add_argument("--split", default="results_ism_v6/ism_subset.tsv")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sp = pd.read_csv(args.split, sep="\t")

    with h5py.File(str(Path(args.tensor) / "nmd_tensor.h5"), "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        row = {s: i for i, s in enumerate(iso)}
        offset, count = f["offset"][:], f["count"][:]
        o_start_all, o_end_all = f["orf_start"][:], f["orf_end"][:]
        struct_all, codes_all = f["structural"][:], f["codes"][:]
        # NORMALISED for the model, RAW for the stratification. The model is fed
        # the normalized block and must be; the question "is a premature stop
        # present" is about the count, and the two are not interchangeable.
        raw_all = f["structural_raw"][:]
        labels = f["labels"][:]
        gene_all = np.array([s.decode() for s in f["gene_id"][:]])
        take = sp[sp.isoform_id.isin(row)]
        take = take.iloc[rng.permutation(len(take))[:args.n]]
        sl = [slice(int(offset[row[s]]), int(offset[row[s]]) + int(count[row[s]]))
              for s in take.isoform_id]
        codes = np.concatenate([codes_all[s] for s in sl])
        o_st = np.concatenate([o_start_all[s] for s in sl]).astype(np.int64)
        o_en = np.concatenate([o_end_all[s] for s in sl]).astype(np.int64)
        ejc = np.concatenate([struct_all[s][:, 0] for s in sl])
        ejc_raw = np.concatenate([raw_all[s][:, 0] for s in sl])
        tx_i = np.concatenate([np.full(s.stop - s.start, i)
                               for i, s in enumerate(sl)])
        is_nmd = np.array([labels[row[s]] for s in take.isoform_id])
        gene = np.array([gene_all[row[s]] for s in take.isoform_id])
        tx_len_tx = take.tx_length.to_numpy()

    print(f"population   {args.split}, {len(take):,} transcripts drawn at "
          f"random from the stratified subset (seed {args.seed})")
    print(f"             {len(codes):,} candidates, NMD prevalence "
          f"{is_nmd.mean():.4f} at the transcript level")
    print(f"             UNWEIGHTED: the subset takes scarce mechanism cells "
          f"whole, so this is the subset, not the pool\n")

    atg9 = decode_windows(codes[:, 0], o_st, ATG_LEFT, o_st)
    stop9 = decode_windows(codes[:, 1], o_en - 1, STOP_LEFT, o_st)
    u = ejc.astype(np.float32)[:, None]
    up, dn = fill_counts(codes[:, 0], ATG_LEFT)
    sup, sdn = fill_counts(codes[:, 1], STOP_LEFT)

    zp_m, zd_m = [], []
    for s in MEMBERS:
        model, _, cols, a = load_model(f"{args.ckpt_dir}/b8_s{s}.pt", args.device)
        assert a["variant"] == "interpretable" and cols == [0], (
            f"seed {s} is {a['variant']} with structural columns {cols}; this "
            f"probe's question is about the variant whose only structural input "
            f"is n_downstream_ejc")
        zp, zd = forward_parts(model, atg9, stop9, u, args.device)
        zp_m.append(zp); zd_m.append(zd)
        print(f"  member {s}   z_d {zd.mean():+.3f} +- {zd.std():.3f}   "
              f"z_p {zp.mean():+.3f} +- {zp.std():.3f}")
    zp_m, zd_m = np.stack(zp_m), np.stack(zd_m)

    mass = np.stack([selection_mass(z, tx_i) for z in zp_m]).mean(0)
    d_junc, n_junc = junction_geometry(codes[:, 1], STOP_LEFT)
    print(f"\njunctions    {100 * (n_junc > 0).mean():.1f}% of candidates have a "
          f"junction downstream of their stop inside the window; "
          f"median distance {np.median(d_junc[n_junc > 0]):.0f} nt")
    print(f"raw count    n_downstream_ejc {ejc_raw.min():.0f}-{ejc_raw.max():.0f}, "
          f"{100 * (ejc_raw > 0).mean():.1f}% above zero "
          f"(this is 'a premature stop is present')")

    rungs = [("1. the EJC count alone", [np.minimum(ejc_raw, 4)]),
             ("2. + the two geometry leaks", [np.minimum(ejc_raw, 4), up, dn, sup, sdn]),
             ("3. + junction geometry", [np.minimum(ejc_raw, 4), up, dn, sup, sdn,
                                         d_junc, n_junc])]

    print("\n" + "=" * 78)
    print("THE LADDER. Each rung conditions on strictly more, so R2 can only rise")
    print("and the correlation can only lose shared structure -- if either moves the")
    print("other way, the strata are not nested and the run is void.\n")
    print(f"  {'conditioning on':<30} {'cells':>6} {'med':>5} {'R2':>6} "
          f"{'pearson':>9} {'spearman':>9} {'null':>7}")
    prev_r2 = -np.inf
    for name, cols in rungs:
        key = strata(cols)
        _, inv = np.unique(key, return_inverse=True)
        sizes = np.bincount(inv)
        r2 = float(np.mean([1 - np.nanmean(residualise(z, key) ** 2) / z.var()
                            for z in zd_m]))
        resid = np.stack([residualise(z, key) for z in zd_m])
        perm = resid.copy()
        for t in np.unique(key):                  # permute WITHIN stratum
            m = np.flatnonzero(key == t)
            if len(m) > 1:
                for i in range(len(perm)):
                    perm[i, m] = perm[i, rng.permutation(m)]
        r_res, n_res = pair_corr(resid, "")
        r_prm, _ = pair_corr(perm, "")
        s_res, _ = pair_corr(resid, "", rank=True)
        assert r2 >= prev_r2 - 1e-9, (
            f"R2 fell from {prev_r2:.4f} to {r2:.4f} when covariates were ADDED; "
            f"nested strata cannot do that, so the stratification is wrong")
        prev_r2 = r2
        print(f"  {name:<30} {len(sizes):>6,} {int(np.median(sizes)):>5,} "
              f"{r2:>6.3f} {r_res:>+9.4f} {s_res:>+9.4f} {r_prm:>+7.4f}   "
              f"n {n_res:,}")
        rungs_last = (key, resid, perm)

    r_raw, _ = pair_corr(zd_m, "")
    print(f"\n  for reference, raw z_d across members:  r {r_raw:+.4f}")
    print("\n  The null column is the same numbers with candidates permuted inside")
    print("  their own stratum. Agreement above it cannot be the covariates and")
    print("  cannot be one member's private solution.")

    # ---- rung 4: conditioning that is not limited by cell counts ------------
    # Quartile strata leave variation INSIDE each cell -- within one
    # junction-distance quartile the distance still runs tens of nucleotides, and
    # both members can read that finer distance off channel 4. Going finer thins
    # cells until a within-cell residual is mostly small-sample noise, so the
    # stratified ladder cannot settle it.
    #
    # So: predict z_d from every non-sequence covariate with a gradient-boosted
    # regressor, OUT OF FOLD AND GROUPED BY GENE, and correlate what is left. The
    # trees take interactions and continuous variables at their own resolution.
    # Out-of-fold because an in-sample residual is shrunk by however much the
    # model overfit, which would understate exactly the quantity of interest.
    from sklearn.ensemble import HistGradientBoostingRegressor       # noqa: E402
    from sklearn.model_selection import GroupKFold                   # noqa: E402

    orf_len = (o_en - o_st).astype(np.float64)
    X = np.column_stack([ejc_raw, up, dn, sup, sdn, d_junc, n_junc,
                         orf_len, o_st.astype(np.float64),
                         tx_len_tx[tx_i].astype(np.float64)])
    groups = gene[tx_i]
    oof = np.zeros_like(zd_m)
    gkf = GroupKFold(n_splits=5)
    for i in range(len(MEMBERS)):
        for tr, te in gkf.split(X, zd_m[i], groups):
            g = HistGradientBoostingRegressor(max_iter=300, random_state=0)
            g.fit(X[tr], zd_m[i][tr])
            oof[i, te] = g.predict(X[te])
    res4 = zd_m - oof
    r2_4 = float(np.mean([1 - (r ** 2).mean() / z.var()
                          for r, z in zip(res4, zd_m)]))
    perm4 = np.stack([r[rng.permutation(len(r))] for r in res4])
    r_res4, n4 = pair_corr(res4, "")
    r_prm4, _ = pair_corr(perm4, "")
    s_res4, _ = pair_corr(res4, "", rank=True)
    print(f"\n  {'4. + a GBM on all 10 covariates':<30} {'oof':>6} {'':>5} "
          f"{r2_4:>6.3f} {r_res4:>+9.4f} {s_res4:>+9.4f} {r_prm4:>+7.4f}   "
          f"n {n4:,}")
    print("     covariates: EJC count, four fill extents, junction distance and")
    print("     count, ORF length, ORF start, transcript length")

    key, resid, perm = rungs_last
    print("\n" + "=" * 78)
    print("BY SELECTION MASS, at the last rung -- a candidate the loss cannot reach")
    print("is not evidence either way: its decay logit is unconstrained.\n")
    qs = np.quantile(mass, [0.25, 0.5, 0.75])
    lab = ["Q1 lowest mass", "Q2", "Q3", "Q4 highest mass"]
    band = np.searchsorted(qs, mass)
    print(f"  {'band':<17} {'mass range':>24} {'n':>8} {'r resid':>9} {'null':>8}")
    for b in range(4):
        m = band == b
        r1, n1 = pair_corr(resid, "", m)
        r0, _ = pair_corr(perm, "", m)
        print(f"  {lab[b]:<17} {mass[m].min():10.2e}-{mass[m].max():<12.2e} "
              f"{int(m.sum()):>8,} {r1:>9.4f} {r0:>8.4f}")


if __name__ == "__main__":
    main()
