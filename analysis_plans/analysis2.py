#!/usr/bin/env python3
"""
analysis2.py — steps 5 to 8 of ANALYSIS2_PLAN: reduce the fifty per-isoform attribution
tables to branch shares, a stop-to-start ratio, and their spreads.

Implements the plan as amended by the independent statistical review of steps 7 and 8
(2026-07-31). The review's five verdicts, and what this file does about each:

  1. TWO SPREADS. The interpretation spread is ensemble-level, the training spread is
     member-level, and they are not centred on the same value -- because |mean_s phi_s| <=
     mean_s |phi_s|, the ensemble's attribution mass is systematically smaller than the
     members' whenever members disagree in sign. So `value +/- a +/- b` attaches a spread to
     a centre it does not belong to. This file computes them separately, adds a delete-one
     MEMBER JACKKNIFE for the ensemble's own seed uncertainty, and emits the member centre
     alongside the member spread so the offset is visible rather than implied.
  2. CROSSED DESIGN. A two-way random-effects decomposition is reported as a DIAGNOSTIC that
     licenses (or does not license) the marginal SDs. It is identified here despite one
     observation per cell, because a cell is a deterministic function of (member, draw):
     exact enumeration over all 2^3 coalitions, fixed 500-isoform reference. No within-cell
     noise, so MS_AB estimates the interaction cleanly on 16 df.
  3. COMPOSITIONAL. NO log-ratio transform. log(m_stop/m_start) already IS an additive
     log-ratio coordinate of the composition, so the coherent statistic is reported already.
     What is added is the consequence the plan omitted: three shares, two degrees of freedom.
     The arithmetic-vs-closed-geometric check below is what licenses this choice.
  4. RATIO. Summarised at ENSEMBLE level on the LOG scale (geometric mean), matching the
     level and scale of everything else. The previous spec pooled 25 member-level cells,
     producing a spread that corresponds to no stated question and a centre inconsistent
     with the reported shares.
  5. PER-CELL TABLE. 110 rows -- per configuration, 25 member + 5 ensemble + 25 leave-one-out
     -- so every estimator here is recomputable by a third party without rerunning anything.
     The leave-one-out rows must be materialised: they CANNOT be derived from member-level
     summaries, because the absolute value is applied after the signed average.

EVERY CHECK PRINTS THE VALUE IT MEASURED, as analysis1.py does. Read the log beside the plan.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python analysis2.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO   = Path.home() / "claude_projects" / "NMD_orf_model_v5_4ct"
CELLS  = REPO / "results_interp_all"
OUT    = REPO / "results_interp_all"
H5     = REPO / "results_4ct_dn" / "nmd_orf_data.h5"

TAGS   = ["atg2000_stop2000", "atg1000_stop1000"]
SEEDS  = [100, 200, 300, 400, 500]
DRAWS  = [1, 2, 3, 4, 5]
BRANCH = ["structural", "stop", "atg"]          # `atg` is the START branch; legacy column name
NICE   = {"structural": "structural", "stop": "stop", "atg": "start"}
TOL    = 1e-12                                   # step 5 residual bar
N_ALL  = 41765
N_NMD  = 9321

_ind = [""]


def log(m=""):
    print(f"{_ind[0]}{m}", flush=True)


def step(n, t):
    _ind[0] = ""
    log(f"\n{'=' * 78}\nSTEP {n} — {t}\n{'=' * 78}")
    _ind[0] = "  "


def check(label, ok, measured):
    log(f"[{'OK  ' if ok else 'FAIL'}] {label}: {measured}")
    if not ok:
        raise SystemExit(f"\nSTOPPED: {label} — measured {measured}")


def path_of(tag, seed, draw):
    return CELLS / f"kernel_shap_branch_{tag}_seed{seed}_all_run{draw}.tsv"


# ---------------------------------------------------------------------------------------
def step5_accept():
    """Accept or reject each of the fifty files, and record the tally.

    The residual is a checksum, not a constraint: with three branches all eight coalitions
    are enumerated and the Shapley values solved exactly, so the sum identity is an algebraic
    consequence of the arithmetic. Anything materially above rounding means the file did not
    come from the calculation it claims to.

    The tally is reported whether or not anything rejects. A silently dropped cell would
    leave the spreads computed over fewer than 25 while the output still said 25.
    """
    step(5, "accept or reject each file, and record the tally")
    import h5py
    with h5py.File(H5, "r") as f:
        ids0 = np.array([x.decode() if isinstance(x, bytes) else x for x in f["isoform_id"][:]])
        y0   = f["labels"][:].astype(int)
    check("step 1 population re-read from Dataset A", len(ids0) == N_ALL and int(y0.sum()) == N_NMD,
          f"n={len(ids0):,}, label==1 {int(y0.sum()):,}")

    with h5py.File(H5, "r") as f:
        sp = np.array([x.decode() if isinstance(x, bytes) else x for x in f["split"][:]])
    test_mask = (sp[y0 == 1] == "test")          # over the 9,321, which are in test_clean
    check("test-split sensitivity population", int(test_mask.sum()) == 2405,
          f"{int(test_mask.sum()):,} of {len(test_mask):,} summarised isoforms lie in split=='test'")

    store, accepted, rejected = {}, 0, []
    for tag in TAGS:
        for s in SEEDS:
            for r in DRAWS:
                p = path_of(tag, s, r)
                if not p.exists():
                    rejected.append((tag, s, r, "missing", np.nan)); continue
                d = pd.read_csv(p, sep="\t")
                bad = None
                if len(d) != N_ALL:                                   bad = "row count"
                elif not (d.isoform_id.values == ids0).all():          bad = "isoform_id order"
                elif not (d.label.values.astype(int) == y0).all():     bad = "label vector"
                mx = float(d.residual.abs().max())
                if bad is None and mx >= TOL:                          bad = "residual"
                if bad: rejected.append((tag, s, r, bad, mx))
                else:
                    accepted += 1
                    store[(tag, s, r)] = d.loc[y0 == 1, ["prediction", "expected_value",
                                                         "shap_structural", "shap_stop",
                                                         "shap_atg"]].to_numpy(np.float64)

    log(f"accepted {accepted}   rejected {len(rejected)}   of 50")
    for t, s, r, why, mx in rejected:
        log(f"      REJECT {t} seed{s} run{r}: {why} (max|residual| {mx:.3e})")
    if rejected:
        raise SystemExit("STOPPED: a rejected cell is a defect to diagnose, not a cell to drop")
    log(f"      every cell restricted to the {N_NMD:,} label==1 isoforms for summarisation")
    return store, test_mask


# ---------------------------------------------------------------------------------------
def cell_stats(a):
    """Reduce one (9321 x 5) block -- prediction, E[f], and three SIGNED phi -- to a row.

    `a` columns: prediction, expected_value, phi_structural, phi_stop, phi_start.
    """
    pred, ev, phi = a[:, 0], a[:, 1], a[:, 2:]
    absphi = np.abs(phi)
    m = absphi.mean(axis=0)
    tot = m.sum()
    pct = 100 * m / tot
    row = dict(n_isoforms_summarised=int(a.shape[0]),
               m_total=float(tot),
               ratio_stop_start=float(m[1] / m[2]),
               mean_pred_logodds=float(pred.mean()),
               mean_expected_value=float(ev.mean()),
               mean_gap=float((pred - ev).mean()))
    for k, b in enumerate(BRANCH):
        row[f"m_{NICE[b]}"]           = float(m[k])
        row[f"pct_{NICE[b]}"]         = float(pct[k])
        row[f"mean_signed_{NICE[b]}"] = float(phi[:, k].mean())
        row[f"sd_abs_{NICE[b]}"]      = float(absphi[:, k].std(ddof=1))
        row[f"median_abs_{NICE[b]}"]  = float(np.median(absphi[:, k]))
        row[f"q25_{NICE[b]}"]         = float(np.percentile(absphi[:, k], 25))
        row[f"q75_{NICE[b]}"]         = float(np.percentile(absphi[:, k], 75))
        row[f"frac_pos_{NICE[b]}"]    = float((phi[:, k] > 0).mean())
    # The efficiency identity, recomputed here rather than trusted from the file.
    row["additivity_gap_vs_signed_sum"] = float(abs(row["mean_gap"] - sum(
        row[f"mean_signed_{NICE[b]}"] for b in BRANCH)))
    return row


def step6_reduce(store, mask=None, quiet=False):
    """Build the 110-row per-cell table: member, ensemble, and leave-one-out ensemble.

    `mask` selects a sub-population of the summarised isoforms. None is the primary population
    (all 9,321 label==1); the test-split sensitivity in step 7 passes the 2,405 that lie in
    split=="test", which is the population the published figures were computed over.
    """
    if not quiet:
        step(6, "reduce every cell — member, ensemble, and leave-one-out ensemble")
    sel = (lambda a: a) if mask is None else (lambda a: a[mask])
    rows = []
    for tag in TAGS:
        for s in SEEDS:
            for r in DRAWS:
                rows.append(dict(config=tag, level="member", member_set=str(s),
                                 member_seed_excluded="NA", n_members=1, draw_id=r,
                                 **cell_stats(sel(store[(tag, s, r)]))))
        for r in DRAWS:
            # SIGNED average across members FIRST -- this is the ensemble's own decomposition,
            # exact because Shapley is linear in the value function.
            ens = np.mean([sel(store[(tag, s, r)]) for s in SEEDS], axis=0)
            rows.append(dict(config=tag, level="ensemble",
                             member_set=",".join(map(str, SEEDS)), member_seed_excluded="NA",
                             n_members=5, draw_id=r, **cell_stats(ens)))
        for excl in SEEDS:
            keep = [s for s in SEEDS if s != excl]
            for r in DRAWS:
                loo = np.mean([sel(store[(tag, s, r)]) for s in keep], axis=0)
                rows.append(dict(config=tag, level="ensemble_loo",
                                 member_set=",".join(map(str, keep)),
                                 member_seed_excluded=str(excl), n_members=4, draw_id=r,
                                 **cell_stats(loo)))
    df = pd.DataFrame(rows)
    if quiet:
        return df
    check("per-cell table has the specified shape",
          len(df) == 110 and len(df[df.level == "member"]) == 50,
          f"{len(df)} rows = 2 configs x (25 member + 5 ensemble + 25 leave-one-out)")
    s = df[["pct_structural", "pct_stop", "pct_start"]].sum(axis=1)
    check("the three shares sum to 100 in every row",
          bool(np.abs(s - 100).max() < 1e-9), f"max |sum - 100| = {np.abs(s - 100).max():.2e}")
    check("efficiency identity holds in every row",
          float(df.additivity_gap_vs_signed_sum.max()) < 1e-9,
          f"max |mean_gap - sum(mean_signed)| = {df.additivity_gap_vs_signed_sum.max():.3e}")
    return df


# ---------------------------------------------------------------------------------------
def anova2(y):
    """Two-way random-effects components from a balanced 5x5 grid, one observation per cell.

    Identified despite n=1 per cell BECAUSE a cell carries no within-cell noise: the Shapley
    enumeration is exact and the reference set is fixed, so the cell is a deterministic
    function of (member, draw) and MS_AB estimates the interaction rather than being aliased
    with a residual. E[MS_A] = s_AB^2 + 5 s_A^2; E[MS_B] = s_AB^2 + 5 s_B^2; E[MS_AB] = s_AB^2.
    """
    a, b = y.shape
    gm, rm, cm = y.mean(), y.mean(axis=1), y.mean(axis=0)
    ms_a  = b * ((rm - gm) ** 2).sum() / (a - 1)
    ms_b  = a * ((cm - gm) ** 2).sum() / (b - 1)
    ms_ab = ((y - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((a - 1) * (b - 1))
    return dict(ms_member=ms_a, ms_draw=ms_b, ms_interaction=ms_ab,
                var_member=max(0.0, (ms_a - ms_ab) / b),
                var_draw=max(0.0, (ms_b - ms_ab) / a),
                var_interaction=ms_ab,
                f_member=ms_a / ms_ab if ms_ab > 0 else float("inf"),
                f_draw=ms_b / ms_ab if ms_ab > 0 else float("inf"),
                df=(a - 1, b - 1, (a - 1) * (b - 1)),
                f_crit_95=3.01)


def step7_summarise(df, sens=None):
    """The two spreads at their own levels, the ensemble jackknife, and the ratio."""
    step(7, "spreads, ensemble jackknife, ratio, and the licensing diagnostic")
    out = {}
    for tag in TAGS:
        d = df[df.config == tag]
        mem = d[d.level == "member"].pivot(index="member_set", columns="draw_id")
        ens = d[d.level == "ensemble"].sort_values("draw_id")
        loo = d[d.level == "ensemble_loo"]
        log(f"\n--- {tag} ---")
        res = {"n_isoforms_summarised": int(ens.n_isoforms_summarised.iloc[0]),
               "population": "isoforms with label == 1, at ORF rank 0, over the whole "
                             "labelled universe; a census of that universe, not a sample",
               "reference_draw_seeds": {str(r): 42 + 1000 * r for r in DRAWS}}

        for b in BRANCH:
            nb = NICE[b]
            e = ens[f"pct_{nb}"].to_numpy()                       # 5 ensemble values
            point, sd_draw_ens = e.mean(), e.std(ddof=1)
            # Delete-one-MEMBER jackknife: the ensemble's own seed uncertainty.
            j = np.array([loo[loo.member_seed_excluded == str(s)][f"pct_{nb}"].mean()
                          for s in SEEDS])
            sd_seed_ens = float(np.sqrt((len(SEEDS) - 1) / len(SEEDS) * ((j - j.mean()) ** 2).sum()))
            g = mem[f"pct_{nb}"].to_numpy()                       # 5 members x 5 draws
            member_pct, draw_pct = g.mean(axis=1), g.mean(axis=0)
            res[nb] = dict(
                point_estimate=float(point),
                sd_draw_ensemble=float(sd_draw_ens), range_draw_ensemble=[float(e.min()), float(e.max())],
                sd_seed_ensemble_jackknife=sd_seed_ens,
                member_centre=float(member_pct.mean()),
                sd_train_member=float(member_pct.std(ddof=1)),
                range_train_member=[float(member_pct.min()), float(member_pct.max())],
                sd_draw_member=float(draw_pct.std(ddof=1)),
                ensemble_minus_member_centre=float(point - member_pct.mean()),
                anova=anova2(g))
            log(f"  {nb:11s} {point:6.2f}%   draw-sd {sd_draw_ens:.3f}  seed-jackknife "
                f"{sd_seed_ens:.3f}  | members {member_pct.min():.2f}-{member_pct.max():.2f} "
                f"(centre {member_pct.mean():.2f}, sd {member_pct.std(ddof=1):.3f})")
            log(f"              ensemble centre - member centre = "
                f"{point - member_pct.mean():+.3f} pp   "
                f"[F_member {res[nb]['anova']['f_member']:.2f}, "
                f"F_draw {res[nb]['anova']['f_draw']:.2f}, crit 3.01 on (4,16)]")

        # RATIO at ensemble level, on the log scale (geometric mean).
        re_ = ens["ratio_stop_start"].to_numpy()
        lj = np.array([np.log(loo[loo.member_seed_excluded == str(s)]["ratio_stop_start"]).mean()
                       for s in SEEDS])
        rm_ = np.exp(np.log(mem["ratio_stop_start"].to_numpy()).mean(axis=1))
        res["ratio_stop_start"] = dict(
            geometric_mean=float(np.exp(np.log(re_).mean())),
            sd_draw_ensemble_log_factor=float(np.exp(np.log(re_).std(ddof=1))),
            range_draw_ensemble=[float(re_.min()), float(re_.max())],
            sd_seed_ensemble_jackknife_log_factor=float(np.exp(np.sqrt(
                (len(SEEDS) - 1) / len(SEEDS) * ((lj - lj.mean()) ** 2).sum()))),
            member_geometric_mean=float(np.exp(np.log(rm_).mean())),
            range_train_member=[float(rm_.min()), float(rm_.max())],
            sd_train_member_log_factor=float(np.exp(np.log(rm_).std(ddof=1))),
            cv_member_cells=float(mem["ratio_stop_start"].to_numpy().std(ddof=1)
                                  / mem["ratio_stop_start"].to_numpy().mean()),
            definition="ratio of two MEANS of |phi| over the summarised isoforms, "
                       "not the mean of per-isoform ratios",
            anova_log_ratio=anova2(np.log(mem["ratio_stop_start"].to_numpy())))
        r = res["ratio_stop_start"]
        log(f"  stop/start  {r['geometric_mean']:.3f}   draw x/÷ "
            f"{r['sd_draw_ensemble_log_factor']:.4f}  seed-jackknife x/÷ "
            f"{r['sd_seed_ensemble_jackknife_log_factor']:.4f}  | members "
            f"{r['range_train_member'][0]:.3f}-{r['range_train_member'][1]:.3f}  "
            f"CV {r['cv_member_cells']:.4f}")

        # THE CHECK THAT LICENSES REPORTING PERCENTAGES RATHER THAN LOG-RATIO COORDINATES.
        p = ens[[f"pct_{NICE[b]}" for b in BRANCH]].to_numpy() / 100
        arith = 100 * p.mean(axis=0)
        g_ = np.exp(np.log(p).mean(axis=0)); closed = 100 * g_ / g_.sum()
        res["compositional_check"] = dict(
            arithmetic_mean=arith.tolist(), closed_geometric_mean=closed.tolist(),
            max_abs_difference_pp=float(np.abs(arith - closed).max()),
            decision="report percentages" if np.abs(arith - closed).max() < 0.05
                     else "spread large enough that log-ratio coordinates matter")
        log(f"  compositional check: arithmetic vs closed geometric differ by at most "
            f"{res['compositional_check']['max_abs_difference_pp']:.4f} pp -> "
            f"{res['compositional_check']['decision']}")
        # SENSITIVITY: the population the published figures used.
        if sens is not None:
            sd = sens[(sens.config == tag) & (sens.level == "ensemble")].sort_values("draw_id")
            res["sensitivity_test_clean"] = {
                "n_isoforms_summarised": int(sd.n_isoforms_summarised.iloc[0]),
                "population": "split == 'test' and label == 1",
                **{NICE[b]: dict(point_estimate=float(sd[f"pct_{NICE[b]}"].mean()),
                                 sd_draw_ensemble=float(sd[f"pct_{NICE[b]}"].std(ddof=1)))
                   for b in BRANCH},
                "ratio_stop_start_geometric_mean":
                    float(np.exp(np.log(sd["ratio_stop_start"]).mean()))}
            t = res["sensitivity_test_clean"]
            log(f"  sensitivity (test_clean, n={t['n_isoforms_summarised']:,}): "
                f"structural {t['structural']['point_estimate']:.2f}%  "
                f"stop {t['stop']['point_estimate']:.2f}%  "
                f"start {t['start']['point_estimate']:.2f}%  "
                f"ratio {t['ratio_stop_start_geometric_mean']:.3f}")
            for b in BRANCH:
                log(f"              {NICE[b]:11s} primary - sensitivity = "
                    f"{res[NICE[b]]['point_estimate'] - t[NICE[b]]['point_estimate']:+.2f} pp")
        out[tag] = res
    return out


def step8_write(df, summary):
    step(8, "write")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "branch_shares_per_cell.tsv"
    df.to_csv(p, sep="\t", index=False)
    log(f"wrote {p}  ({p.stat().st_size:,} bytes, {len(df)} rows, {len(df.columns)} columns)")
    for tag, res in summary.items():
        q = OUT / f"branch_decomposition_{tag}.json"
        q.write_text(json.dumps(dict(
            tag=tag, written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            spec="ANALYSIS2_PLAN steps 7-8 as amended by the statistical review 2026-07-31",
            per_cell_table=p.name, **res), indent=2))
        log(f"wrote {q}  ({q.stat().st_size:,} bytes)")


def main():
    log(f"Analysis 2 — branch decomposition.  {datetime.now().isoformat(timespec='seconds')}")
    log(f"numpy {np.__version__}, pandas {pd.__version__}")
    store, test_mask = step5_accept()
    df = step6_reduce(store)
    sens = step6_reduce(store, mask=test_mask, quiet=True)
    summary = step7_summarise(df, sens)
    step8_write(df, summary)
    _ind[0] = ""
    log(f"\nDone. {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
