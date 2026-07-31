#!/usr/bin/env python3
"""
analysis3.py — steps 5 to 8 of ANALYSIS3_PLAN: reduce the fifty structural-feature
attribution tables to feature shares and their spreads.

Sibling of analysis2.py, and deliberately the same shape: same three levels, same two
spreads at their own levels, same jackknife, same variance components, same compositional
check, same census framing, same test-split sensitivity. What differs:

  TWO GAMES, NOT ONE. Each cell carries an exact five-player decomposition over the
  structural features AND an exact four-player decomposition in which the two annotation
  flags move together. They are SEPARATE GAMES, not a regrouping: merging players changes
  the normaliser, and adding two mean-absolute shares overstates the merged contribution
  wherever the two attributions differ in sign. Measured on a probe: the naive sum gives
  30.1% against the correct 25.4%. Both are reported; neither is derived from the other.

  NO RATIO. Analysis 2's stop-to-start ratio existed because two comparably sized branches
  invited a ranking. With five parts and no natural pairing, a ratio would name a
  comparison nobody asked for.

  A COMPOSITIONAL FAILURE BRANCH THAT MAY ACTUALLY FIRE. Analysis 2 passed its check by two
  orders of magnitude with three comparable parts. Five parts with one expected to dominate
  may not, so when the gap reaches the threshold the shares are additionally reported in
  centred log-ratio coordinates and the percentages are marked descriptive only.

EVERY CHECK PRINTS THE VALUE IT MEASURED. Read the log beside the plan.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python analysis3.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO  = Path.home() / "claude_projects" / "NMD_orf_model_v5_4ct"
CELLS = REPO / "results_interp_all"
OUT   = REPO / "results_interp_all"
H5    = REPO / "results_4ct_dn" / "nmd_orf_data.h5"

TAGS  = ["atg2000_stop2000", "atg1000_stop1000"]
SEEDS = [100, 200, 300, 400, 500]
DRAWS = [1, 2, 3, 4, 5]

# The two games. `parts` are the players, in the order their columns appear.
GAMES = {
    "five_player": dict(
        parts=["frac_start", "frac_stop", "is_ref_cds", "is_sqanti_cds", "n_downstream_ejc"],
        prefix="shap_", residual="residual"),
    "four_player": dict(
        parts=["frac_start", "frac_stop", "annotation", "n_downstream_ejc"],
        prefix="shap4_", residual="residual4"),
}

TOL        = 1e-12          # within-file algebraic identity, BOTH games
XREF_TOL   = 1e-3           # cross-code-path, cross-architecture; see analysis2.py
N_ALL      = 41765
N_NMD      = 9321
N_TEST_NMD = 2405
N_CELLS    = len(TAGS) * len(SEEDS) * len(DRAWS)
F_CRIT_4_16_95 = 3.0069
COMP_THRESHOLD_PP = 0.05

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
    return CELLS / f"feature_shap_{tag}_seed{seed}_all_run{draw}.tsv"


# ---------------------------------------------------------------------------------------
def step5_accept():
    """Accept or reject each of the fifty files, on BOTH games, and record the tally."""
    step(5, "accept or reject each file, and record the tally")
    import h5py
    with h5py.File(H5, "r") as f:
        ids0 = np.array([x.decode() if isinstance(x, bytes) else x for x in f["isoform_id"][:]])
        y0 = f["labels"][:].astype(int)
        sp = np.array([x.decode() if isinstance(x, bytes) else x for x in f["split"][:]])
    check("step 1 population re-read from Dataset A",
          len(ids0) == N_ALL and int(y0.sum()) == N_NMD,
          f"n={len(ids0):,}, label==1 {int(y0.sum()):,}")
    test_mask = sp[y0 == 1] == "test"
    check("test-split sensitivity population", int(test_mask.sum()) == N_TEST_NMD,
          f"{int(test_mask.sum()):,} of {len(test_mask):,} summarised isoforms in split=='test'")

    # The independent per-isoform source. Analysis 2's `prediction` is the SAME quantity --
    # the plain model output with everything observed -- computed by a different script in a
    # different game. It is what ties these values to the isoforms they belong to; row count,
    # id order, labels and the residual are all satisfied by a file whose value columns were
    # permuted together.
    xref = {}
    for tag in TAGS:
        for s in SEEDS:
            q = CELLS / f"kernel_shap_branch_{tag}_seed{s}_all_run1.tsv"
            if q.exists():
                xref[(tag, s)] = pd.read_csv(q, sep="\t", usecols=["isoform_id", "prediction"])
    check("independent predictions available from Analysis 2", len(xref) == len(TAGS) * len(SEEDS),
          f"{len(xref)} of {len(TAGS) * len(SEEDS)} (configuration, member) pairs")

    store, accepted, rejected, xdev = {}, 0, [], []
    r5, r4 = [], []
    for tag in TAGS:
        for s in SEEDS:
            for r in DRAWS:
                p = path_of(tag, s, r)
                if not p.exists():
                    rejected.append((tag, s, r, "missing", np.nan)); continue
                d = pd.read_csv(p, sep="\t")
                bad, worst = None, np.nan
                if len(d) != N_ALL:                                bad = "row count"
                elif not (d.isoform_id.values == ids0).all():       bad = "isoform_id order"
                elif not (d.label.values.astype(int) == y0).all():  bad = "label vector"
                if bad is None:
                    a = float(d.residual.abs().max()); b = float(d.residual4.abs().max())
                    r5.append(a); r4.append(b); worst = max(a, b)
                    if worst >= TOL: bad = f"residual ({worst:.2e})"
                if bad is None and (tag, s) in xref:
                    dev = float(np.abs(d.prediction.values
                                       - xref[(tag, s)].prediction.values).max())
                    xdev.append(dev)
                    if dev >= XREF_TOL: bad = f"prediction vs Analysis 2 ({dev:.2e})"
                if bad:
                    rejected.append((tag, s, r, bad, worst))
                else:
                    accepted += 1
                    cols = ["prediction", "baseline_struct_reference"]
                    for g in GAMES.values():
                        cols += [f"{g['prefix']}{p_}" for p_ in g["parts"]]
                    store[(tag, s, r)] = d.loc[y0 == 1, cols].to_numpy(np.float64)

    log(f"accepted {accepted}   rejected {len(rejected)}   of {N_CELLS}")
    for t, s, r, why, w in rejected:
        log(f"      REJECT {t} seed{s} run{r}: {why}")
    if rejected:
        raise SystemExit("STOPPED: a rejected cell is a defect to diagnose, not a cell to drop")
    log(f"      5-player residual: max {max(r5):.3e}   median {np.median(r5):.3e}")
    log(f"      4-player residual: max {max(r4):.3e}   median {np.median(r4):.3e}")
    log(f"      predictions agree with Analysis 2 on {len(xdev)} cells: "
        f"max deviation {max(xdev):.2e} (bar {XREF_TOL:.0e})")
    check("every cell of the grid is present and accepted", accepted == N_CELLS,
          f"{accepted} of {N_CELLS} = {len(TAGS)} configs x {len(SEEDS)} members x {len(DRAWS)} draws")
    return store, test_mask


# ---------------------------------------------------------------------------------------
def _game_slice(game):
    """Column offsets of one game's parts within the stored block."""
    off = 2
    for name, g in GAMES.items():
        if name == game:
            return slice(off, off + len(g["parts"]))
        off += len(g["parts"])
    raise KeyError(game)


def cell_stats(a, mask=None):
    """Reduce one stored block to a row, for BOTH games."""
    if mask is not None:
        a = a[mask]
    pred, base = a[:, 0], a[:, 1]
    row = dict(n_isoforms_summarised=int(a.shape[0]),
               mean_pred_logodds=float(pred.mean()),
               mean_baseline_struct_reference=float(base.mean()),
               mean_gap=float((pred - base).mean()))
    for game, g in GAMES.items():
        phi = a[:, _game_slice(game)]
        ab = np.abs(phi)
        m = ab.mean(axis=0)
        tot = m.sum()
        row[f"{game}_m_total"] = float(tot)
        for k, p_ in enumerate(g["parts"]):
            row[f"{game}_m_{p_}"] = float(m[k])
            row[f"{game}_pct_{p_}"] = float(100 * m[k] / tot)
            row[f"{game}_mean_signed_{p_}"] = float(phi[:, k].mean())
            row[f"{game}_sd_abs_{p_}"] = float(ab[:, k].std(ddof=1))
            row[f"{game}_frac_pos_{p_}"] = float((phi[:, k] > 0).mean())
        # Each game has its own efficiency identity; both must hold on the same rows.
        row[f"{game}_efficiency_gap"] = float(abs(row["mean_gap"] - phi.mean(axis=0).sum()))
    return row


def step6_reduce(store, mask=None, quiet=False):
    """Member, ensemble and leave-one-out ensemble rows — 110, for both games at once."""
    if not quiet:
        step(6, "reduce every cell — member, ensemble, and leave-one-out ensemble")
    rows = []
    for tag in TAGS:
        for s in SEEDS:
            for r in DRAWS:
                rows.append(dict(config=tag, level="member", member_set=str(s),
                                 member_seed_excluded="NA", n_members=1, draw_id=r,
                                 **cell_stats(store[(tag, s, r)], mask)))
        for r in DRAWS:
            ens = np.mean([store[(tag, s, r)] for s in SEEDS], axis=0)
            rows.append(dict(config=tag, level="ensemble",
                             member_set=",".join(map(str, SEEDS)), member_seed_excluded="NA",
                             n_members=5, draw_id=r, **cell_stats(ens, mask)))
        for excl in SEEDS:
            keep = [s for s in SEEDS if s != excl]
            for r in DRAWS:
                loo = np.mean([store[(tag, s, r)] for s in keep], axis=0)
                rows.append(dict(config=tag, level="ensemble_loo",
                                 member_set=",".join(map(str, keep)),
                                 member_seed_excluded=str(excl), n_members=4, draw_id=r,
                                 **cell_stats(loo, mask)))
    df = pd.DataFrame(rows)
    if quiet:
        return df
    check("per-cell table has the specified shape",
          len(df) == 110 and len(df[df.level == "member"]) == 50,
          f"{len(df)} rows = 2 configs x (25 member + 5 ensemble + 25 leave-one-out)")
    for game, g in GAMES.items():
        gap = (df[[f"{game}_m_{p_}" for p_ in g["parts"]]].sum(axis=1)
               - df[f"{game}_m_total"]).abs().max()
        check(f"{game}: m_total is the normaliser the shares were formed from",
              float(gap) < 1e-12, f"max |sum(m) - m_total| = {gap:.2e}")
        e = df[f"{game}_efficiency_gap"].max()
        check(f"{game}: efficiency identity holds in every row", float(e) < 1e-9,
              f"max |mean_gap - sum(mean_signed)| = {e:.3e}")
    return df


# ---------------------------------------------------------------------------------------
def anova2(y):
    """Balanced two-way random model, one observation per cell. See analysis2.py."""
    a, b = y.shape
    gm, rm, cm = y.mean(), y.mean(axis=1), y.mean(axis=0)
    ms_a = b * ((rm - gm) ** 2).sum() / (a - 1)
    ms_b = a * ((cm - gm) ** 2).sum() / (b - 1)
    ms_ab = ((y - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((a - 1) * (b - 1))
    return dict(ms_member=ms_a, ms_draw=ms_b, ms_interaction=ms_ab,
                var_member=max(0.0, (ms_a - ms_ab) / b),
                var_draw=max(0.0, (ms_b - ms_ab) / a), var_interaction=ms_ab,
                var_member_truncated=bool(ms_a < ms_ab), var_draw_truncated=bool(ms_b < ms_ab),
                f_member=ms_a / ms_ab if ms_ab > 0 else float("inf"),
                f_draw=ms_b / ms_ab if ms_ab > 0 else float("inf"),
                df=(a - 1, b - 1, (a - 1) * (b - 1)), f_crit_95=F_CRIT_4_16_95)


def compositional_check(P):
    """Arithmetic mean of compositions against the closed geometric mean, with a FAILURE
    BRANCH. P is (n_compositions, n_parts) in percent."""
    p = P / 100
    arith = 100 * p.mean(axis=0)
    zero = (p <= 0).any()
    if zero:
        return dict(arithmetic_mean=arith.tolist(), closed_geometric_mean=None,
                    max_abs_difference_pp=None, zero_part_present=True,
                    decision="a part measured zero; the log-ratio form is undefined and the "
                             "percentages are reported as descriptive only")
    g = np.exp(np.log(p).mean(axis=0)); closed = 100 * g / g.sum()
    gap = float(np.abs(arith - closed).max())
    out = dict(arithmetic_mean=arith.tolist(), closed_geometric_mean=closed.tolist(),
               max_abs_difference_pp=gap, zero_part_present=False,
               threshold_pp=COMP_THRESHOLD_PP)
    if gap < COMP_THRESHOLD_PP:
        out["decision"] = "report percentages"
    else:
        # THE BRANCH THAT MAY ACTUALLY FIRE HERE. Centred log-ratio coordinates, whose centre
        # IS the closed geometric mean by construction.
        clr = np.log(p) - np.log(p).mean(axis=1, keepdims=True)
        out["decision"] = ("spread large enough that percentages are descriptive only; "
                           "centred log-ratio coordinates reported alongside")
        out["clr_mean"] = clr.mean(axis=0).tolist()
        out["clr_sd"] = clr.std(axis=0, ddof=1).tolist()
    return out


def step7_summarise(df, sens=None):
    step(7, "each spread at its own level, for both games")
    out = {}
    for tag in TAGS:
        d = df[df.config == tag]
        ens = d[d.level == "ensemble"].sort_values("draw_id")
        loo = d[d.level == "ensemble_loo"]
        mem = d[d.level == "member"]
        log(f"\n--- {tag} ---")
        res = {"n_isoforms_summarised": int(ens.n_isoforms_summarised.iloc[0]),
               "population": "isoforms with label == 1 at ORF rank 0; a census of the labelled "
                             "universe, not a sample",
               "reference_draw_seeds": {str(r): 42 + 1000 * r for r in DRAWS}}

        for game, g in GAMES.items():
            log(f"  [{game}]")
            res[game] = {}
            for p_ in g["parts"]:
                col = f"{game}_pct_{p_}"
                e = ens[col].to_numpy()
                point, sd_draw = e.mean(), e.std(ddof=1)
                se_draw = sd_draw / np.sqrt(len(DRAWS))
                j = np.array([loo[loo.member_seed_excluded == str(s)][col].mean() for s in SEEDS])
                sd_seed = float(np.sqrt((len(SEEDS) - 1) / len(SEEDS) * ((j - j.mean()) ** 2).sum()))
                bias = float((len(SEEDS) - 1) * (j.mean() - point))
                gmat = mem.pivot(index="member_set", columns="draw_id", values=col).to_numpy()
                mp, dp = gmat.mean(axis=1), gmat.mean(axis=0)
                res[game][p_] = dict(
                    point_estimate=float(point),
                    sd_draw_ensemble=float(sd_draw), se_draw_ensemble=float(se_draw),
                    range_draw_ensemble=[float(e.min()), float(e.max())],
                    sd_seed_ensemble_jackknife=sd_seed,
                    range_seed_ensemble_jackknife=[float(j.min()), float(j.max())],
                    ensemble_size_bias_jackknife=bias,
                    infinite_ensemble_estimate=float(point - bias),
                    member_centre=float(mp.mean()), sd_train_member=float(mp.std(ddof=1)),
                    range_train_member=[float(mp.min()), float(mp.max())],
                    sd_draw_member=float(dp.std(ddof=1)),
                    range_all_cells=[float(gmat.min()), float(gmat.max())],
                    ensemble_minus_member_centre=float(point - mp.mean()),
                    anova=anova2(gmat))
                log(f"    {p_:20s} {point:6.2f}%   draw-SD {sd_draw:.3f} (SE {se_draw:.3f})  "
                    f"seed-jk SE {sd_seed:.3f}  bias {bias:+.3f}  | cells "
                    f"{gmat.min():.2f}-{gmat.max():.2f}")
            res[game]["compositional_check"] = compositional_check(
                ens[[f"{game}_pct_{p_}" for p_ in g["parts"]]].to_numpy())
            c = res[game]["compositional_check"]
            log(f"    compositional: gap {c['max_abs_difference_pp']:.4f} pp "
                f"(threshold {COMP_THRESHOLD_PP}) -> {c['decision']}")

        # THE ERROR THIS ANALYSIS EXISTS TO AVOID, computed so it is on the record.
        f5, f4 = res["five_player"], res["four_player"]
        naive = f5["is_ref_cds"]["point_estimate"] + f5["is_sqanti_cds"]["point_estimate"]
        correct = f4["annotation"]["point_estimate"]
        res["annotation_merge"] = dict(
            naive_sum_of_five_player_shares=float(naive), four_player_share=float(correct),
            overstatement_pp=float(naive - correct),
            note="the naive sum is NOT reported as a result; it is recorded to show the size of "
                 "the error that adding two mean-absolute shares would have introduced. Merging "
                 "players also changes the normaliser, so the two are not comparable term by term.")
        log(f"  annotation: four-player {correct:.2f}%   naive sum of the two five-player "
            f"shares {naive:.2f}%   overstatement {naive - correct:+.2f} pp")

        if sens is not None:
            sd = sens[(sens.config == tag) & (sens.level == "ensemble")].sort_values("draw_id")
            res["sensitivity_test_clean"] = {
                "n_isoforms_summarised": int(sd.n_isoforms_summarised.iloc[0]),
                "population": "split == 'test' and label == 1",
                **{game: {p_: dict(
                    point_estimate=float(sd[f"{game}_pct_{p_}"].mean()),
                    sd_draw_ensemble=float(sd[f"{game}_pct_{p_}"].std(ddof=1)))
                    for p_ in g2["parts"]} for game, g2 in GAMES.items()}}
            t = res["sensitivity_test_clean"]
            log(f"  sensitivity (test_clean, n={t['n_isoforms_summarised']:,}): " +
                "  ".join(f"{p_} {t['five_player'][p_]['point_estimate']:.2f}%"
                          for p_ in GAMES["five_player"]["parts"]))
        out[tag] = res
    return out


def step8_write(df, summary):
    step(8, "write")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "feature_shares_per_cell.tsv"
    df.to_csv(p, sep="\t", index=False)
    log(f"wrote {p}  ({p.stat().st_size:,} bytes, {len(df)} rows, {len(df.columns)} columns)")
    for tag, res in summary.items():
        q = OUT / f"feature_decomposition_{tag}.json"
        q.write_text(json.dumps(dict(
            tag=tag, written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            spec="ANALYSIS3_PLAN steps 5-8", per_cell_table=p.name, **res), indent=2))
        log(f"wrote {q}  ({q.stat().st_size:,} bytes)")


def main():
    log(f"Analysis 3 — structural feature decomposition.  "
        f"{datetime.now().isoformat(timespec='seconds')}")
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
