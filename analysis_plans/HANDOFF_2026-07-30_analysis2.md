# Handoff — Analysis 2 is specified and unblocked, not yet run

Written 2026-07-30. Everything referenced here is committed in this repository under
`analysis_plans/`, on branch `master`. The repository is private until submission.

---

## Start here

Read, in order: `ANALYSIS1_PLAN_2026-07-30.md`, then `ANALYSIS2_PLAN_2026-07-30.md`. They are the
specification; code is written from them, not the other way round. `analysis1.py` is the worked
example of what an implementation of one looks like, and `analysis1_runlog.txt` is what its output
should look like — every check prints the value it measured, so the log is the evidence rather than a
claim about it.

Then read the two memory files on how these plans are written and on verifying against artifacts
rather than from memory. Six statements in these plans were wrong when first written and were caught
only by opening the file or running the code.

## State

**Analysis 1 is complete.** Both configurations scored on `test_clean`, n = 10,520.

| configuration | ensemble AUC | 95% CI | ensemble AUPRC | 95% CI | member mean ± sd |
|---|---|---|---|---|---|
| `atg2000_stop2000` | 0.93254 | 0.92648–0.93796 | 0.83065 | 0.81511–0.84427 | 0.92551 ± 0.00192 |
| `atg1000_stop1000` | 0.93102 | 0.92501–0.93697 | 0.83510 | 0.82046–0.84861 | 0.92359 ± 0.00241 |

Outputs are in `results_interp_all/`, which is gitignored — they are data for the Zenodo deposit
under D38, not code. Pete has reviewed these and accepts that they round to the published 0.93 / 0.83
at two decimal places.

**Analysis 2 is specified and not run.** Nothing in `results_interp_all/` relates to it.

## What was fixed today, and why it matters before you run anything

**`11_kernel_shap_branches.py` gained three changes and had never run end to end.** A replicate slot
in the output filename (it overwrote its own reference draws), a `--checkpoint-dir` separate from the
output directory, and the split gate it had never called. `test_guards.py` sections 4–6 cover them.

**`NMDDataset` gained `restrict_to`.** The script built the whole `train` split — 9.0 GB at
`atg1000_stop1000`, 17.9 GB at `atg2000_stop2000` — to use 500 of its 26,711 rows as the SHAP
reference set. With the explained cohort alongside, one run at the wider configuration needed 46 GB
against a 32 GB SLURM request. Found by running it, not by reading it. Every added executable line is
inside `if restrict_to is not None`, so the default path is unchanged by construction.

**`11_kernel_shap_branches.py` is wired to use it**, and the equivalence was measured rather than
assumed. The reference isoforms are the same set and their branch embeddings are identical to exactly
zero. The Shapley values differ by 1.2e-08 — the reference set is now held in sorted rather than drawn
order, and `evaluate_coalition` averages over it, so floating-point summation order changes the last
bits. That is an order of magnitude below float32 machine epsilon (1.19e-07) and four orders below
the 1e-12 residual tolerance. Mathematically identical, numerically identical to within summation
order; not bitwise identical, and it should not be.

Reference set: 172 MB instead of 9.0 GB. One full-cohort run at `atg2000_stop2000` now needs ~28 GB
rather than 46.

## First tasks, in order

1. **Re-measure memory and wall time** for one run at each configuration over `--explain-split all`.
   ~28 GB at `atg2000_stop2000` fits a 32 GB request with little headroom. Size the request from the
   measurement, not from the existing script.
2. **Run one cell end to end on the cluster** — one configuration, one member, one draw — and check
   the residual is at machine precision before committing to the other 49.
3. **Then the remaining 49.**

## Traps that have already cost time here

**Do not loosen a consumer glob to make it find member-tagged files.** It will match all five members
and pool them, collapsing the between-member spread, which is the training-variability estimate the
whole design exists to produce. No error, no warning, a plausible number. `utils.assert_one_member`
guards the one site that globs; the real fix is a `--member-seed` on the consumer.

**The checkpoint's embedded config does not describe the model it belongs to.** In
`best_model_atg2000_stop2000_seed100.pt` it reports window widths of 100 and 1000. The widths come
from the tag.

**Cluster failures here are operational, not analytical.** The recorded ones: a job that died at 17
seconds having produced no log because `slurm_logs/` did not exist when SLURM opened `--output`; an
exit code reporting success while the log reported failure, because the script ended on an `echo`; a
process killed silently by a login node; an oversized time request excluded from backfill and left
pending an hour beside idle nodes. Ask Pete before every cluster login.

## Outstanding: statistical review of Steps 7 and 8

**Pete is doing this in the new window.** It has not been done. Four things I noticed while reading
those steps, recorded as questions for that review rather than as findings, so they are not lost and
not pre-judged:

1. **The two spreads describe different objects.** The interpretation spread is the spread of the
   ENSEMBLE's percentage across five reference draws; the training spread is the spread of MEMBER
   percentages across five members. Averaging five members reduces draw noise, so the first is
   ensemble-level and the second member-level. Reported side by side as `value ± a ± b` they invite a
   comparison that may not be valid. Related: the ensemble has one training realization, so its own
   training uncertainty is not estimable from these data at all — the member spread bounds it only
   loosely, since members share training data and are not independent.

2. **The design is crossed, and the estimators are marginal.** The same five reference draws are used
   for every member, so member × draw is a fully crossed 5×5. That supports separating member
   variance, draw variance and their interaction. The marginal standard deviations as specified fold
   the interaction into the components rather than isolating it.

3. **The three shares are compositional.** They sum to 100% by construction, so they are not
   independent and a standard deviation on each is descriptive only — a rise in one is necessarily a
   fall in another, and any joint statement about two of them is one fact, not two. The plan notes the
   sum constraint but does not draw this consequence.

4. **The ratio's summary may not suit its distribution.** A ratio of two positive quantities is
   right-skewed, so a mean and standard deviation over the 25 cells may describe it poorly. A median
   with a range, or moments on the log ratio, would be more robust.

And one gap in Step 8 rather than a question: **it records the summary but not the 25 per-cell
values.** Without those, the spreads cannot be recomputed or re-estimated by a different method, and
the summary has to be taken on trust. The per-cell table should be written alongside it.

## Open decisions

- **Whether the reference draw stays uniform.** Kept, so any movement in the branch shares is
  attributable to the model rather than to a redefined baseline. It is unstratified, so "absent"
  means "a typical training transcript" — about 22% NMD susceptible — not "a control".
- **Subgroup vocabulary for Analyses 3–5.** Section 4's terms apply, from SF26: PTC+, PTC−, *Ref AUG
  absent*, Control. Panel G already restricts to reference-AUG-traceable isoforms by construction and
  Pete has confirmed that scope stands, which means claims 5.4.3 and 5.6.9 are computed over section
  4's gene-matched pair scope, not the model's universe. n = 54 for the PTC− arm.
- **Transcriptome-wide prediction** over all 162,800 expressed isoforms was raised and parked. It is
  a substrate rebuild rather than a scoring change, and a new claim rather than a repair. Decide once
  the interpretation results exist.
