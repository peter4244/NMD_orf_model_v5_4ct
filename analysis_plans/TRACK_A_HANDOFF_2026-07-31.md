# Handoff to Track A — §5 results, one pipeline departure, and cluster facts

Written 2026-07-31 from the model repo (`peter4244/NMD_orf_model_v5_4ct`, branch `master`).

**What this is and why it is prose.** Track B writes no ledger rows and does not run
`build_current_state.py`. Everything below therefore arrives as findings for Track A to turn into
C-rows, D-rows and W-rows, or to reject. Every number carries the measurement behind it and the
artifact it came from, so nothing here has to be taken on trust.

**Two things are flagged as MANUSCRIPT DEFECTS rather than rebuild observations** — they are
statements in the paper that the measurements do not support, independently of any deposit
question. They are marked ⚠ below.

---

## 1. C71's stasis alarm did not fire

C71 records a test-design consequence for the retrain: the pre-clip §5 numbers are baselines, not
targets, and *"unexpected STASIS is the alarm."*

Mid-session I raised exactly that alarm. A single cell — `atg1000_stop1000`, member seed 100,
reference draw 1 — gave structural / stop / start = **59.4 / 30.5 / 10.1**, against the published
full-cohort 60.0 / 29.1 / 10.9. Every branch within about 1.4 points.

**It was one cell out of twenty-five.** The ensemble over all five members and five draws at that
configuration is **66.31 / 23.88 / 9.81**. The apparent agreement was a coincidence in a single
member at a single draw, and across the fifty computed cells the structural share ranges
59.11–77.59%.

**Action for Track A:** C71's alarm can be recorded as raised and cleared. The underlying
instruction — do not quote the ledger's deposit-native §5 numbers as the deposit-native answer —
is unaffected and still stands.

*Source: `results_interp_all/branch_decomposition_atg1000_stop1000.json`, and
`branch_shares_per_cell.tsv` for the per-cell range. Produced by `analysis_plans/analysis2.py`.*

---

## 2. ⚠ Claim 5.12 is configuration-dependent, and false at one configuration

Published: *"STOP sequence ~3× as important as START sequence."*

| | stop-to-start ratio | cells with ratio > 1 | range across 25 cells |
|---|---|---|---|
| `atg1000_stop1000` | **2.435** | 25 / 25 | 1.459 – 3.178 |
| `atg2000_stop2000` | **1.043** | 15 / 25 | 0.782 – 1.495 |

At the narrower configuration the claim is roughly right. At the wider one the two sequence
branches are approximately equal, the ordering inverts in 10 of 25 cells, and on the population the
published figures were computed over (`split == "test"`, n = 2,405) the ratio is **0.966** — start
slightly exceeds stop.

**This is a claim about the model, not about the deposit.** No rebuild question is involved: the
ranking of the two sequence branches depends on a window-width choice, so a sentence asserting it
as biology is not supportable as written.

*Ratios are geometric means over five reference draws at ensemble level, formed within each cell.
Source: the two `branch_decomposition_*.json` files.*

---

## 3. ⚠ Claim 5.11's "roughly ⅔ structural" is much less determined than one number suggests

Published: *"Roughly ⅔ of predictive information from ORF structural data, ⅓ from START+STOP
sequence."*

| | structural | stop | start |
|---|---|---|---|
| `atg2000_stop2000` | **76.24%** | 12.13% | 11.63% |
| `atg1000_stop1000` | **66.31%** | 23.88% | 9.81% |

Across all fifty cells the structural share ranges **59.11 – 77.59%**. It is the largest branch in
50/50 cells and a majority in 50/50 — that part is robust to training seed, reference draw and
window width, and is the defensible claim. The *value* is not: individual members span 69.7–76.9%
at one configuration and 59.7–68.3% at the other.

Three wording problems independent of the number, all of which need fixing alongside it:

1. It is a share of **mean absolute log-odds displacement**, not "predictive information", not
   variance explained, not an information-theoretic quantity.
2. Shapley divides credit among branches carrying overlapping information, so *"the remainder
   coming from sequence information"* asserts a clean partition the method does not support.
3. The published figure comes from **one trained model**. Training variability dominates every
   other source here: `F_member` runs 49.6 – 4321.5 against a 3.0069 critical value on (4,16) df.

**The spread is arguably the finding**, more than the point estimate.

---

## 4. Analysis 1 supersedes the basis of claim 5.9

Published: *"AUC=0.93, AUPRC=0.833 on held-out test set."*

| configuration | ensemble AUC (95% CI) | ensemble AUPRC (95% CI) | member mean AUC ± sd |
|---|---|---|---|
| `atg2000_stop2000` | 0.93254 (0.92648–0.93796) | 0.83065 (0.81511–0.84427) | 0.92551 ± 0.00192 |
| `atg1000_stop1000` | 0.93102 (0.92501–0.93697) | 0.83510 (0.82046–0.84861) | 0.92359 ± 0.00241 |

Scored on `test_clean`, n = 10,520, of which 2,405 NMD susceptible (22.9%). Intervals are
percentile bootstrap over 2,000 isoform resamples, seed 20260730, shared across both
configurations. Both round to the published 0.93 / 0.83 at two decimals.

**A consequence worth noting:** the two configurations' performance intervals overlap almost
entirely, while their structural shares differ by 9.93 percentage points — 5.0× the combined
seed-jackknife scale. Two models that predict equally well divide their evidence measurably
differently. That bears on how strongly any mechanistic reading of the decomposition can be
stated.

---

## 5. A registered prediction that failed

`ANALYSIS2_PLAN` §3d registered, before any cell ran, that the structural share should be **lower**
at `atg2000_stop2000` than at `atg1000_stop1000` — more sequence input, unchanged structural input,
a mechanical rather than biological reason. It was named in advance so it would not later read as a
finding.

Measured, it is **higher: 76.24% against 66.31%**, 9.93 pp in the opposite direction and more than
an order of magnitude larger than any spread in the analysis.

The prediction is retained in the plan rather than deleted. Its practical consequence: the
direction is no longer available as a mechanical explanation for anything downstream. Why widening
the sequence windows *reduces* the sequence branches' share is not established and is not claimed.

---

## 6. A departure from the pre-registered pipeline — decision material

`docs/SECTION5_REDESIGN_SPEC.md` §6 names `deepshap.py` for the per-member × replicate attribution
stage. **Analysis 3 departs from it** for the structural-feature decomposition, using exact
enumeration of all 32 coalitions over the five features instead.

**Reason.** DeepSHAP is approximate. The acceptance discipline both preceding analyses rest on —
the residual as a checksum, hard reject at 1e-12 — depends on exactness, and `ANALYSIS2_PLAN` says
so: *"a hard reject is affordable here and would not be for an approximate method, where a small
non-zero residual is expected and tells you nothing."* With five players, 2⁵ = 32 coalitions
enumerate, and the features enter through `nn.Linear(5, 32)`, so the expensive CNN encodings are
computed once and reused. The checksum survives. Measured on a 500-isoform probe: max residual
**8.88e-16**.

DeepSHAP is still required for the nucleotide-level analysis that follows, where 2²⁰⁰⁰ does not
enumerate. **This departure is scoped to the structural-feature stage only.**

Pete approved this 2026-07-31. It is recorded here because a change to a pre-registered pipeline
belongs in the ledger, not only in a commit message.

---

## 7. ⚠ A live defect in a committed SLURM script

`slurm_kernel_shap_dn.sh` requests **32 GB**. A full-cohort kernel-SHAP run at `atg2000_stop2000`
under the pre-2026-07-30 code peaks at **35.3 GiB** and would be OOM-killed.

The cause: `NMDDataset` materialises a split's windows as float32 on construction — 28.01 GiB for
the full cohort at that width — and `f[...][idx].astype(np.float32)` holds the float16 read buffer
and the float32 result simultaneously, so the transient peak sits half an array above steady state.
Measured rather than derived: at n = 6,000 / w2000 the model predicts 5.31 GiB and the process
peaked at 5.38.

Fixed in the model repo (commit `3741a58`) by reading the explained split in chunks — peak now
**2.10 GiB** at `atg1000_stop1000` and **3.66 GiB** at `atg2000_stop2000`, Linux `sacct` MaxRSS,
largest over 25 cells each. Proven to change no value: eager and chunked outputs are byte-identical
(same md5, all 4,356 rows agreeing to exactly zero) — see
`analysis_plans/chunked_read_equivalence_runlog.txt`.

**The script itself still carries the 32 GB request**, and any consumer running the older code path
against it will hit this.

---

## 8. Cluster reference material

Measured on Explorer 2026-07-30/31. None of this is recorded anywhere durable and all of it cost
time to learn.

| partition | usable nodes | time limit | note |
|---|---|---|---|
| `short` | ~125 of 232 | 2 days | **parallelises job arrays** — 23 concurrent across 15 nodes measured |
| `sharing` | ~455 | **1:00:00** | would have **rejected** our 1:30:00 jobs outright |
| `gpu` | ~24 | 8:00:00 | contended: 83 pending against 66 running |

- **Array tasks queue about twice as slowly as single jobs**: median 102 s against 54–55 s, measured
  on our own submissions. Immaterial for 8-minute jobs; it would matter for short ones.
- **A GPU is ~8% faster per job and the wrong choice for fifty**: the same cell ran 7 min 53 s on
  `gpu` (node d1012) and 8 min 36 s on `short` (node c0625). Fifty finish sooner on `short` because
  of node count, and `slurm_kernel_shap_dn.sh` already records a fivefold spread between GPU node
  families. An earlier 31-minute GPU run is **not** evidence about devices — it ran the pre-chunking
  code under memory pressure.
- **Fifty full-cohort cells complete in 24 minutes** of wall clock as one array, `--array=0-49%20`.
- **`slurm_logs/` must exist before `sbatch`** — SLURM opens `--output` before the script runs.
- **The Explorer HDF5 is content-identical to the local one** but differs by 23,404 bytes of
  internal layout. Verified: row count, all five split counts, label sum, both normalization
  vectors to 7 dp, w2000 row-0 checksums, and full-array sums of `orf_features` and `orf_mask`. The
  file is stored in sorted `isoform_id` order, so row order is canonical rather than
  build-dependent — which is what makes the position-based reference draw reproducible across
  machines. All ten checkpoints are byte-identical by md5.

---

## 9. What is NOT in here

- **No ledger rows have been written, and `build_current_state.py` has not been run.**
- **Analysis 3 is COMPLETE.** See §10 below. Fifty cells, all accepted.
- **Nucleotide-level claims (5.16–5.20) are untouched.**
- The `results_interp_all/` outputs are data, not code, and are gitignored under D38. They are on
  this machine and on Explorer, not in the repository.

---

## 10. Claim 5.14 — SUPPORTED, and more stable than the branch-level claims

Published: *"Of individual ORF structural features, EJC count was by far the most important."*

Fifty cells, two configurations x five members x five reference draws, all accepted (both games'
residuals at or below 3.55e-15 against a 1e-12 bar). Ensemble shares over the 9,321 summarized
isoforms:

| feature | `atg2000_stop2000` | `atg1000_stop1000` | range over all 25 cells |
|---|---|---|---|
| `n_downstream_ejc` | **60.96%** | **64.44%** | 57.8–62.7 / 56.7–69.1 |
| `is_ref_cds` | 15.10% | 17.86% | 12.2–17.6 / 16.4–18.8 |
| `is_sqanti_cds` | 11.55% | 8.71% | 8.5–15.3 / 6.2–14.2 |
| `frac_start` | 7.91% | 5.26% | 6.8–9.3 / 2.7–8.4 |
| `frac_stop` | 4.48% | 3.73% | 3.7–5.2 / 3.3–5.7 |

**The claim holds, and it holds better than anything in §2 or §3.** The junction count is the
largest feature in both configurations, its share agrees between them to within 3.5 points, and no
cell of the fifty puts it below 56.7%. Contrast claim 5.12, which inverts between configurations,
and claim 5.11, whose value spans 18 points. Reference-draw spreads are 0.10–0.62 pp and
ensemble-size biases are under 0.25 pp throughout.

Test-split sensitivity (n = 2,405, the published figures' population) moves nothing: 61.70% and
64.73%.

**One thing worth the manuscript's attention that the claim does not mention.** The two annotation
flags together carry **22.5% and 23.0%** — computed as a four-player game in which they move as one
player, not by adding their separate shares. That is a substantial share of the structural evidence
resting on *how an ORF was called* rather than on where it sits. It is not a defect, but a sentence
crediting the model with recovering junction-dependent decay should probably not pass over it in
silence.

*A note on that number, because the obvious shortcut is wrong.* Adding the two features' individual
shares gives 26.64% and 26.56% — an overstatement of 4.13 and 3.62 percentage points, because
`mean|a| + mean|b| >= mean|a+b|` wherever the two attributions differ in sign, and because merging
players changes the normalizer. The four-player figure is a separate exact Shapley solve over the
same coalitions. Do not add the five-player shares.

*Source: `results_interp_all/feature_decomposition_*.json` and `feature_shares_per_cell.tsv`,
produced by `analysis_plans/analysis3.py`; run log at `analysis_plans/analysis3_runlog.txt`.*
