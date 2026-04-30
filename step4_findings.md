# Step 4 — Reproducibility & Completeness Findings

**Report:** `orf_model_report_v5.Rmd` at commit `8c28bfa` (post-Step-3)
**Reviewer:** Plan agent + direct grep/read verification (read-only)
**Date:** 2026-04-30

## Summary

The pipeline trail is in good shape for the **classical** outputs (training, evaluation, marginal DeepSHAP, motif analysis, junction/GC/polyA, kernel SHAP, attention) — every figure has a source TSV, every TSV exists on disk for the active best model, and chunk guards protect against silent rendering of empty sections via the data-status panel. The principal Step-4 issues fall into four buckets: (1) **stale provenance comments** that still claim "produced by 07_motif_analysis.py" / "joint DeepSHAP run 1" / "produced by deepshap.py" for files that are actually 5-run pooled outputs of `09b_export_subgroup_profiles.py` or `scripts/export_joint_motif_logos.py`; (2) the three **Step-2-era reproducibility helpers** (`09b_export_subgroup_profiles.py`, `scripts/export_joint_motif_logos.py`, `scripts/patch_stop_codon.py`) are uncommitted-to-SLURM and undocumented in METHODS/README/CLAUDE — yet they produce data feeding §3.1, §4.1, §7.2, §9.4, §9.6, §9.7, §9.8; (3) **dead loads** (`pa_data` from a missing `polya_signal_*.tsv`) and a latent **variable-shadowing trap** in §9.11 (`all_orfs` is overwritten if `deepshap_all_orfs_summary_*.tsv` ever appears); (4) **orphan figure files** (`fig4a_stop_codon_nmd_ctrl.{png,pdf}`, `fig4f_junction_nmd_ctrl.{png,pdf}`) leftover from before Step 2's NMD-vs-Control → per-subgroup restructure. No silent fallback is currently misleading the rendered HTML, but several would activate quietly if someone regenerated data in a different order.

## Top 3 issues

### 1. CRITICAL — Stale provenance comments for joint-pooled SHAP TSVs

Multiple core figure chunks still document inputs as marginal "run 1" outputs of `07_motif_analysis.py` / `deepshap.py`, while the `has_motif_*_joint`/`has_sg_profiles` guards mean the chunk in fact reads 5-run-pooled TSVs produced by `scripts/export_joint_motif_logos.py` and `09b_export_subgroup_profiles.py`. Affected lines:

- `orf_model_report_v5.Rmd:834` — "Source: motif_logo_atg_{shap_run_tag}.tsv — produced by 07_motif_analysis.py from deepshap_atg_{tag}.npz". Active source under current state is `motif_logo_atg_joint_atg500_stop500.tsv` produced by `scripts/export_joint_motif_logos.py` from `deepshap_joint_*_run{1..5}.npz` (5-run pooled).
- `orf_model_report_v5.Rmd:999-1001` — Same wording for stop logo. "Population: ~2,000 DeepSHAP samples per replicate (run 1 shown)" — actually the joint-pooled file uses **all** test samples (n_nmd ≈ 2,268 per Step 2) averaged across 5 runs.
- `orf_model_report_v5.Rmd:733` — `# Population: all test transcripts (NMD only), joint DeepSHAP run 1` — the file `shap_profile_atg_joint_atg500_stop500.tsv` is produced by `09b_export_subgroup_profiles.py` and is averaged across 5 runs (verified at `09b_export_subgroup_profiles.py:72-76`).
- `orf_model_report_v5.Rmd:1001` — see above.
- `orf_model_report_v5.Rmd:1654` — `# Produced by deepshap.py --branches structural --n-explain 0 --n-background 500. Averaged across 5 replicates`. Actual producer of `sample_shap_structural_*.tsv` is `09b_export_subgroup_profiles.py:83,198-200` — the 5-run averaging is inherited from `load_runs()`, but the named producer is wrong.
- `orf_model_report_v5.Rmd:2428` — "Population: all test transcripts from joint DeepSHAP run 1" for `shap_profile_atg_subgroup_joint_*.tsv`. Same misstatement: 5-run pooled.
- `orf_model_report_v5.Rmd:2743` — same.
- `orf_model_report_v5.Rmd:2867` — "Population: all test transcripts, 500 background, joint DeepSHAP run 1" (§9.11). The producer comment at L2866 ("Produced by aggregating deepshap_joint_{tag}[_orf{1-4}]_run1.npz files") is also misleading — the file (when present) would also be 5-run pooled if produced by the same pipeline.

This is the same class of bug Step 2 caught in §3.1/§4.1 (joint vs run1-marginal); the "joint" data is now in place but the in-chunk provenance comments still describe the run-1 marginal sources they replaced.

### 2. CRITICAL — Reproducibility helper scripts not in any SLURM wrapper or methods doc

Three scripts produce data that the rendered Rmd consumes, but none are referenced by any `slurm_*.sh` wrapper or by `METHODS.md`/`README.md`/`CLAUDE.md`:

- `09b_export_subgroup_profiles.py` — produces 6 TSVs (`sample_shap_structural_*.tsv`, `shap_profile_{atg,stop}_{joint,subgroup_joint}_*.tsv`, `motif_logo_{atg,stop}_subgroup_joint_*.tsv`) feeding §3.1, §4.1, §7.2, §9.4, §9.6, §9.7, §9.8.
- `scripts/export_joint_motif_logos.py` — produces the 2 master joint motif logos for §3.1, §4.1.
- `scripts/patch_stop_codon.py` — patches `selected_orfs.tsv` from HDF5 (Step 2 fix #11). Required to reproduce §4.1 chi-square (without this patch the column carries the documented BUGFIX_STOP_CODON-2026-03-31 off-by-one).

A user starting from a clean checkout has no documented path to regenerate these intermediate TSVs. `slurm_export_motif_v5.sh:16-20` only runs 06 and 07 (marginal flow); `slurm_export_features_09.sh` only runs 09_export_{gc_content,polya,junction_ordinal}. There is no `slurm_export_features_09b.sh`. METHODS.md does not name 09b or the two `scripts/*.py` helpers anywhere.

### 3. MAJOR — `all_orfs` variable-shadowing trap in §9.11

`orf_model_report_v5.Rmd:160` defines a global `all_orfs <- read.delim("results_4ct/selected_orfs.tsv")` consumed at L1094 (§4.1 stop-codon test) and L2633 (§9.9 attention subgroup join). At `orf_model_report_v5.Rmd:2878`, inside the `eval=has_all_orfs` chunk, the same name is reassigned: `all_orfs <- read.delim(all_orfs_path)` — i.e. the §9.11 per-ORF DeepSHAP table. Currently dormant because `has_all_orfs == FALSE` (the file `deepshap_all_orfs_summary_atg500_stop500.tsv` is intentionally not generated; §9.11 is gated behind a "Section intentionally not generated" note at L2873-2874, which is fine).

If a future user runs `slurm_deepshap_joint_orf1_4.sh` to populate §9.11, the chunk at L2878 will silently clobber the global `all_orfs` data frame. Subsequent chunks that re-reference the original `all_orfs` (none after L2633 in current Rmd) would be safe — but anyone editing the report later may step on this trap.

## All findings by category

### Category A — Stale or wrong provenance comments

| # | File:line | Severity | Issue |
|---|-----------|----------|-------|
| A1 | Rmd:834 | CRITICAL | "produced by 07_motif_analysis.py" — actually `scripts/export_joint_motif_logos.py` when joint TSV present (current state) |
| A2 | Rmd:999-1001 | CRITICAL | Same as A1 for stop logo; "(run 1 shown)" wrong — joint-pooled across 5 runs |
| A3 | Rmd:733 | MAJOR | "joint DeepSHAP run 1" — actually 5-run pooled by `09b_export_subgroup_profiles.py` |
| A4 | Rmd:1654 | MAJOR | `sample_shap_structural_*.tsv` named producer "deepshap.py" — actual producer is `09b_export_subgroup_profiles.py` |
| A5 | Rmd:2428 | MAJOR | "joint DeepSHAP run 1" for `shap_profile_atg_subgroup_joint_*.tsv` — 5-run pooled |
| A6 | Rmd:2743 | MAJOR | Same misstatement as A5 |
| A7 | Rmd:2867 | MAJOR | "joint DeepSHAP run 1" for §9.11 deepshap_all_orfs_summary; if/when generated, will be 5-run pooled |
| A8 | Rmd:163 | MINOR | "from 05t_ref_cds_features.R" — script does not exist in this repo (lives in upstream `nmd_orf_model`); `ref_cds_features.tsv` is a symlink to `/home/p.castaldi/cc/nmd_orf_model/results/ref_cds_features.tsv` |
| A9 | Rmd:1546, 1630 | MINOR | Same `05t_ref_cds_features.R` reference; not reproducible from this repo alone |
| A10 | Rmd:2624 | MINOR | "subgroup assignments from Section 9.13 logic" — actual definition lives in §7.1 (chunk `sec6-subgroup-definitions` at L1540), not §9.13 |
| A11 | Rmd:1303 | MINOR | "10 isoforms with all-zero weights … excluded" — Step 1 noted actual is 0 isoforms; comment is stale (deferred there, surface here) |

### Category B — Files referenced but missing / dead loads

| # | File:line | Severity | Issue |
|---|-----------|----------|-------|
| B1 | Rmd:239-241 | MINOR | `polya_signal_atg500_stop500_run1.tsv` is loaded into `pa_data` but the file does not exist on disk and `pa_data` is never used elsewhere in the Rmd. Dead read; renders successfully because of `if (has_pa)` guard. Safe to remove the load + path definitions |
| B2 | Rmd:2869-2870 | INFO | `deepshap_all_orfs_summary_atg500_stop500.tsv` missing — handled gracefully via `eval=!has_all_orfs` skip-note at L2873; data-status panel surfaces it. No fix required (intentional) |

### Category C — Reproducibility gaps

| # | Item | Severity | Issue |
|---|------|----------|-------|
| C1 | `09b_export_subgroup_profiles.py` | CRITICAL | Not in any `slurm_*.sh`; not in `METHODS.md`/`README.md`/`CLAUDE.md`; produces 6 TSVs feeding §3.1, §4.1, §7.2, §9.4, §9.6, §9.7, §9.8 |
| C2 | `scripts/export_joint_motif_logos.py` | CRITICAL | Same — not in any wrapper or doc; produces the 2 master joint motif logos for §3.1 / §4.1 |
| C3 | `scripts/patch_stop_codon.py` | MAJOR | Not in any wrapper. BUGFIX_STOP_CODON_2026-03-31.md (4ct addendum) describes purpose but not invocation. §4.1 chi-square breaks without this patch (Step 2 fix #11) |
| C4 | `selected_orfs.tsv` | MAJOR | Not a symlink — it is a real file overwritten by `patch_stop_codon.py` on 2026-04-30 09:48. If someone re-runs `data_prep.py` it will be regenerated with the buggy stop_codon column. Need a runbook step or build-time guard |
| C5 | `ref_cds_features.tsv`, `td2_features.tsv` | MAJOR | Both are symlinks pointing into `/home/p.castaldi/cc/nmd_orf_model/results/`. If the upstream repo moves, the 4ct repo silently breaks. Cross-repo dependency not documented |
| C6 | `audit_report.R` | MINOR | Stale: `results_dir <- "results"` (L13) instead of `results_4ct`. Won't run in this repo as-is. Either update or remove |
| C7 | `tx_summary_6ct.tsv` in `results_4ct/` | MINOR | Bootstrapping input for `relabel_tx_summary_4ct.R`. Used once; no longer needed but kept for re-bootstrapping. Document or delete |

### Category D — Orphaned outputs

| # | Path | Severity | Notes |
|---|------|----------|-------|
| D1 | `figures/fig4a_stop_codon_nmd_ctrl.{png,pdf}` | MINOR | Last-modified 2026-04-30 08:54. No matching `save_fig(..., "fig4a_stop_codon_nmd_ctrl")` in current Rmd. Leftover from pre-Step-2 NMD-vs-Control restructure (Step 2 fix #5 changed §4.1.1 to per-subgroup → fig name became `fig4a_stop_codon_subgroup`) |
| D2 | `figures/fig4f_junction_nmd_ctrl.{png,pdf}` | MINOR | Same — Step 2 fix #6 renamed §9.4 figure to `fig4f_junction_subgroup` |
| D3 | `results_4ct/motif_kmer_enrichment_atg500_stop500_run1.tsv` | MINOR | Produced by 07_motif_analysis.py L411; not consumed by Rmd or any script |
| D4 | `results_4ct/motif_positional_kmer_atg500_stop500_run1.tsv` | MINOR | Same — produced by 07 L494, not consumed |
| D5 | `results_4ct/sample_importance_tx_atg500_stop500.tsv` | MINOR | Produced by 05b_export_sample_importance_tsv.py; not consumed by Rmd or any script |
| D6 | `results_4ct/sample_importance_atg500_stop500.npz` | MINOR | Produced by 05_export_sample_importance.py; only consumed by 05b → also unused output chain |

### Category E — Silent fallbacks worth flagging

| # | File:line | Severity | Issue |
|---|-----------|----------|-------|
| E1 | Rmd:218-230 (load-best-model) | MAJOR | `if (has_motif_atg_joint && has_motif_stop_joint) { … } else if (has_motif_marginal) { … }` swaps motif logo source between joint (5-run pooled, full test set) and marginal (run-1, n=444 NMD subsample). The only signal the reader gets is a `cat()` line in the setup chunk's stdout, which is `include=FALSE`-suppressed in the rendered HTML. The data-status panel at L317 conflates both via `has_motif_logos`. Reader cannot tell from the rendered HTML which dataset is shown. Suggest splitting the data-status flag and appending "(joint, 5-run)" / "(marginal, run 1)" to figure subtitles |
| E2 | Rmd:609-643 (prepare-unified-importance) | MAJOR | `has_structural_shap` controls a fallback from `shap_summary` (mean of 5 deepshap_summary runs) to `imp_rank` (single file `structural_importance_by_rank_*.tsv`). Currently dormant (true), but if any of the 5 `deepshap_summary_*_run*.tsv` is missing, the §2.2 "Structural Feature Importance" figure silently switches to a different methodology with no visual indication |
| E3 | Rmd:160 vs 2878 | MAJOR | `all_orfs` variable shadowed in §9.11 (Issue #3 above). Currently dormant |
| E4 | Rmd:1007-1012 | MINOR | `if ("landmark" %in% names(motif_stop)) { ... filter(landmark == "stop_codon") } else { ... }` — handles the marginal vs joint TSV schema difference inline. Correct, but undocumented in the surrounding provenance comment |

### Category F — Orphan/legacy scripts

| # | Script | Severity | Issue |
|---|--------|----------|-------|
| F1 | `audit_report.R` | MINOR | Stale results_dir (Cat C6). Either update for 4ct or delete |
| F2 | `relabel_tx_summary_4ct.R` | INFO | Bootstrapping script (one-time tx_summary relabel from 6ct → 4ct). Not in any slurm wrapper. Document as one-time setup step |
| F3 | `make_architecture_figure.R`, `make_shap_interpretation_figure.R` | INFO | Not in slurm but invoked via `source()` from Rmd L345/L351 (eval=file.exists). Working as intended |

## Verified-correct inventory

- Every TSV/RDS/CSV/JSON loaded by the Rmd at the active best model (atg500_stop500) **exists on disk** except `polya_signal_atg500_stop500_run1.tsv` (B1, dead load) and `deepshap_all_orfs_summary_atg500_stop500.tsv` (B2, intentionally absent and gracefully handled).
- All 12 `metrics_atg{A}_stop{S}.json` cells are present (sweep heatmaps complete).
- All 12 `predictions_*.tsv`, `attention_weights_*.tsv`, `training_log_*.csv` files for the 12 sweep cells are present.
- All 5 `deepshap_joint_atg500_stop500_run{1..5}.npz` and 5 `deepshap_summary_atg500_stop500_run{1..5}.tsv` files are present (foundation for joint-pooled outputs).
- 40 distinct `save_fig(..., "name")` calls; 40 corresponding {png,pdf} pairs present (zero missing). Two stale figure pairs in `figures/` correspond to renamed save_fig calls (D1, D2).
- Data-status panel at L311-339 surfaces every guard flag for the section consumers; reader sees a "Data status" header with missing inputs called out (currently, two gentle "missing" notices: §9.11 deepshap_all_orfs and possibly polya_signal — but the latter isn't in the panel because `has_pa` flag isn't included, only `has_pa_sqanti`).
- Step-2-introduced `scripts/patch_stop_codon.py` is correctly cited at Rmd:1086-1089 for the §4.1 stop-codon analysis (no provenance gap there).
- Subgroup definition logic is consistent between §7.1 (L1569-1585) and §9.9 (L2637-2649) — same case_when, same category mapping.

## Adjudication needed

1. **Category A (provenance comments).** All 7 stale-source comments are in-chunk `# Source:` notes. Recommend rewriting each to: (a) name the actual producing script (`09b_export_subgroup_profiles.py` or `scripts/export_joint_motif_logos.py`); (b) replace "joint DeepSHAP run 1" with "joint DeepSHAP, 5-run pooled, full test set"; (c) for L1654 specifically, name 09b not deepshap.py. Mechanical edits, no data change. Apply or skip?

2. **Category C1-C3 (helper scripts not in slurm).** Add three new files: `slurm_export_features_09b.sh` invoking `python 09b_export_subgroup_profiles.py --tag atg500_stop500`, `slurm_export_joint_motif_logos.sh` invoking `python scripts/export_joint_motif_logos.py --tag atg500_stop500`, and a one-line addition to either `slurm_build_h5.sh` or a new `slurm_patch_selected_orfs.sh` invoking `python scripts/patch_stop_codon.py`. Plus a "Build order" section in METHODS.md or README.md ordering them. Apply, defer, or document as known limitation?

3. **Category C4-C5 (selected_orfs / ref_cds_features dependencies).** `selected_orfs.tsv` is a real file that gets clobbered by `data_prep.py`. Add a guard or a build-time chained step `data_prep.py && patch_stop_codon.py`? Document the ref/td2 symlinks as cross-repo deps in README?

4. **Category D1-D2 (orphan figure files).** Safe to delete `figures/fig4a_stop_codon_nmd_ctrl.{png,pdf}` and `figures/fig4f_junction_nmd_ctrl.{png,pdf}`?

5. **Category D3-D6 (orphan output TSVs/NPZ).** `motif_kmer_enrichment_*`, `motif_positional_kmer_*`, `sample_importance_tx_*`, `sample_importance_*.npz` are produced but never consumed. Disable the producing branches in 07/05/05b, or accept as legacy for compatibility with the 6ct lineage?

6. **Category E1 (motif logo silent fallback).** Add an explicit visible note to the data-status panel — split `has_motif_logos` into `has_motif_logos_joint` and `has_motif_logos_marginal_fallback`, and surface the fallback condition in the rendered HTML?

7. **Issue #3 / Cat E3 (`all_orfs` shadowing).** Rename §9.11's `all_orfs <- read.delim(all_orfs_path)` to `all_orfs_per_orf` (or similar) at Rmd:2878 to remove the latent trap?

8. **Cat F1 (`audit_report.R`).** Update results_dir to "results_4ct" or delete the script (REPORT_AUDIT.md was a 6ct-era artifact)?
