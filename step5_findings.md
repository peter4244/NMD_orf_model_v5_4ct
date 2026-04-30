# Step 5 — METHODS.md Verification Findings

**METHODS.md commit:** `f50addc` (post-Step-4)
**Reviewer:** Plan agent + read-only Bash/Read against current 4ct codebase
**Date:** 2026-04-30

## Summary

METHODS.md is **substantially out of date** relative to the current 4ct code. Two cross-cutting failures dominate: (1) the **9-channel sequence encoding table is wrong** — METHODS.md describes 10 channels with definitions (ATG marker, stop-codon-frame triplet, ORF body) that do **not** correspond to anything in `data_prep.py::encode_window_v5()`, which actually emits 9 channels (4 nucleotides + junction + rolling GC + 3 reading-frame one-hots); (2) METHODS.md's "Feature Definitions and CDS Sources" section describes a **17-feature transcript-level structural input (8 ref-CDS + 8 TD2 + 1 indicator) and a 9-feature per-ORF block over 10 ORFs**, neither of which exists in the v5 code — the model has no `tx_features` input at all, the per-ORF block is exactly 5 features (`frac_start`, `frac_stop`, `is_ref_cds`, `is_sqanti_cds`, `n_downstream_ejc`), and `MAX_ORFS = 5`. Beyond these, METHODS.md carries (a) a 6ct test-set count (15,584 → actually 10,131), (b) a stale background size for DeepSHAP (100 background → SLURM uses 500), (c) no description of joint-mode DeepSHAP, the joint motif-logo pooling, KernelSHAP branch decomposition, or stop-codon patch — all of which produce data the rendered Rmd consumes — and (d) no cross-repo dependency note for `ref_cds_features.tsv`/`td2_features.tsv` symlinks. The 4ct cell-type/threshold/dataset-size narrative at the top of the doc and the train/val/test split (chr 1,3,5,7 / chr 2,4 / rest) are correct. The pieces that align with code are largely the population/dataset numbers (Step 1's territory) and the "v2 Priority ORF Selection" prose. Almost everything below §3 is materially stale.

---

## Top 3 issues

### 1. CRITICAL — Wrong sequence-encoding channel table

`METHODS.md:61-72` describes a "**10-channel sequence encoding**" with channels:

| METHODS claim | Actual code (`data_prep.py:108-160`) |
|---|---|
| ch 0-3: A/C/G/T (one-hot) | ch 0-3: A/C/G/T one-hot ✓ |
| ch 4: Splice junction | ch 4: Splice junction ✓ |
| ch 5: ATG codon marker | ch 5: rolling GC fraction (continuous, ~50bp window) |
| ch 6-8: Stop codon frame 0/1/2 | ch 6-8: reading-frame one-hot (codon position 0/1/2 relative to ORF's ATG) |
| ch 9: ORF body | (no ch 9 — there are only 9 channels) |

`data_prep.py:41` `N_SEQ_CHANNELS = 9`, `model.py:38` `in_channels=9`, `config.yaml:8` `n_seq_channels: 9`, `deepshap.py:307-309` channel names = `["A","C","G","T","junction","rolling_gc","frame_0","frame_1","frame_2"]`. Step-1 already flagged the channel count as 9 vs 10 in the Rmd cross-cutting observations, but METHODS.md still has the 10-channel table with bug-for-bug wrong **definitions**, not just an extra row. Fix: replace channel rows 5-9 with the actual rolling_gc / frame_0 / frame_1 / frame_2 definitions from the code. (The `STOP_CODONS` constant in `data_prep.py:64` exists only for the verification check at L527-543; it is not encoded as a channel.)

### 2. CRITICAL — "Feature Definitions and CDS Sources" describes nonexistent transcript-level input

`METHODS.md:76-158` ("Feature Definitions and CDS Sources") describes:

- "Per-ORF (9 features) and per-transcript (8 features)" (L78)
- "Transcript-level features (17 total: 8 ref-CDS + 8 TD2 + 1 indicator)" (L90), with a long table of `ref_*`/`td2_*` columns
- "These are computed for each of the 10 priority-ranked ORFs" (L140), with a 9-row per-ORF feature table (L143-153)

**None of this matches v5 code.** `model.py:196` and `model.py:204` have a forward signature `(atg_windows, stop_windows, orf_features, orf_mask)` — there is no `tx_features` argument. `model.py:199` explicitly comments "v5: no tx_features input." `data_prep.py:692-693` explicitly skips writing `tx_features` to HDF5: `# v5: no tx_features dataset`. `utils.py:31` `NMDDataset` docstring: "No tx_features (removed in v5)."

The per-ORF block in code has **5 features**, not 9, defined at `data_prep.py:47-53`:
```python
ORF_FEATURE_COLS = ["frac_start", "frac_stop", "is_ref_cds", "is_sqanti_cds", "n_downstream_ejc"]
```
`config.yaml:11` `n_orf_features: 5`. The METHODS table at `METHODS.md:142-153` lists `orf_length`, `frac_position`, `frac_tx_covered`, `kozak_score`, `n_upstream_atgs`, `n_downstream_ejc`, `has_downstream_ejc`, `is_ref_cds`, `is_sqanti_cds` — only 3 of these (`is_ref_cds`, `is_sqanti_cds`, `n_downstream_ejc`) actually feed the model. The remaining 6 features exist in `selected_orfs.tsv` (used downstream by `04_interpret_attention.py`) but never enter `NMDOrfModel`.

Also wrong: `MAX_ORFS = 5` (`data_prep.py:39`), so "for each of the 10 priority-ranked ORFs" at L140 contradicts `K=5` claimed at L35/L38 in the same document. The Rmd uses `max_k=10` only in `04_interpret_attention.py:30,41` for join purposes, but the model architecture and HDF5 only carry 5.

Fix: drop the "Transcript-level features" section entirely (it describes the 6ct-era dual-channel ref-CDS+TD2 transcript input that v5 removed), rewrite the per-ORF table to the 5 actual features, change "10 priority-ranked ORFs" to "5 (K=5)", and update L158 ("8 + 8 = 16 transcript-level channels plus indicator") which describes a model that no longer exists.

### 3. CRITICAL — Step-2/3/4 pipeline additions undocumented in METHODS

METHODS.md describes only the original v5 marginal flow (DeepSHAP → 06/07 → motif logos). The current pipeline state, surfaced in Steps 2-4, includes four substantial additions that METHODS.md does not name anywhere:

- **Joint DeepSHAP mode** (`deepshap.py:105-166` `JointBranchWrapper` + `--branches joint`). All 5 runs in `results_4ct/deepshap_joint_*_run{1..5}.npz` use this mode at `n_explain=0` (full test set), `n_background=500` per `slurm_deepshap_joint.sh:25-26`. METHODS.md DeepSHAP section (L235-262) describes only the "BranchWrapper" (single ATG/stop branch, fixed context) at `n_explain=2,000`, `n_background=100` (L246-249). The 100-background figure matches the legacy `slurm_deepshap_v5.sh:23-24` but **not** the current `slurm_deepshap_seq_500bg.sh:24-25` / `slurm_deepshap_structural.sh:23-24` / `slurm_deepshap_joint.sh:25-26`, all of which use `n_background=500` and `n_explain=0`.
- **Joint motif-logo pooling** (`scripts/export_joint_motif_logos.py`): not mentioned in METHODS.md. This produces `motif_logo_atg_joint_atg500_stop500.tsv` and `motif_logo_stop_joint_atg500_stop500.tsv` from 5-run-pooled NPZ averaging (`scripts/export_joint_motif_logos.py:127-128`), covering all `n_nmd≈2,268 / n_ctrl≈7,863` test transcripts. Step 4 already added this to README.md but METHODS.md was not updated.
- **Subgroup profile export** (`09b_export_subgroup_profiles.py`): not mentioned. Produces `sample_shap_structural_*.tsv`, `shap_profile_{atg,stop}_{joint,subgroup_joint}_*.tsv`, `motif_logo_{atg,stop}_subgroup_joint_*.tsv` — all consumed by §3.1, §4.1, §7.2, §9.4, §9.6, §9.7, §9.8 of the report. Subgroup logic at `09b_export_subgroup_profiles.py:25-43` mirrors the report §7.1 case_when. METHODS.md §"Subgroup-Specific DeepSHAP Analysis" L267-298 describes only `08_export_subgroup_deepshap_tsv.py` (the marginal flow) and does not mention 09b at all.
- **Stop-codon patch** (`scripts/patch_stop_codon.py`): the BUGFIX_STOP_CODON_2026-03-31.md (4ct addendum) explains that `selected_orfs.tsv` carries an off-by-one stop-codon column that is patched at runtime via this script. METHODS.md does not mention the bug or the fix; the §4.1 chi-square claim in the report only renders correctly because the patch was applied on 2026-04-30 09:48. A user re-running `data_prep.py` cleanly will regenerate the buggy column; METHODS.md needs to call out the patch step.

Step 4 already added all four to README.md's "Build order" section (L76-100). METHODS.md either needs the same build-order block or an explicit cross-reference to README's pipeline-order section.

---

## All findings by category

### Category A — Architecture / model description

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| A1 | L61-72 (10-channel table) | CRITICAL | Wrong channel count and wrong definitions for ch 5-9 (see Top issue 1) | `data_prep.py:41,108-160`; `deepshap.py:307-309` |
| A2 | L78 ("per-ORF (9 features) and per-transcript (8 features)") | CRITICAL | v5 has 5 per-ORF features; no per-transcript features at all | `data_prep.py:47-53`; `config.yaml:11`; `model.py:199` |
| A3 | L90-137 (entire "Transcript-level features" section) | CRITICAL | Describes a 17-feature `ref_*`+`td2_*` TX input that the model does not receive. v5 removed this. | `model.py:196,199`; `data_prep.py:693`; `utils.py:31` |
| A4 | L140 ("for each of the 10 priority-ranked ORFs") | CRITICAL | Inconsistent with K=5 stated at L35/L38 same doc | `data_prep.py:39`; `config.yaml:3` |
| A5 | L142-153 (per-ORF feature table) | CRITICAL | Only 3 of 9 listed (`is_ref_cds`, `is_sqanti_cds`, `n_downstream_ejc`) are model inputs. The other 6 (`orf_length`, `frac_position`, `frac_tx_covered`, `kozak_score`, `n_upstream_atgs`, `has_downstream_ejc`) are downstream-script columns, not model inputs | `data_prep.py:47-53` |
| A6 | L156-158 ("Relationship to the prior structural elastic net") | MAJOR | Says ORF model retains "both ref-CDS and TD2 transcript-level features as parallel channels (8 + 8 = 16)". v5 has zero TX-level features. | Same as A3 |
| A7 | L35 ("ATG CNN + stop CNN + structural feature linear layer") | MINOR | Phrasing is correct but ambiguous: the "structural feature linear layer" is `struct_fc: Linear(5, 32)` (`model.py:85`), per-ORF only — could be misread as TX-level | `model.py:85,106` |
| A8 | L38 ("K=5 captures 83% of attention weight; ranks 5-9 contribute <17% collectively") | MINOR | Citation to Section 5 is fine, but K=5 means there are no ranks 5-9 in this model. The 83% figure must come from a separate ORF-coverage analysis; clarify provenance. | `data_prep.py:442-468` |

### Category B — Dataset / split

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| B1 | L168 ("Test set: 15,584 transcripts") | MAJOR | This is the **6ct** test-set size. 4ct test set is 10,131 transcripts (Step 1 verified); section appears in attention analysis section. The "2,386 NMD-sensitive" is also stale — actual is 2,268. | `predictions_atg500_stop500.tsv` (10,131 rows); CLAUDE.md L16 |
| B2 | L57 ("Validation: chr 2, 4") | CORRECT | Matches `data_prep.py:43 VAL_CHRS = {"chr2", "chr4"}` | `data_prep.py:43` |
| B3 | L56 ("Test (holdout): chr 1, 3, 5, 7 (paralog genes excluded → 'test_paralog' split)") | CORRECT | Matches `data_prep.py:42,556-559` and `utils.py:40-41` (test_clean = test, paralog rows held out). | `data_prep.py:42-43,556-561`; `utils.py:40` |
| B4 | L21-25 (Dataset summary block: 39,938 / 8,840 / 31,098 / 1:3.5; label-change deltas) | CORRECT | Matches CLAUDE.md, README.md, and Step 1's verified numbers | CLAUDE.md L13-19 |
| B5 | L168 ("yielding 155,840 ORF-level rows (including padding)") | MAJOR | 15,584 × 10 = 155,840. With 4ct numbers and K=5: 10,131 × 5 = 50,655 ORF-level rows. Number is wrong on two grounds (test count and K). | n/a |

### Category C — Training procedure

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| C1 | L53 (training summary) | MAJOR-incomplete | Mentions BCEWithLogitsLoss, pos_weight, Adam, early-stop val AUC patience=10, mixed precision. Missing: differential weight decay (`weight_decay_cnn=0.001` vs `weight_decay_other=0.0001`); `ReduceLROnPlateau` scheduler (`mode="max"`, `patience=5`, `factor=0.5`); batch size (256); learning rate (1e-3); seed (42); overfit gap monitoring (gap_threshold=0.05) | `03_train.py:30-45,140-148`; `config.yaml:17-29` |
| C2 | (no explicit pos_weight formula) | MINOR | Pos_weight is computed dynamically from training set as `n_neg / max(n_pos, 1)` (`utils.py:84-88`); METHODS.md says "BCEWithLogitsLoss with pos_weight" without describing the formula | `utils.py:84-88` |
| C3 | L53 ("12 models trained with ATG window ∈ {100, 500, 1000} × stop window ∈ {100, 500, 1000, 2000}") | CORRECT | Matches `data_prep.py:40` `WINDOW_SIZES = [100, 500, 1000, 2000]` and 3 ATG × 4 stop = 12 cells | `data_prep.py:40`; metrics_*.json files |
| C4 | (selection criterion) | MINOR | METHODS.md does not state which model was picked or why. README.md L33 says "Selected configuration: ATG=500, STOP=500 (best AUC)" — METHODS.md should mirror this since Step 1 settled the AUC-vs-AUPRC ambiguity | README.md:33; CLAUDE.md:8 |

### Category D — DeepSHAP description

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| D1 | L235-262 (entire DeepSHAP section) | CRITICAL | Describes only marginal `BranchWrapper` (single ATG or stop branch) at `n_explain=2000`, `n_background=100`. The active 4ct flow runs **joint** mode with `n_explain=0` (all test samples), `n_background=500`, 5 runs. Joint mode is undocumented. | `deepshap.py:105-166`; `slurm_deepshap_joint.sh:25-26` |
| D2 | L246-249 ("Background set: 100 randomly sampled training transcripts ... Explained samples: 2,000") | MAJOR | Stale 6ct numbers. Current SLURM scripts use 500 background, 0 explain (all-test-set) for both joint and structural; the only `n_explain=2000`/`n_background=100` script is the legacy `slurm_deepshap_v5.sh` | `slurm_deepshap_seq_500bg.sh:24-25`; `slurm_deepshap_structural.sh:23-24`; `slurm_deepshap_joint.sh:25-26` |
| D3 | L249 ("seeds (100, 200, 300, 400, 500)") | UNVERIFIABLE | I did not check the actual seeds in the 5 NPZ files. The seeds are passed via `${SEED}` shell var in slurm scripts. Worth confirming. | `slurm_deepshap_joint.sh` |
| D4 | L246-249 (background "reflecting ~15% NMD prevalence") | UNVERIFIABLE | The background sampling is uniform random over training set (`deepshap.py:207-208`), so it inherits ~22% NMD prevalence (8,840 / 39,938 ≈ 22%) for 4ct data, not 15% (which is the 6ct number, 9,274/61,669) | `deepshap.py:207-208`; CLAUDE.md L16 |
| D5 | L255-257 ("CNN-based SHAP values are elevated at the first and last ~25bp of the sequence window") | INFO | Edge artifact note is qualitative and likely still valid; not a defect. | n/a |
| D6 | "Marginal" vs "Joint" distinction | CRITICAL | METHODS.md does not name marginal vs joint at all. The Rmd's silent fallback (Step 4 finding E1) means the section needs to describe both modes and which one the report figures use. | n/a |

### Category E — KernelSHAP / branch decomposition

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| E1 | (no mention) | CRITICAL | `11_kernel_shap_branches.py` produces the §2.1 branch decomposition (60.7% structural / 28.8% stop / 10.5% ATG). 8-coalition embedding-level Shapley over 3 branches per transcript, default `n_background=500`. METHODS.md never mentions this analysis. | `11_kernel_shap_branches.py:131-208,251` |

### Category F — Subgroup classification

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| F1 | L267-276 (subgroup definitions) | CORRECT | Categories match `09b_export_subgroup_profiles.py:25-43` `assign_subgroup()` and Rmd §7.1 (L1597-1610, L2670-2680). "NMD other" exclusion noted. | `09b_export_subgroup_profiles.py:30-43` |
| F2 | L265 ("`08_export_subgroup_deepshap_tsv.py`") | MAJOR | Section header credits `08_…` for the analysis. Per Step 4 finding A4-A6, the active subgroup profile data feeding the report comes from `09b_export_subgroup_profiles.py` (5-run joint pooled), not `08_…`. METHODS.md needs to cover both, distinguishing the marginal (08) and joint (09b) flows. | `09b_export_subgroup_profiles.py`; Step 4 findings |
| F3 | L276 ("A small number of NMD isoforms (~15) … 'NMD other'") | UNVERIFIABLE | Need a recount on actual 4ct NMD-other count; format is "~15" so probably acceptable, but the value is from a 6ct era. Step 1 confirmed the four named subgroups sum to 10,131, so any "NMD other" is currently 0. | n/a |
| F4 | L280 ("the 2,000 explained test samples are mapped to subgroups via their `explain_indices`") | MAJOR | For the **joint/09b flow** the explained-sample count is `n_explain=0` (full test set, ~10,131), not 2,000. Re-uses the stale D2 numbers. | `slurm_deepshap_joint.sh:25` |

### Category G — Attention analysis (`04_interpret_attention.py`)

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| G1 | L162-211 (attention section) | MAJOR | Generally faithful (entropy, max-attention, argmax, MWU tests, no-PTC subset all match). The numeric mistakes (L168 test count 15,584 → 10,131; "2,386 NMD" → 2,268; "10 ORFs" → 5) cascade through. | `04_interpret_attention.py:46-272` |
| G2 | L168 ("up to 10 ORFs (top-K by Kozak score)") | MAJOR | Conflates two things: HDF5 holds K=5; `04_interpret_attention.py:30,41` joins to `selected_orfs.tsv` with `max_k=10` for downstream ORF-level analyses but `selected_orfs.tsv` itself has only ranks 0-4. So the model and the analysis both use 5 ORFs per transcript. | `data_prep.py:39`; `04_interpret_attention.py:30,41`; `selected_orfs.tsv` rank distribution |
| G3 | L166 ("top-K by Kozak score from orfik scan") | MINOR | This conflicts with L37-38 of the same doc which describes priority CDS-then-Kozak selection. L166 should say "priority-ranked (ref CDS > SQANTI CDS > Kozak fill)". | `data_prep.py:338-439` |

### Category H — Reproducibility / pipeline / cross-repo

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| H1 | (no build-order section) | MAJOR | Step 4 added a 15-step build-order block to README.md (L76-100). METHODS.md has nothing analogous; describing methodology without naming `09b`, `export_joint_motif_logos`, `patch_stop_codon`, or `11_kernel_shap_branches` makes the methodology non-reproducible from this doc alone. Either replicate the README block or add a "See README.md for pipeline order" pointer. | README.md:76-100 |
| H2 | L109, L126 ("Source: `ref_cds_features.tsv` (computed by `05t_ref_cds_features.R`)") | MAJOR | These references are unfindable in this repo. `ref_cds_features.tsv` is a symlink → `../nmd_orf_model/results/ref_cds_features.tsv`. METHODS.md does not say so. Step 4 finding C5 already flagged this for the Rmd; same issue here. | `ls -la results_4ct/ref_cds_features.tsv` |
| H3 | L41 ("Junction positions computed from `structures.rds` exon coordinates") | INFO | Not a problem in itself, but `structures.rds` lives upstream and the actual code reads `junctions.tsv` (`data_prep.py:277-284`). METHODS would benefit from naming the TSV the model actually consumes. | `data_prep.py:277-284` |
| H4 | (BUGFIX_STOP_CODON_2026-03-31.md) | MAJOR | The 2026-04-30 4ct addendum (lines 143-176 of that file) describes the patch flow and the `scripts/patch_stop_codon.py` bugfix needed for §4.1's chi-square. METHODS.md silent on this. | `BUGFIX_STOP_CODON_2026-03-31.md:143-176` |

### Category I — Window construction / encoding details

| # | METHODS.md line | Severity | Issue | Code reference |
|---|---|---|---|---|
| I1 | L43-50 (window construction) | CORRECT | Matches `data_prep.py:105-162,591-593`. ATG center at `orf_start - 1 + 1` (T of ATG), stop center at `orf_end - 1 - 1` (middle of stop codon), `half_win = window_size // 2`, span `[center - half_win, center + half_win)`. | `data_prep.py:105-162,591-593` |
| I2 | L41 ("Verification: ATG codon confirmed at `orf_start` position for all rank-0 ORFs") | CORRECT | `data_prep.py:524-543` does this verification check at build time. | `data_prep.py:524-543` |
| I3 | L40-42 (sequence source paragraph) | CORRECT | Matches `FASTA_PATH = .../nmd_lungcells_corrected.fasta` (`data_prep.py:36-37`); the "earlier pipeline used `cnn_data.tsv` truncated to 4,096 bp" historical note is fine. | `data_prep.py:36-37` |
| I4 | L37-38 (Priority ORF Selection) | CORRECT | Matches `data_prep.py:338-439` `select_priority_orfs()`. | `data_prep.py:338-439` |

### Category J — Cell types / labels (4ct narrative)

All Section 1 items at L3-28 (cell types, mashr re-run, NMD/non-NMD criteria, threshold change 0.50→0.30, dataset summary, relabeling procedure via `relabel_tx_summary_4ct.R`) are correct and consistent with CLAUDE.md, README.md, and the underlying code. **CORRECT.**

---

## Verified-correct inventory (high-confidence)

1. 4ct cell-type list (AT, DD, FB, MV) and DD_ALI/DO exclusion rationale (L6-8)
2. Non-NMD threshold 0.30 and rationale (L15-16)
3. Dataset sizes (39,938 / 8,840 / 31,098 / 1:3.5) (L21-25)
4. Train/val/test chromosome split (chr 1,3,5,7 / chr 2,4 / rest) (L56-58) — matches `data_prep.py:42-43`
5. Window construction logic (ATG center = T of ATG; stop center = middle of stop codon; ±half_win zero-padded) (L43-50)
6. Window-sweep grid (3×4 = 12 cells) (L53)
7. Priority ORF selection logic (L37-38) — matches `data_prep.py::select_priority_orfs()`
8. ATG verification at build time (L41) — matches `data_prep.py:524-543`
9. Subgroup definitions (L267-276) — match `09b_export_subgroup_profiles.py:25-43` and Rmd §7.1
10. Edge-artifact qualitative description (L255-257) — sound conceptually

---

## Adjudication needed (before fixes)

1. **Channel table (Issue #1):** Replace L61-72 with the actual 9-channel table from `data_prep.py:108-160` plus `deepshap.py:307-309` channel names. Drop the row for "ch 9: ORF body" entirely; replace ch 5/6-8 with rolling_gc and reading-frame. Apply mechanically?

2. **Per-ORF feature table (Issue #2 / Cat A2-A5):** Replace L142-153 with the 5-row table from `data_prep.py:47-53`. Drop or heavily restructure L76-137 ("Feature Definitions and CDS Sources" / "Transcript-level features"), since v5 has no TX-level features. Recommend: a single "Per-ORF structural features (5)" subsection naming `frac_start`, `frac_stop`, `is_ref_cds`, `is_sqanti_cds`, `n_downstream_ejc` and noting the rationale comment from `data_prep.py:46`. Keep a note that `ref_cds_features.tsv`/`td2_features.tsv` exist for **subgroup classification** (§7.1) and **09b** export, but they do not enter the model. Apply or revise scope?

3. **K=5 vs K=10 (Cat A4, A8, G2):** Change L140 ("each of the 10 priority-ranked ORFs") to "each of the K=5 priority-ranked ORFs". Reword L166 to remove "up to 10 ORFs". The "K=5 captures 83%" claim at L38 is presumably from an earlier ORF-coverage analysis on K=10 selections — this needs to be re-verified or reframed as "median ≤5 ORFs/transcript".

4. **DeepSHAP rewrite (Cat D1-D6, F4):** Add a "Joint vs marginal" subsection. Joint mode (n_explain=0, n_background=500, 5 runs, joint flattened input rank-0 only, fixed context for ranks 1-4) feeds §3.1, §4.1, §7.2, §9.4, §9.6, §9.7, §9.8 via `09b_export_subgroup_profiles.py` and `scripts/export_joint_motif_logos.py`. Marginal mode (the `BranchWrapper` ATG/stop) is the legacy single-branch attribution. Keep both, label which figures use which.

5. **Add KernelSHAP section (Cat E1):** New subsection for `11_kernel_shap_branches.py`: 8-coalition embedding-level Shapley over 3 branches, default `n_background=500`, additivity check residual ≤ 1.8e-15 (Step 2), produces §2.1 branch decomposition.

6. **Stop-codon patch (Cat H4):** Add a brief "Stop-codon column patch" note pointing at `scripts/patch_stop_codon.py` and `BUGFIX_STOP_CODON_2026-03-31.md`'s 4ct addendum.

7. **Build order (Cat H1):** Replicate README.md's 15-step block, or add a one-line "See README.md §Build order" pointer? Recommend the latter to avoid drift.

8. **Cross-repo dependencies (Cat H2):** Add a note that `ref_cds_features.tsv` and `td2_features.tsv` are produced upstream by `../nmd_orf_model/05t_ref_cds_features.R` and `05t_td2_features.R` and symlinked into `results_4ct/`. README.md has this at L102-112; mirror in METHODS.md or cross-reference.

9. **Test set size (Cat B1, B5, G2):** Change L168 from "15,584 transcripts ... 155,840 ORF-level rows (including padding) ... 2,386 are NMD-sensitive" to "10,131 transcripts ... 50,655 ORF-level rows ... 2,268 NMD-sensitive". Same correction wherever 6ct numbers leak.

10. **Training detail completeness (Cat C1):** Decide whether METHODS.md should describe full hyperparameters (differential weight decay, ReduceLROnPlateau, batch size, lr, seed, gap monitoring) or stay summary-level and defer to `config.yaml`.
