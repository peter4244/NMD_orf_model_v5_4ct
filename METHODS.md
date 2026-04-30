# NMD ORF Model — Methods (4 Cell Type Retrain)

## NMD/Non-NMD Classification (4 Cell Types)

### Cell type selection
The model is trained using NMD classifications derived from 4 primary lung cell types: AT (alveolar type II), DD (day differentiated), FB (fibroblast), and MV (microvascular endothelial). Two cell types from the original 6-cell-type experiment were excluded:
- **DD_ALI** (air-liquid interface differentiated): Excluded due to near-zero logFC correlation between short-read and long-read DGE (r = 0.002), indicating unreliable treatment effect estimates.
- **DO** (day organoid): Excluded due to insufficient statistical power (n=2 donors after outlier removal of DD029T), weakest pairwise effect-size correlations with other cell types.

### mashr differential isoform expression
Multivariate adaptive shrinkage (mashr) was re-run using only the 4 retained cell types. Because mashr jointly estimates effect sizes and borrows strength across conditions, removing 2 cell types changes the shrinkage estimates for all remaining cell types. The new mashr results are at `/projects/talisman/shared-data/nmd/mashr/` (original 6-cell-type results archived in `old_6celltype/` subfolder).

### NMD classification criteria
- **NMD-responsive:** Union across 4 cell types of isoforms with `nmd_responsive == TRUE` (lfsr < 0.05 and posterior mean logFC > 0 in mashr).
- **Non-NMD:** Intersection across all 4 cell types of isoforms with `adj.P.Val > 0.30`.
- The non-NMD threshold was lowered from 0.50 (used in the 6-cell-type model) to 0.30 to account for changed p-value distributions under 4-cell-type mashr. With fewer cell types, the mashr shrinkage is more aggressive, shifting adj.P.Val distributions and causing the 0.50 threshold to be overly restrictive (yielding only ~1,890 non-NMD isoforms in the training set). The 0.30 threshold recovers a comparable non-NMD set size (~31,098) with a class ratio of 1:3.5 (NMD:non-NMD).
- Isoforms classified as both NMD and non-NMD (none observed) would be excluded.
- Isoforms in neither set are excluded from training.

### Dataset summary
- **Total isoforms:** 39,938 (vs 61,669 in original v5)
- **NMD:** 8,840 (vs 9,274)
- **Non-NMD:** 31,098 (vs 52,395)
- **Class ratio:** 1:3.5 NMD:non-NMD (vs 1:5.6)
- **Label changes from original:** 15 isoforms dropped (NMD → neither), 2 flipped (NMD → non-NMD), 10 flipped (non-NMD → NMD), 21,731 dropped (non-NMD → neither due to stricter intersection)

### Relabeling procedure
The relabeling is performed by `relabel_tx_summary_4ct.R`, which reads the 4 per-cell-type mashr CSVs, computes the union/intersection aggregate, and updates the `is_nmd` column in `tx_summary.tsv`. ORF features, junction positions, and other structural data are unchanged from the original v5 pipeline.

### Pipeline / build order

For the canonical sequence in which to run the SLURM wrappers (relabel → data_prep → patch_stop_codon → train → evaluate → deepshap_joint → deepshap_structural → 09b → export_joint_motif_logos → 09 → 08 → 11 → render), see **README.md "Build order"**. METHODS does not duplicate the list to avoid drift.

---

## Model Training and Window Size Sweep

### Architecture
The ORF-centric hybrid model processes up to K=5 candidate ORFs per transcript through a shared-weight `ORFEncoder` and aggregates via learned attention. Each `ORFEncoder` (`model.py:70-109`) has three sub-encoders for the rank-k ORF: an ATG CNN over the 9-channel ATG window, a stop CNN over the 9-channel stop window, and a structural linear branch `Linear(5, 32) → ReLU` over the 5 per-ORF features. The three 32-dim sub-embeddings are concatenated (96 dim) and fused via `Linear(96, 64) → ReLU → Dropout(0.2)` to produce the per-ORF embedding. The 5 ORF embeddings are aggregated by a learned attention pooler (softmax-normalized over valid ORFs), yielding a transcript embedding that is fed to a small classification head. See `model.py` (`NMDOrfModel`).

### Priority ORF Selection
ORFs are ranked by priority: (1) reference CDS ORF (if the gene's dominant non-NMD isoform's ATG can be mapped), (2) SQANTI/TransDecoder2 CDS ORF (if different from ref CDS), (3) remaining ORFs ranked by Kozak score, up to K=5. Implemented in `data_prep.py::select_priority_orfs()`. The choice of K=5 is supported by an ORF-coverage analysis: a small fraction of transcripts have more than 5 ORFs that meet the priority criteria, but the median number of priority-eligible ORFs per transcript is well below 5 (see Section 5 of the report for attention-rank dominance figures).

### Sequence source
Full-length spliced transcript sequences are read directly from the SQANTI corrected FASTA (`nmd_lungcells_corrected.fasta`). This replaces an earlier pipeline that used `cnn_data.tsv` which truncated sequences to 4,096 bp, causing 40% of STOP=1000 windows to be partially zero-padded. Verification: ATG codon confirmed at `orf_start` position for all rank-0 ORFs. Junction positions computed from `structures.rds` exon coordinates (strand-aware cumulative exon lengths).

### Sequence window extraction

Each ORF contributes two sequence windows: one centered on the **ATG (start codon)** and one centered on the **stop codon**. The window center is placed on the **middle nucleotide** of the three-nucleotide codon:

- **ATG center:** `orf_start + 1` (0-based), i.e., the T of ATG.
- **Stop center:** `orf_end - 1` (0-based), i.e., the middle nucleotide of the stop codon (A of TAA/TAG, or G of TGA).

The `window_size` parameter specifies the **total window length** in positions. Internally, `half_win = window_size // 2` positions are extracted on each side of the center (center − half_win to center + half_win − 1). A window_size of 500 therefore produces a **500-position window** spanning ±250bp around the center. Positions that fall outside the transcript boundary are zero-padded. Implemented in `data_prep.py::encode_window_v5()`.

### Window size sweep
12 models trained with ATG window ∈ {100, 500, 1000} × stop window ∈ {100, 500, 1000, 2000} positions (each spanning ±half the window size around the codon center).

### Training hyperparameters (`config.yaml`, `03_train.py`)

- **Loss:** BCEWithLogitsLoss with dynamic `pos_weight = n_neg / max(n_pos, 1)` computed from the training set (`utils.py:84-88`).
- **Optimizer:** Adam with **differential weight decay** — `weight_decay = 0.001` for CNN parameters (`atg_cnn`, `stop_cnn`), `weight_decay = 0.0001` for everything else (`03_train.py:30-45`).
- **Learning rate:** `lr = 0.001`, with **`ReduceLROnPlateau`** scheduler (`mode="max"` on val AUC, `factor=0.5`, `patience=5`).
- **Early stopping:** monitor val AUC, `patience=10` epochs, max `epochs=100`.
- **Batch size:** 256.
- **Mixed precision:** enabled.
- **Seed:** 42.
- **Overfit guard:** training halts if `train_auc - val_auc > 0.05` for `patience` consecutive epochs (`overfit_gap_threshold`).

### Best model selection

Selected configuration: **ATG=500, STOP=500** by AUC (0.9306). AUPRC peaks at ATG=500 STOP=1000 (0.8387 vs ATG=500 STOP=500 at 0.8330; difference 0.004) — both are within the AUC tier. The selection prioritizes AUC for consistency with the original v5 model and because AUC is more robust to the 4ct's class-ratio change. See README.md "Model Performance" for the full sweep table.

### Train/val/test split
- Test (holdout): chr 1, 3, 5, 7 (paralog genes excluded → "test_paralog" split)
- Validation: chr 2, 4
- Training: remaining chromosomes
- Splits assigned in `data_prep.py::build_dataset()`.

### 9-channel sequence encoding
Each window is encoded with 9 channels at each position (`data_prep.py:41,108-160`; channel names also surfaced in DeepSHAP NPZs at `deepshap.py:307-309`):

| Channel | Name | Type | Definition |
|---------|------|------|-----------|
| 0-3 | A, C, G, T | Binary (one-hot) | Nucleotide identity. Exactly one is 1 per position; positions outside the transcript are all-zero. |
| 4 | Splice junction | Binary | 1 at exon-exon junction positions within the window (from `junctions.tsv`, transcript-space coordinates). |
| 5 | Rolling GC | Continuous [0, 1] | Local GC fraction over a 50bp sliding window centered on the position (`compute_rolling_gc()`, `data_prep.py:82-102`). |
| 6-8 | Reading frame 0/1/2 | Binary (one-hot) | Codon position (0/1/2) of this position relative to this ORF's ATG, computed as `(genomic_position - orf_start) % 3`. Exactly one is 1 per position within the transcript. |

Implemented in `encode_window_v5()` (`data_prep.py:105-162`). Windows that extend beyond the sequence boundary are zero-padded across all 9 channels. The legacy "ATG codon marker", "stop-codon frame", and "ORF body" channels described in earlier docs do **not** exist in v5; codon identity is determined implicitly via the reading-frame channels and the ATG/stop windows being centered on the codons of interest.

---

## Per-ORF Structural Features

The v5 model receives **5 per-ORF structural features** and **no transcript-level features** — a substantial simplification from earlier versions, which fed parallel ref-CDS and TD2 transcript-level feature blocks. The v5 model recovers cross-ORF context from the per-ORF features alone, plus the two CNN branches and the attention aggregator. The forward signature is `model(atg_windows, stop_windows, orf_features, orf_mask)` (`model.py:200-244`); there is **no** `tx_features` argument.

### The 5 features (`data_prep.py:47-53`)

| Feature | Definition |
|---------|-----------|
| `frac_start` | Fractional start position: `orf_start / tx_length` (0 = 5' end). |
| `frac_stop` | Fractional stop position: `orf_end / tx_length` (1 = 3' end). |
| `is_ref_cds` | Binary: 1 if this ORF's start matches the reference CDS ATG (the gene's dominant non-NMD isoform's ATG, traced through the target isoform). |
| `is_sqanti_cds` | Binary: 1 if this ORF's start matches the SQANTI/TransDecoder2 CDS call. |
| `n_downstream_ejc` | Count of exon-exon junctions downstream of this ORF's stop codon. **Primary PTC indicator.** Included because junctions beyond the stop window are otherwise invisible to the CNN (`data_prep.py:46`). |

All 5 features are z-score normalized by training set statistics before entering the model. The structural sub-encoder is a single linear layer `Linear(5, 32) → ReLU` (`model.py:85,106`).

### CDS identity sourcing

`is_ref_cds` and `is_sqanti_cds` are derived from upstream feature tables that this repo **does not regenerate**:

- `ref_cds_features.tsv` — symlink to `../nmd_orf_model/results/ref_cds_features.tsv`, produced upstream by `05t_ref_cds_features.R` in that repo. Provides the reference CDS ATG position and a `category` column (used downstream for subgroup classification).
- `td2_features.tsv` — symlink to `../nmd_orf_model/results/td2_features.tsv`, produced upstream by `05t_td2_features.R`. Provides the TD2/SQANTI CDS ATG and `td2_downstream_ejc` (used for subgroup classification).

If the upstream repo moves, regenerate these files in the upstream tree and re-link before re-running the 4ct pipeline. See README.md "Cross-repo dependencies" for the canonical statement of this dependency.

### What is NOT in the model

The following are present in `selected_orfs.tsv` (and used by `04_interpret_attention.py` for downstream ORF-level analyses) but **do not enter** `NMDOrfModel`: `orf_length`, `frac_position` (different from `frac_start`), `frac_tx_covered`, `kozak_score`, `n_upstream_atgs`, `has_downstream_ejc`. The transcript-level `ref_*` and `td2_*` blocks (8 + 8 + 1 indicator) described in pre-v5 docs were removed in v5 (`model.py:199` "v5: no tx_features input"; `data_prep.py:692-693` "v5: no tx_features dataset"; `utils.py:31` "No tx_features (removed in v5)").

### Relationship to the prior structural elastic net

The prior 24-feature elastic net (AUC = 0.94) used 12 TD2 features + 12 reference-CDS features at the transcript level. The v5 ORF model replaces this with: (1) a multi-ORF approach evaluating K=5 candidate ORFs via shared-weight encoding and attention selection; (2) only 5 per-ORF structural features (no TX-level features); (3) two CNN branches over ATG and stop windows providing the sequence-level signal that the elastic net could not access.

---

## Attention Analysis (`04_interpret_attention.py`)

### Data structure

The unit of observation for most attention analyses is an **ORF-within-a-transcript**. Each test-set transcript has up to K=5 ORFs, priority-ranked (ref CDS > SQANTI/TD2 CDS > top Kozak fill — see "Priority ORF Selection" above). Each ORF has an attention weight assigned by the model's attention aggregator (softmax-normalized across ORFs within a transcript, so weights sum to 1 per transcript).

Test set: 10,131 transcripts (chr 1, 3, 5, 7; paralog-free), yielding 50,655 ORF-level rows (including padding to K=5). Of these, 2,268 are NMD-sensitive.

### Analysis 1: Attention by ORF type

For each binary ORF indicator (`is_ref_cds`, `is_sqanti_cds`, `has_downstream_ejc`), we compute the mean, median, and SD of attention weight for ORFs where the indicator is 1 vs 0, stratified by transcript NMD class. Mann-Whitney U test (two-sided) compares attention distributions for ref_CDS=1 vs ref_CDS=0 among NMD transcripts.

### Analysis 2: Attention entropy

Per-transcript Shannon entropy (base 2) of the attention weight vector, computed over valid (non-padding) ORFs only:

```
H = -sum(w_i * log2(w_i)) for w_i > 0
```

Normalized entropy divides by log2(n_valid_orfs) for that transcript, giving a value in [0, 1]. Transcripts with 1 valid ORF get normalized entropy = 0.

The "effective number of ORFs" = 2^H provides an intuitive scale: a transcript with H = 2.0 bits has attention equivalent to a uniform distribution over ~4 ORFs.

Mann-Whitney U test compares entropy distributions between NMD and non-NMD transcripts.

### Analysis 3: Attention vs ORF feature correlations

**What the rho values represent:** For each ORF-level feature (e.g., `orf_length`, `frac_position`), we compute the Spearman rank correlation between that feature's value and the attention weight, across all valid ORF-within-transcript observations.

The unit of observation is a single ORF. For example, rho = -0.64 for `frac_position` means: across all ~23,669 valid ORFs in NMD transcripts, ORFs located earlier in the transcript (lower fractional position) tend to receive higher attention weights.

This is a cross-ORF, cross-transcript correlation — it pools all ORFs from all transcripts together. It reflects which ORF-level properties the model uses to allocate attention, but does not account for within-transcript structure (e.g., it does not distinguish "this ORF got more attention than its siblings" from "this transcript's ORFs all got moderate attention").

Correlations are computed separately for NMD, non-NMD, and all transcripts. Features with fewer than 20 valid observations are excluded.

**Features:**
- `orf_length`: ORF length in nucleotides
- `frac_position`: fractional position of the ORF start codon within the transcript (0 = 5' end, 1 = 3' end)
- `frac_tx_covered`: fraction of transcript length covered by this ORF
- `kozak_score`: Kozak consensus score at the ATG
- `n_upstream_atgs`: number of ATG codons upstream of this ORF's start
- `n_downstream_ejc`: number of exon-exon junctions downstream of this ORF's stop codon
- `has_downstream_ejc`: binary indicator for n_downstream_ejc > 0

### Analysis 4: No-PTC NMD isoform attention

Identifies NMD-sensitive test transcripts where no ORF in K=5 has a downstream exon-junction complex (`has_downstream_ejc == 0` for all ORFs). These are "no-PTC" cases — NMD isoforms the model cannot explain via the classical PTC + downstream EJC mechanism.

For these transcripts, reports: model prediction accuracy, attention distribution across ORF ranks, and a feature comparison (mean ORF features of the top-attended ORF) between no-PTC and PTC NMD isoforms.

## Structural Feature Importance (`05_interpret_structural.py`)

### Method: Gradient x input attribution

For each test-set sample, we compute the gradient of the classification logit with respect to the structural feature inputs (ORF-level and TX-level), then multiply element-wise by the input values. This gives a per-feature, per-sample attribution score indicating how much each feature contributed to the model's output for that sample.

The model is set to eval mode (deterministic BatchNorm and no dropout) with gradients enabled on the structural feature tensors. `cls_logits.sum().backward()` computes per-sample gradients in a single batch pass (valid because samples are independent through the model).

### Interpretation in normalized space

All structural features are z-score normalized by the training set statistics before entering the model. The gradient x input attributions are therefore in **normalized space**: a mean |grad x input| of 0.06 for `has_downstream_ejc` means that a one-standard-deviation perturbation of this feature, at its observed value, shifts the logit by 0.06 on average. This makes features with different original scales directly comparable.

The **sign** of mean grad x input indicates direction: positive means the feature (in normalized units) pushes toward NMD classification; negative means it pushes away.

### Stratifications

- **By class:** Mean |grad x input| computed separately for NMD, non-NMD, and all test transcripts
- **By ORF rank:** ORF feature importance broken down by rank (0-4), showing whether the model uses different features for the priority ORF vs lower-ranked ones
- **By CDS status:** ORF feature importance for ref-CDS ORFs vs non-ref-CDS ORFs among NMD transcripts, revealing whether the model treats the reference frame differently

---

## DeepSHAP Sequence Interpretation (`deepshap.py`)

### Method

DeepSHAP (Lundberg & Lee, 2017) computes per-position, per-channel attribution values for the 9-channel sequence encoding. We use the `shap.DeepExplainer` implementation, which applies the DeepLIFT algorithm through the CNN layers.

### Three wrappers — three modes

`deepshap.py` provides three explainer wrappers that expose different slices of the model to DeepSHAP. Each is selected by `--branches {atg, stop, structural, joint}`:

- **`BranchWrapper`** (`deepshap.py:24-65`, `--branches {atg,stop}`) — **Marginal sequence mode.** Varies a single sequence window (ATG or stop) of the rank-0 ORF. All other inputs (rank 1-4 windows, structural features) are held constant at observed values. Used by the legacy single-branch interpretation flow (figures fall back to this when joint TSVs are unavailable).
- **`StructuralWrapper`** (`deepshap.py:69-103`, `--branches structural`) — Varies the 5 per-ORF structural features of the rank-0 ORF; sequence windows fixed. Produces the §7.2 attribution heatmap and the per-sample 5-feature SHAP table.
- **`JointBranchWrapper`** (`deepshap.py:105-166`, `--branches joint`) — **Joint mode (current canonical flow).** Varies the rank-0 ORF's ATG window + stop window + structural features simultaneously as a single flattened input vector. Fixes ranks 1-4 at observed values. Yields an additive decomposition `φ_ATG + φ_stop + φ_struct` in the embedding space that can be allocated back to position/channel level. This is the mode used to produce the SHAP data feeding §3.1 (Kozak motif logo), §4.1 (stop-codon motif logo), §7.2 (structural attribution), §9.4 (junction SHAP), §9.6 (input importance by subgroup), §9.7 (start by subgroup), §9.8 (stop by subgroup).

### Background and sample selection

The current 4ct pipeline runs DeepSHAP on **all test samples** with **500 background**, in 5 replicates with seeds 100, 200, 300, 400, 500 (`slurm_deepshap_joint.sh:19,25-26`, `slurm_deepshap_seq_500bg.sh`, `slurm_deepshap_structural.sh`):

- **Background set:** 500 randomly sampled training transcripts. The background is sampled uniformly over the training set, so it inherits ~22% NMD prevalence (8,840 / 39,938) — not stratified.
- **Explained samples:** `n_explain = 0` flag selects all 10,131 test transcripts (`deepshap.py` interprets 0 as "explain everything").
- **Replication:** 5 independent runs with seeds {100, 200, 300, 400, 500}. The 5 NPZ files (`deepshap_joint_atg500_stop500_run{1..5}.npz`) are pooled by averaging in downstream scripts.

A **legacy** marginal-mode flow exists in `slurm_deepshap_v5.sh` at `n_explain=2000`, `n_background=100` — this matches what earlier docs described, but the current report does not consume that data when joint TSVs are present. Step-4 finding E1 documents the silent fallback behavior.

### Attribution metric

The primary metric is **SHAP × input**: the DeepSHAP value at each position multiplied by the input value at that position. For one-hot encoded nucleotides, this is nonzero only for the nucleotide that is actually present. Positive values push the prediction toward NMD; negative values push away.

### Edge artifacts

CNN-based SHAP values are elevated at the first and last ~25bp of the sequence window due to the CNN's receptive field extending beyond the window boundary. This was confirmed by comparing STOP=500 and STOP=1000 half-windows (1,000 vs 2,000 bp total): the elevated region moves with the window boundary, not with genomic position. The first and last bins of regional SHAP plots should be interpreted with this caveat.

### Pooled outputs from joint NPZs

Two helper scripts pool the 5 joint NPZ files into the TSVs the report consumes (added during Step-2/Step-4 reproducibility work; absent from earlier METHODS):

- **`scripts/export_joint_motif_logos.py`** — pools the 5 NPZs into per-position-per-channel mean SHAP × input + nucleotide frequency for the ATG and stop windows. Outputs: `motif_logo_atg_joint_{tag}.tsv` (§3.1) and `motif_logo_stop_joint_{tag}.tsv` (§4.1). Population: all test samples (n_nmd ≈ 2,268, n_ctrl ≈ 7,863), 5-run averaged.
- **`09b_export_subgroup_profiles.py`** — pools the 5 NPZs into 6 TSVs feeding the subgroup analyses: `sample_shap_structural_{tag}.tsv` (per-sample 5-feature SHAP), `shap_profile_{atg,stop}_joint_{tag}.tsv` (per-position class-average), `shap_profile_{atg,stop}_subgroup_joint_{tag}.tsv` (per-position per-subgroup), `motif_logo_{atg,stop}_subgroup_joint_{tag}.tsv` (per-position per-channel per-subgroup). Subgroup classification logic at `09b_export_subgroup_profiles.py:30-43` mirrors the report's §7.1 case_when (see "Subgroup definitions" below).

### Landmark motif analysis (marginal flow, `07_motif_analysis.py`)

The legacy flow extracts per-nucleotide SHAP logos at ±15bp around biologically meaningful positions (stop codon, exon-exon junctions) from the marginal-mode NPZ. This produces a **multi-landmark** TSV (`motif_logos_stop_*.tsv`) carrying both `stop_codon` and `first_3utr_junction` landmarks. Only the `first_3utr_junction` landmark is consumed by the report (§9.1) — the stop-codon logo is sourced from the joint flow. For variable-position landmarks (junctions), each sample contributes its own landmark location, and SHAP values are aligned to the landmark position.

---

## KernelSHAP Branch Decomposition (`11_kernel_shap_branches.py`)

This produces the §2.1 branch decomposition (60.7% structural / 28.8% stop / 10.5% ATG of NMD branch attribution). Distinct from DeepSHAP, which attributes per-position; KernelSHAP here treats the model's three sub-encoders (ATG branch, stop branch, structural branch) as **3 "players"** and computes their exact additive Shapley contributions.

### Method

For each test transcript:
1. Pre-compute the 3 rank-0 ORF sub-embeddings (32-dim each) from `model.orf_encoder` (`11_kernel_shap_branches.py:33-118`).
2. Evaluate all `2^3 = 8` coalitions of present/absent branches. For absent branches, integrate over background sub-embeddings (uniform sample from the same `n_background=500` training set; `11_kernel_shap_branches.py:251`) — not a mean approximation, since the ReLU fusion layer is nonlinear.
3. Compute exact Shapley values `φ_ATG + φ_stop + φ_struct = f(x) - E[f(x)]` per transcript.

Additivity check: residual `|φ_ATG + φ_stop + φ_struct - (f(x) - E[f(x)])|` ≤ 1.8e-15 (machine epsilon) across the test set (Step-2 verified).

### Output

`kernel_shap_branch_{tag}.tsv` (per-transcript Shapley values per branch), consumed by §2.1 (overall decomposition) and §7.3 (per-subgroup decomposition).

---

## Stop-codon column patch (`scripts/patch_stop_codon.py`)

`selected_orfs.tsv` is generated by `data_prep.py` and was found in March 2026 to carry an off-by-one error in its `stop_codon` column — the column was reading the three nucleotides immediately downstream of the stop codon rather than the stop codon itself (full diagnosis in `BUGFIX_STOP_CODON_2026-03-31.md`). The 4ct addendum (2026-04-30) describes the patch flow:

1. After `data_prep.py` writes `selected_orfs.tsv`, run `python scripts/patch_stop_codon.py` (or `slurm_patch_selected_orfs.sh`).
2. The patch reads the one-hot encoded stop window from `nmd_orf_data.h5`, decodes the actual codon at the rank-0 stop position, and overwrites the `stop_codon` column with the correct value.
3. The patched file has 100% canonical stop codons (TAA/TAG/TGA) for rank-0 ORFs.

The §4.1 chi-square test depends on this patched column. **A user re-running `data_prep.py` cleanly will regenerate the buggy column** — the patch must be re-applied. README.md's "Build order" section places the patch step between `data_prep.py` and `03_train.py`.

---

## Subgroup-Specific DeepSHAP Analysis

There are **two subgroup-specific DeepSHAP flows** in the repo, with different scope and sample size:

- **Marginal subgroup flow** (`08_export_subgroup_deepshap_tsv.py`) — Stratifies the legacy run-1 marginal-mode subset (n=2,000) into subgroups and computes per-subgroup statistics on Kozak, 5'UTR composition, stop codon, junction, periodicity, and rolling GC. This is the older flow.
- **Joint subgroup flow** (`09b_export_subgroup_profiles.py`) — Pools the 5 joint-mode NPZs (all test samples, 500 background each) and emits per-subgroup positional and motif-logo TSVs. This is the **current canonical flow** for §3.1, §4.1, §7.2, §9.4, §9.6, §9.7, §9.8.

### Subgroup definitions

NMD isoforms in the test set are classified into three subgroups based on the `category` field from `ref_cds_features.tsv` and the `td2_downstream_ejc` field from `td2_features.tsv`:

- **NMD PTC+**: `category == "effectively_ptc"`, OR `category ∈ {ref_atg_lost, no_ref_isoform, not_atg_in_target, no_stop_in_target}` with `td2_downstream_ejc > 0` (TD2 reclassification).
- **NMD PTC-, ref ATG retained**: `category ∈ {no_downstream_ejc, truncated_no_ejc}`.
- **NMD PTC-, ref ATG lost**: `category ∈ {ref_atg_lost, no_ref_isoform, not_atg_in_target, no_stop_in_target}` with `td2_downstream_ejc == 0` or NA.
- Non-NMD transcripts are labeled **Control**.

This logic mirrors the report's `case_when` assignment in §7.1 of `orf_model_report_v5.Rmd` and is implemented in `09b_export_subgroup_profiles.py:30-43` (`assign_subgroup()`). NMD isoforms that fall outside all three subgroups are labeled "NMD other" and excluded from subgroup analyses (currently 0 isoforms in the 4ct test set; the four named subgroups sum to 10,131 — Step-1 verified).

### Joint subgroup flow method

Per-subgroup outputs from `09b_export_subgroup_profiles.py`:
- `sample_shap_structural_{tag}.tsv` — per-sample DeepSHAP for the 5 structural features. Used by §7.2.
- `shap_profile_{atg,stop}_joint_{tag}.tsv` — per-position per-channel mean |SHAP| for NMD vs Control. Used by §2.3, §3, §4.
- `shap_profile_{atg,stop}_subgroup_joint_{tag}.tsv` — per-position total nucleotide |SHAP| (A+C+G+T) per subgroup. Used by §9.6.
- `motif_logo_{atg,stop}_subgroup_joint_{tag}.tsv` — per-position per-channel mean SHAP × input + nucleotide frequency, ±15bp, per subgroup. Used by §9.7, §9.8.

Each output averages across the 5 joint NPZ runs; n_samples per subgroup matches the test-set sums per the case_when classification.

### Marginal subgroup flow method (legacy)

For each of the 5 DeepSHAP replicates (runs 1–5), the 2,000 explained test samples are mapped to subgroups via their `explain_indices`. SHAP × input statistics are computed per subgroup for: (1) Kozak context (-6 to +7, A of ATG = +1); (2) 5'UTR composition; (3) stop codon identity; (4) junction SHAP; (5) codon periodicity; (6) rolling GC by region.

### Cross-run stability (marginal flow)

The marginal flow aggregates per-run subgroup statistics into mean, standard deviation, CV, sign consistency (YES if all 5 runs agree on sign), and 95% CI (mean ± 1.96 × SE, SE = std / √5). Findings are reported as robust only when sign-consistent across all 5 runs. The joint flow performs the same averaging implicitly via its 5-run NPZ pooling.

---

## Gene-Matched C2/C4 Analysis

A supplementary report uses gene-matched isoform pairs from the isopair analysis to control for gene-level confounds. C2 (NMD) and C4 (Control) pairs share the same (gene_id, reference_isoform_id). Only pairs where both comparators are in the test set are used. Structural importance is recomputed from per-sample grad × input vectors (`sample_importance_{tag}.npz`) filtered to gene-matched isoforms. DeepSHAP channel summaries are recomputed from the per-sample NPZ arrays filtered to gene-matched isoform IDs.
