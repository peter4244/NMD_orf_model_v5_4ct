# NMD ORF Model — Methods

## Model Training and Window Size Sweep

### Architecture
The ORF-centric hybrid model processes up to K=5 candidate ORFs per transcript through a shared-weight encoder (ATG CNN + stop CNN + structural feature linear layer), aggregates via learned attention, and predicts NMD status. See `model.py` for full architecture (NMDOrfModel class).

### Priority ORF Selection
ORFs are ranked by priority: (1) reference CDS ORF (if the gene's dominant non-NMD isoform's ATG can be mapped), (2) SQANTI/TransDecoder2 CDS ORF (if different from ref CDS), (3) remaining ORFs ranked by Kozak score, up to K=5. Implemented in `data_prep.py::select_priority_orfs()`. K=5 captures 83% of attention weight; ranks 5-9 contribute <17% collectively (see Section 5 of the report).

### Sequence source
Full-length spliced transcript sequences are read directly from the SQANTI corrected FASTA (`nmd_lungcells_corrected.fasta`). This replaces an earlier pipeline that used `cnn_data.tsv` which truncated sequences to 4,096 bp, causing 40% of STOP=1000 windows to be partially zero-padded. Verification: ATG codon confirmed at `orf_start` position for all rank-0 ORFs. Junction positions computed from `structures.rds` exon coordinates (strand-aware cumulative exon lengths).

### Sequence window extraction

Each ORF contributes two sequence windows: one centered on the **ATG (start codon)** and one centered on the **stop codon**. The window center is placed on the **middle nucleotide** of the three-nucleotide codon:

- **ATG center:** `orf_start + 1` (0-based), i.e., the T of ATG.
- **Stop center:** `orf_end - 1` (0-based), i.e., the middle nucleotide of the stop codon (A of TAA/TAG, or G of TGA).

The `window_size` parameter specifies the **total window length** in positions. Internally, `half_win = window_size // 2` positions are extracted on each side of the center (center − half_win to center + half_win − 1). A window_size of 500 therefore produces a **500-position window** spanning ±250bp around the center. Positions that fall outside the transcript boundary are zero-padded. Implemented in `data_prep.py::encode_window_v5()`.

### Window size sweep
12 models trained with ATG window ∈ {100, 500, 1000} × stop window ∈ {100, 500, 1000, 2000} positions (each spanning ±half the window size around the codon center). Training uses BCEWithLogitsLoss with pos_weight for class imbalance, Adam optimizer, early stopping on validation AUC (patience=10), and mixed precision.

### Train/val/test split
- Test (holdout): chr 1, 3, 5, 7 (paralog genes excluded → "test_paralog" split)
- Validation: chr 2, 4
- Training: remaining chromosomes
- Splits assigned in `data_prep.py::build_dataset()`.

### 10-channel sequence encoding
Each window is encoded with 10 binary (0/1) channels at each position:

| Channel | Name | Definition |
|---------|------|-----------|
| 0-3 | A, C, G, T | One-hot nucleotide encoding. Exactly one is 1 per position. |
| 4 | Splice junction | 1 at exon-exon boundary positions (transcript-space junction coordinates from `structures.rds`). |
| 5 | ATG codon marker | 1 at all three positions of any ATG triplet within the window. |
| 6-8 | Stop codon frame 0/1/2 | 1 at positions within stop codons (TAA/TAG/TGA), assigned to the channel matching the reading frame relative to the ORF's ATG. |
| 9 | ORF body | 1 at positions inside the ORF (between start and stop codons), 0 outside. |

Implemented in `data_prep.py::encode_window_vectorized()`. Windows that extend beyond the sequence boundary are zero-padded.

---

## Feature Definitions and CDS Sources

The model uses two levels of structural features: per-ORF (9 features) and per-transcript (8 features). Understanding their provenance is critical for interpreting the results.

### CDS identity indicators

Two binary per-ORF features identify whether an ORF corresponds to a known CDS call:

- **`is_ref_cds`**: 1 if this ORF's start position matches the **reference CDS ATG**. The reference CDS is defined as the reading frame of the gene's dominant non-NMD isoform (highest DMSO CPM across cell types). This is a gene-level anchor: for each gene, we identify the most abundant non-NMD isoform and trace its ATG through the target isoform. Available for 45,008 / 61,697 isoforms (73%). Unavailable when no non-NMD isoform in the gene shares the ATG with a top-10 ORF, or when the gene has no clearly dominant non-NMD isoform.

- **`is_sqanti_cds`**: 1 if this ORF's start position matches the **SQANTI/TransDecoder2 CDS call**. SQANTI assigns CDS via TransDecoder2, which selects the longest ORF with homology support. For novel transcripts not in reference annotation, this is the only CDS call available. Available for all 61,697 coding isoforms.

For annotated (ENST) transcripts, both indicators may flag the same ORF if TD2 and the reference agree on the reading frame. For novel transcripts, `is_ref_cds` may be available if the novel isoform belongs to a gene with a dominant non-NMD isoform whose ATG can be traced, but `is_sqanti_cds` is always available.

### Transcript-level features (17 total: 8 ref-CDS + 8 TD2 + 1 indicator)

The model receives two parallel sets of transcript-level structural features — one derived from the reference CDS frame and one from the TransDecoder2 (TD2) CDS prediction — plus an indicator for reference CDS availability. This dual-channel design ensures the model has structural context for all isoforms, including the ~25% where the reference CDS ATG cannot be mapped.

#### Ref-CDS features (`ref_` prefix, 8 features)

Derived from the **reference CDS frame** — the reading frame of the gene's dominant non-NMD isoform projected onto the target isoform. Available for ~75% of isoforms (`ref_atg_available == 1`). For the remaining ~25%, all ref-CDS features are 0 (filled from NaN).

| Feature | Definition |
|---------|-----------|
| `ref_downstream_ejc` | Number of exon-exon junctions downstream of the reference CDS stop codon. Primary PTC indicator. |
| `ref_log_utr3_length` | Log of the 3'UTR length as defined by the reference CDS stop codon |
| `ref_atg_density` | Density of ATG codons in the 5'UTR (upstream of the reference CDS start) |
| `ref_atg_strong_kozak` | Whether the reference CDS ATG has a strong Kozak consensus |
| `ref_uorf_count_overlapping` | Number of upstream ORFs overlapping the reference CDS |
| `ref_uorf_count_outframe` | Number of out-of-frame upstream ORFs in the 5'UTR |
| `ref_utr5_orf_coverage` | Fraction of the 5'UTR covered by upstream ORFs |
| `ref_stop_density` | Density of stop codons in the vicinity of the reference CDS stop |

Source: `ref_cds_features.tsv` (computed by `05t_ref_cds_features.R`).

#### TD2 features (`td2_` prefix, 8 features)

Derived from the **TransDecoder2 CDS prediction** — SQANTI/TD2's CDS call for each isoform. Available for ~99% of coding isoforms (excludes ~750 with zero-length 5'UTR or failed ATG validation). Unlike ref-CDS features, TD2 features do not require cross-isoform reference tracing, so they have near-complete coverage.

| Feature | Definition |
|---------|-----------|
| `td2_downstream_ejc` | Number of exon-exon junctions downstream of the TD2 CDS stop (capped at 5) |
| `td2_log_utr3_length` | Log of the 3'UTR length from the TD2 CDS stop |
| `td2_atg_density` | Density of ATG codons in the 5'UTR (upstream of TD2 CDS start) |
| `td2_atg_strong_kozak` | Number of strong Kozak ATGs in the 5'UTR |
| `td2_uorf_count_overlapping` | Number of upstream ORFs overlapping the TD2 CDS |
| `td2_uorf_count_outframe` | Number of out-of-frame upstream ORFs |
| `td2_utr5_orf_coverage` | Fraction of the 5'UTR covered by upstream ORFs |
| `td2_stop_density` | Density of stop codons in the 5'UTR |

Source: `td2_features.tsv` (assembled from `utr5_features_all.rds`, `ptc.rds`, and `cds.rds`/`structures.rds`).

#### CDS source indicator (1 feature)

| Feature | Definition |
|---------|-----------|
| `ref_atg_available` | Binary: 1 if the reference CDS ATG is exonic in this isoform, 0 otherwise. Signals to the model whether the ref-CDS features are informative (real values) or uninformative (zeros). |

#### Design rationale

For isoforms where both feature sets are available (~75%), the ref-CDS and TD2 features often agree (when TD2 selects the correct reading frame) but can differ substantially (when TD2 selects an alternative ORF). The model can learn to weight each source appropriately. For isoforms where only TD2 features are available (~25%), the model uses the `ref_atg_available = 0` indicator to discount the zeroed ref-CDS features and rely on TD2 features and per-ORF/sequence inputs.

### Per-ORF features

These are computed for each of the 10 priority-ranked ORFs (ref CDS > SQANTI CDS > Kozak-ranked), independent of any CDS annotation beyond the `is_ref_cds` and `is_sqanti_cds` indicators:

| Feature | Definition |
|---------|-----------|
| `orf_length` | ORF length in nucleotides |
| `frac_position` | Fractional position of ORF start within the transcript (0 = 5' end) |
| `frac_tx_covered` | Fraction of transcript length covered by this ORF |
| `kozak_score` | Kozak consensus score (0-2): count of canonical positions matching (purine at -3, G at +4) |
| `n_upstream_atgs` | Number of ATG codons upstream of this ORF's start |
| `n_downstream_ejc` | Number of exon-exon junctions downstream of this ORF's stop codon |
| `has_downstream_ejc` | Binary: n_downstream_ejc > 0 |
| `is_ref_cds` | Binary: this ORF matches the reference CDS ATG (see above) |
| `is_sqanti_cds` | Binary: this ORF matches the SQANTI/TD2 CDS ATG. TD2 selects the highest-scoring ORF per transcript based on length and homology evidence (BLAST/PFAM); defaults to longest ORF without hits. |

Note that `n_downstream_ejc` and `has_downstream_ejc` at the ORF level are computed for each candidate ORF independently. They differ from the TX-level `ref_downstream_ejc` and `td2_downstream_ejc` which are specific to their respective reading frames. For the ORF that happens to be the reference CDS (or TD2 CDS), the values will agree; for other ORFs, they reflect a different stop codon position.

### Relationship to the prior structural elastic net

The prior 24-feature elastic net model (AUC = 0.94) used 12 TD2 features + 12 reference-CDS features. The ORF model extends this by: (1) replacing the TD2 per-ORF feature set with a multi-ORF approach — evaluating all 10 candidate ORFs via shared-weight encoding and attention selection; (2) retaining both ref-CDS and TD2 transcript-level features as parallel channels (8 + 8 = 16), plus the `ref_atg_available` indicator; (3) adding sequence-level features via CNN branches.

---

## Attention Analysis (`04_interpret_attention.py`)

### Data structure

The unit of observation for most attention analyses is an **ORF-within-a-transcript**. Each test-set transcript has up to 10 ORFs (top-K by Kozak score from orfik scan). Each ORF has an attention weight assigned by the model's attention aggregator (softmax-normalized across ORFs within a transcript, so weights sum to 1 per transcript).

Test set: 15,584 transcripts (chr 1, 3, 5, 7; paralog-free), yielding 155,840 ORF-level rows (including padding). Of these, 2,386 are NMD-sensitive.

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

Identifies NMD-sensitive test transcripts where no ORF in the top-K has a downstream exon-junction complex (`has_downstream_ejc == 0` for all ORFs). These are "no-PTC" cases — NMD isoforms the model cannot explain via the classical PTC + downstream EJC mechanism.

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
- **By ORF rank:** ORF feature importance broken down by rank (0-9), showing whether the model uses different features for the priority ORF vs lower-ranked ones
- **By CDS status:** ORF feature importance for ref-CDS ORFs vs non-ref-CDS ORFs among NMD transcripts, revealing whether the model treats the reference frame differently

---

## DeepSHAP Sequence Interpretation (`deepshap.py`)

### Method

DeepSHAP (Lundberg & Lee, 2017) computes per-position, per-channel attribution values for the 10-channel sequence encoding. We use the `shap.DeepExplainer` implementation, which applies the DeepLIFT algorithm through the CNN layers.

### Wrapper architecture

For each test sample, a `BranchWrapper` isolates a single sequence branch (ATG or stop) of the rank-0 ORF. All other inputs (other ORFs' windows, structural features, TX features) are held constant at their observed values. Only the target window varies, allowing DeepSHAP to attribute the model's output to specific nucleotide positions in that window.

### Background and sample selection

- **Background set:** 100 randomly sampled training transcripts (reflecting ~15% NMD prevalence). The SHAP baseline is the average model prediction over this background.
- **Explained samples:** 2,000 randomly sampled test transcripts.
- **Replication:** 5 independent runs with different random seeds (100, 200, 300, 400, 500) for background and sample selection. Channel-level CVs are 2-5% for nucleotide channels and 3-6% for the junction channel.

### Attribution metric

The primary metric is **SHAP × input**: the DeepSHAP value at each position multiplied by the input value at that position. For one-hot encoded nucleotides, this is nonzero only for the nucleotide that is actually present. Positive values push the prediction toward NMD; negative values push away.

### Edge artifacts

CNN-based SHAP values are elevated at the first and last ~25bp of the sequence window due to the CNN's receptive field extending beyond the window boundary. This was confirmed by comparing STOP=500 and STOP=1000 half-windows (1,000 vs 2,000 bp total): the elevated region moves with the window boundary, not with genomic position. The first and last bins of regional SHAP plots should be interpreted with this caveat.

### Landmark motif analysis (`07_motif_analysis.py`)

Per-nucleotide SHAP logos are extracted at ±15bp around biologically meaningful positions (stop codon, exon-exon junctions). For variable-position landmarks (junctions), each sample contributes its own landmark location, and SHAP values are aligned to the landmark position.

---

## Subgroup-Specific DeepSHAP Analysis (`08_export_subgroup_deepshap_tsv.py`)

### Subgroup definitions

NMD isoforms in the test set are classified into three subgroups based on the `category` field from `ref_cds_features.tsv` and the `td2_downstream_ejc` field from `td2_features.tsv`:

- **NMD PTC+**: `category == "effectively_ptc"`, OR `category ∈ {ref_atg_lost, no_ref_isoform, not_atg_in_target, no_stop_in_target}` with `td2_downstream_ejc > 0` (TD2 reclassification).
- **NMD PTC-, ref ATG retained**: `category ∈ {no_downstream_ejc, truncated_no_ejc}`.
- **NMD PTC-, ref ATG lost**: `category ∈ {ref_atg_lost, no_ref_isoform, not_atg_in_target, no_stop_in_target}` with `td2_downstream_ejc == 0` or NA.
- Non-NMD transcripts are labeled **Control**.

This logic mirrors the R report's `case_when` assignment in Section 7.1 (lines 1565–1581 of `orf_model_report_v5.Rmd`). A small number of NMD isoforms (~15) that do not fit any subgroup are classified as "NMD other" and excluded from subgroup analyses.

### Method

For each of the 5 DeepSHAP replicates (runs 1–5), the 2,000 explained test samples are mapped to subgroups via their `explain_indices` (which index directly into the test set, in the same order as `predictions_{tag}.tsv`). SHAP × input statistics are then computed per subgroup for:

1. **Kozak context**: Per-nucleotide SHAP at standard Kozak positions (-6 to +7, A of ATG = +1).
2. **5'UTR composition**: Mean channel SHAP upstream of the Kozak zone (> 6bp from ATG).
3. **Stop codon identity**: TGA/TAA/TAG frequency and total signed SHAP at variable stop codon positions.
4. **Downstream junction SHAP**: Junction channel SHAP in the 3'UTR portion of the stop window.
5. **Codon periodicity**: Per-frame nucleotide SHAP and lag-3 autocorrelation in the ORF body.
6. **Rolling GC importance**: By transcript region (5'UTR, CDS, ORF body, 3'UTR).

### Cross-run stability

Per-run subgroup statistics are aggregated to produce:
- **Mean** and **standard deviation** across 5 runs.
- **CV** (coefficient of variation): std / |mean|.
- **Sign consistency**: "YES" if the sign of the mean is the same in all 5 runs, "NO" otherwise.
- **95% CI**: mean ± 1.96 × SE, where SE = std / √5.

Findings are reported as robust only when sign-consistent across all 5 runs. Channels or positions with sign inconsistency or CV > 50% are noted but not presented as findings.

---

## Gene-Matched C2/C4 Analysis

A supplementary report uses gene-matched isoform pairs from the isopair analysis to control for gene-level confounds. C2 (NMD) and C4 (Control) pairs share the same (gene_id, reference_isoform_id). Only pairs where both comparators are in the test set are used. Structural importance is recomputed from per-sample grad × input vectors (`sample_importance_{tag}.npz`) filtered to gene-matched isoforms. DeepSHAP channel summaries are recomputed from the per-sample NPZ arrays filtered to gene-matched isoform IDs.
