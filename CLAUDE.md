# NMD ORF Model v5

## Project Overview
Deep learning model predicting nonsense-mediated mRNA decay (NMD) visibility from ORF sequence context. This is a multi-branch transformer that processes up to K=5 candidate ORFs per transcript through shared-weight CNN encoders (ATG window + stop window + structural features), aggregates via learned attention, and classifies NMD status.

**Primary model configuration:** ATG=500, STOP=500 (best AUPRC in 3×4 window sweep).

## Repository Structure

### Source Code (pipeline order)
- `data_prep.py` — HDF5 dataset construction: ORF selection, sequence encoding (10 channels), window extraction, train/val/test split (test = chr 1,3,5,7 paralog-free)
- `model.py` — NMDOrfModel architecture definition
- `config.yaml` — Hyperparameters and paths
- `utils.py` — Shared utilities
- `03_train.py` — Model training (BCEWithLogitsLoss, Adam, early stopping on val AUC)
- `evaluate.py` — Test-set evaluation, metrics JSON, predictions TSV
- `04_interpret_attention.py` — Attention weight extraction and entropy analysis
- `05_interpret_structural.py` — Structural feature importance (DeepSHAP on ORF features)
- `05_export_sample_importance.py` / `05b_export_sample_importance_tsv.py` — Per-sample importance NPZ/TSV export
- `deepshap.py` — DeepSHAP on sequence branches (ATG/stop/joint), 5 independent runs
- `06_export_deepshap_tsv.py` — Positional/regional DeepSHAP summary TSVs
- `07_motif_analysis.py` — Motif logos and k-mer enrichment from DeepSHAP
- `08_export_subgroup_deepshap_tsv.py` — Subgroup-stratified DeepSHAP exports (Kozak, stop codon, junction, frame periodicity, GC, 5'UTR)
- `09_export_*.py` — Additional feature exports (GC content, junction ordinal, poly(A))
- `11_kernel_shap_branches.py` — Kernel SHAP for branch-level Shapley values
- `export_rds.R` — R-side data export
- `make_architecture_figure.R` — Architecture diagram
- `orf_model_report_v5.Rmd` — Full analysis report (R Markdown → HTML)

### Key Documentation
- `METHODS.md` — Detailed methods: architecture, encoding, feature definitions, DeepSHAP methodology
- `BUGFIX_STOP_CODON_2026-03-31.md` — Documents stop codon off-by-one bugs found and fixed (model unaffected, only metadata and one interpretation export were impacted)

### Results Directory (not in git)
- `results/` contains all outputs: model weights (`.pt`), predictions, metrics JSON, DeepSHAP NPZ arrays, attention/importance/motif/subgroup TSVs, HDF5 training data
- `figures/` contains report figures (PDF + PNG)
- These must be copied separately (e.g., `rsync`) — they are gitignored

### SLURM Scripts
`slurm_*.sh` — Cluster job scripts for each pipeline stage. These reference the cluster environment and may need path adjustments for other machines.

## Data Provenance
- Input data originates from the isopair pipeline at `/projects/talisman/shared-data/nmd/isoform_transitions/Version_6.0/isopair_wrapper/`
- Sequences from SQANTI-corrected FASTA; junctions from `structures.rds`; ORFs from ORFik scan
- Training data assembled into `results/nmd_orf_data.h5` (~5.8GB, not in git)

## Known Issues (resolved)
- Stop codon position off-by-one in ORFik scan metadata and subgroup DeepSHAP export — fixed 2026-03-31, documented in `BUGFIX_STOP_CODON_2026-03-31.md`. Model weights and primary DeepSHAP arrays were never affected.

## Working Conventions
- The report (`orf_model_report_v5.Rmd`) reads all data from `results/` relative to project root
- DeepSHAP uses 5 independent runs for stability; joint DeepSHAP NPZs are ~480MB each
- Best model tag throughout the codebase: `atg500_stop500`
