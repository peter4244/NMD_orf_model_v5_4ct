# NMD ORF Model v5 — 4 Cell Type Retrain

## Project Overview
Deep learning model predicting nonsense-mediated mRNA decay (NMD) visibility from ORF sequence context. This is a **retrain of the v5 model** using only 4 non-ALI cell types (AT, DD, FB, MV), excluding DO (insufficient pairs, n=2 after outlier removal) and DD_ALI (poor short/long-read concordance).

The model architecture is identical to the original v5: multi-branch transformer processing up to K=5 candidate ORFs per transcript through shared-weight CNN encoders (ATG window + stop window + structural features), aggregated via learned attention.

**Primary model configuration:** ATG=500, STOP=500.

## Key Differences from Original v5 (nmd_orf_model_v5)
- **Cell types:** AT, DD, FB, MV only (was AT, DD, DO, FB, MV)
- **mashr re-run:** New mashr DE results with 4 cell types (shrinkage estimates change with fewer conditions)
- **Non-NMD threshold:** adj.P.Val > 0.30 (was 0.50) — lowered because 4-cell-type mashr shifts p-value distributions
- **NMD definition:** Union of nmd_responsive == TRUE across 4 cell types
- **Non-NMD definition:** Intersection of adj.P.Val > 0.30 across all 4 cell types
- **Dataset size:** ~39,938 isoforms (8,840 NMD / 31,098 non-NMD, ratio 1:3.5) vs original ~61,669 (9,274 / 52,395, ratio 1:5.6)
- **Results directory:** `results_4ct/` (not `results/`)
- **New mashr results:** `/projects/talisman/shared-data/nmd/mashr/` (old 6ct results in `old_6celltype/` subfolder)

## Repository Structure

### Source Code (pipeline order)
- `export_rds.R` — Isopair RDS → the eight feature tables, incl. `tx_summary.tsv`
  (sole writer) and `tx_summary_provenance.json`. `relabel_tx_summary_4ct.R` is RETIRED
  (D18) and is NOT a build step — do not reinstate it.
- `data_prep.py` — HDF5 dataset construction
- `model.py` — NMDOrfModel architecture definition
- `config.yaml` — Hyperparameters and paths
- `utils.py` — Shared utilities
- `03_train.py` — Model training (BCEWithLogitsLoss, Adam, early stopping on val AUC)
- `evaluate.py` — Test-set evaluation, metrics JSON, predictions TSV
- Interpretation scripts (04–11) — Same as original v5, updated for results_4ct paths
- `export_rds.R` — R-side data export
- `orf_model_report_v5.Rmd` — Analysis report (will need updating for 4ct context)

### Results Directory (not in git)
- `results_4ct/` contains all outputs: model weights, predictions, metrics, HDF5 training data
- Input TSVs (orf_features, junctions, paralogs, etc.) are symlinked from the original nmd_orf_model results
- `tx_summary.tsv` is a real file with 4ct-relabeled NMD/non-NMD assignments

### SLURM Scripts
`slurm_*.sh` — Cluster job scripts, all pointed at this project directory and results_4ct.

## Data Provenance
- ORF features, junctions, sequences: Same as original v5 (from isopair pipeline v6.0)
- NMD labels: New 4-cell-type mashr DE results at `/projects/talisman/shared-data/nmd/mashr/`
- Labels: carried by the ORFik scan and written by `export_rds.R`; vintage recorded in
  `tx_summary_provenance.json`. `relabel_tx_summary_4ct.R` is retired (D18).

## Working Conventions
- Best model tag: `atg500_stop500`
- All output goes to `results_4ct/`
- Original v5 project at `../nmd_orf_model_v5/` — do not modify
