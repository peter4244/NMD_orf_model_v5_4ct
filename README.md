# NMD ORF Model v5 — 4 Cell Type Retrain

Deep learning model predicting nonsense-mediated mRNA decay (NMD) from ORF sequence context, retrained using only 4 primary lung cell types (AT, DD, FB, MV).

## Motivation

The original v5 model ([NMD_orf_model_v5](https://github.com/peter4244/NMD_orf_model_v5)) used NMD classifications derived from 6 cell types via multivariate adaptive shrinkage (mashr). Two cell types were subsequently excluded:

- **DD_ALI** — near-zero logFC correlation between short-read and long-read DGE (r = 0.002)
- **DO** — insufficient statistical power (n=2 donors after outlier removal)

Because mashr jointly estimates effect sizes across conditions, removing these cell types changes the shrinkage for all remaining cell types. This required re-running mashr with only the 4 retained cell types and retraining the model from scratch.

## Model Performance

Window size sweep (3 x 4 grid), sorted by AUPRC:

| ATG | STOP | AUC | AUPRC |
|-----|------|-----|-------|
| 500 | 1000 | 0.928 | 0.839 |
| **500** | **500** | **0.931** | **0.833** |
| 100 | 500 | 0.921 | 0.829 |
| 1000 | 1000 | 0.928 | 0.828 |
| 1000 | 500 | 0.929 | 0.824 |
| 100 | 1000 | 0.924 | 0.823 |
| 1000 | 100 | 0.925 | 0.817 |
| 500 | 2000 | 0.928 | 0.810 |
| 500 | 100 | 0.922 | 0.810 |
| 100 | 100 | 0.923 | 0.809 |
| 1000 | 2000 | 0.924 | 0.803 |
| 100 | 2000 | 0.919 | 0.797 |

**Selected configuration: ATG=500, STOP=500** (best AUC, consistent with original v5).

Compared to original v5 (AUC=0.93, AUPRC=0.78), AUPRC improved substantially (0.83), likely due to less imbalanced class ratio.

## Dataset

| | 4ct Model | Original v5 |
|---|---|---|
| Cell types | AT, DD, FB, MV | AT, DD, DO, FB, MV |
| NMD isoforms | 8,840 | 9,274 |
| Non-NMD isoforms | 31,098 | 52,395 |
| Total | 39,938 | 61,669 |
| NMD:non-NMD ratio | 1:3.5 | 1:5.6 |
| Non-NMD threshold | adj.P > 0.30 | adj.P > 0.50 |
| Test set (chr 1,3,5,7) | 10,131 | 15,584 |

The non-NMD threshold was lowered from 0.50 to 0.30 to account for changed p-value distributions under 4-cell-type mashr shrinkage.

## Architecture

Multi-branch transformer identical to original v5:
- Up to K=5 candidate ORFs per transcript
- Shared-weight CNN encoders (ATG window + stop window)
- Per-ORF structural features (5 features)
- Learned attention aggregation across ORFs
- BCEWithLogitsLoss with dynamic pos_weight

See `METHODS.md` for full details.

## Pipeline

```
relabel_tx_summary_4ct.R   # Reclassify NMD labels from 4ct mashr
data_prep.py               # Build HDF5 dataset
03_train.py                # Train model
evaluate.py                # Evaluate on test set
04_interpret_attention.py   # Attention analysis
05_interpret_structural.py  # Structural feature importance
deepshap.py                # DeepSHAP (5 independent runs)
06-09_export_*.py           # Export interpretation TSVs
11_kernel_shap_branches.py  # Branch-level Shapley values
```

## Data Provenance

- **Sequences:** SQANTI-corrected FASTA from isopair pipeline v6.0
- **ORF features:** ORFik scan (shared with original v5)
- **NMD labels:** 4-cell-type mashr DE results at `/projects/talisman/shared-data/nmd/mashr/`
- **Original v5:** [peter4244/NMD_orf_model_v5](https://github.com/peter4244/NMD_orf_model_v5)
