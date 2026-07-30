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

## Build order (intermediate TSVs feeding the report)

The report `orf_model_report_v5.Rmd` consumes a number of intermediate TSVs that
are not produced directly by the numbered pipeline scripts. Build order:

```
0. export_rds.R                                     (Isopair RDS → orf_features.tsv,
                                                     tx_summary_prelabel.tsv, ref_cds_features.tsv,
                                                     td2_features.tsv, junctions.tsv,
                                                     paralog_genes.tsv, val_paralog_genes.tsv)
1. relabel_tx_summary_4ct.R                         (tx_summary_prelabel.tsv → tx_summary.tsv
                                                     + tx_summary_provenance.json)
2. slurm_build_h5.sh                                (data_prep.py → nmd_orf_data.h5, selected_orfs.tsv)
3. slurm_patch_selected_orfs.sh                     (scripts/patch_stop_codon.py — fixes stop_codon
                                                     column off-by-one; required for §4.1 χ² test)
4. slurm_train_4ct.sh / slurm_train_4ct_sweep*.sh   (03_train.py)
5. slurm_interpret_v5.sh                            (evaluate.py, 04_, 05_)
6. slurm_deepshap_joint.sh                          (deepshap.py × 5 runs → deepshap_joint_*_run{1..5}.npz)
7. slurm_deepshap_structural.sh                     (deepshap.py --branches structural × 5 runs)
8. slurm_export_motif_v5.sh                         (06_, 07_ — marginal motif logos, fallback path)
9. slurm_export_joint_motif_logos.sh                (scripts/export_joint_motif_logos.py — preferred
                                                     5-run pooled motif logos for §3.1, §4.1)
10. slurm_export_subgroup_profiles_09b.sh           (09b_export_subgroup_profiles.py — feeds §3.1, §4.1,
                                                     §7.2, §9.4, §9.6, §9.7, §9.8)
11. slurm_export_features_09.sh                     (09_ GC, polya, junction-ordinal)
12. slurm_export_subgroup_v5.sh                     (08_ subgroup-specific marginal SHAP)
13. slurm_export_importance_v5.sh                   (05b_, 05c_ structural rollups)
14. slurm_kernel_shap.sh                            (11_ KernelSHAP branch decomposition)
15. slurm_render_v5.sh                              (renders orf_model_report_v5.Rmd)
```

## Inputs and regeneration (code-only deposit)

This repository ships **code only**. Nothing under `results_4ct/` is version-controlled
(see `.gitignore`); the HDF5 feature file, all input TSVs, and every interpretation export
are regenerated by running the pipeline in the **Build order** above. Starting data lives in
GEO (**GSE329233**).

The per-ORF/structural input TSVs (`orf_features.tsv`, `tx_summary_prelabel.tsv`,
`ref_cds_features.tsv`, `td2_features.tsv`, `synthetic_cds.tsv`, `junctions.tsv`,
`paralog_genes.tsv`, `orf_scan_metadata.json`) are produced in-repo by **`export_rds.R`**, which
reads the Isopair `Version_6.0/isopair_wrapper/data_mashr/analysis_cache` objects
(`ref_cds_features_all.rds`, `utr5_features_all.rds`, `ptc.rds`, `cds.rds`, `structures.rds`,
ORFik scan). **`relabel_tx_summary_4ct.R`** then reads `tx_summary_prelabel.tsv` and writes
`tx_summary.tsv` — the 4-CT-labeled universe, with neither-class isoforms dropped — plus
`tx_summary_provenance.json`, which `data_prep.py` requires before it will build an HDF5.

> **One writer per filename (2026-07-30).** `export_rds.R` and `relabel_tx_summary_4ct.R` both
> used to write `tx_summary.tsv`, so its contents depended on which ran last and nothing recorded
> which that was — running `export_rds.R` second meant training on the scaffold's original labels,
> silently and at exit 0. `export_rds.R` no longer writes `tx_summary.tsv` at all. Legacy
> `tx_summary_6ct.tsv` / `tx_summary_4ct.tsv` files are **not** valid substitutes for
> `tx_summary_prelabel.tsv`: neither was produced by this repo and neither has a recorded row
> count, so its vintage is unknown. Re-run `export_rds.R`.

The HDF5 dataset
(`nmd_orf_data.h5`) and `selected_orfs.tsv` are built by **`data_prep.py`** (see
`slurm_build_h5.sh`) and patched by **`scripts/patch_stop_codon.py`**.

> Earlier revisions symlinked `ref_cds_features.tsv`/`td2_features.tsv` from a sibling
> `nmd_orf_model` tree; those symlinks were dangling and have been removed. `export_rds.R`
> is the single in-repo producer.

To render `orf_model_report_v5.Rmd` you must run the full Build order first — the report
reads regenerated `results_4ct/` exports that are not shipped.

## Data Provenance

- **Sequences:** SQANTI-corrected FASTA from Isopair pipeline v6.0
- **ORF/structural features:** `export_rds.R` from the Isopair `Version_6.0` analysis cache (ORFik scan shared with original v5)
- **NMD labels:** 4-cell-type mashr DE results at `/projects/talisman/shared-data/nmd/mashr/` (relabeled by `relabel_tx_summary_4ct.R`)
- **Starting data:** GEO **GSE329233**
- **Original v5:** [peter4244/NMD_orf_model_v5](https://github.com/peter4244/NMD_orf_model_v5) — deprecated (6-CT scope; do not cite)

## superseded/

Non-canonical files kept for provenance (QC session logs, redundant slurm wrappers) live in
`superseded/`; nothing there is needed to run the pipeline. See `superseded/README.md`.
