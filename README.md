# NMD ORF Model — Four Primary Lung Cell Types

A deep learning model that predicts nonsense-mediated decay (NMD) from transcript sequence around
the start codon and the stop codon, together with per-ORF structural features. It is trained on NMD
classifications derived from four primary lung cell types: alveolar type 2 (AT2), large airway
epithelial (LAE), fibroblast (FB) and microvascular endothelial (MV).

The architecture is built for interpretability as well as accuracy: each candidate ORF is scored
through separate start-window, stop-window and structural branches, and a learned attention layer
exposes which ORF the model used for each isoform.

**This repository ships code only.** The trained checkpoint, its predictions and every
interpretation export live in the Zenodo record (concept DOI
[10.5281/zenodo.21544336](https://doi.org/10.5281/zenodo.21544336)); the starting sequencing data
are in GEO (**GSE329233**). Nothing under `results_4ct/` is version-controlled — see
**Inputs and regeneration**.

## The model

| | |
|---|---|
| Checkpoint | `best_model_atg1000_stop1000_seed42.pt` |
| Start window | 1000 nt |
| Stop window | 1000 nt |
| Member seed | 42 |
| Epoch selected | 5 (early stopping on validation AUC) |
| Validation AUC | 0.932 |
| Trainable parameters | 34,310 (12,866 in each of the two sequence CNNs) |

Held-out test set (chr1, chr3, chr5, chr7 — never seen in training or validation):

| | |
|---|---|
| AUC | 0.926 |
| AUPRC | 0.818 |
| Isoforms scored | 10,522 |
| NMD susceptible | 2,405 |

> **`config.yaml` does not state the model's window sizes, and running without the flags builds
> the wrong tensors.** `data.window_size_atg` / `data.window_size_stop` in that file are the
> *sweep's starting grid point*, not the selected configuration. Any run that reproduces or serves
> this checkpoint must pass `--atg-window 1000 --stop-window 1000` explicitly; otherwise the
> tensors are built at a shape the checkpoint cannot consume. The selected values are recorded
> inside the checkpoint itself under `config.selected`, which is the authoritative copy.

## Dataset

Isoforms are split by chromosome, so no gene appears in more than one split.

| Split | Chromosomes | Isoforms | NMD | Non-NMD |
|---|---|---:|---:|---:|
| Train | all others (19) | 26,720 | 5,959 | 20,761 |
| Validation | chr2, chr4 | 4,356 | 927 | 3,429 |
| Test | chr1, chr3, chr5, chr7 | 10,522 | 2,405 | 8,117 |
| **Total** | | **41,776** | **9,321** | **32,455** |

Class ratio 1:3.5 (NMD : non-NMD). Counts are those of the deposited model's own prediction table,
`predictions_atg1000_stop1000_seed42_all.tsv`.

**Labels.** Isoform-level NMD calls come from multivariate adaptive shrinkage (mashr) across the
four cell types:

- **NMD susceptible** — the union across cell types of isoforms with `nmd_responsive == TRUE`
  (lfsr < 0.05 and posterior mean logFC > 0).
- **Non-NMD** — the intersection across all four cell types of isoforms with `adj.P.Val > 0.30`.
- Isoforms in neither set are excluded from training. None fell into both.

Read-through loci are excluded: their `gene_id` values are composites of two adjacent genes
transcribed as one unit, which makes every gene-level operation on them — split assignment, the
paralog leakage screen — ill-defined.

## Architecture

Up to K=5 candidate ORFs per transcript pass through an `ORFEncoder` whose weights are shared
across the K ORFs, and are aggregated by learned attention.

Within each `ORFEncoder` three sub-encoders run in parallel and do **not** share weights with one
another: a CNN over the 9-channel start window, a CNN over the 9-channel stop window, and a
structural branch (`Linear(5, 32) → ReLU`) over the 5 per-ORF features. The three 32-dim
sub-embeddings are concatenated and fused (`Linear(96, 64) → ReLU → Dropout(0.2)`). The K ORF
embeddings are pooled by a learned attention layer (softmax over valid ORFs) and passed to a
classification head (`Linear(64,32) → ReLU → Dropout(0.3) → Linear(32,1)`). Training uses
`BCEWithLogitsLoss` with dynamic `pos_weight`.

Each sequence CNN is `Conv1d(9, 32, k1) → BatchNorm → ReLU → MaxPool(4) → Conv1d(32, 32, k2) →
BatchNorm → ReLU → max over the length axis → Linear(32, 32)`, with `k1, k2 = 15, 7` at this
window size.

**One consequence matters for reading the interpretation outputs.** The first convolution spans all
nine input channels at once, so no channel has an independent pathway through a branch. The branch
decomposition therefore attributes importance to **regions** — start window, stop window,
structural — and not to modalities; each region's share already includes its own GC and junction
content.

Full methods are the paper's Supplemental Methods, section "Deep Learning Model" — this repository
does not keep a second copy. Read `RETRAIN_ARCHITECTURE_CHANGES.md` before changing anything about
the architecture or retraining.

### A note on names

Prose here calls the two sequence windows the **start window** and the **stop window**, after the
codon each is anchored on. Code identifiers retain `atg` — the HDF5 key `atg_windows`, the
`--atg-window` flag, `window_size_atg` in config, and the `atg` element of file names. Renaming
them would change the HDF5 schema and break already-deposited artifacts, so the prose terminology
does not extend to the code.

Cell types are likewise `AT` and `DD` in code and in the feature tables, for AT2 and LAE
respectively.

## Pipeline

```
export_rds.R                # Isopair RDS -> the eight feature tables
data_prep.py                # Build HDF5 dataset
03_train.py                 # Train
evaluate.py                 # Evaluate on the test set
04_interpret_attention.py   # Attention analysis
05_interpret_structural.py  # Structural feature importance
deepshap.py                 # DeepSHAP (5 independent runs)
06-09_export_*.py           # Export interpretation TSVs
11_kernel_shap_branches.py  # Branch-level Shapley values
```

## Build order (intermediate TSVs feeding the report)

The report `orf_model_report_v5.Rmd` consumes intermediate TSVs that are not produced directly by
the numbered pipeline scripts. Build order:

```
1. export_rds.R                                     (Isopair RDS → the eight feature tables:
                                                     orf_features.tsv, tx_summary.tsv,
                                                     tx_summary_provenance.json,
                                                     ref_cds_features.tsv, td2_features.tsv,
                                                     junctions.tsv, paralog_genes.tsv,
                                                     val_paralog_genes.tsv, synthetic_cds.tsv)
2. slurm_build_h5.sh                                (data_prep.py → nmd_orf_data.h5, selected_orfs.tsv)
3. slurm_patch_selected_orfs.sh                     (scripts/patch_stop_codon.py — fixes stop codon
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

The `slurm_*_dn.sh` wrappers are the corresponding chain for a rebuild against the deposited
starting data rather than the working tree.

## Repository checks

`check_drivers.py` enforces one property: a **deposit-native** driver (`slurm_*_dn.sh`,
`submit_*_dn.sh`) derives the window, the member tag and the results tree from
`paths_config.py --selected-tag` rather than naming them. It exists because that property has
been broken twice — nine drivers after the 2026-08-04 re-selection, and two more found on
2026-08-11, one exporting subgroup DeepSHAP at 500/500 for a 1000/1000 checkpoint and one
writing into the deprecated `results_4ct_dn`. Every instance exited 0 and produced numbers.

The published-chain drivers are deliberately **not** checked: `slurm_train_4ct.sh`,
`slurm_interpret_v5.sh`, `slurm_kernel_shap.sh` and the rest pin 500/500 correctly, because they
reproduce a fixed historical run whose selection was 500/500.

It runs as a pre-commit hook. Git does not version `.git/hooks`, so enable it once per clone:

```bash
git config core.hooksPath hooks
```

A literal that is genuinely right belongs in `EXEMPT_LINES` with its reason, not behind
`--no-verify`.

## Inputs and regeneration (code-only deposit)

Nothing under `results_4ct/` is version-controlled (see `.gitignore`); the HDF5 feature file, all
input TSVs, and every interpretation export are regenerated by running the pipeline in the
**Build order** above.

The per-ORF and structural input TSVs (`orf_features.tsv`, `tx_summary.tsv`,
`ref_cds_features.tsv`, `td2_features.tsv`, `synthetic_cds.tsv`, `junctions.tsv`,
`paralog_genes.tsv`, `orf_scan_metadata.json`) are produced in-repo by **`export_rds.R`**, which
reads the Isopair `Version_6.0/isopair_wrapper/data_mashr/analysis_cache` objects
(`ref_cds_features_all.rds`, `utr5_features_all.rds`, `ptc.rds`, `cds.rds`, `structures.rds`,
ORFik scan). It also writes `tx_summary_provenance.json` beside `tx_summary.tsv`, which
`data_prep.py` requires before it will build an HDF5.

> **`export_rds.R` is the sole writer of `tx_summary.tsv`**, so there is one writer per artifact and
> the labels are unambiguously the scan's. What is *not* implied by that is **which** scan — the
> hazard `tx_summary_provenance.json` exists to close. It records the scan's identity and class
> counts, so a stale isoform universe is detectable rather than assumed. Without it, 0 of 24
> isoforms added by a rebuilt `structures.rds` reached `tx_summary.tsv` and nothing said so.

The HDF5 dataset (`nmd_orf_data.h5`) and `selected_orfs.tsv` are built by **`data_prep.py`** (see
`slurm_build_h5.sh`) and patched by **`scripts/patch_stop_codon.py`**.

To render `orf_model_report_v5.Rmd` you must run the full build order first — the report reads
regenerated `results_4ct/` exports that are not shipped.

## Data provenance

- **Sequences:** SQANTI-corrected FASTA from the Isopair pipeline v6.0
- **ORF and structural features:** `export_rds.R`, from the Isopair `Version_6.0` analysis cache
- **NMD labels:** carried by the ORFik scan and written by `export_rds.R`; the scan's vintage is
  recorded in `tx_summary_provenance.json`
- **Starting data:** GEO **GSE329233**
- **Trained model and interpretation exports:** Zenodo, concept DOI
  [10.5281/zenodo.21544336](https://doi.org/10.5281/zenodo.21544336)
