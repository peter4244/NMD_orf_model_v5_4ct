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
| Start / stop window | 1000 nt each |
| Member seed | 42 |

Held-out performance, the dataset splits and the class balance are **not restated here**. They are
properties of the deposited artifacts and are read from them:
`metrics_atg1000_stop1000_seed42_test_clean.json` for the test metrics, and
`predictions_atg1000_stop1000_seed42_all.tsv` for the split sizes and label counts. A number copied
into this file is a number that can go stale while the artifact stays right, which has already
happened once.

> **`config.yaml` does not state the model's window sizes, and running without the flags builds
> the wrong tensors.** `data.window_size_atg` / `data.window_size_stop` in that file are the
> *sweep's starting grid point*, not the selected configuration. Any run that reproduces or serves
> this checkpoint must pass `--atg-window 1000 --stop-window 1000` explicitly; otherwise the
> tensors are built at a shape the checkpoint cannot consume. The selected values are recorded
> inside the checkpoint itself under `config.selected`, which is the authoritative copy.

## Labels

Isoform-level NMD calls come from multivariate adaptive shrinkage (mashr) across the four cell
types. Isoforms are split by chromosome, so no gene appears in more than one split.

- **NMD susceptible** — the union across cell types of isoforms with `nmd_responsive == TRUE`
  (lfsr < 0.05 and posterior mean logFC > 0).
- **Non-NMD** — the intersection across all four cell types of isoforms with `adj.P.Val > 0.30`.
- Isoforms in neither set are excluded from training. None fell into both.

Read-through loci are excluded: their `gene_id` values are composites of two adjacent genes
transcribed as one unit, which makes every gene-level operation on them — split assignment, the
paralog leakage screen — ill-defined.

## Architecture

Up to K=5 candidate ORFs per transcript pass through an `ORFEncoder` with weights shared across the
K ORFs, aggregated by learned attention. Within each encoder, three sub-encoders run in parallel and
share no weights: a CNN over the start window, a CNN over the stop window, and a structural branch
over the per-ORF features.

Full methods are the paper's Supplemental Methods, section "Deep Learning Model", and the layer
definitions are `model.py`. This file deliberately keeps neither a second copy. Read
`RETRAIN_ARCHITECTURE_CHANGES.md` before changing anything about the architecture or retraining.

**One consequence matters for reading the interpretation outputs.** The first convolution spans all
nine input channels at once, so no channel has an independent pathway through a branch. The branch
decomposition therefore attributes importance to **regions** — start window, stop window,
structural — and not to modalities; each region's share already includes its own GC and junction
content.

### A note on names

Prose here calls the two sequence windows the **start window** and the **stop window**, after the
codon each is anchored on. Code identifiers retain `atg` — the HDF5 key `atg_windows`, the
`--atg-window` flag, `window_size_atg` in config, and the `atg` element of file names. Renaming
them would change the HDF5 schema and break already-deposited artifacts, so the prose terminology
does not extend to the code.

Cell types are likewise `AT` and `DD` in code and in the feature tables, for AT2 and LAE
respectively.

## Repository layout

The build is fifteen steps. Everything needed to run them is at the top level; work that is not
part of that build sits in a folder, so the pipeline is what a reader sees first.

| where | what |
|---|---|
| top level | the pipeline — the numbered scripts, `model.py`/`utils.py`, both configs, `export_rds.R`, the report, the verification harnesses, and the fourteen `slurm_*_dn.sh` deposit-native drivers |
| `drivers/` | the other SLURM drivers, including the ones that reproduce the **published** run. Kept because the deposit-native rebuild exists to be compared against it |
| `ism/` | in-silico mutagenesis bank machinery. Not used by any result in the paper |
| `v6/` | a separate architecture explored after v5. Not the published model |
| `exploration/` | one-off analyses — cross-seed floor, k-mer controlled, run-length replicate |
| `analysis_plans/` | the plans those analyses were written from, with their run logs |
| `superseded/` | replaced code, kept rather than deleted |

Nothing was deleted in this reorganization. The tree archived at
[10.5281/zenodo.21536501](https://doi.org/10.5281/zenodo.21536501) v2.0.0 is unaffected, and git
history carries every path as it was.

## Build order

Fifteen steps, in order. The report `orf_model_report_v5.Rmd` is the last of them and consumes
intermediate TSVs that the earlier steps write:

```
1. export_rds.R                                     (Isopair RDS → the eight feature tables:
                                                     orf_features.tsv, tx_summary.tsv,
                                                     tx_summary_provenance.json,
                                                     ref_cds_features.tsv, td2_features.tsv,
                                                     junctions.tsv, paralog_genes.tsv,
                                                     val_paralog_genes.tsv, synthetic_cds.tsv)
2. drivers/slurm_build_h5.sh                                (data_prep.py → nmd_orf_data.h5, selected_orfs.tsv)
3. slurm_patch_selected_orfs.sh                     (scripts/patch_stop_codon.py — fixes stop codon
                                                     column off-by-one; required for §4.1 χ² test)
4. drivers/slurm_train_4ct.sh / drivers/slurm_train_4ct_sweep*.sh   (03_train.py)
5. drivers/slurm_interpret_v5.sh                            (evaluate.py, 04_, 05_)
6. drivers/slurm_deepshap_joint.sh                          (deepshap.py × 5 runs → deepshap_joint_*_run{1..5}.npz)
7. drivers/slurm_deepshap_structural.sh                     (deepshap.py --branches structural × 5 runs)
8. drivers/slurm_export_motif_v5.sh                         (06_, 07_ — marginal motif logos, fallback path)
9. drivers/slurm_export_joint_motif_logos.sh                (scripts/export_joint_motif_logos.py — preferred
                                                     5-run pooled motif logos for §3.1, §4.1)
10. drivers/slurm_export_subgroup_profiles_09b.sh           (09b_export_subgroup_profiles.py — feeds §3.1, §4.1,
                                                     §7.2, §9.4, §9.6, §9.7, §9.8)
11. drivers/slurm_export_features_09.sh                     (09_ GC, polya, junction-ordinal)
12. drivers/slurm_export_subgroup_v5.sh                     (08_ subgroup-specific marginal SHAP)
13. drivers/slurm_export_importance_v5.sh                   (05b_, 05c_ structural rollups)
14. drivers/slurm_kernel_shap.sh                            (11_ KernelSHAP branch decomposition)
15. drivers/slurm_render_v5.sh                              (renders orf_model_report_v5.Rmd)
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

The published-chain drivers are deliberately **not** checked: `drivers/slurm_train_4ct.sh`,
`drivers/slurm_interpret_v5.sh`, `drivers/slurm_kernel_shap.sh` and the rest pin 500/500 correctly, because they
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
`drivers/slurm_build_h5.sh`) and patched by **`scripts/patch_stop_codon.py`**.

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
