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

## BEFORE ANY RETRAIN OR RE-ARCHITECTURE

**Read [`RETRAIN_ARCHITECTURE_CHANGES.md`](RETRAIN_ARCHITECTURE_CHANGES.md) first.** It is
the accumulated list of things the current design does that interpretation found — the ATG
window's fill boundary leaking ORF length into the initiation head, the bin-max
representation discarding motif multiplicity and spacing, the forward separation between
the heads being given back by the loss, and four more. Each item states what it is, how we
know, and what a retrain should do.

Pointers to it also sit at the top of `03_train.py` and `train_v6.py`, because a retrain
can be started from either direction. **The file is the single copy — add findings there.**

## Working Conventions
- Best model tag: `atg500_stop500`
- All output goes to `results_4ct/`
- Original v5 project at `../nmd_orf_model_v5/` — do not modify

## Multi-window operating rules

Two or more windows work this repo concurrently, in separate git worktrees. These rules exist
because each one was learned by hitting it — see `analysis_plans/HANDOFF_INTERPRETABILITY_2026-08-02.md`.

- **Review the specification before writing the code.** When implementing an analysis row another
  window wrote, review it first and send back the choices it leaves open. **Do not resolve an
  ambiguity in code** — a choice made inside an implementation is invisible to review, and that is
  the origin of essentially every error this project has had. A row with an empty field in the
  thirteen-field template (`ANALYSIS_SEQUENCING_PROPOSAL.md`) is not ready to implement.
- **Two implementations means one shared specification and independent code.** Writing your own
  specification produces two analyses, not a replication.
- **Check the artifact before repeating a claim about it.** When any document says something is
  owed, blocked, settled or retracted, verify before acting: `git log --date=iso --format='%h %cd %s' -1 <sha>`.
  Prose is the one thing here that never gets independently recomputed, and both errors that survived
  two days of cross-checking lived in prose. A commit message is not evidence of its diff; `git show` is.
  A file mtime is not a commit time.
- **Namespace everything**, including cluster scratch scripts — outputs `interp_*` / `model_*`, jobs
  `hi_*` / `md_*`. Two windows once wrote `autocorr.py` to the same cluster directory and the second
  silently replaced the first.
- **Conference protocol.** A **conference** is called by Pete on a **specific topic**.
  While it is open, **every message goes to all conference members**, not just to
  whoever asked the question. Exceptions can be made per message; this is the
  default. **Communication reverts to baseline once the topic is deemed addressed** —
  and someone has to say so, or the conference never ends.

  *Why it exists, from 2026-08-02:* the results window sent a narrative-process
  design to the interpretability window without copying the model window, which
  owned the work; and the interpretability window summarised state for Pete using a
  framing the model window had already retracted, which the results window could not
  catch because they had never seen it. Both are bilateral-routing failures and both
  are what all-members messaging prevents.

- **Ask before every Explorer login.** A loaded ssh-agent is not standing authorization.
- **Where a document does not cover what you hit, add a row.** None of this structure was designed;
  it is a record of failures, so it stops exactly where our failures stopped. Do not work around a
  gap and do not assume the omission was considered.
