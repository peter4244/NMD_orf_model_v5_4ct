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

## Data facts

*What the artifacts contain, stated as facts rather than warnings — consult when you touch
the thing. These lived only in dated handoffs until 2026-08-02, so every window recopied
them and none could check them. Provenance is given for each; **two are marked unverified
and should be checked before they are relied on**.*

- **`structural` in the tensor is z-scored on the training split only. `structural_raw`
  holds the untransformed values.** `build_tensor.py:224`, `:291-292`; columns at `:55`.
- **`orf_length` in `orf_pool.tsv` is inclusive** — `orf_end − orf_start + 1`. *Verified
  across all 802,035 rows; the offset is exactly −1 everywhere.* Rank statistics are
  unaffected by this; differences, ratios and densities are not.
- **`vals` in the ISM bank has four base slots per position and only the three substitutions
  are filled, so the observed base stays NaN.** `np.isfinite(x).all(1)` is therefore never
  true — use `.sum(1) == 3`. `build_ism_bank.py:243`, `:756`. All three windows hit this.
- **`vals_capture` is roughly a thousandfold smaller than `vals_decay`, and the two have
  different noise floors** — capture carries no clamp, the transcript log-odds does. **A
  single liveness threshold applied across both is wrong in one of them.** h5 attribute
  `branch_resolution`.
- **`vals` ≠ `vals_capture` + `vals_decay`.** Stick-breaking is not additive in (p, d), so
  the residual is the interaction and is recoverable by subtraction. A base that moves
  initiation and a base that moves decay are different events and `vals` alone cannot tell
  them apart. h5 attribute `branch_attribution`.
- **The ISM bank is a stratified subset, not a sample of the pool.** Strata are
  `{is_nmd} × {has_annotation} × {main_orf_stop}`; weights sum to **41,765** over **4,999**
  rows; **annotated transcripts are ~8× over-represented** while recovery is scored against
  the annotation. Population estimates require `sampling_weight`. *Verified in
  `results_ism_v6/ism_subset.tsv`.* Exposure of any given statistic scales with how much its
  quantity depends on the three stratifying variables.
- **The keto and GC backgrounds are numerically identical in both existing measurements**
  (.501/.501 and .502/.502) — G+T equals G+C exactly when T equals C, and they agree here to
  a thousandth. "The background is .50" is therefore true of **two different quantities**,
  and reconciling composition tables by eye can match the wrong pair.
- **Explorer**: `p.castaldi@explorer.northeastern.edu`, `~/cc/nmd_orf_model_v5_4ct`, conda
  env `nmd_model`. Four-job cap shared across windows. The checkpoint every interpretability
  claim rests on is `runs/interp_c32_b8_s100/best.pt`, **which is not local**, and neither is
  the tensor.
- ⚠ **UNVERIFIED, carried forward from handoffs and not re-checked:** fold-over-median
  degenerates on the ~2.8% of transcripts whose median is below 1e-6 (prefer rank statistics
  by default). Stated here so it stops being invisible cargo, not because it has been
  confirmed.

## Roles — defined once, in the other repo

**`nmd_lung_longread_2026:CLAUDE.md`, section "Roles".** Seven: model window, interpretability,
storyteller, guardian of the manuscript, organizer, Pete, and Yul. A **pointer, not a copy** —
duplicating a definition into a second file is how the two drift apart. Change it there.

Two consequences that bite here: **escalation of a narrative to the guardian is Pete's call, not
the producing window's** (D60), and **agents propose, never promote** — only Pete or Yul move a
result to *in the paper* (D50).

## The standing hazard: position and length are entangled

*Restored 2026-08-02 after a restructure dropped it. It is D62, approved by Pete and corrected
once. It is not a data fact — it is an analysis hazard — which is why it needs its own section
rather than a bullet above.*

**No candidate-level correlation is interpretable until position and ORF length are conditioned on
jointly. Never read a single partial alone — each can be the other confounder leaking.
Conditioning moves results in BOTH directions. Assume position is involved; never assume which
way.**

Four instances in one day: **C7**, where holding position made the correlation *stronger*
(−0.582 against −0.460) because position was suppressing; **C8**, where length was the mediator;
the **ATF4 22.8%**, where position was proposed and refuted; and **C16**, the one case where a
partial manufactured a signal — "junction-seeking at matched length" +0.452, because holding
length does not hold position (`orf_start ~ orf_length` is only **−0.150**). Holding both returns
−0.067, agreeing with the marginal.

Candidates within one transcript span ~2 orders of magnitude in length — median within-transcript
max/min **83.5×** — so *holding length* describes a regime the model never occupies. **Prefer the
unconditional comparison for what the model does; use a partial to explain why, never to state
behaviour.** And **zero is not the null** for a rank product: a queue with no model in it scores
**+0.334 raw** against the model's −0.050.

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

  **For a conference about a SCIENTIFIC STORY the bar is higher** (Pete,
  2026-08-02): it ends only when the **storyteller says it should end** *and*
  **every member has read the story** and either agrees with it or agrees to stop
  critiquing it. Settling the *process* for documenting a story is not settling
  the story. The first declaration under the weaker rule was made while the
  guardian had explicitly not yet read the narrative, and was withdrawn.

  *Why it exists, from 2026-08-02:* the results window sent a narrative-process
  design to the interpretability window without copying the model window, which
  owned the work; and the interpretability window summarised state for Pete using a
  framing the model window had already retracted, which the results window could not
  catch because they had never seen it. Both are bilateral-routing failures and both
  are what all-members messaging prevents.

- **Ask before every Explorer login.** A loaded ssh-agent is not standing authorization.
- **Never use `/tmp` on Explorer. It is a shared multi-user filesystem.** Write under
  `$HOME/cc/` or `/scratch/p.castaldi/`, and namespace the filename. On 2026-08-04 a redirect to
  `/tmp/collect.txt` failed with "Permission denied" because that path already existed owned by
  another user — and the `cat` that followed printed **their** file, a pytest run from an
  unrelated project, which was read in full before anyone noticed it was not ours. Two failures
  in one line: our output went nowhere, and we read someone else's data. Neither announced
  itself; the command exited and produced plausible-looking text. The same collision can
  silently overwrite another user's file when the permissions happen to allow it.
- **Where a document does not cover what you hit, add a row.** None of this structure was designed;
  it is a record of failures, so it stops exactly where our failures stopped. Do not work around a
  gap and do not assume the omission was considered.
