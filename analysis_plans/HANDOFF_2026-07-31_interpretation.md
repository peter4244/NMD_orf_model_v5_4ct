# Handoff — §5 interpretation: where it stands and what the next window is for

Written 2026-07-31, end of a long session. Branch `master`, everything below is committed and
pushed.

---

## Start here

Read this file, then `ANALYSIS4_PLAN_2026-07-31.md` §2 and §5. Do **not** read the other analysis
plans on arrival — Analyses 1–3 are complete and their plans are only needed if you touch them.

**Your task is not to execute Analysis 4 as written.** It is to continue a line of thinking about
**how to identify important sequence features by model interpretation**, of which Analysis 4 is one
attempt. Four method choices have already been made and unmade, each on a measurement. That is not
churn; each was correct given what was known, and each was overturned by something measurable that
nobody anticipated. Expect a fifth.

**Model retraining is explicitly on the table** for this work (Pete, 2026-07-31). Several of the open
questions cannot be answered any other way, and the cost is small — the existing members early-stop
around epoch 4–7 and run ~14–17 epochs. Training cost has never actually been measured here; measure
one run before committing to five.

---

## The principle the session arrived at

**Importance is a contrast, not a measurement.** Every "this sequence is important" claim is
comparative, and needs three references. Each failed differently today, and each failure produced a
number that looked like an answer:

| reference | how it failed | measured |
|---|---|---|
| the method's own noise floor | DeepSHAP's completeness error exceeded the effect it decomposed | 129% (start), 307% (stop) |
| other positions | Kozak's *magnitude* was indistinguishable from arbitrary positions | 1.07× control |
| a matched population | restricting to long transcripts changed the same quantity 3.2× and produced a confident null | +0.0048 → +0.0015 |

**Enumeration was how Analyses 2 and 3 got the first reference for free.** With 8 or 32 coalitions the
residual is zero by arithmetic, so any deviation is a defect rather than an estimate. That is the
real reason enumeration mattered — not precision. For sequence positions the player set is 1,000–2,000
and enumeration is impossible, so every reference must be built empirically. That is the harder
problem and it is the one this line of work is actually about.

**The right statistic changes with the number of players.** For three branches or five features,
`mean|φ|` was correct. For positions it is nearly useless — Kozak scored 1.07× control on magnitude
and **100th percentile on across-member sign consistency**. With many weak players, *reliability*
discriminates where *magnitude* cannot. Which is also what a motif is: a consistent directional
preference, not a large one.

**⚠ ANALYSIS4_PLAN's headline positional statistic is `mean_abs`. It would have missed Kozak
entirely.** This is the most important single thing to fix.

---

## State of each analysis

| | status |
|---|---|
| **1 — performance** | Complete. `analysis1.py`, run log committed. AUC 0.9325/0.9310, AUPRC 0.8307/0.8351. |
| **2 — branch decomposition** | Complete. 50 cells, all accepted, reduced by `analysis2.py`. Structural 76.2%/66.3%. |
| **3 — structural features** | Complete. 50 cells, all accepted, reduced by `analysis3.py`. EJC 61.0%/64.4%; annotation pair 22.5%/23.0%. |
| **4 — positional** | **Plan only, and it needs a fifth revision.** Two defects below. |
| **5 — Kozak / ORF selection** | **Plan written 2026-07-31**, `ANALYSIS5_PLAN_2026-07-31.md`, from measurements made against matched control positions. Production run not yet made. |

**Analysis 4 inherits three things from Analysis 5's measurements**, none of them folded in yet:
the composition confound is the main term rather than a sensitivity (§5c treats GC-matching as the
latter); the method floor is ~7×10⁻⁷ at production batch size, not exactly 0.0 as §5a2 states; and
`mean_abs` remains the headline positional statistic.

Track A (the ledger window) has the Analyses 1–3 findings: `TRACK_A_HANDOFF_2026-07-31.md`, landed
there as D41, W143–W145, C76. Twelve sections; §11 and §12 were added after their commit and they
have not seen them.

**Nothing needs sending early.** Pete, 2026-07-31: Track A is standing off all model-related work
until Track B signs off. So findings accumulate in that document and go over in one handover — there
is no urgency even for §12, which measures that the method behind the published nucleotide-level
figures fails on this model. Do not interrupt them; add to the document.

---

## Analysis 4: two defects, both blocking

**1. The occlusion operator does not work.** §5a specifies 3-nt blocks replaced by a
dinucleotide-preserving shuffle. **A dinucleotide-preserving shuffle of a 3-mer is the identity map** —
verified exhaustively, 100% of 64 3-mers have exactly one valid arrangement. Every block effect would
be exactly 0.0. Worse, §6's no-op acceptance check asserts "a block substituted for itself gives
zero", so the criterion is *satisfied by* total failure.

Proposed fix (unimplemented): replace the block with a **GC-matched alternative 3-mer** — support 7
or 23, never degenerate, and channel 5 becomes invariant by construction, which removes the GC
recomputation confound from block occlusion entirely.

**2. The headline statistic is wrong**, per the principle above.

Both are unfixed. The plan also carries corrections from three narrow reviews that ARE folded in
(propagation across ORF windows, padding-aware profiles, created-motif exclusions, three variance
components, sign-stability precondition) — do not redo those.

---

## The Kozak result, and the four confounders that hid it

> **SUPERSEDED 2026-07-31, later the same day. The block below has no code behind it.** No script,
> run log, commit or surviving scratchpad produces these numbers; the search covered the repo, `git
> log --all` and every `/private/tmp/claude-*/…/scratchpad` on disk. The one committed Kozak
> artifact, `probe_kozak_ablation.py`, is a different experiment (n=300, no controls, −3 +0.00519).
> **Replaced by `ANALYSIS5_PLAN_2026-07-31.md` §8 and Track A handoff §13**, both reproducible from
> `measure_position_bank.py` / `analyse_position_bank.py` and their run logs. The re-measurement at
> n=500 with 30 matched controls agrees in direction and is weaker: −3 exceeds 93% of controls on
> the `purine` operator, not 100%. The confounder list below is sound and still applies; a fifth and
> a sixth have since been added (ORF attention, observed-base composition), and the operator itself
> turned out to be the largest defect — see §8 of the Analysis 5 plan.

```
Kozak −3   mean +0.00356   across-member |t| 6.89   all 5 signs agree   exceeds 100% of controls
Kozak +4   mean +0.00225   across-member |t| 2.73   signs disagree      exceeds  80% of controls
control positions: |t| median 1.28, 90th pct 4.66; only 20% show all-5 sign agreement
```

**The −3 purine preference is learned and reproducible; the +4 G preference is not.** That split is
biologically coherent (−3 is the stronger half of the consensus) and fits Pete's framing: Kozak is an
**ORF-selection** signal, not an NMD signal. The model needs to know which AUG the ribosome picks,
because that determines where the stop sits and therefore whether it is premature.

**Four confounders, each of which individually produced a wrong answer:**

1. **Created motifs.** +4 = T creates an in-frame stop codon in 18.9% of isoforms; −3 = A creates an
   in-frame AUG in 4.0%.
2. **Selection heterogeneity.** Rank 0 mixes three populations — purine-at-−3 is 79.6% for `ref_cds`,
   61.9% for `sqanti_cds`, 93.1% for `kozak`-selected. And rank 0 has the **weakest** Kozak context of
   all five slots (1.230 vs 1.665–1.879), because the ranking rule selects rank 0 by annotation and
   fills ranks 1–4 by descending Kozak score.
3. **Global attention re-weighting.** Ablating the annotation flags shrinks Kozak sensitivity 0.58×
   — but shrinks *control* positions 0.66×. The ablation down-weights rank 0 globally and says almost
   nothing about Kozak. **A flag-ablation result must always be normalised against control positions.**
4. **Population and power.** Requiring a wide unpadded span cut the effect 3.2× and left ~40 isoforms,
   which produced a confident null (|t| 0.80, signs disagreeing) that was purely underpowered.

**I concluded twice that there was no Kozak effect. Both conclusions came from a confounder, not from
the model.** Assume the same is true of any negative result in this area until it has been checked
against matched controls at matched n.

---

## What retraining could answer that inference cannot

Pete's open hypothesis: **the ORF structural features may override Kozak during training** — the model
never has to learn initiation context because `is_ref_cds` / `is_sqanti_cds` already identify the
start. Inference-time ablation **cannot** test this (confounder 3 above proves it measures attention
re-weighting). A model trained without those two features would.

Other retrain-answerable questions worth considering:
- Does the model learn +4 if −3 is unavailable, and vice versa?
- Is the structural branch's dominance (60–76%) a property of the task or of feature availability?
- Would a model trained without `n_downstream_ejc` learn junction position from the sequence channels?

---

## Practical facts, so they are not rediscovered

- **Local python:** `~/miniforge3/envs/nmd_model_local/bin/python`. The default has a torch/numpy ABI
  mismatch. **`shap` is NOT installed locally** — but nothing current needs it.
- **Explorer:** `p.castaldi@login.explorer.northeastern.edu`, project at
  `/home/p.castaldi/cc/nmd_orf_model_v5_4ct`, python at
  `/home/p.castaldi/.conda/envs/nmd_model/bin/python`. **Ask Pete before every login.**
- **Cluster:** use `short` (parallelises arrays, 2-day limit, ~125 nodes). `sharing` has a 1-hour cap
  that rejects these jobs. GPU is ~8% faster per job and the wrong choice for fleets. `slurm_logs/`
  must exist before `sbatch`.
- **Data:** windows are float16 in the HDF5, cast to float32 on load. Channel 5 (rolling GC) is
  **window-local** and exactly recomputable from channels 0–3. Channels 4 and 6–8 are
  sequence-invariant. Codon sits at array indices `W/2−1, W/2, W/2+1`; Kozak −3 = `W/2−4`, +4 =
  `W/2+2`. All verified at 100%.
- **Probes written this session** (all committed, all with run logs): `probe_deepshap_additivity.py`,
  `probe_ism_cost.py`, `probe_kozak_ablation.py`. The last is the closest thing to a working
  interpretation harness and is the natural base to build on.

---

## Open decisions for Pete

1. **Split Kozak out as Analysis 5?** It has a result and an understood design; Analysis 4 does not.
2. **Fix Analysis 4's statistic to sign-consistency-against-matched-controls**, which is a real
   redesign of §5b, not an edit.
3. **Run the flags-removed retrain?** It is the only route to the override question.
4. **Does the Kozak result need the propagated substitution** before it can go in the manuscript? All
   current numbers are window-local, which is the off-manifold version.

## What not to redo

The verified-facts list in the three review prompts (see git log for `876f21c` and the reviews that
followed) — coordinate convention, channel semantics, padding figures, reportable-position counts,
ORF window overlap, population sizes, per-pass cost. All measured with evidence. Re-verifying them
was the main cost of the earlier broad reviews; the narrow reviews that skipped them ran in a third
of the time and found more.
