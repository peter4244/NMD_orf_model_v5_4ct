# Start here — handoff to the next model-side window

*Written 2026-08-01. Deliberately without shorthand: two windows invented a lot of it yesterday and
it meant nothing from outside.*

## What you are here to do

Three things, in this order:

1. **Watch the retrain finish**, select one configuration, and run the second phase.
2. **Build the in-silico mutagenesis bank** the interpretability window needs.
3. **Hand it over** and let them compute the metrics.

You are the **model window**. The other window — titled "NMD Harold - Interpretability" — owns
interpretation and motif identification. Message it with `mcp__ccd_session_mgmt__send_message`; find
its id with `list_sessions`. Pete's instruction on 2026-08-01: **talk to it less.** Batch messages,
send one when something changes what the other window would *do*, not per finding.

---

## The one document that matters

`analysis_plans/RETRAIN_PLAN_2026-08-01.md` — sections 1 to 8, all implemented. It is a
**specification of what is done to the data**, not an argument; the reasoning lives separately in
`RETRAIN_RATIONALE_2026-08-01.md`. Keep it that way. The standard it is written against is
`~/.claude/projects/-Users-petecastaldi/memory/feedback_analysis_plan_standards.md` — read that
before editing the plan, not after.

Every number in section 3.4 is produced by the code that implements section 3, and printed in
`analysis_plans/build_orf_pool_runlog.txt`.

---

## Job 1 — the retrain

### What is running

SLURM array **8876808** on Explorer, `~/cc/nmd_orf_model_v5_4ct`. Eight tasks, five runs each,
40 runs total: 8 configurations × 5 seeds of the interpretable model.

The GPU queue allows **8 submitted jobs and 4 running**, which is why it is 8 chunked tasks and not
a 40-task array. Four run at a time; the rest wait.

### Checking it

```bash
ssh p.castaldi@explorer.northeastern.edu
cd ~/cc/nmd_orf_model_v5_4ct
squeue -u $USER
python sweep_v6.py --collect sweep/phase1.tsv      # progress and results so far
```

Ask before every login. A loaded ssh-agent is not standing permission — Pete grants it per session
and granted it for the session that launched this.

### If a task is cut off

**Resubmit the same array.** `sbatch --array=1-8 sweep_v6.sbatch sweep/phase1.tsv 5`. A run that
already finished is skipped; a run that was interrupted resumes from its own checkpoint, bit-exactly
— verified, max difference 0.00e+00 against an uninterrupted run at the same seed. This is the
designed recovery path, not a repair.

### When all 40 are done

```bash
python sweep_v6.py --select sweep/phase1.tsv       # picks a configuration on val_clean ONLY
python sweep_v6.py --phase 2 --out sweep           # writes the arms at that configuration
sbatch --array=1-3 sweep_v6.sbatch sweep/phase2.tsv 5
```

**Nothing has touched the test split and nothing should until a configuration is fixed.** Selection
is on `val_clean` — the paralog-screened validation set, 4,356 transcripts. Test is `test_clean`,
10,520. Reporting test metrics for 40 configurations and then quoting the best is the failure the
two-phase structure exists to prevent.

### The result to watch for

**The permuted-bin control.** It is configuration `interp_c32_b8_perm`: identical parameter count to
`interp_c32_b8`, with the bin order scrambled per candidate at every forward pass so positional
information is destroyed and nothing else is.

If it matches the real configurations, **the model is not using position**, and the central
architectural claim of the rebuild fails. That is a finding, not a configuration — `--select`
already refuses to carry it forward if it wins, and says so.

The permutation is redrawn every pass on purpose. A permutation fixed at initialisation is not a
control: the projection is fully connected over the flattened bins and can undo any fixed reordering
by permuting its own weights.

### Where it stood at handoff

4 running, 4 queued, 0 of 40 complete, 16 minutes in. Epoch time 170–270 s. Convergence looks fast —
the 32-channel run was at patience 2 by epoch 4 — so runs should stop nearer 10–15 epochs than 40.
Estimated **6–8 hours** total.

Validation performance at epoch 2–4, across the four channel configurations: **AUC 0.943–0.948,
AUPRC 0.867–0.881.**

**Do not compare that to the old model's 0.9310 / 0.8351 yet.** Those were the *test* set of a
*different universe* (10,131 transcripts); these are validation on this one (4,356). The comparison
only becomes honest when the selected configuration is evaluated once on `test_clean`, and even then
the old checkpoint should be re-evaluated on the same split rather than quoted — AUPRC is
prevalence-dependent and does not transport.

What it does show: withholding four of five tabular features has not collapsed accuracy.

---

## Job 2 — the mutagenesis bank

This is the substantial piece of new work and **it is not planned yet**. It has no section in the
plan. Write one before building it.

### What the interpretability window asked for

Verbatim from their message, in the format their existing stage reads, so no conversion layer exists
to get wrong:

```
vals           (n_iso, W, 4)  float32   delta logit for substituting each base; NaN where invalid
valid          (n_iso, W)     bool
obs            (n_iso, W)     int8      observed base index, ACGT = 0123
labels         (n_iso,)
floor          scalar                   max |no-op| — substituting the observed base for itself
transcript_id  (n_iso,)
```

**Three requirements they were explicit about, each of which cost them something:**

- **Propagate every perturbation to every window containing that transcript coordinate.** Candidates
  overlap, so changing one window alone presents the same base as two different bases at once — the
  off-manifold condition the method exists to avoid. Their own local version does not do this and
  they flagged it as a limitation yours should not inherit.
- **Record the no-op floor.** Substituting the observed base for itself must come out near zero.
  Theirs is 4.8e-07.
- **Exclude nothing at the edges, but ship the extent of each window in transcript coordinates.**
  They do the exclusion. The reason: channel 5 is a 50-base rolling mean, so within 25 bases of a
  boundary a substitution perturbs a truncated denominator. That artifact put 13 of their top 15
  candidates at consecutive extreme offsets until they caught it.

They corrected themselves on the last point: with 19 candidates spread 5′→3′ the union of window
spans is likely the whole transcript, so the extent they need is **per candidate window, not per
transcript**. Ship the spans explicitly rather than let them reconstruct geometry — reconstructing
instead of reading is what cost an hour on `orf_rank`.

**Subset:** ~1,000 transcripts, stratified by label, respecting **a gene-level discovery /
confirmation split that you fix and ship** so both windows use the same one. That split does not
exist yet. They cannot say what n is required — their pilot on 120 gave a top strength of 2.5 against
a null edge of 1.0 and they do not know where the curve turns.

**Arms:** banks for the interpretable model, the **permuted-bin control** (a null they cannot build
themselves — they asked to protect this one), and sequence-blanked. Scalars only for predictor and
no-junction. If banks get expensive, cut sequence-blanked first, then no-junction.

### What is known about the cost, and what is not

They priced it at **~150 CPU-hours** for 1,000 transcripts whole-transcript. **That is a laptop CPU
measurement and should not be budgeted from.** On the V100 here, forward-and-backward runs at about
85,000 candidates per second, so the compute is minutes, not hours — the estimate is dominated by
hardware, not by the task. **Re-measure on this hardware before requesting nodes.**

Three things they got right that save real work:

- **Propagation does not multiply the cost.** One perturbation is one forward pass however many
  windows it touches; it changes what is written into the input, not how many passes run.
- **The no-op floor is a property of the arithmetic, not the position.** Sample it at a few hundred
  positions instead of all of them: 25% back.
- **Memory is the likelier wall than compute.** `probe_ism_cost.py` in this repo already caught a
  35.3 GiB peak against a 32 GB request on the *old* architecture at 5 slots. This pool is ~19
  candidates per transcript, so roughly 4× that per-batch footprint. Re-run that probe against the
  new pool first — it has already saved this project once.

**One optimisation neither window has costed.** A perturbation at position *p* only changes the
windows covering *p*. The other candidates' capture and decay values are unchanged and can be cached
— but the stick-breaking product couples them, so the aggregation still has to be recomputed for the
whole transcript. With windows spread across the transcript that is plausibly a 4–8× saving on the
encoder, which is the expensive part.

---

## Job 3 — the metrics

Not yours. The interpretability window computes them from the bank. Your job ends at handing over a
bank that states its own population.

---

## What is settled — do not re-derive

- **The pool.** 802,035 candidates over 42,043 transcripts, mean 19.1, ordered 5′→3′ by transcript
  position. Admission: MANE q05 Kozak floor (−1.2508, read from Isopair's calibration at run time),
  start in the first half of the transcript, the reference start codon always, and the five
  highest-scoring candidates where that leaves the pool empty. Rebuilds to the same sha256.
- **The window geometry.** Both windows 1000 wide. The ATG window is **asymmetric** — 900 upstream,
  100 into the ORF, anchored at array index 900. The stop window is symmetric, anchored at 500.
  Filling is bounded at the ORF midpoint so no base is read by both windows; `mid` belongs to the
  ATG window.
- **The storage.** One `uint8` per position per window: bits 0–2 the fill state (0 unfilled, 1–4
  ACGT, 5 filled but not ACGT), bit 3 a junction. `tensor_io.py` holds the only definition of the
  mapping and it is checked against the nine-channel encoder it replaced, not against a description
  of it. 0.48 GB on disk, 1.59 GB resident.
- **The architecture.** Three separate encoders. Capture reads the start-codon window alone, through
  its own weights — verified invariant to the stop window while decay moves with it. Aggregation is
  stick-breaking in log space; selection masses plus the residual sum to exactly 1.
- **The junction feature stays as agreed** — thresholdless count becomes the 50-base rule, and stops
  there. Graded distance was measured and buys nothing for discovery: median shift 0.45 percentage
  points across candidate sequence features.
- **The 2,334 `no_ref_isoform` transcripts stay in training.** They are 24.8% of all positives,
  respond in all four cell types more consistently than the rest, and are enriched ~2× for the
  upstream-ORF mechanism. Deleting them would preferentially delete what the rebuild is for.

## Traps this project has paid for

1. **Read the code, not the description of it.** Nearly every correction today came from opening the
   producing file. Two reviews of the plan found fourteen defects; three would have produced silently
   wrong results.
2. **A control must be power-matched and computed over the same strata as its target.** A retraction
   of a published claim was wrong because a control had 214 observations against 5,818 and a
   different set of strata. Sweeping many positions rather than three is what settled it.
3. **A percentage-point difference cannot be compared across groups with different base rates.**
   Always report the odds ratio beside it. This cost two retractions in one day.
4. **A failing step does not stop the next one.** Twice today a patch raised and the commit went
   ahead anyway; once an scp failed and the job submitted anyway. Chain with `&&`, or check.
5. **`ast.parse` is not `compile`.** A `global` declaration in the wrong place parses and does not
   compile.
6. **numpy string arrays take their width from the first literal.** `np.where(keep, "floor", "")`
   silently truncated "reference" to "refer" and made a comparison against the full word permanently
   false.
7. **Do not encode contextual judgement as a rule in code.** Pete, twice. Make the facts visible and
   let the analyst decide. A guard written after one mistake would not have caught that mistake.

## Practical facts

- Local Python: `~/miniforge3/envs/nmd_model_local/bin/python`. The default has a broken numeric
  library.
- Explorer: `p.castaldi@explorer.northeastern.edu`, `~/cc/nmd_orf_model_v5_4ct`, conda env
  `nmd_model` (torch 2.5.1+cu121, pandas 3.0.1). **Ask before every login.**
- Explorer home has a per-user quota. `~/cc` sits at 38 GB. `superseded_2026-07-29` and
  `nmd_orf_model_v5` were left alone deliberately — the first because its README records decision D4
  that superseded artifacts are marked and never removed, the second because this project's
  `CLAUDE.md` says do not modify it.
- Quota accounting lags after a delete by a minute or two, so a write that fails may succeed shortly
  after. Do not conclude "still over quota" from one attempt.
- The tables on Explorer under `results_4ct_dn/` are byte-identical to the local
  `nmd_w69_tables_2026-07-30/`. Verified by sha256, all five.
- `infer_uorf_attention.py` and the three R metric scripts read **attention weights that no longer
  exist**. Section 4 of the plan specifies the repair; it has not been done, and it cannot be
  finished until the export format is fixed, because stick-breaking emits `p_k`, `P_select` and `d_k`
  rather than one attention weight.
