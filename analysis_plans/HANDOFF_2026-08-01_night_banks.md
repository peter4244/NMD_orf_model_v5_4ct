# Start here — handoff, night of 2026-08-01

*Written plainly, per Pete's standing instruction: no invented shorthand.*

Supersedes `HANDOFF_2026-08-01_evening_decode.md`, whose one job is finished.

## What happened

**The decode optimisation is done, verified, and measured.** The bank build no
longer rebuilds all 1,000 positions of a window for every mutated row; it decodes
each candidate's window once and patches the ~51 positions a substitution can
reach. **8.4× end to end**, measured on 50 real transcripts with startup
amortised (4,315 substitutions/s against 514).

Do not quote 22×. That figure is the input-building microbenchmark alone — real,
repeated, and on the right hardware, but the bank also pays the per-position
writeback, three aggregations and the stop-window encoder. 8.4× is the number.

**The five interpretable banks are built** (job 8885690, seeds 100–500, chunk
4,096). §8.5 is written, amended and deliberately **unread**.

## Why the bank can be trusted

Three checks, all on the GPU the bank runs on, all of which were first shown to be
capable of failing.

    step 1  patched windows vs decode_windows, 449,344 substitutions over 4,030
            candidates, both window kinds: ZERO differing entries, max |diff|
            exactly 0.0. Both channel-5 branches exercised at 224,672 each.
    step 3  a bank rebuilt with the pre-cache builder at K=102, K=103 and K=1,
            and again at 50 transcripts: 37 datasets and every attribute bitwise
            identical.
    step 2  verify_ism_bank against the model's own forward(): obs matches the
            FASTA with 0 mismatches, spans clean both ways, vals match the
            uncached reference to 2.842e-09.

The equality is exact rather than approximate for a reason worth keeping: channel
5 is `num/den` where both are counts over at most 1,000 positions and so are exact
integers in float32. A substitution moves the count by exactly ±1 and leaves the
denominator alone, so patching and recomputing perform the identical division.

Three deliberate mutations — the span patch disabled, the span shortened by one at
either end, the wrong fill states read as GC — each produced thousands of
differing rows. The check can fail.

## The thing that nearly went wrong, and the pattern in it

I launched all five production banks at chunk 16,384 for an 8% speed gain, having
verified everything at 4,096. The same 50 transcripts built at the two shapes are
**not the same bank**: `vals` differ by up to 1.77e-06, `vals_capture` by
4.32e-07, and the reported `batch_shape_offset` itself moves 8.79e-07 → 8.40e-07.
Not the cache — cached and pre-cache builds at 4,096 are bitwise identical — but
chunk shape changing float accumulation order, which the same-chunk baseline
cancels *within* a chunk and not *between* chunk choices. Cancelled four minutes
in, deleted the 1,172 shards (the builder skips shards on disk, so a relaunch over
them would have mixed two shapes with the floor varying by build order), relaunched
at 4,096.

**Every error tonight was in the scaffolding, not in the thing being verified.**
The profiler substituted at positions the bank never uses. The reference builder
ran from the wrong directory. I explained a real discrepancy with a false cause.
I launched at an unverified chunk size. `tx_length` was derived instead of read.
The checkpoint path came from the wrong machine. The cache itself was correct from
the first write and survived every check.

That is the lesson to carry into tomorrow. The bank is verified three ways; the
risk is not that it is wrong. The risk is the **analysis harness around it** —
subset reweighting, floor comparisons, split filtering, seed pooling. Put the
checks there.

## The insight I would act on first

**Verify against the producing code, never against a restatement of it.** This
recurred three times in one evening and the restatement lost every time:

- §8.5 gave right-hand fill as `min(100, (orf_length // 2) + 1)`. Read with the
  pool's `orf_length`, which is inclusive, that overstates fill by one on 36% of
  candidates — and it would have admitted candidates whose true fill is 99 as
  though it were 100, in the one place the midpoint-clip control is enforced.
  `window_spans` is right; the plan now says so and stops restating the geometry.
- `tx_length` derived as the largest `orf_end` is wrong for every transcript whose
  3′-most ORF ends before the transcript does. It is read from
  `discovery_confirmation_split.tsv`.
- The handoff said checkpoints live at `results_interp_all/v6_checkpoints/`. True
  on the laptop; on the cluster they are `runs/interp_c32_b8_s<SEED>/best.pt`.

I also got the *explanation* of the first one wrong — I blamed a transcript-length
clip that is never active, since `orf_end <= tx_len` for all 796,584 candidates.
The interpretability window caught it and I confirmed it independently.

## Three measurements now say the same thing about the capture head

Independent, and they converge:

    decomposition (§8 arms)   sequence ~0.004 AUC against selection's ~0.057
    §8.5 on validation        capture 0.5799 on geometry-matched pairs, gene CI
                              [0.5066, 0.6486] — clears chance by 0.0066
                              kozak_score 0.5167 on the same pairs
                              orf_length 0.9535, which the model cannot see
    the bank itself           p_capture median 0.0132; 88.8% of candidates carry
                              under 1e-4 of the selection mass

Read together: the initiation head is weakly discriminative, most candidates
**cannot express an effect at all**, and what discrimination exists is
seed-dependent — the §8.5 statistic ran 0.528 to 0.595 across the five seeds, a
spread of 0.067 on an effect of 0.080.

The claim that selection dominates sequence is safe; it comes from the ablation
arms. The claim that the model **recognises start codons** is not supported, and is
now contradicted by two independent measurements. Frame tomorrow's analysis as
"what does the model do" rather than as confirming initiation, and keep the
interpretability window's rule: name the structure, never the finding. "5′ scanner"
describes the architecture. It is not a claim about what the architecture learned.

## The gating question for tomorrow, and a prediction

**What fraction of `vals_capture` clears the floor on the real banks?**

On a local three-transcript bank, median |vals_capture| is 1.32e-06 against median
|vals| of 1.69e-03 — three orders of magnitude apart — while the batch-shape floor
on the GPU is ~8e-07. The interpretability window measured the same shape
independently: median |vals_capture| near 9e-05 against |vals| near 1e-02, with 43%
of capture entries below 3e-06.

If that holds, **roughly half the capture arm is unreadable**, and a positional
profile computed over `vals` without conditioning would be a profile of the decay
arm wearing the capture arm's name. `qc_ism_banks.py` answers this directly and its
output is below.

**A hypothesis to test rather than assume:** if `vals_capture` is floor-limited,
`dsel` and `dstart` may still be usable. They are not differences of two nearly
equal aggregates — `dsel` is a total variation summed over K, which accumulates
signal where a difference cancels it. That is a reason to expect better
signal-to-noise, not a demonstration of it. Measure before relying on it.

**A prediction I am recording so it can be wrong:** cross-seed sign agreement will
be high for `vals` and low for `vals_capture`. If it is low, capture-arm findings
need cross-seed replication as a gate, not as a robustness check.

### "Is it robust" hides three different questions

Do not collapse them. Collapsing them is the same error as carrying one design
effect across statistics.

| what varies | what is fixed | instrument | question it answers |
|---|---|---|---|
| initialisation | the transcripts | cross-seed **sign** agreement (`qc_ism_banks.py`) | do the seeds agree which *direction* each substitution pushes? |
| initialisation | the transcripts | cross-seed **magnitude rank** and elevated-set overlap (`analysis_ism_regions.py`) | do the seeds agree which *positions* are extreme? |
| the genes | the seed | the `arm` column — discovery vs confirmation, disjoint genes | does a finding **generalise** to genes it was not found on? |

The first two are the same axis and different comparisons, and they can disagree
without contradiction. Seeds can agree on direction everywhere while disagreeing
about which positions are extreme, because sign agreement is insensitive to
magnitude. The reverse is also possible: agreement about which positions are large,
with the small-magnitude bulk flipping sign at the floor. **Given that roughly half
the capture arm may sit near the noise, expect exactly that pattern in
`vals_capture`.**

**The third axis is shipped and unused.** §9 step 1 assigned every gene to
`discovery` or `confirmation` by a fair draw seeded at 20260801, no gene spans both
arms, and the bank carries `arm` per transcript. Nothing tonight touched it. It is
the only one of the three that answers whether a finding holds on genes it was not
discovered on, which is the question a reviewer asks. Use it before claiming
generality from the other two.

I had this wrong in a message to the interpretability window — I described their
check as varying transcripts. It does not; it varies the seed, as mine does. They
corrected it from their code.

## Anchoring on the reference candidate is a differential exclusion

From the interpretability window, and it is the sharpest trap waiting for tomorrow.
**3,422 of the 4,999 subset transcripts have a reference candidate; the 1,577
without are not evenly spread.** The NMD / no-main-ORF-stop cell — the mechanism
cell this section is about — retains 49.9%, while its matched control, control /
no-main-ORF-stop, retains 93.1%.

So any analysis that filters on `cand_is_ref_cds`, or anchors a positional profile
on the reference start, drops half the mechanism cell and almost none of its
control. A difference between them would then be partly the filter. This is the
same shape as the geometric leaks in §8.5: the control that looks like it holds
something fixed is itself correlated with the comparison.

`qc_ism_banks.py` is **not** exposed — checked, not assumed: it reads only `valid`,
`vals`, `vals_capture`, `vals_decay`, `mass` and `chunk_rows`, and never touches
`cand_is_ref_cds`. Anything built tomorrow needs the same check made explicitly.

## Two worktrees, and the merge that now has to be deliberate

Both windows were committing from one checkout and `git add -A` swept up whatever
the other had in flight. It cost provenance in both directions — several of the
interpretability window's files landed under my commit messages, including the
chunk-invariance probe that stopped five banks shipping at the wrong chunk size,
and half of their anchor fix sits inside my `e7da9b2`. Nothing was lost, but the
history is wrong about who did what.

    ~/claude_projects/NMD_orf_model_v5_4ct          master   this window
    ~/claude_projects/NMD_orf_model_v5_4ct_interp   interp   interpretability

Layout is in `analysis_plans/WORKTREE_LAYOUT.md` (aedf42a).

**The new obligation is merging, in both directions, often:**

    git -C ~/claude_projects/NMD_orf_model_v5_4ct merge interp

Neither window now sees the other's new files until a merge, and today each of us
used the other's code within minutes of it being written. Infrequent merges trade
one failure mode for a worse one.

**Two things worktrees do not fix.** The results directories are shared by symlink,
so two builds writing one shard directory would still collide — the bank build is
unaffected and unprotected in equal measure. And both windows `scp` to the same
cluster directory, which is a second clobbering channel entirely outside git.

Also adopted: **stage explicit paths, never `git add -A`.** That is the fix for the
thing that actually happened.

## Two checks, not a taxonomy

Both windows kept making errors today. Sorted by what would have caught them, not
by what they looked like — and counted rather than impressioned, across both
windows:

**Untraced mechanism — 2 instances.** A plausible cause asserted without opening
the code that would exhibit it. My right-fill diagnosis blamed a transcript-length
clip that is never active. The other window's "capture near the floor ⇒ capture
claims at risk" was stated without checking which quantity those claims are
computed from — they read `p_select` off a forward pass and never touch
`vals_capture`.

> **The check:** grep for the quantity before asserting anything about what
> depends on it.

**Uncharacterized denominator or comparison set — 4 instances, and this is the one
that dominated.** My structural zeros in `vals_capture`. My zeros-based mask, which
fixed that error by introducing another. The other window's reference-anchor
exclusion, 31.5 points differential on the grouping variable. Their circular-shift
null drawn from the whole transcript when the comparison lived inside one window.
The dense-array padding, where 78% of every array is not a position at all.

> **The check:** before reading any proportion or any null, enumerate what is in
> the denominator **and** what is in the comparison set. Both halves. Most of
> today's errors were the second half.

Deliberately not a framework. A third pattern showed up today — a control coarser
than the confound it is controlling — that fits neither cleanly, and three named
categories would not survive contact with tomorrow. Two checks that have each
caught something real are worth more than a taxonomy that hasn't.

**One process note that generalises past this project.** A wrong framing sent
between windows reaches artifacts faster than a correction travels: the other
window's overbroad claim about §5's mechanism arm was in this handoff before they
retracted it. The mitigation is not slower messaging. It is that a claim sent
sideways carries the same provenance mark as one written down — traced, or
explicitly not traced.

## Traps that still apply

1. **The subset is stratified, not random.** 4,999 transcripts, weights summing to
   41,765 exactly. Any population-level estimate must be reweighted or it describes
   this subset. The mechanism cell is 437 transcripts in 41,765; a random 5,000
   would have drawn about fifty.
2. **A design effect is a property of the statistic, not the dataset.** Tonight
   produced 2.88 on the §8.5 pair statistic against 3.15 exploratory. Measure it
   for each statistic; never carry one.
3. **A zero is a claim that needs its own evidence.** The local bank reports a
   batch-shape offset of exactly 0.000e+00 — that is single-threaded execution
   making reduction order independent of batch shape, not the pipeline being
   exact. The same build on the GPU reports 2.31e-07 and 4.93e-07.
4. **`chunk_offset` is not a safe normalizer for the floor.** Measured elsewhere at
   1.5× the shipped value on a K=1 transcript and 20× on a K=10 one. The banks now
   record `chunk_rows` per transcript so a mixed-shape build is visible.
5. **Profile where the code runs.** On a laptop the encoders are on the CPU and
   decode looks like 13–16%; on a V100 decode is 88%.

## §8.5, parked

Written, amended on review, dry-run on validation, and **not read on test**. The
test split is unspent and the design is frozen, so the pre-registration survives.

Under the amended bar it fails on validation — capture beats chance (0.5799, CI
excludes 0.5) but not `kozak_score` (0.5630, CI [0.4431, 0.6756] includes 0.5) —
and the direction split's upstream arm now includes 0.5, so the monotone-positional
alternative is not excluded. Both amendments changed the conclusion, in the
direction of claiming less.

Two questions to settle **before** any future test read:

- **The Kozak arm is underpowered by construction**, computed only on pairs where
  capture and the matrix disagree — 270 on validation, perhaps 650 on test. A wide
  interval failing to exclude 0.5 is absence of evidence, not equivalence.
  "Inconclusive" should be pre-committed as a legitimate outcome.
- **The ATG window carries exon junction marks (channel 4).** Two candidates 50
  bases apart hold different junction patterns, so a residual preference may be
  annotation rather than nucleotides. §8.5 controls window fill and position; it
  does not separate sequence from junction placement.

Recorded because it bears on the pre-registration: the ISM subset spans every split
including test, per the standing convention that only AUC and AUPRC are test-only.
So a later decision to run or skip §8.5 will have been made with test-derived model
behaviour in view. That does not touch the frozen design, but if §8.5 is ever
reported, the ordering is reported with it.

## Also settled tonight

- **The predictor rerun finished** — all five seeds, not three outstanding. Val
  AUC 0.9494, 0.9494, 0.9482, 0.9464, 0.9488; mean 0.9484.
- **The start window is the same for both heads.** `enc_init` (capture) and
  `enc_atg` (the decay branch's start arm) receive the identical (9, 1000) ATG
  tensor — 900 upstream, up to 100 into the ORF. Separate weights, same input.
  Only `enc_stop` sees a different window (500/500, anchored at `orf_end − 1`).
  "up to 100" matters: fill stops at the ORF midpoint, so short candidates get
  fewer, and that clip is the geometric leak §8.5 exists to control.

## Practical

- Local python: `~/miniforge3/envs/nmd_model_local/bin/python`.
- Explorer: `p.castaldi@explorer.northeastern.edu`, `~/cc/nmd_orf_model_v5_4ct`,
  conda `nmd_model`. **Ask before every login.** GPU partition, 8-hour limit, four
  concurrent jobs per user.
- The cluster clone does **not** track this repo's commits — scripts are copied
  onto it, so its `git rev-parse HEAD` is not the provenance of what ran. The
  SLURM scripts print sha256 of the code they execute; trust that.
- The other window is "NMD Harold — Interpretability",
  `local_4c14cee7-60c5-4ecf-ba47-30edebf26c38`. Larry (figures) is
  `local_beccba7a-741b-436d-8db2-4c29e52b4d02`. Batch messages; send one when
  something changes what they would do. Check what they send rather than accept it
  — tonight they corrected me and I corrected nothing of theirs, but they also
  retracted one of their own findings unprompted.

## Bank QC — what the five banks actually say

All five built, job 8885690, chunk 4,096, **all on Tesla V100-SXM2-32GB** across
five nodes (d1002, d1007, d1011, d1017, d1002). Hardware is therefore not a
confound in the cross-seed numbers; disagreement is initialisation. The residual:
the bank script logs GPU model but not driver version per task, so a driver
difference between nodes is not excluded — it is not a plausible cause of the
effect sizes below.

**Completeness, clean on every bank:** 4,999 / 4,999 transcripts, 11,062,149 valid
positions, zero transcripts with no finite response, `chunk_rows` constant at
[4096].

    bank    floor       vals median   >floor   decay median   >floor
    s100   1.664e-06    1.326e-03     82.1%    1.018e-03      81.7%
    s200   1.873e-06    1.488e-03     77.3%    9.651e-04      77.0%
    s300   1.384e-06    1.205e-03     84.7%    8.747e-04      84.5%

The floor is 1.4–1.9e-06, roughly double what a three-transcript build suggested.

### The capture arm is readable. My first measurement of it was not.

I reported 36–46% of `vals_capture` clearing the floor and was about to call the
arm mostly floor. **That was the denominator, not the data.** A substitution in a
stop window changes `e_stop → z_d` only and cannot touch `z_p`, so `vals_capture`
is exactly zero there **by construction**. On 300 transcripts of s100:

    covered by >=1 ATG window           66.2% of valid
    vals_capture exactly zero           35.9% of valid
      ...of those, inside an ATG window  5.7%
    exact zeros outside any ATG window  33.8% of valid

The last two coincide because they are the same set. Conditioned on ATG coverage,
clearance is near 64%, and on a small local bank the conditioned median is 1.015e-05
against 1.316e-06 unconditioned — an order of magnitude from the denominator alone.
`qc_ism_banks.py` (df5cffa) now prints both and raises only on the conditioned one.

**This is the failure this handoff warned about, made by the handoff's author, four
hours after writing the warning.** The bank was fine. The harness was not.

### What the floor does and does not threaten

Corrected by the interpretability window after Larry caught it, and traced to the
code rather than argued:

- **threatened** — D3, novel motifs by ISM on the capture branch, and any
  sequence-level *explanation* of capture's preferences.
- **not threatened** — E4 and E5, which read `parts["p_select"]` off a
  `return_parts=True` forward pass (`analysis2_selection_mass_full.py:126`,
  `verify_atf4_capture_selection.py:68-69`); grepping both for `vals_capture`
  returns zero. Likewise C1–C3 and §8.5, which are AUCs and win rates over
  `p_capture` — also forward-pass.

`vals_capture` is a *sensitivity*, ~1.3e-06: how much one substituted base moves
the logit through the initiation branch. `p_select` is a *probability*, 0.1–0.6.
Six orders of magnitude apart and answering different questions. **An unreadable
sensitivity says nothing about the behaviour it fails to explain.** The diversion
result is not at risk from these numbers and the QC must not be read that way.

### Cross-seed, and the prediction I recorded was wrong

    vals   33,186,447 entries   all 3 seeds same sign 44.1%   r(seed1, mean rest) 0.558
    cap    33,186,447 entries   all 3 seeds same sign 24.7%   r 0.549

Chance unanimity on three seeds is 25%. So `vals` at 44.1% is **above chance but
not high** — I predicted high — and the `cap` figure is mostly my structural-zero
bug, since `np.sign(0)` is 0 and such entries can never be unanimous.

The interpretability window's prediction is the one that held: near-identical
correlations (0.558, 0.549) beside very different sign agreement is exactly
"agreement on which positions are large, with the small-magnitude bulk flipping
sign at the floor."

### The run-length result splits into two findings, and they can come apart

From the interpretability window, and it is the most important thing to carry into
tomorrow. The one positive that survived today is that elevated positions form runs
of four or more bases, 34 times against 0 for random placement. **At a cross-seed
track correlation of 0.558, the specific positions may not replicate even though
the run structure does.**

These are separable and must be reported separately:

- **run-length distribution, per seed** — does each seed independently concentrate
  sensitivity into short blocks?
- **elevated-set overlap, between seeds** — do they agree on *which* blocks?

The first can hold while the second fails, and that outcome is not a weaker version
of the finding — it is a different finding: *"the model concentrates sensitivity
into short blocks"* would be a property of the architecture, while *"these
particular blocks"* would be a property of an initialisation. Only the first is
claimable from five seeds; the second would need the discovery/confirmation arm.

Reporting a single "the runs replicate" number would conflate them.

### Two masks that are not interchangeable

A dead perturbation **inside** an ATG window is a real measurement of zero. A
stop-only position is not a measurement at all. Masking on "where `vals_capture` is
nonzero" deletes both; masking on ATG geometry deletes only the second. The
dead-perturbation rate is about 21% on this model, so the choice is not cosmetic —
and I had this wrong in the first correction to `qc_ism_banks.py`, using the
zeros-based mask for cross-seed agreement before the interpretability window
pointed out the distinction.

The geometric mask also buys a check that can fail, now asserted: every bank must
agree on ATG coverage, because geometry does not depend on the seed. Disagreement
would mean the banks were built over different candidate sets and nothing
comparing them is valid.

**Open for the morning, and the first thing worth doing:** `vals` sign agreement at
44.1% is lower than a quantity intended for interpretation should be, and unlike
the capture figure it is **not** explained by structural zeros. Condition it on
entries clearing the floor before concluding anything from it. If the agreement is
carried entirely by the above-floor minority, that is a usable result; if it stays
near chance there too, positional claims from `vals` need the discovery/confirmation
arm before they mean anything.
