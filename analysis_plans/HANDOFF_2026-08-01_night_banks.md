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

## Bank QC

See `qc_ism_banks_runlog.txt`, produced by `qc_ism_banks.py`.
