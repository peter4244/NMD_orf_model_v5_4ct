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

## The dead-perturbation rate is not hardware-dependent — tested, not argued

The interpretability window's pilot found **0** exact zeros inside ATG windows; my
300-transcript random draw off the GPU banks found **5.7%** of zeros inside. Their
leading hypothesis was CPU against GPU: an exact zero in `vals_capture` needs the
encoder output bitwise unchanged, so accumulation order could plausibly change the
rate. If true, a liveness gate specified per (isoform, position, operator) would not
be reproducible off this cluster.

**Controlled test, same three transcripts, same seed, CPU build against GPU build:**

    transcript                    exact zeros CPU/GPU   inside-ATG zeros CPU/GPU
    ENSG00000000457.15.novel6         501 / 501                 0 / 0
    ENSG00000000457.15.novel8         501 / 501                 0 / 0
    ENSG00000001497.19.novel1         592 / 592                 0 / 0

Identical. `base_logit` agrees to 1.2e-07 across machines, confirming the same
weights and that inter-machine float noise is far too small to flip a dead
perturbation. **The hypothesis is refuted and the gate is reproducible.**

The variable is the **transcript**, not the machine: 5.7% on a random draw, 0 on a
deliberately short draw, 0 on these three. Which transcripts carry inside-ATG dead
perturbations, and why, is open — but it is a property of the data, and any gate
built on it must be characterised across transcripts rather than assumed uniform.

## A liveness gate would be a third differential filter — measured, not predicted

The interpretability window predicted the inside-ATG dead-perturbation rate rises
with upstream fill extent, and that if so, a liveness gate would be the anchor
exclusion one level down. Tested on `bank_interp_s100`, 400 transcripts, positions
covered by **exactly one** ATG window so the covariate is well defined:

    upstream fill extent          n        dead rate
    100-200                     450          0.00%
    200-400                   2,293          0.00%
    400-600                   3,552          0.51%
    600-900                   4,901          0.00%
    900 (saturated)          26,996         10.78%
    point-biserial r = +0.1457

**RESOLVED — the driver is selection mass, and the interpretation flips.** Tested
the interpretability window's third hypothesis on the same positions:

    dead rate by SELECTION MASS
      p_select < 1e-8            n  9,044       32.38%
      p_select >= 1e-8   every remaining stratum, n 29,148      0.00%

    fill extent, HOLDING mass
      p_select < 1e-8      saturated 34.74%   truncated 2.70%
      every other stratum  saturated  0.00%   truncated 0.00%

    r(extent, dead)          = +0.1457
    r(log10 p_select, dead)  = -0.7294

Every dead perturbation is in the near-zero-mass stratum, and it is a **threshold,
not a gradient** — 0.00% in all five strata above 1e-8. Fill extent's association
disappears once mass is held, except *within* the dead-mass stratum where it still
modulates 34.7% against 2.7%, so extent is a partial proxy rather than a pure one.
Saturated upstream extent means `orf_start >= 901`, so many AUGs precede the
candidate and its stick-breaking mass is a product over a long prefix.

**FULLY RESOLVED: the deaths are float64 underflow in the aggregation.** The
interpretability window argued the zeros could not be the mass annihilating a real
change — float64 resolves a `Δz_p` of 1e-3 at mass 2e-8 by ~90,000 ulps — and so
must be `enc_init` returning a bitwise-identical embedding. That is true **above a
boundary they did not state**: unresolvable needs `mass × Δz_p < 2.2e-16`, so with
`Δz_p` ~1e-3 the boundary is mass ~2.2e-13, and stick-breaking mass reaches 1e-15
by slot 50. Splitting the dead stratum by magnitude:

    p_select band            n      dead rate   regime
    [0, 1e-30)              97       100.00%    underflow possible
    [1e-30, 1e-20)       1,022       100.00%    underflow possible
    [1e-20, 1e-16)         590       100.00%    underflow possible
    [1e-16, 1e-13)       1,626        69.31%    underflow possible
    [1e-13, 1e-11)       1,540         5.97%    encoder must be bitwise-equal
    [1e-11, 1e-8)        4,169         0.00%    encoder must be bitwise-equal
    p_select exactly 0      84       100.00%

The rate collapses **exactly at the derived boundary**. So the deaths are
aggregation underflow, and the question does not relocate to `enc_init` except for
the 5.97% residual just above the boundary, where underflow should not reach.

**That also closes the fill-extent puzzle.** Saturated windows mean 3′-proximal
candidates, which means deeper ordinal position, which means lower mass, which
means more underflow. The twelvefold modulation inside the dead stratum was further
mass stratification, not a separate effect. Everything here is arithmetic and none
of it is about the encoder.

**This changes what a liveness gate means.** It is not a geometric artifact
corrupting the measurement. At `p_select` < 1e-8 a dead perturbation is a *true
statement*: nothing done to that window moves the output, because no mass reaches
it. The gate is correctly identifying candidates the model cannot route to.

**The differential exclusion is still real, but it is biological rather than
encoding.** The mechanism cell has long 5′UTRs, therefore more upstream candidates,
therefore more deep low-mass ones. The honest framing is not "the gate is broken"
but *"the mechanism arm contains more unreachable candidates, and any comparison
must say whether it is counting them."* Those two readings imply different fixes,
which is why this was worth testing before the sentence was written.

**The superseded reading, kept because the shape is still unexplained on its own
terms:**

**The direction holds; the shape does not.** This is a step at saturation, not a
gradient. The proposed mechanism — more filled positions competing to be a bin's
pooled maximum — predicts a smooth rise with extent, and there is none: every
truncated bin is at or near zero and the fully-filled bin jumps to 10.78%. What is
special about a *completely* filled upstream window is not explained and is worth
one look before anyone relies on liveness.

**The consequence is confirmed and larger than predicted.** A liveness gate drops
~11% of positions where the upstream window is saturated and ~0% elsewhere.
Saturated upstream means ≥900 bases 5′ of the start, i.e. a long 5′UTR — which is
the NMD / no-main-ORF-stop cell. So gating on liveness silently removes an order of
magnitude more positions from the mechanism arm than from its comparator.

That is the **third** filter today that looks neutral and correlates with the
comparison, after the reference anchor (49.9% vs 93.1% retention) and the
structural zeros. Any liveness-gated comparison between mechanism groups needs this
characterised first.

Caveats: one seed, 400 transcripts, singly-covered positions only.

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
>
> **Sharpened, after a fifth instance:** enumerate the **distribution** of the set,
> not a representative value from it. The fifth error was characterising a stratum
> spanning `p_select` from 1e-30 to 1e-8 by a single value of 2e-8, computing
> float64 resolvability there, and generalising to the whole stratum — which spanned
> the boundary being tested for. A representative value is not a denominator.

**Naming the failure mode did not prevent it.** That fifth instance was committed by
the window that had proposed the category, two hours after we agreed on it, in a
piece of arithmetic done carefully and correctly at one point of a five-decade
range. The lesson is not that the categories are wrong — it is that they do not
work as things to remember. Only running the check works, which is why both are
phrased as an operation on a set rather than as a principle.

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

## The run-length result is motif-scale, and that reframes the whole thing

Pete's point: **RBPs bind RNA through recurring motifs that shift position across
transcripts.** A binding preference is characterised by what it recognises, not by
where it sits.

**What that retires.** Position-level agreement across seeds is the wrong test for
such a mechanism. Two members can agree completely on the pattern while weighting
different instances of it, and a positional Jaccard cannot distinguish that from
disagreement. So the low across-seed overlap (0.125, ~22% of top-1% sets) is **not**
evidence against a shared learned feature. It also retires Q2 as a negative: offsets
measured relative to the start and stop codon can only find *anchored* features, so
"nothing at the start codon" is a question that design cannot answer.

**What it explains, which is the part I had not drawn out.** An RBP motif is
typically 4–8 nucleotides. The run-length finding is that elevated positions form
runs of **4 or more**, mean length 1.2–1.9, with the excess concentrated at exactly
that scale. That is motif scale.

**And it turns the GC control from a control into positive evidence.** If the runs
were an artifact of channel 5's ±25 averaging, they would be *tens of bases long*,
matching the smoothing window. They are 4–6. **Wrong length for the encoding, right
length for a motif** — which is independent of the fact that the clustering also
survives with GC held bitwise constant.

So the results are one story rather than three: the decay branch has learned short
contiguous sequence features, they recur across transcripts, they replicate on genes
the model never saw, and they are not an artifact of the GC encoding.

**A prediction recorded before the k-mer job reports** (8886733, k = 5 and 6,
five banks), so its outcome is interpretable either way:

    k-mer HIGH, positional 0.125  -> a motif the members share and place differently
    both HIGH                     -> the positional Jaccard was underpowered, not negative
    both LOW                      -> see below; NOT cleanly diagnostic

**AMENDED before the job reported.** My first statement of the third branch —
"both low falsifies the motif reading and points at something transcript-specific" —
was too strong, and the interpretability window was right to catch it. A **degenerate**
motif would also give low exact-k-mer agreement while still being a motif: fixed-k
exact counting is a blunt instrument for a variable one, and the method document
already ranks direct PWM fitting above k-mer enrichment for that reason.

So "both low" does not distinguish *no shared motif* from *a shared motif that
exact 5-mers cannot see*, and the next step from there is **PWM fitting, not the
conclusion that the runs are transcript-specific.** The amendment is recorded before
the result for the same reason §8.5's was: the same change afterwards would not be
defensible.

### 2026-08-02: directionality has a null of 0.37, not 0

The two extraction criteria overlap at Jaccard 0.524 (median, quartiles 0.455–0.589),
so a third of each set is criterion-specific and both must be run. Directionality
`|mean_b vals| / max_b |vals|` was then reported as 0.466 at elevated positions
against 0.391 at all positions, on a **ceiling of 0.75** — with `vals[obs] = 0` and
three substitutions agreeing, `mean = 3v/4` against `max = v`. The ceiling
correction was right.

**But the floor was assumed to be 0, and it is not reachable.** Three substitutions
with random signs give

    |sum| = 3  with probability 2/8,   |sum| = 1  with probability 6/8
    E[ |mean| / max ] = (2/8)(3/4) + (6/8)(1/4) = 0.375

so **0.375 is the random-sign null**, and 0 requires perfect opposition, which noise
never produces. Simulated: 0.3750 at equal magnitudes, 0.255–0.263 for heavy-tailed
magnitude distributions.

**The data supplies its own null and it lands on the analytic value.** Measured on
s100, 500 transcripts, 1,036,981 positions:

    |effect| band              n         median      directionality   % of ceiling
    2.2e-16 - 1.4e-05    207,396      7.37e-08          0.370           49.4%
    1.4e-05 - 1.1e-03    207,396      2.33e-04          0.364           48.5%
    1.1e-03 - 5.1e-03    207,396      2.78e-03          0.359           47.9%
    5.1e-03 - 1.4e-02    207,396      8.60e-03          0.375           50.0%
    1.4e-02 - 4.3e-02    155,547      2.25e-02          0.392           52.3%
    4.3e-02 - 9.5e-02     41,480      5.73e-02          0.409           54.5%
    9.5e-02 - 6.7e-01     10,370      1.25e-01          0.427           56.9%

    all positions      0.373    indistinguishable from the null
    sub-floor band     0.370    pure noise (median 7.4e-08 against a floor of 1.7e-06)
    top 1%             0.427    above the null by 14%

**RECONCILED, and my own null was wrong.** The 0.373-against-0.391 gap between the
two windows is **mean versus median**, not data: median of the ratio gives 0.3911
against their reported 0.391. Ruled out first, on one sample: pooling method
(per-transcript 0.3733 against pooled 0.3732), floor restriction (0.3737), column
(`vals` 0.372, `vals_capture` 0.365), dead substitutions (0.3732), mean over three
rather than four (0.4975, far too high).

**And the null is statistic-dependent, which broke my correction of theirs:**

    RANDOM-SIGN NULL          mean      median
      equal magnitudes       0.3749    0.2500
      exponential            0.2585    0.2502
      half-normal            0.2627    0.2503

    MEASURED                  mean      median          n
      all positions          0.3731    0.3913   1,108,267
      sub-floor (noise)      0.3709    0.3888     171,041
      top 1% by |effect|     0.4273    0.4481      11,083

I claimed the sub-floor band "lands on the analytic prediction without being told
to". **That was a coincidence of the mean.** On the median it misses badly — 0.389
measured against 0.250 predicted — so the sub-floor data is **not sign-random
noise**: its three substitutions agree in sign far more than chance, meaning a
shared per-position component dominates at that scale. The analytic random-sign null
is the wrong reference, and I used it to correct the other window while making the
parallel error.

**The right null is the one the data measures — the sub-floor band — and it makes
both statistics agree:**

    against the in-sample sub-floor null      mean      median
      all positions                          +0.6%     +0.6%
      top 1% by |effect|                    +15.2%    +15.3%

**FULLY RECONCILED. Two gaps, both "same name, different set", and the settled
number is 20.5%.**

    set                                median     mean         n
    sub-floor null (measured)          0.3872   0.3700   158,036
    top 1% GLOBAL quantile             0.4469   0.4270    10,370
    top 1% PER-TRANSCRIPT              0.4668   0.4519    10,369

    per-transcript:  +20.5% above null,  21.9% of the range 0.387-0.75
    global:          +15.4% above null,  16.4% of the range
    the two definitions overlap at Jaccard 0.240

Gap one was **mean versus median**. Gap two was **global versus per-transcript top
1%** — and those two definitions share only 24% of their positions, because the
global quantile over-weights transcripts whose effects are large overall.
Per-transcript is what every other elevation analysis here uses, including the
run-length and k-mer work, so it is the right one and the global version was the
outlier.

Independent implementations then agree to four decimals: 0.4668 here against 0.4664
there, 21.9% of range against 22%.

**The settled statement: elevated positions are 20.5% above the measured noise
floor, which is 22% of the available range.** All positions are indistinguishable
from the floor.

**That number was corrected three times in one morning and shrank every time:** 62%
of an attainable ceiling → ~14-15% above a wrong analytic null → 20.5% above the
measured in-sample null. Worth stating plainly wherever it is written, because the
first figure travelled between windows before any of the corrections did.

The conclusion that the branch learned both directional and non-directional features
stands. The size of the directional part is ~15% over noise, and it should be quoted
against the measured sub-floor band rather than against 0 or against an analytic
sign-null.

**The error class, again, and from the other end.** The ceiling was checked and
corrected; the floor was never checked because 0 *looks* like a null. A ratio
statistic has two reference points and both need enumerating. That is the same
lesson as the denominators, arriving through the numerator.

`probe_directionality_null.py`. Note it also caught one of mine: `np.isfinite(x).all(1)`
is never true, because `vals` is NaN at the observed base by construction. It failed
loudly rather than returning an empty comparison, which is the good version.

### 2026-08-02: the seqlet plan, and the four things that decide whether it works

Agreed with the interpretability window. The reason to move off k-mer enrichment is
**not** that clustering is better — it is that a contribution weight matrix is
defined by *what recurs across independent sites* and never computes a ratio against
a reference set. Every problem of the last twelve hours lived in the background.
`vals_decay` is (n, W, 4), which is natively what MoDISco consumes.

**Order of work, and the gate.**

1. **The SpliceAI positive control is BLOCKING.** MoDISco was demoted here on a
   measurement — zero patterns from a model that demonstrably encodes GT/AG, on 56
   seqlets against a floor of 100. That is an underpowered negative. At 110,000
   seqlets the stated cause is gone. **If the pipeline cannot recover GT/AG from
   SpliceAI at scale, nothing it says about our model is worth reading.** It is the
   only control this project has that separates "the method failed" from "the model
   has nothing".
2. **Direct PWM fitting before MoDISco** — no background *and* no clustering, ~40
   parameters against grouping 110,000 seqlets, degeneracy represented natively.
   **Pre-committed: a poor PWM fit does NOT license "no sequence preference."** A
   single PWM assumes one motif; several motifs return a blend, and a poor fit is
   then ambiguous between "nothing" and "more than one". A poor fit licenses going
   to MoDISco, nothing else. Otherwise we replace one underpowered negative with
   another, which is how MoDISco was demoted in the first place.
3. MoDISco as the richer follow-up.

**Four port decisions, each capable of producing a plausible wrong answer.**

- **No reverse complement.** MoDISco revcomp-collapses by default because a TF motif
  is double-stranded. RNA is single-stranded and the model reads one strand;
  revcomp would merge genuinely different motifs. The likeliest naive-port error.
- **The observed base.** `vals` is **NaN at the observed base by construction** — you
  cannot substitute a base for itself — so MoDISco's four-valued input needs a
  convention. Writing `L[b]` for the logit under base `b`, the bank gives
  `vals[b] = L[b] − L[obs]` with `vals[obs] := 0`, so mean-centring is well defined
  and is what hypothetical contributions already are:
  `hyp[b] = vals[b] − mean_b(vals[b])`. The alternative — observed = 0, others
  = −Δ — additionally asserts the observed base contributes nothing, which is false
  exactly when the observed base is the one doing the work. **Run both; predicted in
  advance that the second under-weights the consensus base.**
- **Seqlet width from our data.** The run-length excess concentrates at 4–6 bases, so
  ~7–11 windows are indicated by measurement rather than by ChIP-seq defaults.
- **Per-transcript cap and multiple draws.** One long poly-U tract contributes
  thousands of near-identical seqlets that recur trivially. Stratify the subsample by
  transcript with a cap, take several independent draws, and require the CWM to
  recur **across draws as well as across seeds** — the cross-seed logic applied to
  the sampling axis.

**The composition null, which keeps the no-background property.** A compositional
tendency also recurs, so recurrence alone cannot exclude the keto skew — but
composition produces a **low-information** CWM and a motif a high-information one.
From the measured compositions:

    background   A .256  C .243  G .258  T .243     H = 1.9994 bits
    elevated     A .214  C .206  G .297  T .284     H = 1.9809 bits
    IC of a composition-only column vs background       0.0183 bits

**A CWM column carrying ≤ ~0.016–0.018 bits over background is fully explained by
the keto skew.** Bracketed, not point-quoted: 0.0183 on 600 transcripts here,
0.0159 on 800 in the other window. The difference is sampling, but a single
decimal-quoted figure is how 0.877 and 2.34 both travelled further than they should
have this week.

**Report per-column KL beside every CWM, not a pass/fail.** The bar is worth far
less at the weak end than the strong:

    strong consensus, 90% one base    1.358 bits    75x the bar
    moderate,         60/20/10/10     0.423 bits    24x
    weak,             35/25/25/15     0.055 bits     3x

At 3×, with width 11 across several clusters, multiplicity eats it entirely. The bar
rejects the keto skew and nothing more; a pass/fail would imply a discrimination it
does not have.

**SEQLET CALLING IS A SECOND CONVENTION, and it selects different positions.**
Ours is the top 1% by **max |vals| across substitutions** — unsigned magnitude.
MoDISco calls seqlets on the windowed **actual** contribution `hyp[obs]`, which under
mean-centring is `−mean_b(vals[b])` — signed. A position where one substitution
raises the logit hard and another lowers it hard scores high on ours and near zero on
theirs, because the signed mean cancels. **Highly sensitive but not directionally so
is a seqlet to us and invisible to MoDISco.**

Which is correct depends on what a motif is taken to be — "this base suppresses
decay" (signed) against "this position is load-bearing whichever way it moves"
(unsigned) — and that cannot be settled a priori. **Extract on both and report the
overlap; the overlap fraction is itself informative about which kind of feature the
branch learned.** It also sharpens the SpliceAI control: GT/AG is a *directional*
consensus, so the signed criterion must recover it, and failure there means the port
is wrong on a model where the answer is known.

**Confidence, stated because it was overstated once already:** mean-centring being
correct for ISM contributions is algebra and is verified in both windows. Mean-centring
being what MoDISco *requires* rather than merely tolerates is reasoning from how the
method works, **not** from reading its source, and should be checked there before the
port rather than after.

    keto  (G+T)  elevated .581  background .501   ratio 1.160
    amino (A+C)  elevated .420  background .499   ratio 0.842
    GC    (G+C)  elevated .503  background .501   ratio 1.004

**What this trades, stated plainly:** the background problem for a clustering
problem. Seqlet threshold, similarity metric, cluster granularity and alignment
window are all assumptions capable of manufacturing structure. They are assumptions
about *the signal* rather than about what a fair comparison population is, which is
the better thing to have to defend — but it is a trade, not an escape.

### 2026-08-02: the GC confound was diagnosed by eye and never applied

Measured on 600 transcripts of s100 (`probe_base_composition_bias.py`), base
composition at elevated positions against all valid positions:

    set                              A       C       G       T     G+C
    background (all valid)       0.256   0.243   0.258   0.243   0.501
    elevated, all scoring        0.214   0.206   0.297   0.284   0.503
      ratio to background         0.83    0.85    1.15    1.17
    elevated, neutral scoring    0.159   0.325   0.357   0.159   0.682
      ratio to background         0.62    1.34    1.38    0.65

**Two findings, and the second matters more.**

1. **The GC-preserving control is itself compositionally biased**, by more than
   anything it was testing: it drives GC at elevated positions from 0.501 to
   **0.682**. Under that restriction an A/T position can only be scored by A<->T and
   a C/G position only by C<->G, so if the model responds more strongly to C<->G the
   restriction selects for C/G positions. Every one of the nine top-enriched k-mers
   under neutral scoring has G at the substituted position. **Its k-mer output is an
   artifact of the control and must not be used.**

2. **Under normal scoring the elevated set is not GC-biased at all** — 0.503 against
   a background of 0.501, two parts in a thousand. **So the GC confound never applied
   to the original enrichment.** It was diagnosed by eye, from k-mers that *looked*
   AU-rich, and never measured on the positions themselves. Two jobs were spent
   controlling for it, and the control introduced a bias three times larger than any
   effect it was testing.

**The axis that IS asymmetric is a different one:** G and T enriched (1.15, 1.17),
A and C depleted (0.83, 0.85), with GC flat. That is keto versus amino, orthogonal
to everything we controlled for.

**The methodological lesson, which is the transferable part.** A confound noticed in
the *output* of an analysis must be measured on the *input* before it is controlled
for. The enriched k-mers looked AU-rich, so GC was assumed to be the driver; nobody
checked the base composition of the selected positions, which is one line and
settles it. This is the same class as the four denominator errors of the night —
reasoning about a set without enumerating it — arriving through the door marked
"being careful".

### The k-mer result: the motif branch, with one confound the GC control cannot touch

k = 5, five members, decay (interpretability window, job 8886733):

    positional Jaccard   0.1250  (~22% overlap of top-1% sets)   LOW
    k-mer enrichment     r = 0.7501, range 0.714-0.809           HIGH

    enriched   TTTTT +2.34, TTGTT, TTTAT, TATAT, TTGAT
    depleted   GCACG, ATCCG, GCTCG, TACGC -2.75
    -> U/A-rich up, GC-rich down

**This is the first branch of the prediction recorded in advance**: members agree on
*what the sequence is* and not on *where it sits*, which is what a variable-position
binding preference produces. It is distinguishable from "the positional Jaccard was
underpowered" only because the three outcomes were written down before the job ran.

**Do not call this an AU-rich element yet. Two confounds, and they need different
controls:**

1. **Compositional / GC.** The enriched-depleted axis *is* a GC axis, which is what
   a GC-driven signal looks like. Job 8886938 rescores using only the GC-preserving
   substitution, so channel 5 is bitwise fixed.
2. **Regional composition, uncontrolled and the more serious one.** Elevated
   positions concentrate 3′ of the stop codon; the background is every valid
   position of the same transcripts, 5′UTR and CDS included. **3′UTRs are AU-rich as
   a class**, so a positional bias toward the 3′UTR reproduces this enrichment with
   no motif in it at all. A region-matched background does not exist yet.

**THE TWO CONTROLS ARE NOT SUBSTITUTES, and this is the thing to not get wrong in
the morning.** Job 8886938 answers (1) and does **nothing** for (2). Holding GC
bitwise constant does not change *where* the elevated positions are, so a 3′UTR-biased
set compared against a whole-transcript background still recovers 3′UTR composition
whether or not the substitution moved GC. **A clean GC control must not be read as
resolving the AU question.** Only a region-matched background does that.

Confound 2 is also the same shape as every error caught tonight: a comparison whose
two sides are drawn from different sets.

### The arity-matched GC decomposition

Two implementations, independently and simultaneously. Raw counts are **not**
comparable across them — the `all` arms differ 1.36× — but ratios within an
implementation are:

                                 mine    interp
      all, max of 3             1.000     1.000
      changing, max of 2        0.962         -
      one GC-changing               -     0.732
      GC-PRESERVING             0.419     0.420

**0.4187 against 0.4198.** At matched arity the interpretability window measures
GC-changing substitutions clustering 1.74× more than GC-preserving ones, and **57%
of the arity-matched clustering survives with GC held constant**. That is the number
for §5 — "it survives" understates what GC does, and "GC contributes about half" is
the arity artifact, not a measurement.

## RUN AND ANSWERED: floor conditioning does not rescue cross-seed agreement

`analysis_crossseed_floor.py`, job 8886461, all five banks, exit 0. Full output in
`model_crossseed_floor_runlog.txt`.

    arm            condition                       n    unanimous  x chance      r
    vals           all finite             33,186,447      26.8%      4.3x     0.617
    vals           all seeds clear floor  25,050,638      29.1%      4.6x     0.618
    vals_decay     all finite             33,186,447      27.8%      4.5x     0.608
    vals_decay     all seeds clear floor  24,923,255      30.2%      4.8x     0.608
    vals_capture   all finite             22,332,501      20.4%      3.3x     0.610
    vals_capture   all seeds clear floor  11,380,608      24.3%      3.9x     0.611

Each seed is conditioned on **its own** batch-shape offset (1.258e-06 to 1.873e-06);
an entry is kept only when every seed clears its own, which is the conservative
choice among the three available.

**The hypothesis was that the sub-floor bulk was doing the disagreeing. It is not.**
Removing 8.1 million unreadable entries moves unanimity by 2–3 points and moves the
correlation not at all — r is 0.61–0.62 in every row, conditioned or not. The seeds
disagree where they can resolve perfectly well.

**What follows, and it is a constraint rather than a defeat:**

- **Positional claims from these banks — *which* positions are elevated — need the
  discovery/confirmation arm before they mean anything.** Five seeds at 4.6× chance
  with r = 0.62 is real and modest, and it does not improve on the readable subset.
- It makes the run-length split **more** likely to matter, not less: run *structure*
  may well replicate while run *locations* do not, which is precisely the two-outcome
  design already agreed.
- The ordering across arms is consistent and mild: `vals_decay` (30.2%) > `vals`
  (29.1%) > `vals_capture` (24.3%). The decay branch is the best-behaved on this
  axis as on the other two, which is a third independent reason the run-length
  finding landed on the right arm.

**Not concluded from this:** that the model is unstable. Sign agreement over five
independent initialisations is a demanding statistic, and r = 0.62 across seeds is
a moderate positive. What it forbids is treating a position list from one seed, or
from a pool across seeds, as a finding on its own.

## Who runs what — agreed with the interpretability window

Pete asked for this to be coordinated rather than left to converge. Split by what
each window already has working:

| who | analysis |
|---|---|
| interpretability | run-length: does *structure* replicate across seeds, and separately do *locations* |
| interpretability | run-length across `discovery` vs `confirmation` arms, same seed, disjoint genes — the untouched axis |
| this window | `vals` cross-seed sign agreement conditioned on above-floor entries |
| this window | the 5.97% residual at `p_select` ∈ [1e-13, 1e-11): recompute `z_p` perturbed vs unperturbed on ~92 positions, since it needs the encoder re-run |

**Deliberately duplicated: the run-length replication itself.** It is the
load-bearing positive in §5, and tonight produced six or seven wrong claims between
the two windows with **not one caught by whoever made it**. Independent computation
of elevated-set overlap — their effect tracks and elevation rule against mine from
`vals` — makes agreement worth something and disagreement worth more. Nothing else
is duplicated.

**Mechanics, because worktrees do not protect the shared results directory:**

- output namespace `results_ism_v6/model_*` and `qc_*` for this window,
  `interp_*` for theirs. Two jobs writing one path is the failure git cannot catch.
- job-name prefix `md_*` here, so `squeue` distinguishes them.
- **the four-job cap is per user and shared** — say so before running anything wide
  rather than starving each other.
- access expires ~09:00; long jobs checkpointed or restartable before then.

**Two constraints to design around:** capture magnitude varies 47× across seeds, so
any elevation threshold on that branch must be relative to the seed's own
distribution (their 10×-transcript-median rule already is); and both the
reference-anchor exclusion and the liveness gate are differential on the mechanism
grouping, so anything that filters must report retention per cell, not overall.

**Answered: the run-length positive is on `vals_decay`.** Not `vals`, not
`vals_capture` — confirmed from the runlog, whose first line reads `effect column =
the decay branch`, and it is the only run-length runlog that exists.

**So it is a decay-branch result outright, not "largely" one, and must be described
that way from the start:** *positions where a substitution changes whether the
selected reading frame triggers decay* — **not** "positions the model is sensitive
to." The second phrasing mixes branches the decomposition deliberately separated,
and wording of this kind propagates faster than its correction.

Three things follow, and all of them make the finding **easier** to defend:

- it sits on the arm clearing the floor at **77–89%**, not capture's 53–78%, so the
  readability question does not touch it;
- it sits on the arm whose magnitude varies **2.1×** across seeds (8.747e-04 to
  1.867e-03) rather than capture's 47×, so cross-seed comparison is well
  conditioned and the self-normalising threshold is not doing heavy lifting;
- it is the more interesting arm to have found it on. A localised decay-branch
  sensitivity speaks to what makes a *committed* frame get degraded, which is the
  question §5 asks. The same result on `vals` would have mixed both branches and
  their interaction.

**Consequence for the duplicated check: it must be computed on `vals_decay` too.**
Running the independent replication on `vals` would not be a replication — it would
be a third analysis, and any agreement or disagreement would be uninterpretable.

The interpretability window will run capture separately and report it separately,
never pooled, and has put on record **before looking** that they expect little
there.

## Where the banks live, and where they do not

**The five banks exist ONLY on Explorer**, at
`~/cc/nmd_orf_model_v5_4ct/results_ism_v6/bank_interp_s{100..500}.h5`, 652–703 MB
each, 3.4 GB total. **Decision, 2026-08-01: do not copy them locally without a
specific immediate use.** One authoritative location, so nobody has to work out
which copy is the dataset — a partial local mirror is the artifact that later gets
mistaken for the whole thing, or diverges from it with nothing to say when.

Note the case that already exists and is worth not extending: `results_tensor_v6/`
is present on **both** machines. That is the ambiguity this decision avoids
repeating for the banks.

**The memory picture, corrected — it depends entirely on how the file is read.**
The laptop has 8 GB of RAM, and the bank's `vals` is `(4999, 9833, 4)` float32:

    dense, whole-array reads (`f["vals"][:]`)   0.79 GB per array
      five such arrays                          3.9 GB
      cross-seed, vals + capture over 5 seeds   7.9 GB + a stacked copy  -> impossible locally
    sliced, retaining only real extents         ~0.66 GB for one bank
      per-member effect tracks, five seeds      ~0.4 GB                  -> fits easily

Padding is **77.5%** of the dense array, which is the whole difference. Code that
slices per transcript never materialises the padded form and fits comfortably; code
that slurps does not. `qc_ism_banks.py` slurps by design and is cluster-only;
`analysis_ism_regions.py` slices and is not.

Two size figures were computed tonight over different sets and disagreed: `W` is
**9,833**, the largest *last covered position*, not 18,626, which is the largest
`tx_length` in the subset — a transcript is covered only to its last window, not to
its end. Likewise 11,062,149 valid positions against 15,018,697 as the sum of
`tx_length`. Checked against the bank rather than argued.

So: **every real computation on the banks runs on Explorer.** `qc_ism_banks.py` is
cluster-only by construction — it uses `f["vals"][:]`, which loads 786 MB per array
— and that is correct rather than a defect to fix.

A related regime change for anyone moving from the tensor to a bank: the tensor is
478 MB and comfortable in memory; a bank is 690 MB on disk and 3.9 GB loaded. Code
that was fine against the tensor may not be against a bank. `h5py` slices lazily —
`[:]` is what defeats it.

**Access, as of 2026-08-01:** Explorer approved for 10 hours (to ~09:00 on
2026-08-02) for work analyzing the ISM banks. Not for the §8.5 test read, not for
training or tensor work, and not after it expires. Ask again for those. The standing
rule otherwise remains: ask before every login.

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
[4096], ATG coverage 67.3% identical across seeds as the geometry assertion
requires. Job 8886153, **exit 0, no problems found**; full output in
`qc_ism_banks_runlog.txt`.

                floor      vals med    capture med (ATG-covered)   >floor
      s100   1.664e-06    1.326e-03         2.770e-05              63.1%
      s200   1.873e-06    1.488e-03         6.281e-06              53.4%
      s300   1.384e-06    1.205e-03         8.515e-05              68.9%
      s400   1.869e-06    3.249e-03         2.941e-04              71.6%
      s500   1.258e-06    1.453e-03         1.574e-04              78.3%

      cross-seed   vals  26.8% unanimous sign   r 0.617
                   cap   20.4% unanimous sign   r 0.610

**The capture arm is usable** — 53–78% clears the floor on the honest denominator.
The near-miss conclusion that it was mostly noise came from the wrong denominator
and is retracted.

### START HERE TOMORROW: capture magnitude is 47× seed-dependent

The ATG-covered capture median runs **6.281e-06 (s200) to 2.941e-04 (s400)**, a
factor of **47** across initialisations. Over the same five seeds `vals` spans a
factor of 2.7 and `vals_decay` 2.1.

So the capture arm's *magnitude* is strongly initialisation-dependent in a way the
other two arms are not. It was invisible at three seeds. **Any elevation cut-off on
the capture branch must be defined relative to that seed's own distribution, not in
absolute effect size, and nothing from the capture branch should be pooled across
seeds until this is handled.** An absolute threshold means something different in
s200 than in s400 by a factor of 47.

### Cross-seed at five seeds

Raw unanimity is not comparable to the three-seed numbers — chance falls from 25% to
6.25%. Against chance, `vals` is 4.3× and `cap` is 3.3×. Both real, both modest, and
closer to each other than either window expected.

**My recorded prediction — high for `vals`, low for `vals_capture` — is wrong on
both halves.** They behave similarly and neither is high. At r = 0.62 for `vals` the
caution about the run-length result stands, and splitting it into run *structure*
versus run *locations* is the first thing to test.

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
