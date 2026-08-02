# Specification — identifying regions the model treats as important

Written 2026-08-02 by the interpretability window, before implementation. Pete's framing:

> First we need reliable tools to identify regions that the whole model thinks are important. Once
> we have identified those regions, analyses of *why* the model thinks they are important come after.

**Scope.** This document specifies a region caller and its calibration. It makes **no claim about
sequence, motifs, routing, or mechanism**, and it deliberately does not adjust for anything. If the
model treats a region as important, the caller should find it — whether that importance comes from
sequence, from ORF structure, or from how much translational traffic reaches it. *Why* is a separate
question and is downstream of this one.

---

## 1. What this problem actually is

**Peak calling on an autocorrelated signal track.** Naming it that makes our difficulties ordinary
rather than novel, and imports a standard solution for the part we got wrong.

Three measured properties of the track constrain the design:

| property | measured | consequence |
|---|---|---|
| **smooth** | autocorrelation 0.90 at lag 1, 0.64 at lag 80, no characteristic scale | a null that destroys autocorrelation is invalid. Random placement of marks is **not** a null for this track — that error is what retracted the run-length result |
| **per-transcript scale varies enormously** | per-transcript median importance spans several orders of magnitude; 2.8% of transcripts have a median below 1e-6 | all thresholds are **per-transcript quantiles**. No absolute cutoff, and no fold-over-median, whose denominator degenerates |
| **heavy-tailed** | p99 / median ≈ 30 within a transcript | a Gaussian surrogate is not an adequate null; the surrogate must preserve the marginal distribution |

---

## 2. Input

- **Track:** `vals` — the **whole-model** transcript-level importance. Not `vals_decay`, not
  `vals_capture`. The question is what the model treats as important, and the branch decomposition
  is part of the *why*.
- **Per-position score:** `max_b |vals[p, b]|`, the largest effect over the three substitutions.
  Stated because it is a choice: max-over-three against max-over-one differs by a measured 1.37×,
  and `mean_abs` is excluded — this project measured it unable to discriminate when many weak
  contributors are present.
- **Positions:** valid positions only, i.e. those holding one of ACGT with a measured substitution.
- **Series:** each maximal contiguous run of valid positions of length ≥ 200, handled separately.
  Gaps are not interpolated across.

## 3. What a region is

A **region** is a maximal run of positions at or above a per-transcript quantile threshold, after
merging and width filtering:

```
  q        threshold quantile, per transcript          sweep 0.90, 0.95, 0.98, 0.99
  gap      merge runs separated by <= gap positions    sweep 0, 2, 5
  wmin     discard regions narrower than wmin          sweep 3, 5, 8
```

**All three are swept and all results are reported across the sweep.** A region set that exists only
at one parameter combination is that combination, not a finding — the same rule that caught the
elevation-threshold failure.

## 4. The null: phase-randomized surrogates

**This is the part the previous work got wrong and it is the reason for the whole document.**

The null must answer: *are these regions distinguishable from what this track would produce anyway,
given how smooth it is?* So it has to preserve the autocorrelation exactly and destroy only **where**
the features are.

**Use iAAFT surrogates** — iterated amplitude-adjusted Fourier transform. Take the FFT of the track,
randomize the phases while preserving conjugate symmetry, invert, then iteratively rescale to the
original rank order. This destroys the *location* of any localized feature, which is what we want.

Random placement of marks — the null used previously — preserves neither the spectrum nor the
marginal, which is why it reported the data beating a null that had no structure at all.

### 4.1 ⇒ Two corrections to the above, 2026-08-02, incoming interpretability window

This section previously claimed iAAFT preserves the power spectrum "exactly by construction" **and**
the marginal distribution "exactly, via the rank rescaling." Both were flagged by the author as
chosen-but-unvalidated. The first claim is wrong and the second is incomplete.

**Correction 1 — it cannot preserve both exactly, and which one it sacrifices decides the direction
of the bias.** iAAFT alternates two steps: replace the Fourier amplitudes with the target amplitudes
(spectrum exact, marginal drifts), then rank-remap onto the original values (marginal exact, spectrum
drifts). It **iterates** precisely because the two are not simultaneously satisfiable, and it
terminates on one of them. Terminating on the amplitude step gives an exact spectrum and an
approximate marginal; terminating on the rank step gives the reverse. The residual does not converge
to zero and concentrates in the tails.

That matters here more than it usually would, because **this caller thresholds at per-transcript
quantiles 0.90–0.99** — the tail is not incidental to the statistic, it *is* the statistic. If the
marginal is the approximate one, the surrogate is least faithful exactly at the operating quantile,
and criterion 1 acquires a bias whose sign is set by an implementation detail this document did not
name.

> **Required:** pre-register the termination step, and report, per transcript and across the sweep,
> the surrogate's value at the operating quantile against the real track's. If the marginal is exact
> by construction the check is free and passes; if it is not, criterion 1 is uninterpretable and no
> amount of surrogate count fixes it.

**Correction 2, and it is the load-bearing one — the null repairs the retracted error on one axis and
repeats it on another.** The run-length result died because random placement destroyed autocorrelation
the track has architecturally. iAAFT preserves autocorrelation. But autocorrelation is not the only
architectural structure in this track: the other is **non-stationarity**. The selection-mass envelope
varies systematically along the transcript — log-mass correlates **0.934** with the effect track
(`SEQUENCE_ENRICHMENT_APPROACH.md` §7.1), and 5′UTR, ORF and 3′UTR carry different mass regimes, as do
window fill edges.

Fourier phase randomization is a null for a **stationary** process. It redistributes local variance
uniformly along the series. So surrogates place peaks where the real track structurally cannot have
them, and concentrate less than the real track does — and **criterion 1 passes because the track has a
mass envelope, not because it has localized features.** Same error, new costume.

This is not hindsight: `ANALYSIS_SEQUENCING_PROPOSAL.md` predicts exactly this shape one section
earlier, for a different instrument — "the *between*-candidate correlation will be **high and largely
meaningless**, because both quantities carry the selection-mass envelope." The prediction was written
for the SmoothHess validation and not applied to the null sitting beside it. Its own conclusion
applies: *it is not four mistakes; it is one, in four costumes.*

> **Required, cheapest first:** (i) surrogate the **residual** after dividing by a smoothed mass
> envelope, then re-multiply — this preserves the envelope and randomizes what is left; (ii) failing
> that, iAAFT within locale blocks; (iii) failing that, condition on mass explicitly, which is what A2
> does and would at least make the two analyses consistent with each other.
>
> **And a distinction this document does not currently draw:** §7 excludes adjustment for selection
> mass, which is right for *calling* regions — the caller should find what the model treats as
> important whatever the source — and wrong for the *null*, which is the one place the architecture
> must be preserved rather than ignored.

**Correction 3 — the series floor is too short for a Fourier method on this track.** §2 admits any
maximal run of valid positions of length ≥ 200. Autocorrelation is **0.64 at lag 80**, so a 200-point
series holds roughly 2.5 correlation lengths and phase randomization has almost nothing to randomize;
the FFT's periodicity assumption also puts a wrap-around discontinuity between the last and first
position, which the caller reads as region structure. **Raise the floor to several multiples of the
correlation length (≥ 800), and report the excluded series by count and locale** — exclusion by length
is exclusion by transcript class, and this project has already been bitten once by a differential
exclusion (31.5% dropped, 49.9% retention in the mechanism cell against 93.1% in its control).

**Correction 4, minor.** 20 surrogates puts the smallest attainable per-transcript empirical p at
1/21 ≈ 0.048, so no transcript can clear 0.05 under a strict inequality. Adequate for the aggregate,
inadequate for any per-transcript statement — and §5 aggregates by gene-clustered bootstrap, so state
that no per-transcript claim is available at this surrogate count.

**Per transcript, generate 20 surrogates.** Call regions on each with identical parameters. Compare:

- number of regions per kilobase
- region width distribution
- region height, as mean score within the region relative to the transcript median

## 5. Calibration and the success criterion, pre-registered

The caller is **reliable** only if all three hold, across the parameter sweep:

1. **Excess over surrogates.** The real track yields more regions per kilobase than its own iAAFT
   surrogates, with the empirical p-value per transcript computed from the 20 surrogates and
   aggregated by gene-clustered bootstrap.
2. **Cross-seed reproducibility of *regions*.** Call independently on all five seeds. A region in one
   seed counts as reproduced if it overlaps a region in another by ≥ 50% of the shorter one. Compare
   against a **circular-shift null** — shift one seed's calls within the transcript and recount —
   which controls for the number and width of calls. Position-level agreement is already known to be
   weak (Jaccard 0.125); this asks the question at the level the caller operates.
3. **Stability across the sweep.** Region sets at adjacent parameter settings overlap substantially.
   A caller whose output reorganizes under a small parameter change is describing the parameters.

**If (1) fails**, the track has no localized structure beyond its own smoothness, and region calling
on it is not meaningful. That is a real result and it stops this line of work.

**If (1) passes and (2) fails**, regions are a property of single initializations and cannot be
carried into any claim about the model.

## 6. Reported regardless of outcome

- regions per kilobase, real against surrogate, per parameter setting
- width and height distributions, real against surrogate
- cross-seed reproduction rate against the circular-shift null
- **the fraction of called regions falling in each locale** — 5′ of the start, within the ORF, 3′ of
  the stop — as a description, not a test. **Report the locale composition alongside it.** The
  windows are UTR-dominated, so "3′ of the stop" is for most transcripts a statement about 3′UTR
  sequence, and the locale fraction will be read as a positional finding unless the composition is
  on the same table (model window, 2026-08-02). The same ambiguity is on toolbox axis 4: locale and
  composition decouple exactly at PTC transcripts, which is where the biology is
- the per-transcript variation in call count, since a caller that puts every region in a handful of
  transcripts is describing those transcripts

## 7. Explicitly out of scope

No sequence content. No motif claim. No adjustment for selection mass, coverage, or composition. No
branch decomposition. Those belong to *why a region is important*, which is the next question and
not this one.

## 8. The row

*Added 2026-08-02 against the thirteen-field template in `ANALYSIS_SEQUENCING_PROPOSAL.md`. The
specification above predates the template; writing the row surfaced one thing the prose had run
together, marked ⇒.*

> **1 · Hypothesis.** The whole-model importance track contains localized regions distinguishable
> from what the track's own smoothness produces anyway.
>
> **2 · Selection rule.** Maximal runs of positions at or above a **per-transcript** quantile, merged
> across gaps ≤ `gap`, discarded below width `wmin`. Per-transcript because per-transcript median
> importance spans orders of magnitude; no absolute cutoff and no fold-over-median, whose denominator
> degenerates on the 2.8% of transcripts with a median below 1e-6.
>
> **3 · Background.** Each transcript's own iAAFT surrogates. Not another transcript, not a global
> threshold, not a count-matched placement — placement is invalid on this track and that is what §4
> exists to say.
>
> **4 · Held fixed.** Nothing, deliberately. §7 is the scope statement and it is a real choice: if the
> model treats a region as important the caller should find it, whether that importance comes from
> sequence, ORF structure or traffic.
>
> **5 · Deliberately not held.** Selection mass, coverage, composition, branch decomposition — all of
> them are *why* a region is important, which is the next question and not this one.
> ⇒ **But "not adjusted for" and "not preserved in the null" are two different decisions, and this
> document previously ran them together.** §7 correctly excludes adjusting for mass when *calling*
> regions; §4.1 correction 2 requires the mass envelope to be preserved in the *null*. Leaving it out
> of the null is not neutrality — it is a null that a transcript's mass envelope alone can beat.
>
> **6 · Null.** iAAFT, **with the four corrections in §4.1, and its own validation is a prerequisite
> to reading criterion 1 at all.** B1's null is the same object, so B1 inherits this prerequisite.
>
> **7 · Reference points.** Per transcript, from that transcript's own 20 surrogates: regions per
> kilobase, width distribution, height. Floor on the empirical p is 1/21 ≈ 0.048 — no per-transcript
> claim is available at this surrogate count, only the gene-clustered aggregate.
>
> **8 · Aggregation.** Per transcript, then gene-clustered bootstrap.
>
> **9 · Sweep.** `q` ∈ {0.90, 0.95, 0.98, 0.99}, `gap` ∈ {0, 2, 5}, `wmin` ∈ {3, 5, 8}. All three
> swept, all reported. A region set existing at one combination is that combination.
>
> **10 · Decision rule.** The three pre-registered criteria in §5, with both failure branches already
> stated there: (1) fails → the track has no localized structure beyond its own smoothness and this
> line of work stops, which is a real result; (1) passes and (2) fails → regions are a property of
> single initializations and cannot enter any claim about the model. ⇒ **(3) had no stated failure
> branch:** if region sets reorganize under a small parameter change the caller is describing its
> parameters, and the output is not usable even if (1) and (2) passed.
>
> **11 · Licensed if positive.** "The model treats these regions as important." **Not** why.
> ⇒ **And not "these are decay regions."** The track is `vals` — whole-model, per §2 — so the regions
> are whole-model regions. **A4 is scoped to called regions and every surviving finding is
> `vals_decay`**, so as things stand A4 would ask its additivity question in regions selected by a
> different quantity than the one whose interpretation it exists to bound. Either the caller gains a
> `vals_decay` variant, or A4's licensing is explicitly narrowed to whole-model regions. Recorded here
> because it is a dependency between two documents and belongs where the track is defined.
>
> **12 · Owner.** Interpretability window. The model window reads this specification before
> submission rather than the results after. Second implementation only if criterion 1 passes, since a
> failure stops the line of work and nothing is downstream of it.
>
> **13 · Enumeration.** n transcripts, n series, **n series excluded by the length floor and their
> locale composition**, n regions per parameter setting, surrogate count, seeds, and the mask
> expression. The locale fraction is reported only with the locale *composition* beside it — the
> windows are UTR-dominated, so "3′ of the stop" is for most transcripts a statement about 3′UTR
> sequence and will be read as a positional finding otherwise.

## 9. Deliverable

`analysis_plans/analysis_region_caller.py`, one job over five banks, with a runlog committed
alongside. **`interp_`-prefixed on cluster scratch**, per the namespacing rule — two windows once
wrote `autocorr.py` to the same directory and the second silently replaced the first.
