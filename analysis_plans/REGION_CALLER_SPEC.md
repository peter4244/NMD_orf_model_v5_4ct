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
original rank order. This preserves:

- the **power spectrum**, hence the autocorrelation, exactly by construction
- the **marginal distribution**, exactly, via the rank rescaling — which matters because the track is
  heavy-tailed and a plain Fourier surrogate would Gaussianize it

and destroys the *location* of any localized feature.

Random placement of marks — the null used previously — preserves neither, which is why it reported
the data beating a null that had no structure at all.

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
  the stop — as a description, not a test
- the per-transcript variation in call count, since a caller that puts every region in a handful of
  transcripts is describing those transcripts

## 7. Explicitly out of scope

No sequence content. No motif claim. No adjustment for selection mass, coverage, or composition. No
branch decomposition. Those belong to *why a region is important*, which is the next question and
not this one.

## 8. Deliverable

`analysis_plans/analysis_region_caller.py`, one job over five banks, with a runlog committed
alongside. Hypothesis row recorded here rather than at run time; the model window reads this
specification before submission rather than the results after.
