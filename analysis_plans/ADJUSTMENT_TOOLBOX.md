# The adjustment toolbox, and the rule for using it

Drafted 2026-08-02 by the interpretability window, for §5 of
`SEQUENCE_ENRICHMENT_APPROACH.md`. Pete's design, 2026-08-02:

> One of the challenges of motif enrichment is the ever-shifting background of what to adjust for
> and not adjust for. Assemble a toolbox of adjustments and stratifications, state specific
> functional hypotheses prior to each analysis, and match the adjustments to the hypothesis
> explicitly.

**Why this is the right shape.** There is no universally correct background. The right adjustment
depends on what is being asked, so the question has to be written down first. Every error of the
last two days was a choice made implicitly inside an implementation — which background, which
denominator, which reference point, which estimator, which selection rule, whether to condition on
mass. None was wrong because someone reasoned badly about it; each was wrong because nobody
reasoned about it at all.

**The rule.** Before any enrichment analysis, write (1) the functional hypothesis in one sentence,
(2) the axes held fixed and why, (3) the axes deliberately *not* held fixed and why, (4) what the
null preserves. An analysis whose hypothesis is not written first cannot have its adjustments
checked against anything.

---

## The axes

Each has been hit for real. The "cost of getting it wrong" column is what actually happened.

| # | axis | what holding it fixed does | cost of getting it wrong |
|---|---|---|---|
| **1** | **selection mass** — `p_select` of the covering candidates | removes the routing component; asks what the *decay head* reads given that a ribosome arrives | not held: the effect track is a readout of mass at r ≈ 0.79 and clusters because mass is smooth. Held *away* rather than stratified: asks a counterfactual the model never computes |
| **2** | **window coverage** — `fill_count` | removes how many candidate windows contain the position | correlates with the track at r ≈ 0.54; distinct from mass and not removed by it |
| **3** | **positional region** — 5′ of start / in ORF / 3′ of stop | fixes locale | not held: elevated positions concentrate 3′ of the stop and any whole-transcript background recovers 3′UTR composition |
| **4** | **compositional region** — coding-like vs UTR-like | fixes composition | **decouples from axis 3 exactly at PTC transcripts.** Not a confound — the decoupling *is* what a PTC transcript is, and conditioning it away removes the exposure |
| **5** | **local base composition** — GC, keto/amino, per-base | fixes what letters are present | our GC-preserving operator drove G+C from 0.502 to **0.679**, so the control was three times more biased than the thing it tested |
| **6** | **distance to a boundary** — fill edge, start, stop, junction, transcript end | fixes edge effects | window fill edges and *natural* boundaries are the same hazard, and a natural boundary **moves** in PTC transcripts, so "distance to the stop" is ambiguous exactly where the biology is |
| **7** | **effect magnitude** | fixes signal-to-noise | any statistic conditioned on being elevated is conditioned on magnitude. Directionality rose with \|effect\| across *all* positions, so "elevated positions are more directional" may be tautological |
| **8** | **transcript identity** — per-transcript vs pooled | fixes which positions compete | global and per-transcript top-1% overlap at Jaccard **0.24**. Global over-weights responsive transcripts; per-transcript takes the same share from each |
| **9** | **substitution arity and GC status** — max-over-3 vs one | fixes how many alternatives the statistic maximises over | max-over-3 against max-over-1 costs 1.37× on its own, which was nearly attributed to GC |
| **10** | **effect-track autocorrelation** | — | the track is autocorrelated at r = 0.90 (lag 1) to 0.64 (lag 80) for architectural reasons. **A null that destroys it tests nothing**, which is what random placement did |
| **11** | **candidate anchor** — annotated vs model-selected | fixes the coordinate origin | anchoring on the reference alone drops 31.5% of transcripts, and **differentially**: 49.9% retention in the mechanism cell against 93.1% in its control |

## Two axes that are not adjustments but decide what a number means

| axis | the failure |
|---|---|
| **estimator** — mean vs median | the same quantity read 0.373 and 0.391; two windows disagreed for an hour |
| **reference points of a ratio** — floor *and* ceiling | the ceiling was assumed 1.0 (it is 0.75) and the floor assumed 0 (measured 0.387). Both must be **measured in-sample**, per statistic — the analytic floor of 0.375 is a mean at equal magnitudes and does not describe this noise |

---

## The rule applied, to the live question

### Hypothesis H1 — *the decay head responds to a local sequence feature, independent of routing*

| | |
|---|---|
| **held fixed** | selection mass and coverage (1, 2) — **by stratification, not removal**, since routing is the architecture; transcript identity (8) — per-transcript, so every transcript contributes the same share |
| **deliberately not held** | local base composition (5) — composition *is* the candidate feature; positional region (3) only within a stated stratum |
| **null must preserve** | the architectural autocorrelation (10). Random placement is invalid here. The test is the analysis re-run on the mass-residualised track, with placement then testing only what remains |
| **status** | the run-length result **does not currently meet this** — its null destroyed axis 10, so it demonstrated that the track is autocorrelated, not that a sequence feature exists |

### Hypothesis H2 — *sequence downstream of a PTC behaves as coding sequence in a post-termination position*

| | |
|---|---|
| **held fixed** | nothing on axis 4 — the position/composition decoupling is the exposure |
| **stratified** | the three cells: normal-downstream, PTC-interval, and their controls |
| **not adjusted** | composition, for the same reason |
| **limit** | power. The PTC-interval cell is the small one, and that bounds the conclusion rather than justifying an adjustment |

### Hypothesis H3 — *elevated positions are more directional than chance*

| | |
|---|---|
| **held fixed** | effect magnitude (7) — **this is the one that has not been done**, and without it the hypothesis is untestable because elevation is defined by magnitude |
| **reference points** | both measured in-sample, per estimator |
| **status** | moved to *not settled* pending the magnitude-matched comparison |

---

## What the toolbox does not do

It does not make any single analysis correct. It makes the choices **visible before they are made**,
which is the only thing that would have caught the eleven errors of the last two days — none of
which threw, and all of which produced a plausible table.
