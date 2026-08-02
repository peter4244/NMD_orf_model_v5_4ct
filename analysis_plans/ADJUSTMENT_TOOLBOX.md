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
| **5** | **local base composition** — GC, keto/amino, per-base | fixes what letters are present | our GC-preserving operator drove G+C from **0.501 to 0.682** (measured, `HANDOFF_2026-08-01_night_banks.md:770`, "elevated, neutral scoring"). That is a **0.181 shift** in the control against a **0.002 shift** in the thing it was built to test |
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

## What each adjustment licenses you to say

Pete's addition, 2026-08-02: an adjustment is only useful if you can state, in plain English, what
it entitles you to conclude. Two things travel together and both must be written down:

- **every adjustment narrows the claim.** The inference is always "given that X was held fixed,"
  and dropping that qualifier is where a claim inflates
- **every adjustment blinds you to the thing adjusted.** A null result after adjusting for
  composition is not "no sequence preference" — it is "no preference beyond composition"

| held fixed | you may say | you may **not** say |
|---|---|---|
| **nothing** | "these positions move the model's output most" | anything about sequence. At r ≈ 0.79 with selection mass, this is mostly a statement about routing |
| **selection mass** (stratified) | "among positions the model routes to equally, these bases matter more" — the decay head's sequence response with routing controlled | "these positions matter most to the output." High-mass positions matter more whatever their sequence, and you removed that |
| **window coverage** | "among positions seen by the same number of candidate windows…" | anything about a position's total influence |
| **positional region** (stratified) | "within the 3′UTR, elevated positions differ from *other 3′UTR positions* in X" | "3′UTR positions are enriched for X" — that is a between-region claim and you stratified it away |
| **compositional region left free** (the PTC decoupling) | one of three readings, stated: enriched in **both** downstream cells → tracks position; **normal only** → tracks 3′UTR composition; **PTC interval only** → coding sequence in a post-termination position | any pooled statement over the cells. Pooling is what the decoupling makes meaningless |
| **local base composition** | "beyond what the base frequencies alone predict, the *arrangement* matters" — the motif claim | "the model prefers base X." That is exactly what you adjusted away, and it is undetectable by construction |
| **distance to a boundary** | "independent of proximity to a window edge or a landmark…" | anything about the landmark itself |
| **effect magnitude** | "at equal effect size, elevated positions still differ in X" — so X is not signal-to-noise | any elevated-versus-background comparison where elevation *is* magnitude. That claim is circular |
| **substitution arity** | comparisons between arms of equal arity | any comparison of a max-over-3 statistic to a max-over-1 one. That difference alone is 1.37× |
| **autocorrelation preserved in the null** | "beyond the smoothness the track already has, positions cluster" | "positions cluster" from a null that destroyed the smoothness. That says only that the track is smooth |
| **anchor = annotated ORF** | "relative to the annotated ORF…", scoped to transcripts that have one | anything about the 31.5% without annotation, whose exclusion is differential by mechanism cell |
| **anchor = model-selected ORF** | "relative to the ORF the model committed to…" | anything about whether the model selects correctly. That is circular |

### Combinations, which is where the real claims live

| combination | the sentence it licenses |
|---|---|
| **mass stratified + composition held + autocorrelation-preserving null** | "Among positions the model routes to equally, and beyond what base frequencies predict, sensitivity clusters into short arrangements." **This is the motif claim.** Nothing weaker earns the word |
| **mass stratified, composition free** | "Among equally-routed positions, these bases are preferred." A *composition* claim — which is what we currently have, and it is not a motif |
| **region stratified + composition held** | "Within the 3′UTR, beyond 3′UTR composition, arrangement matters." The regionally-honest version of the motif claim |
| **magnitude held + mass stratified** | "At equal effect size and equal routing, elevated positions have property X." The only form in which a property-of-elevated-positions claim is not partly circular |
| **nothing held, per-transcript elevation** | "Within each transcript, the largest effects fall here." True, weak, and almost entirely architecture |

### The two failure sentences to watch for

Both were written this week before being caught:

- **"X is enriched at elevated positions"** with nothing held fixed — mostly says elevated positions
  are high-mass positions.
- **"X survives the control, therefore X is real"** where the control was adjusted for something on
  the causal path — the GC-preserving operator, which was itself GC-biased at **0.682 against 0.501**.

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
