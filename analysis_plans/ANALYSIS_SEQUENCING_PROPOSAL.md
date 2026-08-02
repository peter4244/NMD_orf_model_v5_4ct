# Proposed analysis sequencing

Drafted 2026-08-02 by the interpretability window, from `ADJUSTMENT_TOOLBOX.md` and
`SEQUENCE_ENRICHMENT_APPROACH.md`. For the model window's comment before anything runs.

**The organizing principle: repair before extend.** Writing the licensed-inference table showed
that none of our results sit in the row that would license the word "motif," and that two of them
rest on foundations we have since retracted. Building a richer instrument on top of that would
validate the instrument, not the finding. So the first phase re-tests what we already claim, using
code that already exists, and it is a go/no-go on everything after it.

**Second principle: a validation gate is scoped to the instrument it validates.** SpliceAI
validates seqlet extraction and clustering. It does **not** validate the composition profile, which
needs no background, or the PWM regression, which is a held-out prediction and carries its own
internal check. Treating it as blocking for everything would delay work it has no bearing on.

**Every item carries its hypothesis row before it runs.** Written here rather than at run time.

---

## Phase A — repair. Cheap, existing code, and a go/no-go on the rest

Each of these can invalidate a current claim. None needs a new instrument. All run on the existing
banks.

### A1 — does the clustering survive removal of the architectural component?

> **Hypothesis.** Among positions the model routes to equally, sensitivity clusters into short
> arrangements beyond the track's own smoothness.
> **Held fixed:** selection mass and coverage, by stratification; transcript identity, per-transcript.
> **Not held:** base composition — it is the candidate feature.
> **Null:** placement on the **mass-residualized** track, so the architectural autocorrelation is
> removed from the data rather than from the null.
> **Licensed if positive:** "among equally-routed positions, these bases are preferred" — the
> *second* row of the combinations table. Composition is still free, so this is not yet a motif.
> **If negative:** the clustering result is architecture, and the run-length line of work ends.

Cost: one array job. **This is the single most informative measurement available.**

### A2 — is the U-rich/keto signature routing or sequence?

> **Hypothesis.** The composition signature at elevated positions holds among positions of equal
> selection mass.
> **Held fixed:** mass and coverage by stratification; positional region, stratified.
> **Not held:** composition — it is the measurement.
> **Licensed if positive:** the signature is a property of what the decay head reads, not of where
> the model routes.
> **If negative:** our best surviving finding is routing, and Phase B has nothing to work on.

Cost: one job. The composition profile machinery exists.

### A3 — is directionality anything but magnitude?

> **Hypothesis.** At equal effect size, elevated positions are more directional than others.
> **Held fixed:** effect magnitude — the axis without which the claim is circular.
> **Owner:** model window (`probe_directionality_null.py`, written, unrun).
> **Licensed if positive:** directionality returns from §6 to §7. **If negative:** it is deleted,
> not softened.

---

## Phase B — the claims that could earn the word

Conditional on Phase A. Each requires the top row of the combinations table.

### B1 — composition-held enrichment on the residual track

> **Hypothesis.** Beyond what base frequencies predict, the *arrangement* of bases at
> equally-routed positions carries information.
> **Held fixed:** mass (stratified), composition (held), region (stratified).
> **Null:** autocorrelation-preserving, on the residual track.
> **Licensed if positive:** the motif claim, in full, with all three qualifiers.
> **Instrument:** direct PWM fitting on the residual, extended to several PWMs by deflation, since a
> single PWM assumes one motif and a blend is indistinguishable from a poor fit.

### B2 — the three-cell PTC comparison

> **Hypothesis.** Sequence downstream of a premature stop behaves as coding sequence in a
> post-termination position.
> **Stratified:** the three cells — normal-downstream, PTC-interval, control.
> **Not adjusted:** composition. The position/composition decoupling *is* the exposure.
> **Licensed:** one of three readings, stated, never pooled.
> **Limit:** power. The PTC-interval cell is the small one, and that bounds the conclusion.

**This is the only analysis in the programme that no artificial background can construct**, and it
is the one most specific to our data rather than to sequence models in general. If Phase A is
positive, I would argue B2 before B1 on those grounds.

---

## Phase C — the richer instrument, gated

### C1 — SpliceAI port and GT/AG recovery. **Blocking for C2 only.**

> The only positive control this project has. The stop-codon control is retired: recovering a stop
> codon from a stop-anchored profile is guaranteed by the anchoring and tests indexing, not method.
> **If the pipeline cannot recover GT/AG at adequate scale, nothing it says about our model is
> readable.** Run at a scale that clears the seqlet floor by an order of magnitude, since the
> original demotion was 56 seqlets against a floor of 100 and that was a scale failure.

### C2 — seqlet extraction and clustering

> Both criteria — signed and unsigned, overlapping at Jaccard 0.52. Per-cluster composition null
> with a **permutation at matched n**, since KL is positively biased at small n and small clusters
> are exactly what a "we found five patterns" claim rests on. Parameter sweep, and the **pairwise
> similarity distribution reported** — multimodal for real clusters, smooth for a manifold.
> **Prior:** our own effect track is a manifold (autocorrelation 0.90 → 0.64 with no scale), so a
> smooth answer is the expected one and should be reportable as a result.

---

## The stopping rule, proposed rather than assumed

The method document leaves "what counts as enough to stop" open. Proposed:

- **If A1 and A2 are both negative**, the honest conclusion is that the decay branch's sequence
  contribution is not separable from routing at this resolution. Write that, and stop. It is a real
  result — it says the model's sequence sensitivity is a readout of its selection distribution — and
  it is more useful than a fourth instrument.
- **If A1 or A2 is positive but B1 fails**, the claim is compositional, not a motif. Write it as
  composition. Do not proceed to C.
- **Proceed to C only if B1 or B2 is positive**, since C is the expensive branch and its output is
  uninterpretable without C1.

---

## Costs, roughly

| phase | jobs | wall | needs new code |
|---|---|---|---|
| A1, A2 | 2 arrays | ~2 h | residualization step only |
| A3 | 1 | <1 h | none, written |
| B1 | 1 array | ~2 h | multi-PWM deflation |
| B2 | 1 array | ~2 h | three-cell stratification |
| C1 | — | ~1 day | the port |
| C2 | several | days | extraction, clustering, nulls |

**Phase A is under three hours and decides whether Phases B and C happen at all.** That is the
argument for the ordering.
