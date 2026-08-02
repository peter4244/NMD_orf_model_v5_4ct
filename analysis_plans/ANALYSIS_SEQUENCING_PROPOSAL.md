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

## Where independent replication sits — and it is not where we have been putting it

The model window's point, 2026-08-02, and it bounds what our two-window arrangement can deliver:

> Replication catches errors the two implementations **do not share**. A common design assumption is
> invisible to it. We both wrote the run-length statistic independently, agreed to four decimals,
> and it was wrong — because we had both reached for random placement without stating the
> hypothesis first.

That is exactly what happened. The fold-rule inversion ratio matched at 0.877 against 0.88, the
normalized GC decomposition at 0.4198 against 0.4187, the arm split near-identically. **All of that
agreement was real and none of it was evidence**, because both implementations shared the design
error rather than the coding error.

**So the order is: hypothesis row → one implementation → replication only for results that survive.**

- A **hypothesis row before the first implementation** is what catches shared design assumptions.
  Nothing else can.
- **Replication after** then means what we want it to mean: two executions of an *agreed
  specification*, so agreement is about execution rather than about a premise neither party stated.
- Replication is expensive and should be **reserved for load-bearing results**. Two implementations
  of an unspecified analysis agree on the wrong thing and cost twice as much doing it.

**The cheapest replication, and it is aimed at the error class that actually got us** (model
window's proposal): whichever window runs an analysis, **the other reads its hypothesis row before
the job is submitted** — not the result after. A row review costs minutes and catches the shared
design assumption; a result review costs a second implementation and cannot.

**Cluster scratch scripts need a namespace too.** We agreed `interp_*` and `model_*` for outputs and
`hi_*` / `md_*` for job names, but not for scripts in the shared working directory. Two windows
wrote `autocorr.py` there and the second silently replaced the first — no conflict, no warning, and
the numbers from the first now have no producer. Same prefixes should apply to scripts.

Under this ordering, Phase A gets rows first and single implementations. Only a result that
survives its own row and is load-bearing for a Phase B or C claim earns a second implementation.

---

## Phase A — repair. Cheap, existing code, and a go/no-go on the rest

**Order: A2 first, then A1 and A3 in parallel, then A4.** Pete asked whether ORF weighting should be
accounted for at all, and the model window independently concluded the same thing from the other
direction: **every surviving claim is downstream of A2**, while A1 can only retire something already
retracted. A4 runs last because it is the only item needing new forward passes, and a negative A2 is
what makes it urgent.

**A correction to how these were first written, and it is the error the toolbox exists to prevent.**
A1's row said *stratification* in one line and *residualized track* in the next. Those are different
operations. Residualizing removes routing from the data and describes a model that weights every ORF
equally, which is neither our model nor ribosomes. **Both A1 and A2 hold mass by stratification** —
within bands of similar routing — never by removal.

Each of these can invalidate a current claim. None needs a new instrument. A1–A3 run on the existing
banks.

### A2 — **THE GATE.** Is the U-rich/keto signature routing or sequence?

> **Hypothesis.** The composition signature at elevated positions holds among positions of equal
> selection mass.
> **Held fixed:** mass and coverage **by stratification**; positional region, stratified.
> **Not held:** composition — it is the measurement.
> **Licensed if positive:** the signature is a property of what the decay head reads, not of where
> the model routes. Everything downstream then means what we have been saying it means.
> **If negative:** our best surviving finding is routing. The composition profile, the cross-seed
> k-mer agreement, and the PWM are all conditional on what elevation selects, so **a negative here
> propagates to all three**.

Cost: one job. **This is the load-bearing measurement of Phase A.**

### A1 — does the clustering survive holding routing fixed? *(secondary, weak instrument)*

> **Hypothesis.** Within bands of similar selection mass, sensitivity clusters into short
> arrangements beyond chance placement.
> **Held fixed:** mass and coverage **by stratification, not residualization**; transcript identity.
> **Not held:** base composition.
> **Licensed if positive:** the *second* combinations row — "among equally-routed positions these
> bases are preferred" — not a motif.
> **If negative:** the run-length line ends, retiring a claim already retracted.

**Labelled a weak instrument deliberately.** Run length was always a proxy for "a motif spans several
bases," and it is weak in both directions: mass-driven smoothness produces it without sequence, and
per A4 a *robust* motif would produce none at all, since no single substitution disturbs it. It is
kept because it is nearly free, not because it is informative.

### A3 — is directionality anything but magnitude?

> **Hypothesis.** At equal effect size, elevated positions are more directional than others.
> **Held fixed:** effect magnitude — the axis without which the claim is circular.
> **Owner:** model window (`probe_directionality_null.py`, written, unrun).
> **Licensed if positive:** directionality returns from §6 to §7. **If negative:** it is deleted,
> not softened.

### A4 — can single-base ISM see a motif at all?

> **Hypothesis.** The model's sequence recognition is additive across positions, so single-base
> substitution measures it fully.
> **Test.** Substitute two positions at once within a window; compare against the sum of the two
> single-base effects.
> **Why it bounds everything else.** Single-base ISM detects a pattern only insofar as it is
> **fragile** to single substitutions. A model recognizing "at least 6 of these 8 match" is
> unaffected by any one substitution and would show ≈0 importance at every position of a real,
> working motif — and conv→ReLU *is* an m-of-n detector when the weights are near-uniform, so this
> is representable here rather than hypothetical.
> **Licensed if additive:** single-base ISM sees everything; the fragility caveat comes out of the
> documents and every negative in the programme strengthens.
> **Licensed if superadditive:** the caveat stands with a number attached, and every negative —
> including this morning's stop-codon nulls — is scoped to "not visible to single-base ISM."

Cost: one job. **It bounds the interpretation of every other result here**, which is why it belongs
in Phase A rather than after it.

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

### B3 — is the pattern position-independent *within* a transcript?

> **Hypothesis.** The same pattern shows elevated importance at multiple distinct offsets in one
> transcript.
> **Why it is not the cross-seed statistic.** Cross-seed agreement on sequence with disagreement on
> position is evidence about the *model's representation*, and it depends on the elevation rule that
> A2 is testing. This is the direct version and does not.
> **Licensed if positive:** the pattern is recognized wherever it occurs rather than at a landmark —
> which is what separates a motif from windowing in a design where the landmarks are given.

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
