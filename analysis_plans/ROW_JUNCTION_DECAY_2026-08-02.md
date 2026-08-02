# Row — does the decay head implement the canonical exon-junction rule?

*Interpretability window, 2026-08-02. Written before any code, per the row template
(`ANALYSIS_SEQUENCING_PROPOSAL.md` → `## The row template — thirteen fields, and an empty one blocks
implementation`). Standards cited, not restated: `PRIMARY_DIRECTIVE.md`, `ADJUSTMENT_TOOLBOX.md`.*

**Scope: this is a MODEL claim.** Not biology, not instrument. It is a statement about what this
trained network's decay head does with junction position.

**One arm of a two-arm question.** Pete's hypothesis is that the model may choose ORFs by whether
they encode a premature stop rather than by whether a ribosome would initiate there — in which case
the ORF competition models label-explanation, not initiation. `n_downstream_ejc` is **decay-side**
information, and the two arms ask opposite things of it:

| arm | question | owner |
|---|---|---|
| selection | does it drive `p_select`, where it does **not** belong? | model window |
| **decay (this row)** | does it drive `d_k`, where it **does** belong — and with what functional form? | interpretability |

Neither reads properly alone. If the same feature is used illegitimately on one head and legitimately
on the other, that is a complete account of the architecture's behaviour.

---

## THE PLAN, in plain terms

*Added after Pete asked whether the row was readable as a plan. It was not — the thirteen fields are
a pre-registration format built to expose implicit choices, and they never state the **procedure**.
That is a gap in the template rather than in this row: our path stages check the work and none of
them makes it legible to a reader who was not in the conversation.*

### The question

The model predicts NMD in two steps: it spreads weight across the candidate reading frames in a
transcript, then asks how likely each one is to trigger decay. We want to know whether the **second**
step has learned real biology.

> ⇒ **Corrected 2026-08-02, Pete's catch.** This paragraph first said "up to five candidate reading
> frames." That is the old v5 design and it is wrong for the current model. **Measured from the
> tensor itself** (`results_tensor_v6/nmd_tensor.h5`, built 2026-08-01): 41,765 transcripts carrying
> **796,584 candidates**, stored ragged via `offset`/`count` rather than in a fixed 5-slot array —
> **median 17 per transcript, mean 19.1, maximum 565**, and 91% of transcripts carry more than five.
> `data_prep.py`'s `MAX_ORFS = 5` and `select_priority_orfs` belong to the superseded pipeline;
> `build_tensor.py` built this one.
>
> **This makes the competition a more interesting object, not a footnote.** A weighting that resolves
> onto premature-stop-encoding ORFs is a far stronger claim across ~19 candidates than across 5, and
> it means the selection arm has substantially more to explain than either of us assumed.

There is a textbook rule for this. A stop codon triggers decay when it sits **more than about 50–55
bases before the last splice junction** in the transcript. That rule is the core of how NMD works,
and it is quantitative, so a model that learned it should show a step change in its output right
around that distance.

**So: does it?**

### Why the answer matters

If it did learn the rule, we have a clear statement about the model reproducing the central
mechanism of the biology it was trained on — the first such statement we would have.

If it did not, that is equally informative, because we **gave** it the junction positions. It would
mean the model was handed the ingredient and never learned the recipe — the same thing we already
found with the stop codon, which it also ignores despite being anchored on one. Two instances stop
being an oddity and start being a description of how this model works.

And it feeds the larger question you posed. If the decay step turns out to be doing very little,
then whatever the model is really doing must be happening in the ORF-weighting step — which is the
arm Maude is taking.

### The trick that makes the answer trustworthy

The worry with any result like this is that something else explains it. Transcripts with distant
junctions differ from transcripts with close ones in many ways — longer tails, different sequence
content — and any of those could produce a pattern we misread as the model knowing biology.

We avoid that by testing at **two** distances:

- **50–55 bases** — where the biology says a step change belongs.
- **500 bases** — where nothing biological belongs, but where *our own engineering* creates one,
  because that is the edge of the window the model reads. Inside it the model sees exactly where a
  junction is; outside it, only that one more exists.

Because everything else about a transcript changes *smoothly* across both distances, a sharp step at
either one can only come from what that distance means. **One is a check that we can detect a real
effect; the other is a check that we are not fooling ourselves.** We have not had either before.

### What we will actually do

1. **Verify the input encoding is correct.** A bug that put junctions in the wrong window was fixed
   in July; confirm on the current data that a junction lands in exactly the window it belongs to.
   The whole analysis is about which side of a boundary a junction falls on, so this is not a
   formality.
2. **Pick the candidates we can read cleanly** — those with exactly one junction after the stop. With
   more than one, a single candidate has some junctions inside the window and some outside, and no
   single distance describes it. Measured: **65,832 candidates** qualify.
3. **Recompute the distances using the stop the model actually commits to**, rather than the
   annotated one. They differ precisely in the transcripts with premature stops, which are the
   interesting ones.
4. **Run the model forward** over those candidates to get its decay probability for each. No
   mutagenesis, no substitution banks — this is the cheap kind of job.
5. **Look for a step change at each distance**, at three different zoom levels, so a result that only
   appears at one setting is visible as such.
6. **Measure the noise floor honestly** by running the identical test at about twenty distances where
   nothing should be happening. Whatever size of apparent step those produce is what "nothing" looks
   like, and our real result has to beat it.

### What each result would mean

| what we see | what it means |
|---|---|
| step at 50–55, none at 500 | the decay step learned the real rule, and our window size is not distorting anything. Best case |
| step at both | it learned the rule, **and** our window size is leaking into predictions. Both true, both reported |
| step at 500 only | predictions partly reflect an arbitrary engineering choice rather than biology. Fixable — the window is too small |
| step at neither | the decay step is just counting junctions and ignoring where they are. Pairs with the stop-codon finding |
| step at 50–55 the **wrong way round** | a threshold exists but runs backwards from the biology. Named here so it cannot get quietly reported as the first row |

### What could go wrong

- **The honest limit:** we supplied junction positions to the model, so if it shows the rule, what it
  learned is the *shape* of the response, not where junctions are. Worth stating plainly rather than
  letting the claim inflate.
- **Sample size at the 500 boundary is thinner** — about 1,800 candidates against 13,400 at the
  biological one. The positive control is better powered than the negative one.
- **Nothing here says anything about ORF weighting.** That is Maude's arm and this row does not
  license a word about it.

### What it costs

One forward-pass job over the candidate pool. Everything before it is local. The follow-up question —
whether the model's *sequence* sensitivity clusters near junctions — is a separate job on data we
already have, and is deliberately not part of this.

---

## Known answer, written down before measuring

*Path stage 3, and the stage we have never had. Recorded so the result cannot be rationalised
afterwards.*

The canonical mammalian NMD rule: a stop codon is recognised as premature when it lies **more than
~50–55 nt upstream of the last exon–exon junction**. This is textbook and quantitative.

**So we predict, in advance:** a discontinuity in decay probability near 50–55 nt, and **no**
discontinuity at 500 nt, which is our stop window's downstream half-width and has no biological
meaning whatever.

**That pairing is what makes this self-calibrating.** The same statistic at two cutoffs gives a
**positive control** (biology says a jump belongs at 50–55) and a **negative control** (only our
architecture could produce one at 500). This project has had no known-answer test since the
stop-codon control was retired; this supplies one without a SpliceAI port.

---

## The row

> **1 · Hypothesis.** The decay head's per-candidate decay probability responds to the position of a
> downstream exon–exon junction with a discontinuity near the canonical 50–55 nt boundary, and with
> no discontinuity at the 500 nt window boundary.
>
> **2 · Selection rule.** Unit = **candidate ORF** (`isoform_id` × `slot`) from the ORF pool,
> **restricted to candidates with exactly one downstream junction**. Running variable = that
> junction's distance from the candidate stop (`junction − orf_end`, 1-based both). The restriction
> is what makes the running variable unambiguous: with several junctions a candidate mixes both
> representations — some inside the window and positionally encoded, some outside and count-only —
> and no single distance describes it. **Measured, local, before this row was written:** 65,832 of
> 802,035 candidates qualify (8.2%); 13,403 lie within ±25 of the biological cutoff and 1,811 within
> ±50 of the architectural one (`interp_junction_density.py`, extended).
>
> **3 · Background.** None, and this is the design's point. A regression discontinuity compares the
> limit from below against the limit from above **at the same cutoff**, so there is no comparison
> population to enumerate wrongly. Every failure this project has retracted came from a background, a
> denominator or a selection rule; this design has none of the three.
>
> **4 · Held fixed.** Junction multiplicity — **by restriction**, see field 2. Gene — by clustered
> inference, since transcripts of one gene are not independent.
>
> **5 · Deliberately not held, and the reason is the design rather than an omission.** 3′UTR length,
> composition, selection mass, ORF length, transcript architecture. **All are smooth in the running
> variable across the cutoff**, so the discontinuity design removes them without matching. Matching on
> them would be the error: they are on the causal path from transcript structure to decay, and this
> project has already built one control that was more biased than the thing it tested.
>
> **6 · Null — measured, not assumed.** **Placebo cutoffs.** Run the identical estimator at ~20
> distances where neither biology nor architecture predicts anything, and take the distribution of
> estimated discontinuities as the noise floor. This is an in-sample measured floor at the analysis's
> own n, which is what axis 12 of the toolbox requires and what the ratio-floor failure cost us.
>
> **7 · Reference points.** Floor **and** ceiling from the placebo distribution, per bandwidth, in
> probability units. No analytic floor. The estimate is reported against its own placebo quantiles,
> never against zero.
>
> **8 · Aggregation.** Local-linear RD estimate per cutoff and bandwidth; interval by
> **gene-clustered bootstrap**. The placebo distribution passes through byte-identical estimator code
> — a floor computed one way and compared against an estimate computed another is two quantities with
> one name.
>
> **9 · Sweep.** Bandwidth {25, 50, 100} nt; cutoff {50, 55} for the biological test; polynomial order
> {1, 2} with **local linear as primary**. All reported. A discontinuity present at one setting is
> that setting.
>
> **10 · Decision rule, all outcomes fixed before the run.**
>
> | outcome | reading |
> |---|---|
> | jump at 50–55, none at 500 | the decay head implements the canonical rule and our window does not leak. Strongest outcome |
> | jump at both | it implements the rule **and** predictions partly reflect window size. Both reported, neither suppressed |
> | jump at 500 only | predictions reflect our architecture rather than the biology. Actionable: the window is too small |
> | jump at neither | the decay head has collapsed to **counting** junctions, discarding position it was given. Same shape as the stop-codon result — a supplied channel unused — making that a **pattern** rather than an isolated oddity |
> | **jump at 50–55 with the wrong sign** | a threshold exists and runs **opposite** to biology. Named explicitly because it would otherwise be rounded into row 1, which is how A2's missing third outcome nearly went |
>
> **11 · Licensed.** A jump at 50–55 licenses *"the decay head's output implements a discontinuity at
> the canonical exon-junction boundary."* **It does not license:** that the model learned this from
> sequence — junction position is **supplied** in channel 4, so what is learned is the *functional
> form*, not the location; that the model is biologically faithful in general; or anything at all
> about selection, which is the other arm.
>
> **12 · Owner.** Interpretability window. **Second implementation only if the outcome is row 1 or
> row 3** — those are the two that anything downstream would rest on. Cross-reviewed by the model
> window before code, both directions.
>
> **13 · Enumeration, beside every statistic.** n candidates, n genes, n within bandwidth per cutoff,
> the stop anchor used, the junction-multiplicity restriction, bandwidth, polynomial order, seed, and
> the mask expression. Never a discontinuity estimate without its placebo floor and its n.

---

## Prerequisites, both step-2 gates

1. **Channel-4 leak assertion.** Confirm on the current tensors that a junction appears in exactly
   the window it belongs to. The fix is in at `553d4c0`, and its own lesson was that the earlier
   overlap fix verified nucleotide channels only and so measured 8 of 9. This analysis is *entirely*
   about which window a junction lands in, so the assertion is load-bearing rather than hygiene.
   **Mine to run** — only this arm touches the tensor.
2. **Density on the operative stop.** The numbers in field 2 use the **annotated** stop. The
   operative stop is what the model commits to, and the two differ exactly in PTC transcripts, which
   is the interesting population. Treat field 2 as a feasibility read, not the sample size.

## Cost

Forward passes over the ORF pool for `d_k`. **No ISM, no substitution banks, no elevation rule.**
The companion question — whether ISM sensitivity concentrates near junctions — is a separate job on
the existing banks, with a different sample and a different scope, and is **not** part of this row.
