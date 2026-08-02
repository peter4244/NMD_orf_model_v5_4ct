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
