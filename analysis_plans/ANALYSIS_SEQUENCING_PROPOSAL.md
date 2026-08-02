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

## The row template — thirteen fields, and an empty one blocks implementation

Added 2026-08-02, after the A2 row was found to fix four lines and leave a dozen implementation
choices open. **The lesson is not "review rows more carefully."** That row was reviewed carefully and
looked finished, because four confident lines look finished. The fix is structural: name the fields,
so a row that omits one is visibly incomplete without anyone having to notice.

Every field below was a real decision that a real implementation would otherwise have made silently.

| # | field | the failure it prevents |
|---|---|---|
| 1 | **hypothesis**, one sentence | the run-length statistic: two implementations, agreement to four decimals, both wrong, no hypothesis written |
| 2 | **selection rule** — what defines the set, *and within what* | global vs per-transcript top-1%, Jaccard 0.24; global-then-binned vs within-stratum |
| 3 | **background** — compared against what, at what scope | a global background collapses a stratified test back to the confounded version |
| 4 | **held fixed**, and **by stratification or by removal** | A1's row said stratification in one line and residualized in the next |
| 5 | **deliberately not held**, and why | composition is the measurement in A2 and the exposure in B2; adjusting it away removes the thing |
| 6 | **null** — what it preserves | random placement destroyed autocorrelation the track has architecturally |
| 7 | **reference points** — floor *and* ceiling, measured in-sample at the analysis's own n | ceiling assumed 1.0 (is 0.75), floor assumed 0 (measured 0.387) |
| 8 | **aggregation** — pooled or per-unit, and the interval | pooling reintroduces the weighting per-transcript selection removes |
| 9 | **sweep** — which parameters, and the primary | the elevation threshold selected 1.7% on the pilot and 10.7% on the banks |
| 10 | **decision rule**, with **all** outcomes fixed before the run | A2 had two outcomes written; the missing third was an interaction that would have been rounded up to a positive |
| 11 | **licensed if positive / negative**, and **what a positive does not license** | every inflated claim this week dropped the qualifier rather than inventing a result |
| 12 | **owner**, and whether it gets a second implementation | replication is for load-bearing survivors, not for everything |
| 13 | **enumeration reported beside every statistic** — n units, n observations, n selected, seeds, exclusion handling, and the mask expression. **Never the ratio alone** | the keto ratio is 1.16× in one document and 1.148× in another, both internally consistent, because a table reporting only `.581` and `.501` makes every candidate cause invisible. Two windows, two days |

**A row with an empty field is not ready to implement.** The implementing window sends it back rather
than resolving the gap in code — a choice made inside an implementation is invisible to review, which
is the toolbox's own diagnosis of all eleven errors.

**This is a template, not a checklist to satisfy.** A field can legitimately read "not applicable, and
here is why." What it cannot do is be absent.

---

## Phase A — repair. Cheap, existing code, and a go/no-go on the rest

**Phase A is A2. One measurement.** A1 is dropped, A3 is tidying the other window owns, A4 is deferred pending the Hill/Dy interaction-attribution work. Pete asked whether ORF weighting should be
accounted for at all, and the model window independently concluded the same thing from the other
direction: **every surviving claim is downstream of A2**, while A1 can only retire something already
retracted. A4 runs last because it is the only item needing new forward passes, and a negative A2 is
what makes it urgent.

**A correction to how these were first written, and it is the error the toolbox exists to prevent.**
A1's row said *stratification* in one line and *residualized track* in the next. Those are different
operations. Residualizing removes routing from the data and describes a model that weights every ORF
equally, which is neither our model nor ribosomes. **A2 holds mass by stratification** — within bands
of similar routing — never by removal. (A1 said the same thing and is dropped; it is named here only
because this is the record of the correction.)

Neither of these needs a new instrument. A2 and A3 run on the existing banks.

### A2 — **THE GATE.** Is the U-rich/keto signature routing or sequence?

*Row rewritten against the thirteen-field template 2026-08-02 by the incoming interpretability
window. The previous five-field row was the one the modeling window found fixed four lines and left a
dozen choices open; the specification below it filled most of those, but the row itself was never
migrated, so the gaps stayed invisible where the template exists to make them visible. Four
substantive changes are marked ⇒ and each is argued in place.*

> **1 · Hypothesis.** The keto (G+T) enrichment at elevated positions holds among positions of equal
> selection mass, equal coverage and equal locale.
>
> **2 · Selection rule.** Top fraction by `max_b |vals_decay|` within each **(transcript × mass band ×
> locale)** cell — within-stratum, never global-then-binned. ⇒ **Ties are broken at random under a
> recorded seed, and the tie fraction is reported per cell.** This is not housekeeping: in the dead
> band the values are exact zeros by float64 cancellation, so the elevated set is *entirely* ties, and
> a deterministic `argsort`/`partition` returns the lowest indices — the 5′-most positions of the
> transcript. The control band would then report 5′UTR composition and be read as an instrumental
> keto signature, in whichever direction 5′UTR composition happens to point. **Verify first** that the
> dead values are bit-identical rather than merely tiny; one line on one bank settles it.
>
> **3 · Background.** The non-elevated positions of the **same cell** — same transcript, same mass
> band, same locale. Never a global background, which collapses the test to the confounded version.
>
> **4 · Held fixed, all by stratification and none by removal.** Mass (8 live quantile bands + dead).
> Locale (⇒ **now a third cell dimension**, see below). Coverage (marginal balance check within band,
> not a cell dimension — jointly stratifying fragments cells below the n the permutation null needs).
>
> **5 · Deliberately not held.** Composition — it is the measurement. Effect magnitude — it cannot be
> held without dissolving the elevated set, and the residual circularity is bounded explicitly under
> *What a positive does not license*.
>
> **6 · Null.** Within-cell permutation of the elevated label at that cell's own n. Preserves cell
> membership, cell size, cell composition and mass band; destroys only the association between the
> label and position.
>
> **7 · Reference points.** Floor and ceiling both from that cell's own permutation distribution, at
> that cell's own n. No analytic floor, and no unstratified ratio as a denominator. The unstratified
> keto ratio is **unresolved** (below) and is not a reference point for anything here.
>
> **8 · Aggregation.** Per transcript, then unweighted mean across transcripts (axis 8); interval by
> gene-clustered bootstrap. ⇒ **The permutation null passes through byte-identical aggregation code.**
> A floor computed per-cell and compared against a band-level statistic is two quantities with one
> name, which is the error class this whole document is organized around.
>
> **9 · Sweep.** Bands × top fraction, jointly, so the elevated count stays roughly constant while
> band resolution varies. **All parameters provisional** pending the descriptive measurement below.
>
> **10 · Decision rule.** Three outcomes, fixed before the run — see the table below. ⇒ **The band
> count is evaluated under gene-clustered resampling**, not as a raw count of qualifying bands.
>
> **11 · Licensed.** Positive: the signature is a property of what the decay head reads, not of where
> the model routes. Negative: our best surviving finding is routing. What a positive does *not*
> license has its own subsection and is part of this field, not an addendum to it.
>
> **12 · Owner.** Interpretability window. **Second independent implementation by the modeling window,
> written against this row** — shared specification, independent code. The one case the replication
> rule was written for.
>
> **13 · Enumeration, beside every statistic.** n transcripts, n cells, n positions, n elevated, n
> excluded and the exclusion reason, seeds, tie fraction, and the mask expression. **Never the ratio
> alone.** This field exists because the unstratified keto ratio is currently two numbers over two
> different sets, both of them reported without their sets.

#### ⇒ Locale is a cell dimension, because the row claimed it and the specification did not deliver it

The previous row said "positional region, stratified" and nothing below it stratified region: bands
were log-mass only and locale never reappeared. The claim and the instrument disagreed, on the gate.

It has to be delivered rather than dropped, and the within-cell background does not already cover it.
Within a single transcript and a single mass band, elevated positions can still sit downstream of the
operative stop while the non-elevated background spans both sides of it — so the comparison recovers
the upstream/downstream composition difference and reports it as a keto effect. That difference is
large and measured: `SEQUENCE_ENRICHMENT_APPROACH.md` §3.2.1 puts the downstream composition null at
**0.0405 bits against 0.0064 upstream**, a factor of six.

**Locale = upstream / downstream of the operative stop**, the same two-way split §3.2.1 already uses,
so the two analyses share a definition rather than inventing a second one.

**The cost, stated because it lands on parameters that are already provisional:** this doubles the
cell count and roughly halves cell size, and it interacts directly with the dead-fraction problem
below. Both must be resolved by the same descriptive measurement, in one pass, before anything
freezes. **The locale-pooled version is reported beside the primary as a diagnostic** — if the two
disagree, that disagreement is the §5.2 decoupling showing up in the gate, and it is a result rather
than a nuisance.

#### The specification, fixed 2026-08-02

Written because the modeling window read the row and found it fixed four lines and left roughly a
dozen implementation decisions open. That is the toolbox's own diagnosis — *every error was a choice
made implicitly inside an implementation* — and a shared row that leaves the choices open puts the
independence back exactly where the arrangement exists to remove it. **This section is the shared
specification. Deviations from it are findings about the specification, not implementation detail.**

**Bands.** Global log-`mass` quantile bands, so a band means the same thing in every transcript.
Primary **8 bands**, swept over 4 and 16 (§3.1: a result that exists at one parameter setting is that
setting).

**Elevation is within-stratum, not global-then-binned.** `|vals| ≈ mass × sensitivity`, so a globally
elevated set concentrates in high-mass bands by construction and leaves the low-mass bands nearly
empty — no power exactly where the test has to discriminate. Each (transcript × band) cell
contributes its own top fraction. This is the same one-name-two-sets shape as the global versus
per-transcript top-1% pair that overlapped at Jaccard 0.24.

**The dead cut is applied first, and the bands are quantiles of the LIVE positions.** Reading fixed
2026-08-02 after the modeling window found that "8 global log-mass quantile bands" and "dead positions
get their own band" do not compose. Dead (`mass` below ~1e-8, §5.5) is a **hard threshold, not a
quantile** — `log(0)` is undefined and `mass` is float32 with a hard floor, so the band edges cannot
be computed across it. Result: **9 cells per transcript** — 8 live quantile bands plus dead.

> **⚠ THREE PARAMETERS BELOW ARE PROVISIONAL AND MUST BE FROZEN AFTER ONE MEASUREMENT.**
> The band count, the top fraction and the ≥100 floor were chosen against **valid** positions as the
> denominator. The reading just fixed bands over **live** positions, which is a smaller set — so the
> three numbers were chosen against a denominator this reading removes. That is the
> enumerate-what-you-divided-by error, in this specification's own arithmetic, found by the window
> implementing it.
>
> **Required first — one descriptive pass, three quantities, because they constrain the same
> parameters and measuring them separately would freeze each against the others' assumptions.**
>
> 1. **The distribution of the dead fraction per transcript** on one bank — not a representative
>    value, because the ≥100 floor bites in the tail and §5.5 puts the tail in the mechanism cell
>    (dead positions concentrate in long-5′UTR transcripts).
> 2. ⇒ **The occupancy histogram: how many of the 8 live bands each transcript actually occupies.**
>    The arithmetic below divides mean positions by band count, which assumes a transcript's positions
>    spread evenly across *global* quantile bands. They do not have to. Mass is per-candidate and
>    structured, so a transcript may sit in two or three bands and be empty in the rest — in which
>    case cells are far larger than ~277 and far fewer than 9 per transcript, and the ≥100 floor
>    removes whole transcripts rather than trimming tails. **This is a reference point assumed rather
>    than measured, in the power calculation of the one gate**, which is the error class named at the
>    top of this document.
> 3. ⇒ **Cell counts under the locale split**, since locale is now a third dimension and halves them.
>
> A descriptive count on data that already exists. **Band parameters freeze after it, and not before.**
>
> ⇒ **Consequence for the decision rule, whatever the occupancy turns out to be.** If transcripts do
> not occupy all bands, then band 1 and band 8 are computed over **different transcript populations**,
> and "≥2/3 of qualifying bands" aggregates over sets that are not comparable. The **conditional**
> outcome — present in high-mass bands, absent in low — then cannot be distinguished from a change in
> which transcripts qualify, which is exactly the reading that outcome was added to protect. So the
> primary analysis is restricted to **transcripts qualifying in every band**, with the full set
> reported beside it and the difference between them stated. If the occupancy measurement shows near-
> complete occupancy, the restriction costs nothing and the two agree; if it does not, the restriction
> is the only version of the test that means what the row says.

**Cell size and the top fraction, jointly — provisional.** At mean 2,213 *valid* positions per
transcript, 8 bands would give ~277 per cell and **top 10% within cell** ~28 positions, swept as
(4 bands, 5%) and (16 bands, 20%) to hold the elevated count roughly constant while band resolution
varies; a cell contributing only with **≥100 valid positions**. **If dead is half of valid, the live
cells are ~138 and ~14 elevated, and a materially larger share falls under the floor** — so these
become (4, 20%) or similar. Excluded cells are reported with their count and mass distribution
regardless, because that exclusion is differential by §5.5's own argument.

**Background is within-cell.** Elevated positions are compared against the non-elevated positions of
**the same transcript in the same band** — never a global background, which collapses the test back
to the confounded version.

**Aggregation** is per-transcript, then unweighted mean across transcripts (axis 8), interval by
gene-clustered bootstrap.

**Mass jointly, coverage marginally.** Mass is the axis in the hypothesis. Coverage correlates at
r ≈ 0.54 but is not the exposure, and stratifying jointly on both fragments cells below the n the
permutation null needs. Coverage is reported *within* band as a balance check; a band that comes out
imbalanced gets a coverage-stratified re-run rather than fragmenting the whole analysis.

**Dead positions get their own band and are the control, not an exclusion.** Every dead perturbation
sits below mass ~1e-8 (§5.5), and §5.5 shows dropping them is differential on the mechanism cell —
long 5′UTRs lose more — so exclusion would build a differential-selection error into the one gate.
As a band they are free and they are the answer to the magnitude flag below: **nothing is read there,
so any keto signature appearing in the dead band is instrumental rather than biological.**

**Seeds.** All five. Cross-seed agreement is reported *beside* the gate, not folded into it — seed is
a replication axis and folding it in conflates "the effect exists" with "the effect is stable across
initializations." One stability floor only: the direction must hold in **≥4 of 5 seeds** for a
positive to read as positive.

**Reference points measured in-sample, per band.** The null is a **within-cell permutation of the
elevated label** at that cell's own n, and every ratio is reported against its own permutation floor
and ceiling. Whatever the unstratified reference turns out to be, it was measured over ~11M positions
and the sampling floor at ~28 elevated per cell is far larger.

> **⚠ THE UNSTRATIFIED KETO RATIO IS UNRESOLVED. Do not quote a number for it.**
> `SEQUENCE_ENRICHMENT_APPROACH.md:849` says **1.16×** (elevated keto .581, background .501);
> `FINDINGS_DECAY_SEQUENCE_2026-08-02.md:29` reconstructs to **1.148×** (elevated .576, background
> .502). **Both are internally consistent** — the backgrounds sum correctly in each — so this is
> neither a transcription slip nor rounding. Same statistic, different set: the fourth instance of
> the error class, sitting in the two documents that govern this gate.
>
> **The gate is unaffected**, because the decision rule is the within-cell permutation percentile and
> "retains X% of the unstratified ratio" was explicitly ruled out. It affects only what may be
> *quoted*, including in figure legends. Resolution is the second implementation's first output, per
> field 13 — the table with its enumeration beside it, never the ratio alone.
>
> **Candidate causes, in order of how much they move .581 to .576:** whether the background is all
> valid positions of the elevated transcripts or of all transcripts; whether dead positions are in
> the background (they cannot be elevated but they are valid, so including them shifts background
> composition without touching the elevated set — **the same open question as the band construction,
> so one measurement may answer both**); seed count; and whether the finite mask was
> `isfinite.sum(1) == 3` or something that silently dropped rows.

#### The decision rule, fixed before the run

A gate whose reading is chosen after the run is not a gate.

| outcome | rule | what it licenses |
|---|---|---|
| **positive** | within-band keto ratio above its own permutation 95th percentile, same direction, in **≥ 2/3 of qualifying bands**, direction holding in ≥4/5 seeds | the signature is a property of what the decay head reads. Everything downstream means what we have been saying |
| **negative** | fewer than half of qualifying bands | our best surviving finding is routing. Propagates to the composition profile and the cross-seed k-mer agreement — **not to the PWM**, see below |
| **conditional** | present in high-mass bands, absent in low | **not the clean positive.** The sequence response is conditional on routing — an interaction, not independence. Reported as such and not rounded up. **Read only against the all-bands transcript restriction**, or it is confounded with which transcripts qualify |

⇒ **The band count is evaluated under gene-clustered resampling, not as a raw count.** Bands are not
independent evidence: every transcript contributes to several of them and carries its own composition,
so a subset of U-rich transcripts produces "positive in 8 of 8" with no per-band evidence at all. The
gene-clustered bootstrap is already specified for the interval and is simply not used by the counting
rule; it must be. This is toolbox axis 8 — transcript identity — re-entering through the decision rule
after being handled in the selection rule.

⇒ **A negative does not propagate to the PWM, and the previous rule said it did.** The composition
profile and the cross-seed k-mer agreement both select positions with the elevation rule, so a
negative reaches them. `analysis_pwm_fit.py` does not: it regresses on **every valid position** with
no elevation threshold anywhere — *"NO FOREGROUND/BACKGROUND SPLIT ANYWHERE"* in its own docstring,
and no elevation rule in `accumulate()`. Nothing about what elevation selects can reach a statistic
that never selects.

**The PWM is threatened by a negative A2, but by a different route, and it needs its own test rather
than inheriting a verdict.** If composition tracks routing, a PWM can predict importance because
sequence predicts locale and locale predicts routing — held-out on disjoint genes throughout, because
that relationship generalizes across genes perfectly well. Held-out prediction tests generalization,
not that the predictor is sequence rather than something sequence stands in for. **Recorded here as an
open item, not folded into the gate.** Getting this wrong in the direction the old rule had it costs a
real result to a verdict that does not apply to it; getting it wrong in the other direction keeps a
result that should have died. Neither is acceptable in a rule that is read once, at the moment it
fires.

Between half and 2/3 is **ambiguous, and is not resolved by looking at it.** It is resolved by more
seeds or a finer sweep, decided before either is examined.

**Not the rule: "retains X% of the unstratified 1.148×."** That makes the confounded quantity the
denominator. Monotone decline across bands is a **diagnostic to report**, not the gate.

#### What a positive does not license

A2 does not escape axis 7. Within a band the elevated positions are still the larger-magnitude ones,
so a positive licenses *"among equally-routed positions, the more sensitive ones are keto-enriched"*
and **not** *"composition is enriched independent of effect magnitude."* This is a weaker circularity
than the directionality claim — directionality mechanically rises with magnitude, composition does
not — but it is not zero: if the instrument's numerical sensitivity is base-dependent, magnitude and
composition correlate through the encoder rather than through what the head reads. **The dead band
tests exactly that**, and A3 owns the general magnitude question. The bound belongs in the row rather
than being discovered when someone asks.

Cost: one job. **This is the load-bearing measurement of Phase A, and the only one.**

**A2 gets two independent implementations — the one case the replication rule was written for.**
The rule reserves replication for results that survive their own row and are load-bearing
downstream. A2 is the most load-bearing measurement in the plan, a negative propagates to three
other results, and with A1 dropped **nothing else in Phase A could catch an error in it**.

Crucially, the second implementation is written **against this row**, not against an independent
reading of the question — so the specification is shared and only the code is independent. That is
the version of replication that catches what replication can catch, and it is what we did *not* have
when both windows implemented the run-length statistic from a premise neither had stated.

### A1 — DROPPED

Pete, 2026-08-02: *"If we don't think A1 is informative we shouldn't do it."*

It was labelled a weak instrument and kept because it was nearly free. That is not a
reason. **A weak instrument returning a positive is worse than no instrument**, because
the positive gets quoted — and eleven errors in two days came mostly from analyses
that returned plausible tables. Cheapness is not an argument for adding surface area.

The claim it would have tested is already retracted, and run length is weak in both
directions: mass-driven smoothness produces it without sequence, and a robust motif
produces none at all.

### A3 — is directionality anything but magnitude? *(tidying, not a gate)*

> **Hypothesis.** At equal effect size, elevated positions are more directional than others.
> **Held fixed:** effect magnitude — the axis without which the claim is circular.
> **Owner:** model window (`probe_directionality_null.py`, written, unrun).
> **Licensed if positive:** directionality returns from §6 to §7. **If negative:** it is deleted,
> not softened.
>
> **Honestly labelled:** this resolves an item in §6 but nothing is downstream of it. A negative
> deletes a line from a document; a positive adds an observation nothing depends on. Kept only
> because it is written and owned by the other window — it is not a Phase A gate, and the same
> criterion that dropped A1 applies to it.

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

## A4 — RESURRECTED, scoped to regions. Decided 2026-08-02 after reading Torop et al.

**Decision: A4 returns, restricted to called regions, and decoupled from SmoothHess.**

Once a region is called, exhaustive pairwise substitution inside it is 28 position-pairs for an
eight-base region. The quadratic cost that deferred A4 was only ever the genome-wide version, so it
is affordable exactly where we want it **whatever any screening method shows**. That decouples two
decisions that were tangled together.

> **Hypothesis (A4).** Substitutions at two positions within a called region produce a joint effect
> exceeding the sum of their individual effects.
> **Held fixed:** the region, hence locale and routing; pairs are drawn within a candidate.
> **Licensed if superadditive:** the model integrates across positions, and single-base ISM
> under-detects whatever it integrates.
> **Licensed if additive — and this is narrower than I first wrote it:** *these pairs did not jointly
> cross a ReLU threshold.* A ReLU network is **exactly additive within a linear region**, so
> additivity is the default rather than evidence. It does not license "single-base ISM sees
> everything."
> **And the positive is sharper than the caveat suggests** (model window): an m-of-n detector built
> from conv→ReLU has its threshold *at* a ReLU boundary, so boundary-crossing is precisely the
> signature being sought.

### SmoothHess as an optional genome-wide screen — a separate, smaller bet

Torop et al. estimate the Hessian of the Gaussian-convolved network via Stein's Lemma, using *n*
gradient calls instead of O(d²) forward passes. Two objections resolved and one live:

- **"The structural channels cannot be perturbed"** — answered. Σ is ours to choose; set zero
  variance on channels 4–8 and only the bases move.
- **"One-hot input is off-simplex"** — answered by projection. A substitution is a direction in
  one-hot space, so the pairwise interaction is a quadratic form `uᵀHv` and the off-simplex entries
  are never interpreted.
- **Live: scale mismatch.** The Hessian is infinitesimal; a substitution is a *finite* move of norm
  √2. `uᵀHv` is curvature at a point, double-ISM is a finite difference across the whole
  displacement. They agree only if the second-order Taylor holds over a unit-scale step, which is
  empirical — and is why the validation is necessary rather than a formality.

> **Validation row, pre-registered.** Correlate `uᵀHv` against exact double-ISM **within candidate**,
> then aggregate the within-candidate correlations. Σ pre-specified over a small declared sweep, all
> of it reported — tuning Σ to maximize agreement is fitting the validation.
>
> **Predicted in advance, so it cannot be believed later:** the *between*-candidate correlation will
> be **high and largely meaningless**, because both quantities carry the selection-mass envelope
> (log-mass correlation 0.93). It is reported beside the within-candidate figure to show exactly how
> much a naive validation would have been routing.

**That prediction is the point.** The mass confound retracted the run-length result, and it would
have passed this validation too. It is not four mistakes; it is one, in four costumes.

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
>
> **BLOCKING PREREQUISITE — an annotation-derived PTC definition.** As currently specified the cell
> is defined by the *model's own* selection: operative stop before annotated stop. Measured on the
> banks, that cell is **550 transcripts, 282 NMD against 268 control — 51.3% NMD against a 44.0%
> background.** It is barely enriched for the thing it is supposed to select, because "model
> committed to a shorter ORF" and "genuine premature termination" are conflated and roughly half the
> cell is the former. Worse, the crossed table shows the annotation flag runs the *opposite* way
> inside it. **Run as written, B2 compares a mixed population against itself.** `main_orf_stop` is
> already in the subset table; the redefinition is small and must precede B2 rather than accompany
> it.

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

## The stopping rule — Phase A turns on A2 alone, and A2 is not the programme's only gate

The method document leaves "what counts as enough to stop" open. **Rewritten 2026-08-02 because A1
was dropped and the rule still said "if A1 and A2 are both negative."** That is not a stale word to
tidy: with A1 gone the whole programme hangs on **one** measurement, and a rule naming a
non-existent analysis would be read loosely at exactly the moment it fires. The "both" was doing
silent work that no longer exists.

⇒ **First, a naming collision that has to go, because this rule is read once and at speed.** This
document said "there is one gate, and it is A2." `SEQUENCE_ENRICHMENT_APPROACH.md` §6 says "The gate,
and why it is the only one" and means **SpliceAI/GT-AG**. Two governing documents, each asserting
uniqueness about a different object, neither acknowledging the other — so a rule that fires on "the
gate" resolves differently depending on which document the reader has open. That is the same failure
as the stopping rule naming a dropped A1, and it was rewritten for exactly that reason. **They are
distinguished by name from here on:**

| | what it gates | if it fails |
|---|---|---|
| **the interpretive gate — A2** | whether our own results mean sequence or routing | Phases B and C do not run |
| **the method-validation gate — SpliceAI/GT-AG** (`SEQUENCE_ENRICHMENT_APPROACH.md` §6) | whether the seqlet pipeline can recover a known motif at all | C2 is unreadable whatever it outputs |

They are independent and neither substitutes for the other: A2 could pass on a pipeline that cannot
find a motif, and the pipeline could recover GT/AG while our own signature is routing. **Two further
gates exist and are named as such elsewhere in this document** — the region caller's excess-over-
surrogates criterion, which A4 is downstream of, and B2's annotation-derived PTC definition. The
honest count is four, not one.

**The interpretive gate is A2.**

- **A2 negative** → the decay branch's sequence contribution is not separable from routing **by
  single-base ISM at this resolution**. Write that and stop. It is a real result: the model's
  apparent sequence sensitivity is a readout of its selection distribution. The scoping to
  single-base ISM is not a hedge — a pattern the model recognizes robustly is invisible to our
  instrument by construction, so the negative belongs to the instrument and not to the model.
  **A negative propagates to what selects on elevation:** the composition profile and the cross-seed
  k-mer agreement. (The model window argued the cross-seed agreement was independent of Phase A; it
  is not, because it uses the elevation rule even though it uses no elevation null. That correction
  is theirs, accepted.) ⇒ **It does not propagate to the PWM**, which uses no elevation rule at all;
  the separate threat to the PWM, and why it needs its own test rather than an inherited verdict, is
  stated with the decision rule above.
- **A2 positive, B1 fails** → the claim is compositional, not a motif. Write it as composition. Do
  not proceed to C.
- **Proceed to C only if B1 or B2 is positive**, since C is the expensive branch and its output is
  uninterpretable without C1.

---

## Costs, roughly

| phase | jobs | wall | needs new code |
|---|---|---|---|
| A2 | 1 array | ~2 h | stratification and the within-band permutation null |
| A3 | 1 | <1 h | none, written |
| B1 | 1 array | ~2 h | multi-PWM deflation |
| B2 | 1 array | ~2 h | three-cell stratification |
| C1 | — | ~1 day | the port |
| C2 | several | days | extraction, clustering, nulls |

**Phase A is under three hours and decides whether Phases B and C happen at all.** That is the
argument for the ordering.
