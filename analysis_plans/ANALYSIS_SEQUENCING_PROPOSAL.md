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
| 13 | **enumeration reported beside every statistic** — n units, n observations, n selected, seeds, exclusion handling, and the mask expression. **Never the ratio alone** | the keto ratio read 1.16× in one document and 1.148× in another for two days across two windows, both internally consistent, because a table reporting only `.581` and `.501` makes every candidate cause invisible. Measured 2026-08-02: 1.148× has no producer and was struck. The enumeration is what would have caught it on day one |

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
> **ANSWERED — the measurement landed while this was being written.** `model_a2_enumeration.py`,
> Explorer job 8896445, runlog at `analysis_plans/model_a2_enumeration_runlog.txt`, bank
> `bank_interp_s100.h5`, 4,999 transcripts, dead cut 1e-8.
>
> **Dead is 5.5% of valid overall and is concentrated, not spread.** 606,774 dead of 11,062,149
> valid. The per-transcript dead fraction is **0.000 through the 80th percentile**, 0.148 at the 90th
> and 0.783 at the maximum — so §5.5's warning holds in the sharpest possible form: dead positions are
> a property of a minority of transcripts, and that minority is the mechanism cell. Mean live per
> transcript is **2,091**, against the 2,213 valid the arithmetic used.
>
> **Occupancy is far from complete, which is the answer to (2) and it goes the way the concern went.**
> At **8 bands: 39,992 cells, 18,395 qualifying at ≥100 live — 46.0%.** The per-cell live-count deciles
> are `0 0 0 169 518 8030`, so **at least 30% of cells are empty outright**. A transcript occupies
> ~3.7 of 8 bands on average, not 8. The `2,213 / 8 ≈ 277` arithmetic described a cell that mostly
> does not exist.
>
>     bands   cells    qualifying        median elevated per qualifying cell (top 10%)
>       4    19,996   14,542  (72.7%)                  ~62
>       8    39,992   18,395  (46.0%)                  ~45
>      16    79,984   21,588  (27.0%)                  ~36
>
> ⇒ **So the primary freezes at 4 bands, not 8**, sweeping 8 and 16 as before. This is a real
> trade — 4 bands is a coarser control on routing, which is the axis the gate exists to hold — and it
> is taken because a finer band that half the transcripts cannot enter does not control routing better,
> it controls routing on a different and unstated population. **Recorded as a proposal rather than
> executed unilaterally**, since the enumeration is the modeling window's and the trade is a judgement
> call: if they prefer 8 with the exclusion profile carried explicitly, that is defensible and I would
> not argue it hard.
>
> ⇒ **And the restriction I wrote before the measurement has to be withdrawn.** I specified that the
> primary analysis be restricted to transcripts qualifying in *every* band. At 8 bands that set is
> near-empty — per-band qualifying counts run 2,027–2,653 of 4,999, so an all-bands intersection is a
> few hundred transcripts at best and possibly far fewer. The restriction was written to make the
> bands comparable and would instead have replaced the analysis with an underpowered one. **What
> replaces it:** the qualifying transcript set is reported **per band** with its overlap matrix, and
> the **conditional** outcome may only be read if the high- and low-mass bands it contrasts are drawn
> from substantially the same transcripts. That is the same guarantee, obtained by measurement rather
> than by exclusion, and it costs no power.
>
> **Still open from this block:** (3), cell counts under the locale split, which halves everything
> above. At 4 bands and 72.7% qualifying there is room for it; at 8 there is not, which is a second
> argument for 4.

**Cell size and the top fraction, jointly — no longer provisional. Measured, job 8896445.** The
earlier arithmetic here read: mean 2,213 *valid* positions per transcript, 8 bands giving ~277 per
cell and top 10% giving ~28 elevated, with a contingency for "if dead is half of valid." **Every part
of that is now superseded and it is worth recording how each part was wrong**, since this is the
specification's own arithmetic and the error class is the one the document is organized around.

- *Denominator.* Bands read **live** positions, not valid. Mean live is **2,091**, not 2,213.
- *The contingency was the wrong shape.* Dead is **5.5% of valid**, not half — but it is not spread
  thinly either. It is **zero through the 80th percentile** and 0.783 at the maximum, so no mean or
  single contingency describes it. A distribution was the right thing to ask for and a representative
  value would have misled in both directions at once.
- *The divisor was the real error.* Positions do not spread evenly across global quantile bands. At 8
  bands **only 46.0% of cells qualify** and at least 30% hold nothing. The ~277-position cell was
  mostly hypothetical.

**Frozen: primary (4 bands, top 10%)** — ~62 elevated per qualifying cell, 72.7% of cells qualifying
— swept over (8, 10%) and (16, 20%). A cell contributes only with **≥100 live positions**. Excluded
cells are reported with their count and mass distribution, because that exclusion is differential by
§5.5's own argument and Q1a now shows exactly how: the transcripts carrying dead positions are the
top two deciles, which is the mechanism cell.

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

> **RESOLVED 2026-08-02, measured. The unstratified keto ratio is ~1.16×, and 1.148× is struck on
> Pete's call.**
>
> **The ratio is insensitive to the set**, which was the hypothesis and it was wrong. Six definitions
> spanning 600 against 4,999 transcripts, ±60 edge trim against none, and dead-in against dead-out
> backgrounds all return **1.155–1.162**. So the disagreement was never a set-definition problem, and
> the candidate causes listed here previously — background scope, dead handling, seed count, the
> finite mask — are all ruled out by measurement rather than by argument.
>
> **1.148× has no producer.** `FINDINGS_DECAY_SEQUENCE_2026-08-02.md:29` cited
> `probe_elevated_composition_profile.py`, which prints a background row and a per-offset
> U/C/A+T/G+C table and nothing else — never an elevated A/C/G/T row, never keto or amino. Re-running
> its exact set definition on the same bank returns 1.158. The script is unmodified since it was
> committed with those findings.
>
> **Quote 1.16× with its enumeration attached**, per field 13: 4,999 transcripts, 11,062,149 valid
> positions, 110,631 elevated at top 1% within transcript, seed 100, dead included, finite mask
> `(np.isfinite(vals_decay).sum(1) == 3)`. Producer `analysis_plans/model_a2_enumeration.py`,
> Explorer job 8896445.
>
> **The gate was never affected**, because the decision rule is the within-cell permutation
> percentile and "retains X% of the unstratified ratio" was explicitly ruled out. Writing the rule
> defensively is what made this a documentation problem rather than a re-run.

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

**Not the rule: "retains X% of the unstratified ratio."** That makes the confounded quantity the
denominator, and it holds whatever the unstratified value turns out to be — it was written when that
value was thought to be 1.148× and stands unchanged now that it is measured at ~1.16×. Monotone
decline across bands is a **diagnostic to report**, not the gate.

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

*Written against the thirteen-field template 2026-08-02. Two things came out of writing it that were
not visible in the five-field version, both marked ⇒.*

> **1 · Hypothesis.** Beyond what base frequencies predict, the *arrangement* of bases at
> equally-routed positions carries information.
>
> **2 · Selection rule.** None — and this is the property being bought. PWM fitting regresses over
> **every valid position** of the mass-residualized track, as `analysis_pwm_fit.py` already does on
> the raw track. There is no elevated set to enumerate wrongly. B1 is gated on A2 for *meaning* (a
> negative A2 makes the composition claim a routing claim, and there is then nothing to go beyond),
> not because it shares A2's selection.
>
> **3 · Background.** Composition, held. The claim is against what local base frequencies predict.
> ⇒ **The composition bar must be local and the code currently uses the global one.**
> `SEQUENCE_ENRICHMENT_APPROACH.md` §3.2.1 measures the global bar at 0.0186 bits and the applicable
> **downstream** bar at **0.0405 — more than twice** — and states that "the ~0.016–0.018 figure quoted
> so far is the global one and is a floor on the bar, not the bar." `analysis_pwm_fit.py:64` hardcodes
> `KL_BAR = (0.0159, 0.0183)` and reports every fitted column against it. Elevated positions
> concentrate downstream. **This must be fixed before B1 runs**, or every column called "above the
> composition floor" is being judged against under half the bar that applies to it.
>
> **4 · Held fixed, by stratification unless stated.** Mass (A2's bands). Locale (A2's split).
> Composition — **held, not stratified**, which is the one removal in the programme and is legitimate
> because composition is the null here rather than the exposure.
>
> **5 · Deliberately not held.** Arrangement — it is the measurement. Effect magnitude — no elevation
> threshold is applied, so magnitude never enters the selection and axis 7 does not bite here. This is
> the only analysis in the programme with that property.
>
> **6 · Null.** Autocorrelation-preserving, on the mass-residualized track. ⇒ **This is the same null
> the region caller specifies, and it is not yet validated on this track — so B1 inherits the region
> caller's null validation as a prerequisite, which was recorded nowhere before this row was
> written.** See `REGION_CALLER_SPEC.md` §4 and the correction now standing there.
> The residual track is the right substrate, and that part is measured: after removing coverage and
> mass the track holds 0.724 at lag 1 and decays to 0.107 by lag 80 — "high at short lag and decaying,
> which is the shape a local sequence feature makes and the raw track does not"
> (`SEQUENCE_ENRICHMENT_APPROACH.md` §7.1).
>
> **7 · Reference points.** Per column, KL against the **local** composition bar, with the floor from
> a permutation at that column's own n — not the analytic bar, and not the global one. KL is
> positively biased at small n (≈ (K−1)/(2N ln2), which is 0.108 bits at N≈20 and accounts for most of
> the 0.142 per-transcript median), so a permutation at matched n is what absorbs the bias.
>
> **8 · Aggregation.** Fit on discovery genes, evaluate on confirmation genes — disjoint by gene.
> Interval by gene-clustered bootstrap. The null passes through byte-identical aggregation code.
>
> **9 · Sweep.** PWM width; number of components in the deflation series. Primary declared before the
> run. A single PWM assumes one motif and a blend is indistinguishable from a poor fit, which is why
> deflation is in the instrument rather than optional.
>
> **10 · Decision rule, all outcomes fixed before the run.**
> **Positive:** ≥1 column carrying information above its *local* bar at matched-n permutation, held
> out on disjoint genes, at the primary width. → the motif claim, in full.
> **Negative:** no column clears its local bar. → the claim is compositional, not a motif. Write it as
> composition; do not proceed to C.
> **The third outcome, and it is the one that gets rounded up:** columns clear the bar in-sample and
> not held out. → the fit is describing discovery genes, and licenses nothing about the model. Named
> here because a poor held-out fit is the expected shape of overfitting and will otherwise be read as
> the negative, which is a different result.
> **A poor fit does not license "no sequence preference"** — pre-registered in
> `analysis_pwm_fit.py`'s own docstring, and it is the underpowered negative that demoted MoDISco the
> first time.
>
> **11 · Licensed if positive.** "Among positions the model routes to equally, and beyond what base
> frequencies predict, sensitivity depends on the arrangement of bases." The top row of the
> combinations table, with all three qualifiers attached. **It does not license** any statement about
> which base the model prefers — that is exactly what was held fixed and is undetectable here by
> construction.
>
> **12 · Owner.** Interpretability window. Second implementation **only if positive**, per the
> replication rule — a negative B1 stops the branch and nothing is downstream of it.
>
> **13 · Enumeration.** n positions fitted, n genes per arm, n columns, per-column n, the local bar
> used and which population it came from, the seed, and the mask expression. Never a KL value without
> the bar it is being compared against.

### B2 — the three-cell PTC comparison

*Written against the thirteen-field template 2026-08-02. Field 10 was empty in the five-field version
and filling it found a missing outcome, which is the second time that has happened on this template.*

> **1 · Hypothesis.** Sequence downstream of a premature stop behaves as coding sequence in a
> post-termination position.
>
> **2 · Selection rule.** Three cells, defined on transcripts rather than on positions:
> **normal-downstream** (positions 3′ of the annotated stop in transcripts with no premature stop),
> **PTC-interval** (positions between the operative stop and the annotated stop), and **control**
> (positions 3′ of the annotated stop in the same PTC transcripts). The third is what makes the
> comparison within-transcript and is why the design does not need an artificial background.
>
> **3 · Background.** Each cell against its own positions. **Never pooled across cells** — pooling is
> precisely what the position/composition decoupling makes meaningless.
>
> **4 · Held fixed.** Nothing on toolbox axis 4. Transcript identity, by the within-transcript
> control cell.
>
> **5 · Deliberately not held — and this is the whole design.** Composition. The position/composition
> decoupling *is* the exposure: a PTC interval is downstream by position and coding by composition,
> and conditioning that away removes the thing being measured, in the same way adjusting for a
> variable on the causal path removes the effect. **No artificial background can construct this cell**
> — a dinucleotide shuffle, a GC-matched set and a region-matched set all fail to produce "downstream
> position with coding composition," because in a normal transcriptome that combination does not
> exist.
>
> **6 · Null.** Within-cell permutation at that cell's own n. No placement null — the track's
> autocorrelation makes placement invalid here for the same reason it was invalid for run length.
>
> **7 · Reference points.** Floor and ceiling per cell, measured in-sample at that cell's n. The
> PTC-interval cell is the small one, so its floor is the widest and must not be compared against a
> floor measured on the large cell.
>
> **8 · Aggregation.** Per transcript, then across transcripts; gene-clustered bootstrap.
>
> **9 · Sweep.** Elevation fraction, matching A2's primary and its sweep so the two are readable
> against each other.
>
> **10 · Decision rule — the three readings, plus the fourth that was missing.**
> Enriched in **both** downstream cells → tracks **position**. Enriched in the **normal cell only** →
> tracks **3′UTR composition**. Enriched in the **PTC interval only** → **coding sequence in a
> post-termination position**, the most mechanistically interesting outcome and the one this design is
> uniquely able to see. ⇒ **Enriched in neither → the model does not respond to this contrast at all**,
> which is a real outcome, is not any of the three readings, and would otherwise be reported as
> "underpowered" — a claim about the instrument standing in for a result about the model. Which of the
> four is being reported is stated explicitly; **never pooled**.
>
> **11 · Licensed.** Exactly one of the four readings above, named. **Limit:** power. The PTC-interval
> cell is the small one and that bounds the conclusion — it is not a reason to adjust anything.
>
> **12 · Owner.** Interpretability window. Second implementation only if the reading is the
> PTC-interval-only one, which is the only outcome anything downstream rests on.
>
> **13 · Enumeration.** n transcripts and n positions **per cell**, the NMD/control split per cell,
> the definition used for "premature," the seed, and the mask expression. This field is why the
> prerequisite below is a prerequisite.

> **BLOCKING PREREQUISITE — an annotation-derived PTC definition.** As currently specified the cell is
> defined by the *model's own* selection: operative stop before annotated stop. That conflates "the
> model committed to a shorter ORF" with "genuine premature termination," and roughly half the cell is
> the former. **Run as written, B2 compares a mixed population against itself.** `main_orf_stop` is
> already in the subset table; the redefinition is small and must precede B2 rather than accompany it.
>
> ⇒ **The provenance of the numbers that justify this block, corrected 2026-08-02.** An earlier
> reviewer — me — reported that the quoted figures had no producer in either worktree. That was wrong,
> and wrong for an instructive reason: the producer is
> `8725053:analysis_plans/probe_ptc_interval_cell.py`, committed 08:20, **before** the claim was
> quoted at `4dd9f69` 09:37, and it was invisible from an interpretability worktree that was 26
> commits behind. Checking the artifact requires being on the branch that has it.
>
> What survives is narrower and still blocks: **there is no committed runlog**, and the script prints
> counts and weighted counts only — it never computes `51.3%`, and no quantity in it is a `44.0%`
> background. Both percentages are hand-derived, the denominator of the second is defined nowhere, and
> the only 44.0% elsewhere in the corpus is unrelated (7,896 of 17,944 transcripts whose triggering
> upstream ORFs are all admitted, `RETRAIN_PLAN_2026-08-01.md:280`). The script also reads
> `bank_interp_s100.h5` alone. **Run it, commit the runlog, and quote the cell with its enumeration**
> — field 13 applied to a number already in use. The block itself is almost certainly right; what is
> missing is the evidence for it in the form this document requires of everything else.

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
