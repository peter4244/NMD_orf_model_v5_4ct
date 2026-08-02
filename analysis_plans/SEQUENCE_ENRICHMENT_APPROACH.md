# Identifying sequence enrichments from model importance measures

*Living document. Started 2026-08-02. Owner: model window. Reviewed by the
interpretability window.*

> **CANONICAL COPY:** `NMD_orf_model_v5_4ct/analysis_plans/` on branch `master`.
> The worktree split means this file also exists in
> `NMD_orf_model_v5_4ct_interp/` on `interp`, where it is a **read-only mirror that
> goes stale between merges**. Edit only the master copy. If you are reading this in
> the `_interp` tree, check `git log -1 --format=%cd -- <this file>` against master
> before trusting it.

Answers three questions: **what we do specifically, how it relates to prior work in
sequence analysis, and what we have changed for our situation.** Written to be read
by someone who has not followed the day-to-day.

---

## 1. What we are measuring, and what we want from it

We change one base at a time and record what happens to the model's output. That is
**in-silico mutagenesis (ISM)**. For every transcript position it gives the effect of
each of the three possible substitutions — the most direct importance measure
available, with no approximation. One bank is ~33 million such numbers.

ISM tells us **where** the model is sensitive. We want to know **what it is sensitive
to**: whether the decay branch has learned a recognizable sequence feature.

Getting from the first to the second is the whole problem, and it is harder than it
looks.

---

## 2. The core difficulty: the background *is* the claim

The obvious method is to take high-importance positions, count the sequences under
them, and compare against a reference set. **"Enriched" has no meaning until you say
"compared to what,"** and every possible answer embeds an assumption about what
counts as unremarkable.

This is not a subtlety we discovered. It is the central problem of **enrichment
testing**, which is what we are doing — scoped deliberately, because it would
overstate the case to call it the central problem of motif discovery generally.
Classical de novo discovery (MEME, Gibbs sampling) has its principal difficulties
elsewhere: search over a combinatorial space, significance under heavy multiple
testing, and representing degeneracy. The background model is one issue among those.
For *our* method — take a selected set of positions, count what is under them,
compare to a reference — it is the whole game. What is specific to us is *which*
backgrounds fail, and why.

**Four failures in one working session, all of which produced plausible tables and
none of which raised an error:**

| failure | what happened |
|---|---|
| regional composition | high-importance positions concentrate 3′ of the stop; 3′UTRs are AU-rich as a class, so an all-positions background recovers 3′UTR composition with no motif involved |
| a confound we invented | two jobs spent controlling for GC — then measured, and GC at high-importance positions is **0.503 against a background of 0.501**. It was diagnosed by eye from a table that *looked* AU-rich |
| the control was worse than the disease | restricting to GC-preserving substitutions (A↔T, C↔G) means A/T positions are scored one way and C/G positions another, which *selects for* C/G positions: GC 0.501 → **0.682** |
| a truncated tail | a k-mer table iterated the foreground, so k-mers absent from the high-importance set were invisible — exactly the most-depleted end |

The lesson we take from this is not "be more careful." It is that **a background is a
claim that must be measured on the inputs, not inferred from the outputs.** Failure
two is the clearest case: the confound was read off the result and controlled for
without ever checking the positions themselves.

**The structural fix is §3.2.2** — a stated functional hypothesis before each
analysis, with adjustments and stratifications chosen to match it in advance. The
background problem has no general solution; it has a determinate one once the claim
is fixed.

---

## 3. What we do specifically

### 3.1 Elevation is self-normalizing, and the rule is fixed in advance

"High-importance" is the **top fraction of each transcript's own valid positions**,
not a fold-change over that transcript's median.

The fold rule was tried first and **inverted on real data** — random placement
produced *more* long runs than the data. Cause: fold-over-median normalizes for
magnitude but not for tail shape, so the same rule selected 1.7% of positions on a
short-transcript pilot and 10.7% on the real banks, reaching 43% on transcripts with
very small medians. A fixed fraction makes the count identical by construction, so a
count-matched null is matched exactly rather than approximately.

**Per-transcript, not a pooled quantile.** These differ more than they sound: the two
definitions of "top 1%" overlap at Jaccard 0.24. A pooled quantile concentrates on
transcripts that are responsive overall, which is precisely the bias the
per-transcript rule exists to remove.

The threshold is **swept**, not fixed at one value, because the effect size is a
function of a number we chose and reporting one point hides that.

### 3.2 Every claim is compared against a null that is measured, not assumed

Three nulls are in use, and the distinction between them matters:

- **count-matched placement null** — the same number of elevated positions redrawn
  uniformly within the same transcript. Answers "would this many positions clump this
  much by chance."
- **in-sample noise null** — positions whose effect lies below the pipeline's own
  numerical floor. These are noise by construction and give the value a statistic
  takes when nothing is there.
- **composition null** — for motif work, the information a CWM column could carry
  from base composition alone, as `KL(elevated composition ‖ background composition)`.
  A column carrying less than this is explained by composition and is not a motif
  position. **This null is currently GLOBAL, which is a defect rather than a choice —
  see §3.2.1.**

**An analytic null is not a substitute for a measured one, even when they agree.** We
found the noise floor matching an analytic random-sign prediction and treated that as
validation. It matched on one summary statistic and missed by half on the other. The
agreement was real and meaningless — and it was the agreement that stopped us
looking.

#### 3.2.1 The composition null is currently global. It should be local.

`KL(elevated ‖ background)` needs a background, and **which positions are pooled into
it changes the bar by a factor of six.** Measured, seed 100, 600 transcripts, top-1%
elevation (`probe_composition_null_locality.py`):

    locality                              A      C      G      T     KL (bits)
    GLOBAL      background             0.256  0.243  0.258  0.243
                elevated               0.214  0.206  0.297  0.284      0.0186
    REGIONAL    upstream background    0.243  0.260  0.278  0.219
                upstream elevated      0.258  0.226  0.307  0.209      0.0064
                downstream background  0.269  0.226  0.238  0.267
                downstream elevated    0.191  0.194  0.316  0.299      0.0405
    PER-TRANSCRIPT                                            median   0.1421

**The global null is anti-conservative exactly where our data is.** Elevated positions
concentrate downstream of the operative stop, and the correct downstream bar is
**0.0405 bits — more than twice the global 0.0186**. A CWM column drawn mostly from
downstream seqlets and carrying 0.03 bits would clear the global bar and fail the one
that applies to it. Pooling regions averages a strict bar and a lenient one into a
middling bar that is right for neither.

**The per-transcript figure looks strictest and is mostly artifact.** KL is positively
biased at small samples: with ~20 elevated positions per transcript and four
categories, the expected bias alone is ≈ (K−1)/(2N ln2) ≈ **0.108 bits**, which
accounts for most of the 0.142 median. Per-transcript nulls are unusable at this
sample size without a bias correction, and quoting one would overstate the bar
roughly threefold.

**What each locality costs, in both directions:**

| null | tests | lets through | wrongly rejects |
|---|---|---|---|
| global | non-random given transcriptome-wide composition | regional preferences, credited as motif | little |
| regional | non-random *given where it sits* | nothing regional | a genuine preference *for* a region |
| per-transcript | non-random given this transcript's composition | nothing at transcript scale | anything learned about transcript class |

No locality is correct in general. **The null must be matched to the claim** — "the
model has learned a motif" needs one local enough to exclude composition and no more
local than the feature itself.

**The principled version for the seqlet work: build each cluster's null from the
composition of the positions that actually contributed to it.** A CWM averages
seqlets drawn from identifiable places; its columns should be judged against those
places. That is local by construction, requires no region scheme to be chosen, and
sidesteps §5.2 — a cluster drawn mostly from PTC intervals gets a PTC-interval null,
which is neither the upstream nor the downstream one.

Not yet implemented. **The ~0.016–0.018 figure quoted so far is the global one and is
a floor on the bar, not the bar.**

### 3.2.2 The toolbox, and the rule that selects from it

*Pete's proposal, 2026-08-02, and it is the governing discipline for everything
below.*

The background problem has **no context-free solution** — "unremarkable" has no
meaning in the abstract. It has a **determinate solution relative to a stated
hypothesis**, because a hypothesis says what is being claimed and therefore what must
be held fixed. So:

> **State a specific functional hypothesis before each analysis, then choose
> adjustments and stratifications to match it, explicitly and in advance.**

This blocks the failure mode that produced two of today's errors: notice something in
the *output*, infer a confound from it, control for that, and discover later that the
confound did not exist (§2) or that the control was worse than the disease (§3.3).
**A control chosen after seeing the result is a control chosen to explain the
result.**

#### The rule for matching

For a hypothesis that X drives the response:

- **adjust for** anything that correlates with X, affects the response, and is **not
  on the causal path from X**
- **stratify by** anything that is part of the exposure or lies on the path — never
  adjust it away (§5.2 is the worked case: PTC composition *is* what a PTC transcript
  is)
- **never adjust for a consequence of X**

The adjust/stratify distinction is the one that keeps being got wrong. Adjusting
removes; stratifying preserves the variable and shows whether the effect differs
across it. When in doubt, stratify — it is recoverable and adjustment is not.

#### The toolbox

**Composition**

| tool | holds fixed | do not use when |
|---|---|---|
| global composition null | transcriptome-wide base frequencies | the feature is regionally concentrated (anti-conservative — §3.2.1) |
| regional composition null | base frequencies within 5′UTR / CDS / 3′UTR, or upstream / downstream of the operative stop | the claim *is* about a region |
| per-transcript null | that transcript's own composition | n per transcript is small — KL bias ≈ 0.108 bits at n≈20 |
| seqlet-matched null | composition of the positions that produced the cluster | not yet implemented; preferred for motif claims |
| dinucleotide-preserving shuffle | local dinucleotide structure | positional claims — it destroys position |

**Geometry and position**

| tool | holds fixed | note |
|---|---|---|
| window fill extent | how much of the model's window carries sequence | encodes 5′ distance and ORF length (§5.3) — invisible to ablation |
| distance to nearest boundary | proximity to start, stop, junction, transcript end | a boundary can *move* (§5.2) |
| anchor choice | which ORF defines "downstream" — model-selected or annotated | tested: same k-mers either way, but that predates §5.2 |
| offset stratification | position relative to an anchor | only finds anchored features; absence proves nothing (§6) |

**Model-internal**

| tool | holds fixed | note |
|---|---|---|
| selection-mass threshold | restrict to candidates the model can route to | differential on the mechanism cell — long 5′UTRs lose more (§5.5) |
| magnitude matching | effect size | required before any claim that a property is enriched among *elevated* positions, since elevation is defined by magnitude (§6, directionality) |
| substitution class | GC-preserving vs GC-changing; transition vs transversion | the GC-preserving restriction is itself compositionally biased (§3.3) |
| branch | capture vs decay | everything here is `vals_decay` |

**Replication axes** — these are not adjustments; they answer different questions

| axis | varies | fixes | question |
|---|---|---|---|
| seed | initialization | transcripts | architecture or one initialization? |
| gene arm | genes (disjoint) | seed | does it hold on genes it was not found on? |
| subsample draw | which seqlets | everything | is it a sampling artifact? |

**Stratifications** — region, NMD label, PTC status, magnitude band, selection-mass
band, discovery/confirmation arm, transcript length, seed.

#### Worked examples, using our own hypotheses

| hypothesis | adjust for | stratify by | do NOT adjust |
|---|---|---|---|
| the decay branch has learned a sequence motif | composition, locally (seqlet-matched) | region; seed | position — a motif may have positional preference |
| importance is elevated 3′ of the stop | composition, within region | PTC status (§5.2) | region — it is the exposure |
| elevated positions are more directional | **magnitude** | magnitude band | nothing else; this claim is currently unadjusted and therefore unsupported (§6) |
| the model reads termination context | offset from the stop | stop identity (TAA/TAG/TGA) | composition at +4 — it is the thing being read |
| runs are not a GC-window artifact | substitution class (GC-preserving) | — | run length — it is the outcome |

#### What each adjustment licenses you to say

*Pete's addition, 2026-08-02.* An adjustment is only worth making if you know what it
buys, so each one carries the sentence you may write if the effect survives it — and,
where it matters, the sentence you may **not**.

| if the effect survives... | you may say | you may NOT say |
|---|---|---|
| **global composition null** | "the sequence here differs from the transcriptome average" | that it is a motif — a regional preference passes this |
| **regional composition null** | "the sequence differs from other sequence in the same region" | that it holds within a transcript |
| **seqlet-matched null** | "the pattern differs from the places it was found in" | — this is the strongest composition claim available |
| **dinucleotide-preserving shuffle** | "not explained by local dinucleotide structure" | anything positional — the shuffle destroyed position |
| **window-fill adjustment** | "not an artifact of how much sequence the model can see" | that it is independent of ORF length, which fill also encodes |
| **boundary-distance adjustment** | "not simply proximity to a landmark" | that the landmark is irrelevant — a moving boundary (§5.2) is not held by this |
| **anchor swapped, same answer** | "does not depend on which stop defines position" | that position and composition have been separated (§5.2) |
| **selection mass STRATIFIED** | "holds at each level of routable mass" | "independent of mass" — mass is the architecture and cannot be adjusted away |
| **magnitude matched** | "not explained by effect size" | — required before *any* claim about elevated positions, since elevation is defined by magnitude |
| **GC-preserving substitutions** | "not driven by the GC channel" | that composition is controlled — the restriction is itself compositionally biased (§3.3) |
| **autocorrelation-preserving null** | "clumps more than the track's own smoothness explains" | — random placement licenses **nothing** here (§7) |
| **across seeds** | "a property of the architecture, not one initialization" | that it generalizes to other genes |
| **across gene arms** | "generalizes to genes it was not found on" | that it is seed-independent |
| **across subsample draws** | "not a sampling artifact" | anything about the population, if the draw was not stratified |

**Combinations are where the useful sentences live:**

| combination | the claim it earns |
|---|---|
| local composition + geometry | "a sequence pattern, not where it sits or what surrounds it" |
| magnitude + local composition | "elevated positions carry distinctive sequence beyond simply being large" |
| seeds + gene arms | "a property of the architecture that holds on genes it was not found on" — the strongest available, and the only one a reviewer will not immediately question |
| mass-stratified + local composition | "a sequence pattern among candidates the model can actually route to" |
| autocorrelation-preserving null + local composition | "a *local* feature, not the smooth envelope" — the claim §7's run-length result was thought to have and does not |

**Every analysis records its row before it runs.** A row filled in afterwards is not a
pre-specification, and this document should say which analyses have one.

**Worked failure, today.** The run-length analysis had no row. Written afterwards it
would read: hypothesis "elevated positions cluster into short sequence features";
adjustments required — local composition, **autocorrelation-preserving null**,
selection mass stratified. It had none of the three. Its null was random placement,
which licenses only "these positions clump more than if they were scattered
independently" — and that was never in doubt, because the effect track is
autocorrelated at 0.74 across 80 bases by construction.

### 3.3 Controls are checked for bias of their own

Every control is itself a selection and can introduce what it removes. The
GC-preserving control is the worked example: built to remove a compositional bias, it
introduced one three times larger. **Before a control is trusted, the quantity it is
supposed to hold fixed is measured under it.**

### 3.4 Two implementations, independently written

The load-bearing statistics are computed twice, in code sharing nothing. This is not
belt-and-braces — it is the only thing that has actually worked. Across two days,
**nine errors of this class, none caught by the person who wrote them.** They do not
raise exceptions; they return tables.

Where the two implementations disagree, the disagreement is *attributed* rather than
averaged. Two such reconciliations: mean-versus-median, and pooled-versus-
per-transcript selection. Both were "same name, different set."

---

## 4. How this relates to previous work

### 4.1 The measurement

ISM is standard and long-established for sequence models. Attribution methods —
DeepLIFT and SHAP-family approaches — were developed largely to approximate it at
lower cost, since ISM requires a forward pass per substitution. We use ISM directly
because we can afford it: the decode optimisation brought a full bank from ~32 hours
to under an hour per seed.

**Consequence worth noting, stated carefully:** ISM is a *direct measurement of the
model's output* under a defined intervention, so there is no completeness property to
verify and no attribution error to bound. That is not the same as saying it gives
"exactly what attribution methods approximate" — ISM measures the causal effect of a
substitution, while DeepLIFT and SHAP decompose an output into per-feature
contributions against a reference. The two answer related questions and coincide only
under local linearity. What matters here is the weaker and sufficient claim: anything
built to consume per-base attribution can consume ours, and ours carries no method
error of its own.

### 4.2 The background problem

Well-trodden. Standard mitigations each fix one axis and leave others:

- **dinucleotide-shuffled backgrounds** preserve local composition, destroy position
- **GC- or region-matched backgrounds** fix locale, force an arbitrary region
  definition
- **position-matched controls** hold geometry, not sequence

There is no neutral background. There are only backgrounds whose biases have been
characterized.

### 4.3 The alternative: TF-MoDISco

TF-MoDISco (from the DeepLIFT group) makes a different move: **it does not compute a
ratio against a reference set at all.** It extracts short high-importance windows
("seqlets"), represents each by its **attribution matrix** rather than its letters,
clusters those, and averages within clusters into a motif-like pattern. The evidence
is **recurrence across many independent sites**. BPNet used this approach to recover
transcription-factor syntax, and it is now the standard route from a DNA sequence
model to motifs.

**Our data is natively its input.** MoDISco consumes "hypothetical contributions" —
what the output would be under each base at each position. That is what ISM measures.
Collapsing our four-valued object to one number per position and counting strings
discards three-quarters of the measurement *and* forces the background choice that
produced every failure in §2.

**What the switch trades, stated plainly:** the background problem for a clustering
problem. Seqlet threshold, similarity metric, cluster granularity and alignment
window are all assumptions capable of manufacturing structure. They are assumptions
about *the signal* rather than about what a fair comparison population is, which is
the better thing to have to defend — but it is a trade, not an escape.


**The clustering problem is smaller than it looks, if the standard is set correctly**
(Pete, 2026-08-02). Much of a similarity structure is usually **manifold rather than
clustered** — a smooth continuum with no natural partition — and the failure mode is
imposing a partition on it and then interpreting the pieces. The discipline that
follows: **take only the obvious clusters, and reject readily whenever the response
surface is flat.** Committing to that in advance makes the parameter sensitivity
largely moot, because an obvious cluster is stable across reasonable parameter
choices, and a cluster that appears at one setting and not its neighbors is by
construction not obvious.

Two operational consequences:

- **sweep the clustering parameters and keep only what survives the sweep** — the
  same move as sweeping the elevation threshold (§3.1), for the same reason: a result
  that depends on a number we chose is a property of that number.
- **check whether cluster structure exists at all before partitioning.** The
  distribution of pairwise seqlet similarities is diagnostic — multimodal for genuine
  clusters, smooth and unimodal for a manifold. Reporting it alongside any clusters
  makes "we found five patterns" falsifiable rather than assumed.

---

## 5. Modifications for our situation

Five, of which the first three are Pete's and the second is the one that generates a
requirement no existing method supplies.

### 5.1 RNA is single-stranded

MoDISco reverse-complement-collapses by default, because a transcription-factor motif
reads the same on both strands of DNA. **RNA does not have a second strand and our
model reads one.** Reverse-complement collapsing would merge genuinely different
motifs into one. This is the likeliest error in a naive port and it is silent.

More generally: any method imported from the DNA/TF literature carries
double-strandedness as an unexamined assumption, and it must be checked rather than
inherited.

### 5.2 The regions are natural, and their identity is not fixed

**The data are regional by construction** — 5′UTR, CDS and 3′UTR each have
characteristic base composition, and any enrichment computed across them recovers
that composition. Our first fix was to compute enrichment **within** each region
against that region's own positions, which is conditioning rather than matching and
avoids having to choose a matched background. The AU signal survived it (r =
0.77–0.81 against the pooled vector), so regional composition alone does not explain
it.

**But region identity is not a fixed property of a sequence, and this is specific to
NMD.** Consider a premature termination codon spliced into the middle of a coding
sequence. The interval between that PTC and the annotated stop is:

- **CDS by composition and evolutionary history** — codon-structured, GC-rich,
  selected as protein-coding
- **3′UTR by position and function** — downstream of the operative stop, carrying the
  exon-junction complexes that trigger decay

**So positional region and compositional region decouple, and they decouple exactly
where the biology is.**

**This is NOT something to correct for.** That framing was wrong in the first draft
of this document and Pete corrected it. The composition of the sequence downstream of
a PTC is not a nuisance variable sitting beside the exposure — **it is part of what a
PTC transcript is.** Conditioning it away would remove the thing we are trying to
measure, in the same way that adjusting for a variable on the causal path removes the
effect. There is no version of this analysis in which we want a PTC transcript's
downstream region to look compositionally like a normal 3′UTR.

**Framed correctly it is an asset, not a liability — a natural experiment the data
provides for free.** The two properties are ordinarily confounded across the
transcriptome, and here they are pulled apart:

| | positionally downstream of the operative stop | composition |
|---|---|---|
| normal transcript | yes | 3′UTR-like, AU-rich |
| PTC transcript, PTC-to-annotated-stop interval | yes | CDS-like, codon-structured, GC-rich |

**No artificial background can construct that cell.** A dinucleotide shuffle, a
GC-matched set, a region-matched set — none of them produce "downstream position with
coding composition," because in a normal transcriptome that combination does not
exist. Our data contains it because NMD substrates are what they are.

**What follows is an interpretive obligation rather than a statistical adjustment:**

- if an enrichment holds in the normal-transcript downstream region **and** in the
  PTC interval, it tracks **position**, not composition — and it did not need a
  background to establish that
- if it holds only in the normal downstream region, it tracks **3′UTR composition**
  and is not about the model's response to position
- if it holds only in the PTC interval, it is about **coding sequence in a
  post-termination position**, which is the most mechanistically interesting outcome
  and the one this design is uniquely able to see

**Every positional result must say which of these it is.** Not adjusted for — stated.

**Sample size, measured (bank_interp_s100, 2026-08-02):**

    transcripts in bank                             4,999
    with an annotated reference candidate           3,422   (weighted 28,739)
    operative stop BEFORE annotated stop              550   (weighted  3,914)
      ...NMD 282     ...control 268

    PTC-interval positions           696,340   median 1,068/tx   max 4,561
    comparison cell (true 3'UTR)   3,320,350   across 3,413 transcripts
    elevated positions in the interval at top 1%:  ~6,963

**Power: adequate for composition, thin for 5-mers.** ~7,000 elevated positions
supports a base-composition comparison comfortably, but spread over 1,024 possible
5-mers that is ~7 each — too thin for individual k-mer enrichments. 4-mers (~27
each) or 3-mers (~109 each) are sound. The comparison is runnable at reduced
resolution.

**The definition above is not yet the right one, and the split says so.** The cell
was expected to be predominantly NMD. It is **282 NMD against 268 control** —
essentially balanced. "Operative stop before annotated stop" therefore conflates two
populations: transcripts with a genuine premature termination codon, and transcripts
where the model simply commits to a shorter ORF than the annotation. A control
transcript does not have NMD-triggering premature termination, so those 268 are
largely the second — a fact about model behavior, not about the transcript.

**The interval must be defined from an annotation-derived PTC call**, not from the
model's selection, so that the cell does not depend on what the model chose. The
subset table carries `main_orf_stop` for this. Until that is done, the counts above
describe a mixed population and the three-cell comparison should not be run on it.

*Also noted:* we currently define "downstream" by the stop of the ORF the model
commits to. Defining it by the annotation gives the same enrichment (tested: identical
k-mers, r = 0.842/0.779 against 0.806/0.774) — but that test was run before this was
articulated and answers a different question. It shows the k-mer result is robust to
the anchor; it says nothing about the position/composition decoupling above.

### 5.3 Edge effects, from windows and from biology

**Two kinds, and both must be considered before any positional result is declared.**

**Window edges.** The model reads fixed windows around each candidate ORF, so the
*fill pattern* of those windows carries information: how much of the upstream window
is filled encodes distance to the 5′ end, and where the downstream fill stops encodes
ORF length. Two independent leaks of this kind have already been found, and neither
was visible to ablation, because every channel is blank outside the filled region —
so any "turn off feature X" test silently retains the fill mask. Both were found by
*conditioning*, not by ablation.

**Natural boundaries.** Start codons, stop codons, splice junctions and transcript
ends are real discontinuities in composition and in function. A signal that peaks at a
boundary may be about the boundary rather than about sequence. Per §5.2, a boundary
can also *move* — the operative stop is not the annotated stop in a PTC transcript —
so "distance to the boundary" is itself ambiguous.

**Standing requirement:** any positional claim states which boundary it is anchored
on, and whether the result survives anchoring on the other one.

### 5.4 Our importance measure has a hole at the observed base

You cannot substitute a base for itself, so ISM gives three values where attribution
methods give four. Filling that gap is a convention with consequences:

- **mean-centering**, `hyp[b] = vals[b] − mean_b(vals[b])` with `vals[obs] := 0`, is
  well-defined and recovers mean-centered contributions exactly — the common offset
  cancels, nothing is invented
- the alternative — observed = 0, others = −Δ — additionally asserts that the observed
  base contributes nothing, which is false precisely when the observed base is the
  consensus, i.e. the case a motif consists of

Both will be run. **Prediction recorded in advance: the second under-weights the
consensus base.**

*Practical note, because three scripts across two windows hit it:* `vals` is NaN at
the observed base, so `np.isfinite(x).all(1)` is **never true**. Use
`(np.isfinite(x).sum(1) == 3)`.

*And a second recurring one:* dividing by a transcript's median importance blows up
on the ~2.8% of transcripts whose median is near zero — importance values of 1e5 to
1e8. Two independent occurrences on the same day. **Rank statistics are immune and
should be the default reach**, which is also why elevation is a top fraction rather
than a fold change (§3.1).

### 5.5 Most positions cannot respond at all

The model chooses among candidate reading frames by a stick-breaking competition, so
a candidate that loses carries almost no weight and a substitution in its window
cannot move the output however the sequence changes. **A zero there is a true
statement, not a failed measurement.** Measured: every dead perturbation sits below a
selection mass of about 1e-8, and the rate tracks a float64 resolution boundary
exactly.

Two consequences: the population these positions are drawn from concentrates in
transcripts with long 5′UTRs, which is the mechanism cell — so any filter on
"responsiveness" is differential on the comparison of interest. And **any comparison
must state whether it counts unreachable candidates.**

---

## 6. What is not settled

Ordered by how much each bounds what the rest can be worth.

- **The PWM ceiling: a single position weight matrix explains 1.73% of importance
  variance at width 9, held out.** That bounds how much *any* single motif can
  account for, and it belongs beside every enrichment claim rather than after it. An
  enrichment can be real and still be a small part of what the branch is doing.
- **Directionality is NOT established and was moved here from §7** (Pete's call,
  2026-08-02). Elevated positions run ~21% above the measured in-sample noise floor
  on `|mean_b vals| / max_b |vals|`. But elevated positions are *defined* as the
  largest effects, and small effects sit near the floor where the three substitutions
  have near-random signs and the signed mean cancels — so the gap may be a
  signal-to-noise property of magnitude rather than learned directional structure.
  The banded profile rises monotonically with |effect| across all positions
  (0.359 → 0.427), which is what the magnitude explanation predicts. **If
  directionality is a function of effect size, "elevated positions are more
  directional" is tautological.** `probe_directionality_null.py` tests it and has not
  been run.
- **The observed-base convention for the port is unsettled and consequential.**
  Mean-centering against observed-equals-zero produce **different motifs from
  identical data**. Both will be run; the prediction that the second under-weights
  the consensus base is recorded in §5.4.
- **Whether the U-rich context is causal or merely associated.** The profile shows
  association. Nothing run so far distinguishes "the model responds to this context"
  from "this context co-occurs with whatever the model responds to."
- **§5.2 is an interpretive requirement with a live definitional problem.** The
  three-cell comparison is specified but the PTC interval is currently defined by the
  model's own selection, which conflates genuine premature termination with the model
  choosing a shorter ORF — evidenced by the near-balanced 282 NMD / 268 control
  split. It needs an annotation-derived definition, and the cell is the small one, so
  power bounds the comparison as well.
- **The sequence enrichment itself is not a claim.** It survives the regional
  control; the k-mer instrument is the wrong one and we are switching.
- **Seqlet calling has two defensible criteria** — signed and unsigned — selecting
  sets that overlap at Jaccard 0.52. Both will be run and the overlap reported.

### The gate, and why it is the only one

**Run the pipeline on a model known to encode a sequence feature: SpliceAI, and
GT/AG.** If it cannot recover a known motif, nothing it says about our model is worth
reading. This separates "the method failed" from "the model has nothing," a
distinction this project has not previously been able to make.

**It is the only such gate, which is sharper than it first appeared.** Recovering the
stop codon from a *stop-anchored* importance profile is guaranteed by the anchoring —
it checks indexing, not the enrichment method. So the stop-codon profile was never a
positive control, and SpliceAI is not merely the best available check but the only
one. *(Pete's point; it retires a check we had been counting.)*

### A negative result worth keeping: no stop-codon context preference

Presence of a stop codon has no variation to learn from, but its **identity and
context** do. Median percentile rank of importance by offset from the stop:

    offset        TAA      TAG      TGA
    -2 .. 0     71-73    80-84    72-78     the codon itself
    +1           70.4     81.4     73.9     the +4 readthrough position
    +2 .. +6    72-78    79-85    75-79

Flat everywhere. No peak at +4, no gradient away from the stop. The only separation
is a whole-window offset — TAG runs ~8 points above TAA at *every* offset including
the codon — which is a property of TAG-terminating transcripts as a class rather than
a context preference, and is not worth chasing at n = 124.

**The pairing is what makes this useful.** The +4 composition in the *data* is real
and structured — A at 0.32/0.35 after TAA/TAG, G at 0.33 after TGA, C depleted after
all three, against a 0.25 background. **So the termination-context bias is present in
the sequence and absent from what the model reads.** A negative result with a
demonstrated positive alongside it is worth far more than a bare absence.

## 7. Established, with its controls

**Read §7.1 first. The run-length result has been retracted.**

What currently survives, stated in the form the measurements support:

- **the five model seeds agree on the *sequence* far better than on the *positions***
  — k-mer enrichment r = 0.75 against a positional Jaccard of 0.125. This is what a
  binding preference at variable locations produces, and it uses no elevation null.
- **the composition profile of elevated positions**: keto (G+T) at 1.16× background,
  amino (A+C) at 0.84×, GC flat at 1.004×. A different measurement from the
  clustering — it conditions on being elevated and asks what base is there.
- **the PWM held-out fit**, which regresses on every position with no elevation
  threshold at all — and therefore bounds the rest: **1.73% of importance variance at
  width 9**.
- **the enrichment survives the regional control** — computed within each region
  against that region's own positions, r = 0.77–0.81 against the pooled vector.

**Everything above is `vals_decay`.** The capture branch is out of scope here, not
merely unreported.

### 7.1 RETRACTED: the run-length clustering result

**Both windows reported that decay-branch importance clusters into short runs at
hundreds of times a count-matched null, and that the run lengths were "wrong for the
encoding." Neither claim survives.**

The null was random placement of elevated positions. **The effect track is
autocorrelated at every scale and does not decay**, so random placement destroys
structure the architecture puts there and the comparison measures smoothness rather
than a sequence feature. Measured independently in both windows:

    correlation of the log effect track with     median r
      fill_count                                 +0.674
      log selection mass                         +0.934

      lag     raw autocorr    after coverage + mass
        1          0.965                   0.724
       10          0.930                   0.594
       25          0.863                   0.289
       50          0.795                   0.136
       80          0.740                   0.107

**The track is smooth because selection mass is smooth.** And the mass correlation is
**largely arithmetic** — |vals| ≈ mass × sensitivity, so a log-scale correlation is
close to definitional and is not a discovery about the model.

**This also retires the "wrong length for the encoding" argument.** Raw
autocorrelation persists past lag 80, which the ±25 GC window cannot explain either,
so the length-scale argument was never doing the work we credited it with.

**Two things worth recording about how this got through.**

*Independent implementation did not catch it.* Both windows wrote the statistic
separately and agreed closely — which validated the *implementation* and said nothing
about the *design*, because both used the same null. **Replication catches errors the
two implementations do not share.** A shared design assumption is invisible to it, and
we had been treating agreement as stronger evidence than it was.

*Mass is not to be adjusted away.* `P(NMD) = Σ P(select k)·d_k`, so a substitution at
a zero-mass candidate genuinely cannot move the output. Conditioning mass away asks
what would matter if ribosomes distributed uniformly, which the model never computes.
**Stratify and state** — the same resolution as the PTC composition point in §5.2,
reached from the other direction.

**The constructive residual, and it is a real test.** After removing coverage and
mass the track retains 0.724 at lag 1 and decays to 0.107 by lag 80 — **high at short
lag and decaying, which is the shape a local sequence feature makes and the raw track
does not.** Re-running the run-length analysis on the mass-residualized track, against
an autocorrelation-preserving null, is a test with a real chance of failing. That is
more than the original had.
