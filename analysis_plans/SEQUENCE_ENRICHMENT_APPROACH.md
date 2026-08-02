# Identifying sequence enrichments from model importance measures

*Living document. Started 2026-08-02. Owner: model window. Reviewed by the
interpretability window.*

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
to**: whether the decay branch has learned a recognisable sequence feature.

Getting from the first to the second is the whole problem, and it is harder than it
looks.

---

## 2. The core difficulty: the background *is* the claim

The obvious method is to take high-importance positions, count the sequences under
them, and compare against a reference set. **"Enriched" has no meaning until you say
"compared to what,"** and every possible answer embeds an assumption about what
counts as unremarkable.

This is not a subtlety we discovered. It is the central problem of motif discovery
and the main thing distinguishing MEME, HOMER and their successors. What is specific
to us is *which* backgrounds fail, and why.

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

---

## 3. What we do specifically

### 3.1 Elevation is self-normalising, and the rule is fixed in advance

"High-importance" is the **top fraction of each transcript's own valid positions**,
not a fold-change over that transcript's median.

The fold rule was tried first and **inverted on real data** — random placement
produced *more* long runs than the data. Cause: fold-over-median normalises for
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
- **composition null** — for motif work, the information content a column would carry
  from base composition alone. Currently ~0.016–0.018 bits, from the measured
  composition of the elevated set.

**An analytic null is not a substitute for a measured one, even when they agree.** We
found the noise floor matching an analytic random-sign prediction and treated that as
validation. It matched on one summary statistic and missed by half on the other. The
agreement was real and meaningless — and it was the agreement that stopped us
looking.

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

**Consequence worth noting:** ISM gives *exactly* what attribution methods
approximate. Anything built to consume attribution output can consume ours, and ours
does not carry the approximation error those methods are known for.

### 4.2 The background problem

Well-trodden. Standard mitigations each fix one axis and leave others:

- **dinucleotide-shuffled backgrounds** preserve local composition, destroy position
- **GC- or region-matched backgrounds** fix locale, force an arbitrary region
  definition
- **position-matched controls** hold geometry, not sequence

There is no neutral background. There are only backgrounds whose biases have been
characterised.

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

*Practical caveat:* the PTC interval exists only in transcripts with a premature stop,
so the cell is smaller than the others and the comparison may be underpowered. That
is a limit on what can be concluded, not a reason to adjust.

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

- **mean-centring**, `hyp[b] = vals[b] − mean_b(vals[b])` with `vals[obs] := 0`, is
  well-defined and recovers mean-centred contributions exactly — the common offset
  cancels, nothing is invented
- the alternative — observed = 0, others = −Δ — additionally asserts that the observed
  base contributes nothing, which is false precisely when the observed base is the
  consensus, i.e. the case a motif consists of

Both will be run. **Prediction recorded in advance: the second under-weights the
consensus base.**

*Practical note, because three scripts across two windows hit it:* `vals` is NaN at
the observed base, so `np.isfinite(x).all(1)` is **never true**. Use
`(np.isfinite(x).sum(1) == 3)`.

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

- **The sequence enrichment is not yet a claim.** It survives the regional control;
  the k-mer instrument is the wrong one and we are switching.
- **§5.2 is an interpretive requirement, not an open technical problem.** The
  position/composition decoupling is not corrected for; every positional result states
  which of the three readings it supports. The three-cell comparison that would settle
  it has not been run.
- **Directionality is NOT established and has been moved here from §7** (Pete,
  2026-08-02). Elevated positions run ~21% above the measured in-sample noise floor
  on the ratio `|mean_b vals| / max_b |vals|`. But elevated positions are *defined*
  as the largest effects, and small effects sit near the floor where the three
  substitutions have near-random signs and the signed mean cancels — so the gap may
  be a signal-to-noise property of magnitude rather than learned directional
  structure. The banded profile already rises monotonically with |effect| across all
  positions, which is what the magnitude explanation predicts. **If directionality is
  a function of effect size, "elevated positions are more directional" is
  tautological.** `probe_directionality_null.py` tests it and has not been run.

- **Seqlet calling has two defensible criteria** — signed and unsigned — selecting
  sets that overlap at Jaccard 0.52. Both will be run and the overlap reported.
- **The gate before any of this counts:** run the pipeline on a model known to encode
  a sequence feature — SpliceAI and GT/AG. If it cannot recover a known motif,
  nothing it says about our model is worth reading. This separates "the method
  failed" from "the model has nothing," a distinction this project has not previously
  been able to make.

---

## 7. Established, with its controls

For completeness, what currently survives:

- decay-branch importance clusters into short runs at every threshold from 0.2% to
  5%. The count-matched null is **exactly 0** below the 5% threshold, so the ratio is
  undefined there rather than large; at 5% it is 19,039 against 50. **A zero null is
  correct rather than suspicious here, and that is checkable:** random placement
  predicts 0.08 runs of ≥4 at p=0.01 and 45 at p=0.05, against 0 and 50 observed. The
  null implementation reproducing closed-form binomial arithmetic is a stronger
  statement about it than any ratio
- it replicates on **disjoint gene sets** — the discovery/confirmation split, which is
  the only axis that answers "does this hold on genes it was not found on"
- it survives with the GC channel held **bitwise constant**, at 57% of its
  arity-matched level
- the **excess** sits at runs of **4 or more bases** — which is motif scale and
  *wrong* for the ±25 averaging window of the GC channel, so the encoding cannot be
  producing it. The *typical* run is 1.38 bases and runs of ≥4 are 3.3% of runs
  (1,865 of 56,675, seed 100, top 1%). The distinction matters: a reader takes
  "run lengths are 4–6" as typical when it is the tail
- the five model seeds agree on the *sequence* far better than on the *positions*,
  which is what a binding preference at variable locations produces

**Everything in §7 is `vals_decay`.** The capture branch is out of scope here, not
merely unreported.

The directionality figure that used to sit in this section was corrected three times
in one morning, shrank each time, and has now moved to §6 as not established. The
first version travelled between windows before any of the corrections did, which is
the part worth remembering about all of these.
