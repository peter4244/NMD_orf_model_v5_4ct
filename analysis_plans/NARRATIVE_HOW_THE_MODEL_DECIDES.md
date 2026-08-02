# How the model decides — the running narrative

**Owner: model window (storyteller), from 2026-08-02.** Drafted by the interpretability
window and handed over on Pete's instruction; reconciled against the committed record and
rewritten. Every sentence must be traceable to a numbered claim with a producer; where it
is not, it says so.

*This document exists because a summary is where our claims go wrong. Four framings
survived in prose on 2026-08-02 after the measurement behind them had been retracted or
was never taken. Analyses get re-run; narratives get repeated. This is the artifact that
has to be checked like an analysis.*

**Standards cited, not restated:** `PRIMARY_DIRECTIVE.md`; `CAPTURE_HEAD_STORY.md` for the
claim→code map; `RETRAIN_ARCHITECTURE_CHANGES.md`.

    [unclaimed]                 nothing measured behind it
    [unclaimed — job NNNNNNN]   measured, producer and runlog exist, never filed as a claim

**Markers are deliberate. Do not tidy them.**

## Depends on

**Claims** C1–C15, C17–C21 (`CAPTURE_HEAD_STORY.md`); **C16 retracted**. **Decisions** D48, D50, D55, D56.
**Findings** `FINDINGS_ORF_SCANNER_2026-08-02.md`, `FINDINGS_TILED_PERTURBATION_2026-08-02.md`,
`FINDINGS_DECAY_SEQUENCE_2026-08-02.md`. **Retrain items** 3, 4, 6, 8.
**Jobs** 8898926 · 8898939 · 8899132 · 8899353 · 8899766 · 8899820 · 8899905 · 8899965 ·
8900114 · 8900209 · 8900229 · 8900407 · 8900420 · 8900473 · 8900631 · 8900643 · 8900685 ·
8900746 · 8900942 · 8900950.

---

## The question

The model predicts decay in two stages: **pick** which of ~15 candidate reading frames
matters, then **judge** whether it triggers decay. Pete's question: does the picking stage
model ribosome initiation, or does it pick whichever frame best explains the label?

**The answer is neither, and the third thing is more interesting than both.**

---

## 1. The picker gates a queue; it does not rank

`p_select_k = p_capture_k × Π_{j<k}(1 − p_capture_j)`, ordered 5′→3′. A candidate wins by
being competent **and** by everything upstream being judged incompetent — which is what
leaky scanning is.

So the head's job is to be **quiet** at the wrong candidates.

Three ways of choosing, scored on GENCODE's own `nonsense_mediated_decay` call — where the
annotated CDS **is** the frame that causes decay, so recovery and decay-causation coincide
(n = 1,099 transcripts in 915 genes; job 8900942):

| | |
|---|---|
| the head's own argmax | 0.304 |
| the most 5′ candidate | 0.702 |
| **the queue built from the head** | **0.793** |

**The head's raw preference is nearly worthless, and the queue built from that same head is
worth +0.489** — CI [+0.455, +0.522], and it wins on 546 transcripts while losing on 9. The
head is not being overruled by the queue; it *is* the queue's only input. That gap is the
whole claim: a gate, not a ranker. The head's **low** scores upstream are what let a
ribosome pass through to the right frame.

**Its margin over pure position is real but modest: +0.091**, CI [+0.053, +0.129], 241 won
against 141 lost (job 8900950, gene-clustered). Position alone recovers **0.702** here, so
most of the queue's accuracy is reachable with no model at all, and the model's contribution
over knowing nothing but the order of the candidates is a tenth.

*Both intervals are gene-clustered bootstraps, not McNemar, because transcripts of one gene
share architecture. McNemar agrees and is quoted in the runlog; it is the optimistic bound.*

## 2. What the picker reads

**Clean, and measured: initiation context immediately 5′ of the start codon.** Perturbing
those 25 bases moves the head's logit more than anything else in its 900-base upstream
window — 0.413 / 0.283 / 0.241 / 0.275 across ORF-length bands, **at the same position in
every band**, in a tile 99.6% filled throughout and therefore carrying no fill confound.
That position is where Kozak context sits. *Jobs 8900209, 8900420.*

**On a diffuse background.** Across the upstream window, 32 consecutive tiles differ from
their neighbours by ~6% and never by more than 50%. Capture's sensitivity there is
**diffuse rather than sparse** — a broad low response with one sharp peak at the start
codon, not a scatter of isolated hot positions. *Job 8900420.*

**Unresolved: a second component tracking ORF length.** The response peak moves with length
(−13, +12, +37, +87). Whether that is the head reading **our fill boundary** or reading
**whatever happens to be filled** cannot be separated by this design, and the question is
**closed to tiling** — downstream tiles already sit below the ~42-position receptive field.

**And that length route is partly ours.** The ATG window's fill stops at the ORF midpoint,
so fill extent = `min(100, length/2)`, and fill saturation alone is a **47× marker** for
"this is the real ORF" — `P(reference | saturated)` 11.1% against 0.24%. *Retrain item 3.*
`[unclaimed]` **Whether the head uses it is not established.**

**What it does not read is the thing biology says decides.** Among ORFs under 200 nt — 69%
of candidates, median 81 nt — `capture ~ kozak` is +0.061 against `capture ~ length` +0.429.
Reference ORFs that are short are recovered **0.276** of the time against **0.735** for long
ones. **The head fails precisely where initiation biology says context should decide.**

## 3. What the picker is actually selecting for

**Length, and position — in that order, and neither is what biology would nominate.**

**The head selects for ORF length, at +0.760** (C8, job 8899132) — the strongest single
association measured anywhere in this document. Among candidates under 200 nt it is +0.429,
**seven times** the Kozak association. §2 is the same finding from the sequence side: the
head fails precisely where initiation context should decide.

**Part of that is our own encoding, not gene-finding.** The ATG window fills to
`min(100, length/2)`, so for any ORF shorter than 200 nt — 69% of candidates — **where the
fill stops encodes ORF length exactly** (C10, retrain item 3). How much of +0.760 is the
model reading sequence and how much is it reading our boundary is not separated.

**The queue adds position**, mechanically rather than by preference: candidates are ordered
5′→3′ and each one's survival is discounted by everything upstream. On GENCODE's NMD call
the most 5′ candidate *alone* recovers the decay-causing frame **0.702** of the time (§1),
so ordering carries most of what the picker achieves.

`[unclaimed]` **`p_select ~ length` has never been measured.** Both positive numbers above
are `p_capture` — the head — and this section's subject is the product. Nothing establishes
that the *picker* selects for length; it is inferred from the head plus the queue's
construction. **This is the gap that makes the section read as negative**, and it is the
cheapest thing outstanding: the same conditioning structure as the table below, on length
instead of junction count.

**What it is not selecting for.**

| `p_select ~ junction count`, conditioning on | median |
|---|---|
| nothing (marginal) | **−0.050** |
| ORF length only | +0.447 |
| position only | −0.553 |
| **length and position together** | **−0.070** |

*Job 8900746. The head's own aversion, −0.453, collapses to −0.009 holding length — that
arm is entirely length.*

**Routing is indifferent to junction structure.** Length and position each mask the
other in opposite directions, so holding either alone manufactures a signal that is
not there; holding both returns −0.070, agreeing with the marginal −0.050.

⇒ **Pete's founding hypothesis does not hold at the routing step under the only valid
conditioning.** The two single partials show large signal (+0.447, −0.553) and both are
artifacts — each is the other confounder leaking. The junction preference enters at the
decay multiplication.

**And an independent second route reaches the same place.** `p_select ~ ejc` is positive
*by construction*: the queue's survival factor falls monotonically with slot, and earlier
slots carry more downstream junctions. Measured in-bank, a **queue with no model in it**
scores +0.334 raw and **+0.568** holding length, against the model's −0.050 and +0.447. So
zero was the wrong reference, and against the right one the model routes toward
junction-bearing candidates *less* than pure ordering does — a deficit of 0.125.
*Job f523f72.* **Bound:** the degenerate null maximises queue influence, so 0.125 is an
upper bound on head aversion and may be generic dilution.

**The two stages are aligned, and the queue does it.** `p_select ~ d` +0.399 against
`p_capture ~ d` +0.091 with the queue removed; the mixture runs **1.29×** above independent
factors, so the alignment is part of what the model computes rather than a description of it
(C13, job 8899905).

## 4. The benchmark was wrong; fixed, the number is 0.883

**The right target is the frame that causes decay, not the annotated main ORF.** **ATF4 is
the case:** the prior picks the 1055-nt main ORF, the posterior flips **18×** onto a 179-nt
uORF and puts **93%** of the signal there. That is the textbook mechanism, and against the
main ORF it scores as a miss.

Scored against GENCODE's curated `nonsense_mediated_decay` biotype — a call made
independently of anything we compute (job 8900631):

| GENCODE biotype | n | prior | posterior |
|---|---|---|---|
| `nonsense_mediated_decay` | 1,099 | 0.793 | **0.883** |
| `protein_coding` | 1,285 | 0.844 | 0.591 |

**The posterior is a decay-seeking correction** — +0.090 where the annotated frame is
decay-causing, −0.253 where it is not, both gene-clustered and both excluding zero (job
8900950).

*Levels here are unweighted means over a stratified bank. For this row that is bounded at
±0.095 worst case and far less in practice; the `protein_coding` row is not — see appendix.*

**The floor under 0.883 is 0.460** — longest ORF, on these same transcripts, headroom
**+0.423** (job 8900942). The interval measured is the prior's margin over that floor,
**+0.333, CI [+0.298, +0.367]**; the posterior's own margin over it was not tested
separately. This closes a gap that was `[unclaimed]` all day; the 0.678 figure that used to
be quoted here belongs to main-ORF recovery and does not apply.

**But this benchmark is one NMD mechanism, not both, and the same run says so.** On these
transcripts the most 5′ candidate recovers the target **0.702** of the time and the longest
ORF only **0.460** — exactly inverted from `protein_coding`, where position gets 0.493 and
length 0.958. Read together: in a GENCODE NMD-biotype transcript the decay-causing frame
**starts at the normal start codon with nothing in front of it**, and is **truncated** by
the premature stop codon so it is no longer the longest frame. That is premature stop codon
in the main frame — poison exons, retained introns, frameshifts.

**The uORF mechanism is largely absent from it.** A transcript whose decay is driven by an
upstream ORF keeps an intact main CDS and is annotated `protein_coding`. ATF4 is in the
second row, not the first. So **0.883 is the model's accuracy on premature stop codon in the
main frame**, and on the row where the uORF mechanism actually lives the posterior *loses*
0.253 to the prior — some of which is the posterior correctly leaving the main ORF, as it
does for ATF4, and some of which is it being wrong. `[unclaimed]` **Nothing separates those
two.** It is the sharpest open question in this document.

## 5. The judge

**It is handed the answer.** In the `interpretable` variant the decay head's entire
non-sequence input is one column: `n_downstream_ejc`. *Retrain item 8.* But `d ~ ejc` is only
+0.445 and `capture ~ d` survives holding that column at **+0.400**, so `d` is not a readout
of it.

**It does not read the stop codon it is anchored on** — every candidate had one by
construction, so no negative example and no gradient. *Retrain item 6.*

**It has a composition preference 3′ of the stop** surviving mass stratification: keto
1.084–1.150 across all eight bands, 5 of 5 seeds. Not established at 5′. **Bound: a single
PWM explains 1.73% of importance variance.**

## 6. How the two stages relate

**The forward separation is verified** — three encoders, and an invariance test that
scrambles the stop window and leaves `p_capture` unmoved. **The coupling is derived** —
`∂L/∂z_p_k ∝ d_k` is calculus on verified code, not an observation. **The consequence is
measured** — `capture ~ d` is +0.362 among short candidates, +0.400 holding the junction
column (**C17**, jobs 8900114 / 8900473). *Same measurement as §5's, read there
against the supplied column and here against the head.*

**But scoped.** In aggregate the separation *holds*: `p_capture ~ d` is +0.091 and the two
heads read **different bases**, agreement ~0.02 within mass band (job 8899820). It is given
back **only among short candidates**, where the head must choose among uORFs.

---

## How to read this

**Sections 1–6 are the science**, and every number in them is measured. Two `[unclaimed]`
markers flag assertions that are not — they are deliberate and greppable.

**The appendix carries provenance, retractions and the predictions this work lost.** It is
separate so the story reads, and it is not optional: a narrative that lists only what
survived cannot be checked.

---

## Not established

- **Whether the head reads the fill boundary or the start codon.** Closed to tiling.
- **Whether the model is a single junction detector.** n = 49 against a pre-registered floor
  of 50. Needs pool-scale forward passes.
- **Whether the posterior's −0.253 on `protein_coding` is error or the uORF mechanism
  working.** Leaving the main ORF is correct for an ATF4-like transcript and wrong for an
  ordinary one, and the GENCODE biotype cannot tell them apart because both are annotated
  `protein_coding`. **The sharpest open question here**, and the one that decides whether
  0.883 generalises past premature stop codon in the main frame (§4).
- **C15 — why routing is junction-biased in NMD transcripts (+0.119) and anti-biased in
  controls (−0.189).** The model cannot see labels, so sequence carries a 0.31 gap and we
  have no mechanism for it. All three windows called this the most interesting open item.
  **It should be designed expecting a confound, not a mechanism** — three claims reversed
  under conditioning on 2026-08-02 and this has the shape of a fourth. First cut: does it
  survive matching NMD and control transcripts on 5′UTR architecture (uORF count, UTR
  length, candidate count)? If it collapses, it is population composition, not routing.
- **Whether the composition signal is an encoder artifact.** No valid instrumental control
  exists for a scale-free statistic.
- **Anything about motifs.** The region caller's criterion points the wrong way.
- **Any biology claim. Zero.** The +4 stop-context bias is real in the data and absent from
  what the model reads — the only candidate, unworked.

# Appendix — provenance and corrections

*Separated from the narrative so the science reads clean. Nothing here is optional: it is
what makes sections 1–6 checkable.*

## The §3 table, and what it cost to get right

*C14. Producer `model_c16_retraction.py`, **job 8900746**, candidate filter k ≥ 4 —
the same population as job 8900685. An earlier version of this table quoted +0.452
and −0.067, which came from a k ≥ 6 run in an uncommitted temp file; the producer
now reports both filters so the difference is visible rather than resolved by
preference, and the conclusion is identical under each. Internal check: the
rank-residual partial agrees with the closed form at one covariate.*

*Also (jobs 8900643, 8900685). The head's own aversion, `p_capture ~ junction count`
−0.453, collapses to −0.009 holding length — that arm is entirely length.*

*This paragraph asserted the opposite for forty minutes.* C16 claimed routing was
junction-seeking at matched length (+0.442); it held length and not position, and is
retracted. Kept visible because the retraction table alone would not have shown that
the body of this document once said it.

*Why holding length does not hold position, measured rather than argued:*
`orf_start ~ orf_length` within transcript is only **−0.150** (n 4,815, |r| < 0.30 in 54.8%
of transcripts). The two covariates are nearly free of each other, which is why holding
either one alone leaves the other's channel fully intact.

**The producer was missing.** The retraction was first run from an uncommitted temp file:
the conclusion was committed, the script, runlog and job id were not. The interpretability
window caught it and stated it exactly — **the retraction was less well-evidenced than the
claim it retracted**, since C16 had a committed script, a runlog, a job id and predictions
registered before the run. Fixed at `b76bef8`.

**And that caused a number discrepancy.** The table read +0.452 while job 8900685 read
+0.442 for what looked like the same quantity. Cause: candidate filters k ≥ 6 against
k ≥ 4. Both are now reported by the producer; the conclusion is identical under each.

    k >= 4   marginal -0.050   length +0.447   position -0.553   both -0.070
    k >= 6   marginal -0.023   length +0.452   position -0.546   both -0.067

## The standing hazard this work generated

Adopted project-wide and now in the model repo's `CLAUDE.md` (D62), so it is recorded here
as provenance rather than as part of the story:

> **⚠ STANDING HAZARD — in this model, any candidate-level correlation is position until
> proven otherwise.** Four instances in one day: C7 (holding position *strengthened* the
> capture–junction partial), the ATF4 uORF (queue position explains 22.8%, not the 100% we
> assumed), C16 (a length partial that unmasked position), and the queue-geometry null
> (`p_select ~ ejc` positive with no model in it). Pair this with the null rule: **for a
> rank product, a monotone ordering or a normalised share, zero is not the null.**

Two of its four instances are cases where position was *refuted* rather than confirmed, and
in C7 position was *suppressing* a real association rather than manufacturing one — so the
operative half is the guardian's clause: **use a partial to explain why, never to state
behaviour.**

## Retracted, with reasons kept

| | why |
|---|---|
| keto ratio **1.148×** | no reproducible producer; the cited script cannot emit that row |
| recovery **0.941** | `astype(bool)` on a −1 sentinel → 0.885 → 0.883 against GENCODE |
| "selects premature-stop frames" | never held: −0.050 marginal, −0.070 holding length and position |
| "separation given back by the objective" as a **global** claim | true only among short candidates |
| capture sensitivity is **sparse** upstream | diffuse — 32 tiles within ~6% (job 8900420) |
| **C16 — routing junction-seeking at matched length** | held length, not position; holding both gives −0.070, and the figure is below a no-model queue null. **C14 stands.** |

**Six entries. Of those, four were caught by a reader outside the derivation asking what
a sentence rested on** — 0.941 by an arithmetic ceiling, the marginal routing claim by a
demand for the direct measurement, the objective sentence by "measured or argued?", the
sparsity prediction by someone else's tiling. **One (C16) was caught by the author, using
a rule a reader had proposed an hour earlier. One (1.148) surfaced from a cross-window
discrepancy.** **None was caught by a check.**

## §1's superseded table, and the prediction the re-score broke

**§1 was scored against the annotated main ORF until 2026-08-02**, the target §4 establishes
is wrong. Pete caught it: *"that is not necessarily the ORF that is actually being picked
under the NMD case, and that is also not what the model is trying to do."* The document had
contradicted itself two sections apart for a full day, and neither window caught it —
§1 and §4 were each checked against their own producer and never against each other.

| arm | main ORF *(superseded)* | GENCODE NMD *(current)* |
|---|---|---|
| the head's own argmax | 0.305 | 0.304 |
| the most 5′ candidate | 0.424 | **0.702** |
| the queue built from the head | 0.697 | 0.793 |

**The conclusion survived; the argument for it did not.** The gate signature got *stronger*
(+0.392 → +0.489) and the margin over position collapsed (+0.273 → +0.091). §1 had been
resting the second half of its claim on the target's weakness.

**I registered the wrong prediction, in the direction that flattered the model.** I predicted
position would *fall* below 0.424 because "on NMD transcripts the most 5′ candidate is
frequently a uORF." It rose to 0.702. The error was conflating two NMD mechanisms: that
sentence is true of uORF-mediated decay and false of the premature-stop-in-main-frame decay
that GENCODE's NMD biotype actually contains. Chasing down *why* the prediction failed is
what produced §4's scope caveat, which is the more useful result of the two.

## The bank is stratified, and it is bounded

`build_ism_subset.py` takes scarce mechanism cells whole and down-samples abundant ones,
recording a `sampling_weight`; `build_ism_bank.py` writes the consequence into the h5 —
population estimates must be reweighted. **No recovery producer applies it**, here or in the
earlier work. Pete's call was that this does not change the conclusion. **It is bounded, and
he is right about the rows this document quotes** (`model_reweight_exposure.py`, local).

Reweighting can only move a within-group mean to the extent the weights **vary within that
group**, so the question is whether the strata cut across the rows we report.

| row | weights within it | worst-case shift |
|---|---|---|
| **`nonsense_mediated_decay`** — every headline number | **98.4% at weight ≈1**, 18 transcripts heavy | **+0.095** |
| `protein_coding` — the contrast | 66.4% at weight > 10 | **+0.295** |

**The NMD-biotype row is nearly flat because GENCODE's NMD biotype is close to coextensive
with the `main_orf_stop` strata, and both of those took weight ~1.** 0.883, 0.793, 0.702,
0.460 and 0.304 are therefore safe to within a bound that is itself adversarial — it assumes
every hit lands on the 18 heavy transcripts. The realistic movement is an order smaller.

**The contrast row is not safe.** 63.9% of `protein_coding` sits in one stratum at weight
13.20, so weighted it would be largely that stratum. **Anything resting on that row needs
weights first — including the posterior's −0.253**, which §4 names as the sharpest open
question. That question is now partly an instrument question.

§4's mechanism conclusion is unaffected either way, because it is **definitional**: GENCODE
assigns `nonsense_mediated_decay` on the annotated CDS terminating prematurely, so
uORF-driven decay is annotated `protein_coding` whatever we sample.

**Two provenance gaps recorded, not closed.** `gencode_biotype_bank.tsv` has no producer —
committed in `d63b5bd` as data beside its consumer, so the GENCODE release and the
ID-matching rule are unrecorded, and that is upstream of every recovery number. And the
benchmark drops 2,354 of 4,999 bank transcripts as unannotated, a population
`build_ism_subset.py` calls *"where most NMD lives."*

## Predictions this document made and lost

**Capture's upstream sensitivity is sparse.** Predicted from the `vals_capture` floor
analysis — its lower quartile sits at zero while its upper exceeds decay's — that adjacent
25-nt upstream tiles would be highly variable. Registered before the tiling ran. **Refuted**
(job 8900420): 32 consecutive tiles within ~6%. The sensitivity is diffuse.

**Routing favours junction-bearing frames at matched length.** C16, +0.442. **Retracted** —
it held length and not position, and it sits below a queue-only null. Two independent routes.

**The keto composition ratio differs by set definition.** Predicted four candidate causes
for the 1.16 / 1.148 discrepancy; six set definitions returned 1.155–1.162 and all four were
refuted. The figure had no reproducible producer.

## Why §1's three numbers are not the headline

They are scored on **main-ORF recovery**, which for an NMD substrate is close to a
*disjoint* concept from the decay-causing frame — ATF4 does the textbook-correct thing and
scores as a miss (§4). That does not invalidate the comparison in §1, because all three
share the target and the claim there is about mechanism, not accuracy.

It does mean **0.697 is not this model's accuracy**, and for most of a day this document
and both windows quoted it as though it were. The benchmark was rebuilt against GENCODE's
curated `nonsense_mediated_decay` biotype, and the figure is 0.883. The earlier framing is
recorded in the retraction table.

## The same defect, four times, inside this document

**A correction that reaches one part of a document and not another** is the failure this
narrative exists to record, and it has now happened four times *in this narrative*:

1. The C16 retraction reached the claim map and the retraction table but **not §3's body** —
   for one commit the document asserted C16 and retracted it simultaneously.
2. The `+0.362 / +0.400` marker was **removed during a rewrite**, against this document's own
   instruction not to tidy markers.
3. The retraction table's count was **wrong twice** — six rows described as "four of the
   five" — until replaced with an explicit accounting, because a ratio drifts from its
   contents.
4. When the §3 table was corrected from k ≥ 6 to k ≥ 4 figures, **the prose beneath it kept
   quoting the old ones** (−0.067, +0.452, −0.543) for one commit — and then **the
   retraction table kept them for one commit more**, after the prose was fixed. One
   correction, three locations, three separate passes.

Each was caught by grepping for the *result* rather than trusting the edit. **None would
have been caught by reading.**

## Four things this document asserted and had to withdraw

Beyond the retraction table: the C16 retraction initially **did not reach the prose** — for
one commit §3 asserted C16 in its body and retracted it in its table, inside the commit
performing the retraction. Caught by grepping for the result rather than trusting the
commit. The marker on `+0.362 / +0.400` was **removed during a rewrite** against this
document's own instruction not to tidy markers, and restored. The retraction table's own
count was **wrong twice** — six rows described as "four of the five" — and is now an
explicit accounting rather than a ratio, because a ratio drifts from its contents.

## Who caught what

Of the six retractions, four were caught by a reader outside the derivation, one by the
author using a rule a reader had proposed an hour earlier, one from a cross-window
discrepancy. **None was caught by a check.** That is the case for reading narratives the way
analyses are read.
