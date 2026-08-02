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

**Claims** C1–C15 (`CAPTURE_HEAD_STORY.md`); **C16 retracted**. **Decisions** D48, D50, D55, D56.
**Findings** `FINDINGS_ORF_SCANNER_2026-08-02.md`, `FINDINGS_TILED_PERTURBATION_2026-08-02.md`,
`FINDINGS_DECAY_SEQUENCE_2026-08-02.md`. **Retrain items** 3, 4, 6, 8.
**Jobs** 8898926 · 8898939 · 8899132 · 8899353 · 8899766 · 8899820 · 8899905 · 8899965 ·
8900114 · 8900209 · 8900229 · 8900407 · 8900420 · 8900473 · 8900631 · 8900643 · 8900685.

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

So the head's job is to be *quiet* at the wrong candidates. Its own argmax scores **0.305**,
worse than "take the most 5′ candidate" at 0.424, while the queue reaches 0.697 (C2, C3).
We misread it for most of a day by looking at where it was loudest.

⚠ **Those numbers are scored on main-ORF recovery**, which for an NMD substrate is close to
a disjoint concept from the decay-causing frame. **On the question that matters — finding
the frame that causes decay — the figure is 0.883**, and §4 derives it. If you read no
further, carry 0.883 and not 0.697.

## 2. What the picker reads

**Clean, and measured: initiation context immediately 5′ of the start codon.** Perturbing
those 25 bases moves the head's logit more than anything else in its 900-base upstream
window — 0.413 / 0.283 / 0.241 / 0.275 across ORF-length bands, **at the same position in
every band**, in a tile 99.6% filled throughout and therefore carrying no fill confound.
That position is where Kozak context sits. *Interpretability window, jobs 8900209, 8900420.*

**On a diffuse background — and this refutes my own registered prediction.** I predicted
from the sparsity of `vals_capture` that adjacent upstream tiles would be highly variable.
They are not: 32 consecutive tiles differ from their neighbours by ~6% and never by more
than 50%. Capture's upstream sensitivity is **diffuse, not sparse**. *Job 8900420.*
**Prediction A refuted.**

**Unresolved: a second component tracking ORF length.** The response peak moves with length
(−13, +12, +37, +87). Whether that is the head reading **our fill boundary** or reading
**whatever happens to be filled** cannot be separated by this design, and the question is
**closed to tiling** — downstream tiles already sit below the ~42-position receptive field.

**And that length route is partly ours.** The ATG window's fill stops at the ORF midpoint,
so fill extent = `min(100, length/2)`, and fill saturation alone is a **47× marker** for
"this is the real ORF" — `P(reference | saturated)` 11.1% against 0.24%. *Verified on
`orf_pool.tsv`; retrain item 3.* `[unclaimed]` **Whether the head uses it is not
established** — the information is present and discriminative; that the head reads it has
no measurement behind it.

**What it does not read is the thing biology says decides.** Among ORFs under 200 nt — 69%
of candidates, median 81 nt — `capture ~ kozak` is +0.061 against `capture ~ length` +0.429.
Reference ORFs that are short are recovered **0.276** of the time against **0.735** for long
ones. **The head fails precisely where initiation biology says context should decide.**

## 3. What the picker is actually selecting for

| `p_select ~ junction count`, conditioning on | median |
|---|---|
| nothing (marginal) | **−0.050** |
| ORF length only | +0.452 |
| position only | −0.543 |
| **length and position together** | **−0.067** |

*C14 (jobs 8900643, 8900685). The head's own aversion, `p_capture ~ junction count`
−0.453, collapses to −0.009 holding length — that arm is entirely length.*

**Routing is indifferent to junction structure.** Length and position each mask the
other in opposite directions, so holding either alone manufactures a signal that is
not there; holding both returns −0.067, agreeing with the marginal −0.050.

⇒ **Pete's founding hypothesis does not hold at the routing step under the only valid
conditioning.** The two single partials show large signal (+0.452, −0.543) and both are
artifacts — each is the other confounder leaking. The junction preference enters at the
decay multiplication.

**And an independent second route reaches the same place.** `p_select ~ ejc` is positive
*by construction*: the queue's survival factor falls monotonically with slot, and earlier
slots carry more downstream junctions. Measured in-bank, a **queue with no model in it**
scores +0.334 raw and **+0.568** holding length, against the model's −0.050 and +0.442. So
zero was the wrong reference, and against the right one the model routes toward
junction-bearing candidates *less* than pure ordering does — a deficit of 0.125.
*Interpretability window's argument, run in-bank; producer `model_queue_null_inbank.py`,
f523f72.* **Bound:** the degenerate null maximises queue influence, so 0.125 is an upper
bound on head aversion and may be generic dilution.

*Why holding length does not hold position, measured rather than argued:*
`orf_start ~ orf_length` within transcript is only **−0.150** (n 4,815, |r| < 0.30 in 54.8%
of transcripts). The two covariates are nearly free of each other, which is why holding
either one alone leaves the other's channel fully intact.

> **⚠ STANDING HAZARD — in this model, any candidate-level correlation is position until
> proven otherwise.** Four instances in one day: C7 (holding position *strengthened* the
> capture–junction partial), the ATF4 uORF (queue position explains 22.8%, not the 100% we
> assumed), C16 (a length partial that unmasked position), and the queue-geometry null
> (`p_select ~ ejc` positive with no model in it). Pair this with the null rule: **for a
> rank product, a monotone ordering or a normalised share, zero is not the null.**

*This paragraph asserted the opposite for forty minutes.* C16 claimed routing was
junction-seeking at matched length (+0.442); it held length and not position, and is
retracted. Kept visible because the retraction table alone would not have shown that
the body of this document once said it.

**The two stages are aligned, and the queue does it.** `p_select ~ d` +0.399 against
`p_capture ~ d` +0.091 with the queue removed; the mixture runs **1.29×** above independent
factors, so the alignment is part of what the model computes rather than a description of it
(C13, job 8899905).

## 4. The benchmark was wrong; fixed, the number is 0.883

Recovery scored against the annotated CDS is the wrong target. **ATF4 proves it:** the prior
picks the 1055-nt main ORF, the posterior flips **18×** onto a 179-nt uORF and puts **93%**
of the signal there — the textbook mechanism — and it scores as a **miss**.

Against GENCODE's curated `nonsense_mediated_decay` biotype, with our EJC rule nowhere in
the target (job 8900631):

| GENCODE biotype | n | prior | posterior |
|---|---|---|---|
| `nonsense_mediated_decay` | 1,099 | 0.793 | **0.883** |
| `protein_coding` | 1,285 | 0.844 | 0.591 |

**The posterior is a decay-seeking correction** — +0.090 where the annotated frame is
decay-causing, −0.253 where it is not.

`[unclaimed]` **No heuristic baseline has been measured against 0.883.** The 0.678 figure
belongs to main-ORF recovery. This number currently has no floor under it.

## 5. The judge

**It is handed the answer.** In the `interpretable` variant the decay head's entire
non-sequence input is one column: `n_downstream_ejc`. *Retrain item 8.* But `d ~ ejc` is only
+0.445 and `capture ~ d` survives holding that column at **+0.400**, so `d` is not a readout
of it. *That +0.400 is a single measurement — `capture ~ d` among short candidates, holding
the junction column — and it is used twice: here, to show `d` is more than the column, and
in §6, to show the head became decay-predictive. One number, two readings, and both are
legitimate; flagged so it is not mistaken for two independent results.*

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
column. *Same measurement as §5's; see the note there.*

**But scoped.** In aggregate the separation *holds*: `p_capture ~ d` is +0.091 and the two
heads read **different bases**, agreement ~0.02 within mass band (job 8899820). It is given
back **only among short candidates**, where the head must choose among uORFs.

---

## Not established

- **Whether the head reads the fill boundary or the start codon.** Closed to tiling.
- **Whether the model is a single junction detector.** n = 49 against a pre-registered floor
  of 50. Needs pool-scale forward passes.
- **C15 — why routing is junction-biased in NMD transcripts (+0.119) and anti-biased in
  controls (−0.189).** The model cannot see labels, so sequence carries a 0.31 gap and
  neither window has a mechanism. **The most interesting open item.**
- **Whether the composition signal is an encoder artifact.** No valid instrumental control
  exists for a scale-free statistic.
- **Anything about motifs.** The region caller's criterion points the wrong way.
- **Any biology claim. Zero.** The +4 stop-context bias is real in the data and absent from
  what the model reads — the only candidate, unworked.

## Retracted, with reasons kept

| | why |
|---|---|
| keto ratio **1.148×** | no reproducible producer; the cited script cannot emit that row |
| recovery **0.941** | `astype(bool)` on a −1 sentinel → 0.885 → 0.883 against GENCODE |
| "selects premature-stop frames" | never held: −0.050 marginal, −0.067 holding length and position |
| "separation given back by the objective" as a **global** claim | true only among short candidates |
| capture sensitivity is **sparse** upstream | diffuse — 32 tiles within ~6% (job 8900420) |
| **C16 — routing junction-seeking at matched length (+0.442)** | held length, not position; holding both gives −0.067. **C14 stands.** |

**Six entries. Of those, four were caught by a reader outside the derivation asking what
a sentence rested on** — 0.941 by an arithmetic ceiling, the marginal routing claim by a
demand for the direct measurement, the objective sentence by "measured or argued?", the
sparsity prediction by someone else's tiling. **One (C16) was caught by the author, using
a rule a reader had proposed an hour earlier. One (1.148) surfaced from a cross-window
discrepancy.** **None was caught by a check.**
