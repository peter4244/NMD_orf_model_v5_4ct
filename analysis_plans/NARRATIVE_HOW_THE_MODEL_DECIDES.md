# How the model decides — the running narrative

**Owner: model window (storyteller), from 2026-08-02.** Drafted by the interpretability window
and handed over on Pete's instruction. Every sentence here must be traceable to a numbered claim
with a producer; where it is not, it says so.

*This document exists because a summary is where our claims go wrong. Twice on 2026-08-02 a
framing survived in prose after the measurement behind it had been retracted — once by each
window. Analyses get re-run; narratives get repeated. This is the artifact that has to be
checked like an analysis.*

**Standards cited, not restated:** `PRIMARY_DIRECTIVE.md` for what a claim needs;
`CAPTURE_HEAD_STORY.md` for the claim→code map; `RETRAIN_ARCHITECTURE_CHANGES.md` for what a
retrain should fix.

---

## The question

The model predicts NMD in two stages: **pick** which of ~19 candidate reading frames matters, then
**judge** whether it triggers decay. Pete's question, 2026-08-02: does the picking stage model
ribosome initiation, or does it pick whichever frame best explains the NMD label?

## What the picker is

**It gates a queue we imposed rather than ranking candidates.** `p_select_k = p_capture_k ×
Π_{j<k}(1 − p_capture_j)`, 5′→3′. A candidate wins by being competent *and* by everything upstream
being judged incompetent — which is what leaky scanning is.

⚠ **The numbers usually quoted for this are scored on the wrong target.** 0.305 argmax against
0.424 for position and 0.697 combined are **main-ORF recovery**, and for an NMD substrate the
operative frame is frequently not the annotated CDS. See the gold-standard block in
`CAPTURE_HEAD_STORY.md`; ATF4 scores as a miss while doing the textbook-correct thing.

## What the picker reads — two components, one clean

**Clean: it reads initiation context immediately 5′ of the start codon.** Perturbing those 25 bases
moves the head's logit more than perturbing anything else in its 900-base upstream window — 0.413 /
0.283 / 0.241 / 0.275 across ORF-length bands, **at the same position in all of them**, in a tile
that is 99.6% filled in every band and therefore carries no fill confound. That position is where
Kozak context sits. *Jobs 8900209, 8900420.*

**On a diffuse background.** 32 consecutive upstream tiles differ from their neighbours by ~6% and
never by more than 50%. Capture's upstream sensitivity is diffuse, not sparse. *Job 8900420.*

**Unresolved: a second component that tracks ORF length.** The response peak moves with length
(−13, +12, +37, +87). Whether that is the head reading our **fill boundary** or reading **whatever
is filled** cannot be separated by this design, and the question is closed to tiling — downstream
tiles are already below the ~42-position receptive field.

**And the length signal has a route that is ours, not the model's.** The ATG window's fill stops at
the ORF midpoint, so fill extent = `min(100, length/2)`; fill saturation alone is a **~47× odds
marker** for "this is the real ORF." *Retrain item 3.*

## What the judge is

**It is handed the answer.** In the `interpretable` variant the decay head's entire non-sequence
input is one column: `n_downstream_ejc`, the premature-stop indicator. *Retrain item 8.*

**It does not read the stop codon it is anchored on** — every candidate had one by construction, so
there was no negative example and no gradient. *Retrain item 6.*

**It has a real composition preference 3′ of the stop** that survives mass stratification: keto
1.084–1.150 across all eight mass bands, 5 of 5 seeds. Not established at 5′. *Job 8896969.*
**Bound:** a single PWM explains **1.73%** of importance variance.

## How the two stages relate

**The forward separation is real and the loss gives it back.** Three encoders, an invariance test,
and a comment saying the separation is what licenses reading `p_k` as initiation — and then
`P(NMD) = Σ p_k·d_k` with the loss on the product, so `∂L/∂p_k ∝ d_k`. **The head cannot observe
termination and is trained to predict it.** *Retrain item 4.*

**And it does.** `capture ~ p_decay` is **+0.362** among short candidates and **+0.400** holding the
EJC column constant — tie-proof, so not a re-derivation of the junction correlation.

**The heads do not read the same bases**: agreement ~0.02 within mass band. *Job 8899820.*

## NOT ESTABLISHED — and the first item was claimed in error

**That the model selects premature-stop-bearing frames.** `p_select ~ n_downstream_ejc` **has never
been measured.** The claim currently rests on a two-step chain (`p_select ~ d`, `d ~ ejc`), and the
`p_select ~ d` result itself is not a numbered claim with an enumeration. **One measurement would
settle it and it should be run before this sentence is used.**

**That the ordering produces the alignment.** Retracted 2026-08-02. Position was proposed as the
route twice and refuted twice — holding position, the capture-junction correlation *strengthened*
(C7); queue position explains 22.8% of the ATF4 class.

**Whether the composition signal is an encoder artifact.** No valid instrumental control exists for
a scale-free statistic.

**Anything about motifs.** The region caller does not work — its success criterion points the wrong
way and the proposed null fix was refuted.

**Any biology claim.** Zero. The +4 stop-context composition bias is real in the data and absent
from what the model reads — the only candidate, and unworked.

## The one biological case that runs end to end

**ATF4.** The prior picks the 1055-nt main ORF; the posterior flips **18×** onto a 179-nt uORF and
puts **93%** of the NMD signal there. The textbook mechanism, recovered. *Job 8900114.*

## Where the model fails, diagnostically

Reference-ORF accuracy is **0.735 when the real ORF is long and 0.276 when it is short**, and in
that short class length beats Kozak **7×**. **The head fails precisely where initiation biology says
context should decide.**
