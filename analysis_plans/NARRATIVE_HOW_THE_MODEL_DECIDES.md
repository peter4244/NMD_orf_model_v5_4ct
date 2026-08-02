# How the model decides

**Owner: model window.** Every number here has a producer and a job id; the claim→code map
is `CAPTURE_HEAD_STORY.md` (C1–C22). The corrections this document went through are in the
git history rather than in your way. `[unclaimed]` marks an assertion with no measurement
behind it — deliberate, and greppable.

---

## The question

The model predicts decay in two stages: **pick** which of ~15 candidate reading frames
matters, then **judge** whether that frame triggers decay — `P(NMD) = Σ_k p_select_k · d_k`.

Does the picking stage model ribosome initiation, or does it pick whichever frame best
explains the label?

**Neither — and the reason is that it does not have one answer.** The picker behaves
differently depending on what it is choosing between, and that split explains almost
everything below, including why our headline number is measured on the half where the model
is least interesting.

---

## 1. The picker gates a queue; it does not rank

`p_select_k = p_capture_k × Π_{j<k}(1 − p_capture_j)`, candidates ordered 5′→3′. A frame
wins by being competent **and** by everything upstream being judged incompetent. That is
leaky scanning, written as arithmetic.

Three ways of choosing, scored on GENCODE's own `nonsense_mediated_decay` call — where the
annotated CDS **is** the decay-causing frame (n = 1,099 in 915 genes; job 8900942):

| | |
|---|---|
| the head's own argmax | 0.304 |
| the most 5′ candidate | 0.702 |
| **the queue built from that same head** | **0.793** |

**The head's raw preference is nearly worthless. Routed through the queue it is worth
+0.489** — CI [+0.455, +0.522], winning 546 transcripts and losing 9.

The head is not being overruled by the queue; it *is* the queue's only input. So its job is
not to score the right frame highly — it is to be **quiet at the wrong ones**, and let the
product turn a run of vetoes into a choice. A gate, not a ranker.

**Its margin over pure position is thin: +0.091**, CI [+0.053, +0.129], 241 won against 141
lost (job 8900950). Taking the first candidate and ignoring the model gets 0.702. Everything
the picker knows is worth a tenth on top of candidate order.

## 2. What the picker reads

**A real initiation-context signal, and it is positional.** Perturbing the 25 bases
immediately 5′ of the start codon moves the head's logit more than anything else in its
900-base upstream window — 0.413 / 0.283 / 0.241 / 0.275 across four ORF-length bands, **at
the same position in every band**, in a tile 99.6% filled throughout so no fill artifact
explains it (jobs 8900209, 8900420). That position is where Kozak context sits. **This is
the strongest candidate in this document for a statement about biology**, and the one result
that got stronger under scrutiny rather than dissolving.

`[unclaimed]` **That the head reads Kozak *content* is not established** — only that it
responds at that *position*. The encoder's receptive field is ~42 nt against a ~10 nt motif,
so the instrument may not be able to separate the two.

**Against a diffuse background.** Across the upstream window, 32 consecutive tiles differ
from their neighbours by ~6% and never by more than 50%: a broad low response with one sharp
peak, not a scatter of hot spots.

**And a second component that tracks ORF length.** The response peak moves with length (−13,
+12, +37, +87). **Part of that route is ours, not the model's** — the ATG window fills to
`min(100, length/2)`, so for any ORF under 200 nt *where the fill stops encodes ORF length
exactly*, and fill saturation alone is a **47× marker** for the real ORF (11.1% against
0.24%). Whether the head exploits it is `[unclaimed]`, and the question is closed to tiling
because downstream tiles already sit below the receptive field. *Retrain item 3.*

**The head runs both components, and which one prevails depends on the candidate:**

| | position | fill confound | dominates for |
|---|---|---|---|
| initiation-proximal | fixed at −13, in all four bands | **none** — 99.6% filled | **short ORFs**, 0.413 against 0.220 |
| length-tracking | moves, +12 → +87 | unresolved | **long ORFs**, 0.590 against 0.275 |

**This is the same regime split §3 finds by a different route.** Where the ORF is long, the
head reads length. Where it is short — where uORFs compete — the clean initiation-proximal
signal is the larger of the two. Two instruments, one boundary.

## 3. What the picker is actually selecting for

**In aggregate, length — and length is not what initiation biology would nominate. But the
criterion changes with the regime, and the aggregate hides that.**

`p_capture ~ ORF length` is **+0.760** (C8, job 8899132), the strongest association measured
anywhere in this work. Among candidates under 200 nt — 69% of them, median 81 nt — it is
+0.429 against **+0.061** for Kozak. **Seven to one.**

The consequence lands exactly where you would predict: reference ORFs that are short are
recovered **0.276** of the time against **0.735** for long ones. **The head fails precisely
where initiation context is the thing that decides.**

**What it is not selecting for: premature stops.**

| `p_select ~ junction count`, conditioning on | median |
|---|---|
| nothing (marginal) | **−0.050** |
| ORF length only | +0.447 |
| position only | −0.553 |
| **length and position together** | **−0.070** |

*Job 8900746.* Length and position mask each other in opposite directions, so holding either
alone manufactures a signal that is not there; holding both returns −0.070, agreeing with the
marginal. The head's own apparent aversion, −0.453, is entirely length — it collapses to
−0.009 when you hold it.

⇒ **The founding hypothesis — that the model picks frames because they carry a premature
stop — does not hold at the routing step.** A queue with **no model in it** scores +0.568
holding length against the model's +0.447, so the model routes toward junction-bearing frames
*less* than pure ordering does. Zero was never the right reference. *Commit f523f72; the
degenerate null maximises queue influence, so the 0.125 deficit is an upper bound.*

**But the aggregate is hiding a sign flip.** Among candidates under 200 nt — where uORFs
compete, and where uORF-driven NMD arises — the head's criterion inverts (job 8900229,
within transcript):

| among short candidates | |
|---|---|
| `p_capture ~ junction count` | **+0.100** — against **−0.453** over all candidates |
| `p_capture ~ d` | **+0.362** |
| top-scoring candidate carries a downstream junction | **76.3%** |
| top-scoring candidate is the 5′-most | 22.8% |

**Over all candidates the head appears to avoid junction-bearing frames, and that arm is
entirely length. Within the class where it has a real choice, it prefers them** — and
predicts the judge at +0.362.

**So the criterion is regime-dependent**: length where the contest is a main ORF against
background, **decay-relevance where the contest is among uORFs**. That settles ATF4 as well —
the uORF wins because it is decay-relevant, not because it is upstream. Position accounts for
22.8% of these cases.

`[unclaimed]` **Both regime numbers are measured on `p_capture`, the head — not on
`p_select`, the picker.** Neither `p_select ~ length` nor `p_select ~ junction count` among
short candidates has been run, so the step from head to picker rides on the queue's
construction rather than on a measurement. These are the cheapest things outstanding and this
section needs them.

## 4. The benchmark was wrong; fixed, the number is 0.883

**The right target is the frame that causes decay, not the annotated main ORF.** ATF4 is the
case: the prior picks the 1055-nt main ORF, the posterior flips **18×** onto a 179-nt uORF
and puts **93%** of the signal there — the textbook mechanism — and against the main ORF
that scores as a **miss**.

Scored against GENCODE's curated biotype, a call made independently of anything we compute
(job 8900631):

| GENCODE biotype | n | prior | posterior |
|---|---|---|---|
| `nonsense_mediated_decay` | 1,099 | 0.793 | **0.883** |
| `protein_coding` | 1,285 | 0.844 | 0.591 |

**The posterior is a decay-seeking correction** — +0.090 where the annotated frame causes
decay, −0.253 where it does not, both gene-clustered and both excluding zero (job 8900950).
**The floor under it is 0.460**, longest-ORF on the same transcripts, so the headroom is
real: this is not an elaborate length heuristic.

**But 0.883 is one NMD mechanism, not both.** On these transcripts position recovers 0.702
and length only 0.460 — inverted from `protein_coding` at 0.493 and 0.958. So the
decay-causing frame here **starts at the normal start codon** and is **truncated** by the
premature stop: poison exons, retained introns, frameshifts. A transcript whose decay is
uORF-driven keeps an intact CDS and is annotated `protein_coding`. **ATF4 is in the second
row.** The mechanism the model demonstrably nails is not in the benchmark that scores it.

**And the mechanism it does contain is the long-ORF regime — the one where the model is
least interesting.** A premature-stop-in-main-frame substrate has a single long annotated
CDS beginning at the normal start codon, which is exactly why position alone scores 0.702
here. That is the regime where §2 and §3 both find the head reading **length** and the queue
doing most of the work. **So the benchmark scores the model where it behaves like a
length-and-position heuristic, and does not score it at all where it reads initiation
context and decay-relevance.** The 0.460 floor and the thin +0.091 margin over position are
measurements of the model at its most mechanical.

## 5. The judge

**It is handed the answer, and is not simply repeating it.** In the `interpretable` variant
the decay head's entire non-sequence input is one column, `n_downstream_ejc`. But `d ~ ejc`
is only +0.445, and `capture ~ d` survives holding that column at **+0.400** — so `d` is not
a readout of the feature it was given. *Retrain item 8.*

**It never learned to read the stop codon it is anchored on.** Every candidate had one by
construction, so there was no negative example and no gradient. *Retrain item 6.*

**It has a composition preference 3′ of the stop, and it is specific.** Taking the top 1% of
positions per transcript by decay sensitivity (job 8896445, 4,999 transcripts, 110,631
elevated positions):

| | |
|---|---|
| keto (G+T) | **1.156×** |
| amino (A+C) | 0.845× |
| **G+C** | **1.004×** |

**The enrichment is keto versus amino and it is orthogonal to GC** — G+C is flat to four
parts in a thousand, so positions are not selected by their own GC status, and since A+T and
G+C are complementary there is no AU bias either.

**The centre and the flanks are different signatures.** At the sensitive position itself the
signal is keto and GC-neutral. From ±2 to ±8 it becomes **uridine-rich and cytosine-poor** —
U runs 1.18–1.41 while **A stays flat at 0.98–1.09**. So it is **uridine-rich, not AU-rich,
and the canonical ARE reading (AUUUA, A and U together) is not what this is.**

It survives mass stratification — keto 1.084–1.150 across all eight mass bands, 5 of 5
seeds — and four other controls: a region-matched background inside the 3′UTR (so the
enrichment is *within* the 3′UTR, not because of it), the stop-anchor definition, five
independently trained seeds (mean pairwise r 0.753 over all 1,024 5-mers), and a PWM held
out on disjoint genes that scores **r = 0.1316 held out against 0.1316 in-sample** — the
sequence→importance relationship generalises.

**Bounds.** A single PWM at width 9 explains **1.73%** of importance variance at best, so
whatever this is, **it is not one motif**. Nothing here has been compared to a binding
database, and whether the uridine enrichment causes the model's sensitivity or merely
accompanies it is untested.

*This is a claim about the model, not about transcript biology: `vals_decay` is the trained
network's own sensitivity, so every number above says what this network responds to. The
biology-shaped reading is the more interesting one and therefore the easier one to drift
into.*

## 6. How the two stages relate

**Forward, they are separate — verified, not assumed.** Three encoders, and an invariance
test that scrambles the stop window and leaves `p_capture` unmoved. In aggregate the
separation holds: `p_capture ~ d` is +0.091 and the two heads respond to **different bases**,
agreement ~0.02 within mass band (job 8899820).

**Backward, the loss couples them.** BCE on the *product* means `∂L/∂z_p_k ∝ d_k` — the
picker's gradient is scaled by the judge's output. That is calculus on verified code.

**And the consequence is measured.** `capture ~ d` is +0.362 among short candidates and
**+0.400** holding the junction column (C17, jobs 8900114, 8900473). **The picker became
decay-predictive within its information limit** — and only among short candidates, which is
exactly where it has room to choose, because that is where uORFs compete.

**The alignment is something the model computes, not a description of it.** `p_select ~ d`
is +0.399 against `p_capture ~ d` +0.091 with the queue removed, and the mixture runs
**1.29×** above independent factors (C13, job 8899905). The queue is doing the aligning.

---

## What this adds up to

**The model has two modes, and the boundary is ORF length.**

Where the contest is a long main ORF against background, it reads **length** — seven times
more than initiation context, and part of that length signal is our own window boundary
rather than sequence. Position does most of the remaining work: taking the first candidate
and ignoring the model scores 0.702, and everything the picker knows adds **+0.091** on top.

Where the contest is **among short ORFs** — where uORFs compete, and where uORF-driven decay
actually comes from — it reads something else. Its junction association flips from −0.453 to
**+0.100**, it predicts the judge at **+0.362**, and 76.3% of its top picks carry a
downstream junction while only 22.8% are simply the first candidate. Perturbation finds the
same boundary independently: the clean initiation-proximal signal at −13 dominates for short
ORFs, length-tracking for long. **Two instruments, one line.**

**The architecture is what makes the first mode work and the second mode possible.** The
picker's own preference is nearly worthless — 0.304 — and becomes 0.793 only through
stick-breaking, which turns a run of vetoes into a choice. That is a gate, not a ranker, and
gating is precisely what leaky scanning past a uORF requires. Meanwhile the judge, handed
the junction count outright, is not a readout of it, never learned to read stop codons
because we gave it no negative examples, and responds at keto bases in a uridine-rich,
GC-neutral window that is not a motif and not an ARE.

**And the two stages do not stay separate.** Forward they are — different encoders, different
bases, agreement ~0.02. Backward the loss couples them, because BCE on a product scales the
picker's gradient by the judge's output. The coupling is measurable exactly where the picker
has a real choice to make: among short candidates, `capture ~ d` is +0.400 holding the
junction column. **The regime where the model is interesting is the regime where its two
halves have merged.**

---

**The sting is in the benchmark.** 0.883 against GENCODE's own NMD call, on a floor of 0.460,
is a sound number. But GENCODE assigns that biotype on the annotated CDS terminating
prematurely — a single long ORF starting at the normal start codon. **That is the long-ORF
regime: the half where the model behaves like a length-and-position heuristic.** The uORF
mechanism, where it reads initiation context and decay-relevance and where it gets ATF4
right by 18× with 93% of the signal on the correct 179-nt frame, is annotated
`protein_coding` and sits in the row we treat as the contrast.

**We measured the model where it is least interesting, and scored it there.** What it does in
the other regime is the better result and it is currently unbenchmarked.

## Not established

- **Whether the head reads Kozak content or only responds at that position.** §2. Receptive
  field ~42 nt against a ~10 nt motif. **The most promising open item.**
- **Whether the head reads the fill boundary or what is inside it.** Closed to tiling; needs
  a retrain varying window extent. *Retrain item 7.*
- **`p_select ~ length`, and `p_select ~ junction count` among short candidates.** §3 — the
  regime flip is measured on the *head*; whether it survives the queue into the *picker* is
  not. The cheapest things outstanding.
- **Whether the posterior's −0.253 on `protein_coding` is error or the uORF mechanism
  working.** Leaving the main ORF is correct for an ATF4-like transcript and wrong for an
  ordinary one, and the biotype cannot separate them. **This decides whether 0.883
  generalises.**
- **C15 — routing is junction-biased +0.119 in NMD transcripts and −0.189 in controls.** The
  model cannot see labels, so sequence carries a 0.31 gap. **Design it expecting a
  confound** — first cut is whether it survives matching the groups on 5′UTR architecture.
- **Whether the model is a single junction detector.** n = 49 against a pre-registered floor
  of 50.
- **Anything about motifs.** The region caller's success criterion points the wrong way.
- **Any biology claim except §2's positional result.** The +4 stop-context bias is real in
  the data and absent from what the model reads — the only other candidate, unworked.

*One instrument caveat travels with every level above. The ISM bank is stratified and no
recovery producer applies `sampling_weight`. On the `nonsense_mediated_decay` row — where
every headline number is scored — 98.4% of transcripts sit at weight ≈1 and the worst-case
shift is +0.095, adversarially; realistically it is an order smaller. On the `protein_coding`
contrast row it is +0.295, so the −0.253 above needs weights before it is quoted. C22,
`model_reweight_exposure.py`.*
