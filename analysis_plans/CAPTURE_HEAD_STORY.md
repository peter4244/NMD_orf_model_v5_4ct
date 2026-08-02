# What the capture head is actually doing — running story and claim→code map

*Started 2026-08-02, model window. Canonical on `master`. Pete's instruction: keep
this as a coherent story AND a code-to-claims sequence, so it can be relayed to the
results window at the proper time rather than reconstructed then.*

**SCOPE: every claim below is about the model, not about transcript biology**, with
one exception marked BIOLOGY. These are read from model outputs; `vals_decay` and
`p_capture` are the network's own quantities.

---

## The story in one paragraph

A transcript carries a median of 15–17 candidate start codons. The model picks one
by a stick-breaking queue in 5′→3′ order, and the head that scores each candidate
(`p_capture`) is shown only the start window — 900 bases upstream plus 100 into the
ORF — and is architecturally blind to where translation ends and to the junction
count. It nonetheless picks the annotated start well. But its scores track ORF
**length** at +0.76, and every association it has with decay-side structure runs
through length and vanishes when length is held. The leading account is that the
head is doing **gene-finding**, possibly as simply as preferring the longest ORF —
a real and sensible signal, learned from the right window, but not the initiation
biology the head is named for.

---

## Claims, each with its producer

| # | claim | scope | evidence | producer / job |
|---|---|---|---|---|
| C1 | The capture head cannot see the stop window or the structural block; capture is computed from the ATG window alone, and the stop window and structural features (including `n_downstream_ejc`) route to decay only. | model / architecture | `model_v6.py:160-164`; assertion at `:279-285` — but that assertion runs on **random** tensors | code read |
| C2 | The model lands on the annotated start **0.697** of the time, against **0.424** for a most-5′-candidate baseline and **0.055** chance. The scanner beats position alone. | model | 3,412 transcripts with a reference candidate, 62,149 candidates | `model_capture_premise.py`, job 8898926 |
| C3 | **The head's own argmax is worse than position** — 0.305 against 0.424. It is not ranking candidates; it gates passage. Low scores upstream are how the right answer wins, which is what scanning is. | model | same run | job 8898926 |
| C4 | Capture is **not flat**: within-transcript CV median 1.23. Selection is concentrated: median normalised entropy 0.29, median max share 0.76. The stick-breaking prior is not doing the work alone. | model | same run | job 8898926 |
| C5 | Decay **does** discriminate among candidates: `p_decay` CV median 0.77, and decay at the selected candidate is sharply bimodal (deciles 0.008 / 0.033 / 0.171 / 0.807 / 0.975). | model | same run | job 8898939 |
| C6 | Capture's scores track downstream junction count at median **−0.460**, against **+0.124** for `kozak_score` — roughly four times stronger than the feature it is supposed to use. | model | within-transcript rank correlation, n=4,917 | job 8898926 |
| C7 | **Position does not mediate C6.** Holding candidate start position fixed, the partial correlation is **−0.582**, stronger than the raw −0.460. Position was suppressing the association. | model | n=4,526 | job 8898939 |
| C8 | **ORF length fully mediates C6.** `capture ~ ORF length` is **+0.760**; holding length collapses the junction association to **−0.009**. | model | n=4,917 / 4,695 | job 8899132 |
| C9 | The reading-frame channels (6–8) are written across the **entire** window, including all 900 upstream UTR positions — phase relative to a downstream AUG. So a periodic 3-cycle grid is supplied throughout the 5′UTR. | model / instrument | `data_prep.py:207-211`: `genomic_positions = arange(w_start, w_end)`, written to every filled position | code read |
| C10 | **The capture window's downstream fill extent is a deterministic function of ORF length, so C8's +0.760 has a geometric channel that is not gene-finding.** The ATG window is filled with `limit_hi = mid` where `mid = (atg_pos + stop_pos) // 2`, and the window runs 100 nt past the AUG. So downstream fill = **min(100, ORF_length / 2)**: for any ORF shorter than **200 nt**, where the fill stops encodes ORF length exactly. | model / instrument | `data_prep.py:262-274` and the clip in `encode_window_v5:141-144`; §5.3 of `SEQUENCE_ENRICHMENT_APPROACH.md` already names this leak class — *"where the downstream fill stops encodes ORF length"* — and records that two of its kind were **invisible to ablation** | code read, interpretability window |

| C11 | **The model beats "pick the longest ORF" by 1.9 points.** Longest-candidate baseline recovers the annotated start **0.678** of the time against the model's **0.697** (most-5′ 0.424, chance 0.055). A one-line heuristic reaches 97% of the model's selection accuracy. | model | 3,412 transcripts | job 8899353 |
| C12 | **C10 is a contributor, not the account.** The length association is **+0.442** among ORFs < 200 nt, where fill encodes length exactly, and **+0.200** at or above 200 nt, where fill is pinned. It weakens by more than half but does not collapse. | model / instrument | n = 4,548 / 3,777 | job 8899353 |

### C11 is the headline and it is deflating

The capture head is handed 900 bases of upstream context, 100 into the ORF, codon
phase, junction positions and rolling GC — and its contribution over *"take the
longest candidate"* is **1.9 percentage points**. Whatever it has learned, a
one-line heuristic captures nearly all of it. That is the strongest support yet for
the gene-finder account, in its most naive form.

**Stated as a limit, not a dismissal:** 0.697 against 0.678 is a real gap and
selection accuracy against the *annotated* ORF is not the only thing the head could
be for. Its low scores upstream are what let the queue work (C3), and that behaviour
is invisible to this baseline.

**AND C11 DOES NOT IDENTIFY A MECHANISM** (interpretability window, and they are
right). If the head reads the *fill boundary* rather than length-as-such, then
"prefer the longest ORF" and "read where the fill stops" make the **same prediction**
on this baseline, so it cannot separate them. C11 bounds how much the head can be
*adding* over a heuristic; it says nothing about what the head is *doing*. C12 is
what separates them, and it says: partly fill, not only fill.

**A consequence worth testing, and the artifacts already exist.** If the head's way
of saying "not this one" is reading how early the fill stops, its behaviour should
change with window size in a specific way — and `results_4ct_sweep/` holds
checkpoints at `atg1000` and `atg2000` alongside the `atg500` primary. Different
window, different fill geometry, same sequences. Not run.

### The caveat on C12, which is mine and applies to both band splits

Comparing correlations between an under-200 band and an over-200 band shares the
weakness that broke my own band split: the two bands differ in tie structure and in
how much length variation survives inside them, so a difference in correlation is
not purely a difference in mechanism. **The direction rescues it here** — range
restriction would predict a *weaker* association in the narrow low band, and we see a
*stronger* one — so C12 survives the objection that killed my version. Recorded
because the next person will hit the same shape.

## C10's falsifiable prediction, recorded before the measurement

*Written by the interpretability window before anyone splits the data, so it cannot be fitted after.*

If C8's `capture ~ ORF length = +0.760` runs through fill geometry, the association must be **carried
by ORFs shorter than ~200 nt** and must **weaken sharply above** that, because fill saturates at 100
and stops varying with length. Concretely:

- **below 200 nt** — fill boundary moves with length, so the correlation should be strong
- **above 200 nt** — fill is pinned at 100 for every candidate, so length is invisible through this
  channel and the correlation should collapse toward the level that genuine coding-likeness supports

**If it collapses:** C8 is a window-geometry artifact, "prefer the longest ORF" is not gene-finding,
and the third leak of the §5.3 class has been found. **If it persists above 200:** the geometric
channel is not the explanation, coding-likeness or another route survives, and C10 is a contributor
rather than the account.

**This also bears on the band-split error recorded below.** A split that "tested where ORF length
still has range" is the same boundary from the other side — above 200 nt, length has range in the
data and none in the fill, which would produce exactly a backwards-looking result. Worth re-reading
that failure against this prediction rather than treating it as a bad test.

**Cost: none.** It needs no new job — capture and ORF length are already in hand from job 8899132.

## ⇒ THE GOLD STANDARD IS WRONG FOR HALF THE DATA — caveat on C2, C3 and C11

*Pete, 2026-08-02. Registered against all three rather than restated in each.*

**C2, C3 and C11 all score the model on recovering the ANNOTATED REFERENCE CDS. For an
NMD substrate that is frequently the wrong ORF.** The mechanism of NMD is that a
non-canonical frame — a uORF, or a PTC-bearing frame — is the one that matters. Scoring
the model on how often it returns the annotated main ORF asks it to do the opposite of
its job on exactly the transcripts the model exists for.

**The proof is our own ATF4 result.** The posterior flips **18×** onto a 179-nt uORF and
puts **93%** of the NMD signal there — the textbook mechanism, recovered. On this
benchmark that flip counts as a **miss**, because the uORF is not the annotated CDS.
**The gold standard penalises the model for being right.**

**And the split we already have is consistent with it, filed under a different claim:**
reference-ORF recovery is **0.658 in NMD transcripts against 0.729 in controls**. The
model looks worse precisely where the annotation is least likely to be the operative
frame.

**What this does to C3 specifically, and it is the load-bearing one.** The gating
interpretation rests on the head's argmax (0.305) being *worse* than position (0.424).
But the head is measured to weight decay-relevant candidates — `capture ~ p_decay`
**+0.362** among short candidates, its top pick carrying a downstream junction 76.3% of
the time. **A head doing that will systematically diverge from the annotation on NMD
transcripts.** So part of the 0.305 may be the head working, and the comparison that
produced the gating reading penalises the behaviour we have separately measured it to
have.

**AND IT IS WORSE THAN A SPLIT CAN FIX** (Pete, sharpening the above). The reference ORF
and the NMD-causing ORF are not noisy versions of one another. For a substrate they are
**close to disjoint**, because the whole mechanism is that a non-canonical frame carries
the premature stop. So reporting by label does not rescue the benchmark; it only shows
where it stops applying.

**C2, C3 and C11 are therefore claims about MAIN-ORF RECOVERY and should be relabelled as
such rather than caveated.** "A one-line length heuristic reproduces 97% of the model's
accuracy" is a statement about finding the annotated CDS. It says **nothing** about
whether the model finds the ORF that causes decay, and we have never measured that.

**What is required.** Relabel the three claims to their actual target, report split by NMD
label so the boundary is visible, and state plainly that **no measurement of
NMD-causing-ORF recovery exists, because no target for it exists.**

**One constructive route, untested and needing a GTF we do not have locally.** GENCODE
marks some transcripts with a nonsense-mediated-decay biotype, and for those the
**annotated CDS IS the PTC-bearing frame** — so reference-ORF would be a valid target for
exactly that subset. If a workable share of our NMD-labelled transcripts are that biotype,
a real benchmark exists for them. Worth checking before anyone builds a surrogate.

**And for NMD transcripts there may be no gold standard available at all.** We have no
ribosome profiling. Defining the target as "the frame whose stop has a downstream
junction" is close to circular, since that is the property under test. **Until that is
resolved, accuracy on NMD transcripts is uninterpretable rather than merely lower**, and
no claim should rest on it.

## ⇒ RETRACTED FRAMING: "the alignment comes from the ordering, not the head"

*Pete, 2026-08-02, challenging a sentence in the interpretability window's summary. It does
not survive, and it contradicts two measurements already in this document.*

**The sentence was:** *the model selects premature-stop-bearing frames, not by looking at
stops, but because the queue's order correlates with them.* **Three problems.**

**1. The mechanism has been tested twice and refuted twice.** C7 — holding candidate start
position fixed, the capture-to-junction correlation **strengthened**, −0.460 to −0.582. The
errors list already records this: *"I proposed position as the route for C6. Measurement
refuted it."* And the ATF4 correction measured queue position at **22.8%** of cases.

**2. The evidence points the other way.** `capture ~ p_decay` is **+0.362** among short
candidates and **+0.400** holding EJC constant — the head detects decay-relevance from the
start window *directly*, which is the opposite of the retracted clause.

**3. The result the sentence rested on is not in this document.** The pair `p_select ~ d`
**+0.399** against `p_capture ~ d` **+0.091** (job 8899905) has never been written up as a
numbered claim — no enumeration, no producer row — and was carried into a summary as
though established. **That is the failure mode this project has spent two days on: a number
that lives in prose acquiring authority it never earned.** It should be recorded properly
or dropped.

**WHAT IS ACTUALLY SUPPORTED.** The head uses decay-relevant information detectable from
the start-codon window. The product correlates with decay more than the head's own scores
do — **but the cause of that gap is not established, and the one candidate cause we tested
is refuted.** A defensible restatement, if the underlying pair is recorded: *the head's
suppression decisions upstream shape which candidate survives the queue, and it makes those
decisions using decay-relevant information* — which is the head acting **through** the
queue, not the queue acting instead of the head.

**AND `p_select ~ n_downstream_ejc` HAS NEVER BEEN MEASURED DIRECTLY.** So "the model
selects premature-stop-bearing frames" currently rests on a two-step chain, `p_select ~ d`
and `d ~ ejc`, rather than on the direct link. One measurement would settle it.

## What follows from C9, and it matters for a prior observation

"ORF periodicity bleeding into the 5′UTR" is most likely **the encoding, not the
sequence.** We hand the model a perfectly periodic phase grid across the whole UTR,
so any composition preference read through those channels looks periodic upstream by
construction. Not pure artifact — the same grid lets the model spot in-frame upstream
stops and AUGs, which is legitimate — but any upstream periodicity claim must be
attributed to the channel before it is attributed to the sequence.

## The open implication (Pete, 2026-08-02) — BIOLOGY, and not yet tested

If the head is a gene-finder **and** the model correctly selects premature-stop ORFs
(which it must, to predict decay), then **NMD-susceptible ORFs must look like real
genes** to a gene-finder. That is not a necessary truth — they could have been
spurious short ORFs — so it is a biology claim derivable from a model claim. First
supporting hint: the model finds the reference ORF in 0.658 of NMD transcripts
against 0.729 of controls, so NMD ORFs are not invisible to it. **Not established.**

---

## What is NOT established, stated so it cannot be quietly assumed

- ~~Whether "gene-finding" is as simple as "prefer the longest ORF."~~ **ANSWERED
  by C11: very nearly yes.** 0.678 against 0.697. What remains open is what the
  residual 1.9 points consists of, and whether the head's *upstream silence* (C3)
  is doing work this baseline cannot see.
- **Whether the head reads UTR sequence or ORF sequence.** Pete's framing of the key
  determination. `vals_capture` exists in the banks and would answer it directly:
  is capture's sensitivity concentrated in the 900 upstream positions or the 100
  in-ORF ones? Not run.
- **Whether codon phase is used at all.** The interpretability window's proposed
  test: rotate channels 6–8 by one position. Fill mask stays bit-identical, only
  codon alignment changes — a conditioning test rather than an ablation, which
  matters because §5.3 records that both known window leaks were invisible to
  ablation. Not run.
- **C1's assertion has never been run on real inputs.** It is a unit test on random
  tensors. C6–C8 are consistent with the invariance holding, since the whole
  association is mediated, but that is inference rather than measurement.

## Errors made and corrected in this thread, kept because they shaped the result

1. I proposed **position** as the route for C6. Measurement refuted it — the
   association strengthened when position was held (C7).
2. I predicted the junction association would be **strong for ORFs ≤100 nt** and
   weak above, on the reasoning that a short ORF's in-window portion runs past its
   stop. Measured +0.176 and −0.490 — backwards. The split does not test that
   mechanism; it tests where ORF length still has range, so it re-expressed C8
   rather than probing its cause. A badly designed test, not a surprising world.
3. The interpretability window argues my band-split failure was not a design error
   but the same 200 nt boundary seen from the other side. **I do not accept the
   rescue.** My split was at 100 nt and my `>100` band mixes the fill-encodes regime
   (100–200) with the fill-pinned regime (>200), so it cannot test C10 cleanly in
   either direction, and it measured `capture ~ EJC` rather than `capture ~ length`.
   Recorded because a generous reinterpretation I do not believe is worse than the
   error it excuses.
4. I cited `model.py` for architecture earlier in the day. It is the superseded v5
   file; the current model is `model_v6.py`. Caught by the interpretability window.


---

## The intervention we cannot run, and the retrain that would make it free

**The design.** C12's boundary sits at 200 nt *because* the ATG window's downstream
extent is 100 — fill = min(100, length/2). So a model trained with a different
extent must show the boundary at **exactly twice the new extent**. A numeric
prediction, at a specific location, registered before the run, that retraining noise
cannot manufacture. It is the only route from observation to intervention available
to this thread.

**Why it is dead, on evidence rather than inference.**

| artifact | date | size | family |
|---|---|---|---|
| `results_4ct_sweep/best_model_atg1000_*` | Jul 30 | 453,130 B | v5 |
| `results_4ct/best_model_atg500_stop500.pt` | Jun 5 | 453,242 B | v5 |
| `runs/interp_c32_b8_s100/best.pt` — **what every claim here is about** | Aug 1 | **281,978 B** | v6 |

The sweep checkpoints carry essentially v5's parameter count and are 1.6× larger
than v6. They are a **different architecture**, not a different window on the same
one — so comparing across them varies the tensor builder, the candidate set (fixed
5 against ragged ~19) and the architecture at once, and the moving-boundary
prediction has no purchase. Naming and the absence of `.h5` files in the sweep
directory were suggestive; the parameter count settles it.

`build_tensor.py:47` also hardcodes `ATG_LEFT, ATG_RIGHT = 900, 100` as module
constants, and its argparse exposes no window parameter, so **every tensor the
current builder produces has the same extent**. There is nothing to compare even in
principle.

**⇒ A DESIGN REQUIREMENT FOR THE DEFERRED RETRAIN, not a dead experiment.**
*Filed as item 7 of [`RETRAIN_ARCHITECTURE_CHANGES.md`](../RETRAIN_ARCHITECTURE_CHANGES.md), which is the single copy of the retrain list and is pointed at from `03_train.py`, `train_v6.py` and `CLAUDE.md`. Keep the reasoning here; keep the requirement there.*
 Pete has
deferred retraining and re-architecture behind infrastructure. **If that retrain
varies the downstream extent across even two settings, this test comes free** — and
it is the only way to convert the geometric account from "the boundary lands at
twice the encoded extent" into "the boundary *moves when we move the extent*."
Recorded here so the requirement reaches the retrain rather than being rediscovered
after it.

**What the write-up must therefore say:** the geometric account is supported by
where the boundary falls and **has not been tested by intervention, because no model
exists that varies the quantity.** A named gap, not a hedge.
