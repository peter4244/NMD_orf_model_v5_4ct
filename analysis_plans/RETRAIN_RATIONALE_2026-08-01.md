# Plan — a model built to be read

*Model-side window, 2026-08-01. Plain English, per Pete's standing instruction.*

**What we are building:** a model that predicts NMD well enough to be worth reading, and is
arranged so that reading it names sequence features that trigger decay. Prediction is the entry
ticket. The design question throughout is not "what makes it accurate" but **"what does this force
the model to compute from sequence, and what does it let the model look up instead."**

**What this builds on**

| | |
|---|---|
| Pete's rulings | `EXPERIMENT_AND_RETRAIN_PLAN_2026-07-31.md` — keep the junction feature but fix it; no minimum ORF length; accuracy floor deferred; put the rigour on confidence that a *specific feature* matters |
| The scanning form | `nmd_lung_longread_2026/docs/SCANNING_SELECTION_SPEC_2026-07-31.md` (Track A) — ribosomes scan, each start codon captures some and the rest leak past. Pete: *"I'm pretty convinced this framing is correct."* Blockers B1–B5 are theirs |
| The architecture facts | `SEQUENCE_DISCOVERY_BRIEF.md` §3, and `model.py` read directly |
| What died and why | `SEQUENCE_DISCOVERY_BRIEF.md` §6 — do not re-propose |
| Today's measurements | `RESULTS_2026-08-01_model_free.md`, `design1_orf_pool_size_runlog.txt` |

---

## 1. What we are aiming the model at, and why that changed today

Everything at the **back** of the transcript has died or shrunk to nothing. Three 3'UTR motifs were
base composition wearing a motif's name. The +4 base failed against an equal-sized control. The stop
codon survives as an association with no mechanism that holds up — and the other window's route-2
measurement put the effect on the stop that *cannot* be triggering decay, with nothing on the one
that can.

Everything at the **front** is an order of magnitude larger:

| | odds ratio |
|---|---|
| upstream start-codon count, ≥4 vs ≤1, at fixed 5'UTR length | **6.97** [5.01, 10.58] |
| 5'UTR more than half covered by ORFs vs not at all | **5.31** [3.92, 7.44] |
| 5'UTR length, at fixed start-codon count | 3.26 [1.48, 8.31] |
| stop codon TGA vs TAG | 1.24 [1.06, 1.47] |
| every 3'UTR motif tested | ~1.0 |

And within that, the part that is a **nameable sequence element** rather than a count. Holding the
number of upstream start codons fixed and varying how many sit in strong initiation context:

| upstream AUGs | 0 strong | 1 strong | 2 strong | 3 strong |
|---|---|---|---|---|
| 2 | 1.9% | 3.3% | 3.4% | — |
| 3 | 2.3% | 4.1% | 5.3% | — |
| ≥4 | **4.8%** | 9.4% | **13.9%** | 14.1% |

Same number of start codons, ~3× difference. That is a specific motif — the bases either side of an
upstream AUG — at a specific place, with a testable prediction.

**So the model's job is to be forced to discover that, and things like it.** Not to be told.

*Owed before this is quoted anywhere: strong initiation context is purine-rich, so this contrast is
not composition-matched. It needs the anagram-style control that killed the three 3'UTR motifs.*

---

## 2. The one principle the whole design follows

**Whatever the model is handed, it will not learn.** Measured four times over, independently:

| the model never learned | because it was handed |
|---|---|
| that a junction must be >50 nt past the stop | a count of junctions at any distance |
| what makes a good start codon | a flag saying "this one is annotated" |
| which ORF the ribosome actually uses | the same flag — the answer was in slot 0 |
| how the ORFs sit relative to each other | features giving their positions directly |

So the design reduces to two lists, and getting them right matters more than any hyperparameter.

**Supplied** — solved biology, no discovery value, and soaking it up makes the residual cleaner:
- exon junction positions (already a sequence channel)
- **the junction rule, fixed**: does a junction lie more than 50 nt past this ORF's stop

**Withheld** — the discovery targets:
- `is_ref_cds`, `is_sqanti_cds` — these *are* the answer to "which ORF is real"
- `frac_start`, `frac_stop` — ORF geometry, handed over
- any Kozak score as an input feature

### One thing that is settled and should not be reopened

Pete asked how graded junction distance would help identify *other* sequence features rather than a
binary indicator. It doesn't — measured, `exp7`. Each candidate sequence feature adjusted for the
binary rule versus six distance bins: median shift 0.45pp, and for the two 5'-end features that
matter, 1% and 8% of their effect. **The junction fix stays as already agreed** — thresholdless
count becomes the 50 nt rule — and stops there. The 18-point gradient beyond 50 nt is a biology
finding to report, not a design change.

---

## 3. The four things to change, each with the measurement behind it

### 3.1 The candidate pool — the biggest single defect

The model sees **5 ORFs out of a mean of 55**, chosen by annotation priority. Measured today
against the population that matters — upstream ORFs whose own stop has a junction more than 50 nt
downstream, each one a decay trigger in its own right, 133,765 of them across 17,944 transcripts:

| admission rule | slots per transcript | of those triggers, how many the model can see |
|---|---|---|
| **current: top 5 by priority** | 5 | **10.5%** |
| top 50 by initiation score | 50 | 72.4% |
| **floor at the 1st percentile of real start codons** | **50.1** | **93.4%** |
| floor at the 5th percentile | 41.6 | 79.3% |
| floor at the 25th percentile | 21.9 | 44.4% |

Two things fall out. **A floor beats a fixed count at matched cost** — 93.4% against 72.4% at ~50
slots, because transcripts differ enormously in how many plausible start codons they carry. And
**the current model is blind to nine-tenths of the mechanism we most want it to learn.**

**Recommendation: floor at the 1st percentile of real annotated start codons (PWM ≥ −2.705).**

The reason for the *1st* percentile and not a stricter one is a trap, and it is the same shape as
one already on the dead list. **We must not admit ORFs by the thing we are trying to discover.** If
we filter on initiation strength and then ask whether the model learned initiation strength, the
weak examples were removed before it ever saw them — the answer is contaminated by construction. A
floor at the 1st percentile is defensible as *"weaker than 99% of sites biology actually uses"*
rather than as a selection on strength. Anything stricter is selecting on the discovery target.

**Also settled:** no length floor. Track A's correction stands — `findORFs(minimumLength = 9)`
counts codons excluding start and stop, so 9 gives exactly the 33 nt floor we observe. Pete's ruling
means **`minimumLength = 0`**. Anyone reading that field as nucleotides sets 9 again and gets 33
again. Dropping it adds 756,247 ORFs, a third of the new pool, and 112,497 of them are a start codon
immediately followed by a stop — start-stop elements, which are real.

**Cost: ~50 slots per transcript, roughly 29 GB of training data, ~10× the current compute.** This
does not fit on this machine and needs the cluster. I will ask before connecting.

### 3.2 The model cannot represent *where* anything is

`SequenceCNN.forward` ends in `x.max(dim=-1).values` — a maximum over the whole window. It records
*whether* a pattern appeared, never *where*. Combined with a 42-base receptive field, the model
physically cannot see a stop codon and a junction together beyond ~42 bases, and the biology's
action is at 51–200.

**Change:** keep the maximum, add a small positional summary alongside it. Cheap — on the order of
10,000 extra parameters against the current 34,050.

### 3.3 The aggregator can say *which* ORF matters but not *how many*

`AttentionAggregator` is a softmax over slots. A softmax normalises to 1, so "this transcript has
twelve upstream start codons" is not expressible — and that is half of Pete's mechanism.

**Change:** the stick-breaking form from Track A's spec. Each start codon captures a fraction of
scanning ribosomes and the rest leak past, so selection probability is the product of everything
upstream having leaked, times capture here. Both halves of the mechanism then fall out of one
computation: five weak upstream ORFs and one strong one are the same equation at different values.

**And one restriction the spec does not contain, which I think is load-bearing.** The initiation
head must see **only the start-codon window**. If it can also see the stop window or the structural
block, it can identify the real ORF from its stop context and never learn initiation at all — which
is exactly the failure the whole rebuild exists to prevent. Architecturally enforcing what each head
may look at is what makes its output readable as initiation.

Compute the product in log space; at 50 slots it underflows otherwise.

### 3.4 Vocabulary size has never been checked

64 learned sequence patterns, total, across both branches. If we intend to report "we found N
sequence features", 64 is the hard ceiling and nobody has tested whether it binds. Train at 16, 32,
64, 128 and find where performance stops improving.

---

## 4. Two models, not one

| | **the predictor** | **the one we read** |
|---|---|---|
| junction rule | supplied, fixed | supplied, fixed |
| "this is the annotated start" flags | kept | **removed** |
| ORF position features | kept | **removed** |
| how it picks the ORF | reads the flag | **must work it out from sequence** |
| what it licenses | any claim about accuracy | any claim about biology |

Plus one cheap arm without the junction rule, as a check that supplying it is not masking something
unexpected. Not the main design.

---

## 5. How we will read it — written now, before training

The failure mode this guards against is real and recent: I retracted a published claim today on the
strength of a control I had not checked, then retracted the retraction. Fixing the analysis after
seeing the answer is how that happens.

**Before any interpretation runs, these are fixed:**

1. **Every claim carries a composition-matched control.** For a motif, its own anagrams — same
   bases, same length, different order. This killed all three 3'UTR motifs today when GC-quintile
   matching had not.
2. **Every control is power-matched to its target.** Compare like sample sizes, or subsample the
   target and show the distributions overlap. An unmatched control produced a wrong retraction today
   through a 25-fold sample-size difference.
3. **Both scales, always.** Percentage points and odds ratio. A percentage-point difference cannot
   be compared across groups with different base rates, and reading one without the other cost two
   retractions in one day.
4. **Uncertainty clustered by gene**, and confirmed across all five independently trained copies.
5. **A null is a distribution, not three estimates.** Sweep many mechanism-free positions.

**The pre-registered prediction, and it can fail:** the initiation head's learned capture
probability should correlate with the Kozak PWM **without ever being supervised on it**. If it does
not, the head learned something else, however well the model predicts. Track A's check; it is the
right one.

**Two state checks:** on transcripts degraded through an upstream ORF, selection mass should sit on
an upstream ORF; on the rest, on the main ORF. And as the pool grows, the main ORF must keep
sensible mass — it is where 51.5% of all positives live, and drowning it out would look like a uORF
success while breaking the common case.

---

## 6. What would make us stop

- **Performance stops improving at 16 patterns** → the vocabulary is tiny and there are far fewer
  findable features than hoped.
- **The interpretable model cannot beat its own sequence-blanked version** → sequence carries no
  usable signal at this window size. A real result, and it stops the project rather than being
  worked around.
- **Learned capture probability does not track initiation context** → the selection head is not
  doing what its name says, and no reading of it is safe.

---

## 7. Order of work

1. **Fix `infer_uorf_attention.py` first.** It hardcodes `attn_0..attn_4` and `range(5)` at lines
   174 and 207. Above 5 slots it does not fail — it silently uses the first five and produces
   plausible numbers. This must be fixed *before* the pool changes, because nothing about its output
   would look wrong.
2. Rebuild the pool: `minimumLength = 0`, floor at the 1st percentile of annotated starts, ordered
   by position in the transcript rather than by annotation priority.
3. Architecture: positional summary, stick-breaking aggregator with the initiation head restricted
   to the start window, filter-count sweep.
4. Train both models plus the two check arms. Cluster.
5. Hand the checkpoints to the interpretation window with the contract in §5 already fixed.

**One thing that is not on this list, deliberately.** Step 1 item 3 of the current plan says to
delete 2,334 transcripts as "NMD+ by construction". That is backwards — the labels are measurements
and the *category* is what was constructed by filtering on them. They are 24.8% of all positives,
they respond in all four cell types more consistently than the rest of the positive class, and they
are enriched roughly 2× for the upstream-ORF mechanism. **Deleting them would preferentially delete
the mechanism this entire rebuild is aimed at.** Keep them; run one sensitivity arm without them,
which now has a clean meaning.
