# What the five model-free experiments found

*Model-side window, 2026-08-01. Every number below was computed by me from the input tables and is
reproducible from a committed script and its run log. Nothing here is relayed from an agent.*

Code and logs: `analysis_plans/exp0*`, `exp2_*`, `exp3_*`, `exp4_*`, `exp5_*`. Commits `30970a8`
and `ebe8582`.

---

## The short version

| | result |
|---|---|
| the junction-distance headline | **holds, exactly** — and the gradient printed beside it is a different population |
| Exp 2, stop codon identity | the published claim **does not survive** a positional control |
| Exp 3, 3'UTR motifs | all three are **composition, not motif** |
| Exp 4, junction phase | reframed as a **calibration null** — and the floor is several points, not zero |
| Exp 5, uORF restart gap | **flat**, and the negative is trustworthy |
| Exp 1, front vs back | the other window's |

**The single most consequential number of the day is the noise floor**: the machinery everyone has
been using returns 2–5 percentage points, at p as low as 0.027, on quantities that provably have no
mechanism. It resolves the +37pp junction effect and does not resolve anything at +2pp.

---

## 0. The headline reproduces, and the table beside it does not belong to it

No committed script in either repository produced 10.8% / 46.8%, and no worklog entry records it.
Recomputed with independent code, it reproduces **to the digit** — but under exactly one definition,
which none of the four documents citing it states:

> slot 0 (`orf_rank == 1`), `n_downstream_ejc == 1`, restricted to the **41,765 isoforms in the
> HDF5**, not the 42,043 in the tables.

On the full tables it is 1,397 / 2,107 at 10.7% / 46.8%. Under the **reference-CDS** anchor instead
it is 1,022 / 1,415 at **7.1% / 43.9%** — same direction, 6.1× rather than 4.3×, different
transcripts. One of the two has to be the one we report.

Coordinate convention was verified rather than assumed: recounting `#{j > orf_end}` reproduces the
file's own `n_downstream_ejc` for **28,723 of 28,723** reference-CDS slots.

### The gradient beside it is `count >= 1`, not `count == 1`

`45.8 / 61.4 / 75.5 / 76.3 / 32.8` also reproduces exactly — on `n_downstream_ejc >= 1`. The two
tables are printed as one analysis in four committed documents and cannot be: any weighted mean of
45.8 and 61.4 lies in [45.8, 61.4], and the headline's near cell is 10.8.

On the headline's own population, count held fixed:

| bin | n | NMD+ |
|---|---|---|
| ≤37 | 1,171 | 10.2% |
| 38–50 | 218 | 13.8% |
| 51–100 | 587 | 57.6% |
| 101–200 | 629 | 70.6% |
| >200 | 868 | 22.2% |

**The step is at >50 nt, not at 38–50.** In the published version that step is 45.8 → 61.4 and looks
like the action begins right at the encoder's 37 nt horizon. It doesn't — it begins just past 50,
which is the textbook rule. The architectural conclusion survives (51–200 is still past 37 nt); the
sentence stating the evidence for it does not. Mean junction count across the published gradient's
bins runs 3.49 / 4.51 / 4.83 / 4.67 / 3.41, so part of that gradient is count, not distance.

### The terminal drop is real, and the stated attribution of it is circular

> *"it falls within class (PTC+ 78.7% → 34.1%), so it is a real ceiling on the rule"*

PTC+ means junction >50 nt downstream. **Every** isoform in the 51–100, 101–200 and >200 bins
satisfies that by construction, so conditioning on it removes nothing from exactly the bins at
issue.

I tried to kill the drop and could not. It is **not** slot-0 identity: where slot 0 *is* the
reference CDS, 70.4% → 19.6%; where it is not, 70.9% → 28.6%. What does change at >200 nt: median
3'UTR **1,970 nt** against 1,041–1,252 elsewhere, median junction count **3** against 6–7, and the
sign of the 3'UTR effect **flips** (long 3'UTR is 77.5% vs 61.3% at 101–200, and 18.4% vs 37.1% at
>200).

So there is a real class of transcripts that satisfy the canonical PTC rule, have long 3'UTRs, and
**escape decay**. That is an NMD-*escape* question and it is on target — protective elements are
sequence elements. Flagged, not opened.

---

## 4. The calibration null — read this before the others

Junction phase relative to a codon boundary has no mechanism for the junction that decides NMD: the
3'UTR is not translated and has no reading frame. So rather than run it as a hypothesis, I ran it as
a measurement of what our machinery returns when there is provably nothing there.

| arm | result |
|---|---|
| null, `count == 1` stratum | −2.07 / −4.18 / −2.12 pp (p 0.44 / 0.075 / 0.28) |
| null, full cohort | **−4.11 pp, CI [−6.54, −0.59], p = 0.027** |
| positive control, the >50 nt rule, identical code path | **+37.37 pp, CI [+33.29, +41.57]** |

The machinery works. It also fires at p < 0.05 on nothing.

**Withdrawn:** my second null, CDS-internal junction phase, is not a null. Exons of length divisible
by three are exactly the ones alternative splicing can skip without shifting the frame — that is a
mechanism — and the arms differ in CDS junction count, which I did not match. It gave −4.74pp at
p = 0.001 and I am reporting it as inconclusive rather than banking it.

---

## 2. Stop codon identity — the published claim does not survive

§5.3.3 says UGA turns up more often in degraded transcripts. Crude, it does: TGA is 59.1% of
degraded against 50.2% of stable, **+8.9pp**.

TAG and TGA are anagrams — same three bases — so swapping them holds composition, GC and length
fixed by construction. Standardised over PTC status × 3'UTR quartile × GC quartile, gene-clustered
bootstrap:

    TGA − TAG  =  +2.00pp   CI [+0.84, +3.18]   p = 0.001
      in PTC+  =  +4.41pp   CI [+1.11, +8.03]   p = 0.010
      in PTC−  =  +1.04pp   CI [−0.01, +2.04]   p = 0.055

Direction confirmed, and larger where readthrough has something to act on — the pattern a
termination-efficiency mechanism predicts. I was ready to call it confirmed.

**Then the control that decides it.** The identical TGA-vs-TAG 3-mer contrast, run at 3'UTR
positions where neither string is a stop codon and no mechanism exists:

| position | TGA − TAG | 95% CI | p |
|---|---|---|---|
| **the stop codon** | **+2.00pp** | [+0.84, +3.18] | 0.001 |
| 3'UTR +25..+27 | +11.66pp | [+3.67, +23.48] | 0.007 |
| 3'UTR +50..+52 | +9.11pp | [+2.67, +16.15] | 0.007 |
| 3'UTR +100..+102 | +11.13pp | [−2.02, +19.64] | 0.104 |

The null is larger than the claim. Stated fairly: control n is ~500 against ~14,000 at the stop, so
those intervals are wide — the reading is not "the control is bigger" but **"the two are not
separable, and the stop position is not distinguished."**

The +4 base fails the same way: spread across A/C/G/T is **1.8pp at +4 against 1.7pp at the +10
control**, and A−T is −0.92pp (p = 0.180) at +4 against +1.24pp (p = 0.052) at +10.

Of the "stop identity × +4 context" convergence — 7 of 8 lenses on one side, 9 analyses in total —
**neither half survives a positional control.**

*A pre-registration that failed, recorded because I wrote it before looking:* I predicted TGA →
**less** NMD, on the grounds that leaky termination lets ribosomes read through and displace the
downstream junction complexes. The data ran the other way, which fits the mainstream slow-termination
model instead. I had the mechanism backwards.

---

## 3. 3'UTR motifs — all three are composition

Each motif was compared not against its absence but against **its own anagrams**: strings with
exactly the same bases and length, differing only in order. This is what the dinucleotide shuffle was
meant to be and could not be at 3 bases (BRIEF §6 — a shuffled 3-mer is the identity map). At 6 and
9 bases real alternatives exist, so the control is available again.

| motif | effect | its anagrams | distance |
|---|---|---|---|
| AATAAA (poly-A signal) | −4.62pp | mean −4.14, range [−6.17, −3.08] | **−0.37 sd** |
| TTATTTATT (AU-rich element) | −17.76pp | mean −16.14, range [−21.74, −10.22] | **−0.47 sd** |
| pyrimidine runs (PTBP1) | −3.84 to −10.28pp | none possible — the motif *is* a composition | — |

The plan's three-direction test is explicit that all three moving the same way is the signature of
composition rather than mechanism. All three signs are negative.

**GC-quintile matching did not catch this. The anagram control did.** Worth carrying forward, because
GC matching is load-bearing elsewhere in the design.

---

## 5. uORF restart gap — flat, and the negative is trustworthy

Gap from the last uORF's stop to the main AUG, standardised over uORF count × 5'UTR quintile:

| gap | 1–10 | 11–25 | 26–50 | 51–100 | 101–250 | >250 |
|---|---|---|---|---|---|---|
| NMD+ | 8.8% | 11.2% | 10.1% | 9.0% | 8.5% | 8.3% |

No gradient. Reinitiation distance does not predict decay, and uORF Kozak strength does not modify
it at fixed gap.

A confident negative in this project has three times turned out to be a confounder, so the same rows
and the same code were pointed at something already known: **uORF count moves 2.3% → 15.8%** (3.5% →
14.5% standardised) from 1 to ≥5 uORFs. The machinery is not simply flat.

### The thing that did move — logged, not opened

**5'UTR length at fixed uORF count.**

| uORFs | 5'UTR Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| 1 | 1.4% | 2.5% | 2.7% | 7.2% | — |
| 2 | 2.6% | 3.4% | 5.5% | 5.0% | — |
| 3 | 3.2% | 4.0% | 8.5% | 6.6% | 3.7% |
| ≥4 | 0.0% | 8.4% | 12.0% | 16.1% | 15.7% |

Both axes carry signal independently — length is not merely a proxy for uORF count. This is the
strongest unexplored thing I turned up.

---

## A landmine for Part 2, Step 1

`orf_scan_metadata.json` for these very tables records `"min_orf_length": 9`. The data it describes
has a hard floor of **33**, with not one ORF below it. **The parameter was set and did not take
effect.**

Step 1 of the retraining plan turns on lowering that floor so ATF4's textbook uORF becomes
admissible. Whoever does it must verify the **output**, not the parameter. (BRIEF §4's 33 nt claim is
right; the metadata is wrong.)

---

## 6. The threshold is a window, and that contradicts a standing decision

Opened after Track A argued I had undersold it. They were right, and their reason is the good one:
they had withdrawn an earlier version of this drop as a nearest-versus-last-junction artifact, and
**that explanation cannot apply at `count == 1`, where nearest IS last by construction.**

Splitting the >200 bin and standardising over baseline expression:

| stop → junction | n | NMD+ crude | standardised |
|---|---|---|---|
| ≤37 | 847 | 7.0% | 10.2% |
| 38–50 | 175 | 8.0% | 9.8% |
| 51–100 | 387 | 54.8% | 53.1% |
| **101–200** | 414 | 69.6% | **63.7%** |
| 201–500 | 252 | 34.1% | 32.2% |
| >500 | 362 | 9.7% | 10.0% |

It rises to a peak and comes all the way back down. Excluded as explanations:

- **Baseline expression.** Computed from DMSO samples only. The confound I expected was that lowly
  expressed isoforms fall into the negative class by lack of power — the negative class requires
  adj.P > 0.30 in all four cell types. It runs the other way: expression Q1 is 60.7% NMD+ and Q5 is
  14.8%, because substrates are degraded and therefore lowly expressed at baseline. And the far bins
  are **better** expressed (median 0.37, 0.43) than the peak bin (0.24). Restricting to the top two
  expression quintiles keeps the shape: 3.7 / 5.6 / 49.3 / 58.2 / 22.7 / 8.9.
- **Nearest vs last junction** — impossible at `count == 1`.
- **Slot-0 selection** — the slot-0 anchor gives 12.9 / 15.8 / 56.4 / 65.8 / 36.5 / 11.5.
- **Junction-to-3'-end position** — flat within each distance bin (at >500: 10.8 / 13.5 / 5.7 / 9.0
  across quartiles).
- **Junction count** — standardised over junction-count band and expression: 9.9 / 6.2 / 56.4 /
  70.7 / 45.4 / 10.6.

### What is NOT excluded, and it matters

**Junction-call quality degrades sharply with distance.** SQANTI's `all_canonical` runs
97.6 / 98.3 / 95.3 / 89.4 / **68.3** / **61.6** % across the bins, and the share of
`novel_not_in_catalog` transcripts runs 10.9 / 14.9 / 31.8 / 36.5 / **50.4** / **53.0** %. The far
bins are largely novel transcripts carrying non-canonical splice calls.

Restricted to canonical, non-RTS-flagged transcripts the shape attenuates but does not vanish:

| | ≤37 | 38–50 | 51–100 | 101–200 | 201–500 | >500 |
|---|---|---|---|---|---|---|
| all | 10.2% | 9.8% | 53.1% | 63.7% | 32.2% | 10.0% |
| **canonical, not RTS** | 9.1% | 10.4% | 53.9% | **72.1%** | **45.1%** | **17.4%** |

(Short-read support columns are unpopulated in this classification file — `min_cov` is entirely NaN
— so that filter could not be applied.)

### The conclusion, stated at the confidence it earns

- **The rise from 51–100 to 101–200 is solid.** Both bins have good junction quality (95.3% and
  89.4% canonical) and similar structural composition, and on the clean subset the rise is
  **53.9% → 72.1%, about 18 points, entirely beyond the 50 nt threshold.**
- **The decline at 201–500 is real but roughly half what the raw numbers say** — 45.1% against a
  ~72% peak once junction quality is controlled, not 32%.
- **The collapse at >500 cannot be separated from annotation quality.** 53% novel-not-in-catalog and
  38% non-canonical. I cannot distinguish "distant junctions stop triggering decay" from "distant
  junction calls in this dataset are largely wrong." Do not report it.

**This contradicts a decision on the record.** The plan states, and the brief lists as dead: *the 50
nt rule is a threshold, so there is no dose to respond to, and the question is empty even for a
perfect model.* The 51–100 versus 101–200 contrast is a dose, it is 18 points, and it is on the
cleanest data in the set.

**Consequence for Part 2 Step 1.** The fix to `n_downstream_ejc` is currently specified as applying
the 50 nt rule — i.e. replacing a thresholdless count with a thresholded one. That would still
discard the 18-point difference between a junction at 80 nt and one at 150 nt. If the feature is
being rebuilt anyway, it should carry graded distance, not a second threshold.

---

## What this implies for the plan

**The design resolves +37pp and does not resolve +2pp.**

The retraining plan's centrepiece is six checks that must all pass before a sequence feature can be
called important. Five of the six are about *what to compare against*. Today says the harder problem
is upstream of that: at the effect sizes single sequence elements produce here, stratified comparison
of observational transcripts cannot separate signal from its own noise floor, however carefully the
strata are built.

That is an argument for the perturbation arm of the plan — implanting a motif into sequences that
lack it and re-encoding — rather than for more careful matching. A perturbation has a within-molecule
control; a matched comparison never does.

It is **not** an argument against the retraining. The one effect that is far above the floor is
exactly the one the model was never given and never learned, and fixing that is Step 1.
