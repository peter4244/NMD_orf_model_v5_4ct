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
| Exp 2, stop codon identity | **the published claim SURVIVES** at about a quarter its crude size — see the retraction below |
| Exp 3, 3'UTR motifs | all three are **composition, not motif** |
| Exp 4, junction phase | the null arms are **not significant** once the estimator is fixed |
| Exp 5, uORF restart gap | **flat**, and the negative is trustworthy |
| Exp 6, the >200 nt window | there **is** a dose beyond the 50 nt threshold, and it is ~18 points |
| Exp 1, front vs back | the other window's |

> ## RETRACTION, 2026-08-01
>
> An earlier version of this document concluded that the §5.3.3 stop-codon claim does not survive a
> positional control. **That was wrong and the conclusion is withdrawn.** Pete pushed back on it and
> he was right.
>
> The error was in my estimator, not in the biology. `standardised()` selected which strata
> contribute **separately for each arm** of a contrast. When one arm is small, only a few strata
> clear the minimum cell size, the two arms end up averaged over *different* strata, and the estimate
> is biased upward — badly.
>
> **The proof is direct.** Subsample the stop-codon contrast — where the answer is known to be
> +1.80pp — down to a control position's sample size (509 TGA / 246 TAG) and run the identical
> estimator 500 times. It returns a **median of +14.04pp**. The control positions returned +13.40pp.
> They were never measuring a different thing; they were measuring the *same* thing with 25× less
> data through a biased estimator.
>
> Fixing the estimator so a stratum contributes only when **both** arms clear the threshold, and
> running **99** control positions instead of 3, gives a null centred on **+0.01pp** (IQR
> [−1.58, +2.28]) — which is what a null should look like. The stop codon gives +1.80pp with all
> 32 of 32 strata contributing in both arms, where the two estimators agree exactly.
>
> The same defect inflated Experiment 4's "noise floor". Every null arm loses significance under the
> corrected estimator (below), so the claim that this design "cannot resolve +2pp" is withdrawn too.
>
> Scripts: `exp2b_control_validity.py`, `exp4b_recheck_nulls.py`.

---

## 0. The headline reproduces, and the table beside it does not belong to it

No committed script in either repository produced 10.8% / 46.8%, and no worklog entry records it.
Recomputed with independent code, it reproduces **to the digit** — but under exactly one definition,
which none of the four documents citing it states:

> slot 0 (`orf_rank == 0` — the column is ZERO-based; earlier prose of mine said `== 1`, though the
> code used `.min()` and was right), `n_downstream_ejc == 1`, restricted to the **41,765 isoforms in the
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

## 4. The calibration null, and what it was worth once the estimator was fixed

Junction phase relative to a codon boundary has no mechanism for the junction that decides NMD: the
3'UTR is not translated and has no reading frame. So rather than run it as a hypothesis, I ran it as
a measurement of what our machinery returns when there is provably nothing there.

| arm | separate-stratum (the broken estimator) | **common-stratum (corrected)** |
|---|---|---|
| null, `count == 1`, phase 0−1 | −2.07 pp, p = 0.440 | −2.20 pp, p = 0.383 |
| null, `count == 1`, phase 0−2 | −4.18 pp, p = 0.087 | −4.18 pp, p = 0.063 |
| null, full cohort, phase 0−1 | −4.11 pp, **p = 0.012** | −2.55 pp, p = 0.099 |
| null, full cohort, phase 0−2 | −3.00 pp, p = 0.060 | −2.28 pp, p = 0.112 |
| withdrawn null 2, all − none | −4.74 pp, **p = 0.001** | −1.70 pp, p = 0.123 |
| withdrawn null 2, + CDS-junction count matched | −2.29 pp, p = 0.237 | −0.20 pp, p = 0.907 |
| **positive control**, the >50 nt rule | **+37.37 pp, p = 0.001** | **+37.37 pp, p = 0.001** |

**Under the corrected estimator not one null arm reaches significance, and the positive control is
unchanged to two decimals.** The "floor of 2–5 points at p = 0.001" was a property of my estimator,
not of the design.

The largest surviving null point estimate is −4.18 pp at p = 0.063, on a comparison with n ≈ 800 per
arm. That is imprecision, not bias — and the 99-position null in `exp2b` puts the centre at
**+0.01 pp**, which is the right way to read it.

**Withdrawn:** my second null, CDS-internal junction phase, is not a null. Exons of length divisible
by three are exactly the ones alternative splicing can skip without shifting the frame — that is a
mechanism — and the arms differ in CDS junction count, which I did not match. It gave −4.74pp at
p = 0.001 and I am reporting it as inconclusive rather than banking it.

---

## 2. Stop codon identity — the published claim survives, at a quarter its crude size

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

**The control I first ran, and why it was worthless.** I ran the identical contrast at three 3'UTR
positions, got +11.66 / +9.11 / +11.13 pp, and concluded the stop result did not stand outside its
own null. Three defects, none of which I checked:

1. **Three estimates are not a distribution.** Their CIs were [+3.67, +23.48], [+2.67, +16.15] and
   [−2.02, +19.64].
2. **The sample sizes are not comparable.** Every transcript carries a stop codon, so TGA n = 14,477
   and TAG n = 5,818. A specific 3-mer at a fixed 3'UTR offset appears at roughly the 1/64 chance
   rate: n ≈ 500 and 220.
3. **The estimands differ.** My estimator chose contributing strata *per arm*. At the stop all 32
   strata contribute; at a control position the median is **10 of 32**, and the two arms are averaged
   over different subsets.

**Defect 3 causes an upward bias, and the size of it is measurable.** Subsample the stop-codon
contrast — true value +1.80 pp — to 509 TGA / 246 TAG and run the same estimator 500 times:

    stop contrast at control sample size   median +14.04pp   sd 3.69
    control positions at native size       median +13.40pp   sd 5.47

The controls were measuring the same thing with 25× less data through a biased estimator.

**The control done properly.** 99 positions from +4 to +300 nt, and a corrected estimator where a
stratum contributes only if **both** arms clear the cell minimum:

| estimator | null median | IQR | 2.5–97.5 pct | stop codon |
|---|---|---|---|---|
| separate-stratum (broken) | +13.40pp | [+9.69, +17.42] | [+3.04, +24.79] | +1.80pp |
| **common-stratum (correct)** | **+0.01pp** | [−1.58, +2.28] | [−5.30, +6.62] | **+1.80pp** |

The corrected null is centred on zero, which is what a null is. At the stop the two estimators agree
exactly (+1.80 and +1.80, 32 of 32 strata) because both arms are large — the bias is absent
precisely where the claim lives.

**So the claim stands.** Direction as published, magnitude about a quarter of the crude contrast
after matching. The mechanistic gradient also holds under both estimators, with all 16 strata
populated in each arm:

| | n TGA | n TAG | crude | separate | **common** |
|---|---|---|---|---|---|
| PTC+ | 4,017 | 1,282 | +4.32pp | +4.50pp | **+4.50pp** |
| PTC− | 8,780 | 3,975 | +0.71pp | +0.68pp | **+0.68pp** |

Six-fold larger where a downstream junction exists for termination efficiency to matter at. That is
the pattern the mechanism predicts and it is not something the matching could manufacture.

*A pre-registration that failed, kept because I wrote it before looking:* I predicted TGA → **less**
NMD, on the grounds that leaky termination lets ribosomes read through and displace downstream
junction complexes. The data ran the other way, which is the mainstream slow-termination model. I
had the mechanism backwards, and my wrong prediction is part of why I was too willing to believe the
bad control.

**The +4 base is a different matter and it still fails.** Spread across A/C/G/T is 1.8pp at +4
against **1.7pp at the +10 control**, and A−T is −0.92pp (p = 0.180) at +4 against +1.24pp
(p = 0.052) at +10. Those two contrasts have comparable sample sizes — every transcript has a base at
+4 and at +10 — so defect 2 does not apply and the comparison is fair. **Stop identity survives; +4
context does not.**

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
has a hard floor of **33**, with not one ORF below it.

**My diagnosis of this was wrong and Track A corrected it.** I wrote "the parameter was set and did
not take effect." It took effect exactly as documented. `05s_orfik_scan.R:112` calls
`ORFik::findORFs(..., minimumLength = 9)`, and ORFik's `minimumLength` is measured in **codons
excluding the start and stop**: 3 + 9x3 + 3 = **33 nt**, precisely the observed floor, with 59,425
ORFs sitting exactly on it. It is a unit confusion, not a silent failure — the metadata field is
named `min_orf_length`, which reads as nucleotides.

This changes the Step 1 fix rather than just flagging it. Pete's ruling is admission by start-codon
score alone; under the true semantics that is **`minimumLength = 0`**, which ORFik documents as
START + STOP = 6 bp — a start-stop element, which is what he described. Someone who believes the
parameter is ignored goes hunting for a bug that is not there; someone who reads the field as
nucleotides sets 9 again and gets 33 again. ATF4's uORF1 is 3 codons and is inadmissible above
`minimumLength = 1`.

The advice survives its own wrong reasoning: **verify the output floor, not the parameter.**

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

**Withdrawn: "the design resolves +37pp and does not resolve +2pp."** That was built on the inflated
noise floor and does not survive its correction. With a common-stratum estimator the null sits at
+0.01pp and a +2pp effect at n ≈ 18,000 is resolvable, with a gene-clustered interval that excludes
zero.

What replaces it is narrower and more useful:

**The binding constraint is sample size in the smaller arm, not the design.** Every contrast that
failed today failed for a reason that can be named. The three-motif result failed because the motifs
were composition — caught by anagram controls, not by matching. The +4 base failed at equal sample
size against an equal-sized control, which is a fair fight it lost. Only my original stop-codon
control failed for a reason that was my own error.

**Two things to carry into Part 2:**

1. **The six-check standard is sound, but check 3 — "against its own matched controls, not against
   zero" — needs a fourth clause: compare the control's sample size to the target's before reading
   anything into the difference.** Today that single omission produced a confident, wrong retraction
   of a published claim.

   *My first response to this was to write the rule into the code — a helper that raised when two
   arms could not be averaged over the same strata. Pete's objection was that this substitutes a rule
   for contextual judgement, and checking settles it: **the guard would not have fired on the case
   that motivated it.** The offending control positions still shared 10 of 32 strata, so it would
   have returned a number and raised nothing, while firing on legitimate small-n contrasts where
   whoever hit it would just loosen the threshold. What caught the error was noticing that n = 214
   against n = 5,818 is not like-for-like, and power-matching to check. No assertion produces that.*

   So `contrast_lib.py` reports rather than enforces: it returns **both** weightings side by side
   with the per-arm sample sizes and shared-strata counts, so a divergence is visible instead of
   latent. At the stop the two agree (+2.02 and +1.92); across 99 control positions they are +13.40
   and +0.01. That contrast is the finding, and it is information, not a verdict.

2. **Anagram controls should be standard for any motif claim.** They caught what GC-quintile
   matching missed, on all three motifs, and they cost nothing.

3. **`power_match` and `sweep_null` are the two things worth having as tools** — not because they
   enforce anything, but because they make the right check cheap. Sweeping 99 positions instead of 3
   is one call.

The perturbation arm is still worth building — a within-molecule control is strictly better than a
matched one — but today is no longer an argument that matched comparison cannot work.
