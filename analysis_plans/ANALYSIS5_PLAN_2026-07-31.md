# Analysis 5 Plan — start codon context, and whether the model reads it

*Written 2026-07-31, before the production run exists. Split out of Analysis 4, which measures
where in the windows the model is sensitive; this one tests two named motif positions.*

This document specifies what is done to the data. Code is written from it.

**Claim ids are the ledger's `section.paragraph.sentence` scheme.**

---

# 1. Purpose

> We have trained a model that predicts, from transcript sequence alone, whether an isoform is
> degraded by nonsense-mediated decay. This analysis establishes two things: how accurately it does
> so on transcripts and chromosomes it never saw, and what kinds of sequence and structural
> information it relies on to do it. Both are reported with uncertainty that reflects the two ways
> the answer could have come out differently — a different training run, and a different draw of the
> reference transcripts used to compute an attribution.

**Approved by Pete, 2026-07-30.** *Amended for Analysis 4 with Pete's approval, 2026-07-31, and the
amendment carries here unchanged: this analysis measures the model's response to a sequence
perturbation rather than computing an attribution against reference transcripts, so it has no
reference draw. Its uncertainty is the training spread across members, the sampling error of the
subsample, and the position-to-position spread of its control reference.*

Kozak context is an **ORF-selection** signal rather than an NMD signal. The model must identify
which AUG the ribosome selects, because that determines where the stop codon sits and therefore
whether it is premature. Analysis 5 tests whether the model uses that context, and whether it uses
it more where the annotation features do not already supply the answer.

---

# 2. Why the operator is composition-neutral

A single-base substitution changes two things at once: the identity of the base, and the local
composition, because channel 5 is a rolling GC fraction recomputed over the window at every
perturbation (`data_prep.py:178-205`). The composition term is not a residual. It is the dominant
term, and it is present at every position.

*Measured here 2026-07-31, `atg1000_stop1000`, five members (seeds 100–500), 500 isoforms, the 37
frame-matched control positions of §5a. Each row is the same contrast applied at control positions
only — positions where no motif is hypothesized, so a reference behaving as a null is what a usable
operator looks like:*

| operator | control \|mean\| median | control \|t\| median | control \|t\| max | controls same-signed | all-5 members agree |
|---|---|---|---|---|---|
| `GC_vs_AT`  {G,C}−{A,T} | 0.00567 | 5.66 | 12.08 | 100% | 86% |
| `G_vs_notG` G−{A,C,T}   | 0.00735 | 5.79 | 37.04 | 100% | 92% |
| `purine`    {A,G}−{C,T} | 0.00348 | 3.65 | 7.13 | 100% | 65% |
| `G_vs_C`    G−C         | 0.00497 | 4.30 | 13.87 | 100% | 81% |
| **`A_vs_T`  A−T**       | **0.00111** | **1.00** | **4.03** | **73%** | **8%** |

Independent member signs would agree 6.25% of the time. **Members agree at control positions far
more often than that under every operator except `A_vs_T`**, so across-member agreement is not by
itself evidence that a position carries a motif — members share training data and architecture, and
they share systematic artifacts too. Agreement is evidence only when read against the control rate.

`A_vs_T` and `G_vs_C` hold GC fixed by construction. Only `A_vs_T` also has a directionless control
reference; `G_vs_C` carries a systematic model-wide preference for G over C at matched GC.

**So the primary operator is `A_vs_T`, `G_vs_C` is secondary, and `GC_vs_AT` is reported as the
quantity being controlled for rather than avoided.** `G_vs_notG` — the operator the Kozak +4
question is naturally posed in — is reported and interpreted only against its own control
distribution, which its table row shows is far from null.

**Every contrast is a re-cut of the same four measured values per position**, so the operator is
chosen after the forward passes, not before. The measurement records the change in predicted
log-odds for each of A/C/G/T at each position; contrast definitions consume that array. Two
operators can therefore never differ in anything but the operator.

---

# 3. Data

## 3.1 Terms

Datasets A and B are those of the preceding plans; their terms carry over. Six are new.

**Position bank.** The set of positions at which substitutions are made: the **hypothesis
positions** (Kozak −3 and +4) and **control positions** spread over ±60 of the start codon. Every
control carries the identical operator, the identical exclusions, the identical GC recomputation and
the identical population test. Only the position differs, so anything systematic — the off-manifold
contamination of a window-local substitution, a composition asymmetry, a positional prior — is
present in the reference as well as in the target.

**Control reference.** The distribution of a statistic across control positions. Every number
reported for a hypothesis position is accompanied by its percentile within this distribution.
Producer: step c. Consumer: §6 and every claim in §8.

**Reliability statistic.** The across-member `|t|` of a contrast — |mean| over the five members
divided by their standard error — and whether all five member signs agree. It measures consistency
across training runs, not isoform-level evidence. It is the headline statistic because magnitude
ranks the operators the wrong way round: *measured here, Kozak −3 has a mean effect of +0.00898
under `G_vs_notG` and +0.00262 under `A_vs_T` — 3.4× larger under the first — while against
frame-matched controls it sits at the 22nd percentile of reliability under the first and the 97th
under the second.* Selecting on magnitude selects the confounded operator.

**Composition-neutral operator.** A contrast whose two sides have equal GC content, so channel 5
moves only by the sampling of individual isoforms and not by construction. `A_vs_T` and `G_vs_C`.

**Created-motif exclusion.** A substituted cell is dropped when the substitution creates an in-frame
stop codon in the codon containing the position, or creates an AUG in any of the three 3-mers
containing it. Applied identically at every position in the bank, evaluated per isoform before any
contrast is formed. *Measured here: at −3 this drops 33% of the isoform-cells entering `A_vs_T` (n=334 of 500),
almost all of them T-cells creating TAA/TAG/TGA in the codon spanning −3…−1. Exclusion is
strongly base-asymmetric — at −3 the rates are A 0.044, C 0.000, G 0.034, T 0.288 — which is why
a contrast is formed only when BOTH sides survive intact (§5c). Guarding each side with "at least
one base survives" instead would turn `purine` into mean(A,G) − C, whose low side is 100% GC
rather than 50%: the exact confound the operator exists to avoid, on about a third of isoforms.*

**Method floor.** The effect measured when the observed base is substituted back onto itself,
through the same GC-recomputation path. *Measured here: max |effect| 9.5×10⁻⁷ to 1.4×10⁻⁶ per
member over 500 isoforms × 39 positions, from float32 reduction order rather than from the
perturbation. Effects below ~10⁻⁶ are arithmetic noise, and the floor rises with batch size —
it was 4.8×10⁻⁷ at 32 positions.* It is exactly 0.0 only at batch sizes small
enough to avoid the reordering, so it is reported per run rather than asserted.

## 3.2 Datasets

**A** = `results_4ct_dn/nmd_orf_data.h5`; **B** = the ten checkpoints, five members per
configuration. The model contract of the preceding plans applies unchanged.

**Dataset C — outputs, written to `results_interp_all/`**

| file | contents |
|---|---|
| `bank_{tag}_slot{R}_n{N}.npz` | per (member, isoform, position, base) change in predicted log-odds; eligibility; created-motif exclusions; observed base; per-slot attention weights; per-member method floor |
| `bank_contrasts_{tag}.tsv` | one row per (configuration, member, slot, position, operator, stratum): contrast, n, and the control percentile |
| `bank_summary_{tag}.json` | across-member statistics, control references, the stratified comparisons, the anchor-shift and random-weight controls |

**Three variance components, as in Analysis 4.** Training spread across the five members; sampling
error of the subsample, by nonparametric bootstrap over isoforms; and the position-to-position
spread of the control reference, which is reported as the reference distribution itself rather than
collapsed to an interval.

---

## 3.3 Mechanism classes, and their relationship to section 4

Section 4 classifies **isopairs** — an NMD isoform together with its gene-matched comparator — and
asks *why this NMD isoform differs from its comparator*. Section 5 classifies **isoforms** and asks
*what NMD-triggering feature this isoform contains*. Same biology, different question, different
unit, and a hundredfold difference in n. The classes are therefore not interchangeable, and the two
definitions that could silently drift between the sections are taken from section 4 rather than
chosen here.

**Taken from section 4** (`figures/multipanel/figure4_ptcneg_and_model/RATIONALE.md` §3):

1. **A PTC is a stop with an exon junction more than 50 nt downstream** — not "at least one
   downstream junction". *Measured here: 7.6% of slots with `n_downstream_ejc` ≥ 1 fail the 50-nt
   rule, and counting them as PTCs costs the PTC+ class 8 points of NMD+ rate.*
2. **The CDS anchor is the GENCODE-projected reference AUG, never the TD2 call.** Section 4's
   classification is 100% ref-AUG-derived for this reason. *Measured here: anchoring on
   `is_sqanti_cds` instead moves 1,556 isoforms out of PTC+ (−23%) and inflates the uORF class by
   135%, because TD2's ATG lies downstream of the reference AUG in 99.1% of occult-PTC pairs, so a
   main-ORF PTC is re-read as an upstream ORF.*

| section 4 group (isopairs, n = 1,385) | n | section 5 class (isoforms, n = 41,765) | n | NMD+ |
|---|---:|---|---:|---:|
| **NMD+/PTC+** | 1,080 | **PTC+** | 6,864 | 70.0% |
| **NMD+/PTC− ORF match** — mechanism inferred by elimination as uORF burden | 53 | **PTC− uORF-PTC** — the upstream PTC-bearing ORF detected directly | 5,913 | 12.6% |
| **NMD+/PTC− ORF diff** — extended 4 + truncated 34 | 38 | *(no counterpart; pooled into the two PTC− classes)* | — | — |
| **Ref AUG absent** — excluded from mechanism inference | 214 | **Ref AUG absent** — reported separately, never pooled | 13,223 | 25.9% |
| **Control** — gene-matched non-NMD comparators | 885 | *(no counterpart; section 5 partitions rather than contrasts)* | — | — |
| | | **PTC− no trigger** | 15,765 | 2.2% |

**The section 4 column is published-§4 scope and is NOT deposit-native.** Those counts were
recomputed by Track A from `isopair_wrapper/data_mashr`, which is the tree section 4 publishes
from; the deposit-native rebuild of the same population gives 1,578 against 1,548 for claim 4.1.2,
and that gap is claim 4.1.2's own recorded defect. The section 5 column is computed entirely from
`results_4ct_dn`. The two columns are therefore on different trees and the correspondence is
structural, not arithmetic.

**And `Ref AUG absent` is not the same object on the two sides.** Section 4's 214 pools 93 pairs
where ref-AUG tracing ran and found no usable reference AUG with 121 that produced no tracing
record at all. The section 5 class is the first kind only — a slot set with no match to the
reference isoform's projected AUG.

**Three structural differences, so the correspondence is read correctly.**

- **Label conditioning.** Section 4's group names begin "NMD+" because section 4 classifies NMD+
  pairs. The section 5 classes partition every isoform without reference to the label, so a class's
  NMD+ rate is a **result**, not part of its definition. PTC+ at 70.0% and PTC− no trigger at 2.2%
  are therefore evidence that the classification captures mechanism.
- **The ORF match / ORF diff split does not exist here.** Section 4 subdivides its PTC− pairs by
  whether the comparator encodes the reference protein, which needs an ORF-length comparison against
  the reference. `ref_cds_features.tsv` carries `ref_orf_length` and `comp_orf_length_from_ref_atg`,
  but *measured here, both are present for only 67.9% of the model universe, and the missingness is
  label-dependent — 62.5% coverage among NMD+ isoforms against 69.4% among negatives* — so
  conditioning on it biases. Until that is resolved, both section 5 PTC− classes mix section 4's
  ORF match and ORF diff.
- **`Ref AUG absent` is the one exact correspondence**, in definition and in treatment: section 4
  excludes it from mechanism inference, and section 5 reports it separately and never pools it.

**One class is a gain rather than a translation.** Section 4's cleanest mechanism claim — that the
residual PTC− NMD is uORF burden — rests on n = 53 and is reached *by elimination*: not PTC, not
3′UTR length, therefore 5′UTR features. `PTC− uORF-PTC` detects the upstream PTC-bearing ORF
directly, at n = 5,913.

**The class carries a mechanism split, not a purity check.** An out-of-frame **oORF** reads past the
main AUG in the wrong frame, so there is no reinitiation rescue; a **uORF** terminating in the 5′UTR
permits reinitiation and is the weaker trigger. *Measured here: uORF 5,287 (89.4%) at 12.6% NMD+,
oORF 626 (10.6%) at 12.8%.* An in-frame upstream ORF cannot appear at all — it terminates at the
main ORF's own stop and inherits its PTC status, and the class requires that stop to be clean;
measured, 0 of the class.

**Half the true uORF population is invisible to the model, and the invisible half is the weak
half.** *Measured here: asking the same question of the full ORFik scan rather than the 5 model
slots gives 12,010 isoforms rather than 5,913, so 6,097 (50.8% of the true set) carry an upstream
PTC-bearing ORF the model cannot see. Those isoforms are 3.9% NMD+, against 12.6% for the visible
ones and 2.2% for the no-trigger baseline.* The slot fill is Kozak-priority and the enrichment is
strong between tiers — *78.7% of selected qualifying upstream ORFs are top-tier against 7.9% of
unselected* — so the candidates it keeps are the ones plausibly initiated. Both versions are
computed and written; the gap between them is the uORF evidence the model is structurally unable
to use.

**Every result in this plan is reported per class, and `Ref AUG absent` is never pooled with the
other three.** *Producer: `build_mechanism_classes.py`, run log
`build_mechanism_classes_runlog.txt`, output `results_interp_all/mechanism_classes.{npz,tsv}`.*

---

# 4. Scientific questions

1. **Does the model use the −3 position of the start codon context?** (claim 5.3.2, legend 5.6.7)
2. **Does it use +4?** (claim 5.3.2)
3. **Is the −3 sensitivity larger where the annotation features do not already identify the start
   codon?** — the override question. Answered here as a between-isoform contrast in the native
   model, which an inference-time ablation cannot answer: *measured 2026-07-31, ablating both
   annotation flags shrinks the Kozak sensitivity to 0.58× but shrinks control positions to 0.66×,
   so the ablation measures global attention re-weighting.*

**Population.** A stratified subsample of 2,000 isoforms, stratified by label class and by PTC
status, reported as distributions and never as exemplars. Both configurations, `atg1000_stop1000`
and `atg2000_stop2000`.

**Slots.** The perturbation is applied to ORF slot 0 and the analysis is stratified by the attention
weight that slot receives. *Measured here at `atg1000_stop1000`, n=500: the slot-0 attention terciles have means 0.343, 0.709
and 0.910 — so a substitution in slot 0's window reaches most but not all of the prediction, and the
fraction it reaches varies enough between isoforms to reverse the sign of the answer (§8).*

---

# 5. Approach

### Step a — the position bank

```
BANK = hypothesis positions {-3, +4} + control positions
FRAME OFFSET of a position = (array_index - (W/2 - 1)) mod 3
CONTROLS are every position within +/-60 of the start codon whose frame offset EQUALS the
  hypothesis positions' offset, excluding the codon itself and the hypothesis positions.
  In the codon convention that is p = -6, -9, ... -60 upstream (19 positions) and
  p = +7, +10, ... +58 downstream (18), so 37 controls.
```

**Controls are selected on frame offset, not on spacing.** The created-motif exclusion depends on
where a position sits within its in-frame codon, so a mixed-offset control set carries a different
exclusion rate from the target and is not a matched reference. *Measured here: at offset 0 a T
substitution is excluded for creating a stop codon in 23.6% of isoform-cells, against 8.7% at offset
2 — a 2.7× difference in how much of one side of the contrast is removed.* Uniform 3-nt spacing does
**not** achieve this, because the codon convention has no zero and skips the codon: spacing by 3
from −60 lands on offset 0 upstream but offset 2 downstream, which is how an earlier bank ended up
with a reference that was 100% upstream and left the downstream hypothesis position with no
region-matched control at all.

*At the codon convention of §3.1 of the Analysis 4 plan, −3 and +4 both sit at offset 0.*

**Both arms are represented deliberately**, because Kozak −3 is upstream and Kozak +4 is
downstream and a reference drawn from one side is not matched to a target on the other. *Measured
here, upstream and downstream controls behave differently under every operator, so the arms are
reported separately as well as pooled.*

Reported positions use that same codon convention — the A of the start codon is +1, the base before
it −1, no zero — so Kozak −3 is array index `W/2 − 4` and Kozak +4 is `W/2 + 2`.

### Step b — the measurement

```
for TAG, for each of the five members:
  for each isoform i in the subsample:
      BASELINE = the observed window with channel 5 recomputed over its filled extent
      for each position p in BANK, for each base b in A,C,G,T:
          substitute b at p in slot R's window
          RECOMPUTE channel 5 over the window's filled extent
          score the model
          effect[member, i, p, b] = score - BASELINE
      record the attention weight on every ORF slot, from the baseline pass
```

**The baseline is built through the same recomputation path as the perturbations.** Channel 5 is
stored as float16 and recomputed in float32, so a recomputed perturbation against a stored baseline
carries a systematic offset shared by every effect, which does not average away in a profile.

**Eligibility, applied identically at every position.** An isoform contributes at a position when
the window's filled extent covers the position ±2 and all five of those bases are A, C, G or T. That
is the minimum span the operator and the exclusion check need. *Measured here at
`atg1000_stop1000`: 94.0% of isoforms are eligible at all 32 positions of a 30-control bank, and
per-position n runs 476–500 of 500.* A wider span requirement is not imposed: a requirement of that
kind cut the same quantity 3.2× and produced a confident null at n≈40.

**The substitution is window-local.** It is not propagated to the other ORF windows that overlap
these transcript coordinates. The propagated version is step e.

### Step c — contrasts against the control reference

```
for each operator in {A_vs_T, G_vs_C, GC_vs_AT, purine, G_vs_notG}:
    for each member m, each position p:
        contrast[m,p] = mean over eligible isoforms of
                          ( mean of effect over the operator's HIGH bases, exclusions dropped
                          - mean of effect over the operator's LOW  bases, exclusions dropped )
    mean[p], sd[p] over the five members;  t[p] = |mean[p]| / (sd[p]/sqrt(5))
    agree[p] = all five member signs equal
    for each hypothesis position:
        report percentile of t[p] and of |mean[p]| within the control positions
        report the control reference itself: median and max |t|, agreement rate, and the
          fraction of controls sharing the median sign
```

Signed effects are averaged across isoforms first and across members second; absolute values and
percentiles afterwards.

### Step d — stratification

The same contrast is recomputed within strata, **each against the control reference recomputed in
the same stratum**, because the reference moves with the stratum. *Measured here: the control `|t|`
median under `A_vs_T` is 1.52 in isoforms carrying only `is_sqanti_cds` and 1.09 in isoforms
carrying `is_ref_cds`.*

```
  ANNOTATION FLAGS: is_ref_cds present / is_sqanti_cds only / neither, on slot R
  ATTENTION:        terciles of the slot-R attention weight, averaged over members
  OBSERVED BASE:    the base actually present at the position, since the hypothesis positions
                    and the controls differ in composition -- measured here, purine fraction
                    0.742 at -3 and 0.694 at +4 against a control median of 0.509 -- and the
                    observed base's own cell is the no-op
```

### Step e — controls

| control | what it breaks | what it tests |
|---|---|---|
| **anchor shift**: run the identical bank with positions defined relative to an anchor displaced +37 nt from the start codon | the anchor's meaning, keeping geometry, padding and operator identical | whether the −3 result is a property of the start codon or of window geometry |
| **five randomly-initialized members**, per member and not ensembled | all learned structure | whether the architecture and input distribution alone produce the contrast |
| **propagated substitution**: the same bank with each substitution applied to every ORF window containing that transcript coordinate, channel 5 recomputed in each | the off-manifold condition of a window-local substitution | whether the result survives the on-manifold perturbation, which is what the manuscript claim needs |

The propagated run is the one that decides whether the result can be stated in the manuscript. The
window-local version is retained beside it because it is what the control reference was built on.

---

# 6. Acceptance

**There is no completeness property to check**, because nothing here is a decomposition. Acceptance
is that the perturbation was constructed correctly and the reference is the matched one.

1. The method floor is reported per member and is at least three orders of magnitude below the
   effects being interpreted.
2. Substituting the observed base reproduces the baseline; the no-op cell is the floor, not zero by
   assertion.
3. Array indices `W/2−1, W/2, W/2+1` spell A, T, G in every eligible isoform.
4. Every hypothesis-position number is reported with its control percentile. A number without one is
   not reportable.
5. The control reference is reported in full — median, max, agreement rate, same-signed fraction —
   so a reader can see whether the operator's reference is null.
6. Per-position n and the created-motif exclusion rate are reported beside every contrast.
7. All five member values are printed beside every across-member statistic.

**A direction is stated only when the five members agree on its sign AND the position exceeds the
control reference on reliability.** Either alone is insufficient: agreement alone is met by 65–92%
of control positions under the confounded operators.

8. **The percentile's resolution and its multiplicity are stated wherever it is used.** With 37
   controls the statistic takes 38 values, so "exceeds 97%" means rank 2 of 38 and the ceiling is
   attainable by chance at roughly 1 in 38. The operator is chosen after the forward passes (§2),
   so a best-of-10 selection across operator × position cells is a family-wise error near 0.4. No
   cell is reported as significant; percentiles are comparisons, not tests.

---

# 7. Cost

*Measured here 2026-07-31 on the local CPU (`~/miniforge3/envs/nmd_model_local/bin/python`), 500
isoforms × 39 positions × 4 bases at `atg1000_stop1000`: 116.6, 160.7, 170.9, 166.5 and 177.6 s per
member, 13.2 minutes for five members.* Cost is linear in isoforms and in bank size, so the
production run at 2,000 isoforms is ~53 minutes at this configuration for the window-local pass. Two costs are **predicted, not
measured**: `atg2000_stop2000` scales with window width, and the propagated pass multiplies the
forward passes by the number of windows a coordinate touches — *measured for Analysis 4: 5.1 of 10
windows at W=1000 and 7.5 at W=2000.* Time one propagated cell before scoping the full run. Nothing
here needs the cluster.

---

# 8. What this supports in the manuscript

**Claim 5.3.2 and legend 5.6.7** are the consumers. On the evidence measured while writing this
plan **neither half is statable**: −3 meets the reference test and fails the sign-agreement test,
and +4 fails both. The plan is written so that the production run can overturn either — including
by supporting them.

*Measured here 2026-07-31, `atg1000_stop1000`, five members, 500 isoforms, window-local, slot 0,
scored against the 37 frame-matched controls of §5a, complete-case:*

| operator | control \|t\| median | controls same-signed | Kozak −3 | Kozak +4 |
|---|---|---|---|---|
| **`A_vs_T`** | **1.00** | **73%** | n=334, \|t\| 3.26, **exceeds 97%** | n=353, \|t\| 0.79, exceeds 43% |
| `purine` | 3.65 | 100% | n=323, \|t\| 5.36, exceeds 78% | n=353, \|t\| 2.39, exceeds 32% |
| `G_vs_C` | 4.30 | 100% | n=483, \|t\| 4.29, exceeds 49% | n=500, \|t\| 8.95, exceeds 84% |
| `G_vs_notG` | 5.79 | 100% | n=323, \|t\| 4.28, exceeds 22% | n=353, \|t\| 5.59, exceeds 49% |
| `GC_vs_AT` | 5.66 | 100% | n=323, \|t\| 3.28, exceeds 16% | n=353, \|t\| 2.06, exceeds 3% |

**`A_vs_T` is the only operator whose control reference approaches a null** — 73% same-signed and
all five members agreeing at 8% of control positions against a 6.25% chance rate. Every other
operator's reference is 100% same-signed, which is the composition gradient of §2.

**Kozak −3 meets the reference test and fails the agreement test.** Under `A_vs_T`: mean +0.00262,
\|t\| 3.26, the 97th percentile of the control reference, 2.37× the median control magnitude. But
the five member values are −0.00023, +0.00340, +0.00363, +0.00200, +0.00432 — one member is
effectively zero and negative, so the signs do not all agree. **§6 requires both, so no direction is
stated for −3.** This is the strongest cell measured and it is not sufficient.

**And the percentile is a weak statistic here.** 97% of 37 controls is rank 2 of 38, selected after
the fact from 10 operator × position cells. A best-of-10 result at roughly p = 0.05 has a
family-wise error near 0.4, and nothing in this design controls it. The percentile is reported
because it is the right comparison, not because it is significant.

**Kozak +4 is unsupported pooled**, under every operator. Its natural operator `G_vs_notG` puts it
at the 49th percentile of a reference that is 100% same-signed with a \|t\| median of 5.79 — the
model prefers G nearly everywhere and +4 is not distinguished from that. `G_vs_C` holds GC fixed and
reaches the 84th percentile, which is the strongest pooled evidence for +4 and is not enough.

**But one +4 cell clears both criteria, and omitting it would be selective reporting.** `G_vs_C` in
the lowest slot-attention tercile (mean attention 0.343, n=167): mean +0.00591, \|t\| 6.75,
**exceeds 100% of that stratum's controls, all five members agreeing**. On §6's own terms that is
stronger than the −3 headline, which fails the agreement half. It is a post hoc stratum of a
secondary operator and is reported as such, not promoted.

**The −3 effect reverses with attention, so the pooled estimate averages opposing populations.**
Under `A_vs_T`, by slot-0 attention tercile, each against its own within-stratum control null:
+0.00469 (\|t\| 1.79, 65th pct, signs disagree) at attention 0.343; **+0.00580 (\|t\| 4.27, 100th
pct, signs agree)** at 0.709; **−0.00236** (\|t\| 1.59, 73rd pct, signs disagree) at 0.910. The
middle tercile is the only cell in this analysis that satisfies §6 outright.

**The result survives conditioning on the observed base**, which differs between target and
controls — 74.2% purine at −3 against a control median of 54.5%. Within observed-purine isoforms
−3 reaches the 92nd percentile; within observed-pyrimidine, the 97th.

**The override question (question 3) has a first answer, and it is the one that motivates a
retrain.** *Measured here under `A_vs_T` at −3, stratified by the annotation flags on slot 0:*

| stratum | stratum n | mean | \|t\| | exceeds |
|---|---|---|---|---|
| `is_sqanti_cds` only, no `is_ref_cds` | 186 | **+0.00538** | 4.16 | **97% of controls** |
| `is_ref_cds` present | 302 | +0.00095 | 1.44 | 68% of controls |

*The n column is stratum size. The isoforms actually contributing are fewer, because the
created-motif exclusion removes roughly a third before any contrast is formed; the production run
reports contributing n per cell, and this table does not.*

**The −3 effect is 5.7× larger where the reference-CDS flag is absent.** This is a between-isoform
comparison in the native model, so it is not the global re-weighting an ablation produces; it is
also confounded by everything else that differs between reference-annotated and novel isoforms, and
it is stated as evidence rather than as a demonstration. The clean test is a model trained without
those features, and this measurement supplies its pre-registered prediction: **a model that no
longer has the annotation features should show a −3 `A_vs_T` contrast across all isoforms at or
above the +0.0054 the flag-absent stratum shows now, scored against its own control reference.**
