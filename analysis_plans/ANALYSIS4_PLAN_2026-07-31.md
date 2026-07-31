# Analysis 4 Plan — where in the sequence windows the model looks

*Rewritten 2026-07-31, before any result exists. Supersedes the DeepSHAP-based draft of the same
date; §2 gives the measurement that replaced it.*

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

**Approved by Pete, 2026-07-30. Do not edit this paragraph.**

Analyses 2 and 3 divided the evidence among branches and among structural features. Analysis 4 opens
the two sequence branches: **where inside each window the model output is sensitive, and to what.**

---

# 2. Why this analysis measures rather than attributes

The first draft used DeepSHAP. It is replaced, and the decisive reason is measured rather than
argued.

**DeepSHAP's completeness error on this model is larger than the effect it decomposes.** Measured
here 2026-07-31, member `atg1000_stop1000` seed 100, 100 reference isoforms, 20 explained isoforms,
comparing the sum of attributions against `f(x) − E_bg[f]` evaluated directly:

| branch | median completeness error | median effect | error as % of effect |
|---|---|---|---|
| start window | 0.2685 | 0.2083 | **128.9%** |
| stop window | 0.5470 | 0.1784 | **306.6%** |

`shap_values(..., check_additivity=True)` raises `AssertionError` on both branches. A positional
figure built on this would be reporting noise. *Run log: `analysis_plans/probe_deepshap_additivity.py`
and its SLURM log, Explorer job 8861228.*

**The mechanism is known.** `shap`'s PyTorch `DeepExplainer` attaches DeepLIFT rules by walking
`nn.Module` instances. This model's dominant nonlinearities are functional calls — `F.relu` after
each convolution and batch-norm, a global `max(dim=-1)` pool, a masked softmax and `bmm` in the
attention aggregator (`model.py:88-93, 140-166`). Rules do not attach to those. `deepshap.py` passes
`check_additivity=False` at all three call sites, which is what one does after this check has failed.

**Gradient attribution also has a documented genomics-specific failure on one-hot input.**
Majdandzic, Rajesh & Koo, *Genome Biology* 24:109 (2023): one-hot sequence lies on a simplex, the
network may behave arbitrarily off it at no cost to generalization, and the input gradient acquires a
component orthogonal to the simplex. **In silico mutagenesis is explicitly exempt** — a forward pass
on real sequence never leaves the simplex.

**And the nearest published model to ours did it by measurement.** Saluki (Agarwal & Kelley, *Genome
Biology* 23:245, 2022) predicts mRNA half-life from transcript sequence using a **6-channel input:
four nucleotide one-hot channels, a binary first-codon-frame track and a binary splice-site track** —
our encoding minus the GC channel. Convolutional with a GRU head, and trained as **five replicate
models averaged as an ensemble**, the design our five members already implement. Its published
interpretation is **ISM, TF-MoDISco on the ISM scores, and insertional analysis**: three perturbation
methods, no gradient attribution anywhere.

**So Analysis 4 measures the model response to perturbation instead of attributing its output.** The
consequence is mostly simplifying: a measurement of the model own output has no completeness property
to verify, no reference distribution, and no method error. §6 replaces the acceptance criterion
accordingly.

---

# 3. Data

## 3.1 Terms

Datasets A and B are those of the preceding plans; their terms carry over. Five are new.

**In silico mutagenesis (ISM).** For one isoform, one position and one alternative base: substitute
and record the change in predicted log-odds. Saturation ISM does this for all three alternatives at
every position. It measures the model; it does not estimate an attribution.

**Insertional analysis.** Placing a chosen motif at a chosen position in otherwise-unchanged real
transcripts and recording the mean predicted change. Saluki method for showing an effect is
positional; the template for §5c.

**Derived channels, and the rule for updating them under perturbation.** Of the nine channels, four
are the nucleotide one-hot, **channel 5 is a 50-bp rolling GC fraction computed from the window own
sequence** (`data_prep.py:204`), channel 4 marks exon junctions, channels 6-8 mark reading frame.
Substituting a base changes the sequence, so:

- **The GC channel is recomputed** over its support at every perturbation. Leaving it stale feeds the
  model an input that cannot occur, reintroducing the off-manifold problem ISM exists to avoid.
- **Junction and frame channels are unchanged.** A point substitution does not move a splice junction
  or shift the reading frame.

*Saluki faced the same decision and documented it: for mutations altering a stop codon they chose NOT
to modify the coding track downstream, because doing so "creates disproportionately large effect
predictions". The analogous choice here is stated rather than left to the code.*

**Window coordinate.** Windows are centered on the **middle** nucleotide of the codon, at array index
`W/2` (`data_prep.py:857-858`). Verified here 2026-07-31 over 400 rank-0 isoforms at both widths:
indices `W/2-1`, `W/2`, `W/2+1` spell A,T,G in 100% of start windows.

Reported positions use the **codon convention** — the A of the start codon is **+1**, the base before
it **-1**, no zero:

```
pos(i) = i - W/2 + 2    for i >= W/2 - 1     (the codon and downstream)
pos(i) = i - W/2 + 1    for i <= W/2 - 2     (upstream, negative)
```

so Kozak -3 is array index `W/2 - 4` and Kozak +4 is `W/2 + 2`. *The earlier draft had this off by
one, putting both Kozak probes on positions with no motif: measured purine content 50.8% at the wrong
index against 76.5% at the right one.*

**Padding.** Windows are zero-padded by the ORF-midpoint clip and by transcript ends. Padding is
position-dependent and large — 98.1% of isoforms unpadded within +/-50 of the anchor against **41.5%**
outside it at `atg2000_stop2000`. **Every positional statistic is computed over unpadded isoforms
only, with the per-position unpadded count reported beside it.** Without this the concentration result
is mechanically guaranteed, and guaranteed larger at the wider configuration.

## 3.2 Datasets

**A** = `results_4ct_dn/nmd_orf_data.h5`; **B** = the ten checkpoints, five members per
configuration. The model contract of the preceding plans applies unchanged.

**Dataset C — outputs, written to `results_interp_all/`**

| file | contents |
|---|---|
| `ism_{branch}_{tag}_seed{S}.npz` | per-isoform x position x alternative-base change in predicted log-odds, plus the per-position unpadded mask |
| `ism_profile_per_cell.tsv` | one row per (configuration, level, member, branch, position): mean and mean-absolute effect by label class, and the unpadded count |
| `context_enumeration_{tag}.tsv` | the designed-context results of §5c |
| `faithfulness_{tag}.tsv` | the occlusion panel of §5d |
| `ism_summary_{tag}.json` | positional summaries, the training spread, the pre-registered contrasts, the control comparisons |

**There is no reference draw and no interpretation spread here.** ISM has no background distribution:
the comparison is to the unperturbed sequence. The five reference draws that carried interpretation
variance in Analyses 2 and 3 have no role, and **the only variance component is the training spread
across five members.** This is a real difference between the analyses and is stated wherever a spread
is reported.

---

# 4. Scientific questions

1. **Where is the model output sensitive within each window?** (claim 5.3.1)
2. **Does start-codon context behave as Kozak predicts?** (claim 5.3.2, legend 5.6.7)
3. **Does stop codon identity change the prediction, and does UGA differ from UAA and UAG?** (5.3.2)

**Not in scope.** Claim 5.3.3 (*UGA more common in NMD-susceptible transcripts, 56% vs 48%*) is a
sequence-composition claim, not a model claim, and is already a recorded ledger defect.

**Population.** Sensitivity is a per-isoform property, so ISM runs on a **stratified subsample of
2,000 isoforms**, stratified by label class and by PTC status, reported as distributions over the
subsample and never as exemplars. *Size fixed 2026-07-31 from the measurement in §7: 2,000 isoforms
costs 24 CPU-hours at `atg1000_stop1000` and roughly 97 at `atg2000_stop2000`, about six hours of
wall clock across both at twenty-way parallelism. The full 9,321-isoform population would be roughly
five times that and buys precision on a distribution that 2,000 already characterises.*

---

# 5. Approach

### Step a — saturation ISM over both windows

```
for TAG, for each of the five members:
  for each isoform i in the subsample, for each branch:
      for each position p unpadded in i:
          for each alternative base b != observed:
              rebuild the window with b at p, RECOMPUTING the GC channel over its support
              effect[i,p,b] = model(perturbed) - model(BASELINE)      # log-odds
```

**The baseline is built through the same recomputation path as the perturbations** — the observed
window with its GC channel recomputed, not the window as stored. Channel 5 is stored as float16 and
recomputed in float32, so comparing a recomputed perturbation against a stored baseline puts a
systematic offset into every effect. *Measured 2026-07-31: a no-op substitution against the stored
baseline gives 2.6e-06 to 1.3e-05 log-odds; against a recomputed baseline it gives exactly 0.0 on
every isoform tested. The offset is small but it is systematic and shared by every perturbation, so
it does not average away in a positional profile.*

Batched over (position x base) so each isoform costs a bounded number of forward passes.

### Step b — the positional profile, padding-aware

```
      mean_effect[p] = mean over unpadded isoforms of mean over b of effect[i,p,b]
      mean_abs[p]    = mean over unpadded isoforms of mean over b of |effect[i,p,b]|
      n_unpadded[p]  = how many isoforms contributed
```
Reported separately for `label == 1` and `label == 0`, only where at least half the subsample is
unpadded.

**Concentration (claim 5.3.1)** is the fraction of total `mean_abs` within +/-50 of the anchor over
unpadded positions only, reported beside the unpadded profile so a reader can see the denominator is
not padding.

### Step c — designed context enumeration, which is how the motif claims are made

Kozak and stop codon identity are **known motifs at known positions**, so they are established by
enumerating contexts rather than inferred from a sensitivity peak. This follows Saluki insertional
analysis and the Optimus 5-Prime enumeration of `NNNAUGNN` contexts (Sample et al., *Nat Biotechnol*
2019), which is how the Kozak consensus was established in a comparable model.

```
  START CONTEXT. Hold everything fixed and substitute the pre-registered Kozak positions
  -3 and +4 through all 4 x 4 = 16 combinations.
      record predicted log-odds for each
      report the marginal effect of purine vs pyrimidine at -3, and G vs not-G at +4

  STOP IDENTITY. Substitute the stop codon through UAA / UAG / UGA, context unchanged.
      report predicted log-odds by identity, and the UGA-vs-others contrast
```

**Positions +2 and +3 both vary between stop codons** — UAA/UAG/UGA are A,A,G at +2 and A,G,A at +3 —
so the whole codon is substituted and the contrast is by identity. *An earlier draft pre-registered
+3 alone to separate UGA from the others, where UGA and UAA are both A; it could not have answered its
own question.*

Anything else described later is labelled post hoc and reported against §6 reference band.

### Step d — the faithfulness panel

```
  COMPREHENSIVENESS: substitute the k positions of largest |mean effect| with the base
      minimising predicted log-odds; measure the actual change.
  SUFFICIENCY: substitute all OTHER unpadded positions likewise, retaining only the top k.
  BASELINE: the same for k random unpadded positions, and for k positions matched on
      observed base composition.
  for k in {1, 5, 10, 25, 50, 100}
```
This converts "the model evidence sits here" from a description of a profile into a measurement of
the model output.

### Step e — controls

| control | what it breaks | what it tests |
|---|---|---|
| **dinucleotide-preserving shuffle** of the window, scored under the **trained** model | motif structure, composition preserved | whether the signal is the motif or the model positional prior |
| **five randomly-initialized members**, reported per member and NOT ensembled | all learned structure | whether the architecture and input distribution alone produce the pattern |
| **label permutation at summarisation** | only the NMD/Control split | the reference distribution for the comparative half of 5.3.2, and nothing else |

The permuted-label retrain of the earlier draft is **dropped**. Its purpose was to control an
attribution artifact; ISM measures the model own output, so that artifact does not arise, and it is
not worth ten training runs. The random-weight control is kept because it is nearly free and still
answers whether the architecture imposes the pattern.

Random-weight controls are **per member, not ensembled** — averaging five untrained models drives the
profile toward zero by cancellation rather than by absence of structure, which would make the control
unable to fail.

---

# 6. Acceptance

**ISM has no completeness property to check, because it is not a decomposition.** Each number is a
difference between two forward passes. So acceptance is not about method error; it is about whether
the perturbation was constructed correctly.

**1. Coordinate alignment — an assertion, not a convention.** In the one function converting array
indices to reported positions:
```
assert nucleotide channels at reported +1,+2,+3 spell A,T,G in >= 99.9% of start windows
assert reported +1 is T and (+2,+3) in {AA, AG, GA} in >= 99.9% of stop windows
```
An off-by-one is what the earlier draft shipped; this makes it impossible to ship silently.

**2. Perturbation validity.** On a sample of perturbed windows: the one-hot remains a valid one-hot,
the GC channel equals a fresh recomputation from the perturbed sequence, and junction and frame
channels are unchanged.

**3. Determinism.** The same perturbation scored twice gives the same number. ISM has no RNG.

**4. A no-op perturbation.** Substituting the observed base for itself must give **exactly** zero,
not approximately. Free, and it has already earned its place: it caught the float16 baseline offset
above, which no positional profile would have revealed. Exact zero is achievable and is therefore the
bar — anything else means the baseline and the perturbation are not going through the same path.

**5. A reference band for post hoc positions.** The same statistic over far-field unpadded positions
(|pos| > 200) with its 2.5th and 97.5th percentiles. A post hoc position is notable only outside that
band — selection-aware, no hypothesis test.

---

# 7. Cost, and what is measured before anything is committed

ISM is 3 substitutions per unpadded position. **Measured 2026-07-31** on the development machine's
CPU, `atg1000_stop1000` member 100, 8 isoforms, batch 512:

| | measured |
|---|---|
| unpadded positions per window | **648 of 1,000 (64.8%)** |
| wall time per isoform, one branch | **4.38 s** |
| per isoform, both branches x five members | **43.8 s** |
| peak memory | **2.02 GiB** |

| subsample | `atg1000_stop1000` | wall at 20-way |
|---|---|---|
| 500 | 6.1 CPU-h | 0.3 h |
| **2,000** | **24.3 CPU-h** | **1.2 h** |
| 9,321 (the full population) | 113.5 CPU-h | 5.7 h |

`atg2000_stop2000` roughly **quadruples** this: the window doubles, so both the position count and
the per-row cost double. The chosen 2,000-isoform subsample is therefore about 121 CPU-hours across
both configurations, roughly six hours of wall clock at twenty-way parallelism.

*Run log: `analysis_plans/probe_ism_cost_runlog.txt`. The 64.8% unpadded figure independently
corroborates the padding measurement in §3.1 and is the reason positions are filtered rather than
averaged over.*

This measurement is the discipline that has already caught a 35.3 GiB peak against a 32 GB request
and a per-isoform rate wrong by 6x.

`deepshap.py` is **not** used, so its eager dual-split read — 28 GB at W=1000 and 57 GB at W=2000,
never given the chunked fix — is not on this path. The ISM producer reads windows through the chunked
path `11_kernel_shap_branches.py` already uses.

---

# 8. What this supports in the manuscript

Claims **5.3.1** and **5.3.2**, and figure legend **5.6.7**.

**A wording consequence.** ISM measures how the model output responds to a substitution. It does not
measure "importance" and does not decompose the prediction, so sentences of the form *"X% of the
signal comes from..."* cannot be written from it, and *"the model learned Kozak"* becomes the
enumeration result of §5c — the measured effect of context on the model output — rather than an
inference from a profile. That is a stronger claim than the one it replaces, and it is the form the
comparable literature reports.
