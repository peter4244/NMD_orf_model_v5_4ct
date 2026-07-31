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

**Approved by Pete, 2026-07-30.** *Amended for Analysis 4 with Pete's approval, 2026-07-31: the
paragraph is the shared purpose of Analyses 1–4 and its second uncertainty does not apply here.
Analysis 4 measures the model's response to a sequence perturbation rather than computing an
attribution against reference transcripts, so it has no reference draw. Its uncertainty is the
training spread across members, the sampling error of the subsample it is measured on, and the
stochastic elements of its controls — enumerated in §3.2. The paragraph is otherwise unchanged and
remains binding on Analyses 1–3.*

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
consequence is mostly simplifying: a measurement of the model's own output has no completeness
property to verify, no reference distribution, and no method error. §6 replaces the acceptance
criterion accordingly.

**But a perturbation must be one a real sequence could produce, and a window-local one is not.** The
five ORF slots' windows overlap heavily in transcript coordinates, because the ORFs themselves
overlap. *Measured here 2026-07-31 over 3,000 isoforms: rank 0's start window shares at least 30 nt
with another slot's window in 92.4% of multi-ORF isoforms at W=1000 and 98.8% at W=2000, median
overlap 78% and 89% of the window; 99.89% of isoforms carry more than one ORF.* Perturbing only rank
0's window would present the same transcript base to the model as two different bases in two branches
at once — an input no sequence can generate, which is the off-manifold condition this method was
adopted to avoid. **Every perturbation is therefore propagated to every window containing that
transcript coordinate** (§5a). *Measured: a perturbation touches 5.1 of the 10 windows on average at
W=1000 and 7.5 at W=2000.*

The midpoint clip guarantees that ONE ORF's own start and stop windows are disjoint, and it does
that correctly; it says nothing about rank 0's start window overlapping another ORF's stop window,
which happens for 78% of isoforms at W=1000 and 95% at W=2000.

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

**There is no reference draw**, because the comparison is to the unperturbed sequence rather than to
a background distribution. The five reference draws that carried interpretation variance in Analyses
2 and 3 have no role here. But that does not leave one variance component; it leaves three, and the
plan reports all three:

| component | source | how estimated |
|---|---|---|
| **training** | five members | spread across members, at its own level, with the delete-one-member jackknife for the ensemble's own uncertainty |
| **sampling** | the 2,000-isoform subsample is a SAMPLE, not a census | nonparametric bootstrap over isoforms, needing no additional forward passes |
| **occlusion** | three dinucleotide shuffles per block | spread across shuffles, reported per block |

**The census licence of Analyses 2 and 3 does not transfer.** Those summarised the whole labelled
universe and were right to attach no isoform-level standard error. This analysis draws 2,000 from a
larger frame, so a sampling error exists and omitting it would be a reporting failure. *Measured:
between-isoform CV of a per-position mean is ~1.3, so the standard error at n = 2,000 is about 3.2%
of the value — small, which is why omitting it would be indefensible rather than defensible.* The
bootstrap resamples ISOFORMS, not positions, which also handles position-to-position dependence
correctly — *measured lag-1 autocorrelation of the positional profile is 0.60, so positions are very
far from independent.*

**Three levels, as in Analyses 2 and 3.** Member, ensemble, and leave-one-member-out. Signed effects
are averaged across members first; absolute values and any normalisation afterwards. `mean_abs` and
the concentration fraction are non-linear in the effects, so the ensemble and member centres differ
by a bias and not only by noise — the same property Analysis 2 documented.

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

### Step a — block occlusion, the primary instrument

The question claim 5.3.1 asks is **where**, at the scale of a motif rather than a base. Block
occlusion answers it directly and at a third of saturation ISM's cost.

```
BLOCK   = 3 nt
STRIDE  = 3, ANCHORED ON THE CODON  -- so inside the ORF each block is exactly one codon
SHUFFLE = 3 dinucleotide-preserving shuffles per block, averaged

for TAG, for each of the five members:
  for each isoform i in the subsample, for each branch:
      for each block B tiling the reportable positions:
          for s in 1..SHUFFLE:
              replace B's bases with a dinucleotide-preserving shuffle of B
              PROPAGATE that substitution to every ORF window containing those
                  transcript coordinates, recomputing each window's GC channel
              score the model
          effect[i,B] = mean over s of (score - BASELINE)
```

**The occluded block is replaced with shuffled real bases, never with zeros.** Padded positions are
all-zero in channels 0-3, so zeroing a block makes the model see the pattern it has learned means
"no sequence here". That would produce large artifacts strongest near the window edges — exactly
where the concentration claim is made. A dinucleotide-preserving shuffle also holds local composition
fixed, so the contrast isolates arrangement rather than confounding with GC content.

**Three shuffles per block, averaged**, so a block's effect is not one arbitrary draw. The shuffle is
the only stochastic element of the measurement and is seeded per (isoform, block) for reproducibility.

**Codon alignment is anchored, not assumed.** Blocks tile outward from the anchor codon in steps of
3, so within the ORF a block is one codon — the unit Saluki showed carries stability signal. Upstream
of the start codon and downstream of the stop codon a block is simply a 3-nt tile, which is harmless.
Both windows straddle that boundary, so alignment is meaningful for roughly half of each.

*A stride of 1 is available and costs three times as much for ~1-nt resolution via averaging the
three blocks covering each position. It is not used for claim 5.3.1, which concerns a ~50-nt scale.*

### Step a2 — base-level ISM, at pre-registered positions only

Saturation ISM is no longer the primary instrument. It is retained **only where single-base
resolution is the question** — the Kozak positions and the stop codon of §5c — where it is a handful
of substitutions per isoform rather than thousands.



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
      mean_effect[B] = mean over isoforms unpadded at B of effect[i,B]
      mean_abs[B]    = mean over isoforms unpadded at B of |effect[i,B]|
      n_unpadded[B]  = how many isoforms contributed
      in_frame[B]    = whether B is codon-aligned AND inside the ORF
```
Reported separately for `label == 1` and `label == 0`, and the in-frame subset reported separately
again, where a block is one codon.

**The reportable position set is fixed in advance, and it is not what the window width implies.**
*Measured here 2026-07-31 over 1,500 isoforms, positions where at least half the subsample is
unpadded:*

| branch | W=1000 | W=2000 | new at the wider width |
|---|---|---|---|
| start | 727 positions, −323 … +404 | 727, −323 … +404 | **0** |
| stop | 904, −403 … +501 | 1,404, −403 … **+1001** | **500** |

The start window is bounded by the ORF-midpoint clip and the transcript start, so widening it adds
nothing. The stop window is not: the wider configuration reaches 500 positions further downstream,
which is the exon-junction region Analysis 3 found dominates the structural evidence. **That is why
`atg2000_stop2000` is retained**, and why blocks are tiled over the reportable set rather than the
whole window — computing what will not be reported buys nothing.

**The class contrast is confounded by geometry unless it is matched.** Padding differs by label
class — *measured: 64.0% of start-window positions unpadded for `label == 1` against 68.6% for
`label == 0` at W=1000* — so the ≥50% rule keeps a different position set for each class. The
primary class contrast is therefore computed on a **padding-matched cohort**, with the unmatched
version reported beside it.

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
      record predicted log-odds for each of the 16 cells, PAIRED WITHIN ISOFORM
      report both main effects AND their interaction, under a stated uniform weighting

  EXCLUSIONS, enumerated per isoform before any contrast is formed. A substitution that
  creates a different motif is not measuring initiation context:
      +4 = T creates an in-frame stop codon at +4..+6 in 18.9% of isoforms
      -3 = A creates an in-frame AUG at -3..-1 in  4.0%
  Cells flagged for a created stop or AUG are excluded from the primary contrast and the
  excluded fraction is reported; the unexcluded version is a sensitivity.

  The +4 contrast moves the recomputed GC channel in one direction by construction ({G} is
  100% GC, {A,C,T} 33%), so a GC-matched G-vs-C contrast is reported beside it. The -3
  purine/pyrimidine contrast is already GC-balanced.

  STOP IDENTITY. Substitute the stop codon through UAA / UAG / UGA, context unchanged.
      report predicted log-odds by identity, and the UGA-vs-others contrast
      ALSO report UGA vs UAG, which is the GC-matched version of the same question
      ALSO substitute one NON-stop codon (UGG), which separates "does the model detect a
        stop at all" from "does it distinguish among stops" -- one extra pass per isoform
```

**Positions +2 and +3 both vary between stop codons** — UAA/UAG/UGA are A,A,G at +2 and A,G,A at +3 —
so the whole codon is substituted and the contrast is by identity. *An earlier draft pre-registered
+3 alone to separate UGA from the others, where UGA and UAA are both A; it could not have answered its
own question.*

**A direction is not stated unless the five members agree on its sign.** *Measured on a 150-isoform
pilot at `atg1000_stop1000`: the five members' −3 purine contrasts are +0.0043, +0.0019, −0.0050,
+0.0053, +0.0029 — they disagree in sign — and the training spread is 12× the subsample sampling
error. The ensemble estimate is smaller than one member standard deviation.* So sign disagreement is
the expected outcome, not a remote risk, and reporting it is a result rather than a failure. All five
member values are printed alongside every ensemble contrast.

Anything else described later is labelled post hoc and reported against §6's reference band.

### Step d — the faithfulness panel

```
  COMPREHENSIVENESS: occlude the top-k BLOCKS jointly; measure the actual change.
  SUFFICIENCY:       occlude every OTHER reportable block, retaining only the top k.
  BASELINES:         k random reportable blocks; k composition-matched blocks; and
                     the top-k chosen by a DIFFERENT member.
  for k in {1, 3, 5, 10, 25}
```

**Selection and evaluation must not use the same numbers, or the result is guaranteed.** The top-k
are chosen on four members' block effects and evaluated on the held-out fifth, five folds. Without
that, comprehensiveness at small k is arithmetic rather than evidence.

**Report the joint change AND the sum of the k individual block effects.** Their ratio is a direct
measurement of epistasis in the model — how far from additive the positions are — which is new
information the plan would otherwise discard, at no extra cost.

**Sufficiency uses composition-preserving corruption, not an argmin base.** Occluding ~500 blocks
with a prediction-minimising substitution produces a sequence unlike any mRNA, so the model's output
there is an extrapolation rather than a measurement. Shuffled occlusion keeps the corrupted input in
the neighbourhood of real sequence, and where the corrupted predictions sit relative to the
distribution of real predictions is reported so a reader can see how far the model was pushed.

### Step e — controls

| control | what it breaks | what it tests |
|---|---|---|
| **dinucleotide-preserving shuffle** of the window, scored under the **trained** model | motif structure, composition preserved | whether the signal is the motif or the model positional prior |
| **five randomly-initialized members**, reported per member and NOT ensembled | all learned structure | whether the architecture and input distribution alone produce the pattern |
| **label permutation at summarisation**, permuted WITHIN padding-extent strata | only the NMD/Control split | the reference distribution for the comparative half of 5.3.2, and nothing else |
| **anchor-shift**: recompute the profile with the window deliberately re-centred off the codon (+37 nt) | the anchor's meaning, keeping geometry and padding identical | whether the ±50 concentration is a property of the start codon or of window geometry |

**The anchor-shift control is the one that threatens claim 5.3.1 directly.** If concentration
survives an anchor pointing at nothing, the concentration is geometry, not biology. It is the
perturbation-era analogue of the reference-draw control the earlier analyses had.

**The label permutation must be stratified.** Padding differs systematically by label class, so an
unstratified permutation breaks a real label-to-geometry association and yields a null that is too
narrow for a padding-dependent contrast.

**The dinucleotide shuffle control preserves the anchor codon** and shuffles around it. A whole-window
shuffle would destroy the start or stop codon, break §6's coordinate assertions on the control, and
make "±50 of the anchor" meaningless. It is a shuffled-sequence *profile*, compared against the real
profile — not a single prediction.

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

**4. A no-op perturbation, checked where exactness is actually achievable.** Substituting a block for
itself must give zero. *Measured: against a recomputed baseline scored at batch 1 the residual is
2.4e-07 to 4.8e-07, not zero — the model's CPU path is batch-shape dependent, so a baseline scored
alone and a perturbation scored inside a chunk of 512 do not agree bitwise.* An "exactly zero"
criterion in output space would therefore stop the analysis on its first cell.

So the check moves to where exactness holds and stays able to fail:
- **Input space, exact:** the no-op perturbed window must be **bit-identical** to the baseline window.
- **Output space, calibrated:** the no-op is placed as **row 0 of every scoring chunk**, so baseline
  and perturbation share a batch shape; within a chunk the residual must then be exactly zero.
- The float16 offset the earlier draft found is still real and is still handled by routing the
  baseline through the same recomputation path.

**5. Propagation validity.** For a sample of perturbations, assert that every window whose transcript
span contains the perturbed coordinate was updated, and that no window outside that set changed.
This is the check for §5a's propagation, and it is the one guarding the plan's central claim to be
on-manifold.

**6. A reference band for post hoc positions — a MAXIMUM-statistic band, not a marginal one.** The
same statistic over far-field reportable positions (|pos| > 200), but the band is the distribution of
the **maximum over blocks of a region the size of the screened one**, obtained by block resampling. A
marginal 2.5/97.5 band would be exceeded by ~5% of screened positions under the null by construction,
which is not a filter. *Measured on a pilot: far-field lag-1 autocorrelation 0.60 and roughly 13
effective independent positions out of 383, so a marginal percentile band rests on far fewer
observations than it appears to.* A post hoc position is notable only if it clears both the
max-statistic band and a pre-registered effect-size floor, and the near-field exceedance count is
reported as a calibration statistic so a reader can see whether the filter is doing anything.

---

# 7. Cost, and what is measured before anything is committed

**Block occlusion at stride 3 with three shuffles costs about a third of saturation ISM.** Stride,
not block size, is the cost lever: at stride 1 the three shuffles would put the cost back to ISM's.

| | blocks per isoform, both branches | passes (x3 shuffles) | CPU-hours for the full design |
|---|---|---|---|
| `atg1000_stop1000` | 545 | 1,635 | **12** |
| `atg2000_stop2000` | 711 | 2,133 | **36** |
| **total** | | | **48 CPU-h, ~2.4 h wall at 20-way** |

*Computed from the measured per-pass cost below and the reportable-position counts in §5b; the full
design is 2,000 isoforms x 5 members x 2 branches.* Base-level ISM at the pre-registered positions
adds about 20 passes per isoform, which is negligible.

The per-pass cost that these rest on, **measured 2026-07-31** on the development machine's CPU,
`atg1000_stop1000` member 100, 8 isoforms, batch 512:

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
