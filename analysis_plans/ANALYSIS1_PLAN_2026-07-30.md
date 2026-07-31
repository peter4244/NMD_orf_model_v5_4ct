# Analysis 1 Plan — held-out performance of the NMD sequence model

*Written 2026-07-30, before any result exists.*

This document specifies what is done to the data. Code is written from it.

---

# 1. Purpose

> We have trained a model that predicts, from transcript sequence alone, whether an isoform is
> degraded by nonsense-mediated decay. This analysis establishes two things: how accurately it does
> so on transcripts and chromosomes it never saw, and what kinds of sequence and structural
> information it relies on to do it. Both are reported with uncertainty that reflects the two ways
> the answer could have come out differently — a different training run, and a different draw of the
> reference transcripts used to compute an attribution.

**Approved by Pete, 2026-07-30. Do not edit this paragraph.**

Analysis 1 addresses the first: **how accurately.**

---

# 2. Data

## 2.1 Terms

**Isoform.** One distinct transcript sequence assembled from the long-read data. It is the unit of
observation: everything below holds one entry, or one prediction, per isoform.

**A note on the word "dataset."** Sections 2.2 to 2.4 label three data *sources* — Dataset A, B and
C. Within Dataset A, HDF5's own term for a named array inside the file is also "dataset". To keep
the two apart, the contents of the file are called **arrays** throughout.

**NMD susceptible.** An isoform whose abundance rises when nonsense-mediated decay is inhibited with
Smg1i. The label is established upstream by the Isopair analysis and arrives fixed. **Control** is an
isoform whose abundance does not rise.

**Split.** Which of fitting, validation or testing an isoform is used for, assigned by chromosome so
a gene never appears in two splits.

**Paralog leakage.** A gene in one split having a near-duplicate elsewhere — at least 80% protein
identity, both expressed — whose sequence would let the model recognize a test isoform it had
effectively already seen. Isoforms of such genes are held in a separate split and excluded from
scoring.

**Configuration.** A pair of sequence-window widths: one for the window around each ORF's start
codon, one for the window around its stop codon. Written as a **tag**, `atg2000_stop2000`. Prose says
*start window*; the code says `atg`, for legacy reasons.

**Seed.** The random number initializing model fitting.

**Member.** One fitted model: one configuration at one seed.

**Ensemble.** The five members of one configuration, their predicted log-odds averaged per isoform.

**Log-odds.** The scale the model outputs, before conversion to a probability. Zero is even odds;
positive favors NMD susceptible.

## 2.2 Dataset A — model inputs

**One file: `results_4ct_dn/nmd_orf_data.h5`.**

It is an HDF5 file, which is a container rather than a table — more like a folder. It holds named
**arrays**, and everything listed in this section lives inside that single file. There are no other
input files. Every array has 41,765 entries along its first axis, and all of them share one row
order: entry *i* of any array refers to the same isoform as entry *i* of any other.

Built by `data_prep.py` from the upstream Isopair feature tables. **41,765 isoforms.**

The upstream tables carry 42,043. The difference is a single exclusion applied here. Some transcripts
run out of one gene and into the next, and the annotation gives each of them a fused gene identifier
naming both genes rather than one — written `ENSGa.v::ENSGb.v`. A fused identifier never matches a
plain one, so these isoforms are invisible to the paralog screen: it cannot ask whether their gene
has a near-duplicate in another split, because they have no single gene to ask about. **278 such
isoforms, drawn from 233 fused loci, are removed**, leaving 41,765.
*Measured here 2026-07-30 from `results_4ct_dn/nmd_orf_data.h5`: 41,765 rows.*

**Arrays Analysis 1 reads directly.** Each is one value per isoform:

| array | one entry |
|---|---|
| `labels` | `int8`; 1 if NMD susceptible, 0 if control |
| `split` | `train`, `val`, `val_paralog`, `test` or `test_paralog` |
| `isoform_id` | variable-length string; the alignment key for everything downstream |
| `chr` | the chromosome |

**Arrays the model consumes when scoring.** Analysis 1 does not read these itself; it hands the
relevant rows to a member, which does:

| array | shape | contents |
|---|---|---|
| `w{W}/atg_windows` | 41,765 × 5 × 9 × W | encoded sequence around each of five ORFs' start codons, at W ∈ {100, 500, 1000, 2000} |
| `w{W}/stop_windows` | 41,765 × 5 × 9 × W | the same around each stop codon |
| `orf_features` | 41,765 × 5 × 5 | five numeric features per ORF |
| `orf_mask` | 41,765 × 5 | `bool`; which ORF slots hold a real ORF |
| `normalization/orf_feat_mean`, `normalization/orf_feat_std` | 5 each | feature normalization, computed on fitting rows only |

A configuration selects which `w{W}` group each branch reads.

**Where the codon sits in a window.** Each window is centered on the **middle** nucleotide of its
codon, at array index `W/2`. A start window therefore holds the A of the start codon at `W/2 − 1`,
the T at `W/2`, and the G at `W/2 + 1`; a stop window is centered the same way on the middle base of
its stop codon. *Set at `data_prep.py:856-858`, `orf_start − 1 + 1`, with the comment "Center on
middle nucleotide of codon". Measured here 2026-07-30 over 3,000 isoforms at ORF rank 0: 97.7% of
start windows and 88.7% of stop windows place the codon there — both figures are floors, since the
check matches the first codon in the window and an upstream one can mask the true anchor.*

This offsets array coordinates by one from the convention that numbers the A of the start codon as
+1. Analysis 1 does not use positional coordinates. Any analysis that does must state which
convention its axis uses.

**Splits.**

| split | chromosomes | n |
|---|---|---|
| `train` | all others | 26,711 |
| `val` | chr2, chr4 | 4,356 |
| `val_paralog` | chr2, chr4 | 56 |
| `test` | chr1, chr3, chr5, chr7 | 10,520 |
| `test_paralog` | chr1, chr3, chr5, chr7 | 122 |
| **total** | | **41,765** |

*Measured here 2026-07-30 from `results_4ct_dn/nmd_orf_data.h5`. All five counts match the values
predicted in `REBUILD_BASELINE_2026-07-30.md` before the file was built. Step 1 re-confirms them at
run time.*

**`test_clean`** is the selection this analysis scores: rows where `split == "test"`. It excludes
`test_paralog`. **n = 10,520, of which 2,405 are NMD susceptible — 22.9%.** Across all 41,765 rows
the figure is 22.3%.
*Measured here 2026-07-30 from `results_4ct_dn/nmd_orf_data.h5`. Verified at the same time: every
isoform in the selection lies on chr1, chr3, chr5 or chr7, and none carries `test_paralog`.*

## 2.3 Dataset B — trained members, `results_4ct_sweep/best_model_{tag}_seed{S}.pt`

One checkpoint per (configuration, seed), holding fitted weights and the epoch at which early
stopping fired. Every member was fitted on `train` with early stopping on `val`, and no test-split
quantity informed fitting or configuration choice.

Analysis 1 uses ten:

| configuration | start window | stop window | seeds | trainable parameters |
|---|---|---|---|---|
| `atg2000_stop2000` | 2000 | 2000 | 100, 200, 300, 400, 500 | 34,050 |
| `atg1000_stop1000` | 1000 | 1000 | 100, 200, 300, 400, 500 | 34,050 |

*Trainable parameters: measured here 2026-07-30 by building each model from `config_dn.yaml`.*

Both configurations are scored and both are reported.

**Each checkpoint also stores the validation AUC of its own member**, which makes the sweep's summary
verifiable without the cluster. Measured here 2026-07-30 by reading all ten:

| configuration | member validation AUC, five seeds | mean | sd |
|---|---|---|---|
| `atg2000_stop2000` | 0.93212, 0.93519, 0.92893, 0.93009, 0.93357 | **0.93198** | 0.00254 |
| `atg1000_stop1000` | 0.92434, 0.93176, 0.92729, 0.93056, 0.92400 | **0.92759** | 0.00353 |

Both means reproduce the figures cited from the sweep exactly. Early stopping fired at epoch 4 to 7.
**All ten members are distinct models**, verified by hashing each state dictionary: five distinct
digests per configuration.

**What a member requires, and what it returns.** Each of the following changes the predictions if
done differently, and none of them raises an error when done wrong.

**Input**, for one isoform: the start and stop windows at the configuration's two widths, for all
five ORF slots; the five structural features, for all five slots; and the ORF mask.

**The structural features are normalized before scoring**, by subtracting
`normalization/orf_feat_mean` and dividing by `normalization/orf_feat_std`, both read from Dataset A.
Members were fitted on normalized features. Passing raw ones yields plausible predictions that are
wrong. `n_downstream_ejc` has mean 1.67 and standard deviation 3.51, an order of magnitude away from
the other four, so the error is largest on the feature carrying the most signal.

**The ORF mask is passed, not ignored.** It marks which of the five slots hold a real ORF. The
attention layer that combines the five excludes masked slots; treating a padded slot as real changes
the prediction. This affects the 1.2% of isoforms with fewer than five ORFs. *Measured here
2026-07-30 from `tx_summary.tsv`.*

**The model runs in evaluation mode.** The encoders contain batch-normalization layers. In training
mode those normalize using the statistics of whatever batch an isoform happens to land in, so a
prediction would depend on which other isoforms were scored alongside it. In evaluation mode they use
the stored running statistics instead. This is a silent difference: both modes return a number.

**The batch-normalization running statistics are part of the checkpoint and must be loaded with the
weights.** The state dictionary holds 34,310 elements — the 34,050 trainable parameters plus 260
buffer entries: `running_mean` and `running_var` for each of four batch-normalization layers, at 32
channels each, and four scalar batch counters. *Measured here 2026-07-30 from
`best_model_atg2000_stop2000_seed100.pt`.*

**Output:** one log-odds per isoform — the model's raw output, before any conversion to a
probability.

**The two window widths come from the tag, not from the checkpoint.** Each checkpoint stores a copy
of the configuration file, and it does not describe the model it belongs to: in
`best_model_atg2000_stop2000_seed100.pt` that copy reports `data.window_size_atg = 100`,
`data.window_size_stop = 1000` and `selected = 500/500`, while the model is 2000/2000. The widths
were supplied as command-line overrides at fitting time and the stored copy never saw them. Building
a model from the embedded configuration produces the wrong architecture. *Measured here 2026-07-30.*
Other architecture hyperparameters — channel counts, embedding dimensions, dropout — are consistent
between the embedded copy and `config_dn.yaml`.

How a checkpoint is loaded, how scoring is batched, and which device it runs on are implementation
choices and do not change the result.

## 2.4 Dataset C — outputs, written to `results_interp_all/`

| file | contents |
|---|---|
| `ensemble_metrics_{tag}_ens5_test_clean.json` | `n`, `n_nmd`, ensemble AUC and AUPRC, their bootstrap intervals, the five member AUCs and AUPRCs, member mean and sd for each, the two mean-probability sensitivity values, the bootstrap seed and resample count |
| `ensemble_predictions_{tag}_ens5_test_clean.tsv` | one row per scored isoform: `isoform_id`, `chr`, `label`, `logit_seed{S}` for each of the five, `logit_ensemble` |

One of each per configuration; two of each in total.

---

# 3. Analysis 1

## 3a. Scientific question

**Can an isoform's susceptibility to nonsense-mediated decay be predicted from its sequence and ORF
structure alone?**

The determinants of NMD are largely visible in the transcript — a stop codon lying upstream of an
exon–exon junction, upstream ORFs in the 5′ untranslated region, a long 3′ untranslated region, stop
codon context that promotes readthrough. The labels come from an experiment. The question is whether
the sequence-visible determinants suffice to recover the experimental outcome, on a population the
model has not seen in any form.

**And how much would the answer have differed had a single fitting run produced it?**

## 3b. Starting data

| | dataset | what is used |
|---|---|---|
| inputs | **Dataset A** | the `test_clean` selection, with `labels`, `isoform_id`, `chr`, and the window and feature arrays at each configuration's widths |
| models | **Dataset B** | all ten members |
| outputs | **Dataset C** | written by this analysis |

Nothing else is read.

## 3c. Approach

Seven steps in order. Steps 1 and 2 run once and are shared; steps 3 to 7 run once per
configuration.

```
TAGS  = [atg2000_stop2000, atg1000_stop1000]
SEEDS = [100, 200, 300, 400, 500]
B     = 2000            # bootstrap resamples
RSEED = 20260730        # bootstrap RNG seed, fixed
```

### Step 1 — define the population

Read the four per-isoform arrays and select the test split. The assertions are not diagnostics: they
establish `n` and fix the isoform order that every later step is aligned to. A failure stops the
analysis rather than being reconciled afterwards.

```
read split, chr, labels, isoform_id from Dataset A      # 41,765 entries each, one shared order
keep  = entries where split == "test"                   # this is test_clean
assert every chr[keep] is one of chr1, chr3, chr5, chr7
assert no entry in keep has split == "test_paralog"
n     = len(keep)                                       # expect 10,520
y     = labels[keep]
ids   = isoform_id[keep]                                # the alignment key for all later steps
n_nmd = sum(y)
```

### Step 2 — draw the bootstrap resamples

Draw them once, before any scoring, and reuse the same resamples for both configurations and both
metrics. Sharing them makes the two configurations' intervals paired rather than independently
noisy. The seed is fixed so the intervals are reproducible.

```
rng = RNG(RSEED)
for b in 1..B:
    R[b] = n indices drawn from 0..n-1 with replacement
    assert 0 < sum(y[R[b]]) < n        # both classes present, or AUC is undefined
```

### Step 3 — score the five members

Load the test rows once per configuration and reuse them across its five members, so every member
sees identical isoforms in identical order. Two assertions carry weight: that the order scored
matches `ids`, and that no two members produced the same predictions. Five identical vectors would
mean one checkpoint was written five times, which gives a member spread of exactly zero and reads as
a stability result.

```
for TAG in TAGS:
    W_start, W_stop = the two widths named by TAG
    X = w{W_start}/atg_windows[keep], w{W_stop}/stop_windows[keep],
        orf_features[keep], orf_mask[keep]              # rows in `ids` order

    for i, S in enumerate(SEEDS):
        L[i] = member(TAG, S) applied to X              # n log-odds
        assert the order scored equals ids
    assert no two rows of L are identical
```

### Step 4 — form the ensemble and the point estimates

The ensemble is the mean log-odds across the five members, per isoform; its AUC and AUPRC are the
headline numbers. Scoring each member separately as well gives the spread across fitting runs.

```
    ens = column mean of L                              # mean log-odds, never mean weights

    auc_ens   = AUC(y, ens)
    auprc_ens = AUPRC(y, ens)

    for i in 1..5:
        auc_mem[i]   = AUC(y, L[i])
        auprc_mem[i] = AUPRC(y, L[i])
    auc_mem_mean,   auc_mem_sd   = mean(auc_mem),   sd(auc_mem)     # sd denominator 4
    auprc_mem_mean, auprc_mem_sd = mean(auprc_mem), sd(auprc_mem)
```

### Step 5 — interval on the ensemble

Recompute the ensemble's two metrics on each bootstrap resample, with the model held fixed. This
varies the test set and nothing else, so the interval answers how precisely a test set of this size
determines the score.

```
    for b in 1..B:
        boot_auc[b]   = AUC(y[R[b]],   ens[R[b]])
        boot_auprc[b] = AUPRC(y[R[b]], ens[R[b]])
    auc_ci   = 2.5th and 97.5th percentiles of boot_auc
    auprc_ci = 2.5th and 97.5th percentiles of boot_auprc
```

### Step 6 — aggregation sensitivity

Repeat the headline calculation combining members on the probability scale instead of the log-odds
scale. The two are not interchangeable and produce different rankings; log-odds is the primary rule
and this records what the alternative would have given.

```
    ens_prob  = column mean of sigmoid(L)               # mean PROBABILITY, not mean log-odds
    auc_alt   = AUC(y, ens_prob)
    auprc_alt = AUPRC(y, ens_prob)
```

### Step 7 — write

Two files per configuration, into `results_interp_all/`, with the contents listed under Dataset C.
The bootstrap seed and resample count are written alongside the numbers so an interval can be
reproduced from the record.

```
    write ensemble_metrics_{TAG}_ens5_test_clean.json
    write ensemble_predictions_{TAG}_ens5_test_clean.tsv
```

## 3d. Statistical models and estimators

**No hypothesis test is performed.** This analysis produces estimates with uncertainty. Nothing is
compared against a null, and the two configurations are not compared against each other.

### The model

The reported model is the **ensemble**: the mean of five members' predicted log-odds, per isoform.
Averaging is on the log-odds scale, not the probability scale, and the two are not interchangeable —
`mean(sigmoid(x)) ≠ sigmoid(mean(x))`, so they induce different rankings and therefore different AUC
and AUPRC. Log-odds is fixed as the primary rule; the probability-scale value is computed at step 6
and reported alongside as a declared sensitivity check.

Weights are never averaged. Independently seeded networks occupy different points in a
permutation-symmetric weight space, and their weight-space mean is not a model.

### The estimators

**AUC** estimates the probability that a randomly chosen NMD-susceptible isoform is scored above a
randomly chosen control. **AUPRC** is the corresponding summary of the precision–recall curve. Both
are rank-based and invariant to any monotone transform of the score, so applying a sigmoid before
computing them changes nothing — while the aggregation at step 3 does.

Both are reported for every model. At about 22% positives AUPRC is the more sensitive to changes in
the low-score region, and AUC alone can look stable while AUPRC moves.

### The two uncertainties

| | question | how estimated |
|---|---|---|
| **bootstrap interval** | how precisely is the ensemble's score determined by a test set of this size? | percentile interval over 2,000 isoform resamples, model held fixed (step 5) |
| **member spread** | would another fitting run have given a different answer? | standard deviation of the five members' scores (step 4) |

They describe different objects — the interval belongs to the ensemble, which is what gets reported;
the spread describes single models, which do not. They are reported side by side and are not
combined into one figure.

The member spread is estimated from five observations and is reported as a descriptive spread rather
than converted into an interval.

---

# 4. What this supports in the manuscript

Claims 5.2.1 and 5.6.4, each of which currently quotes one AUC and one AUPRC and names a window
configuration. Both are rewritten from this analysis's outputs, carrying two configurations' numbers
where they now carry one, each with its interval and member spread.
