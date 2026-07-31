# Analysis 3 Plan — which structural features the model uses

*Written 2026-07-31, before any result exists.*

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

Analysis 2 established that the structural branch carries the majority of the attributed evidence.
Analysis 3 opens that branch: **which of its five features, and in what proportion.**

---

# 2. Data

## 2.1 Terms

Datasets A and B are those of Analysis Plans 1 and 2, and their terms — isoform, NMD susceptible,
split, configuration, seed, member, ensemble, log-odds, branch, reference draw, ORF rank 0 — carry
over unchanged. Three terms are new or sharpened.

**Structural feature.** One of the five numbers describing an ORF's placement and context, which
together form the structural branch's entire input. In array order:

| # | name | one entry | at ORF rank 0 |
|---|---|---|---|
| 0 | `frac_start` | ORF start position ÷ transcript length | 0.000–0.970, mean 0.188 |
| 1 | `frac_stop` | ORF end position ÷ transcript length | 0.015–1.000, mean 0.535 |
| 2 | `is_ref_cds` | 1 if the ORF's start matches the reference CDS start | binary; 1 in 68.3% |
| 3 | `is_sqanti_cds` | 1 if it matches the TransDecoder2 CDS start | binary; 1 in 79.9% |
| 4 | `n_downstream_ejc` | count of exon junctions downstream of this ORF's stop codon | 0–33, mean 1.077 |

*Measured here 2026-07-31 from `results_4ct_dn/nmd_orf_data.h5`, at ORF rank 0 over all 41,765
isoforms.* The five enter the model through one linear layer, `nn.Linear(5, 32)` followed by ReLU
(`model.py:105,126`); nothing else reaches the structural branch.

`n_downstream_ejc` is the feature carrying the best-established trigger of nonsense-mediated decay.
It exists because junctions beyond the stop window are invisible to the sequence encoders
(`data_prep.py:51`).

**Feature attribution.** How much one structural feature contributed to one isoform's prediction,
in log-odds. Absence is operationalized exactly as in Analysis 2 — by substituting the value that
feature takes in a reference isoform, and averaging the model output over the 500 of them. With
five features all 32 present/absent combinations are evaluated, so the calculation is **exact**
rather than approximate, and the five attributions plus the baseline sum to the prediction to
machine precision.

**A feature decomposition is not a sub-division of a branch decomposition.** Analysis 2 played a
three-player game whose players were branches; Analysis 3 plays a five-player game whose players are
the structural features, with the two sequence branches held at their observed values. Shapley
values are not additive under regrouping: the five feature attributions do **not** sum to the
structural branch's attribution from Analysis 2, and no quantity from one analysis may be divided by
or subtracted from a quantity in the other. The two are reported side by side and never composed.

## 2.2 Dataset A — model inputs

Unchanged from Analysis Plan 2: one file, `results_4ct_dn/nmd_orf_data.h5`, 41,765 isoforms in one
shared row order. Analysis 3 reads `labels`, `isoform_id` and `split` directly, and the model
consumes the window arrays, `orf_features`, `orf_mask` and the two `normalization/` arrays.

**The features are normalized before scoring**, by subtracting `normalization/orf_feat_mean` and
dividing by `normalization/orf_feat_std`. Attribution is computed on the model's input scale, which
is the normalized one; a share is invariant to that choice but a signed attribution is not, so the
scale is stated wherever a signed value is reported.

**Padded ORF slots carry zeroed normalized features.** This affects the 1.2% of isoforms with fewer
than five ORFs and is a property of the code path that serves every consumer, not of this analysis.

## 2.3 Dataset B — trained members

Unchanged: ten checkpoints, two configurations at five seeds each, `results_4ct_sweep/`. The model
contract of Analyses 1 and 2 applies unchanged and is not restated — normalization applied, ORF mask
passed, evaluation mode, window widths from the tag and never from the checkpoint's embedded config.

## 2.4 Dataset C — outputs, written to `results_interp_all/`

| file | one row |
|---|---|
| `feature_shap_{tag}_seed{S}_all_run{R}.tsv` | one isoform: `isoform_id`, `label`, `prediction`, `baseline_struct_reference`, `shap_frac_start`, `shap_frac_stop`, `shap_is_ref_cds`, `shap_is_sqanti_cds`, `shap_n_downstream_ejc`, `shap_sum`, `residual` |
| `feature_shares_per_cell.tsv` | one cell at one level, as in Analysis 2: identity, scope, the five feature measures, their distribution over isoforms, verification anchors |
| `feature_decomposition_{tag}.json` | one configuration: the five shares, each spread named for its level, the variance components, the compositional check, the test-split sensitivity, `n`, the population in words, the reference-draw seeds |

Fifty of the first, one per-cell table, two summaries.

---

# 3. Analysis 3

## 3a. Scientific question

The structural branch carries the majority of the model's attributed evidence. **Within it, how is
that evidence divided among the five features?**

The question is a test of whether the model recovered the mechanism rather than a correlate. The
junction-dependent account of nonsense-mediated decay predicts that `n_downstream_ejc` should
dominate: a stop codon with exon junctions downstream is the canonical trigger. Two of the other
four features, `is_ref_cds` and `is_sqanti_cds`, are annotation-agreement flags rather than
mechanism, and weight on them would indicate the model leaning on how an ORF was called rather than
on where it sits.

And, as everywhere in this work: **how much would the answer have differed under a different
training run, or a different draw of reference isoforms?**

## 3b. Starting data

| | dataset | what is used |
|---|---|---|
| inputs | **Dataset A** | all 41,765 isoforms |
| models | **Dataset B** | all ten members |
| outputs | **Dataset C** | written by this analysis |

**Population for summarization:** isoforms with `label == 1`, at ORF rank 0 — 9,321 isoforms, a
census of the labelled universe rather than a sample from it. Identical to Analysis 2, so the two
decompositions describe the same isoforms.

## 3c. Approach

Eight steps, mirroring Analysis 2 so the two are directly comparable. Steps 1 and 2 are shared with
it and are re-run rather than inherited, so this analysis stands alone.

```
TAGS   = [atg2000_stop2000, atg1000_stop1000]
SEEDS  = [100, 200, 300, 400, 500]
DRAWS  = [1, 2, 3, 4, 5]
NBG    = 500
BASE   = 42                             # draw R uses RandomState(BASE + 1000*R)
FEATS  = [frac_start, frac_stop, is_ref_cds, is_sqanti_cds, n_downstream_ejc]
```

### Step 1 — define the population

As Analysis 2 step 1: read `labels`, `isoform_id` and `split`; assert 41,765 entries and 9,321
positives; record the split per isoform for the step 7 sensitivity.

### Step 2 — fix the reference draws

**The same five reference sets as Analysis 2**, drawn from the train split only, uniformly and
without replacement, using `RandomState(42 + 1000*R)`.

Reusing them does **not** make a feature share and a branch share comparable — those are fractions
of different totals in different games, and §2.1 forbids composing them. What it buys is narrower
and real: the two analyses are played against identical baselines, so the empty coalition of this
five-player game is exactly the structural-absent state of the three-player game, the draw-to-draw
variation is paired across the two analyses, and any difference between them is a property of the
games rather than of the reference sets they happened to get.

**All absent features take their values from the same reference isoform, jointly.** Not
independently per feature. This is what makes the five-player game nest inside the three-player one;
drawn independently, the reused draws would buy nothing.

The draw is unstratified, so "absent" means *this feature took the value it takes in a typical
training transcript* — a mixture that is about 22% NMD susceptible, not a control.

### Step 3 — encode what does not vary, once per member

The two sequence branches and the ORFs at ranks 1 to 4 are held at their observed values throughout,
so their embeddings are computed once per isoform per member and reused across all 32 coalitions.

```
for TAG in TAGS:
  for SEED in SEEDS:
    model = member(TAG, SEED)                        # eval mode; normalization applied
    for every isoform i:
        start_vec[i], stop_vec[i] = the two sequence branch embeddings at ORF rank 0
        ctx[i] = embeddings of ORFs 1..4
        m[i]   = orf_mask[i]
        x[i]   = the five normalized structural features at ORF rank 0
```

### Step 4 — evaluate the 32 coalitions and solve for the attributions

For each isoform the model is run with each subset of the five features present. Features not in the
subset take their values from the reference isoforms, and the prediction is averaged over the 500 of
them. The exact Shapley value of each feature follows from the 32 coalition values by enumeration.

```
    for R in DRAWS:
      for every isoform i:
        for each of the 32 subsets S of FEATS:
            build the rank-0 feature vector: observed for features in S, reference for the rest
            v[S] = mean over the 500 reference isoforms of
                   model(start_vec[i], stop_vec[i], struct_fc(that vector), ctx[i], m[i])
        phi[0..4]      = shapley_from_32_coalitions(v)
        prediction     = v[FEATS]
        baseline_struct_reference = v[{}]          # NOT Analysis 2's expected_value: there all
                                                   # three branches are absent, here only the
                                                   # structural features are. Different quantity,
                                                   # so deliberately a different column name.
        residual       = prediction - baseline_struct_reference - sum(phi)
      write Dataset C row per isoform
```

Only the structural branch's own encoder is re-evaluated per coalition; the sequence encodings are
not. The 32 coalitions cost roughly four times Analysis 2's eight, against an unchanged encoding
cost, so a run is expected to take between one and two times Analysis 2's — *predicted here
2026-07-31 from Analysis 2's measured split between encoding and coalition evaluation; to be
replaced by measurement before the fifty are submitted.*

### Step 5 — accept or reject each file, and record the tally

Identical in form to Analysis 2 step 5, and affordable for the same reason: all 32 coalitions are
enumerated and the Shapley values solved exactly, so the sum identity is an algebraic consequence of
the arithmetic and the residual can only be floating-point rounding.

**The 1e-12 tolerance is inherited as a prediction, not a measurement.** Shapley weights over five
players accumulate more rounding than over three. It is verified on the first cell before the fifty
are submitted; landing at 1e-11 would be a tolerance to restate, not a defect — and discovering it
after fifty jobs would be waste. The two tolerances are kept distinct: 1e-12 for the within-file
algebraic identity, and the wider cross-code-path bar for the prediction agreement above.

```
    for each of the 50 files:
        assert row count == 41,765
        assert the isoform_id vector equals Dataset A's order      # every later step is positional
        assert the label vector matches step 1
        assert prediction agrees with Analysis 2's for the same (tag, member), which is an
               INDEPENDENT per-isoform value and is what catches a file whose value columns
               are attached to the wrong isoforms
        r = max|residual|
        accept if r < 1e-12, else record (tag, member, draw, r) and STOP
    report accepted, rejected, and every rejected cell with its residual
```

The tally is reported whether or not anything is rejected.

### Step 6 — reduce every cell to its feature measures

As Analysis 2 step 6, over five players rather than three, at three levels — member, ensemble, and
leave-one-out ensemble — giving 110 rows.

```
    m_f   = mean(|phi_f|) over the summarised isoforms,  f in FEATS
    pct_f = 100 * m_f / sum over FEATS of m_f
```

**Signed values are averaged across members first; absolute value and normalization afterwards**,
for the reason given in Analysis 2 step 6, and with the same consequence: the ensemble and member
centres differ by a bias, not only by noise.

### Step 7 — each spread at its own level

Identical in structure to Analysis 2 step 7: ensemble point estimate over the five draws with its
draw spread and range; the ensemble's own training uncertainty by delete-one-member jackknife; the
member cloud reported with its own centre; the two-way variance components on the crossed 5×5 as a
licensing diagnostic; the compositional check; and the sensitivity restricted to `split == "test"`
and `label == 1`, n = 2,405.

**The composition has five parts and four degrees of freedom.** The five shares sum to 100 by
construction, so a rise in one is a fall in the others, and the five spreads are never combined in
quadrature.

**And here the compositional check has a failure branch, because it may actually fail.** Analysis 2
passed it by two orders of magnitude with three comparable parts. Five parts, one of which is
expected to dominate, may not. If the arithmetic mean and the closed geometric mean of the
compositions differ by 0.05 percentage points or more, the shares are additionally reported in
centred log-ratio coordinates, whose centre is the closed geometric mean by construction, and the
percentages are described as descriptive only. A part measuring exactly zero is reported as such
and excluded from the log-ratio form rather than allowed to produce an infinity.

**No ratio is formed.** Analysis 2's stop-to-start ratio existed because two comparably sized
branches invited a ranking. Here the question is whether one feature dominates, which the shares
answer directly; a ratio between two of five parts would name a comparison nobody asked for.

### Step 8 — write

The fifty per-isoform tables, one per-cell table, and one summary per configuration, with the
contents listed under Dataset C. The summarised isoforms are a census, so no isoform-level standard
error is attached to a reported share.

## 3d. Statistical models and estimators

**No hypothesis test is performed.** The two configurations are reported side by side and are not
tested against each other.

### The estimator

A feature share is `mean_i |phi_f(i)| / sum over features`, the mean taken over the summarised
isoforms. It is a share of **mean absolute log-odds displacement** — not variance explained, not an
information-theoretic quantity, and not "importance" in any sense that survives being quoted without
its definition.

### The two annotation flags are anti-correlated, not duplicated

`is_ref_cds` and `is_sqanti_cds` both assert that the ORF's start matches an externally annotated
coding sequence, which invites treating them as near-duplicates whose credit Shapley would split.
**Measured, they are not.** At ORF rank 0 they agree on 56.9% of all 41,765 isoforms and on only
**31.0%** of the 9,321 summarized ones, where their phi coefficient is **−0.477**.
*Measured here 2026-07-31 from `results_4ct_dn/nmd_orf_data.h5`.*

The mechanism is the ORF ranking rule. Tier 1 selects an ORF whose start matches the reference CDS;
tier 2 selects one matching the TransDecoder2 CDS **when it differs from the reference**. The rule
that chooses rank 0 therefore makes the two flags alternatives wherever the annotations disagree,
and both are 1 only on the 52.6% of isoforms where the two annotations coincide.

Two consequences, both of which change what is reported:

- **The two shares are individually meaningful.** There is no duplicate-splitting artifact to
  correct for, and neither share should be described as half of a shared contribution.
- **A merged player answers a different question** — how much annotation agreement of any kind
  contributes — and it is computed exactly rather than by adding shares. Summing two
  mean-absolute shares overstates the merged contribution wherever the two attributions differ in
  sign, which is the absolute-value-before-sum error this plan forbids between members.

**The merged player costs nothing.** The four-player game over `{frac_start, frac_stop, annotation,
n_downstream_ejc}` needs only the coalitions in which both flags are present or both absent — 16 of
the 32 already evaluated, and no additional model evaluations. Its four-part composition is
reported beside the five-part one **as a separate game, not a derived row**: merging two players
changes the normalizer, so a four-part share is not the sum of two five-part shares.

### What the decomposition does and does not establish

It establishes how the model's structural evidence divides among the five inputs it was given. It
does not establish that the model implements the junction rule: `n_downstream_ejc` dominating is
consistent with that account and is the prediction it makes, but the feature is a count the model was
handed, not a mechanism it discovered.

It says nothing about which ORF the model relied on — ranks 1 to 4 are held fixed throughout — nor
about the sequence branches, which are held at observed values and are not players in this game.

---

# 4. What this supports in the manuscript

Claim 5.14, *"Of individual ORF structural features, EJC count was by far the most important"*, and
Figure 5 Panel C, the ranked per-feature structural importance. Both are rewritten from this
analysis's outputs, carrying two configurations' numbers where they now carry one, each share with
its interpretation spread (as both a standard deviation and a standard error), its training spread,
the ensemble-size bias, and the range across all fifty member cells — which is a wider envelope than
the range across the five draw-averaged members and is reported as its own field.
