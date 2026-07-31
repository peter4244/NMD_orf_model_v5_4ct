# Analysis 2 Plan — what kinds of information the model uses

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

Analysis 2 addresses the second: **what kinds of information.**

---

# 2. Data

## 2.1 Terms

Datasets A and B are those of Analysis Plan 1; their terms — isoform, NMD susceptible, split,
configuration, seed, member, ensemble, log-odds — carry over unchanged. Four terms are new or
sharpened here.

**Branch.** One of the three inputs the model reads: the sequence window around the start codon, the
sequence window around the stop codon, and the five structural features. Each is encoded separately
into a 32-dimensional vector, and the three are fused into one per-ORF representation.

**The ORF scan.** Before any of this, every isoform's sequence was searched exhaustively for open
reading frames — every ATG with an in-frame stop codon at least 9 nucleotides downstream. That scan
produced 1,540,674 ORFs across 42,043 isoforms, a median of 32 per isoform. Everything below selects
from those.

**ORF rank 0.** The highest-priority of an isoform's five candidate ORFs. Priority is decided by
asking, in order, whether a *scanned ORF starts at exactly the position an externally annotated
coding sequence says it should*:

1. **Reference CDS.** The reference isoform for this gene has an annotated coding sequence, and its
   start codon has been mapped onto this isoform — recorded as `ref_utr5_length`, the length of this
   isoform's 5′UTR under that mapping, so the expected start is `ref_utr5_length + 1` in this
   isoform's own 1-based coordinates. The tier is satisfied when some scanned ORF begins at that
   exact position. Not nearby, not in the same frame — the same base.
2. **TransDecoder2 CDS.** The same test against the CDS start called by TransDecoder2, taken from
   the SQANTI classification file, when it differs from the reference.
3. **Highest Kozak score.** If neither annotation lands on a scanned ORF, the best-scoring ORF by
   Kozak context is taken instead.

Ties within a tier are broken by Kozak score; the remaining slots up to five are filled by descending
Kozak score.

Measured at rank 0 over all 42,043 isoforms: **68.3% reference CDS, 27.4% TransDecoder2, 4.3%
Kozak-only**. *Measured here 2026-07-30 from `results_4ct_dn/selected_orfs.tsv`, `selection_reason`
at `orf_rank == 0`.* So rank 0 is the annotated reference coding sequence for about two thirds of
isoforms and a predicted ORF for the rest.

Tier 1 can fail two ways, and both are recorded in the `category` field that the subgroup definitions
use: the reference start codon may not map onto this isoform at all, because the exon carrying it is
absent; or it may map to a position where the scan found no valid ORF. Those cases are what section 4
calls *Ref AUG absent*.

Every attribution in this analysis is computed at rank 0.

**Branch attribution.** How much one branch contributed to one isoform's prediction, in log-odds.
Absence is operationalized by substituting a reference isoform's encoded branch. With only three
branches, all eight present/absent combinations are evaluated, so the calculation is **exact** rather
than approximate, and the three attributions plus the baseline sum to the prediction to machine
precision.

**Reference draw.** The 500 training isoforms whose encoded branches stand in for "branch absent".
Which 500 are drawn is random, and it is the only stochastic element of this analysis.

## 2.2 Dataset A — model inputs

Unchanged from Analysis Plan 1: **one file**, `results_4ct_dn/nmd_orf_data.h5`, holding named arrays
that share one row order over 41,765 isoforms. Analysis 2 reads `labels`, `isoform_id` and `split`
directly, and the model consumes the window arrays, `orf_features`, `orf_mask` and the two
`normalization/` arrays.

**Two properties of Dataset A that this analysis depends on and Analysis 1 did not.**

Each window is centered on the **middle** nucleotide of its codon, at array index `W/2`. Analysis 2
does not use positional coordinates, so this affects nothing here; it is restated because the
positional analyses that follow do.

The five ORF slots are ordered by priority and `orf_mask` marks which hold a real ORF. Rank 0 is slot
0. 98.8% of isoforms fill all five slots; 1.2% fill fewer.

## 2.3 Dataset B — trained members

Unchanged from Analysis Plan 1: ten checkpoints, `results_4ct_sweep/best_model_{tag}_seed{S}.pt`, two
configurations at five seeds each, verified as ten distinct models.

**The model contract is the same as Analysis 1's and is repeated because getting any of it wrong
changes the attributions without raising an error.** The structural features are normalized using the
stored mean and standard deviation before scoring; the ORF mask is passed rather than ignored; the
model runs in evaluation mode so batch normalization uses its stored running statistics; the two
window widths come from the tag, never from the configuration copy stored inside the checkpoint,
which does not describe the model it belongs to.

**One requirement specific to this analysis.** The attribution is computed at ORF rank 0 with ORFs 1
to 4 **held at their observed values throughout**. The decomposition therefore does not take the
whole prediction apart. Its baseline is the model's prediction when rank 0's three branches are drawn
from reference isoforms and the other four ORFs are left alone, and the three attributions sum to
*prediction minus that baseline*.

## 2.4 Dataset C — outputs, written to `results_interp_all/`

| file | one row |
|---|---|
| `kernel_shap_branch_{tag}_seed{S}_all_run{R}.tsv` | one isoform: `isoform_id`, `label`, `prediction`, `expected_value`, `shap_atg`, `shap_stop`, `shap_structural`, `shap_sum`, `residual` |
| `branch_decomposition_{tag}.json` | one configuration: the three shares and the stop-to-start ratio, each with both spreads, plus `n`, the population in words, and the reference-draw seeds |

Fifty of the first — two configurations × five members × five reference draws — and two of the second.

`shap_atg` is the **start**-branch attribution; the column keeps the legacy name.

---

# 3. Analysis 2

## 3a. Scientific question

The model reads three things: sequence around the start codon, sequence around the stop codon, and
five structural features including the count of exon junctions downstream of the stop codon. When it
calls an isoform NMD susceptible, **how is the evidence divided among the three?**

This is a test of whether the model recovered known biology. Junction position downstream of the stop
codon is the best-established trigger of nonsense-mediated decay and enters through the structural
branch. Stop codon identity and its immediate context enter through the stop window. Upstream ORFs
and initiation context enter through the start window. A decomposition dominated by structural
features would say the model reached the junction-dependent account from data alone; substantial
weight on the stop window would point toward stop codon context.

Two quantities answer it. The **structural share** — what fraction of the attributed evidence the
structural features carry. The **stop-to-start ratio** — how much more the stop codon context matters
than the start codon context.

And, as everywhere in this work: **how much would either have differed under a different training run,
or a different draw of reference isoforms?**

## 3b. Starting data

| | dataset | what is used |
|---|---|---|
| inputs | **Dataset A** | all 41,765 isoforms, with `labels`, `isoform_id`, `split`, and the arrays the model consumes at each configuration's widths |
| models | **Dataset B** | all ten members |
| outputs | **Dataset C** | written by this analysis |

Nothing else is read.

**Population.** The claims are about how the model reaches a *positive* call, so controls are
excluded: **isoforms with `label == 1`, at ORF rank 0**. Over the full universe that is 9,321
isoforms. *Measured 2026-07-30 from `results_4ct_dn/nmd_orf_data.h5`.*

The attributions are computed over every isoform and the restriction is applied when they are
summarized, so the same files also support the `test_clean` sensitivity analysis in step 7 and the
subgroup breakdowns later analyses need.

*The published percentages 60.7 / 28.8 / 10.5 reproduce exactly only under a `label == 1` restriction
on the test split; over all rows of the same file they are 56.3 / 32.1 / 11.6. Measured here
2026-07-30 from the deposited `kernel_shap_branch_atg500_stop500.tsv`, n = 2,268 of 10,131.*

## 3c. Approach

Eight steps. Steps 1 and 2 run once and are shared; steps 3 to 8 run once per configuration.

```
TAGS   = [atg2000_stop2000, atg1000_stop1000]
SEEDS  = [100, 200, 300, 400, 500]      # which trained member
DRAWS  = [1, 2, 3, 4, 5]                # which reference draw
NBG    = 500                            # reference isoforms per draw
BASE   = 42                             # config training seed; draw R uses BASE + 1000*R
```

### Step 1 — define the population

Read the per-isoform arrays and record what is being explained. Unlike Analysis 1 this is the whole
labeled universe, not a split; the split is recorded per isoform so step 7 can restrict to it.

```
read labels, isoform_id, split from Dataset A          # 41,765 entries, one shared order
n      = 41,765
y      = labels                                        # 1 = NMD susceptible
ids    = isoform_id                                    # alignment key
assert n == 41,765 and sum(y) == 9,321
record  n_test_clean = count(split == "test")          # expect 10,520, used at step 7
```

### Step 2 — fix the reference draws

The reference set is what "branch absent" means. Five sets of 500 are drawn from the **train split
only**, so no isoform is ever its own reference and no held-out data enters the comparison. The draw
is uniform and without replacement, and the same five sets are reused across both configurations and
all members — so a difference between two members is a difference between models rather than between
the reference sets they happened to get.

Within a run, all 41,765 explained isoforms are attributed against the same 500. The draw is the
only stochastic element in the analysis, which is what makes it a clean interpretation-variance knob.

```
train_pos = positions within the train split           # 26,711 isoforms
for R in DRAWS:
    ref[R] = 500 positions drawn from train_pos, uniformly, without replacement,
             using RandomState(BASE_SEED + 1000 * R)   # BASE_SEED = 42, the config's training seed
    assert the five sets are distinct from one another
    record the label composition of each
```

**The draw is not stratified**, so each reference set inherits the training split's composition:
about 22% NMD susceptible, 78% control. *Measured here 2026-07-30 over three draws from
`results_4ct_dn/nmd_orf_data.h5`: 109, 117 and 104 NMD of 500, against 5,959 of 26,711 (22.3%) in
train.*

So "absent" operationally means *this branch looked like a randomly chosen training transcript* — a
mixture that is mostly, but not entirely, controls — and `expected_value` is the model's mean
prediction over that mixture. A control-only reference would answer a cleaner question, how far a
branch pushes the prediction away from a non-NMD transcript. The uniform draw is retained because it
is what the published figures used and changing it would break comparability, but the choice is
recorded here rather than inherited silently.

### Step 3 — encode every isoform once per configuration

The three branch encoders are deterministic given a member, so each isoform's three 32-dimensional
branch vectors are computed once and reused across all eight coalitions. This is the dominant cost of
the analysis and is not repeated per coalition.

```
for TAG in TAGS:
  for SEED in SEEDS:
    model = member(TAG, SEED)                          # eval mode; normalization applied
    for every isoform i:
        e[i] = (start_vec, stop_vec, struct_vec)       # three 32-dim vectors, ORF rank 0
        ctx[i] = embeddings of ORFs 1..4               # held fixed for every coalition
        m[i]   = orf_mask[i]
```

### Step 4 — evaluate the eight coalitions and solve for the attributions

For each isoform, the model is run with each subset of rank 0's three branches present. Absent
branches are replaced by the reference isoforms' corresponding vectors and the prediction is averaged
over the 500 of them. The exact Shapley value of each branch follows from the eight coalition values
by enumeration — no sampling and no approximation.

```
    for R in DRAWS:
      for every isoform i:
        for each of the 8 subsets S of {start, stop, structural}:
            build rank-0 vector: observed for branches in S, reference for those not in S
            v[S] = mean over the 500 reference isoforms of model(rank-0 vector, ctx[i], m[i])
        phi_start, phi_stop, phi_struct = shapley_from_8_coalitions(v)
        prediction     = v[{start, stop, structural}]
        expected_value = v[{}]
        residual       = prediction - expected_value - (phi_start + phi_stop + phi_struct)
      write Dataset C row per isoform
```

### Step 5 — accept or reject each file, and record the tally

**The residual is a checksum, not a constraint.** With three branches all eight coalitions are
enumerated and the Shapley values solved exactly, so
`phi_start + phi_stop + phi_struct = prediction - expected_value` is an algebraic consequence of the
arithmetic. Nothing rescales to make it hold. The residual can therefore only be floating-point
rounding — around 1e-15 — and anything materially larger means the file did not come from the
calculation it claims to: coalitions evaluated against different reference sets, a model left in
training mode, a mislabelled coalition, or a file assembled from two runs. It does not say which.

This is why a hard reject is affordable here and would not be for an approximate method, where a
small non-zero residual is expected and tells you nothing.

```
    accepted = 0; rejected = []
    for each of the 50 files:
        assert row count == 41,765 and label vector matches step 1
        r = max|residual|
        if r < 1e-12:  accepted += 1
        else:          rejected.append((tag, member, draw, r))   # reason recorded, not just count

    report accepted, len(rejected), and every rejected cell with its residual
    if rejected: STOP -- a rejected cell is a defect to diagnose, not a cell to drop
```

**The tally is reported whether or not anything is rejected**, in the run log and in the summary
file. A silently dropped cell would leave the spreads computed over fewer than 25 cells while the
output still said 25, and an unreported exclusion is the reporting failure rather than the fix.

Rejection stops the analysis rather than proceeding on the survivors. A residual above tolerance
means an upstream fault that would also affect the cells that happened to pass.

### Step 6 — reduce each file to three numbers

```
    nmd   = rows where label == 1                      # n emitted, expect 9,321
    m_b   = mean(|shap_b|) over nmd,  b in {structural, stop, start}
    pct_b = 100 * m_b / (m_structural + m_stop + m_start)
```

Fifty rows result, one per (configuration, member, reference draw).

### Step 7 — the two spreads, the ensemble, and the ratio

The ensemble's attribution for an isoform is the mean of its members' attributions, because members
are combined by averaging log-odds and attributions are linear on that scale. **Signed values are
averaged across members first, and absolute values and normalization are applied afterwards.**
Reversing either order answers a different question: averaging absolute values discards sign
disagreement between members, and averaging percentages is not the percentage of averaged
attributions, because normalizing is not linear.

```
    # the ensemble's own answer — the headline figure
    for R in DRAWS:
        join the 5 members on isoform_id
        phi_b(i) = mean over members of shap_b(i)      # signed, before absolute value
        apply step 6 to the averaged table  ->  pct_b(R)
    point_estimate_b    = mean over DRAWS of pct_b(R)
    interpretation_sd_b = sd   over DRAWS of pct_b(R)

    # the member distribution — how much the answer depends on the training run
    for SEED: member_pct_b(SEED) = mean over its 5 draws of pct_b
    training_sd_b = sd over the 5 member_pct_b

    # the stop-to-start ratio, formed inside each cell
    for each of the 25 cells: ratio = m_stop / m_start
    ratio point estimate, and both spreads, from those 25 values

    # sensitivity: the population the published figures used
    repeat step 6 restricted to rows with split == "test" and label == 1
```

The ratio is formed per cell and summarized afterwards. Its numerator and denominator come from the
same reference draw and are correlated, so a ratio of two separately averaged means is a different
quantity with no interpretable spread.

### Step 8 — write

Two files per configuration: the fifty per-isoform tables from step 4, and one summary carrying the
three shares, the ratio, both spreads for each, the measured `n`, the population in words, and the
reference-draw seeds.

## 3d. Statistical models and estimators

**No hypothesis test is performed.** This analysis produces estimates with uncertainty. The two
configurations are reported side by side and are not tested against each other.

### The estimator

A branch share is `mean_i |phi_b(i)| / sum over branches`, the mean taken over NMD-susceptible
isoforms. It is a share of **mean absolute log-odds displacement** — not variance explained, and not
an information-theoretic quantity. The published wording, "60% of the predictive information," names
neither the population nor the quantity, and both must be stated when the number is rewritten.

The three shares sum to 100% by construction within every cell, so they are not independent and no
uncertainty statement should imply they are.

### What the decomposition does and does not partition

Shapley values divide credit among branches that may carry overlapping information; where two
branches encode the same signal, the credit is split between them rather than assigned to one. The
published phrase "the remainder coming from sequence information" asserts a clean partition that this
does not support, and is a wording change to be made alongside the number.

The decomposition covers rank 0 with ORFs 1 to 4 held fixed, so it says nothing about which ORF the
model relied on. That question belongs to the attention analysis.

### The two spreads

| | question | how estimated |
|---|---|---|
| **interpretation spread** | given this model, how precisely is the share determined? | sd across five reference draws of the ensemble's share |
| **training spread** | would another fitting run have given a different share? | sd across five members of their per-member shares |

They describe different objects — the interval belongs to the ensemble, the spread to single models —
and are reported side by side rather than combined. Each is estimated from five observations and is
reported as a descriptive spread.

### An expected difference between configurations, named so it is not read as a finding

At the larger window width the two sequence branches receive more input while the structural branch
stays at five features. The structural share is therefore expected to be lower at
`atg2000_stop2000` than at `atg1000_stop1000` for a reason that is mechanical rather than
biological. The comparison is reported as a direction and magnitude, not as a test.

### Memory is the binding constraint, not time

`NMDDataset` materialises a whole split's window arrays as float32 when it is constructed
(`utils.py:248-249`). Analysis 2 constructs two of them per run — the explained cohort, and the
train split from which 500 reference isoforms are drawn:

| | train split | explained cohort (`all`) | **per run** |
|---|---|---|---|
| `atg1000_stop1000` | 9.0 GB | 14.0 GB | **23 GB** |
| `atg2000_stop2000` | 17.9 GB | 28.0 GB | **46 GB** |

*Measured here 2026-07-30 from the array shapes; confirmed by two test runs at
`atg1000_stop1000` that spent minutes at full CPU without emitting a line, which was the load
rather than the computation.*

The train split is loaded in full to use **500 of its 26,711 rows**. That is the whole of the
excess, and it explains a note in the record that reads as caution but is not:
`slurm_kernel_shap_dn.sh` requests 32 GB, and the handoff observes that kernel SHAP's "narrow
margin is memory, not time". At 46 GB a full-cohort run at the wider configuration would exceed
that request.

**Before any of the fifty runs, the reference set is read as 500 rows rather than as a split.**
The normalization and padded-ORF handling must come from the same code path that serves the
explained cohort, so the change belongs inside `NMDDataset` as an optional restriction to a subset
of a split's rows, not as a second reader in this script. `NMDDataset` is shared with
`evaluate.py`, `deepshap.py` and `ensemble_evaluate.py`, so the change is additive with an
unchanged default, and is verified by scoring one member before and after and asserting identical
predictions.

### Cost, and where this runs

Measured 2026-07-30 on this machine's CPU: the coalition evaluation is about 5 ms per isoform, and
the branch encoding that precedes it dominates, at roughly 10 minutes per run over 41,765 isoforms.
That is **~13 minutes per run and ~11 hours for all fifty** locally, against roughly an hour on the
cluster with the runs in parallel.

The code is written and proven on a small subset locally before any cluster submission. The
recorded failures on that cluster — a job that died before its log existed, an exit code that
reported success while the log reported failure, a process killed silently on a login node — were all
found by submission rather than by running, which is the slowest way to find them.
