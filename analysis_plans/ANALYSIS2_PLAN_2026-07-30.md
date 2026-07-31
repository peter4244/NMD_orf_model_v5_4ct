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
| `branch_shares_per_cell.tsv` | one cell at one level: identity, scope, the branch measures, their distribution over isoforms, and the verification anchors. 110 rows — per configuration, 25 member, 5 ensemble, 25 leave-one-out |
| `branch_decomposition_{tag}.json` | one configuration: the three shares and the stop-to-start ratio, each spread named for its level, the variance components with their F statistics, the compositional check, the test-split sensitivity, `n`, the population in words, and the reference-draw seeds |

Fifty of the first — two configurations × five members × five reference draws — one per-cell table,
and two summaries.

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

### Step 6 — reduce every cell to its branch measures

Three levels are reduced, not one. The member level is the fifty computed cells. The ensemble
level is what the analysis reports. The leave-one-out level exists because the ensemble's own
training uncertainty cannot be recovered from member summaries — the absolute value is applied
after the signed average, so a four-member ensemble has to be formed from the per-isoform
attributions themselves.

```
    nmd   = rows where label == 1                      # n emitted, expect 9,321
    m_b   = mean(|phi_b|) over nmd,  b in {structural, stop, start}
    pct_b = 100 * m_b / (m_structural + m_stop + m_start)
    ratio = m_stop / m_start

    member        (tag, seed, draw)     phi_b = that cell's attributions        25 per configuration
    ensemble      (tag, draw)           phi_b = mean over the 5 members          5 per configuration
    ensemble_loo  (tag, seed out, draw) phi_b = mean over the other 4           25 per configuration
```

**Signed values are averaged across members first; absolute value and normalization come
afterwards.** The ensemble's attribution for an isoform is exactly the mean of its members'
attributions — Shapley values are linear in the value function, and the ensemble's value function
is the mean of the members' — so this order decomposes the ensemble itself. Reversing it answers a
different question: averaging absolute values discards sign disagreement between members, and
averaging percentages is not the percentage of averaged attributions, because normalizing is not
linear.

One consequence carries into step 7. Since `|mean_s phi_s| <= mean_s |phi_s|`, wherever members
disagree in sign the ensemble carries less attribution mass than the members average, by a
branch-specific amount. **The ensemble share and the mean of the member shares therefore differ by
a bias, not only by noise**, and neither is a spread about the other's centre.

### Step 7 — each spread at its own level, and the ratio

Every spread is reported against the centre it belongs to. The interpretation spread belongs to the
ensemble; the training spread describes single members, whose centre differs from the ensemble's by
the bias named in step 6. They are never combined, and never written as one `value ± a ± b`.

```
    # the headline: the ensemble's answer, and how much the reference draw moves it
    point_estimate_b  = mean over DRAWS of pct_b^ens(R)
    sd_draw_ens_b     = sd   over DRAWS of pct_b^ens(R)
    range_draw_ens_b  = min and max over DRAWS

    # the ensemble's OWN training uncertainty, by delete-one-member jackknife
    pct_b^(-s)        = mean over DRAWS of pct_b^loo(s out, R)
    sd_seed_ens_b     = sqrt( (5-1)/5 * sum_s ( pct_b^(-s) - mean_s pct_b^(-s) )^2 )

    # the member cloud, carrying its own centre so the offset from the ensemble is visible
    member_pct_b(s)   = mean over DRAWS of pct_b(s, R)
    member_centre_b   = mean over SEEDS of member_pct_b(s)
    sd_train_mem_b    = sd   over SEEDS of member_pct_b(s)
    sd_draw_mem_b     = sd   over DRAWS of ( mean over SEEDS of pct_b(s, R) )

    # the ratio: ensemble level, log scale
    ratio_point       = exp( mean over DRAWS of log ratio^ens(R) )
    ratio_sd_draw     = exp( sd   over DRAWS of log ratio^ens(R) )     # a ×/÷ factor
    ratio_sd_seed     = exp( jackknife sd over SEEDS of log ratio^(-s) )
    ratio_member(s)   = exp( mean over DRAWS of log ratio(s, R) ), summarized over SEEDS

    # sensitivity: the population the published figures used
    repeat the above restricted to rows with split == "test" and label == 1    # n = 2,405
```

`sd_train_mem_b` and `sd_draw_mem_b` are the two comparable quantities, both member-level;
`sd_seed_ens_b` and `sd_draw_ens_b` are the other comparable pair, both ensemble-level. Spreads are
never compared across levels.

Every spread rests on five observations and four degrees of freedom, where the relative standard
error of a standard deviation is about 35%. Each is therefore given to two significant figures and
always beside its minimum and maximum, which assume nothing about the distribution.

The ratio is formed inside each cell before being summarized: its numerator and denominator come
from the same reference draw and are correlated, so a ratio of two separately averaged means is a
different quantity with no interpretable spread. It is summarized on the log scale because a
ratio's centre is multiplicative — the geometric mean satisfies `GM(stop/start) = 1 /
GM(start/stop)`, and which branch is the numerator is arbitrary. It is a ratio of two means of
`|phi|` over the summarized isoforms, never the mean of per-isoform ratios, which has an unbounded
tail wherever `|phi_start(i)|` approaches zero.

**Two diagnostics decide how those numbers are described, and both are reported.**

*Variance components.* Member × draw is a fully crossed 5×5, and a cell carries no within-cell
noise — the coalition enumeration is exact and the reference set fixed — so the interaction is
identified rather than aliased with a residual. From the 25 member cells, with
`E[MS_member] = σ²_AB + 5σ²_A`, `E[MS_draw] = σ²_AB + 5σ²_B`, `E[MS_AB] = σ²_AB` on 4, 4 and 16
degrees of freedom. Where a main effect's F against `MS_AB` is small, its marginal standard
deviation is described as including the interaction rather than as clean. A negative moment
estimate is truncated at zero and the truncation stated.

*Compositional scale.* The three shares sum to 100 by construction and so carry two degrees of
freedom, not three: a rise in one is a fall in another, a joint statement about two of them is one
fact, and the three spreads are never combined in quadrature. Shares are reported as percentages,
licensed by comparing the arithmetic mean of the five ensemble compositions against the closed
geometric mean; agreement within 0.05 percentage points is the condition. The stop-to-start ratio
is already a log-ratio coordinate of the composition, so the compositionally coherent statistic is
reported beside them.

### Step 8 — write

Three artifacts. The fifty per-isoform tables from step 4 are retained, being the only route to any
isoform-level re-analysis. One per-cell table, and one summary per configuration.

**The per-cell table carries every value the summary is computed from**, so any estimator above can
be recomputed — or replaced with a different one — without rerunning the model. 110 rows: per
configuration, 25 member, 5 ensemble, 25 leave-one-out.

| group | columns |
|---|---|
| identity | `config`, `level`, `member_set`, `member_seed_excluded`, `n_members`, `draw_id` |
| scope | `n_isoforms_summarised` |
| estimator inputs | `m_structural`, `m_stop`, `m_start`, `m_total`, `pct_*`, `ratio_stop_start` |
| distribution | `mean_signed_*`, `sd_abs_*`, `median_abs_*`, `q25_*`, `q75_*`, `frac_pos_*` |
| verification | `mean_pred_logodds`, `mean_expected_value`, `mean_gap`, efficiency residual |

`m_total` is carried because shares alone cannot show whether total attribution mass moved, and it
is what makes step 6's bias visible between levels. `sd_abs_*` is carried so that a reader who
treats the summarized isoforms as a sample rather than a census can form the isoform-level standard
error without the per-isoform tables.

The summary states, per configuration: the three shares and the ratio, each spread named for its
level, the variance components with their F statistics, the compositional check, the test-split
sensitivity, the measured `n`, the population in words, and the reference-draw seeds.

**The summarized isoforms are a census** of the labelled universe at ORF rank 0, not a sample drawn
from a larger one, so no isoform-level standard error is attached to a reported share.

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

### A difference between configurations that was predicted, and came out the other way

**The prediction, registered before any cell was run.** At the larger window width the two sequence
branches receive more input while the structural branch stays at five features, so the structural
share was expected to be **lower** at `atg2000_stop2000` than at `atg1000_stop1000`, for a reason
mechanical rather than biological. It was named in advance precisely so it would not later be read
as a finding.

**Measured, it is higher: 76.24% against 66.31%**, a difference of 9.93 percentage points in the
opposite direction, and larger than every spread in this analysis by more than an order of magnitude
— the ensemble draw standard deviations are 0.54 and 0.42 pp, and the seed jackknife standard errors
1.28 and 1.52 pp. *Measured here 2026-07-31 from the fifty accepted cells.*

The prediction is recorded rather than deleted because a registered expectation that fails is
evidence, and because the direction is no longer available as a mechanical explanation for anything
downstream. **The comparison is reported as a measured direction and magnitude, not as a test, and
not as an expected artifact.** Why widening the sequence windows *reduces* the sequence branches'
share of attributed evidence is not established here and is not claimed.

### Memory is the binding constraint, not time

`NMDDataset` materialises a split's window arrays as float32 when it is constructed. Both of the
sets a run needs are therefore read as **rows**, not as whole splits.

The reference set is 500 rows of `train`. The explained cohort is read in **chunks of 2,048
rows**, in the split's own order, each row read exactly once and each chunk released before the
next is read. Both restrictions go through the one code path that serves the whole split, so
normalization and padded-ORF handling cannot drift between them.

What a run retains is each isoform's three 32-dimensional branch embeddings, its ranks 1–4
context and its ORF mask — **59 MB over all 41,765 isoforms**. *Computed here 2026-07-30 from the
array shapes.* Window data is read once, consumed once and released.

| | one chunk, held | reference set | **peak per run** | mean over 25 |
|---|---|---|---|---|
| `atg1000_stop1000` | 0.69 GiB | 0.17 GiB | **2.10 GiB** | 1.66 GiB |
| `atg2000_stop2000` | 1.37 GiB | 0.34 GiB | **3.66 GiB** | 3.29 GiB |

*Chunk and reference figures computed here 2026-07-30 from the array shapes. Peak measured on
Explorer 2026-07-31 as the largest `sacct` MaxRSS over the twenty-five cells of each
configuration, array job 8850292 — the largest rather than the mean, because a request is sized
against the worst cell it must hold, and on the machine the request is made to.*

The request is **8 GB**, 2.2× the largest measured peak. Peak exceeds the sum of the parts above
because the sequential chunk allocations leave the allocator's high-water mark above any single
chunk's footprint.

Read eagerly instead, one full-cohort run holds 14.00 GiB at `atg1000_stop1000` and 28.01 GiB at
`atg2000_stop2000`, and peaks half an array higher again — 35.3 GiB — because the float16 read
buffer and the float32 result are alive together. `slurm_kernel_shap_dn.sh` requests 32 GB.

The chunk width is a multiple of the 256-row extraction batch, so each forward pass receives the
same rows in the same order it would receive from a single read of the whole split.

### Cost, and where this runs

One full-cohort run takes **5m31s to 10m58s**, and the fifty run as one array in **24 minutes**
of wall clock. *Measured on Explorer 2026-07-31, array job 8850292, fifty tasks at up to
twenty-three concurrent across fifteen nodes.* On the development machine's CPU the same runs
take 6m16s and 8m50s at the two configurations, which is what makes a full-cohort cell a local
debugging step rather than a cluster round trip.

Each configuration is run at full cohort on the development machine before any cell of it is
submitted. The fifty are then submitted to the cluster's **CPU** partition as one array, one task
per cell.

**The device is chosen on queue capacity, not on per-job speed.** The same cell — 
`atg2000_stop2000`, member 100, draw 1 — was run on both: **7 min 53 s on a GPU node, 8 min 36 s
on a CPU node**, so a GPU is about 8% faster per job. It is also the scarcer resource, at 24
usable nodes against roughly 125, with a fivefold spread between GPU node families on record
here. Fifty jobs therefore finish sooner on the CPU partition despite the per-job penalty.
*Measured on Explorer 2026-07-31: jobs 8849990 (gpu, node d1012) and 8849989 (short, node
c0625), `sacct` elapsed.*

Every run's progress output is unbuffered, so a log distinguishes a working job from a dead one
while it is still running. The recorded failures on that cluster — a job that died before its log
existed, an exit code that reported success while the log reported failure, a process killed
silently on a login node — were all found by submission rather than by running, which is the
slowest way to find them.
