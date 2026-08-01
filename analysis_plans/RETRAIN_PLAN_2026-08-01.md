# Retrain plan — sequence interpretability model

Specification of what is done to the data. Reasoning lives in
`RETRAIN_RATIONALE_2026-08-01.md`.

All positions in this document are **1-based transcript coordinates**: position 1 is the first base
of the transcript, and a range `[a, b]` includes both ends.

---

## 1. Purpose

> "The goal of this is a 'sequence interpretability' model that acheves good performance but is
> designed in such as way that interpretation of that model will identify sequence features that
> trigger NMD."
>
> — Pete, 2026-08-01

> "identify the best approach to use sequence-based modelling to identify sequence elements that
> influence NMD susceptibility"
>
> — Pete, 2026-08-01

---

## 2. Data

Every dataset the plan touches. All paths are on this machine.

### 2.1 `SEQ` — transcript sequences

`~/claude_projects/nmd_deposit_2026/source_data/sqanti/nmd_lungcells_corrected.fasta`

One record per transcript assembled by SQANTI3 from the long-read data: 614,992 records. A record
is the spliced mRNA sequence in 5′→3′ transcript coordinates, so position 1 is the first
transcribed base and introns are already removed. 42,043 of these records are the model universe
(§2.2); the rest are transcripts SQANTI called but the labelling pipeline did not retain.

Positions throughout this plan are 1-based transcript coordinates on this sequence.

### 2.2 `TX` — transcript labels and split

`~/claude_projects/nmd_w69_tables_2026-07-30/tx_summary.tsv`, 42,043 rows, one per transcript,
written by `export_rds.R`.

`isoform_id` keys every other table. `is_nmd` is the training label: 1 where mashr called the
transcript NMD-responsive (lfsr < 0.05 and posterior logFC > 0) in **any** of the four cell types
AT, DD, FB, MV; 0 where adj.P.Val > 0.30 in **all** four. Transcripts satisfying neither are absent
from the file. Counts: 9,425 label 1 and 32,618 label 0 (measured here from the file).

`chr` gives the split. Test is chr1, chr3, chr5, chr7 — 10,719 transcripts. Validation is chr2 and
chr4 — 4,437. Training is everything else — 26,887 (measured here). No gene spans two splits.

`tx_length` is the transcript length in bases and equals the length of the matching `SEQ` record.

### 2.3 `JUNC` — exon junction positions

`~/claude_projects/nmd_w69_tables_2026-07-30/junctions.tsv`, 95,623 rows, one per transcript in
`structures.rds` — the parsed exon-structure set built from the combined GTF, which holds both the
GENCODE reference transcripts and the SQANTI-called novel isoforms. It is a **wider set than the
model universe**: 40,486 reference and 55,137 novel entries, against 16,068 and 25,975 in `TX`
(measured here). `TX` is a strict subset of it, so every transcript this plan looks up is present.
The 53,580 extra transcripts carry no label and are not used by this plan.

`junctions` is a comma-separated ascending list of transcript coordinates. Each value is the
position of the last base before an exon–exon junction, so a junction listed at 393 means the
boundary falls between bases 393 and 394. A transcript with no junctions has an empty field.

### 2.4 `REFCDS` — the annotated coding sequence projected onto each transcript

`~/claude_projects/nmd_w69_tables_2026-07-30/ref_cds_features.tsv`, 42,063 rows, written by
`05t_ref_cds_features.R`.

For each transcript, its gene's reference isoform is the highest-DMSO-CPM isoform of that gene that
is both coding and labelled non-NMD. `ref_utr5_length` is the number of bases in the transcript
before the first base of that reference start codon, so the reference start codon begins at
transcript position `ref_utr5_length + 1`. `ref_atg_available` is 1 where that projection succeeded, and every rule in this document that
refers to the reference start codon applies only to those transcripts. `ref_utr5_length` is NA where
it is 0 — 13,288 rows — and 0 where the transcript begins at the start codon — 121 rows — so the two
are distinguished by `ref_atg_available`, never by value. `gene_id` groups transcripts for clustered
uncertainty.

The file carries 42,063 unique transcripts against `TX`'s 42,043; the 20 extra have no label and are
dropped by the join.

**"Reference start codon" throughout this document means this projection.** It is not a GENCODE
annotation: the reference isoform is chosen by expression among the gene's non-NMD coding isoforms,
so the term is label-derived and §3.7 records what follows from that.

### 2.5 `HOLDOUT` — paralog gene lists

`paralog_genes.tsv` (56 genes) and `val_paralog_genes.tsv` (19 genes) in the same directory.

Each file lists genes with a close paralog on the **other side** of a split boundary. Leakage into
the test set and leakage into the validation set are computed against different boundaries, so the
two lists are disjoint sets on disjoint chromosomes: all 56 test-side genes lie on chr1/3/5/7 and
all 19 validation-side genes on chr2/chr4 (measured here). Neither list contains a training-split
gene, so these are genes screened out of **evaluation**, not genes withheld from training.

### 2.6 `SQCDS` — the SQANTI coding-sequence call

`~/claude_projects/nmd_deposit_2026/source_data/sqanti/nmd_lungcells_classification.txt`, columns
`isoform`, `coding` and `CDS_start`. `CDS_start` is the 1-based transcript position of the first base
of the coding sequence SQANTI called, for the transcripts it calls coding. Used only to populate one
column supplied to the predictor variant (§3.6); the interpretable model never receives it.

### 2.7 Datasets this plan replaces

`orf_features.tsv` (1,540,674 rows) and `selected_orfs.tsv` (209,174 rows) are the current candidate
ORF scan and its 5-slot selection. Section 3 produces their replacements. They are not inputs to
anything specified here.

---

## 3. Build the candidate ORF pool

### 3.1 Question

Which reading frames does the model see, and does that set contain the ORFs through which NMD is
actually triggered?

### 3.2 Starting data

`SEQ`, `TX`, `JUNC`, `REFCDS`.

### 3.3 Approach

**Step 1 — enumerate every ORF in every transcript.**

For each transcript, every ATG that has an in-frame stop codon downstream defines a candidate ORF,
running from the A of the ATG to the last base of that stop. There is no minimum length: an ATG
immediately followed by a stop is a candidate. Overlapping ORFs in different frames are separate
candidates; two ATGs in the same frame sharing one stop are separate candidates.

```
for each transcript t in TX:
    s = SEQ[t]                                   # 1-based; s[a..a+2] is a codon
    L = tx_length[t]
    for a in every position where s[a..a+2] == "ATG":
        j = a + 3
        while j + 2 <= L:
            if s[j..j+2] in {"TAA","TAG","TGA"}:
                emit ORF(transcript=t, start=a, end=j+2, length=j+2-a+1)
                break
            j = j + 3
```

`start` is the position of the A of the ATG and `end` the last base of the stop codon, both 1-based,
so `length = end − start + 1` and the reference start codon is matched by `start == ref_utr5_length + 1`
(§2.4). Every later step uses these.

**Step 2 — score initiation context at each candidate's start codon.**

Each start codon is scored by the Cavener–Ray position weight matrix over the eight positions −6 to
−1 and +4, +5 relative to the A of the ATG, which is the matrix `Isopair::scoreKozakPWM` uses. The
score is the sum of log2(observed frequency / 0.25) over the positions that exist; positions running
off either end of the transcript are skipped and the remainder summed. A candidate whose start codon
has no scorable position gets no score and is not admitted.

The matrix has eight columns and is indexed by **column ordinal**, not by the position label. The
labels number the A of the ATG as +1, so a label must be converted to a displacement before it is
added to a coordinate.

```
COLS      = [-6,-5,-4,-3,-2,-1,+4,+5]         # Cavener-Ray column labels, A of ATG at +1
disp(c)   = c if c < 0 else c - 1             # label -> displacement from the A
PWM[b, k] = log2(CavenerRay_freq[b, k] / 0.25)     # k indexes COLS, 0..7

score(t, a) = sum over k in 0..7,
              where 1 <= a + disp(COLS[k]) <= tx_length[t],
              of PWM[ SEQ[t][ a + disp(COLS[k]) ], k ]
```

**Step 3 — read the admission floor.**

`FLOOR` is the MANE-calibrated Kozak threshold, `threshold_q05` in
`~/claude_projects/Isopair/inst/extdata/kozak_mane_calibration.rds` — **−1.2508**. It is the 5th
percentile of the PWM score at the annotated CDS start of 19,226 of 19,276 MANE Select transcripts
in GENCODE v49, computed 2026-04-19 by `Isopair/data-raw/calibrate_mane_kozak.R`, and it is
Isopair's own default, returned by `defaultKozakThreshold()`.

The threshold is read from the `.rds` at run time rather than copied into code, so a recalibration
propagates. The scale matches: the calibration scores with `Isopair::scoreKozakPWM` under
`.defaultKozakPWM()`, and the step-2 matrix is the validated Python port of that function.

```
FLOOR = readRDS(kozak_mane_calibration.rds)$threshold_q05      # -1.2508
```

**Step 4 — admit, discount by position, and order.**

A candidate is admitted when its score is at or above `FLOOR` **and** its start lies in the first
half of the transcript. The position rule discounts ORFs a scanning ribosome is unlikely to reach.

The reference start codon (§2.4) is always admitted, whatever its score and wherever it sits. It reaches the
pool by this rule alone for 13.3% of the transcripts that have one (measured, §3.4).

Where the pool is still empty — which happens only for transcripts with no enumerable reference
start codon — the five highest-scoring candidates are admitted.

Admitted candidates are ordered by `start` ascending, which is the order a scanning ribosome
encounters them, and the slot index is that position in the order. Slot index carries no priority:
slot 0 is the 5′-most admitted candidate, not the annotated one.

```
for each transcript t:
    L        = tx_length[t]
    cds      = the orf with orf.start == ref_utr5_length[t] + 1, if any
    admitted = [ orf for orf in ORFs(t)
                 if (score(orf) >= FLOOR and orf.start <= L/2) or orf is cds ]

    if admitted is empty:
        admitted = sort(ORFs(t), key = score, descending)[:5]

    admitted = sort(admitted, key = orf.start, ascending)
    for k, orf in enumerate(admitted):
        orf.slot = k
```

Each transcript records five counts: ORFs enumerated, ORFs at or above `FLOOR`, candidates the
position rule discounted, whether the reference start codon needed the always-admit rule, and
whether the fallback fired. The first two are what §3.4's first rows report; none of the last three
is a silent exclusion or a silent rescue.

Each **non-admitted** ORF is also classified as upstream or not, and as triggering or not, by the
same definitions §3.4 uses, so the denominators there are produced by this step rather than assumed.

**Step 5 — attach the per-candidate quantities the model is given.**

Each admitted candidate carries one supplied structural number and nothing else. `n_downstream_ejc`
is the count of junctions lying more than 50 bases past the last base of that candidate's stop
codon. This is the canonical exon-junction rule; the count is over junctions from `JUNC` for that
transcript.

```
n_downstream_ejc(orf) = count of j in JUNC[orf.transcript] where j > orf.end + 50
```

Four further quantities are **recorded in the pool table but supplied only to the predictor variant**
(§6.3): which candidate matches the reference start codon, which matches the SQANTI coding-sequence
call, and each candidate's fractional start and stop position. The interpretable model never receives
them.

The PWM score is recorded and supplied to neither variant. It decides admission in step 4 and is not
a feature.

**Step 6 — emit the pool table.**

One row per admitted candidate, replacing `selected_orfs.tsv`:

| column | definition |
|---|---|
| `isoform_id` | transcript, keys to `TX` |
| `slot` | 0-based index in 5′→3′ order among admitted candidates of this transcript |
| `orf_start` | 1-based transcript position of the A of the ATG |
| `orf_end` | 1-based transcript position of the last base of the stop codon |
| `orf_length` | `orf_end − orf_start + 1` |
| `n_downstream_ejc` | step 5. Supplied to both variants |
| `kozak_score` | step 2. Supplied to neither variant |
| `is_ref_cds` | 1 where `orf_start == ref_utr5_length + 1` and `ref_atg_available` is 1. Predictor only |
| `is_sqanti_cds` | 1 where `orf_start == CDS_start` in `SQCDS` and SQANTI calls the transcript coding. Predictor only |
| `frac_start` | `orf_start / tx_length`. Predictor only |
| `frac_stop` | `orf_end / tx_length`. Predictor only |
| `admitted_by` | one of `floor`, `reference`, `fallback` — which rule put this candidate in the pool |

### 3.4 Quantities this step reports

| quantity | population | predicted |
|---|---|---|
| candidates per transcript, no floor | 42,043 transcripts | mean 54.6 |
| ...above `FLOOR` | same | mean 36.4 |
| **...after all four admission rules** | same | **mean 19.1**, median 17, p90 35, p99 59, max 565 |
| candidates total | same | 802,035 |
| transcripts with an empty pool | same | 0 |
| candidates discounted by position | 42,043 transcripts | 732,509 |
| transcripts where the fallback fires | 42,043 transcripts | 58 |
| GENCODE CDS start in the pool | 28,775 with an enumerable CDS start | 28,775 (100%) |
| ...reaching it only by the always-admit rule | same | 3,825 (13.3%) |
| coverage of upstream ORFs whose own stop has a junction >50 bases downstream | 133,765 such ORFs | 89,062 (66.6%) |
| transcripts all of whose triggering upstream ORFs are admitted | 17,944 carrying at least one | **7,896 (44.0%)** |

"Upstream" means `orf_start < ref_utr5_length + 1`, over transcripts with `ref_atg_available` = 1;
"triggering" means the ORF's own stop has a junction more than 50 bases past it. The transcript-level
row is the one the aggregation of §6.2 Step 4 depends on: a transcript missing one of its triggering
ORFs has the wrong leak product, which the ORF-level percentage does not show.

Source: `build_orf_pool_runlog.txt`, which is this step's own output.

### 3.5 Cost

Training-tensor size scales linearly in candidates per transcript and in window width. 802,035 candidates, each with two
1000-base 9-channel windows stored as float16, is **28.9 GB** — measured by building chr21 and
scaling by candidate count, not by scaling the old file. This does not fit in the 8 GiB on this machine; the tensor is built on the cluster.

### 3.6 The tail

No cap applies. Every transcript keeps all of its admitted candidates, and batches are bucketed by
candidate count so that a batch holds transcripts of similar width. Candidates per transcript after all four admission
rules: median 17, p90 35, p99 59, maximum 565 (measured, `design5_final_pool_runlog.txt`).

If a cap is ever imposed, the pool table carries a per-transcript column recording whether it was
hit and how many candidates were dropped.

### 3.7 What the always-admit rule puts in the pool

2,493 admitted candidates — 0.3% of the pool — score below `FLOOR`: 2,423 are the reference start
codon and 70 arrive by the fallback, which by definition fires only where nothing clears the floor
(measured, `build_orf_pool_runlog.txt`). Within the pool, therefore, a below-floor score identifies the reference start codon
exactly. It identifies 8.4% of reference start codons, not all of them, and it does not recover the
transcript-level properties that carry the label — a transcript with no reference isoform is 100%
NMD-positive and a self-reference transcript 0%, and neither is inferable from which candidate is
the reference.

The consequence is narrow and directional: those 2,423 sit in the low-score tail, where the true
relation between initiation context and being the main ORF is inverted. That is the tail the
interpretation window's pre-registered check reads, so the check is reported alongside the count of
below-floor admitted candidates in the population it is computed on.

---

## 4. Prerequisite code repair

The aggregation of §6.2 Step 4 emits no attention weights, so the scripts that read them do not
carry over unchanged. Three changes, in this order.

**Step 1 — the slot count comes from the data.** `infer_uorf_attention.py:174` builds its output
columns from a literal list of five and `:207` iterates `range(5)`;
`compute_uorf_attention_metrics_pathB.R:95` selects five named columns.
`compute_uorf_attention_metrics.R:89` and `audit_uorf_attention.R:92,188` already use
`starts_with("attn_")` and need no change. The slot count is read from the tensor file, which records
it. `data_prep.py:44` (`MAX_ORFS = 5`), `config.yaml:44` and `config_dn.yaml:10`
(`max_orfs_per_tx: 5`) and `verify_determinism.py:59` carry the same literal.

**Step 2 — the exported quantities change.** Stick-breaking produces three per candidate — capture
probability `p_k`, selection probability `P_select[k]`, and decay probability `d_k` — where the
softmax produced one. `P_select` sums to at most 1 over a transcript, with the remainder being the
ribosome passing every candidate; `p_k` and `d_k` do not sum to anything. The export carries all
three, in long form, one row per (transcript, slot).

**Step 3 — the join key changes.** These scripts join on `selected_orfs.tsv`, whose columns include
`orf_rank`, `frac_position`, `kozak_score`, `is_sqanti_cds` and `n_upstream_atgs`. The replacement
table (§3.6) is keyed on `isoform_id` and `slot`. Every join is repointed.

## 5. Build the training tensor

### 5.1 Question

What array does the model consume, and what is in each channel?

### 5.2 Starting data

The pool table from §3, `SEQ`, `JUNC`, `TX`, `HOLDOUT`.

### 5.3 Approach

**Step 1 — give each candidate two sequence windows.**

Each window spans `[anchor − left, anchor + right)` and is 1000 bases wide. The anchor sits at array
index `left`, so a window that cannot be filled to its full extent is zero-padded without the anchor
moving; the reading-frame channels then mean the same thing for an ORF of any length.

| window | anchor | left | right |
|---|---|---|---|
| ATG | first base of the start codon | 900 | 100 |
| stop | middle base of the stop codon | 500 | 500 |

The ATG window is **asymmetric**: initiation depends on what a scanning ribosome has already
traversed, so the window reaches 900 bases upstream and 100 into the ORF. That covers the whole
region 5′ of the candidate for **67.0%** of admitted upstream candidates and the whole annotated
5′UTR for **85.8%** of transcripts (measured, `design6_upstream_reach_runlog.txt`).

Filling is bounded at the ORF's midpoint so the two windows never read the same base: the ATG window
fills no further than `mid`, the stop window no earlier. The bound applies to the fill, not to the
frame, and is inert whenever the ORF is longer than the window reaches.

```
mid       = (orf_start + orf_end) // 2
atg_anchor  = orf_start                       # the A of the ATG
stop_anchor = orf_end - 1                     # the middle base of the stop codon

atg[k]  = SEQ[t][ atg_anchor  - 900 + k ]  for k in 0..999, filled only where
          1 <= position <= mid          ;  anchor lands at k = 900
stop[k] = SEQ[t][ stop_anchor - 500 + k ]  for k in 0..999, filled only where
          mid < position <= tx_length[t];  anchor lands at k = 500
```

`mid` belongs to the ATG window: the ATG window may fill up to and including it, the stop window
strictly after it, so no base is read by both. Every unfilled index is zero across all nine
channels.

For an ORF longer than 1000 bases the region between `orf_start + 100` and `mid` is unobserved.
Features there are covered by the candidates that start in it, each of which carries its own windows.

**Step 2 — encode nine channels at each window position.**

| channel | content |
|---|---|
| 0–3 | one-hot A, C, G, T; all zero at a padded position |
| 4 | 1 where the transcript position at this index is listed in `JUNC` for this transcript, else 0. `JUNC` is 1-based (§2.3) and the index is converted before lookup |
| 5 | fraction of G or C over the 50 bases centred here, computed from this window's own sequence |
| 6–8 | one-hot reading frame of this position relative to this candidate's start codon |

**Step 3 — attach the structural block.**

The interpretable model's block is the single number `n_downstream_ejc` from §3 step 5. The
predictor's block is five numbers: that one plus `is_ref_cds`, `is_sqanti_cds`, `frac_start` and
`frac_stop`, all recorded in the pool table by §3.6.

Each number is centred and scaled by its mean and standard deviation over candidates belonging to
transcripts whose split label is exactly `train` — computed after step 5 assigns splits, so no
screened-out transcript contributes. Those constants are written into the tensor file and are read
back at inference rather than recomputed.

**Step 4 — assemble ragged, not padded.**

Candidates per transcript run from 1 to 565 with a median of 17 (§3.4), so padding every transcript
to the maximum would store mostly zeros. Candidates are stored as one flat array in transcript order
with a per-transcript start offset and count.

```
candidates[i]  = (atg_window[9,1000], stop_window[9,1000], structural[1 or 5])
offset[t], count[t]   such that transcript t owns candidates[offset[t] : offset[t]+count[t]]
```

**Step 5 — assign splits, screening each evaluation side against its own paralog list.**

Split comes from `chr` in `TX`: test chr1/3/5/7, validation chr2/4, training the rest. A transcript
on a test chromosome whose gene is in `paralog_genes.tsv` is labelled `test_paralog` rather than
`test`; a transcript on a validation chromosome whose gene is in `val_paralog_genes.tsv` is labelled
`val_paralog` rather than `val`. Each branch consults its own side's list: the two lists are disjoint
and sit on disjoint chromosomes, so a single shared test can never fire for one of the sides.

```
for each transcript t:
    if chr[t] in {chr1, chr3, chr5, chr7}:
        split[t] = "test_paralog" if gene[t] in paralog_genes      else "test"
    elif chr[t] in {chr2, chr4}:
        split[t] = "val_paralog"  if gene[t] in val_paralog_genes  else "val"
    else:
        split[t] = "train"
```

Transcripts on the 233 read-through composite loci — `gene_id` of the form `ENSGa::ENSGb` in
`REFCDS`, 278 transcripts — are removed from every split before this assignment. A composite is two
genes transcribed as one unit, so the split, the paralog screen and the gene clustering are all
ill-defined on it, and the paralog screen cannot see inside one.

`test_clean` is `split == "test"`, **10,520** transcripts; `val_clean` is `split == "val"`, **4,356**
(measured: of 10,719 test transcripts 122 are paralog-screened and 77 sit on composite loci; of 4,437
validation transcripts, 56 and 25). The five split labels total 41,765. An
absent `val_paralog` label is an error rather than an empty category — it means the screen did not
run.

### 5.4 Quantities this step reports

Candidates written, transcripts written, bytes on disk, and the normalization constants with the
number of training candidates each was computed over.

---

## 6. Architecture

### 6.1 Question

What does each part of the model compute, and from which inputs?

### 6.2 Approach

**Step 1 — encode each window, preserving coarse position.**

Each window passes through two convolutions, and the length axis is then reduced by **binned** max
pooling: the axis is split into `B` equal bins and the maximum is taken within each, then
concatenated. At `B = 1` this is the current global maximum, which records whether a pattern
appeared anywhere and not where; at `B > 1` the position of a pattern survives to the resolution of
one bin.

```
h = relu(bn1(conv1(x, 9 -> C, k=15)));  h = maxpool(h, 4)      # length 1000 -> 250
h = relu(bn2(conv2(h, C -> C, k=7)))
h = concat over the B parts of tensor_split(h, B, axis=-1) of max(part)   # C*B
e = linear(h, C*B -> 32)
```

250 is not divisible by 8, 16 or 32, so the split is `tensor_split`: the first `250 mod B` bins take
one extra position and the rest are equal. Bin widths on the input are 1000, 125, 63 and 31 bases.

`B` is swept over 1, 8, 16 and 32. Bins finer than the 42-base receptive field of `conv2` cannot
resolve position more precisely than that receptive field, which is what bounds the sweep at 32.

**`B = 1` is not the control.** It differs from `B = 8` in parameter count as well as in positional
resolution — 1,024 projection weights against 8,192, on a model with 34,050 — so a difference between
them is consistent with either. The control is a **`B = 8` arm whose bin order is randomly permuted
before concatenation**, once per model at initialisation and fixed thereafter: identical parameter
count, positional information destroyed. The sweep is read against that arm.

`B = 1` is also not the current architecture. The encoder at `B = 1` is the current `SequenceCNN`,
but §6 replaces the fusion, the aggregator and the head, so no arm of this sweep is the model whose
0.9310 AUC §8.3 quotes.

The only parameter count that changes is the projection: `C*B -> 32` instead of `C -> 32`, which at
`C = 32, B = 8` is 8,192 weights against 1,024.

**Step 2 — initiation head, from the ATG window.**

The initiation head reads the whole ATG window — 900 bases upstream of the start codon and 100 into
the ORF — through **its own** convolutional encoder, and emits one number per candidate through a
sigmoid. It does not read the stop window or the structural block.

The encoder is not shared with the decay head's ATG branch. Both read the same window; each learns
its own filters. A shared encoder would make "what the initiation head learned" and "what the decay
head learned" the same object, and the initiation head's filters are what the interpretation window
reads.

```
p_k = sigmoid( linear( encode_init( atg_window ), 32 -> 1 ) )
```

**Step 3 — decay head, from everything else.**

The decay head reads the full ATG embedding, the stop embedding and the structural block, and emits
one number per candidate through a sigmoid.

```
d_k = sigmoid( linear( concat[ e_atg(32), e_stop(32), relu(linear(structural)) (32) ] ) )
```

**Step 4 — aggregate by stick-breaking in transcript order.**

Candidates are already ordered 5′→3′ by slot index. Selection probability at candidate *k* is
capture at *k* times the probability that every earlier candidate was passed over. The product is
computed in log space.

```
log_pass[k] = sum over j < k of log(1 - p_j)
P_select[k] = exp( log_pass[k] + log(p_k) )
P_nmd       = sum over k of P_select[k] * d_k
```

The residual mass `exp(sum over all k of log(1 - p_k))` is the ribosome passing every candidate;
it contributes nothing to `P_nmd`.

**Step 5 — output.**

The training output is `logit(P_nmd)`, clamped away from 0 and 1 by 1e-6.

### 6.3 What each model variant receives

| | interpretable | predictor |
|---|---|---|
| structural block | `n_downstream_ejc` | that plus `is_ref_cds`, `is_sqanti_cds`, `frac_start`, `frac_stop` |
| everything else | identical | identical |

---

## 7. Training

### 7.1 Question

How is each variant fit, and what is varied?

### 7.2 Starting data

The tensor from §5.

### 7.3 Approach

**Step 1 — loss and optimizer.** Binary cross-entropy on the transcript logit against `is_nmd`.
Adam. Batches are bucketed by candidate count (§3.6).

**Step 2 — stopping.** Training stops when AUC on `val_clean` has not improved for 5 epochs; the
checkpoint with the best `val_clean` AUC is kept. Early stopping selects a checkpoint, so it runs on
the screened set.

**Step 3 — seeds.** Each configuration is trained from 5 random initialisations.

**Step 4 — sweeps.** On the interpretable variant, at 5 seeds each: `conv_channels` over 16, 32, 64,
128 at `B = 8`; `B` over 1, 8, 16, 32 at `conv_channels = 32`; and the permuted-bin control at
`B = 8, conv_channels = 32`.

The sweep is a cross, not a grid, so it establishes channel saturation **at `B = 8`** and bin
sensitivity **at `conv_channels = 32`** and nothing joint. The projection width is a function of both,
so a capacity ceiling is a property of the pair and this design cannot locate it.

**Step 5 — select one configuration, on `val_clean` only.** The configuration handed to §8 is the one
with the highest mean `val_clean` AUC over its 5 seeds. Test metrics are computed once, for that
configuration, after it is fixed. The predictor variant is trained at the same configuration so the
two differ only in the structural block.

### 7.4 Quantities this step reports

Per run: epochs to stop, `val_clean` AUC at the kept checkpoint, wall time. Per configuration: the
five seeds' `val_clean` AUC, mean and range. Test metrics are not computed here.

---

## 8. What training produces

### 8.1 Question

What is handed to the interpretation window, and what is each artefact for?

### 8.2 Approach

Training produces checkpoints, not conclusions. No comparison between model variants establishes
that a sequence feature matters; that standard is the six checks on a specific feature, and it is
the interpretation window's.

**Step 1 — hand over the two models of §6.3**, five seeds each, with their test AUC and AUPRC
reported as the mean and range over seeds. The interpretable variant is the one read. The predictor
is the reference for what accuracy costs.

**Step 2 — run two arms that can only say "stop", never "yes".**

| arm | differs by | what it can say |
|---|---|---|
| sequence-blanked | channels 0–3 **and 5** set to zero | if it matches the interpretable model, sequence carries nothing usable at this window size and interpretation will come up empty. Stop early. |
| no-junction | channel **4** zeroed **and** the structural block dropped | if it finds something the interpretable model does not, supplying the junction rule is masking it |

Channel 5 is a rolling GC fraction computed from the window's own sequence, so zeroing 0–3 alone
leaves base composition in place — the confound that accounted for all three 3′UTR motif findings.
Channel 4 lists junction positions and the stop anchor sits at a fixed index, so the 50-base rule
stays computable from the window unless channel 4 goes too; dropping only the structural block
removes the supplied number, not the information.

Blanking sets channels 0–3 to zero, which is also what a padded position looks like (§5.3 Step 2).
The blanked arm therefore cannot distinguish sequence from padding, and its result is read as
"channels 0–3 and 5 carry nothing **beyond what padding already encodes**".

Neither is evidence for anything. A difference in either direction is an operational signal about
whether to continue, and is reported as such.

**Step 3 — record, do not arm, the label question.** The 2,334 `no_ref_isoform` transcripts stay in
training. Whether they are trained on is recorded in the run metadata so the question can be asked
later against a specific finding, rather than answered in advance against nothing.

### 8.3 Quantities this step reports

Per variant and seed: `test_clean` AUC and AUPRC, epochs to stop, wall time.

Each metric carries a **bootstrap interval resampled over genes**, computed within each seed. Seeds
vary initialisation only — the split is fixed — so the range over seeds contains no sampling variance
at all, and the sampling variance it omits is inflated by clustering: `test_clean` holds 10,520
transcripts in 3,234 genes, 90.0% of them in a multi-transcript gene, and this project measured the
inflation from ignoring that at 5.47. A comparison between arms in §8.2 is read as an overlap of
gene-clustered intervals, not as a difference of point estimates.

The reference points are re-derived rather than quoted. The published 0.9310 AUC / 0.8351 AUPRC was
computed on 10,131 transcripts of a different universe with a different pool and feature set, and
AUPRC is prevalence-dependent, so it does not transport. The existing checkpoint is re-evaluated on
`test_clean` and that number is the comparison; the tabular-only GBM is refitted on the same split.
