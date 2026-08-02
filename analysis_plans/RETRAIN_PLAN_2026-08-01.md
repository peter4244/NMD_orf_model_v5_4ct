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

Training-tensor size scales linearly in candidates per transcript and in window width. 802,035 candidates at two 1000-base windows,
one byte per position, is **1.6 GB** before compression — measured by building chr21 and scaling by
candidate count. Storing the nine channels as float16 instead costs 28.8 GB, which exceeded the
cluster's per-user quota. This does not fit in the 8 GiB on this machine; the tensor is built on the cluster.

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

**Step 4 — assemble ragged, and store one byte per position rather than nine channels.**

Candidates per transcript run from 1 to 565 with a median of 17 (§3.4), so padding every transcript
to the maximum would store mostly zeros. Candidates are stored as one flat array in transcript order
with a per-transcript start offset and count.

Eight of the nine channels are binary and the ninth is derivable from them, so the windows are stored
as **one `uint8` per position per window** and the channels are reconstructed in the data loader:

| bits | meaning |
|---|---|
| 0–2 | fill state: 0 unfilled, 1–4 for A, C, G, T, 5 filled but not one of those |
| 3 | a junction lies at this transcript position |

"Filled but not ACGT" is a distinct state from "unfilled": an N inside the window counts toward the
GC denominator and carries a reading frame, while a position outside the window or past the midpoint
clip carries neither.

```
codes[i]       = uint8[2, 1000]                # ATG window, stop window
orf_start[i], orf_end[i]                       # for the reading-frame channels
structural[i]  = float32[5]
offset[t], count[t]   such that transcript t owns candidates[offset[t] : offset[t]+count[t]]
```

Encoding and decoding are one definition, in `tensor_io.py`.

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
before concatenation, redrawn per candidate at every forward pass**: identical parameter count,
positional information destroyed. The sweep is read against that arm.

The permutation must be redrawn rather than fixed at initialisation. The projection that follows is
fully connected over the flattened bins, so it can undo any fixed reordering by permuting its own
weights, and a fixed-permutation arm would measure nothing.

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

**The headline metric is the five-seed ENSEMBLE**, and the ensembling rule is the mean predicted
**probability** — the arithmetic mixture of the members, which does not depend on the link function.
The mean-logit ensemble is reported beside it; the two differ by about 0.0005 on `test_clean` and
selecting whichever is higher after seeing both would be a selection effect, however small. The
seed-mean is also reported, because it is the single-model quantity and the two are not
interchangeable.

**The ensemble carries its own bootstrap interval.** The seed-mean statistic is the mean over seeds
of per-seed AUC; the ensemble is the AUC of the averaged prediction. They are different estimators
and an interval computed for one does not cover the other, so each is resampled separately.

**The interval reported with a headline metric is the ordinary bootstrap over transcripts.** The
gene-resampled interval is computed and reported beside it as a **check**, together with their
variance ratio, which is the design effect on that metric. Applying a clustering correction by
default would assume the thing this project has twice got wrong: the design effect is a property of
the statistic, not of the dataset. Measured on `test_clean`, AUC for the selected configuration gives
**1.00** — gene and transcript resampling are indistinguishable, sd 0.00343 against 0.00344 — while
the matched-pair statistic of §8.5 gives 3.15 on the same transcripts. Both are measured, neither is
carried.

Transcripts of one gene are not independent, but they are far less dependent than this project has
been assuming. Measured over the test chromosomes: 41.0% of multi-transcript genes carry **both**
labels, sibling pairs share a label 75.2% of the time against 64.8% expected by chance, and the ICC
of `is_nmd` within gene is **0.300** at an effective cluster size of 3.36 — a design effect of
**1.71**, so an interval widens by 1.31×. Isoforms of one gene having different NMD fates is the
phenomenon this model exists to predict, and the clustering correction has to be sized to that rather
than to an assumption that siblings agree.

The **5.47** previously cited here and in `SEQUENCE_DISCOVERY_BRIEF.md` is not a design effect. It
originates in `exp2b_control_validity_runlog.txt` as `sd 5.47`, a standard deviation in percentage
points of a bootstrap distribution, reported beside a companion `sd 3.69` in a section comparing two
spreads to test power matching. It concerns neither genes nor clustering.

The label ICC above bounds nothing directly: what governs the interval is how the model's *errors*
correlate within a gene, which is why both resamplings are computed on the predictions rather than
inferred from the labels.

Seeds vary initialisation only — the split is fixed — so the range over seeds contains no sampling
variance at all. It is reported beside the interval and labelled differently.

**No interval here covers the split.** Test is four fixed chromosomes, so resampling within them
cannot express what a different chromosome partition would give, and that variation is plausibly
larger than either interval. A reported metric states the estimand it is conditional on: this model,
these four chromosomes.

**A comparison between arms is a paired difference, bootstrapped on the same gene resamples.** Both
arms are scored on the same transcripts, so most of the sampling uncertainty is shared and cancels;
asking whether two separately computed intervals overlap is a strictly weaker test and can hide a
real difference. One resample of genes is drawn per replicate and every arm is scored on it, and the
interval reported for a comparison is the interval of the difference.

**The permuted-bin arm's score is a draw, not a number.** It redraws its bin permutation at every
forward pass, so it is evaluated under `R` draws and **the average is taken over AUCs, not over
predicted scores** — averaging the scores would build an ensemble, which is a different and better
model than the one under evaluation. `R` is not fixed in advance: the across-draw spread is measured,
and `R` is raised until that spread divided by √`R` is small against the gene-clustered half-width.
Both quantities are reported.

### 8.4 The two arms that answer the position question

The comparison that identifies the permutation is **`interp_c32_b8` against `interp_c32_b8_perm`,
matched**, and it is run whatever configuration §7 step 5 selects. Comparing the *selected*
configuration against `c32_b8_perm` would confound bin count with the permutation whenever selection
does not land on `c32_b8`, and it would set a maximum over seven configurations against an arm that
was not selected that way. The selected configuration's own `test_clean` metrics are reported
separately, for the accuracy claim rather than for this one.

`interp_c32_b1` is not this comparison and is not read as it: it differs from `c32_b8` in parameter
count as well as in positional resolution (§6.2 step 1).

**AUC is secondary evidence for the position claim, and this is stated before the numbers exist.**
The binned pooling of §6.2 step 1 exists to make position *readable*, not to raise accuracy, and a
model can encode position that barely helps prediction or route around it and score the same. The
primary evidence is:

1. the **selection-mass profile** of §9.3 step 7, which reads the initiation head's selectivity
   directly; and
2. the **positional profile of the mutagenesis bank**, mean effect by offset from the anchor,
   compared between `c32_b8` and `c32_b1`. Structure at specific offsets in one and a flat profile in
   the other is position being used, measured on the mechanism.

Because AUC is secondary, a small AUC difference is **not** evidence against position either. That
symmetry is recorded here so it cannot be applied in one direction only.

**Where the two arms disagree, and on which transcripts.** A mechanism that decides a small fraction
of transcripts produces a small average difference and is still a mechanism, so the per-transcript
disagreement between the two arms is examined as well as its mean. Concentration alone is weak —
two initialisations also disagree somewhere — so the disagreement is stratified by three quantities
**named here before the predictions exist**, to keep this from becoming a search:

| stratum | why it is a place position should matter |
|---|---|
| upstream ORF count | more upstream candidates, more for a positional rule to do |
| slot of the reference start codon | the 11.2% of transcripts whose main ORF sits beyond slot 9 (§9.4) |
| stop-to-junction distance, banded at 50 bases | the one distance relation the previous architecture provably could not express |

Disagreement that concentrates **and** concentrates on these is a mechanism. Disagreement that
concentrates but scatters across them is initialisation noise with a pattern read into it.

The reference points are re-derived rather than quoted. The published 0.9310 AUC / 0.8351 AUPRC was
computed on 10,131 transcripts of a different universe with a different pool and feature set, and
AUPRC is prevalence-dependent, so it does not transport. The existing checkpoint is re-evaluated on
`test_clean` and that number is the comparison; the tabular-only GBM is refitted on the same split.

### 8.5 The matched-pair initiation test, pre-registered

**Written before `test_clean` is read, and before the configuration is selected.** The exploratory
version of this analysis ran on validation and is not reported: two independent geometric leaks were
found in it, each of which inflated the estimate, and the corrected residual did not survive
gene-clustered inference at that sample size. This section fixes the analysis so that the test-set
run is confirmatory rather than another exploration.

> **AMENDED 2026-08-01, before `test_clean` was read.** Reviewed by the interpretability window
> against a mechanics run of `test85_matched_pair_initiation.py` on validation. Four changes, all of
> which make the claim **harder** to support, not easier; every one is recorded here rather than
> applied silently, because the same change made after the read would not be defensible.
>
> 1. **The primary is now capture against chance AND capture against `kozak_score`**, both
>    gene-clustered, rather than against chance alone. Clearing 0.5 does not answer "compared to
>    what", and on validation it cleared by 0.0066 while `orf_length` beat capture three to one.
> 2. **`orf_length` and 5′ proximity move from baselines to bounding context**, and are reported
>    with the statement that they use information `p_capture` cannot access. `p_capture` reads the
>    ATG window only, and the full-right-fill filter guarantees both members carry all 100
>    downstream bases, so capture *cannot see* ORF length while the baseline can. That is a model
>    against an oracle, and is not evidence the model failed at its own task. `kozak_score` is the
>    information-matched comparison: strictly less context than capture (~10 nt against 900).
> 3. **The direction split requires intervals, not point estimates.** Both arms' gene-clustered
>    intervals must exclude 0.5. The downstream arm was n=45 on validation and is expected near 110
>    on test; "both point estimates above 0.5" is not a claim at that size.
> **HELD 2026-08-01, unread.** Written, amended and dry-run on validation, then deliberately NOT
> run on `test_clean` — the ISM bank of §9 measures the capture arm directly (`vals_capture`) where
> this asks the same question through a pairwise proxy, so §8.5 is held until the bank is processed
> and may not be needed. **The test split is unspent and the design is frozen, so the
> pre-registration still stands if it is revived.**
>
> Recorded because it bears on that: the ISM subset spans every split, test included, per the
> project's standing convention that only AUC and AUPRC are test-only while interpretability uses
> all data. So a later decision to run or skip §8.5 will have been made with test-derived model
> behaviour already in view. That does not touch the frozen design — nothing about §8.5 may be
> tuned to what the bank shows — but if §8.5 is ever reported, this ordering is reported with it.
>
> Two questions left open at the hold, both to settle **before** any test read:
> - **The Kozak arm is underpowered by construction.** It is computed only on pairs where capture
>   and the matrix disagree — 270 on validation, perhaps 650 on test. A wide interval failing to
>   exclude 0.5 is absence of evidence, not equivalence, and "inconclusive" should be pre-committed
>   as a legitimate outcome rather than written around afterwards.
> - **The ATG window carries exon junction marks (channel 4).** Two candidates 50 bases apart hold
>   different junction patterns, so a residual preference may be annotation rather than nucleotides.
>   §8.5 controls window fill and position; it does not separate sequence from junction placement.
>
> 4. **The direction split's conclusion is narrowed to what it establishes.** "Both above 0.5 is
>    evidence no positional account can produce" overclaims: it excludes a *monotone* positional
>    preference only. A preference peaked at a particular distance from the midpoint cap puts both
>    arms above 0.5 and is purely positional, and references sit at a characteristic distance from
>    that cap by construction. Read "no **monotone** positional account can produce."

**The question.** Does the capture head prefer the annotated start codon over a competing start codon
in the same transcript, once every geometric property of the reading window is held fixed?

**The two leaks this design controls.** Both live in *which positions of the ATG window are filled*,
and neither is visible to an ablation over channels.

| leak | mechanism | control |
|---|---|---|
| 5′ padding | the window reaches 900 upstream, so its left fill boundary is `max(901 − orf_start, 0)` — distance to the 5′ end, exactly | pairs matched on `orf_start` to ±50, and the direction split below |
| midpoint clip | fill stops at the ORF midpoint, so right-hand fill is a readout of ORF length. Reference coding sequences are long and competing ORFs are usually very short | **both** candidates required to have the full 100 bases of right-hand fill |

Right-hand fill is **whatever `window_spans` returns**, and this section deliberately does not restate
it. An earlier draft gave it as `min(100, (orf_length // 2) + 1)`, which is exact when `orf_length`
means the span but one too many when read with the pool's `orf_length`, which is inclusive
(`orf_end - orf_start + 1`). That reading overstates fill by one on every odd span under the cap —
36% of candidates — and would admit candidates whose true fill is 99 as though it were 100, in the
one place the midpoint-clip control is enforced. A restatement of an authority is a thing that drifts
from it; the authority is the function that built the tensor.

**Construction.** Over `test_clean`:

```
eligible   = candidates with kozak_score >= -1.2508          # both arms cleared the same bar
pairs      = (r, c) with r reference, c not, same transcript,
             |orf_start[r] - orf_start[c]| <= 50,
             right_fill[r] == 100 and right_fill[c] == 100
statistic  = fraction of pairs where p_capture[r] > p_capture[c],
             ties counted as one half
```

`p_capture` is the mean over the five seeds of the configuration selected in §7 step 5; the per-seed
range is reported beside it.

**Inference is a bootstrap resampling GENES**, not pairs — the pairs of one gene are not independent
and the design effect measured on this statistic in the exploratory run was 3.15, against 1.71 for
the label-level ICC. The pair-resampled interval is reported alongside so that ratio is visible
again on the test split.

**The claim is supported only if the gene-clustered 95% interval excludes 0.5 AND the
gene-clustered interval for capture against `kozak_score` on the identical pairs also excludes 0.5.**
If either includes 0.5 that is reported as the result. The section is drafted to report a null before
the split is read, not after.

**Pre-specified secondary — the direction split.** Matching on `orf_start` to ±50 still leaves one
member upstream. The statistic is reported separately for pairs where the reference is upstream and
where it is downstream, **each with its own gene-clustered 95% interval**. A monotone positional
preference must push one of those below 0.5; **both intervals excluding 0.5** is evidence no
*monotone* positional account can produce. It does not exclude a non-monotone one — a preference
peaked at a particular distance from the midpoint cap is purely positional and puts both arms above
0.5 — and references sit at a characteristic distance from that cap by construction.

**Pre-specified comparisons, on the identical pairs**, ties counted as one half throughout.

*Information-matched, and part of the primary claim:* `kozak_score` — strictly less context than
`p_capture`, ~10 nt against 900.

*Bounding context, reported with the statement that they use information `p_capture` cannot access:*
`orf_length` and 5′ proximity (`−orf_start`). The ATG window carries 900 upstream and 100 into the
ORF, and both members carry all 100 by the filter above, so ORF length is not visible to the model
and is visible to these.

*Also reported:* `n_downstream_ejc`. Expect this **below** 0.5 rather than at it: reference ORFs are
long, so their stops sit 3′-proximal and carry fewer downstream junctions than their short
competitors. Below 0.5 here is signal in reverse, not a failed baseline.

*Design guard, not a baseline:* right-hand window fill is constant within every pair by construction,
so every comparison is a tie and the value must be **exactly 0.5000**. Anything else means the
full-right-fill filter did not engage.

*The tabular model:* a gradient-boosted model predicting `is_ref_cds` from the other tabular columns,
fitted on `train` candidates and scored on the test pairs — §8.3's own model is transcript-level and
cannot pick a within-pair winner, and "with `is_ref_cds` withheld, since that column is the label of
this test" only coheres if the baseline is predicting it. Run **twice**, with and without
`is_sqanti_cds`: that column fingers the same candidate about two-thirds of the time
(`P(is_ref | is_sqanti)` = 0.63 over the pool), so with it the number is a ceiling on what any tabular
model could do and without it an honest baseline, and the gap measures how much of the tabular signal
is annotation echo. Expect the without version near `orf_length`'s rate regardless, since
`orf_length` and `orf_start` are among its inputs.

**Reported whatever it shows**, with the number of pairs, the number of genes, and the pairs-per-gene.


---

## 9. The in-silico mutagenesis bank

### 9.1 Question

What does the trained model's transcript-level prediction do when each base of a transcript is
substituted, over what population is that measured, and which transcript positions can be measured
at all?

### 9.2 Starting data

The tensor from §5, the pool table from §3, `TX` (§2.2) for `is_nmd`, `tx_length` and `chr`,
`REFCDS` (§2.4) for `gene_id`, and the checkpoints from §7 step 5 and §8 step 2.

The bank is consumed by the interpretation window, which computes every metric from it. This section
produces the bank and the population statement that travels with it, and stops there.

### 9.3 Approach

**Step 1 — partition the genes into a discovery half and a confirmation half.**

The population is the 41,765 transcripts the tensor holds: the pool's 42,043 less the 278 on the 233
read-through composite loci, which §5 step 5 removes from every split and which therefore cannot
appear in a bank built from the tensor. A composite is two genes transcribed as one unit, so an arm
assignment on it is ill-defined for the same reason a split is.

Each of the 12,380 `gene_id` values remaining (measured, §9.4) is assigned to `discovery` or
`confirmation` by an independent fair draw from a generator seeded at 20260801, over genes taken in
sorted order so the assignment depends on the gene set and the seed and not on row order in any input
file. A gene is assigned once, over the whole universe rather than over any subset, so the assignment
does not change when the subset of §9.3 step 2 grows.

Every transcript inherits its gene's arm. A gene lies entirely on one chromosome, so the arm never
crosses the `train` / `val` / `test` boundary of §5 step 5, and a transcript's arm and its split are
separate facts that both travel with the bank.

```
for each gene g in REFCDS:
    arm[g] = "discovery" if rng.random() < 0.5 else "confirmation"
for each transcript t:
    arm[t] = arm[gene_id[t]]
```

Paralogy is **not** screened across this boundary. The two lists of `HOLDOUT` (§2.5) are computed
against the test and validation boundaries and say nothing about this one, so a close paralog pair
can be split across the two arms. The count of transcripts whose gene has a paralog anywhere is
reported (§9.4) rather than removed.

**Step 2 — fix the order in which transcripts enter the bank.**

The 41,765 transcripts of step 1 are put in one fixed random order by the same generator, drawn
independently within each `is_nmd` stratum and interleaved so that any prefix of the order holds
their prevalence, 0.2232. The bank of size *n* is the first *n* transcripts of that order.

A larger bank is therefore the same bank with rows appended: the subset at *n* = 1,000 is a prefix of
the subset at *n* = 2,000, and step 1's arm assignment is unchanged by the growth. The order is
written out in full so *n* is a parameter of the run rather than a property of the file.

```
order = interleave( shuffle(transcripts with is_nmd == 1),
                    shuffle(transcripts with is_nmd == 0),
                    at the prevalence 0.2232 )
subset(n) = order[:n]
```

Position *i* of the order takes a positive when `floor((i+1) * 0.2232)` exceeds `floor(i * 0.2232)`,
so the positive count of any prefix is within one of `i × 0.2232` rather than only correct in
expectation.

**Step 3 — establish which transcript positions are measurable.**

A candidate's two windows each cover a contiguous run of transcript positions, fixed by §5 step 1:

```
mid_k    = (orf_start_k + orf_end_k) // 2
ATG_k    = [ max(1, orf_start_k - 900),      min(tx_length, mid_k, orf_start_k + 99) ]
STOP_k   = [ max(mid_k + 1, orf_end_k - 501), min(tx_length, orf_end_k + 498) ]

covering(p) = { (k, ATG)  : p in ATG_k }  union  { (k, STOP) : p in STOP_k }
valid(p)    = covering(p) is not empty
```

A position outside every window is read by no part of the model and carries no effect to measure.
Over the 42,043 transcripts, 77.8% of positions are covered and 17.0% of transcripts are covered
completely; 95.3% of the uncovered mass lies 3′ of the last window, because §3 step 4 admits only
candidates starting in the first half of the transcript and the 3′-most stop window reaches 498 bases
past the 3′-most stop codon (measured, §9.4). `valid` is emitted so that an uncovered position is
distinguishable from a measured zero.

**Step 4 — define one substitution.**

`sub(t, p, b)` replaces the base at transcript position *p* of transcript *t* with base *b*, in
**every** window in `covering(p)`, and recomputes channel 5 of each of those windows from that
window's own bases.

Everything else is held fixed: the junction channel, the reading-frame channels, the structural
block, the candidate coordinates, and the candidate set itself. A substitution can create or destroy
an ATG or a stop codon, and the pool is **not** re-derived when it does; the bank measures the
model's response to its input, not to a re-scanned transcript.

Applying the substitution to one window and not the others would present the same transcript
coordinate as two different bases within one forward pass, which is a state no transcript can occupy.

```
sub(t, p, b):
    for (k, w) in covering(p):
        i = index of p in window w of candidate k        # p - anchor(k,w) + left(w)
        fill_state[k, w, i] = b
        channel5[k, w] = rolling_GC(window w of candidate k, span 50)
```

The observed base `obs(p)` is read back from any covering window. All covering windows encode the
same transcript position, so they agree; disagreement is an error in the geometry rather than a
property of the data.

**Step 5 — recompute only what the substitution changed.**

The model runs in evaluation mode, so batch normalization uses its running statistics and dropout is
off. Each candidate's two embeddings then depend only on that candidate's own windows, and a
candidate no window of which contains *p* has exactly the values it had in the unperturbed pass.

```
base pass:  for every candidate k, compute and keep
            e_init_k = enc_init(atg_k),  e_atg_k = enc_atg(atg_k),  e_stop_k = enc_stop(stop_k)

for a substitution at p:
    for (k, ATG)  in covering(p):  recompute e_init_k and e_atg_k
    for (k, STOP) in covering(p):  recompute e_stop_k
    p_k, d_k  from the current embeddings, for every k
    logit     from the stick-breaking aggregation of §6.2 step 4, over all K candidates
```

The aggregation is recomputed over the whole transcript every time, because the stick-breaking
product couples the candidates: an earlier candidate's capture changes every later candidate's
selection mass. Only the encoders are cached among the model's own computations. That cache is
4.2× fewer encoder passes than re-encoding every candidate for every substitution (measured, §9.4).

**Step 5c — the decoded window is cached too, and a substitution patches it.**

The nine channels of §5 are reconstructed from the stored codes for every window the model reads.
Rebuilding all 1,000 positions per substitution was 88.2% of the run on a V100, against 4.4% for
all three encoders (measured, `analysis_plans/profile_ism_cluster.py`; the same profile on a
laptop says the opposite, because there the encoders are on the CPU).

A substitution changes about 51 of the 1,000 positions: the substituted base in channels 0–3, and
the ±25 span over which channel 5 averages local GC. Channel 4 does not change, because a junction
is annotation. Channels 6–8 do not change, because the frame grid is anchored on the candidate's
own start codon, which the substitution does not move, and because a base is replaced by a base so
the fill mask is unchanged. Each candidate's two windows are therefore decoded once and kept on
the device, and a substitution copies the decoded window and patches that span
(`window_cache.py`).

The patch is **bitwise equal** to a full decode, not close to one. Channel 5 is `num / den` where
both are counts over at most 1,000 positions and so are exact integers in float32; a substitution
moves the count over the span by exactly −1, 0 or +1 and leaves the denominator alone, so
`(num ± 1) / den` is the same division of the same two exact integers a full recompute performs.
The three possible numerators are precomputed per candidate, so the substitution path performs no
channel-5 arithmetic at all.

Equality is checked rather than argued, by `verify_window_cache.py`, which compares the patched
window against `decode_windows` on real windows of both kinds and requires zero differing entries.
The check reports how many substitutions took the channel-5 branch and how many skipped it, and
fails if either is zero: half of all substitutions leave GC status unchanged, and a check drawn
only from those would pass without testing the span at all.

**Step 5b — the response variable is the unclamped log-odds.**

The model's training output clamps `P(NMD)` to [1e-6, 1 − 1e-6] and forms the logit from the clamped
probability. Both operations destroy the response to a substitution, in different places:

- at the clamp the logit is constant, so **every** substitution returns exactly 0.0 — measured at
  `z_d` = −14 with 6 candidates, where the true response is 7.916e-02;
- away from the clamp the round-trip through a float32 probability loses it — measured at `z_d` = +16,
  where `P` is 0.984 and its float32 spacing exceeds the change the substitution makes.

Both are indistinguishable from a position the model does not use, and the no-op floor is zero in
both, so the floor does not catch either.

The bank's response is therefore `log P(NMD) − log(1 − P(NMD))` formed from `log P(NMD)` directly,
never through `P`, with the second term evaluated in two regimes split at log(1/2), and the whole
aggregation carried in float64. It equals the training output to 4.2e-06 wherever the clamp is
inactive. The encoders remain float32; only the arithmetic over the *K* candidate values changes.

The model's own output layer is unchanged — the clamp is correct for training. `base_logit_training`
and `pinned_in_training` record which transcripts that output would have pinned, so that regime can
be conditioned on rather than inferred.

**Step 6 — every position carries its own baseline, computed at the same batch shape.**

`sub(t, p, obs(p))` substitutes the observed base for itself, and **the other three substitutions at
that position are differenced against it** rather than against the unperturbed pass of step 5.

The reason is that the encoder's output for a fixed input row depends on how many rows share its
batch. Measured on this model in evaluation mode, there are three regimes — batch 1, batch 2 to 7,
and batch 8 upward — that differ by 3.278e-07; row position within a batch does not matter. The
unperturbed pass runs at batch *K* and a chunk of substitutions runs at the chunk's own size, so a
difference taken between them carries that offset whenever the two fall in different regimes. *K* is
below 8 for 6,537 of 42,043 transcripts, so the offset is present for 15.5% of the cohort and absent
for the rest: an error **correlated with candidate count**, not noise.

Differencing within one chunk cancels it exactly. All four bases are therefore evaluated at every
position, and the fourth is the baseline rather than a check.

The distance between each chunk's baseline and the unperturbed pass is recorded as
`batch_shape_offset` — the quantity removed. It is measured at every valid position, and it is the
scale an effect is read against: a difference this pipeline produces from a change that is not a
substitution at all.

**Step 7 — emit the bank.**

One file per (arm, seed). `W` is the largest `last_covered` over the transcripts in the subset, so no
stored column is invalid for every transcript.

| name | shape | type | content |
|---|---|---|---|
| `vals` | (n_iso, W, 4) | float32 | the response of step 5b under `sub(t, p, b)` minus its unperturbed value; NaN where `valid` is false, and at the observed base |
| `valid` | (n_iso, W) | bool | `valid(p)` of step 3 |
| `obs` | (n_iso, W) | int8 | observed base, ACGT = 0123; −1 where `valid` is false or the base is not one of ACGT |
| `labels` | (n_iso,) | int8 | `is_nmd` |
| `batch_shape_offset` | scalar attribute | float | step 6, the maximum over every valid position and draw |
| `transcript_id` | (n_iso,) | string | keys to `TX` |
| `spans` | (n_cand, 6) | int32 | one row per candidate: transcript row index, slot, and the four bounds of `ATG_k` and `STOP_k` |
| `cand_offset`, `cand_count` | (n_iso,) | int32 | the rows of `spans` belonging to each transcript |
| `p_capture` | (n_cand,) | float32 | `p_k` of §6.2 step 2, the initiation head's output, unperturbed |
| `p_select` | (n_cand,) | float32 | `P_select[k]` of §6.2 step 4, unperturbed |
| `p_decay` | (n_cand,) | float32 | `d_k` of §6.2 step 3, unperturbed |
| `arm` | (n_iso,) | string | `discovery` or `confirmation`, from step 1 |
| `split` | (n_iso,) | string | `train`, `val`, `val_paralog`, `test`, `test_paralog`, from §5 step 5 |
| `gene_id` | (n_iso,) | string | for clustering |
| `base_logit` | (n_iso,) | float64 | the unperturbed response of step 5b |
| `base_logit_training` | (n_iso,) | float32 | what the model's own output layer emits, clamp included |
| `pinned_in_training` | (n_iso,) | bool | `|base_logit| >= 13.8155`, where that clamp is active |

Provenance travels as file attributes beside these: the checkpoint path, the model configuration, the pool `sha256`, the split file, the number of paired draws, the number of positions the floor was sampled at, and a one-paragraph statement of what one `vals` entry means.

`p_capture`, `p_select` and `p_decay` share `spans`' row order but are separate datasets: `spans` is
geometry and survives a change of checkpoint, these are model outputs and do not.

`p_capture` ships alongside `p_select` rather than only the product. `P_select[k]` is `p_k` times the
probability that every earlier candidate was passed over, so it confounds "this candidate is a strong
initiation context" with "everything upstream of it was weak"; a check on initiation context reads
the first, and only `p_select` decides whether a substitution can move the output at all.

**A candidate carrying no selection mass cannot move the transcript logit however its sequence
changes.** A zero there is "not expressible", which is the same distinction as `valid` against a
measured zero, one level up. Stick-breaking halves the mass at every slot when `p_k` is near 0.5, so
this is not a rare corner: at `p_k` = 0.5 exactly, 99.8% of the mass falls in slots 0 to 9, while
73.3% of transcripts hold more than 10 candidates and 37.6% hold more than 20 (measured, §9.4).

The depth the mass reaches is a **function of the fitted `p_k`**, so the same number is a readout of
whether the initiation head learned anything: selectivity and depth are one axis, and mass dying
early says `p_k` is close to uniform. It is measured on the selected checkpoint before any bank is
built, and reported as a result rather than only as a restriction.

`spans` is emitted because the interpretation window excludes positions within 25 bases of a window
boundary, where a substitution perturbs the truncated denominator of the 50-base rolling mean of
channel 5. The bounds are stated rather than left to be reconstructed from the pool table.

**Step 8 — the arms.**

| arm | model | bank |
|---|---|---|
| interpretable | the configuration selected in §7 step 5 | full, **one per seed** |
| permuted-bin control | `interp_c32_b8_perm` | full, one per seed, by step 9 |
| sequence-blanked | §8.2 | **none** — see below |
| predictor, no-junction | §8.2, §6.3 | none; §8.3 metrics only |

A bank is built for **each of the five seeds** of an arm, over the same subset and the same
discovery/confirmation split, and shipped as five files. Seeds differ in initialisation only, so a
sequence feature present in one and absent in the other four is a property of that initialisation
rather than of the model, and nothing in the gene-level split of step 1 can detect that — it holds
the seed fixed and varies the transcripts. Requiring a finding to survive across seeds is the check
that varies the other axis.

The sequence-blanked arm is trained and evaluated with channels 0–3 and 5 set to zero, and a
substitution changes channels 0–3 and 5 and nothing else. Every entry of its `vals` is therefore
exactly zero by construction, whatever the model learned, so the arm has no mutagenesis bank. Its
evidence is the §8.3 accuracy comparison.

**Step 9 — the permuted-bin control is stochastic, and is measured with common random numbers.**

The control redraws its bin permutation at every forward pass (§6.2 step 1), so a single forward pass
of it is a draw rather than a prediction. Its effect at a position is the expectation over
permutations, estimated by pairing: one permutation is drawn **per candidate and per encoder** and
held fixed across the unperturbed pass and every substitution of that transcript, the whole bank is
computed under it, and the result is averaged over `R` such draws.

Per encoder, not per window: `enc_init` and `enc_atg` both read the ATG window, and in training each
draws its own permutation at every pass. Sharing one between them at bank time would pair two
encoders that the trained model never paired.

```
for r in 1..R:
    P_r = one permutation per (candidate, encoder), drawn once   # 3 encoders, 2 windows
    vals_r = the bank of steps 4-7 computed with every forward pass using P_r
vals = mean over r of vals_r
```

Pairing is what makes this measurable: an unpaired difference of two passes is dominated by the
permutation, not by the substitution. The size of that domination is measured rather than assumed —
the standard deviation of the **unperturbed** logit across permutation draws is what an unpaired
difference would carry as noise, and it is reported beside the effect it would swamp (§9.4).

`R` is set from the spread across paired draws, on the trained control checkpoint, and is not set in
advance: the ratio of that spread to the effect is a property of the fitted weights, and a value read
off an untrained model does not transfer. The training path is unchanged — a permutation supplied to
the encoder replaces the draw, and supplying none redraws exactly as §6.2 step 1 specifies.

### 9.4 Quantities this step reports

Measured over all 42,043 transcripts of the pool, by
`measure_ism_geometry.py` / `measure_ism_geometry_runlog.txt`:

| quantity | population | measured |
|---|---|---|
| filled ATG-window positions per candidate | 802,035 candidates | mean 742 of 1000, median 906 |
| filled stop-window positions per candidate | same | mean 613 of 1000, median 537 |
| transcript positions covered by ≥1 window | 42,043 transcripts | mean 2,353 of 3,273 |
| fraction of the transcript covered | same | mean 0.778, median 0.755 |
| transcripts covered completely | same | 7,156 (17.0%) |
| share of uncovered mass lying 3′ of the last window | same | 95.3% |
| encoder passes per transcript, caching unaffected candidates | same | mean 159,942 |
| ...re-encoding every candidate instead | same | mean 669,647 |
| saving from the cache, on the totals | same | 4.2× |
| float32 input if one transcript's substitutions are batched at once | same | mean 3.72 GB, p90 7.19 GB, max 119.96 GB |
| ...transcripts above 8 GB on that basis | same | 3,584 |
| genes | 42,043 transcripts | 12,613, mean 3.3 transcripts each |
| genes carrying both labels | 12,613 genes | 3,546 |
| `is_nmd` prevalence | 42,043 transcripts | 0.2242 |
| transcripts with more than 10 candidates | same | 30,799 (73.3%) |
| ...more than 20 | same | 15,788 (37.6%) |
| slot of the reference start codon in 5′→3′ order | 28,775 with one in the pool | median 1, mean 3.7, p90 11, max 564 |
| ...within slots 0 to 9 | same | 25,559 (88.8%) |
| selection mass in slots 0 to 9, at `p_k` = 0.5 | 60 chr21 transcripts | 99.8% |

Measured over the 41,765 transcripts of step 1, by `build_ism_split.py` /
`build_ism_split_runlog.txt`:

| quantity | population | measured |
|---|---|---|
| transcripts on a composite locus, excluded | 42,043 transcripts | 278 on 233 loci |
| genes | 41,765 transcripts | 12,380 |
| `is_nmd` prevalence | same | 0.2232 |
| transcripts assigned `discovery` | same | 20,897 (50.0%), prevalence 0.2254 |
| transcripts assigned `confirmation` | same | 20,868 (50.0%), prevalence 0.2210 |
| genes spanning both arms | 12,380 genes | 0 |
| genes spanning more than one chromosome | same | 0 |
| at *n* = 1,000: discovery / confirmation | first 1,000 of the order | 463 / 537, over 936 genes |
| ...positives among them | same | 105 / 118 |

Reported by the bank build itself: `n`, `W`, bytes per arm, transcripts and genes per arm of step 1,
the achieved discovery/confirmation balance within each (`is_nmd`, split) cell, the
`batch_shape_offset` of step 6 with the number of positions it was measured at, **the distribution of
`|vals|` against it and the share of substitutions clearing it**, the measured substitution rate and peak
device memory on the hardware used, and for the control three numbers from the `R` draws of step 9:
the mean absolute effect, the standard deviation of the effect across paired draws, and the standard
deviation of the unperturbed logit across draws.

The third of those is what an unpaired estimator would carry as noise. Measured on chr21 with
**untrained** weights, where the effect itself is near zero and the ratio therefore says nothing about
the fitted model: mean absolute effect 3.15e-05, paired spread 4.50e-05 (1.43×), unperturbed-logit
spread 1.18e-02 (373×). The structure — that the unpaired term is orders of magnitude larger than the
paired one — is a property of the architecture. The ratio is not, and is re-measured on the trained
control checkpoint before `R` is chosen.

Whole-transcript batching is not used: at a mean of 3.72 GB of model input per transcript before any
activation, 3,584 transcripts exceed 8 GB on input alone. The bank is chunked by perturbation, and the
chunk size is set from a measured per-row cost on the hardware the run uses.
