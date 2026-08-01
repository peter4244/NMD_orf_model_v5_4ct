# Retrain plan — sequence interpretability model

Specification of what is done to the data. Reasoning lives in
`RETRAIN_RATIONALE_2026-08-01.md`.

Section 3 is written in full for review. Sections 4–8 are listed by name only and are written after
section 3 is agreed.

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
transcript position `ref_utr5_length + 1`. `ref_atg_available` is 1 where that projection succeeded.
`gene_id` groups transcripts for clustered uncertainty.

### 2.5 `HOLDOUT` — paralog gene lists

`paralog_genes.tsv` (56 genes) and `val_paralog_genes.tsv` (19 genes) in the same directory. Genes
withheld from training so that a paralog of a training gene cannot appear in evaluation.

### 2.6 Datasets this plan replaces

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
    s = SEQ[t]
    for i in every position where s[i:i+3] == "ATG":
        j = i + 3
        while j + 3 <= len(s):
            if s[j:j+3] in {"TAA","TAG","TGA"}:
                emit ORF(transcript=t, start=i, end=j+3, length=j+3-i)
                break
            j = j + 3
```

**Step 2 — score initiation context at each candidate's start codon.**

Each start codon is scored by the Cavener–Ray position weight matrix over the eight positions −6 to
−1 and +4, +5 relative to the A of the ATG, which is the matrix `Isopair::scoreKozakPWM` uses. The
score is the sum of log2(observed frequency / 0.25) over the positions that exist; positions running
off either end of the transcript are skipped and the remainder summed. A candidate whose start codon
has no scorable position gets no score and is not admitted.

```
PWM[base, position] = log2(CavenerRay_freq[base, position] / 0.25)
offsets = [-6,-5,-4,-3,-2,-1,+3,+4]        # relative to the A of ATG at offset 0

score(t, i) = sum over o in offsets, where 0 <= i+o < len(SEQ[t]),
              of PWM[ SEQ[t][i+o], o ]
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

**Step 4 — admit and order.**

A candidate is admitted when its score is at or above `FLOOR`. The candidate matching the annotated
CDS start is admitted regardless of score, where the transcript has one: at `FLOOR` it otherwise
drops out for 8.4% of transcripts (measured, §3.4), leaving those transcripts with no candidate at
their annotated coding sequence. Unconditional admission places the ORF in the pool; nothing marks
which one it is.

Admitted candidates are ordered by `start` ascending, which is the order a scanning ribosome
encounters them, and the slot index is that position in the order. Slot index carries no priority:
slot 0 is the 5′-most admitted candidate, not the annotated one.

```
for each transcript t:
    admitted = [ orf for orf in ORFs(t)
                 if score(orf) >= FLOOR
                 or orf.start == ref_utr5_length[t] ]
    admitted = sort(admitted, key = orf.start, ascending)
    for k, orf in enumerate(admitted):
        orf.slot = k
```

**Step 5 — attach the per-candidate quantities the model is given.**

Each admitted candidate carries one supplied structural number and nothing else. `n_downstream_ejc`
is the count of junctions lying more than 50 bases past the last base of that candidate's stop
codon. This is the canonical exon-junction rule; the count is over junctions from `JUNC` for that
transcript.

```
n_downstream_ejc(orf) = count of j in JUNC[orf.transcript] where j > orf.end + 50
```

Withheld, and therefore not written into the pool table as model inputs: which candidate matches the
reference CDS, which matches the TD2 CDS, each candidate's fractional start and stop position, and
the PWM score itself. The PWM score decides admission in step 4 and is not a feature.

**Step 6 — emit the pool table.**

One row per admitted candidate, replacing `selected_orfs.tsv`:

| column | definition |
|---|---|
| `isoform_id` | transcript, keys to `TX` |
| `slot` | 0-based index in 5′→3′ order among admitted candidates of this transcript |
| `orf_start` | 1-based transcript position of the A of the ATG |
| `orf_end` | 1-based transcript position of the last base of the stop codon |
| `orf_length` | `orf_end − orf_start + 1` |
| `n_downstream_ejc` | step 5 |
| `is_ref_cds` | 1 where `orf_start == ref_utr5_length + 1`. Recorded for evaluation and for the interpretation window; **not a model input** |

### 3.4 Quantities this step reports

Each is a count or proportion over the pool, and each is compared against the value predicted from
the measurement in `design1_orf_pool_size_runlog.txt`, which used the same enumeration on the same
sequences.

| quantity | population | predicted |
|---|---|---|
| candidates per transcript before the floor | 42,043 transcripts | mean 54.6 |
| candidates per transcript after the floor | 42,043 transcripts | mean 36.4 |
| candidates in the largest transcript | 42,043 transcripts | 1,190 |
| share of upstream ORFs whose own stop has a junction >50 bases downstream that are admitted | 133,765 such ORFs over 17,944 transcripts | 70.6% |
| share of transcripts whose annotated start codon clears `FLOOR` on its own | 28,775 transcripts with `ref_atg_available == 1` | 91.6% |
| the same share after unconditional admission | same | 100% |

All predicted values are measured in `design2_mane_floor_runlog.txt` on the same sequences with the
same enumeration.

### 3.5 Cost

Training-tensor size scales linearly in candidates per transcript. The current 5-slot tensor is
2.9 GB, so 36.4 candidates projects to roughly 21 GB. This does not fit in the 8 GiB on this
machine; the tensor is built on the cluster.

### 3.6 The tail

No cap applies. Every transcript keeps all of its admitted candidates, and batches are bucketed by
candidate count so that a batch holds transcripts of similar width. Candidates per transcript at
`FLOOR` with the annotated CDS forced in: median 32, p90 65, p99 107, maximum 1,191 (measured,
`design3_check_track_a_runlog.txt`).

If a cap is ever imposed, the pool table carries a per-transcript column recording whether it was
hit and how many candidates were dropped.

---

## 4. Prerequisite code repair

`infer_uorf_attention.py` derives its attention columns from a literal list of five and a `range(5)`
at lines 174 and 207, and `compute_uorf_attention_metrics_pathB.R:95` selects five named columns.
These take the slot count from the data instead. `compute_uorf_attention_metrics.R:89` and
`audit_uorf_attention.R:92,188` already do.

---

## 5. Build the training tensor

### 5.1 Question

What array does the model consume, and what is in each channel?

### 5.2 Starting data

The pool table from §3, `SEQ`, `JUNC`, `TX`, `HOLDOUT`.

### 5.3 Approach

**Step 1 — give each candidate two sequence windows.**

A candidate carries an ATG window centred on the middle base of its start codon and a stop window
centred on the middle base of its stop codon, each 500 bases wide. A window running off either end
of the transcript is zero-padded on that side. Where the two windows would overlap, each is clipped
at the midpoint of the ORF so that together they partition it and neither reads the other's half;
the clipped side is padded.

```
mid   = (orf_start + orf_end) / 2
atg  = SEQ[t][ orf_start+1 - 250 : orf_start+1 + 250 ]   clipped to [1, mid], zero-padded
stop = SEQ[t][ orf_end-1  - 250 : orf_end-1  + 250 ]     clipped to [mid, tx_length], zero-padded
```

**Step 2 — encode nine channels at each window position.**

| channel | content |
|---|---|
| 0–3 | one-hot A, C, G, T; all zero at a padded position |
| 4 | 1 where the position is listed in `JUNC` for this transcript, else 0 |
| 5 | fraction of G or C over the 50 bases centred here, computed from this window's own sequence |
| 6–8 | one-hot reading frame of this position relative to this candidate's start codon |

**Step 3 — attach the structural block.**

The interpretable model's block is the single number `n_downstream_ejc` from §3 step 5. The
predictor's block is five numbers: that one plus `is_ref_cds`, `is_sqanti_cds`, `frac_start` and
`frac_stop`, where the fractions are `orf_start / tx_length` and `orf_end / tx_length`.

Each number is centred and scaled by its mean and standard deviation over candidates belonging to
**training-split transcripts only**. Those constants are written into the tensor file and are read
back at inference rather than recomputed.

**Step 4 — assemble ragged, not padded.**

Candidates per transcript run from 1 to 1,191 with a median of 32, so padding every transcript to
the maximum would store mostly zeros. Candidates are stored as one flat array in transcript order
with a per-transcript start offset and count.

```
candidates[i]  = (atg_window[9,500], stop_window[9,500], structural[1 or 5])
offset[t], count[t]   such that transcript t owns candidates[offset[t] : offset[t]+count[t]]
```

**Step 5 — assign splits.**

Split comes from `chr` in `TX`: test chr1/3/5/7, validation chr2/4, training the rest. Transcripts
of genes listed in `HOLDOUT` are removed from the training split.

### 5.4 Quantities this step reports

Candidates written, transcripts written, bytes on disk, and the normalization constants with the
number of training candidates each was computed over.

---

## 6. Architecture

### 6.1 Question

What does each part of the model compute, and from which inputs?

### 6.2 Approach

**Step 1 — encode each candidate with shared weights.**

Each window passes through the existing two-layer convolution, and the length axis is then reduced
by **binned** max pooling rather than a single global max: the axis is split into 8 equal bins and
the maximum is taken within each. The output is 32 channels × 8 bins, flattened and projected to 32.

```
h = relu(bn1(conv1(x, 9->32, k=15)));  h = maxpool(h, 4)
h = relu(bn2(conv2(h, 32->32, k=7)))
h = concat over b in 0..7 of  max(h[:, b*L/8 : (b+1)*L/8])      # 32*8
e = linear(h, 256 -> 32)
```

**Step 2 — initiation head, from a narrow slice of the ATG window only.**

The initiation head reads the central 51 bases of the ATG window — positions −25 to +25 relative to
the A of the start codon — through its own two-layer convolution, and emits one number per
candidate through a sigmoid. It does not see the stop window, the structural block, or the rest of
the ATG window.

```
p_k = sigmoid( linear( encode_narrow( atg_window[:, 225:276] ) ) )
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

**Step 2 — stopping.** Training stops when validation AUC has not improved for 5 epochs; the
checkpoint with the best validation AUC is kept.

**Step 3 — seeds.** Each configuration is trained from 5 random initialisations.

**Step 4 — convolution width sweep.** `conv_channels` takes each of 16, 32, 64, 128, at 5 seeds
each, on the interpretable variant.

### 7.4 Quantities this step reports

Per run: epochs to stop, validation AUC at the kept checkpoint, wall time. Per configuration:
the five seeds' test AUC and AUPRC.

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
| sequence-blanked | channels 0–3 set to zero | if it matches the interpretable model, sequence carries nothing usable at this window size and interpretation will come up empty. Stop early. |
| no-junction-feature | structural block empty | if it finds something the interpretable model does not, supplying the junction rule is masking it |

Neither is evidence for anything. A difference in either direction is an operational signal about
whether to continue, and is reported as such.

**Step 3 — record, do not arm, the label question.** The 2,334 `no_ref_isoform` transcripts stay in
training. Whether they are trained on is recorded in the run metadata so the question can be asked
later against a specific finding, rather than answered in advance against nothing.

### 8.3 Quantities this step reports

Per variant and seed: test AUC, test AUPRC, epochs to stop, wall time. Reported against the current
model at 0.9310 AUC / 0.8351 AUPRC and a tabular-only GBM at 0.8035 AUPRC (both measured elsewhere,
`SEQUENCE_DISCOVERY_BRIEF.md` §5).
