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

**Step 3 — set the admission floor from real start codons.**

The floor is the 1st percentile of the PWM score of annotated start codons. The annotated start
codons are the candidates whose `start` equals `ref_utr5_length` (0-based), taken over the
transcripts where `ref_atg_available` is 1. Admitting at the 1st percentile means a candidate is
kept when its initiation context is at least as good as that of the weakest 1% of start codons the
annotation actually uses.

```
annotated = [ score(t, ref_utr5_length[t]) for t in REFCDS where ref_atg_available == 1 ]
FLOOR = percentile(annotated, 1)
```

**Step 4 — admit and order.**

A candidate is admitted when its score is at or above `FLOOR`. Admitted candidates are ordered by
`start` ascending, which is the order a scanning ribosome encounters them, and the slot index is
that position in the order. Slot index carries no priority: slot 0 is the 5′-most admitted
candidate, not the annotated one.

```
for each transcript t:
    admitted = [ orf for orf in ORFs(t) if score(orf) >= FLOOR ]
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
| candidates per transcript after the floor | 42,043 transcripts | mean 50.1 |
| candidates in the largest transcript | 42,043 transcripts | 1,789 |
| share of upstream ORFs whose own stop has a junction >50 bases downstream that are admitted | 133,765 such ORFs over 17,944 transcripts | 93.4% |
| share of transcripts whose annotated start codon is admitted | 28,775 transcripts with `ref_atg_available == 1` | not yet measured |

The last row is the one with no prediction: admission by initiation score does not guarantee the
annotated CDS a slot, and 51.5% of NMD-positive transcripts are degraded through the main ORF
(measured elsewhere, `build_mechanism_classes_runlog.txt`, over 41,765 transcripts).

### 3.5 Decisions this step needs before it runs

**The floor percentile.** 1st percentile gives mean 50.1 candidates per transcript and 93.4%
coverage of triggering upstream ORFs; 5th gives 41.6 and 79.3%; 25th gives 21.9 and 44.4% (all
measured in `design1_orf_pool_size_runlog.txt`). The choice sets the size of the training tensor and
therefore the compute for everything downstream.

**The tail.** The largest transcript carries 1,789 admitted candidates. Either every transcript
keeps all of its candidates and batches are bucketed by candidate count, or a cap applies and the
number of dropped candidates is recorded per transcript.

---

## 4. Prerequisite code repair

`infer_uorf_attention.py` derives its attention columns from a literal list of five and a `range(5)`
at lines 174 and 207, and `compute_uorf_attention_metrics_pathB.R:95` selects five named columns.
These take the slot count from the data instead. `compute_uorf_attention_metrics.R:89` and
`audit_uorf_attention.R:92,188` already do.

---

## 5. Build the training tensor

## 6. Architecture

## 7. Training

## 8. Model comparisons

*Sections 5–8 are written after section 3 is agreed.*
