# Resume here — interpretability, morning of 2026-08-02

Written at the stop. Explorer access was approved for ten hours from ~22:00 and is scoped to
**ISM bank analysis only** — not the §8.5 test read, not training, not the tensor build.

## Two jobs were left running. Read them first.

```bash
ssh p.castaldi@explorer.northeastern.edu 'cd ~/cc/nmd_orf_model_v5_4ct && \
  tail -40 results_ism_v6/interp_kmer_gc_8886938.log; \
  sed -n "/k=6/,\$p" results_ism_v6/interp_kmer_8886733.log'
```

- **`8886938` is the one that matters.** It rescores the k-mer enrichment using only the
  GC-preserving substitution at each position, so channel 5 is bitwise fixed.
- `8886733` was finishing k=6; k=5 is already read and is below.

Neither log is in git. Copy them into `analysis_plans/` and commit before doing anything else, or
they exist in one place only.

## Where the result stands

**Established, and controlled:**

Decay-branch ISM sensitivity clusters into short runs of elevated positions. Present at every
threshold from 0.2% to 5%. Replicates on disjoint gene sets (confirmation 914 / discovery 951 runs
of ≥4, seed 100). Reproduced by the model window's independent implementation. Survives with the
GC channel held bitwise constant at **57%** of its arity-matched level — which is the number to
quote, because `all` maxes over three substitutions and the GC arms over one, and attributing that
gap to GC would be an arity artifact.

**The new result, k=5, five members:**

    positional Jaccard   0.1250   (~22% overlap of top-1% sets)   LOW
    k-mer enrichment     r = 0.7501, range 0.714 to 0.809         HIGH

High agreement on *what the sequence is*, low agreement on *where it sits* — the outcome
pre-registered as a motif the members share and place differently. Enriched: `TTTTT` +2.34,
`TTGTT`, `TTTAT`, `TATAT`, `TTGAT`. Depleted: `GCACG`, `ATCCG`, `GCTCG`, `TACGC` −2.75.
U-rich and AU-rich up, GC-rich down.

## Do not call that an AU-rich element yet. Two confounds.

1. **GC composition.** The enriched/depleted axis IS a compositional axis. Job `8886938` tests it.
2. **Regional composition — NOT controlled, and the one I would worry about.** Elevated positions
   concentrate downstream of the stop codon; the background is every valid position of the same
   transcripts, 5′UTR and CDS included. 3′UTRs are AU-rich as a class, so a positional bias toward
   the 3′UTR reproduces this enrichment with no motif involved. **A region-matched background is
   the next thing to write.** It does not exist.

### READ THIS BEFORE OPENING `8886938` — the model window's catch, and it is easy to misread tired

**The two controls are not substitutes, and `8886938` does not answer confound 2.** Holding
channel 5 bitwise constant does not move the elevated positions. A 3′UTR-biased set scored against
a whole-transcript background recovers 3′UTR composition whether the scoring substitution preserved
GC or not.

So if `8886938` comes back clean, the only supported reading is **"not driven by the GC channel."**
It is *not* "the AU enrichment is real." Only a region-matched background can say that.

Their reason for flagging it is the sharper half: **a clean result on the control you have is
exactly when the control you do not have stops being noticed.** Same structure as the agreement
observation below, one level up — a passing check is another place where both parties stop looking.

This is also the fourth appearance of one structure in a single session: the reference anchor, the
structural zeros, the liveness gate, and now regional composition. All four are a filter or a
comparison whose two sides are drawn from different sets, and all four looked neutral until someone
split them by cell.

## Retracted tonight — do not re-derive

| claim | why it died |
|---|---|
| runs of ≥4 at 34-against-0, from the pilot | fold-over-median threshold selected 1.7% of positions on short transcripts and 10.7% on real ones. On the banks the null BEAT the data, 17,454 to 15,304. Superseded by the top-1% rule |
| "structure replicates, locations don't, so only structure is claimable" | wrong if the object is a motif — members can agree on the pattern and disagree on instances. Pete's RBP point |
| Q2's "nothing at the start codon" as a negative result | start/stop-anchored offsets can only find anchored features. A floating motif fails Q2 by construction; say so wherever Q2 appears |
| "§5's mechanism claim rests on the capture arm" | E4/E5 read `p_select`/`p` off forward passes and never touch `vals_capture`. The floor threatens D3 and sequence-level explanations of capture, nothing else |

## The habit this session earned

Seven instances in one night of **one quantity name over two different sets**: padded vs unpadded
arrays, valid vs ATG-covered positions, a `p_select` stratum vs a representative value inside it,
`tx_length` vs last-covered position, the reference-anchor exclusion, structural zeros in the
capture arm, and raw run counts across two implementations differing 1.36× in scale.

The sharpest one was the last: it was committed **inside a message asserting that two
implementations agreed**. Agreement is when both parties stop looking. Before quoting any
proportion, size or cross-implementation comparison, enumerate what is in the set — the naming did
not prevent recurrence, only the check does.

## Coordination

Model window owns cross-seed sign agreement and the ~92-position encoder residual at
`p_select ∈ [1e-13, 1e-11)`. This window owns run-length, the arms, and the k-mer work. The
run-length replication is deliberately duplicated in both implementations, both on `vals_decay`.
Namespaces `interp_*` and `model_*`; job prefixes `hi_*` and `md_*`; four-job cap is shared.

Figures window works in `nmd_lung_longread_2026`, which is **not** worktree-split and whose commit
gate currently fails on six pre-existing artifacts. Open with Pete: whether to split it, and who
clears those.
