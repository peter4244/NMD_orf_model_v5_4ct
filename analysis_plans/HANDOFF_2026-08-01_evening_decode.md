# Start here — handoff, evening of 2026-08-01

*Written deliberately plainly. Pete's standing instruction: no invented shorthand, and a
plain-English closer on every response.*

## The one job

**Make `decode_windows` faster, then build the mutagenesis banks.** Everything else on the
model side is finished. The bottleneck is measured, the fix is specified below, and the
verification bar is non-negotiable because the entire bank's correctness rests on this
one function.

Do not start by reading the whole plan. Read this file, then §9 of
`analysis_plans/RETRAIN_PLAN_2026-08-01.md`, then `tensor_io.py`. That is enough.

---

## What is finished

**Section 5's performance case.** Selected `interp_c32_b8` on validation only, before test was
opened; the selection file is quoted in the evaluation job's own log, so the audit trail is in
the artifact rather than in anyone's account of it.

    ensemble (mean probability, the declared headline)  AUC 0.9322  AUPRC 0.8679
    ensemble (mean logit)                               AUC 0.9327  AUPRC 0.8684
    single model, mean of five seeds                    AUC 0.9265  AUPRC 0.8551
                                                        95% CI [0.9198, 0.9332]

The comparator is `atg1000_stop1000` **re-scored on the v6 universe** — n 10,520, n_nmd 2,405,
identical to ours, verified from both sides — at 0.9310 / 0.8351 under mean logit, which the
file declares. So it is a genuine architecture-versus-architecture comparison. **AUC is a tie.
AUPRC is +0.033.** Lead on AUPRC and say the AUC is a wash; the published figure was chosen
across twelve test-set comparisons and ours chose nothing.

**The decomposition, on one split, validation:**

    interpretable (sequence + selection)   0.9469
    sequence-blanked (selection only)      0.9426
    tabular GBM (neither)                  0.8815
    no-junction                            0.8904

    candidate weighting  ~0.057   upper bound
    positional detail    ~0.004   lower bound (the permuted arm keeps fill extent)
    sequence             ~0.004

Selection is worth roughly fifteen times sequence. That is Section 5's central claim and it
came from Pete's observation that the junction rule cannot be applied without first choosing a
reading frame.

**The bank code.** Complete, reviewed twice by the interpretability window, 36 datasets.
Verified against an uncached reference implementation that shares no code path with it.

---

## The job: decode is 88% of the time

Measured on a V100, five repeats each, `torch.cuda.synchronize()` before every timer,
`analysis_plans/profile_ism_cluster.py`:

    chunk 49,152 rows      decode_windows   88.2%
                           host->device      6.7%
                           both encoders     4.4%
                           three aggregations 0.0%

    same work in 1/4/16/64 calls: 393k / 397k / 392k / 371k rows/s  -> NOT per-call overhead

**Do not profile this on the laptop.** I did, it said decode was 13-16% and the encoders 85%,
and I abandoned the correct hypothesis on the strength of it. On a laptop the encoders run on
the CPU; on the cluster they are on the GPU and cost almost nothing. Laptop timings on this
work also vary twofold on identical input — I quoted a 2.9x speedup from one before/after pair
and the true figure on the hardware was 10%.

### The fix

A single base substitution changes **about 51 positions of a 1000-wide window**: the base
itself, and the ±25 span over which channel 5 averages local GC. The other 949 are identical to
the unmutated window. Today the code rebuilds all 1000 from scratch for every mutated row —
cumulative sums, one-hot writes, frame assignment across the whole axis.

So: **decode each candidate's window once, cache it, and for each mutation copy the cache and
patch the ~51 changed positions.** That converts the dominant cost from computation to a memory
copy. The copy is unavoidable — the encoder needs a contiguous `(n, 9, 1000)` tensor — but at
memory bandwidth 590 MB is tens of milliseconds against the current 1.6 s.

Plausibly 10-25x overall, which turns 32 hours of wall time into one to three.

Channels 0-3 change at one index. Channel 5 changes over ±25. Channels 4 and 6-8 **do not
change at all** — junctions are annotation, and the frame grid is anchored, which was verified
on 400 real substitutions this morning.

### The verification bar, and it is not negotiable

`decode_windows` is what every number in the bank is computed from.

1. **Exact agreement with the current implementation** on real windows from
   `results_tensor_v6`, both ATG and stop, over at least a few thousand candidates. Not close —
   equal. It is deterministic integer-and-float arithmetic on the same inputs.
2. **Then re-run `verify_ism_bank.py`**, which recomputes the bank the slow way through the
   model's own `forward()` and compares entry by entry. It currently passes at max difference
   3.5e-08 against a floor of 1.5e-08.
3. **Then rebuild a small bank and compare arrays against a pre-change build**, as was done for
   every change today: `vals`, `vals_capture`, `vals_decay`, `dsel`, `dstart`, `dgc`, `mass`,
   `fill_count`, `obs`, `valid`.

If any of those three is skipped, the speedup is not worth having.

---

## Then build the banks

    python build_ism_bank.py --tensor results_tensor_v6 \
        --checkpoint runs/interp_c32_b8_s<SEED>/best.pt \
        --split results_ism_v6/ism_subset.tsv \
        --out results_ism_v6/bank_interp_s<SEED>.h5 --n 5000

Five banks, seeds 100-500, interpretable variant only. **Not** the permuted-bin control — the
discovery null is a circular shift on the real bank's own tracks and costs no forward passes,
so banking the control would spend half the budget answering a question nobody asked. **Not**
the `nosqanti` predictor — performance arm only.

The subset is `results_ism_v6/ism_subset.tsv`, 4,999 transcripts over 3,830 genes,
sha256 in `ism_subset_provenance.json`. It is **stratified, not random**: the mechanism cell
this section is about is 437 transcripts in 41,765, and a random 5,000 would draw about fifty.
Scarce cells are taken whole; every transcript carries a `sampling_weight` and the weights sum
to 41,765 exactly. **Any population-level estimate must be reweighted or it describes this
subset rather than the pool.**

Shards make this restartable: each transcript is written as it completes, a rerun skips what is
on disk, and `--from/--to` lets array tasks share a shard directory. The gpu partition kills at
8 hours and that costs one transcript, not a run.

---

## Traps this session paid for

1. **Profile where the code runs.** Above. Cost: a wrong hypothesis and a wrong number quoted
   to Pete.
2. **A speedup from one before/after pair is not a measurement.** Repeat it and look at the
   spread first.
3. **Copy files up before submitting the job that needs them.** Two jobs failed on this; one of
   them computed a nonsense loop bound, ran zero iterations and **exited 0 with SLURM recording
   COMPLETED.** Check the log, not the exit status.
4. **`pandas` casts NaN to 0 on `.to_numpy(np.int8)`** with only a RuntimeWarning. It silently
   turned "no GENCODE annotation" into "not upstream of the annotated start" for 62.8% of the
   pool. `fillna(-1)` first, and ship an explicit mask beside any sentinel.
5. **`np.savez` appends `.npz`** to any path not ending in it, so a `.npz.tmp` write lands
   somewhere else and the rename fails.
6. **Ablation cannot separate geometry from content in this encoding.** Every channel is zero
   outside the filled region, so every "keep only X" condition secretly keeps the fill mask.
   Two independent geometry leaks were found today — the 5' padding boundary encodes distance
   to the transcript start, and the midpoint clip encodes ORF length — and **both were found by
   conditioning, neither by ablation**. Any positional claim from this encoding needs a
   conditional control designed in from the start.
7. **Difference-in-means is scale-dependent; use a rank statistic.** The same ablation read
   "18% of the effect survives" on one scale and "114%" on the other.
8. **A design effect is a property of the statistic, not of the dataset.** Today produced 5.47
   (never a design effect at all — a transcription error that spread across eight documents),
   1.71 (a label ICC, which does not govern an interval), 3.15 (a matched-pair statistic) and
   1.00 (AUC). Measure it every time; never carry it.
9. **A zero is a claim that needs its own evidence.** Every zero encountered today was
   arithmetic or configuration until proven otherwise — a clamp pinning the output, a float32
   round-trip, a degenerate test, a silent no-op.

---

## Open, and who owns it

- **The launch decision.** Profile-then-launch is what Pete chose. If the optimisation lands,
  launch at full 5,000. If it does not, the choice is 32 hours at full size or 16 at half, and
  the scarce cells stay whole under any cut.
- **Re-scoring the old checkpoint on this split** — the interpretability window took it.
- **The claim list** — they owe it; the four scoping answers that mattered are already given
  and are recorded in §9.
- **§4 code repair** — dropped, off the critical path, do not pick it up.
- **`predictor_nosqanti_c32_b8`** — running, 2 of 5 seeds at 0.9494, above the with-column arm.

## The other window

Titled "NMD Harold - Interpretability", session `local_eeea272c-b38b-4d63-b63d-0cd73276aa6a`,
reachable with `mcp__ccd_session_mgmt__send_message`. **Batch messages; send one when something
changes what they would do.** Check what they send rather than accept it — today each window
overturned several of the other's conclusions and that is the main value of the arrangement.
They are good: they found the clamp defect, the batch-shape regime problem, the
`blank_junctions` bug, and they retracted two of their own findings unprompted.

They are owed nothing blocking. They have the full tensor locally and are running mechanism
analyses.

## Practical

- Local python: `~/miniforge3/envs/nmd_model_local/bin/python`.
- Explorer: `p.castaldi@explorer.northeastern.edu`, `~/cc/nmd_orf_model_v5_4ct`, conda env
  `nmd_model`. **Ask before every login.** GPU partition: 8 submitted, 4 running, 8-hour limit.
  `short` is the CPU partition, 2-day limit.
- Checkpoints are durable at `results_interp_all/v6_checkpoints/` — do not rely on anything
  under `/private/tmp`.
