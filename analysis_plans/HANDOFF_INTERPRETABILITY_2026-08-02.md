# Handoff — interpretability lead, 2026-08-02

You are taking over the interpretability thread. **Your first task is to review two documents and
tell Pete what you think is wrong with them:** `ANALYSIS_SEQUENCING_PROPOSAL.md` (mine) and the model
window's `SEQUENCE_ENRICHMENT_APPROACH.md`. Read them adversarially. Both have been through two
rounds of cross-review and both still contained errors at the end of the second.

## Read in this order

1. `analysis_plans/ANALYSIS_SEQUENCING_PROPOSAL.md` — the plan, one gate, with hypothesis rows
2. `analysis_plans/ADJUSTMENT_TOOLBOX.md` — the eleven adjustment axes and, more importantly, **what
   each one licenses you to conclude**
3. `analysis_plans/SEQUENCE_ENRICHMENT_APPROACH.md` — the model window's method document, including
   the motif definition
4. `analysis_plans/REGION_CALLER_SPEC.md` — written, unimplemented
5. `analysis_plans/FINDINGS_DECAY_SEQUENCE_2026-08-02.md` — the two findings that survived

Do **not** start by reading the older handoffs. They contain claims retracted since.

**State on arrival.** Nothing running on the cluster; both worktrees clean. The model window's review
of the plan is **already folded in** at `4dd9f69` — three fixes: the stopping rule rewritten (it still
named A1 after A1 was dropped, and a rule naming a nonexistent analysis gets read loosely at exactly
the moment it fires); A2 given two independent implementations against one shared row; B2 formally
blocked. There is no pending review queue. The figures window has the retractions below.

---

## What is true now

All on the five production ISM banks: `results_ism_v6/bank_interp_s{100..500}.h5` on Explorer,
4,999 transcripts, 11,062,149 valid positions each, `chunk_rows` constant at 4,096, QC clean.

| claim | evidence | caveat that must travel with it |
|---|---|---|
| **Elevated decay positions sit at a keto base (G or U) in a uridine-rich, cytosine-poor window** | keto 1.148×, amino 0.851×, **G+C flat at 1.004×**; flanks ±2–6 carry U at 1.19–1.41 with A flat | **U-rich, not AU-rich** — A contributes nothing. The center is GC-neutral; the AU enrichment is in the *flanks*, and reporting the center while describing the whole signal was an error Pete caught |
| **The signal is not the GC encoding window** | profile decays to baseline by ~8 bases and is gone by 20; channel 5 averages over ±25 | wrong length for the encoding, right length for a sequence feature |
| **Five members agree on sequence far better than on position** | k-mer enrichment r = 0.7529 (0.7176–0.8111) against positional Jaccard 0.125 | uses the **elevation rule**, so it does not survive a negative A2 untouched |
| **A PWM predicts importance on held-out genes** | held-out r = 0.1316 = in-sample, 5.5M confirmation positions, disjoint genes | **1.73% of variance.** This is a ceiling on what any single motif can be worth, not a finding |
| **The decay branch does not read the stop codon it is anchored on** | codon bases at the 75th percentile, indistinguishable from controls 25–30 bases away | the pool guarantees a stop at every anchor, so there was no negative example and no gradient — same structural blindness already on record for capture |
| **No stop-codon context preference** | flat importance across ±6 by codon identity, no +4 peak — while the +4 *composition* in the data is real and structured | the informative pairing: the signal is present in the sequence and absent from what the model reads |

## What is retracted — do not re-derive

| retracted | why it died |
|---|---|
| run-length clustering, "hundreds of times a count-matched null" | the null placed marks at random, destroying autocorrelation the track has **architecturally**. The track is smooth because selection mass is smooth (log-mass correlation 0.93 with the effect track). The statistic showed the track is autocorrelated, not that a sequence feature exists |
| "AU-rich element" | G+C is flat at the elevated position — 0.505 against 0.502. AU and GC are complementary, so there is no AU enrichment at the center |
| the GC confound, and the control built for it | the elevated set was never GC-biased. The GC-preserving operator drove G+C to **0.679** — the control was three times more biased than the thing it tested. Its k-mer output is discarded, not fixed |
| directionality, "21% above the noise floor" | moved to *not settled*. Elevated positions are **defined** as the largest effects, and directionality rises with effect magnitude across all positions, so the claim may be tautological. Untestable until magnitude is held fixed |
| "§5's mechanism claim rests on the capture arm" | E4/E5 read `p_select` / `p` off forward passes and never touch `vals_capture`. The capture floor threatens D3 and sequence-level *explanations* of capture, nothing else |
| the fold-over-median elevation rule | selected 1.7% of positions on a short-transcript pilot and 10.7% on the real banks. On the banks the **null beat the data**. Replaced by a fixed top-fraction per transcript |

---

## The plan: one gate

**A2** — is the U-rich/keto signature something the decay head reads, or a reflection of where the
model routes? Routing held fixed **by stratification, never by removal** — the weighting is the
model, and residualizing describes a model that weights every ORF equally, which is neither ours nor
ribosomes.

Everything is downstream of it. A2 gets **two implementations against one shared row** — the model
window writes the second against my specification, so only the code is independent.

Also live: **A4** resurrected but scoped to called regions (28 pairs for an 8-base region — the
quadratic cost was only ever the genome-wide version). **B2 is blocked** on an annotation-derived
PTC definition; as specified its cell is 51.3% NMD against a 44.0% background. **C** is gated on
recovering GT/AG from SpliceAI, which is now the *only* positive control — the stop-codon one is
retired, because recovering a stop codon from a stop-anchored profile is guaranteed by the anchoring
and tests indexing, not method.

---

## The one thing to internalize

**Eleven errors in two days, and they are one error.** Two quantities with the same name computed
over different sets, or a reference point assumed rather than measured:

padded vs unpadded arrays · valid vs ATG-covered positions · a `p_select` stratum vs a representative
value inside it · `tx_length` vs last-covered position · the reference-anchor exclusion (31.5%
dropped, **differentially** — 49.9% retention in the mechanism cell against 93.1% in its control) ·
structural zeros in the capture arm · raw counts across two implementations differing 1.36× in scale
· the ratio floor assumed 0 when measured 0.387 · mean vs median (0.373 vs 0.391) · global vs
per-transcript top-1% (Jaccard 0.24 between them) · the elevation threshold's selectivity

**None of them threw. All of them produced a plausible table.** The check is mechanical: before
quoting any proportion, size, ratio or cross-implementation comparison, enumerate what is in the set
and measure both reference points in-sample.

Two failure modes that are *not* that one, and both bit:

- **Treating a feature of the exposure as a confounder.** Selection mass is the architecture;
  the position/composition decoupling at a PTC is what a PTC transcript *is*. Stratify and state,
  never adjust away. No amount of checking denominators catches this — it needs someone who knows the
  biology.
- **Compression.** A correction was made, accepted by both windows, and then **reintroduced when the
  document was summarized for a reader.** Summarizing is where corrections go to die, and this
  handoff is the artifact most exposed to it.

And the limit of the arrangement: **replication catches errors the two implementations do not
share.** Both windows built the run-length statistic independently, agreed to four decimals, and were
both wrong — neither had written the hypothesis down first. Hence: hypothesis row → one
implementation → replication only for load-bearing survivors.

---

## Practical

- **Worktrees.** Model window has `NMD_orf_model_v5_4ct` on `master` and all the data. You have
  `NMD_orf_model_v5_4ct_interp` on `interp`, data by symlink. See `WORKTREE_LAYOUT.md`. Merge often,
  both ways. Namespaces: outputs `interp_*` / `model_*`, jobs `hi_*` / `md_*` — **and cluster scratch
  scripts too**, which we learned when both windows wrote `autocorr.py` and the second silently
  replaced the first.
- **Explorer.** `p.castaldi@explorer.northeastern.edu`, `~/cc/nmd_orf_model_v5_4ct`, conda env
  `nmd_model`. **Ask Pete before every login** — access has been granted in explicit time-boxed
  windows. Four-job cap shared with the model window. Copy files up, `sha256sum` both ends, then
  submit; `git rev-parse HEAD` on the cluster is *not* the provenance of what runs there.
- **Traps in the data itself.** `structural` in the tensor is **normalized** — the raw count is
  `structural_raw`. `orf_length` in the pool is **inclusive** (`orf_end − orf_start + 1`). `vals` is
  **NaN at the observed base**, so `np.isfinite(x).all(1)` is never true — use
  `.sum(1) == 3`; all three windows hit this. Fold-over-median degenerates on the 2.8% of transcripts
  whose median is below 1e-6 — **reach for rank statistics by default**.
- **The other windows.** Model window (Maude) owns training, banks, and the second A2 implementation.
  A figures window (Larry) works in `nmd_lung_longread_2026`, which is **not** worktree-split and
  whose commit gate currently fails on six pre-existing artifacts.

## Your first task

Review both documents adversarially and report to Pete. Two things I would look at hardest, because
they are mine and unreviewed by anyone else:

1. **The region-caller spec's iAAFT null.** I chose it because it preserves autocorrelation and the
   marginal distribution while destroying feature location. I have not validated that it behaves as
   intended on this track.
2. **The claim that A2 is the only gate.** I arrived at it by dropping items on Pete's criterion —
   "if we don't think it is informative we shouldn't do it" — and it is possible I cut something that
   was load-bearing.
