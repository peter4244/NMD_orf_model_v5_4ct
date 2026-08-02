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
| **Elevated decay positions sit at a keto base (G or U) in a uridine-rich, cytosine-poor window** | keto **1.16×** (measured; 1.148× struck 2026-08-02 — see `ANALYSIS_SEQUENCING_PROPOSAL.md`), amino 0.845×, **G+C flat at 1.004×**; flanks ±2–6 carry U at 1.19–1.41 with A flat | **U-rich, not AU-rich** — A contributes nothing. The center is GC-neutral; the AU enrichment is in the *flanks*, and reporting the center while describing the whole signal was an error Pete caught |
| ~~**The signal is not the GC encoding window**~~ **— RETIRED 2026-08-02** | *was:* profile decays to baseline by ~8 bases and is gone by 20; channel 5 averages over ±25 | **Dead by two independent routes.** The decay quoted is the **U** profile while the channel is **G+C**, and G+C is still 0.95–0.96 at ±20, returning only past ±40 — the scale of the channel, not one that excludes it. Independently, `SEQUENCE_ENRICHMENT_APPROACH.md` §7.1: raw autocorrelation persists past lag 80, which ±25 cannot explain either, so length scale never discriminated. **The composition profile survives; the inference on top of it does not, and nothing currently excludes the GC channel.** Settling it needs conditioning on channel 5, not ablation — see `FINDINGS_DECAY_SEQUENCE_2026-08-02.md` |
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

Three failure modes that are *not* that one, and all three bit:

- **Treating a feature of the exposure as a confounder.** Selection mass is the architecture;
  the position/composition decoupling at a PTC is what a PTC transcript *is*. Stratify and state,
  never adjust away. No amount of checking denominators catches this — it needs someone who knows the
  biology.
- **Provenance.** A commit that claimed edits it did not contain. The message is not evidence of the
  diff; `git show` is. Same rule as the cluster, where `git rev-parse HEAD` is not the provenance of
  what ran.
- **Compression.** A correction made, accepted by both windows, verified, committed — and then
  **reintroduced when the state was summarized for a new reader.** Twice in two days, both times in
  a document written *for a reader* rather than in an analysis. The second: the modeling handoff
  (`eb48a54`, 09:40:51) told its incoming window not to start until three fixes landed that had
  landed at 09:37:38 (`4dd9f69`). Three minutes. A blocking instruction, inside a document listing
  compression as a failure mode. Corrected at `d1a0941`.

  **Then the report of it repeated the error class.** I gave the gap as seven minutes, from the
  file's mtime rather than the commit time — and the mtime had since moved to 09:47:36, because the
  correcting commit rewrote the file. A filesystem timestamp and a commit timestamp are two
  quantities with one name. Caught by the model window checking my number.

  Neither instance was caught by anyone being careful. Each was caught by going to the artifact
  before repeating a claim about it, which is a step and not a virtue. **When any document —
  including this one, including this bullet — says something is owed, blocked, settled or retracted,
  check the commit before acting on it.** `git log --date=iso --format='%h %cd %s' -1 <sha>`.

Two limits of the two-window arrangement, both found by hitting them:

**Replication catches only the errors the two implementations do not share.** Both windows built the
run-length statistic independently, agreed to four decimals, and were both wrong — neither had
written the hypothesis down first, so the error sat upstream of both implementations where no amount
of independence reaches it. Hence the ordering: hypothesis row → one implementation → replication
only for load-bearing survivors.

**Every number here has been independently recomputed at least once. No paragraph has.** That is the
gap, and it is exactly where both surviving failures live (model window, `7225cdb`). The reason is
structural: an analysis gets *re-run*, but a document gets *read* — a compressed error in prose
arrives at the next window as a fact with no producer attached, nothing to re-execute, nothing to
diff, no runlog to open. You inherit this gap. The fix has the same shape as everything else above:
**when a document states a result, open the producer before believing it.**

---

## How to read all of this

**None of it was designed. It is a record of what we ran into.** The toolbox exists because a GC
control had to be discarded; the autocorrelation-preserving null because the run-length null was
invalid; the magnitude clause because the directionality claim was circular; the motif definition
because we used the word for two days without one. Every piece of structure is the scar of a specific
failure, and the documents therefore **stop exactly where our errors stopped**.

So expect classes they do not cover — a third limit of the arrangement should be assumed, since both
known ones were found by something going wrong that no document had an entry for. **When you hit a
gap, add a row.** Do not work around it, and do not assume the omission was considered.

The corollary, already demonstrated once: the incoming modeling window reviewed the A2 row before
implementing it and found the row fixed four lines while leaving a dozen implementation choices open
— two of which changed the design rather than filling a blank. Reviewing prose before it becomes code
caught what no amount of independent implementation would have, because both implementations would
have shared the error. That review is now the standing order, not a courtesy.

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
- **A coincidence that will mislead you.** The keto background and the GC background are numerically
  identical in both existing measurements (.501/.501 and .502/.502). Not a bug: G+T equals G+C
  exactly when T equals C, and C and T are within a thousandth of each other here. So "the background
  is .50" is true of **two different quantities**, and anyone reconciling composition tables by eye
  can match the wrong pair and conclude two sets agree when they do not. Found by the modeling
  window while reconciling 1.16× against 1.148×.
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
