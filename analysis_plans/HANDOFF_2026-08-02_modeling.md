# Start here — modeling window, 2026-08-02

*Written plainly. Pete's standing instruction: no invented shorthand, and a
plain-English closer on every response.*

You have primary responsibility for **implementing and managing the modeling side**
of the analysis plan. This document is self-contained; you do not need to read the
two previous handoffs.

---

## The one job

**Write the second implementation of A2, the single gating measurement, and run it.**

A2 asks: *is the U-rich / keto composition signature at high-importance positions a
property of what the decay head reads, or of where the model routes?* It is the only
gate in the programme. A negative propagates to every surviving result.

The interpretability window ("Harold") owns the first implementation and the
specification. **You write the second against their row, not your own reading of the
question** — shared specification, independent code. That is deliberate and §"How we
got burned" below explains why.

**Nothing blocks you. The plan is ready.** An earlier version of this document told
you to wait for three fixes — see §"A correction to this document, and why it is
instructive".

---

## Read these three, in this order, then stop reading

1. **`analysis_plans/SEQUENCE_ENRICHMENT_APPROACH.md`** — the method document. §1.1
   defines "motif"; §3.2.2 is the toolbox and the rule that selects from it. This is
   the governing document and it is the one to update when the approach changes.
2. **`analysis_plans/ANALYSIS_SEQUENCING_PROPOSAL.md`** — the plan. A2 is the gate;
   A1 is dropped; A3 and A4 are non-gating; B and C are downstream.
3. **`analysis_plans/ADJUSTMENT_TOOLBOX.md`** — the enumerated adjustments and what
   each licenses you to conclude.

`REGION_CALLER_SPEC.md` becomes relevant at Phase B, not before.

---

## What exists and is trustworthy

**Five ISM banks**, on Explorer only, at
`~/cc/nmd_orf_model_v5_4ct/results_ism_v6/bank_interp_s{100..500}.h5` — 652–703 MB
each. Seeds 100–500 of the interpretable arm, 4,999 transcripts, 11,062,149 valid
positions, `chunk_rows` constant at 4,096, zero dead transcripts. QC in
`analysis_plans/qc_ism_banks_runlog.txt`.

**The decode optimization**, which produced them: 8.4× end to end, and *bitwise*
identical to the implementation it replaced. Verified three ways on the GPU — 449,344
substitutions with zero differing entries, 37 bank datasets bitwise identical against
the pre-cache builder, and agreement with the model's own `forward()` to 2.842e-09.
Each check was first shown capable of failing, by deliberate mutation.

**Hardware is not a confound in the cross-seed numbers, and this is already checked.**
All five array tasks ran on **Tesla V100-SXM2-32GB** across five nodes (d1002, d1007,
d1011, d1017, d1002). So cross-seed disagreement is initialization, not hardware.
*Residual, stated because it is unlogged rather than excluded:* the bank script
recorded GPU model but not driver version per task. Not plausible as a cause at these
effect sizes; the script now logs driver, the banks predate the fix.

**A name collision to resolve before it costs a day.** The sequencing plan's **A2**
(the routing-versus-sequence gate) and the §5 claim list's **A2** (performance against
the legacy architecture) are unrelated. Proposed: call the gate **"the routing gate"**
in prose and prefix the plan's identifiers (`SEQ-A2`, `SEQ-B1`). A possessive label
like "Maude's A2" will not survive this handover and should not be adopted.

**`vals_decay` is the column.** Everything in the programme is the decay branch;
capture is out of scope, not merely unreported.

---

## What is retracted — do not rebuild anything on these

| claim | status |
|---|---|
| importance clusters into short runs at hundreds of times chance | **RETRACTED.** The null was random placement; the effect track is autocorrelated 0.965 at lag 1 and still 0.740 at lag 80, because it tracks selection mass (log-log r = 0.93). Random placement destroys structure the architecture puts there. |
| run lengths are "wrong for the encoding" | **RETRACTED** with it — autocorrelation persists past lag 80, which the ±25 GC window cannot explain either. |
| a GC confound in the composition signature | **NEVER EXISTED.** Elevated positions have GC 0.503 against a background of 0.501. It was diagnosed by eye from a k-mer table and controlled for without being measured. |
| the GC-preserving control | **BIASED, 3× harder than what it tested.** It drives GC at elevated positions from 0.501 to 0.682, because A/T positions can only be scored by A↔T and C/G only by C↔G. Its k-mer output is an artifact. |
| directionality at elevated positions | **NOT ESTABLISHED.** ~21% above the measured noise floor, but elevation is *defined* by magnitude and directionality rises monotonically with magnitude. A3 tests it. |

---

## What survives, and how weak it is

- cross-seed agreement on **sequence** (k-mer r = 0.75) far exceeding agreement on
  **position** (Jaccard 0.125) — but this uses the elevation rule, so **A2 negative
  propagates to it**
- the composition profile: keto (G+T) 1.16× background, amino (A+C) 0.84×, GC flat
- **the PWM ceiling: 1.73% of importance variance at width 9, held out.** This bounds
  everything else in the document and belongs beside any enrichment claim
- the enrichment survives the regional control (within-region r = 0.77–0.81)

**Under the definition in §1.1, what we have is a *composition* claim and not a motif
claim.** "Among equally-routed positions, these bases are preferred." Nothing in the
programme has yet earned the word "motif," and the ladder in §3.2.2 says exactly what
would.

---

## The discipline, which is not optional

**Every analysis records a hypothesis row before it runs** — hypothesis, what is held
fixed, what is deliberately not held, the null, and what a positive licenses you to
say. A row written afterwards is not a pre-specification. The toolbox (§3.2.2) is
the menu; the rule for choosing is causal: adjust for what correlates with the
exposure and is *not* on the causal path, **stratify** by anything that is part of
the exposure, never adjust for a consequence. **When in doubt, stratify** — it is
recoverable and adjustment is not.

Two things that are part of the exposure and must never be adjusted away:

- **selection mass.** `P(NMD) = Σ P(select k)·d_k`, so a substitution at a zero-mass
  candidate genuinely cannot move the output. Conditioning it away asks what would
  matter if ribosomes distributed uniformly, which the model never computes.
- **the composition of sequence downstream of a PTC.** It is part of what a PTC
  transcript *is*. See §5.2 — it is a natural experiment, not a confound.

---

## How we got burned, so you don't

**Eleven errors in two days across two windows, and not one was caught by whoever
made it.** They do not raise exceptions. They return tables.

**One pattern accounts for most of them: a set or a reference point enumerated
wrongly.** It appeared as a denominator containing a third of positions that were
zero by construction; as a "top 1%" that meant two different sets (Jaccard 0.24
between them); as a ratio statistic whose floor was assumed to be 0 when it was 0.39;
as a mean quoted against a median; as a null that destroyed structure the data has
architecturally. **Before quoting any proportion or comparing to any null, enumerate
the distribution of what you divided by — not a representative value from it.**

**Independent replication does not catch design errors.** Both windows implemented
the run-length statistic separately, agreed to three decimal places, and were both
wrong — because they shared the null, not the code. That is why the order is now
**hypothesis row → one implementation → replication only for load-bearing survivors**,
and why you write A2's second implementation against Harold's row rather than your
own.

**A correction can be undone by summarizing.** A run-length figure was corrected,
accepted by both windows, and then reintroduced as an error while compressing the
method document for a reader. Check the artifact, not the commit message — one commit
this session claimed two edits and contained neither, because an assertion failed
mid-script and the file was never written.

---

## A correction to this document, and why it is instructive

**This handoff originally told you not to start until three fixes landed. They had
already landed — at `4dd9f69`, 09:37:38, three minutes before this document was
committed at 09:40:51.** Verify it yourself rather than take it from me:

```bash
git log --format='%h %cd %s' --date=format:'%H:%M:%S' -1 4dd9f69
sed -n '/^## The stopping rule/,/^---/p' analysis_plans/ANALYSIS_SEQUENCING_PROPOSAL.md
grep -n -A6 'BLOCKING PREREQUISITE' analysis_plans/ANALYSIS_SEQUENCING_PROPOSAL.md
```

All three are in the plan: the stopping rule rewritten with A1 gone and the negative
scoped to single-base ISM; A2 given two implementations with the second written
against Harold's row; and B2 carrying a **BLOCKING PREREQUISITE** for an
annotation-derived PTC definition — the cell is 550 transcripts at **51.3% NMD
against a 44.0% background**, so it is barely enriched for what it is meant to
select.

**This is the third failure mode on the list below, committed inside the document
that lists it.** Not staleness: a correction that was made, verified, committed, and
then reintroduced as an open item while the state was summarized for a new reader.
Summarizing is where corrections go to die, and a handoff is the artifact most
exposed to it.

**It was caught mechanically, not by care.** The interpretability window went to the
commit before repeating a claim about it. That is the whole technique. Neither window
has ever caught one of these by being careful, and both have caught several by
checking the artifact.

**So: check this document against the repository before relying on any state it
asserts.** Including this section.

**And there is a second layer, which is the best instance of the class we produced.**
The interpretability window reported the gap as seven minutes. It is three minutes
thirteen. Their number came from the file's **mtime**, not the **commit timestamp** —
and the mtime has since moved again, because correcting this very section rewrote the
file. **A filesystem timestamp and a commit timestamp are two things with one name**,
and the quoted figure was not merely wrong but non-stationary. They checked *an*
artifact rather than *the* artifact, inside a message telling someone else to check
the artifact.

### The gap in the arrangement that you inherit

**Both compression failures were in documents written for readers, not in analyses.**
The run-length restatement in the method document, and the false blocking instruction
above. That is not a coincidence, and it explains why this class survived two windows
cross-checking each other continuously:

> **An analysis gets re-run. A document gets read.** A compressed error in prose
> arrives at the next window as a fact with no producer attached — nothing to
> re-execute, nothing to diff, no runlog to open.

**The two windows cross-checked each other's numbers and neither cross-checked the
other's prose.** Every number in this programme has been independently recomputed at
least once; no paragraph has. That is a gap in the arrangement rather than in either
window, and you inherit it.

The cheap fix, and it is the same shape as everything else here: **when a document
states a result, open the producer before believing it** — the runlog, the commit,
the script. Not vigilance. A step.

## The other windows

- **Harold — interpretability**, session `local_4c14cee7-60c5-4ecf-ba47-30edebf26c38`.
  Owns the plan, the first A2 implementation, seqlet extraction, and the SpliceAI
  gate. Reachable by `mcp__ccd_session_mgmt__send_message`. **Check what they send
  rather than accept it** — over two days each window overturned several of the
  other's conclusions and that is the arrangement's main value, not a friction in it.
- **Larry — results and figures**, session
  `local_beccba7a-741b-436d-8db2-4c29e52b4d02`. Do not let a retracted result reach a
  figure; they have been sent the retraction list.

Batch messages. Send one when something changes what the other window would do.

---

## Practical

- Local python `~/miniforge3/envs/nmd_model_local/bin/python`; cluster
  `/home/p.castaldi/.conda/envs/nmd_model/bin/python`.
- **Explorer: `p.castaldi@explorer.northeastern.edu`, `~/cc/nmd_orf_model_v5_4ct`.
  ASK PETE BEFORE EVERY LOGIN** unless he has granted a scoped window; grants are
  time-boxed and purpose-scoped and do not carry over.
- **The banks live only on Explorer. Do not copy them local** — one authoritative
  location, and the laptop has 8 GB of RAM against a bank that is 3.9 GB if read with
  `[:]`. Slice per transcript; `h5py` is lazy and `[:]` is what defeats it.
- **`vals` is NaN at the observed base by construction**, so
  `np.isfinite(x).all(1)` is **never true**. Use `(np.isfinite(x).sum(1) == 3)`.
  Three scripts across two windows hit this.
- **Copy files up, `sha256sum` both ends, then submit.** Two jobs failed this session
  from submitting first, and one of them exited 0 with SLURM recording COMPLETED.
- **The cluster clone's `git rev-parse HEAD` is not the provenance of what runs
  there** — scripts are copied on, not pulled. The SLURM scripts print the sha256 of
  the code they execute; trust that.
- **Run from the repo root.** `build_ism_bank` resolves the pool table and GENCODE
  flags from `Path(__file__).parent`.
- Two worktrees: this one is `master`; Harold's is
  `NMD_orf_model_v5_4ct_interp` on `interp`. **Merge often, both directions** — every
  tracked file exists twice and drifts. Stage explicit paths; **never `git add -A`**,
  which swept several of Harold's files into commits with unrelated messages.

---

## How to read the method document — this matters more than the credit

**It was not designed. It is a record of what we ran into.**

The ladder in §3.2.2 exists because a GC control had to be thrown away. The
autocorrelation-preserving null exists because the run-length null was invalid. The
magnitude clause exists because the directionality claim was circular. The definition
in §1.1 exists because we used the word "motif" for two days without one. **Every
piece of structure in that document is the scar of a specific failure**, and it is
worth knowing which, because:

**You will hit failures it does not anticipate.** It is not a standard handed down
finished — it is the current state of what two windows learned by getting things
wrong, and it stops exactly where our errors stopped. When you find a class it does
not cover, the correct response is to add a row, not to work around the gap or
assume the omission was considered.

The two limits of the arrangement that are already known and written down — that
**independent replication misses shared design errors**, and that **we cross-checked
numbers and never prose** — were both discovered the same way, by something going
wrong that the document had no entry for. Expect a third.

---

## The sentence to keep

We spent two days building instruments and the most valuable output was learning that
the word "motif" was doing work our adjustments had not earned. **If A2 comes back
negative, the honest finding is that the decay branch's sequence sensitivity is a
readout of its selection distribution** — a real result about this architecture, and
a more interesting one than a weak motif would have been. Write that rather than
reaching for a fourth instrument.
