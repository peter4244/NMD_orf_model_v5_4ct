# Analysis 4 Plan — where in the sequence windows the model looks

*Written 2026-07-31, before any result exists.*

This document specifies what is done to the data. Code is written from it.

**Claim ids are the ledger's `section.paragraph.sentence` scheme.**

---

# 1. Purpose

> We have trained a model that predicts, from transcript sequence alone, whether an isoform is
> degraded by nonsense-mediated decay. This analysis establishes two things: how accurately it does
> so on transcripts and chromosomes it never saw, and what kinds of sequence and structural
> information it relies on to do it. Both are reported with uncertainty that reflects the two ways
> the answer could have come out differently — a different training run, and a different draw of the
> reference transcripts used to compute an attribution.

**Approved by Pete, 2026-07-30. Do not edit this paragraph.**

Analyses 2 and 3 divided the evidence among branches and among structural features. Analysis 4 opens
the two sequence branches: **where inside each window the attribution sits, and what it falls on.**

---

# 2. What makes this analysis different, and the problem it has to solve first

**Every acceptance decision in Analyses 1 to 3 rested on exactness.** Three branches enumerate to 8
coalitions and five features to 32, so the Shapley values were solved exactly and the efficiency
residual could only be floating-point rounding. That made a hard reject at 1e-12 affordable, and
`ANALYSIS2_PLAN` says why it would not be otherwise: *"a hard reject is affordable here and would
not be for an approximate method, where a small non-zero residual is expected and tells you
nothing."*

**A sequence window has 1,000 or 2,000 positions and nine channels. Nothing enumerates.** Attribution
is computed with DeepSHAP, which is approximate, and D41's foreclosure is explicit that Analysis 3's
exact-enumeration argument does not carry here.

So **this plan cannot inherit its acceptance criterion, and most of it is about what replaces one.**
Section 3e is the substantive part of this document; the rest is structure carried over from
Analyses 2 and 3.

---

# 3. Data

## 3.1 Terms

Datasets A and B are those of the preceding plans and their terms carry over unchanged. Four are new.

**Per-nucleotide attribution.** How much one position in one window contributed to one isoform's
prediction, in log-odds. Signed: positive pushes the prediction toward NMD susceptible.

**Window coordinate.** `ANALYSIS1_PLAN` §2.2 records that each window is centered on the **middle
nucleotide** of its codon, at array index `W/2`, and states: *"Any analysis that does use positional
coordinates must state which convention its axis uses."* **This is that analysis.**

Every position reported by Analysis 4 uses the **codon convention**, in which the A of the start
codon is **+1**, the base before it is **−1**, and there is no zero. Array index `W/2` is the middle
base of the codon, so the mapping from array index `i` to reported position is:

```
pos(i) = i - W/2      for i <  W/2      (upstream, negative)
pos(i) = i - W/2 + 1  for i >= W/2      (the codon and downstream, positive)
```

The start codon therefore occupies reported positions +1, +2, +3 at array indices `W/2 − 1`, `W/2`,
`W/2 + 1`. A figure axis, a table column and a sentence in the manuscript all use this convention or
say which other one they use. **Claim 5.3.1 speaks of importance "around the start and stop codons"
without a convention, so the rewritten sentence states one.**

**Reference draw.** As in Analyses 2 and 3: 500 training isoforms, five draws, `RandomState(42 +
1000*R)`. Reused unchanged, so the three analyses share baselines.

**Channel.** One of nine per position: four nucleotide indicators, an exon-junction indicator, a
50-bp rolling GC value, and three reading-frame indicators. Attribution is per position **per
channel**, and a claim about a nucleotide is a claim about the four nucleotide channels only.

## 3.2 Datasets

Unchanged: **A** = `results_4ct_dn/nmd_orf_data.h5`; **B** = the ten checkpoints in
`results_4ct_sweep/`. The model contract of the preceding plans applies and is not restated.

**Dataset C — outputs, written to `results_interp_all/`**

| file | contents |
|---|---|
| `nt_shap_{branch}_{tag}_seed{S}_all_run{R}.npz` | per-position per-channel signed attribution for the explained isoforms, one array per branch ∈ {start, stop} |
| `nt_profile_per_cell.tsv` | one row per (configuration, level, member, draw, branch, position): mean signed and mean absolute attribution, by label class |
| `nt_decomposition_{tag}.json` | the positional summaries with both spreads, the convergence and stability diagnostics of §3e, and the exact-anchor comparison |

---

# 4. Analysis 4

## 4a. Scientific question

**Where inside the start and stop windows does the model's evidence sit, and what does it fall on?**

Three sub-questions, each attached to a claim:

1. **Is attribution concentrated near the codons?** (claim 5.3.1) A model that had learned initiation
   and termination context should weight positions close to the anchor and little far from it.
2. **Does the start window carry Kozak information?** (claim 5.3.2, legend 5.6.7) The Kozak consensus
   makes specific predictions — a purine at −3 and a G at +4 — which are positions and channels
   stated in advance here, not selected after seeing the profile.
3. **Does the stop window distinguish stop codon identity?** (claim 5.3.2) In particular whether UGA
   is weighted differently from UAA and UAG.

And, as everywhere in this work: **how much would any of it have differed under a different training
run, or a different draw of reference isoforms?**

**Not in scope.** Claim 5.3.3 (*UGA more common in NMD-susceptible transcripts, 56% vs 48%*) is a
direct analysis of sequence composition, not of model attribution, and is already a recorded ledger
defect. Analysis 4 may report stop codon identity as a covariate but does not address 5.3.3.

## 4b. Starting data

All 41,765 isoforms; ten members; the five shared reference draws. Summarized over the same
population as Analyses 2 and 3 — `label == 1`, 9,321 isoforms, a census — with the control class
retained separately because claim 5.3.2 and legend 5.6.7 are comparative.

## 4c. Approach

Eight steps, mirroring Analyses 2 and 3 so the three are comparable. Steps 1 and 2 are the shared
population and reference draws, re-run rather than inherited.

### Step 3 — attribute, per branch

For each (configuration, member, draw) and each branch, run DeepSHAP with the branch's wrapper —
which fixes every other input at its observed value and varies only that window — against the draw's
500 reference isoforms, over the explained cohort.

### Step 4 — collapse to profiles

```
    for each branch, position p, channel c:
        signed[p,c] = mean over isoforms of phi[i,p,c]
        absol[p,c]  = mean over isoforms of |phi[i,p,c]|
    reported separately for label == 1 and label == 0
```

Positions are converted to the codon convention of §3.1 **once, here**, and every downstream artifact
carries converted coordinates.

### Step 5 — accept or reject: see §3e, which replaces the residual gate

### Step 6 — reduce, at three levels

Member, ensemble and leave-one-out ensemble, as in Analyses 2 and 3, with **signed values averaged
across members first**. The order matters more here than anywhere: per-position attributions change
sign across the window, so averaging magnitudes first would manufacture importance where members
disagree.

### Step 7 — the two spreads, and the pre-registered positional tests

Both spreads at their own levels, as amended by the statistical review: interpretation spread across
draws with its standard deviation **and** standard error, training spread across members with its
own centre, the delete-one-member jackknife, the ensemble-size bias, and the variance components.

**Positions and channels are named before the profiles are seen:**

| question | quantity | pre-registered target |
|---|---|---|
| concentration (5.3.1) | fraction of total mean \|attribution\| within ±50 of the anchor | reported for both branches; no threshold asserted |
| Kozak (5.3.2) | mean signed attribution at −3 on the A and G channels, and at +4 on the G channel | these four, fixed in advance |
| stop identity (5.3.2) | mean signed attribution on the channel spelling position +3 of the stop codon, by codon | UGA vs UAA vs UAG |

A position or channel not in this table may be described but is reported as **post hoc** and labelled
so.

### Step 8 — write

The three artifacts of Dataset C, plus the per-cell table that makes every estimator recomputable.

## 4d. What the decomposition does and does not establish

Attribution near a position says the model's output is sensitive to that position given this
reference set. It does not establish that the model represents a motif, and *"the model learned
Kozak sequence information"* (claim 5.3.2) is a stronger statement than a profile can support. The
rewritten sentence says what was measured.

The branch wrapper holds the other branch and all structural features at observed values, so this is
a **within-branch** decomposition and its numbers are not comparable term-by-term with Analysis 2's
branch shares.

## 4e. Acceptance, when the residual is no longer a checksum

**The residual is reported and is not a gate.** DeepSHAP's completeness error is method error, not
arithmetic error; a small value would not prove the calculation was the intended one, and a large one
is uninformative without a scale. It is recorded per cell as a magnitude relative to the prediction
delta, and a distribution over the fifty is reported. **No cell is accepted or rejected on it.**

Four checks replace it. **The first is the strongest and is the reason this analysis is tractable at
all.**

**1. An exact anchor.** Analysis 2 decomposed the same model over three branches *exactly*. The start
and stop branches' Shapley values from that analysis are known without approximation, for the same
isoforms, members and reference draws. DeepSHAP's per-position attributions within a branch are a
finer decomposition of a quantity we already know exactly, so their **sum over positions and channels
is compared against Analysis 2's φ for that branch**, per isoform. Agreement is the acceptance
criterion; the tolerance is set from the two measured regimes — honest method error versus a
misattributed or misaligned file — exactly as the cross-file prediction tolerance was set in
Analysis 2, and **not fitted to the observed maximum**.

**2. Convergence in the reference set.** Attribution is computed at n = 100, 250 and 500 reference
isoforms on one cell per configuration. If the profile has not converged by 500 the number of
references is a free parameter the result depends on, and that is reported rather than absorbed.

**3. Replicate stability.** The five draws already give an interpretation spread. Here it is also a
diagnostic: a position whose attribution is not stable across draws is not a position about which
anything is claimed. The claim-bearing positions of step 7 are reported with their across-draw
spread beside them, and a spread exceeding the effect is stated as such.

**4. A negative control.** The same pipeline is run on **shuffled labels** for one cell per
configuration. It must not produce the pre-registered Kozak or stop codon signal. This is the check
that distinguishes "the model learned a motif" from "this pipeline produces motif-shaped output".

**Why four rather than one.** The exact anchor tests alignment and completeness; convergence tests
whether 500 references suffice; stability tests whether the answer is determined; the negative
control tests whether the method manufactures structure. The residual tested none of these and, for
an approximate method, would have tested nothing at all.

## 4f. Cost

Unknown, and **measured before anything is submitted** — the discipline that has now caught a 35.3
GiB peak against a 32 GB request and a per-isoform rate I had wrong by 6×. DeepSHAP over windows of
1,000–2,000 positions is expected to cost substantially more per isoform than either preceding
analysis, and whether the explained cohort is the full 41,765 isoforms or a declared subsample is a
decision taken **after** the measurement and recorded here, not assumed now.

---

# 5. What this supports in the manuscript

Claims **5.3.1** and **5.3.2**, and figure legend **5.6.7**. Not 5.3.3, which is a sequence-composition
claim, and not 5.3.4, which is a citation.
