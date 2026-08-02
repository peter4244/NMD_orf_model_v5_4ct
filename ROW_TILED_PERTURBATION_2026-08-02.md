# Row — does the initiation head read sequence, or read our window geometry?

*Interpretability window, 2026-08-02. Written before any code. Row template:
`ANALYSIS_SEQUENCING_PROPOSAL.md` → `## The row template`. Standards cited, not restated.*

**Scope: this decides between a MODEL claim and an INSTRUMENT claim**, which is unusual and
is the reason it is worth running. One outcome says the head does something; the other says
it reads an artifact we built.

---

## THE PLAN, in plain terms

### The question

The initiation head picks which reading frame matters. We know it correlates with ORF
length at +0.760, that a one-line "take the longest candidate" heuristic reproduces 97% of
its accuracy, and that Kozak — the actual biology of start-site strength — comes in weak at
+0.124.

We also know the encoding hands it a shortcut. The window's fill stops at the ORF midpoint,
so **fill extent = min(100, ORF length / 2)**, and fill saturation alone is a **~47× odds
marker** for "this is the real ORF." The head could be reading that instead of reading
sequence.

**So: does it read sequence, or does it read where the fill stops?**

### Two predictions, from two independent measurements, registered before the test

**PREDICTION A — the head reads initiation context** (model window, from the sparsity of
capture sensitivity: lower quartile at zero, upper quartile exceeding decay's in floor
units, so a minority of positions carry everything). If that minority is the start-codon
neighbourhood, sensitivity sits at a **fixed offset from the AUG anchor and does not move
when ORF length changes.**

**PREDICTION B — the head reads the fill boundary** (interpretability window, from the 47×
odds measurement). Then sensitivity sits at **offset min(100, length/2) from the anchor —
which *moves* with ORF length below 200 nt and pins at +100 above it.**

**They are mutually exclusive on the discriminating axis, which is not *where* the peak is
but *whether it moves*.** For a 60 nt ORF both predictions point at roughly the same tiles;
for a 190 nt ORF they point 85 bases apart; above 200 nt B is pinned while A is unchanged.
So the test is a **moving target**, and ORF length supplies the variation for free.

*This is the design we could not run for window size, because window geometry is hardcoded
and no model varies it. Here the variation is already in the data.*

### What we will do

1. **Bring down the checkpoint** (the tensor is already local, 478 MB; only the weights are
   missing) and read `n_bins` and `permute_bins` from its saved arguments — they set the
   tile size and, if bins are permuted, they veto the analysis entirely.
2. **Sample candidates**, stratified by ORF-length band and reference status. A profile
   averages across candidates, so 10–20k is ample and 800k is waste.
3. **Tile the start-codon window** and, for each tile, **shuffle the sequence within it**,
   leaving everything else identical.
4. **Measure the change in the head's own logit** for that candidate.
5. **Plot sensitivity against tile position, separately for each ORF-length band**, and ask
   whether the peak sits still or moves.

### What each outcome means

| result | reading |
|---|---|
| peak at a fixed offset from the AUG, unmoved across length bands | **A** — the head reads initiation context. The length correlation is something else, and the fill leak is present but unused |
| peak tracks min(100, length/2) | **B** — the head reads our fill boundary. Its selection competence is substantially an artifact of the encoding, and the retrain has a specific fix |
| both — a fixed peak *and* a moving one | the head does both. Their relative size is then the finding |
| neither, sensitivity flat | single-tile perturbation cannot see what the head uses. Escalate to multi-tile before concluding anything |

### What it costs

Minutes. The tensor is local; the capture path is one small convolutional encoder plus a
linear layer. 10–20k candidates × 20–40 tiles × 2 arms is a few hundred thousand encoder
passes — seconds of GPU, and the read is tens of megabytes.

### What could go wrong

- **Shuffled windows are off-manifold.** The model has never seen one. Shuffling preserves
  composition *and* the reading-frame channels, so it is closer to real input than
  substitution — but this is a real caveat and it travels with the claim.
- **Sub-bin tiles saturate.** The encoder takes a maximum within each bin, so perturbing
  part of a bin only registers if it held that bin's maximum. Finer tiles will produce many
  exact zeros. That is the instrument, not noise, and gets reported as a rate.
- **Nothing here says anything about the decay head**, or about motifs.

---

## The row

> **1 · Hypothesis.** The initiation head's per-candidate logit responds to perturbation of
> the start-codon window at an offset that is **fixed relative to the AUG** (it reads
> initiation context) rather than one that **tracks min(100, ORF length/2)** (it reads the
> fill boundary).
>
> **2 · Selection rule.** Unit = candidate ORF. Sampled, not enumerated: 10–20k stratified
> by **ORF-length band** — the axis the predictions diverge on — and by reference status.
> No elevation rule anywhere; every sampled candidate contributes every tile.
>
> **3 · Background.** None. Each candidate is its own control: Δ is measured against that
> candidate's own unperturbed logit. There is no comparison population to enumerate wrongly,
> which is what killed four analyses this week.
>
> **4 · Held fixed.** Everything except the perturbed tile — by construction, since the
> perturbation is local and the same candidate supplies the baseline. Reading frame held:
> the frame channels are positional, so shuffling nucleotides leaves them untouched.
>
> **5 · Deliberately not held.** ORF length — **it is the axis the test runs on.** The
> predictions differ precisely in whether the response moves with it, so conditioning it
> away would delete the measurement. (Third time this week that adjusting for length would
> have removed the mechanism; stated here so it is not proposed a fourth.)
>
> **6 · Null.** No placement null and none needed — tiles are pre-specified, so nothing has
> to establish that they exist. **In-sample reference:** far-upstream filled tiles, which
> should carry little, give the scale of "a tile that does not matter."
>
> **7 · Reference points.** Floor from the far-upstream tiles, measured in-sample per
> length band. Ceiling from the largest single-tile effect observed. Both reported; neither
> assumed.
>
> **8 · Aggregation.** Mean Δ per (tile × length band), interval by gene-clustered
> bootstrap. Signed **and** absolute reported — signed matters, because a gating head should
> show its logit *rise* when evidence against a spurious candidate is destroyed.
>
> **9 · Sweep.** Tile size: **bin width primary**, half-bin as the finer arm. Set by the
> encoder, not by us — receptive field is ~42 positions, so anything below that measures
> blur rather than location, and the bin is the finest position the model retains. A second
> **anatomically-aligned** scheme puts the in-ORF 100 in its own tile, because bins do not
> align with the AUG. Arms: **shuffle** (preserves composition, destroys arrangement) and
> **random substitution** (destroys both); their difference is the composition contribution.
>
> **10 · Decision rule, all outcomes fixed before the run.** The four rows of the table
> above. The fourth — flat sensitivity — is named explicitly because it would otherwise be
> read as "the head uses nothing," when it means single-tile perturbation cannot see what
> the head uses.
>
> **11 · Licensed.** A moving peak licenses *"the initiation head's selection response
> tracks the encoding's fill boundary."* A fixed peak licenses *"the head responds to
> sequence at a fixed position relative to the start codon."* **Neither licenses** any
> statement about motifs, about the decay head, or about what the model would do with a
> different window — that last one is unavailable and item 7 of
> `RETRAIN_ARCHITECTURE_CHANGES.md` says why.
>
> **12 · Owner.** Interpretability window. Second implementation only if the outcome is B,
> since that is the one that reaches the retrain.
>
> **13 · Enumeration.** n candidates, n genes, n per length band, tile size and scheme,
> shuffle seed, fill fraction **per tile**, the rate of exactly-zero effects, the checkpoint
> sha and its `n_bins`/`permute_bins`, and the mask expression. Never a profile without the
> fill fractions beside it.

---

## Second run — registered 2026-08-02, before it ran

*The first run refuted prediction A and left B unearned. This run tests a different
prediction and I am narrowing its stated purpose because my first proposal for it was
wrong.*

**WHAT IT CANNOT DO, corrected.** I proposed re-running with a finer grid to settle
whether the offset between peak and predicted fill boundary is constant (supporting B) or
rising (against it). **That is not available.** Downstream tiles are already 25 nt, below
the ~42-position receptive field, so finer tiles there measure blur rather than location.
**The offset question is bounded by the architecture, not by tile choice, and no tiling
run can settle it.** Recorded so it is not proposed a third time.

**WHAT IT DOES TEST — the model window's sparsity prediction, registered by them before
this run.** Their measurement found capture sensitivity *sparse rather than weak*: lower
quartile at zero, upper quartile exceeding decay's in floor units. If that is right, then
at 25 nt upstream resolution:

> **Adjacent upstream tiles should be HIGHLY VARIABLE**, because sparse critical positions
> land in some tiles and not others — **not the smooth monotone rise** the 125 nt tiling
> showed (0.031 at −838 to 0.18 at −88).

**The first run cannot distinguish these.** At 125 nt a tile holding one critical base and
a tile with weak signal spread over 125 produce the same median. That is their point and
it is correct.

**Outcomes, fixed before the run.**

| result | reading |
|---|---|
| upstream tiles highly variable, adjacent tiles differing severalfold | sparsity confirmed; the smooth rise at 125 nt was an averaging artifact |
| upstream profile still smooth at 25 nt | capture's upstream sensitivity is genuinely diffuse, and the sparsity seen at position level does not aggregate into localised tiles |
| variable only near the anchor, smooth far upstream | both — a localised initiation signal on a diffuse background |

**Implementation: no code change.** `--coarse 25` gives 25 nt tiles throughout, 40 tiles
against 12. Same script, same sha, same sample and seed as job 8900209 so the two runs are
directly comparable.

## Prerequisites

1. **The checkpoint.** Not local. `n_bins` and `permute_bins` live in its saved arguments
   and set the tile size. **If `permute_bins` is true, this analysis is void** — bins are
   shuffled per pass and position is destroyed by design; the bank's own code calls that
   variant the control arm. Confirm before anything else.
2. **If bin permutation is active at inference, one fixed permutation must be held** across
   the baseline and every tile, per `build_ism_bank`'s own note that an unpaired difference
   of two passes is otherwise dominated by the permutation rather than the perturbation.

## Assertions that must pass before any result is read

- A tile entirely outside the filled region gives **exactly zero**.
- Re-encoding without shuffling reproduces the stored channels **bitwise**.
- A one-position tile gives zero — nothing to permute.
