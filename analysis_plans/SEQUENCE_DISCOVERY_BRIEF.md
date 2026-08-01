# Brief — discovering sequence elements that increase NMD susceptibility, by model interpretation

*Written 2026-07-31 as the substrate for a hypothesis-generation fan-out. Track B (model repo).
Companion: Track A's `docs/SEQUENCE_DISCOVERY_SYNTHESIS_2026-07-31.md` in `nmd_lung_longread_2026`.*

---

## 1. The objective

We have a model that predicts, from transcript sequence and a small tabular block, whether an
isoform is degraded by nonsense-mediated decay. **It already performs well and performance is no
longer the goal.** The goal is to use the model as an instrument for **discovering which sequence
elements increase NMD susceptibility** — elements we could then state as biology, not as model
behavior.

A proposal is useful here if it produces a *sequence element* (a motif, a positional relation, a
compositional property, a structural feature of the transcript) that we can name, localize, and
defend as causal for NMD — and a way to test it that could come out negative.

---

## 2. The biology being discovered

**The canonical mechanism.** A ribosome terminates at a stop codon. If an exon–exon junction lies
**more than ~50 nt downstream** of that stop, the exon junction complex is still bound, UPF1
assembles, and the transcript is degraded. A stop with no downstream junction beyond 50 nt is a
normal termination and the transcript survives.

**Two consequences that make this a two-part problem.**

1. **Which ORF is translated determines where the stop is.** Scanning ribosomes select a start
   codon probabilistically, influenced by Kozak context (`gccRccAUGG`; −3 purine and +4 G are the
   strong positions), 5′ proximity, and competition from upstream AUGs. A different start means a
   different frame means a different stop means a different NMD verdict.
2. **Upstream ORFs are a second trigger, and are mechanistically the same trigger.** A uORF's stop
   codon is a stop codon with many downstream junctions — i.e. a PTC. Two sub-cases differ:
   - **uORF terminating in the 5′UTR** — the 40S can resume scanning and reinitiate at the main
     AUG, so this is a *leaky* trigger.
   - **oORF (overlapping ORF)** — starts upstream, reads past the main AUG **out of frame**,
     terminates inside the CDS. No reinitiation rescue. Mechanistically the more potent class.
   - An *in-frame* upstream ORF is an N-terminal extension, terminates at the main ORF's own stop,
     and is not a competing ORF at all.

**Other known contributors, less central here:** long 3′UTRs (EJC-independent NMD), stop-codon
identity and readthrough context, and 5′UTR structure affecting scanning.

**Two things the current pipeline pre-computes away**, and this may matter more than any motif:
splice-site context determines *where junctions are*, and start-codon context determines *where the
stop lands*. Both are sequence determinants of NMD whose consequences are summarized into tabular
features before the model sees them.

---

## 3. The model, precisely

Per transcript, up to **K = 5 candidate ORFs**. Each slot carries two 9-channel sequence windows and
5 tabular numbers.

**Sequence channels (per window):** 0–3 one-hot A/C/G/T; **4 = exon-junction positions**; 5 = rolling
GC fraction over 50 nt, computed from *this window's own sequence*; 6–8 = reading-frame one-hot
relative to this ORF's AUG.

**Windows.** An ATG window centred on the middle base of the start codon and a stop window centred on
the middle base of the stop codon, at width 1000 or 2000. A **midpoint clip** stops each window
crossing the ORF's midpoint, so the two windows partition the ORF; padding is on the inward side.
The stop window's *downstream* half is unclipped except by the transcript end.

**Tabular features per slot** (`ORF_FEATURE_COLS`): `frac_start`, `frac_stop`, `is_ref_cds`,
`is_sqanti_cds`, `n_downstream_ejc`.

**Encoder (shared weights across all 5 slots):**
```
conv1(9 -> 32, k=15) -> BN -> ReLU -> MaxPool1d(4) -> conv2(32 -> 32, k=7) -> BN -> ReLU
-> x.max(dim=-1)            # GLOBAL max over the length axis
-> fc -> 32-dim
```
Receptive field of `conv2` over the input is **42 nt**. Global max pooling discards absolute
position; only within a 42 nt receptive field is relative position preserved, and the mid-pool
quantises it to 4 nt.

**Per-ORF fusion:** `[atg_emb(32), stop_emb(32), ReLU(Linear(5 -> 32) of the tabular block)]` ->
`Linear(96 -> 64)` -> ReLU -> Dropout -> the ORF embedding.

**Aggregation:** `attn_score = Linear(64 -> 1)` applied to **each slot independently**, masked
softmax over slots, weighted sum of slot embeddings. Then `Linear(64 -> 32) -> ReLU -> Dropout ->
Linear(32 -> 1)`.

**Architectural facts that constrain interpretation, all verified in code:**
- **No parameter depends on K.** The checkpoint runs unchanged at K = 5, 6, 8.
- **No positional encoding over slots.** The model is permutation-invariant across slots; it cannot
  tell slot 0 from slot 3 by index.
- **`attn_score` sees one slot at a time.** Slot r's score cannot depend on slot q's features, so
  no cross-slot relation (e.g. "this ORF overlaps that one") is computable, even though
  `frac_start`/`frac_stop` contain the information.
- **The classifier sits OUTSIDE the attention mixture** — `g(Σ α_r e_r)`, not `Σ α_r g(e_r)`.
  Moving it inside costs 0.0008 AUC on the trained model with no retraining, and makes a
  selection-vs-verdict decomposition exact.
- Total 34,050 trainable parameters; 32 conv channels × 2 branches bounds the number of distinct
  sequence features the encoder can hold.

---

## 4. The data, and its floors and ceilings

- 41,765 isoforms, 22.3% NMD+. Labels from mashr differential expression across 4 cell types.
- **Split is by CHROMOSOME** (test chr1/3/5/7, val chr2/4). Zero genes span splits. Separate
  paralog holdouts exist. Gene-identity leakage is not a live concern.
- **ORF scan has a 33 nt floor.** Minimum `orf_length` is 33; zero ORFs are shorter. A canonical
  3-codon regulatory uORF (e.g. ATF4 uORF1) **cannot be a candidate at all**.
- **Only 5 of a mean 36.6 ORFs per isoform are kept**, by priority: reference CDS, then TD2 CDS,
  then top Kozak. `kozak_score` is a 3-level integer (has −3 purine + has +4 G), so ties are
  common — mean 6.3 tied at each isoform's maximum — and are broken by transcript order.
  Between-tier selection is nonetheless strong: 78.7% of selected qualifying upstream ORFs are
  top-tier against 7.9% of unselected.
- **Half the true upstream-PTC population is invisible to the model** (12,010 isoforms in the full
  scan vs 5,913 in the slots) — and the invisible half is the weak half (3.9% NMD+ vs 12.6%).
- **`is_ref_cds` and `is_sqanti_cds` disagreement is a strong NMD proxy**: 49.5% NMD+ where TD2
  disagrees with the reference at slot 0, against 12.0% where they agree. TD2 avoids calling
  PTC-bearing ORFs as the CDS, so the disagreement encodes PTC status via a tool's bias.
- **2,240 isoforms (5.4%) are NMD+ by construction** (`no_ref_isoform`: the gene has no dominant
  non-NMD coding isoform). A deterministic label with no biological content.

**Mechanism classes over the model universe** (reference-AUG anchored, PTC = junction >50 nt
downstream): PTC+ 6,864 (70.0% NMD+); PTC− with an upstream PTC-bearing ORF 5,913 (12.6%, of which
89.4% terminate in the 5′UTR and 10.6% are out-of-frame oORFs); PTC− no trigger 15,765 (2.2%); no
reference-AUG anchor 13,223 (25.9%).

---

## 5. What has been measured

- **Test performance:** ensemble AUC 0.9310, AUPRC 0.8351 at `atg1000_stop1000`.
- **A GBM on 25 tabular numbers alone** reaches AUPRC 0.8035. *Caveat established since: this is
  not "sequence only adds 0.038", because `n_downstream_ejc` is itself a pre-computed summary of
  sequence-determined events — splice sites, frame, stop position — so the comparison measures what
  sequence adds after the problem's most important sequence computation has been done for it.*
- **Branch decomposition (exact enumeration):** structural 59–78% of mean absolute log-odds
  displacement across 50 cells; EJC count dominates within it.
- **Kozak −3, GC-matched operator (A vs T):** cohort mean +0.00262, across-member |t| 3.26, 97th
  percentile of 37 frame-matched control positions. **But per isoform the median is +0.00007 and
  only 52.1% are positive** — a small mean shift on a near-symmetric distribution.
- **Kozak +4:** unsupported under every operator; indistinguishable from a global preference for G.
- **Per-isoform max |Δ| across banked positions has median 0.031 against a cohort-mean maximum of
  0.0035** — roughly 9× dilution from averaging across isoforms at a fixed anchor-relative index.
  Independent measurement puts single-base ISM per-isoform maxima at median 0.137.
- **The Kozak effect's sign follows the perturbed slot's own PTC status**: +0.0087 where slot 0's
  stop is EJC-triggering, −0.0001 where it is not, at matched attention.
- **The effect reverses with attention**: +0.0058 in the middle slot-attention tercile, −0.0024 in
  the top.
- **Within-gene concordance over label-discordant siblings is 0.919, and 0.803 on the 21.9% of
  pairs where both siblings share the same slot-0 EJC count** — a validity metric that controls
  for gene identity and for the dominant structural feature.

---

## 6. What has already died, and why — do not re-propose these

- **DeepSHAP / gradient attribution.** Completeness error 129–307% of the effect it decomposes on
  this architecture; rules do not attach to functional (non-`nn.Module`) nonlinearities. One-hot
  sequence also lies on a simplex where input gradients acquire an off-simplex component.
- **Dinucleotide-preserving shuffle of a 3-mer.** It is the identity map — 100% of 64 3-mers have
  exactly one valid arrangement.
- **`mean_abs` as a positional headline statistic.** With many weak players it cannot discriminate;
  it ranked the composition-confounded operator above the clean one.
- **Across-member sign agreement as standalone evidence.** Members agree at *meaningless* control
  positions 65–92% of the time against a 6.25% chance rate.
- **Inference-time flag ablation as a measure of sequence contribution.** It shrinks control
  positions (0.66×) alongside targets (0.42×) — it measures attention dilution.
- **Branch-Shapley share as a success criterion for a training regime.** Modality dropout
  manufactures redundancy and Shapley splits credit among redundant players, so the structural
  share falls whether or not anything was learned.
- **Permuted-label FDR at the causal gate** — the null count is ~0 by construction.
- **Naive zeroing as modality dropout.** With the input identically zero, `∂L/∂W = δ·xᵀ = 0`
  exactly; the weights reading it cannot relearn and the branch becomes a learned constant.
- **Frame-rotation probes** — off-manifold, and rotate-by-3 is the identity.

---

## 7. Measurement traps this project has paid for

1. **Importance is a contrast, not a measurement.** Every claim needs three references: the
   method's own noise floor, other positions, and a matched population. Each has silently failed
   here while still producing a number that looked like an answer.
2. **Composition dominates identity.** A single-base substitution moves the recomputed GC channel.
   Under a G-vs-not-G operator, control positions are 100% same-signed with |t| median 5.79 —
   confident structure at *every* position. Only GC-matched operators (A vs T, G vs C) have a
   reference approaching null.
3. **Controls must be matched on frame offset.** The created-motif exclusion rate is 23.6% at one
   codon offset and 8.7% at another.
4. **Cross-isoform averaging at a fixed anchor-relative index destroys per-isoform signal** —
   roughly 9× here. Gate per isoform first, aggregate second.
5. **Bootstrap must be clustered by gene** (measured design effect **1.71** (`is_nmd` ICC 0.300 within gene, effective cluster size 3.36 — intervals 1.31× wider)). *Corrected 2026-08-01:* this read
   "variance inflation 5.47", which was `sd 5.47` — a percentage-point standard
   deviation lifted out of the power-matching section of
   `exp2b_control_validity_runlog.txt`, where it has nothing to do with genes or
   clustering. The instruction is unchanged; the magnitude was wrong, and it was wrong
   in the direction of overstating what clustering costs.
6. **Conditioning on the annotated ORF is circular** when ORF identification is half of what the
   model is supposed to do.
7. **A confident negative in this area has twice been a confounder rather than the model.**
8. **A comparison statistic can have a null that is not 50%.** `|slope fitted on a short window| >
   |slope fitted on a long window|` held for 83.2% of isoforms in the junction-distance probe and
   means nothing: a fit over a shorter x-range has higher variance under the same noise, so the
   inequality holds almost mechanically whatever the truth. Establish a statistic's null before
   reading it as evidence — the same shape as trap 2, where control positions were 100% same-signed.
9. **The only ground truth for "which ORF is translated" is GENCODE traced through the isoform.**
   Absent ribosome profiling, any discovered initiation feature is capped at "predicts the
   annotated start", not "predicts ribosome selection".

---

## 8. What a good proposal looks like here

- It names a **sequence element**, not a model property.
- It says what would make the finding **spurious**, and how that is excluded.
- Its test has a **matched reference** — what is the comparison, and why is it matched?
- It states whether the readout is **per isoform** or a cohort aggregate, and if aggregate, why
  averaging does not destroy it.
- It is feasible: local CPU, five trained checkpoints, ~40k isoforms, the full ORF scan, junction
  tables and transcript FASTA. Retraining is on the table (~14–17 epochs, wall time never measured).
- It says what it would cost and what it would rule out.
