# One page — NMD sequence-discovery day, 2026-07-31

*Track B. Shared with Pete and Track A. Every number here was sourced by me from a run log, a
committed script, or my own recomputation — nothing relayed unverified from an agent.*

**Scale.** ~46 agents across three fan-outs here (70 hypotheses / 14 lenses; 50 / 10; 8 architecture
designs + 4 judges) plus Track A's 17 / 8 lenses / 48 hypotheses. 22 commits.

---

## The result of the day, and it needs no model

Holding the model's dominant tabular feature **exactly** constant at `n_downstream_ejc == 1`, and
varying only the distance that feature cannot express:

| nearest downstream junction | n | NMD+ |
|---|---|---|
| ≤ 50 nt | 1,389 | **10.8%** |
| > 50 nt | 2,084 | **46.8%** |

**A 4.3× difference invisible to the model's inputs.** Computed from two TSVs, no forward passes.
The full gradient (≤37 / 38–50 / 51–100 / 101–200 / >200 nt) is 45.8 / 61.4 / 75.5 / 76.3 / **32.8%**.
The terminal drop is **not** class composition — it falls within class (PTC+ 78.7% → 34.1%), so it
is a real ceiling on the rule and must be attributed before this is reported.

## The synthesis

**Every sequence computation the model failed to learn corresponds to a tabular feature that made
learning it unnecessary.** Four measured independently today:

| not learned | made unnecessary by | withheld by D45? |
|---|---|---|
| the >50 nt stop-to-junction relation | `n_downstream_ejc` — a *thresholdless* presence count | **no** |
| initiation context at the start codon | `is_ref_cds` / `is_sqanti_cds` | yes |
| which ORF is translated | `is_ref_cds` — slot 0 *is* the answer | yes |
| cross-slot ORF geometry | `frac_start` / `frac_stop` | yes |

The one not withheld carries ~43% of the decision. That is now an argument from four measurements
rather than from symmetry.

## We have been using the wrong instrument

**21.5% of single-base substitutions are bitwise dead** (never move a pooled maximum); **0% of
channel-4 interventions are.** Liveness is a property of the (operator, channel) pair, not the model.
The encoder's co-visibility horizon is ~+37 nt — and the largest label gradient sits at 38–200 nt,
straddling and then exceeding it. The natural unit is the filter (64 of them, enumerable), not the
base.

## What converged independently

Across four fan-outs that could not see each other: **bounded filter catalog / argmax coverage**;
**stop identity × +4 context** (7 of 8 Track A lenses); **Kozak-gated uORF/oORF** (7 of 8);
**same-parse sibling contrast**. Independent convergence is the only merit signal available here.

## Reliability — two of four syntheses fabricated

Round 2 cited a "Round 3" that never ran. Track A's S3 invented an 11,276-vs-5,911 discrepancy
matching nothing on any tree. Round 1's headline verified to a decimal. **Rule adopted: no synthesis
number is relayed without being personally sourced to a journal entry or run log.** The failure is
invisible because fabricated material is formatted identically to real material.

## Corrections made today

- **§4's `category` is thresholdless** (`05t_ref_cds_features.R:385` rolls its own count while the
  file loads Isopair). A second, independent implementation of the same defect as the model feature.
  Manuscript issue, Track A's; scope deliberately not extrapolated to published ENST-only numbers.
- **My liveness gate deleted the reference cell**, making a favourable result look unfavourable.
  Corrected: Kozak −3 *strengthens* under the gate (+0.00262 → +0.00361, 97th percentile unchanged).
- **Kozak reads correctly once the operator matches the position.** −3 clears under `A_vs_T`, +4
  under `G_vs_C` — and the Cavener–Ray weights say `A_vs_T` carries 2.25× the information at −3 and
  `G_vs_C` 2.5× at +4. Two for two, not engineered. Gives a pre-registerable 8-position test.
- **PWM rescore**: validated against `Isopair::scoreKozakPWM` at 4.99e-13; pre-registered prediction
  held 4/4; ties at the isoform maximum 6.32 → 1.00. **It does not fix coverage** (46.9% vs 48.4% at
  K=5) — coverage is bounded by K, not by the score.
- **Kozak thresholding beats fixed-K** at every matched cost: MANE q05 gives 92.4% coverage at 26.5
  mean slots vs 93.6% at fixed K=30. The tail is real biology (long GENCODE transcripts, normal ORF
  density), so no cap — bucketed batching.

## Agreed order

**Sequential:** argmax-coverage catalog → implant-and-reencode harness → same-parse sibling causal
stack → reweighting test (mechanism readout only, never AUPRC).
**Parallel:** pool rebuild — PWM, MANE q05 threshold, ORF-length floor lowered, no cap.
**Track A:** §4 defect on published scope. **Not yet:** retrains, full-cohort ISM.

## Two landmines and one open question

`infer_uorf_attention.py` hardcodes `attn_0..attn_4` and `range(5)` (lines 174, 207) — it will
**silently truncate** at K>5. Both configs' `selected:` block says 500/500, where no five-member
fleet exists.

**Nobody has asked whether the sibling residual lives in the selection half or the trigger half.**
It decides whether the scanning-architecture work points at the discovery target. It is cheap.
