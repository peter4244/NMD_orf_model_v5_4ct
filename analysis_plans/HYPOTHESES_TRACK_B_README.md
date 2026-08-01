# Track B hypotheses for joint ranking — 2026-07-31

`HYPOTHESES_TRACK_B_2026-07-31.tsv` — **128 hypotheses**, extracted from the per-agent journals
rather than from the syntheses, because two of four syntheses fabricated material and the raw
agent returns are the only schema-validated record.

| round | n | lenses |
|---|---|---|
| R1 | 70 | nmd-core, initiation, splicing, rbp, utr3, elongation, perturbation, probing, architecture, training, counterfactual, stats, evolution, priorart |
| R2 | 50 | the-relation, uorf-efficiency, stop-context, training-dynamics, cross-config, internals, external-validation, conditional-content, sibling-contrast, controls |
| ARCH | 8 | query-key, stick-breaking, kozak-module, set-attention, factorized, minimal, reinitiation, sceptic |

Self-declared cost: 35 hours, 45 day, 27 days, 14 retrain, 7 retrain+build.

## Flags to apply before ranking — each measured today, each kills or demotes a class of proposal

1. **Anything premised on the model having been GIVEN the 50 nt rule is dead.** `n_downstream_ejc`
   is `sum(junctions > orf_end)` — thresholdless. The rule was neither supplied nor learned.
2. **Anything premised on the model ignoring channel 4 is dead.** A junction mark is worth a median
   0.253 logit, ~2 nucleotide substitutions.
3. **Anything predicting a distance response, knee or decay beyond ~+37 nt is unrepresentable** by
   this encoder (conv2 RF 42 nt, but the 4-nt pool grid caps stop/downstream co-visibility at +37).
4. **Anything predicting "flat in d" or "ratio ≈ 1" must be reframed as a BOUND, not a test** —
   the architecture entails it. This retires three of my own predictions and Track A's T1 framing.
5. **Single-base operators need a liveness gate.** 21.5% of substitutions are bitwise dead;
   channel-4 interventions are 0% dead. Liveness is per (operator, channel), not per model.
6. **Composition-moving operators are not usable.** Only `A_vs_T` has a near-null control reference
   (8% five-seed sign agreement vs 6.2% nominal); `G_vs_C` is GC-invariant *by construction* and
   still 81%. GC-invariance does not entail nullity — verify the reference empirically.
7. **Proposals specifying five checkpoints at `atg500_stop500` cannot run** — the only five-member
   fleets are `atg1000_stop1000` and `atg2000_stop2000`.
8. **The transcript FASTA IS available locally** at
   `nmd_deposit_2026/source_data/sqanti/nmd_lungcells_corrected.fasta`, 100% coverage of the h5
   isoforms. Any proposal routed to the cluster for sequence access can stay local.

## Already run, so rank as done rather than proposed

- Kozak PWM rescore (validated vs `Isopair::scoreKozakPWM` at 4.99e-13; prediction held 4/4; ties
  6.32 → 1.00; **does not** fix pool coverage).
- Threshold-vs-fixed-K pool coverage curves (MANE q05: 92.4% at 26.5 mean slots).
- Head-swap / factorised-mixture evaluation (costs 0.0008 AUC untrained).
- Liveness rates, and the corrected live-gated Kozak contrasts.
- The label-level fixed-count distance table (10.8% vs 46.8% at `n_downstream_ejc == 1`).
