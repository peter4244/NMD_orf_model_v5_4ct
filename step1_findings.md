# Step 1 — Factual Accuracy Findings

**Report:** `orf_model_report_v5.Rmd` at commit `b890e77`
**Reviewer:** Plan agent, sequential 5-step protocol
**Date:** 2026-04-30

## Summary

~75 numeric claims checked. **~58 correct, 11 incorrect, ~6 unverifiable.** Most inline-R values render correctly from live TSVs; almost all errors are in hardcoded prose that was not updated when data was regenerated for the 4ct retrain.

The major sample-size cascade (39,938 → 10,131 test → 2,268 NMD / 7,863 Control), the subgroup partition (Control 7,863; PTC+ 1,798; PTC- ATG retained 194; PTC- ATG lost 276), the 47% reclassification figure, the 12-cell window grid, K=5 ORFs, 9-channel encoding, and stability CVs all check out.

---

## Top 3 issues (highest impact)

### Issue A — Best-model selection rationale
- L74-81 (§0) and L540-541 (§1) claim atg500_stop500 was "selected by AUPRC" and that "AUPRC peaks at STOP=500 across all ATG sizes."
- **Reality:** at ATG=500, STOP=**1000** has higher AUPRC (0.8387 vs 0.8330). atg500_stop500 is the **AUC winner**, not the AUPRC winner.
- Decision needed: change rationale to AUC-based selection, OR re-examine which model should be primary.

### Issue B — Stale 6ct hardcoded subgroup-branch numbers
- L2440: "27% ATG branch vs 8% for PTC+" → actual 4ct: **29% vs 9%**
- L2924: "from 8% in PTC+ to 27% in PTC- ref ATG retained, dominating 25%" → actual: **9%, 29%, 29%**
- Fix: update numbers, or convert to inline R for stability.

### Issue C — §3.1 Kozak / §4.1 stop-codon prose contradicts data
- L931: "At -3, direction reverses (NMD positive, Control slightly negative)" → **both negative** in run1 TSV.
- L932: "G at +4 has negative SHAP in NMD but positive in controls" → **both positive**, NMD ~7x larger.
- L978: "NMD has ~2x higher overall |SHAP| than Controls" (applied to stop branch) → actual stop ratio **1.32x** (the 2x figure fits the ATG branch).
- L1189: "TGA NMD magnitude is ~2x larger than Control" → actual NMD is **smaller** (0.0055 vs 0.0076, ratio 0.72x).

---

## All findings by section

### §0 Data Loading & Model Overview

1. **L74-81** — Best model "selected by AUPRC". **INCORRECT** — atg500_stop500 wins by AUC; atg500_stop1000 has higher AUPRC (0.8387 vs 0.8330). Fix: change "AUPRC" → "AUC" in rationale.
2. **L284** — DeepSHAP subset "~448 NMD, ~1552 Control". Actual 444/1556 written. **MINOR** (acceptable with "~").
3. **L350** — "Reference CDS rank-0 available for ~73% of transcripts". Actual: **69.6%** (test set). **INCORRECT** — likely 6ct leftover. Fix: change to "~70%".
4. **L398** — Branch decomposition "structural 61%, stop 29%, ATG 10%". Actual: 60.7%, 28.8%, 10.5%. **CORRECT** within rounding.

### §1 Performance Comparison

5. **L540** — STOP=500 optimal by AUPRC. **INCORRECT** — STOP=1000 wins at ATG=500. Fix: rewrite to "AUPRC peaks at STOP=500 for ATG=100 and ATG=1000, but at STOP=1000 for ATG=500. STOP=500 with ATG=500 was selected as the best model by AUC."
6. **L541** — "AUPRC peaks at STOP=500 across all ATG sizes". **INCORRECT** (same data).
7. **L541** — "ATG=100 and ATG=500 perform similarly by AUPRC (< 0.002 difference at STOP=500)". Actual: **0.0041**. Fix: change to "< 0.005" or "0.004".
8. **L540** — `auc_stop2000` inline. **CORRECT**.
9. **L542** — AUPRC gain stop100→500 inline. **CORRECT**.

### §2 Feature Importance

10. **L679** — Branch percentages NMD-only (inline). **CORRECT**.
11. **L683** — "Within structural branch (61%)". **CORRECT**.
12. **L764** — "lag-3 autocorrelation for G is 0.95... 90% of FFT power at period 3". Actual: autocorr **0.86**, single-frequency period-3 power **45.3%** (narrow-band ~75%). **INCORRECT**. Fix: recompute or remove specific numerics.

### §3 Start Codon Signals

13. **L774-781** — Stability inline. **CORRECT**.
14. **L930** — Inline `kz_A_m3_nmd`. **CORRECT** (renders -0.001 / 0).
15. **L931** — "At -3, direction reverses (NMD positive, Control slightly negative)". Actual: NMD -0.0012, Control -0.0002, both negative. **INCORRECT**. Fix: rewrite to match data.
16. **L932** — "G at +4 has negative SHAP in NMD but positive in controls". Actual: NMD +0.0031, Control +0.0004, both positive. **INCORRECT**. Fix: "G at +4 is positive in both classes, with NMD ~7x larger than Control".
17. **L932** — "+4 signal in NMD comes from A and T, not the canonical G". Actual: A=-0.0009, T=-0.0010, G=+0.0031. **INCORRECT** — G is the only positive contributor.
18. **L874** — Kozak NMD logo n_nmd. **CORRECT** (444).
19. **L885** — Kozak Control logo n_ctrl. **CORRECT** (1556).

### §4 Stop Codon and 3'UTR Signals

20. **L978** — "NMD ~2x higher overall |SHAP| than Controls" (stop branch). Actual ratio: **1.32x**. (The 2x figure fits the ATG branch, not stop.) **INCORRECT**.
21. **L1040** — Stop codon NMD logo n. **CORRECT** (444).
22. **L1064** — TGA enrichment in NMD (49.6% vs 48.4%). Inline values **CORRECT**; "enrichment" framing debatable but minor.
23. **L1189** — "TGA has positive signed SHAP in both NMD and Control, but NMD magnitude is approximately 2x larger". Actual: NMD 0.0055, Control 0.0076 — NMD is **smaller** (0.72x). **INCORRECT**. Fix: rewrite — TGA positive in both, Control magnitude slightly larger.

### §5 Attention

24. **L1266** — Code comment "10 isoforms with all-zero weights excluded". Actual: 0 isoforms have all-zero weights. **INCORRECT** (stale comment, no user-facing impact).
25. **L1296** — Median max attention (inline). **CORRECT**.
26. **L1321** — Rank-0 dominance (inline). **CORRECT** (NMD 84.3%, Control 87.1%).
27. **L1322** — Mean entropy (inline). **CORRECT** (NMD 0.913, Control 0.948 bits).

### §6 GC Enrichment

28. **L1378-1471** — Hardcoded "n=2,000 samples" titles. **CORRECT**.
29. **L1489** — Hardcoded GC trajectory endpoints "~48%→~36% controls, ~50%→~45% NMD". **UNVERIFIABLE** without recomputation against `gc_content_across_stop_window_atg500_stop500.tsv`. Flag for follow-up.

### §7 Subgroup Definitions and Attribution

30. **L1602** — "Reclassifies ~47% of original ref-ATG-lost as PTC+". Actual: 47.4%. **CORRECT**.
31. Subgroup counts (`subgroup_counts` chunk). Sums correctly. **CORRECT**.
32. **L1724-1726** — `n_downstream_ejc`, `is_ref_cds` inline. **CORRECT**.
33. **L1806-1812** — Inline branch percentages by subgroup. **CORRECT** (renders from live data).

### §8 Subgroup Predictive Performance

34. **L1928** — Sensitivity inline values. **CORRECT** (PTC+ 92.0%, PTC- ret 33.0%, PTC- lost 35.1%).

### §9 Additional Analyses

35. **L1989, L2018** — Junction motif n inline. **CORRECT**.
36. **L2025** — Hardcoded "65% A at -1, 82% G at 0" for first 3'UTR junction motif. **UNVERIFIABLE** without focused recomputation against motif_logos_stop run1 TSV. Flag for follow-up.
37. **L2093** — EJC dose-response inline percentages. **CORRECT** (renders from data).
38. **L2185** — Poly(A) prevalence inline. **CORRECT** (84.3% NMD, 86.0% Control).
39. **L2185** — AATAAA fraction inline. **CORRECT** (61% in NMD).
40. **L2440** — "27% ATG branch vs 8% for PTC+". Actual: 29.2% vs 9.1%. **INCORRECT** (stale 6ct). Fix: "29% vs 9%".
41. **L2924** — "from 8% in PTC+ to 27% in PTC- ref ATG retained, dominating 25%". Actual: 9.1%, 29.2%, 29.4%. **INCORRECT** (3 stale numbers). Fix: replace or convert to inline R.
42. **L2678-2680** — Hardcoded attention dominance "PTC+ ~95%/0.88, PTC- ret ~50%, PTC- lost ~63%". **UNVERIFIABLE** without recomputation; likely 6ct-era numbers, may diverge a few pp.
43. **L2802** — Quartile accuracy inline. **CORRECT**.
44. **L2928** — Subgroup sensitivities inline. **CORRECT**.

### §9.13 Error Analysis

45. **L2952** — Confusion matrix inline. **CORRECT** (TP=1816, FN=452, FP=595, TN=7268, sens 80.1%, spec 92.4% at threshold 0.5).
46. **L3037** — No-PTC NMD inline. **CORRECT** (94/2268, 4.1%, mean prob 0.166).

---

## Cross-cutting observations

- **No 6ct cell-type list slipped into prose.** No prose enumerates cell types. METHODS.md needs separate check.
- **No date references in prose.** Nothing to verify on DE/training dates.
- **Channel count = 9** (not 10), confirmed against `nmd_orf_data.h5`. The CLAUDE.md "10 channels" may be inaccurate; report's "9 channels" is correct.
- **Joint SHAP motif TSVs missing from results_4ct/.** Only `motif_logo_atg_*_run1.tsv` and joint-subgroup variants exist. The chunk has fallback logic so it renders, but provenance comments saying "joint SHAP logos (all test samples)" mislead — the §3.1 Kozak and §4.1 stop-codon logos actually come from run1 marginal SHAP. Worth flagging in data-status panel.
- **Inline R values are reliable.** Errors concentrate in hardcoded prose left over from the 6ct version.

---

## Adjudication needed (before fixes are applied)

1. **Issue A (best-model rationale):** Switch text to "selected by AUC", or revisit selection criterion?
2. **Issue C, L978 / L1189:** Remove the "~2x" claim entirely, or rewrite to the actual ratios? Same question for the §3.1 Kozak prose at L931-932.
3. **L2678-2680 (attention dominance), L1489 (GC endpoints), L2025 (junction motif freqs):** Should I dispatch a follow-up agent to recompute these, or update inline R, or leave as-is and accept "~" qualifiers?
4. **L764 (period-3 G stats):** Recompute and write the actual numbers, or remove the specific numerics in favor of "strong period-3 modulation"?
