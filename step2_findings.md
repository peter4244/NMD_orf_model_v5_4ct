# Step 2 — Result Correctness Findings

**Report:** `orf_model_report_v5.Rmd` at commit `47af68b` (post-Step-1)
**Reviewer:** Plan agent + direct Bash verification
**Date:** 2026-04-30

## Summary

Step 1 verified that prose numbers match data files. Step 2 asks a different question: are the **results themselves correct** — right populations, right tests, right denominators, biologically plausible? The picture is mixed.

**Population-level analyses are largely correct** (predictions, calibration, confusion matrix, EJC dose-response, 5'UTR features, KernelSHAP additivity, subgroup branch decomposition).

**Three substantive concerns:**
1. **CRITICAL** — §4.1 stop-codon chi-squared test is computed from SHAP-subset reconstructed counts with unweighted aggregation across NMD subgroups. The figure caption shows p=0.86 (NS) while the prose claims "TGA is enriched in NMD" — direct contradiction. Properly computed from the full test set (10,131 ORFs, filtering to canonical TGA/TAA/TAG = 389): χ² = 8.40, p = 0.015. **And TAG, not TGA, is the most enriched stop codon in NMD** (NMD 26.3% vs Control 16.2%); TGA is barely different (NMD 56.6% vs Control 54.1%). This conflicts with the SHAP claim that TGA is the only stop codon with positive SHAP toward NMD.
2. **MAJOR** — §5 attention entropy contains an internal contradiction. L1302 says "NMD has **higher** attention entropy than Controls" but L1327 (and the underlying data) shows NMD has **lower** mean entropy (0.913 vs 0.948). Mann-Whitney p=0.68 (NS); KS p=4.6e-15 (distributional difference). The "higher entropy in NMD" claim is wrong; the "similar between classes" claim at L1327 is correct.
3. **MAJOR** — §3.1 / §4.1 motif logos are sourced from `motif_logo_atg_atg500_stop500_run1.tsv` (single-run **marginal** SHAP, n_nmd=444, n_ctrl=1556) but the prose and methodology text describe "joint DeepSHAP averaged across 5 replicates, all test samples." Either generate the joint motif logos or update the methodology text.

---

## Cross-cutting concerns

### Unweighted aggregation across NMD subgroups (recurring)
Multiple sections aggregate NMD metrics by taking an **unweighted mean across the three NMD subgroups** (PTC+ n=1,798 / PTC- ATG retained n=194 / PTC- ATG lost n=276) — equally weighting subgroups that are imbalanced 9.3:1:1.4. Affected:

- **§4.1 (line 1086-1088)**: `group_by(class, stop_codon) %>% summarise(pct = mean(pct_mean))` — NMD pct uses unweighted mean
- **§4.1.1 (line 1163-1168)**: same pattern via `sg_sc_class`
- **§9.4 (junction SHAP)**: similar pattern

Step 1 fix to §4.1.1 prose ("NMD/Control ≈ 0.7x") inherited the unweighted aggregate. Properly weighted-by-size: ratio is ~1.15x (not 0.7x).

### Population mismatch in §3 / §4 motif logos
Methodology text in §0 (lines 280-290) says SHAP samples are stratified, joint, 5-replicate. The §3.1 and §4.1 figures actually load `motif_logo_atg_atg500_stop500_run1.tsv` (single replicate, marginal). The joint motif logo TSV does not exist on disk. Reader cannot tell from the rendered HTML which dataset they're looking at.

### Calibration not discussed
§9.13 Figure shows the model is systematically overconfident in middle probability bins (e.g., predicted 0.45 → observed rate 0.23; predicted 0.75 → observed 0.59). The figure renders but there's no quantitative observation. Honest reporting would note this.

---

## Section-by-section findings

### §1 Performance heatmaps (lines 412-542)
- ✅ **Verified correct.** All 12 cells of AUC and AUPRC heatmaps match `metrics_*.json`. AUC range 0.9195–0.9306, AUPRC range 0.7966–0.8387. Best AUC at atg500_stop500 (0.9306). AUPRC peak at atg500_stop1000 (0.8387). STOP=2000 AUC = 0.9277 ≈ 0.928.

### §2.1 KernelSHAP branch decomposition (lines 640-679)
- ✅ **Verified correct.** Additivity residual ≤ 1.8e-15 (machine epsilon). Sum of branch SHAP equals prediction-baseline. NMD percentages 60.7% / 28.8% / 10.5% match prose 61% / 29% / 10%.

### §2.3 Per-nucleotide SHAP zoom (lines 717-765)
- ⚠️ **MINOR.** At positions -1, 0, +1 of the ATG codon (the invariant A, T, G), all four nucleotide |SHAP| are exactly zero (DeepSHAP attribution to channel-input is zero on fixed positions). Reader may be confused about the "missing" codon in the plot. Add a one-line note.

### §3.1 Kozak motif logo (lines 821-937)
- ⚠️ **MAJOR.** Source data is single-run marginal SHAP (n_nmd=444). Prose at line 875 says "SHAP × input" referring to joint methodology in §0. Either generate joint motif logos or update.
- ✅ **Verified correct (within shown data).** G ratios at -3 / -1 / +4 are 6.7x / 10.2x / 8.0x. Step 1's "~7-8x" headline is fair (-1 is closer to 10x but bracketed).

### §4 Stop branch ratio (lines 942-984)
- ⚠️ **MINOR.** "NMD/Control ratio ≈ 1.3x" is correct only for the per-position positional sum metric. Branch-level KernelSHAP gives different ratios: |SHAP_stop| 1.73x; |SHAP_atg| 1.76x; |SHAP_struct| 2.31x. Clarify which metric.

### §4.1 Stop codon frequency chi-squared — **CRITICAL**
- 🔴 **CRITICAL.** Multiple problems:
  1. Test uses SHAP-subset reconstructed counts (`approx_n` in `sc_pop`), not actual test-set counts.
  2. NMD aggregation is unweighted across subgroups.
  3. Result shown in figure: χ² = 0.3, p = 8.6e-01 (NS).
  4. Prose at line 1152 (chunk `sec4-stop-obs`): "TGA is enriched in NMD (49.6% vs 48.4% in Control)" — but figure caption shows the test as non-significant. **Direct contradiction.**

  **Properly computed from full test set + selected_orfs (rank-0 ORFs with canonical TGA/TAA/TAG stop codon, n=389):**
  - TGA: NMD 56.6% (n=56), Control 54.1% (n=157) — barely different
  - TAA: NMD 17.2% (n=17), Control 29.7% (n=86) — strongly **depleted** in NMD
  - TAG: NMD 26.3% (n=26), Control 16.2% (n=47) — strongly **enriched** in NMD
  - χ² = 8.40, df=2, p = 0.015 (significant)

  **Notable biological tension:** The DeepSHAP analysis claims TGA-positive / TAA-negative / TAG-negative, but the actual frequency analysis shows TAG is the most-enriched stop codon in NMD. The model's "TGA preference" interpretation is not supported by the population-level frequency data. Worth investigating or framing carefully.

### §4.1.1 TGA SHAP NMD vs Control (lines 1158-1196)
- ⚠️ **MAJOR.** Step 1 fix said "NMD/Control ≈ 0.7x" — that's the unweighted aggregate. **Properly weighted by subgroup size: NMD TGA = 0.00871 vs Ctrl 0.00764 — ratio 1.15x.** The Step 1 wording reversed the direction; should be "NMD slightly larger than Control on weighted aggregate (1.15x), driven by PTC+ (1.33x); PTC- subgroups have weak TGA signal that drag down the unweighted mean."

### §5 Attention entropy (lines 1199-1330) — **MAJOR contradiction**
- 🟠 **MAJOR.** Internal contradiction:
  - L1302: "NMD isoforms have slightly lower max attention weights than Controls, consistent with the **higher attention entropy** observed above"
  - L1327: "Entropy is **similar** between classes (NMD: 0.913 bits, Control: 0.948 bits)"
  - Underlying data: mean NMD 0.913 < Control 0.948. **NMD entropy is LOWER on the mean.** The claim "higher" at L1302 is wrong. The "similar" claim at L1327 is honest given Mann-Whitney p=0.68 (NS).
- Distributional note: KS test p=4.6e-15 — there IS a distributional difference, but it's not "NMD higher". Median NMD 0.843 > Control 0.783, but mean is opposite. Heavy tails in Control (more concentrated NMD distribution).

### §6 GC enrichment (lines 1333-1496)
- ✅ **Verified correct.** Trajectory endpoints match (Step 1 update). SHAP/raw correspondence intact.
- ⚠️ **MINOR.** Panel B "raw GC content" sums G+C frequency over 4 nucleotide channels, but the per-position denominator includes zero-padding. At window edges, sums approach less than 1.0 due to padding. For the bulk of the window this is fine (sum ≈ 1.0); at edges it's a small distortion.

### §7.3 Branch decomposition by subgroup (lines 1734-1819)
- ✅ **Verified correct.** Subgroup percentages sum to 100% per subgroup. PTC+ struct/stop/atg = 62/29/9.1%. ATG-dominant in PTC- ret = 57/194 = 29.4%.

### §8 Subgroup performance (lines 1827-1936)
- ✅ **Verified correct.** Sensitivities: PTC+ 92.0%, PTC- ret 33.0%, PTC- lost 35.1%. AUC/AUPRC one-vs-rest verified. AUPRC for PTC- subgroups is very low (~7-10%) due to low prevalence (≈ 2.5–3.5% vs Control); honest interpretation note worth adding.

### §9.1 Junction motif (lines 1946-2030)
- ✅ **Verified correct.** A at -1: NMD 64.8%, Ctrl 54.0%. G at 0: NMD 79.4%, Ctrl 75.8%. Step 1 fix matches.

### §9.2 EJC dose-response (lines 2032-2099)
- ✅ **Verified correct.** Mean P(NMD) by EJC count: 0=39%, 1=61%, 2=83%, 3=89%, 4=92%, 5+=92%.

### §9.3 PolyA (lines 2102-2191)
- ✅ **Verified correct.** Prevalence NMD 84.3% / Ctrl 86.0%. Fisher OR=0.871, p=0.038. "Nearly identical" framing fair.

### §9.4 Junction SHAP (lines 2204-2236)
- ⚠️ **MAJOR.** Same unweighted-aggregation issue. PTC+ junction SHAP 0.00157 (n=1798) dominates. Unweighted NMD mean = 0.00053; weighted = 0.00125. Ratio NMD/Ctrl: unweighted 3.5x vs weighted 8.4x. Direction supports claim but magnitude differs.

### §9.5 5'UTR by subgroup (lines 2240-2349)
- ✅ **Verified correct.** Wilcoxon p ≈ 7.7e-35 for PTC- ret vs Ctrl utr5_length. Medians match.

### §9.9 Attention by subgroup (lines 2568-2686)
- ✅ Rank-0 dominance matches Step 1 update.
- ⚠️ **MINOR.** L2685 "median attention ≈ 0.59" for PTC- lost is the median of `max_attn` (max attention regardless of which rank), not median of rank-0 attention specifically (which is much lower for diffuse subgroups). PTC+ "0.87" similarly. Should clarify the metric or pick one consistent definition.

### §9.10 Entropy quartile accuracy (lines 2728-2807)
- ✅ **Verified correct.** Q1 90.8% → Q4 56.6% monotonic decline. NMD subgroups within-class ranking valid.

### §9.13 Confusion matrix (lines 2943-3050)
- ✅ **Verified correct.** TP=1816, FN=452, FP=595, TN=7268. Sensitivity 80.1%, specificity 92.4%.
- ⚠️ **MINOR.** Calibration plot rendered, no quantitative commentary. Model is overconfident in middle bins. Honest reporting would note this. Suggested addition: "The model is well-calibrated at the extremes (predicted P(NMD) <0.1 and >0.9) but somewhat overconfident in the middle (predicted 0.45 → observed 0.23; predicted 0.75 → observed 0.59), indicating boundary-region uncertainty that is not fully reflected in the predicted probabilities."

---

## Verified-correct inventory (high-confidence checks)

1. AUC/AUPRC heatmap cells (12) match metrics_*.json
2. KernelSHAP additivity residual ≈ 0
3. NMD branch percentages (struct/stop/atg)
4. Subgroup branch percentages (sum to 100% per subgroup)
5. Subgroup counts: 7,863 / 1,798 / 194 / 276 sum to 10,131
6. Confusion matrix counts and derived metrics
7. Subgroup AUC/AUPRC one-vs-rest
8. EJC dose-response P(NMD) at each level
9. 5'UTR length and coverage by subgroup
10. Junction motif position frequencies
11. Calibration deciles computation
12. Strict no-PTC count (94/2268, 4.1%)
13. Attention rank-0 dominance per subgroup
14. Entropy quartile accuracy decline
15. PolyA Fisher test
16. GC trajectory endpoints
17. ATG-dominated PTC- ret count (57/194)
18. §3.1 G dominance ratios at Kozak landmarks
19. §4.1.1 PTC+ TGA SHAP magnitude (~+0.010)

---

## Adjudication needed

1. **§4.1 chi-squared** — fix the test computation? Use full test-set counts via `predictions × selected_orfs` join. And (separately) the prose claim "TGA enriched in NMD" needs reframing — TAG is the most-enriched in NMD; TGA is barely different. The biological tension between SHAP (TGA-positive) and frequency (TAG-enriched) is interesting and may warrant a (Hypothesis) note.

2. **§5 entropy contradiction** — replace L1302 "higher attention entropy" with "lower mean entropy / similar central tendency / Control distribution heavier-tailed"?

3. **§3/§4 motif logo population mismatch** — three options:
   (a) Generate joint motif logos via deepshap.py + 09b_export_subgroup_profiles.py extension
   (b) Update prose/methodology to acknowledge run1-marginal source
   (c) Both — generate joint and use those, falling back to marginal with explicit notice

4. **§4.1.1 TGA aggregate** — re-do as size-weighted (NMD/Ctrl ≈ 1.15x), or compute directly from selected_orfs? Either fix invalidates Step 1's "0.7x" framing.

5. **§9.4 junction SHAP** — same: weighted aggregation, or skip and rely on per-subgroup analyses?

6. **§9.13 calibration commentary** — add quantitative note about middle-bin overconfidence?

7. **§5 / §9.9 max-attention vs rank-0 attention metric** — clarify in legends/prose?
