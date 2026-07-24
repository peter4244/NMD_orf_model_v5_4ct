# Step 3 — Documentation Accuracy Findings

**Report:** `orf_model_report_v5.Rmd` at commit `b88d513`
**Reviewer:** Plan agent + verification
**Date:** 2026-04-30

## Summary

Step-1/2 substantive rewrites have propagated correctly. Numeric values from inline R hooks (`r ...`) are reliable. Figures generated from patched data (stop codon, branch decomposition, sensitivity, calibration ECE) match prose. Remaining inconsistencies concentrate in (a) static hand-written prose that disagrees with the values it cites, and (b) by-subgroup motif logos with per-panel y-axes that frustrate the prose's magnitude comparison.

## Top 3 issues

### 1. CRITICAL — §7.2 wrong directional verb (Rmd L1763)

Prose: **"is_ref_cds importance is 0.104 for PTC+ ... but drops to 0.135 for ref-ATG-lost"** — 0.135 is **higher** than 0.104, so "drops" is the wrong direction. Figure `fig6c_attribution_heatmap.png` confirms PTC+ is_ref_cds=0.104 < ATG-lost is_ref_cds=0.135. Inline R values are correct; only the static word "drops" is wrong.

### 2. CRITICAL — §9.13 stale hardcoded calibration percentages (Rmd L3052)

My Step-2 calibration bullet said "Most test transcripts (60%) fall in the well-calibrated extremes, so this overconfidence affects ~25% of the test set in the boundary region." Actual decile distribution from `predictions_atg500_stop500.tsv`:
- **Extremes (P<0.1 ∪ P>0.9): 5077/10131 = 50.1% (not 60%)**
- **Middle (P=0.4–0.8): 939/10131 = 9.3% (not 25%)**

Both percentages are wrong. The figure data renders correctly; only the prose narrative is stale.

### 3. MAJOR — §9.7/§9.8 by-subgroup motif logos with per-panel y-axes

Figures `fig6_atg_subgroup.png` and `fig6_stop_subgroup.png` use independent y-axes per subgroup panel:
- `fig6_atg_subgroup.png`: PTC+ ±0.0025; PTC- ret ±0.001; PTC- lost ±0.0005; Control ±0.001
- `fig6_stop_subgroup.png`: PTC+ ±0.008; PTC- ret ±0.0025; PTC- lost ±0.002; Control ±0.004

Prose at L2543 says "Kozak signal at -3 is **strongest in PTC+**" and at L2601 "PTC+ shows the strongest stop codon SHAP signal" — these visual claims are only true if the reader notices the y-axis differs. A reader skimming the figure may infer the panels show comparable magnitudes. §3.1 and §4.1 master logos already use shared `kz_ylim` / `sc_ylim` approach — apply same here.

## Other findings

| # | Section | Severity | Issue |
|---|---------|----------|-------|
| 4 | §4.1.1 (L1227) | MINOR | "Weighted aggregate ratio 1.15x" claim is not derivable from `fig4a_stop_codon_subgroup.png` or Table 4a (which only show per-subgroup values). Either add a note in figure subtitle or leave the claim to prose only with a reference to the calculation. |
| 5 | §5 (L1360) | MINOR | Bullet says "Entropy is similar between classes (NMD: 0.913, Control: 0.948 bits)" using mean values without labeling them. The preceding paragraph (L1335) discusses median direction (NMD higher). A reader scanning bullets only will infer wrong direction. Add "(mean)" label to L1360. |
| 6 | §2.2 (L693) | MINOR | "Within the structural branch (61%)" rounds 60.7% to 61% within one sentence of where 60.7% was used. Use 60.7% or 61% consistently. |
| 7 | §3.1 (L935) | MINOR | "G at -3 carries the largest single-channel positive signal" but G at +4 is "similarly dominant" with same value (0.003 in both). Wording implies a ranking that doesn't exist. Could say "G at -3 and G at +4 carry comparable, dominant positive signals". |
| 8 | Cross-cutting | MINOR | Subgroup palette inconsistency: §4.1.1, §9.4, §9.7, §9.8 use `subgroup_pal_short` (pastel oranges); §7, §8, §9.5, §9.10 use `subgroup_pal` (saturated). Same subgroup label appears in two different colors. Pick one palette. |
| 9 | §9.4 (L2272) | MINOR | Prose silent on Control's non-zero junction SHAP. Figure shows Control bar ~0.0002 (positive, CI excluding 0). Reader sees Control bar but prose doesn't mention it. |
| 10 | File naming | MINOR | Several figures still saved with `fig4*` / `fig6*` legacy prefixes that don't match current section structure. Cosmetic. |
| 11 | §8 (L1966) | MINOR | "92%" prints without trailing decimal because `round(92.046, 1)=92` strips it. `sprintf("%.1f%%", ...)` would force "92.0%" for consistency with PTC- subgroups (33.0%, 35.1%). |

## Verified correct (high-confidence)
- §0 data status panel + test population n
- §1 AUC/AUPRC heatmap values vs prose
- §2.1 branch percentages (60.7/28.8/10.5)
- §3.1 Kozak inline values match figure
- §4.1 stop codon frequency figure (Chi-sq 46.6, p=7.5e-11, per-codon Fisher p)
- §4.1.1 PTC+ TGA largest in figure (matches "PTC+ carries the largest TGA")
- §5 mean/median entropy and median max attention values
- §6 GC trajectory endpoints
- §7.3 branch decomposition by subgroup
- §8 sensitivity matches figure
- §9.5 5'UTR by-subgroup
- §9.9 rank-0 dominance and median max attention
- §9.10 entropy quartile accuracy
- §9.13 confusion matrix, ECE, Brier, no-PTC count

## Adjudication needed

1. **§7.2 "drops" vs "rises"** — fix directional verb to match data
2. **§9.13 60%/25% percentages** — recompute from cal_data and use inline R values
3. **§9.7 / §9.8 motif logos with per-panel y-axes** — apply shared y-limit (same approach as §3.1 / §4.1)
4. **§4.1.1 weighted aggregate visibility** — add to figure subtitle, or leave prose-only?
5. **§5 L1360 mean labeling** — add "(mean)" qualifier?
6. **Subgroup palette consistency** — pick one palette (subgroup_pal vs subgroup_pal_short)?
7. **Other minor wording fixes (#6, #7, #9, #11)** — apply or skip?
