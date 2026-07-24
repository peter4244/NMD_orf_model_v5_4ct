# 4ct Report Verification Log

**Report:** `orf_model_report_v5.Rmd` at commit `b890e77`
**Protocol:** 5-step scientific-research report verification (global/CLAUDE.md)
**Started:** 2026-04-30

Each step is a focused, dimension-specific pass over the entire report. Steps run sequentially because fixes in earlier steps may invalidate later findings.

---

## Step 1: Factual accuracy

**Status:** Complete (2026-04-30). 13 fixes applied to `orf_model_report_v5.Rmd`. Rmd parses cleanly. Full findings in `step1_findings.md`.

**Headline:** ~75 claims checked, ~58 correct, 11 incorrect, ~6 unverifiable in Round 1. All identified incorrect claims fixed. No new issues introduced (parse + semantic spot-check passed).

| # | Section | Issue | Fix applied |
|---|---------|-------|-------------|
| 1 | §0 (L76) | Code comment "selected by AUPRC" but atg500_stop500 wins by AUC | Changed comment to "selected by AUC" |
| 2 | §0 (L350) | "Reference CDS available for ~73%" — actually 70% | Updated to "~70%" |
| 3 | §1 (L540-541) | "STOP=500 optimal by AUPRC", "AUPRC peaks at STOP=500 across all ATG sizes" — both false | Rewrote to AUC framing; noted STOP=1000 wins AUPRC at ATG=500 (0.839 vs 0.833); diff 0.004 |
| 4 | §2 (L764) | Period-3 G stats "0.95 autocorr, 90% FFT" — actual 0.86 NMD / 0.92 Ctrl, ~77% narrow band | Updated to recomputed values (0.86 NMD, 0.92 Ctrl) |
| 5 | §3 chunk | New inline R bindings needed (`kz_G_m3_*`, `kz_G_m1_*`) | Added to chunk `sec3-kozak-stats` |
| 6 | §3.1 (L928-933) | 4 sign/magnitude errors at -3 and +4 Kozak positions; conclusion "model is not recognizing Kozak context" was based on wrong data | **Rewrote interpretation: model DOES recognize Kozak (G dominates at -3, -1, +4 in both classes; NMD/Control magnitude ratio ~7-8x).** Reverses prior conclusion. |
| 7 | §4 (L978) | "NMD ~2x higher \|SHAP\|" applied to stop branch — actual 1.3x; 2x figure fits ATG branch | Rewrote: stop ratio 1.3x; ATG ratio ~2x; stop has highest absolute attribution |
| 8 | §4.1.1 (L1189) | "TGA NMD ~2x larger than Control" — actual aggregate 0.72x; PTC+ specifically is ~1.3x | Rewrote: TGA positive in both, PTC+ specifically has largest signed SHAP (+0.010 vs Ctrl +0.008) |
| 9 | §6 (L1489) | GC trajectory "controls 48%→36%, NMD 50%→45%" stale | Updated to "controls ~50%→~40%, NMD ~50%→~47%" |
| 10 | §9.1 (L2025) | Junction motif "65%, 82%" hardcoded | Updated to "65% NMD/54% Ctrl at -1; 79% NMD/76% Ctrl at 0" |
| 11 | §9.7 (L2440) | Stale "27% / 8%" hardcoded | Converted to inline R: `` `r ptcneg_ret$pct_atg` `` and `` `r ptcpos$pct_atg` `` |
| 12 | §9.9 (L2678-2680) | Attention "PTC+ 95%/0.88, PTC- ret ~50%, PTC- lost ~63%" — actual 93%/0.87, 51%, 50%/0.59 | Updated to recomputed values (PTC- lost 50% not 63% — narrative now treats PTC- subgroups as similarly diffuse) |
| 13 | §9.12 (L2924) | Stale "8% / 27% / 25%" hardcoded | Converted to inline R: `pct_atg` and `nrow(atg_dom_ret)/ptcneg_ret$n` |

**Notable interpretive change:** §3.1 conclusion reversed from "model is not simply recognizing strong Kozak context" to "model recognizes Kozak-consensus residues, with substantially higher attribution in NMD" — flagged because it's a meaningful narrative shift driven by the data correction.

**Deferred (low-impact, code-comment only):** L264 inline says "~448 NMD" vs actual 444 (acceptable with "~"); L1266 stale code comment "10 isoforms with all-zero weights excluded" (no all-zero rows actually exist; not user-facing).

---

## Step 2: Result correctness

**Status:** Complete (2026-04-30). All identified issues fixed. Findings in `step2_findings.md`.

**Headline:** Population-level analyses correct (predictions, calibration, confusion matrix, EJC dose-response, 5'UTR features, KernelSHAP additivity, subgroup branch decomposition all verified). Bigger-than-expected discoveries during the pass: §4.1 chi-squared p-value flipped (NS → p=7.5e-11) once the stop-codon bug was identified; §3.1/§4.1 motif logos switched from run-1 marginal SHAP (n=444 NMD) to joint 5-run pooled SHAP across the full test set (n=2,268 NMD).

**Most consequential discovery — stop-codon column bug propagated to 4ct.** `selected_orfs.tsv` carried the buggy column documented in BUGFIX_STOP_CODON_2026-03-31.md (reading post-stop nucleotides instead of stop codon). When my Step 2 review used this column to compute marginal stop-codon frequencies, it produced a misleading "TAG enriched in NMD" finding (an artifact of post-stop sequence). Resolved by patching `selected_orfs.tsv` from HDF5 one-hot encoding (`scripts/patch_stop_codon.py`); now 100% canonical stops. With clean data, the model's SHAP and the population frequencies are directionally consistent (TGA enriched in NMD; TAA depleted; TAG slightly depleted).

**Fixes applied:**

| # | Issue | Fix |
|---|-------|-----|
| 1 | §5 attention entropy: L1302 said NMD has "higher" entropy (mean is lower) | Rewrote with median direction, mean direction, KS distributional difference, Mann-Whitney NS |
| 2 | §4.1 chi-squared from SHAP-subset reconstructed counts (p=0.86) | Rewrote chunk to compute from `predictions × selected_orfs` (full test set; χ²=46.6, p=7.5e-11) |
| 3 | §4.1 prose: "TGA enriched in NMD" had been numerically contradicted | Updated with correct direction matching figure (still TGA-enriched, but now consistent across data and SHAP) |
| 4 | §4.1.1 prose: Step 1's "0.7x" was unweighted aggregate | Updated to weighted aggregate (1.15x) + PTC+ specific (1.3x) |
| 5 | §4.1.1 figure: unweighted aggregation hid PTC+ dominance | Changed to per-subgroup view (all 4 subgroups as separate bars) |
| 6 | §9.4 junction SHAP: same unweighted aggregation issue | Same per-subgroup figure restructure |
| 7 | §3.1/§4.1 motif logos: used run-1 marginal SHAP (n=444 NMD) instead of joint | Wrote `scripts/export_joint_motif_logos.py`; joint TSVs now exist (n=2,268 NMD), Rmd auto-falls-through |
| 8 | §3.1 ratios re-derived: joint NMD/Ctrl is ~4x not ~7-8x | Updated prose to "approximately 4x"; rewrote -1 bullet to reflect joint values |
| 9 | §9.9: median attention metric was ambiguous | Standardized to "median max-attention" with values for all 3 subgroups |
| 10 | §9.13 calibration plot rendered without commentary | Added (Data) bullet with ECE+Brier and overconfidence note |
| 11 | `selected_orfs.tsv`: 6ct legacy rows + buggy stop_codon | Patched + filtered to 4ct (198,816 rows, 100% canonical stops) |
| 12 | Bug fix doc | Added 2026-04-30 4ct addendum to BUGFIX_STOP_CODON_2026-03-31.md |

**Verified-correct (high-confidence, no fix needed):** AUC/AUPRC heatmap cells; KernelSHAP additivity; NMD branch percentages; subgroup branch percentages sum to 100%; subgroup counts sum to n_test; confusion matrix; subgroup AUC/AUPRC one-vs-rest; EJC dose-response; 5'UTR length/coverage by subgroup; junction motif position frequencies; calibration deciles; strict no-PTC count; attention rank-0 dominance per subgroup; entropy quartile accuracy; PolyA Fisher test; GC trajectory endpoints; ATG-dominated PTC- ret count.

**New scripts checked into repo:**
- `scripts/patch_stop_codon.py` (fixes selected_orfs.tsv from HDF5; reproducible)
- `scripts/export_joint_motif_logos.py` (pools 5 joint DeepSHAP NPZs to motif logo TSVs)

---

## Step 3: Documentation accuracy

**Status:** Complete (2026-04-30). 11 fixes applied. Rendered cleanly (job 6434900, 1m47s, 171 chunks).

**Headline:** Step-1/2 substantive rewrites had propagated correctly. Step 3 caught hand-written prose that disagreed with values it cited, plus per-panel y-axis issues that masked subgroup magnitude differences.

| # | Section | Issue | Fix applied |
|---|---------|-------|-------------|
| 1 | §7.2 (L1763) | "is_ref_cds importance is 0.104 ... but **drops** to 0.135" — wrong direction | Rewrote: "is somewhat larger (0.135) for ref-ATG-lost — where the feature is always 0, and the model has learned to interpret that absence itself as evidence for NMD" |
| 2 | §9.13 (L3052) | Hardcoded "60% in extremes, ~25% in middle" — actually 50.1% / 9.3% | Replaced with inline R `pct_extreme` and `pct_middle` computed from preds |
| 3 | §9.7 / §9.8 | Per-panel y-axes (PTC+ ±0.0025, PTC- ret ±0.001 etc) masked the magnitude prose | Two-pass build: compute shared `atg_sg_ylim` / `stop_sg_ylim` across all 4 subgroup matrices, apply via `coord_cartesian` |
| 4 | §4.1.1 (L1227) | Weighted-aggregate "1.15x" claim not derivable from figure | Dropped sentence; rely on per-subgroup figure + PTC+ specific (1.3x) |
| 5 | §5 (L1360) | Bullet quoted mean entropy (NMD lower) without labeling, contradicting paragraph above (median, NMD higher) | Changed to "Mean entropy is similar between classes... see paragraph above for the median direction" |
| 6 | Cross-cutting | Two subgroup palettes (`subgroup_pal_short` pastel; `subgroup_pal` saturated) used in different sections | Unified on `subgroup_pal` (saturated). Defined globally at §0; removed late definition; replaced all `subgroup_pal_short` references |
| 7 | §2.2 (L693) | "Within the structural branch (61%)" rounds 60.7% to 61% within one sentence | Use 60.7% with reference to §2.1 |
| 8 | §3.1 (L935) | "G at -3 carries the largest" but G at +4 has same value (tie) | Rewrote: "G at positions -3 and +4 carries comparable, dominant positive signals" |
| 9 | §9.4 (L2272) | Prose silent on Control's non-zero junction SHAP | Added: "Control transcripts show a small positive signal (~10× smaller than PTC+); the PTC- subgroups are near-zero" |
| 10 | §8 (L1966), §9.12 (L2970) | "92%" rendered without trailing decimal because `round(92.04, 1)=92` | Replaced with `sprintf("%.1f", ...)` to force "92.0%", "33.0%", "35.1%" |

**Verified-correct list (high-confidence, no fix needed):** §0 data status panel + n_test; §1 AUC/AUPRC heatmap values vs prose; §2.1 branch percentages (60.7/28.8/10.5); §3.1 Kozak inline values match figure; §4.1 stop codon frequency figure (Chi-sq 46.6, p=7.5e-11, per-codon Fisher); §4.1.1 PTC+ TGA largest in figure; §5 mean/median/max-attn values; §6 GC trajectory; §7.3 branch decomposition by subgroup; §8 sensitivity; §9.5 5'UTR by subgroup; §9.9 attention rank-0 dominance; §9.10 entropy quartile accuracy; §9.13 confusion matrix, ECE, Brier, no-PTC.

---

## Step 4: Reproducibility & completeness

**Status:** pending

| # | Section | Missing reference | Fix |
|---|---------|-------------------|-----|

---

## Step 5: METHODS.md verification

**Status:** pending

| # | METHODS.md section | Code location | Discrepancy | Fix |
|---|--------------------|---------------|-------------|-----|
