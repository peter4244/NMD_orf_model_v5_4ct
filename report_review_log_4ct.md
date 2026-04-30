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

**Status:** pending

| # | Section | Result | Issue | Fix |
|---|---------|--------|-------|-----|

---

## Step 3: Documentation accuracy

**Status:** pending

| # | Section | Prose claim | Table/figure shows | Fix |
|---|---------|-------------|--------------------|-----|

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
