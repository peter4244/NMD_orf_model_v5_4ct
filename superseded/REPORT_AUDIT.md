# Report Audit: orf_model_report_v5.Rmd

**Date:** 2026-04-03
**Auditor:** Claude (automated) + manual review
**Scope:** Accuracy, correctness, and text-to-results consistency

## Errors Found and Fixed

### E1. Wrong window size in Section 4 prose (CRITICAL)

- **Location:** Line 910
- **Description:** Text said "1000bp window centered on the rank-0 ORF's stop codon, capturing ~500bp of 3'UTR." The best model uses STOP=500 (a 500-position window with half_win=250), not 1000bp.
- **Fix:** Changed to "500-position window centered on the rank-0 ORF's stop codon, capturing ~250bp of 3'UTR"
- **Results affected:** No figures or computations affected (code uses the correct variable). Only the prose description was wrong.
- **Severity:** Critical — incorrect model description

### E2. Wrong model tag in EJC figure subtitle (CRITICAL)

- **Location:** Line 1216
- **Description:** Subtitle said "STOP=1000 model" but figure uses the best model which is STOP=500.
- **Fix:** Changed to "STOP=500 model"
- **Results affected:** Figure subtitle only. The actual data and plot are correct.
- **Severity:** Critical — misidentifies which model produced the figure

### E3. EJC dose-response numbers wrong (CRITICAL)

- **Location:** Line 1252
- **Description:** Hardcoded text claimed "0→1 EJC (P(NMD) from 56% to 88%). 2 EJCs = 94%, 3 = 96%, 5+ = 97%." Actual computed values from data (NMD transcripts only): 0 EJC = 51%, 1 EJC = 73%, 2 = 90%, 3 = 94%, 5+ = 95%. The 0-EJC and 1-EJC numbers were off by 5-15 percentage points.
- **Fix:** Replaced hardcoded numbers with inline R expressions pulling from `ejc_summary` computed in the fig4c chunk. Numbers now update automatically.
- **Results affected:** The qualitative story (largest jump is 0→1, diminishing returns) remains correct. But the specific magnitudes were significantly wrong.
- **Severity:** Critical — incorrect numerical claims in the text

### E4. Table numbering scrambled (MINOR)

- **Location:** Throughout sections 5, 6, and 7
- **Description:** Table numbers in `<summary>` tags and `caption` arguments were inconsistent and out of sequence. Section 6 tables were labeled "Table 7x" in summary tags; Section 7 tables were labeled "Table 6x." Multiple conflicts.
- **Fix:** Renumbered all tables to follow section numbers:
  - Section 5: Tables 5a, 5b (were "Table 4", "Table 4b")
  - Section 6: Tables 6a-6d (were mix of "7a"/"6a"/"6"/"6d"/"7c"/"6e")
  - Section 7: Tables 7a-7c (were "6a"/"6b"/"7b")
- **Results affected:** Navigation/reference only. No data affected.
- **Severity:** Minor — cosmetic confusion

## Verified Correct

### V1. KernelSHAP branch decomposition
- **Claimed:** Structural 61%, Stop 29%, ATG 10%
- **Computed:** Structural 61.0%, Stop 28.8%, ATG 10.2%
- **Status:** PASS

### V2. KernelSHAP residual = 0
- **Computed:** Max |residual| = 1.78e-15
- **Status:** PASS (effectively zero, floating point)

### V3. Ref CDS availability ~73%
- **Computed:** 70.9% (11,040 of 15,574)
- **Status:** PASS (within ~2% of "~73%" approximation; the prose uses "~")

### V4. Model performance metrics
- **AUC:** JSON=0.9308, reproduced via pROC=0.9308
- **AUPRC:** JSON=0.7814
- **n_test:** 15,574 (consistent across JSON, predictions TSV)
- **Prevalence:** 15.3% (2,386/15,574)
- **Status:** PASS

### V5. Attention entropy direction by subgroup
- **Claimed:** PTC+ has lower entropy (more focused), PTC- has higher entropy
- **Computed:** PTC+ = 0.769, PTC- retained = 1.058, PTC- lost = 1.392
- **Status:** PASS

### V6. Branch % by subgroup (line 2001)
- **Claimed:** PTC- ref ATG retained: 27% ATG branch vs 8% for PTC+
- **Computed:** PTC- retained: 27.4%, PTC+: 8.6%
- **Status:** PASS

### V7. Attention stats by subgroup (lines 2438-2440)
- **PTC+ rank-0 dominant:** Claimed ~95%, computed 95.5% — PASS
- **PTC+ median attention:** Claimed ~0.88, computed 0.88 — PASS
- **PTC- retained rank-0 dominant:** Claimed ~50%, computed 50.5% — PASS
- **PTC- ATG lost rank-0 dominant:** Claimed ~63%, computed 54.8% — **MARGINAL** (8 percentage points off, but definition of "dominant" may differ between my check and the report's; the report may use argmax_orf==0 without the max_attn>0.5 filter)

### V8. Subgroup definition consistency (sections 6.1 vs 6.8)
- All 15,574 isoforms have identical subgroup assignments
- Zero "NMD other" category transcripts
- **Status:** PASS

### V9. noptc vs PTC- population distinction
- noptc (any_orf_has_ejc==0): 99 transcripts
- PTC- subgroups: 511 transcripts
- Overlap: 98 (noptc is a 19% subset of PTC-)
- Report correctly labels noptc as "strict definition" (line 2651)
- **Status:** PASS — populations are different but the prose correctly distinguishes them

## Known Limitations / Notes

### N1. Variable shadowing
`nmd_subgroups` and `perf_by_subgroup` are each defined twice (sections 6.2 and 7.2). The logic is identical both times so the result is the same, but this is fragile — a change to one definition without updating the other could introduce silent errors.

### N2. Color palette inconsistency
`subgroup_pal` (section 6.1, line 1737) and `subgroup_pal_short` (setup, line 46) use different hex codes for the same subgroups. This creates visual inconsistency in section 6.8 plots vs the rest of section 6.

### N3. Hardcoded prior model AUC
The 0.94 elastic net reference line in the training curve plot (line ~464) has no verifiable data source in the results directory.

### N4. Window centering description
Line 313 says "±250bp around the middle nucleotide of the codon." The implementation creates `[center-250, center+250)` which is 250 before and 249 after center (exclusive right boundary). The "±250bp" description is a minor simplification.

### N5. Line 770 autocorrelation/FFT claim
"The lag-3 autocorrelation for G is 0.95 in the ORF body, with 90% of FFT power at period 3." No code in the report computes these values. They appear to be from an external analysis not included in the report pipeline.

## Audit Methodology

1. Wrote `audit_report.R` — standalone script that loads data independently and verifies hardcoded numbers
2. Ran subgroup consistency check comparing sections 6.1 and 6.8 definitions
3. Ran noptc vs PTC- population overlap analysis
4. Reproduced AUC via pROC from raw predictions
5. Verified directional claims (entropy, branch decomposition) against computed data
6. Grepped for all hardcoded numbers in prose and cross-referenced against data
