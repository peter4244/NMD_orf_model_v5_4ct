# Bug Fix: Stop Codon Position Off-by-One in ORFik Scan and Subgroup Export

**Date:** 2026-03-31
**Affected files:**
1. `/projects/talisman/shared-data/nmd/isoform_transitions/Version_6.0/isopair_wrapper/05s_orfik_scan.R`
2. `/home/p.castaldi/cc/nmd_orf_model_v5/08_export_subgroup_deepshap_tsv.py`

---

## Root Cause

The ORFik `findORFs()` function returns ORF coordinates where `end(orf)` is the **last nucleotide of the stop codon** (i.e., the stop codon IS included in the ORF range). A comment in `05s_orfik_scan.R` incorrectly stated "ORFik excludes stop from range," leading to two downstream bugs.

### Evidence

The v5 model's stop branch window is centered using `orf_end - 1 - 1` (0-based) in `data_prep.py` line 593. Direct inspection of the DeepSHAP NPZ inputs confirms the window center is the **middle nucleotide of the stop codon**:

```
offset -1: T = 100%           (1st nt of stop — always T)
offset  0: A = 49%, G = 51%   (2nd nt — variable, A or G only)
offset +1: A = 79%, G = 21%   (3rd nt — variable, A or G only)
offset +2: all 4 nucleotides   (post-stop sequence)
```

This is consistent with `orf_end` being the last nt of the stop codon, and `orf_end - 2` (0-based) being its middle nucleotide.

---

## Bug 1: `stop_codon` column in `orf_features.tsv` (05s_orfik_scan.R)

### Problem

Line 158-164 extracted the stop codon as:
```r
stop_pos <- oe + 1L
stop_codon[j] <- as.character(subseq(tx_seq, stop_pos, stop_pos + 2L))
```

This reads the **3 nucleotides after the stop codon**, not the stop codon itself. Result: only 3.5% of rank-0 ORFs showed canonical stop codons (TAA/TAG/TGA) — the rest showed random triplets (AAA, AGA, GCC, etc.).

### Fix

```r
# Stop codon: last 3 nt of the ORF (ORFik includes the stop codon in the range)
if (oe >= 3L && oe <= tx_len) {
  stop_codon[j] <- as.character(subseq(tx_seq, oe - 2L, oe))
} else {
  stop_codon[j] <- NA_character_
}
```

### Impact on v5 model

**None.** The `stop_codon` column is metadata only. No Python script in the v5 pipeline reads it. Model training, predictions, DeepSHAP, and all interpretation scripts derive stop codon identity directly from the one-hot sequence encoding in the HDF5/NPZ files.

### Regeneration required

To produce a corrected `orf_features.tsv` and `selected_orfs.tsv`:
1. Re-run `05s_orfik_scan.R` to regenerate `orfik_scan.rds`
2. Re-run `export_rds.R` to regenerate `orf_features.tsv`
3. Re-run `data_prep.py` to regenerate `selected_orfs.tsv`

No model retraining or DeepSHAP regeneration needed.

---

## Bug 1b: `n_downstream_ejc` threshold off by 3nt (05s_orfik_scan.R)

### Problem

Line 186-189 counted downstream EJCs using:
```r
stop_end <- oe + 3L
n_downstream_ejc[j] <- sum(junctions > stop_end)
```

Since the stop codon ends at `oe` (not `oe + 3`), this requires junctions to be >3nt past the actual stop end.

### Fix

```r
# ORFik ORF end = last nt of ORF (including stop codon). Stop ends at oe.
n_downstream_ejc[j] <- sum(junctions > oe)
```

### Impact on v5 model

**Negligible.** The EJC model's >50nt rule means junctions must be >50nt downstream of the stop to count. Being off by 3nt (checking >oe+3 instead of >oe) would only affect junctions in the narrow 0-3nt window after the stop — which is biologically implausible for real exon-exon junctions. For all practical purposes, `n_downstream_ejc` values are unchanged.

---

## Bug 2: `compute_stop_codon_shap()` position error (08_export_subgroup_deepshap_tsv.py)

### Problem

Line 159-162 (before fix):
```python
# Stop codon: pos stop_pos = T (constant), stop_pos+1 = A or G, stop_pos+2 = A or G
pos1 = stop_pos + 1
pos2 = stop_pos + 2
```

The function assumed `stop_pos` (window center) was the T (1st nucleotide of the stop codon). In reality, the window is centered on the **2nd nucleotide** (middle) of the stop codon. So `pos2 = stop_pos + 2` read the first nucleotide AFTER the stop codon — a random nucleotide that could be any of A/C/G/T.

Result: only ~54% of samples were identified as TAA/TAG/TGA (the rest classified as "other"), and the frequency/SHAP statistics in `subgroup_stop_codon_shap_atg500_stop500.tsv` were unreliable.

### Fix

```python
# Stop codon layout: stop_pos-1 = T (constant), stop_pos = A or G, stop_pos+1 = A or G
# (window is centered on the middle nucleotide of the stop codon)
pos1 = stop_pos
pos2 = stop_pos + 1
```

### Impact on v5 model

**None on model itself.** The model sees correct sequence windows — only the post-hoc interpretation export had the wrong offset. Model training, predictions, attention weights, structural importance, and all DeepSHAP channel-level attributions are unaffected.

**Report impact:** The stop codon frequency and SHAP figures in Section 4.1 / 4.1.1 of `orf_model_report_v5.Rmd` were based on the buggy export. Regenerated via SLURM job 5528000.

### Regeneration

```bash
cd /home/p.castaldi/cc/nmd_orf_model_v5
sbatch slurm_export_subgroup_v5.sh   # submitted as job 5528000
```

This regenerates all 6 subgroup TSVs including the corrected `subgroup_stop_codon_shap_atg500_stop500.tsv`.

---

## Summary

| Bug | File | Impact on model | Impact on report | Regeneration |
|-----|------|-----------------|------------------|-------------|
| Wrong `stop_codon` column | 05s_orfik_scan.R | None | None (column unused) | Deferred |
| EJC threshold off by 3nt | 05s_orfik_scan.R | Negligible | Negligible | Deferred |
| Stop codon SHAP positions | 08_export_subgroup_deepshap_tsv.py | None | Yes — wrong freqs/SHAP | Job 5528000 running |

---

## 2026-04-30 4ct addendum

The 5-step verification of `orf_model_report_v5.Rmd` (Step 2, result correctness)
discovered that 4ct's `selected_orfs.tsv` was carrying over the buggy `stop_codon`
column despite the upstream R fix being applied in March. The Rmd's §4.1 chi-squared
test and stop-codon frequency claims were affected: the figure caption read p=0.86 (NS)
while the prose claimed "TGA enriched in NMD" — a direct contradiction created by the
test being computed from SHAP-subset reconstructed counts of post-stop nucleotides.

**Resolution applied 2026-04-30:**

1. Wrote `scripts/patch_stop_codon.py`, which re-derives `stop_codon` from the bug-free
   HDF5 one-hot encoding (positions [-1, 0, +1] of each ORF's stop window) and patches
   `selected_orfs.tsv` in place. It also filters the file to the 4ct isoform set
   (the file was carrying 6ct legacy rows). Post-patch: 198,816 rows, 39,938 unique
   isoforms, 100% canonical (TGA/TAA/TAG) stop rate.

2. Rewrote the §4.1 R chunk to compute the chi-squared from the proper
   `predictions × selected_orfs` join across the full test set. With clean data:
   χ² = 46.6, p ≈ 7.5e-11 (highly significant). TGA is enriched in NMD (55.9% vs
   48.1% in Control); TAA depleted (23.9% vs 29.9%); TAG slightly depleted (20.2%
   vs 22.0%). The model's TGA-positive/TAA-negative/TAG-negative SHAP signs are now
   directionally consistent with the population frequencies.

3. Wrote `scripts/export_joint_motif_logos.py` to pool the 5 joint DeepSHAP NPZs into
   `motif_logo_atg_joint_atg500_stop500.tsv` and `motif_logo_stop_joint_atg500_stop500.tsv`
   (covering the full test set: n_nmd=2,268, n_ctrl=7,863). The Rmd's existing fallback
   logic (§0, line ~207) now picks these up in preference to the marginal run-1 TSVs,
   so the §3.1 Kozak and §4.1 stop motif logos cover all test transcripts.

No model retraining or DeepSHAP regeneration needed. The patcher is reproducible from
the HDF5 (which was always bug-free); if the upstream R pipeline is ever re-run from
the 2026-03-31-fixed `05s_orfik_scan.R`, the patcher becomes redundant.
