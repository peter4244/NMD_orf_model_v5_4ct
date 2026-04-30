#!/bin/bash
#SBATCH --job-name=patch_orfs
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=results_4ct/patch_selected_orfs_%j.log

# Patch the stop_codon column in selected_orfs.tsv from the HDF5 one-hot encoding.
# Required after re-running data_prep.py to fix the off-by-one bug documented in
# BUGFIX_STOP_CODON_2026-03-31.md (4ct addendum, 2026-04-30).
# Without this patch, §4.1 of the report (chi-square test on stop-codon frequencies)
# silently uses post-stop nucleotides instead of stop codons.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

echo "=== Patching selected_orfs.tsv stop_codon column from HDF5 ==="
python scripts/patch_stop_codon.py

echo ""
echo "=== Done ==="
