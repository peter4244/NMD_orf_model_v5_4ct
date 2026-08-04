#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_margin
#SBATCH --output=results_ism_v6/model_gate_margin_paired_%j.log

# Re-scores section 1's gate-vs-ranker comparison against GENCODE's own NMD call
# instead of the annotated main ORF (Pete, 2026-08-02). Metadata read only --
# p_capture, p_select, p_decay and the candidate coordinate columns. Does not
# touch vals_decay or vals_capture.
# Local provenance: analysis_plans commit 9dad906.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== code (sha256) ==="
sha256sum analysis_plans/model_gate_margin_paired.py
echo ""
$PY analysis_plans/model_gate_margin_paired.py
echo "=== exit: $? ==="
