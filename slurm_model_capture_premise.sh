#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_cap_prem
#SBATCH --output=results_ism_v6/model_capture_premise_%j.log

# Does the 5' scanner beat position alone? Metadata read only -- p_capture,
# p_select and the candidate features. Does not touch vals_decay.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== code (sha256) ==="
sha256sum analysis_plans/model_capture_premise.py
echo ""
$PY analysis_plans/model_capture_premise.py --bank results_ism_v6/bank_interp_s100.h5
echo "=== exit: $? ==="
