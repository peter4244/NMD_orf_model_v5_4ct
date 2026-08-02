#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=md_a2_dead
#SBATCH --output=results_ism_v6/model_a2_deadband_%j.log

# What is the dead band made of? The SEQ-A2 control returned keto 1.081 3' of
# the stop and that number bounds how much of the live positive belongs to the
# decay head. Descriptive only.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== code (sha256) ==="
sha256sum analysis_plans/model_a2_deadband_diag.py
echo ""
$PY analysis_plans/model_a2_deadband_diag.py --bank results_ism_v6/bank_interp_s100.h5
echo "=== exit: $? ==="
