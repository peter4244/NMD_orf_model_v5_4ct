#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_snr
#SBATCH --output=results_ism_v6/model_branch_snr_%j.log

# Does the 5' scanner beat position alone? Metadata read only -- p_capture,
# p_select and the candidate features. Does not touch vals_decay.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
echo "=== code (sha256) ==="
sha256sum analysis_plans/model_branch_snr.py
echo ""
$PY analysis_plans/model_branch_snr.py --bank results_ism_v6/bank_interp_s100.h5
echo "=== exit: $? ==="
