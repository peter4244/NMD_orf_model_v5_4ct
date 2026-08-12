#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=md_a2_excl
#SBATCH --output=results_ism_v6/model_a2_exclusion_%j.log

# Is the >=100 live floor differential on the mechanism cell? The one unmeasured
# risk in the SEQ-A2 banding. The hypothesis row is in the script docstring and
# was committed before this ran.
#
# One pass builds the per-transcript live log-mass; the 9 (band, floor)
# combinations are then computed in memory, so the bank is read once.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"

echo "=== code (sha256) ==="
sha256sum analysis_plans/model_a2_exclusion_check.py
echo ""

$PY analysis_plans/model_a2_exclusion_check.py \
    --bank results_ism_v6/bank_interp_s100.h5
echo "=== exit: $? ==="
