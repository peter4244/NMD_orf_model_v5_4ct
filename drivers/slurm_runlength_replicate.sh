#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_runlen
#SBATCH --output=results_ism_v6/model_runlength_%j.log

# Independent second computation of the run-length result on vals_decay, sharing no
# code with analysis_ism_regions.py.
#
# TWO THINGS, IN ORDER:
#   1. the RETRACTED fold-over-median rule on seed 100, to confirm the inversion
#      the interpretability window reported (15,304 observed vs 17,454 null) rather
#      than accept it. A retraction taken on trust is not verified.
#   2. the fixed-fraction rule, swept, which is the corrected statistic.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
echo "=== code actually running (sha256, first 16) ==="
sha256sum analysis_runlength_replicate.py | awk '{print "  " substr($1,1,16), $2}'
echo "=== git HEAD on this clone is NOT the provenance: $(git rev-parse --short HEAD) ==="
echo ""

echo "########## 1. CONFIRMING THE RETRACTION: fold rule, seed 100 ##########"
$PY analysis_runlength_replicate.py results_ism_v6/bank_interp_s100.h5 \
    --rule fold --k 10 --k-sweep 5,10,20
echo ""
echo "########## 2. CORRECTED STATISTIC: fixed fraction, seed 100 ##########"
$PY analysis_runlength_replicate.py results_ism_v6/bank_interp_s100.h5 \
    --rule fraction --k 0.01 --k-sweep 0.005,0.01,0.02,0.05
echo "=== exit: $? ==="
