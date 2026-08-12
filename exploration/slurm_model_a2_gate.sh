#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=md_a2_gate
#SBATCH --output=results_ism_v6/model_a2_gate_%j.log

# SEQ-A2, the second implementation. Five banks, 8 live mass bands plus a dead
# band, three regions, within-cell elevation at top 10%, within-cell exact
# permutation null, gene-clustered bootstrap.
#
# --verify-null is ON for this run. It shuffles labels for real on a subsample
# and compares against the closed-form draw the script uses. Two earlier versions
# of that check were shown incapable of failing by deliberate mutation before
# this one was kept; see the docstring.
#
# The primary is the frozen row: 8 bands, top 10%, >=100 live floor. The sweeps
# run after and are separate invocations so a failure in one does not lose the
# other.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"

echo "=== code (sha256) ==="
sha256sum analysis_plans/model_a2_gate.py
echo ""

echo "################ PRIMARY: 8 bands, top 10% ################"
$PY analysis_plans/model_a2_gate.py \
    --bands 8 --top-frac 0.10 --floor 100 --verify-null \
    --json-out results_ism_v6/model_a2_gate_primary.json
echo "=== primary exit: $? ==="

echo ""
echo "################ SWEEP: 4 bands, top 5% ################"
$PY analysis_plans/model_a2_gate.py \
    --bands 4 --top-frac 0.05 --floor 100 \
    --json-out results_ism_v6/model_a2_gate_sweep4.json
echo "=== sweep4 exit: $? ==="

echo ""
echo "################ SWEEP: 16 bands, top 20% ################"
$PY analysis_plans/model_a2_gate.py \
    --bands 16 --top-frac 0.20 --floor 100 \
    --json-out results_ism_v6/model_a2_gate_sweep16.json
echo "=== sweep16 exit: $? ==="
