#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=md_a2_gate2
#SBATCH --output=results_ism_v6/model_a2_gate2_%j.log

# SEQ-A2 rerun with three defects fixed:
#   1. unresponsive positions SUB-STRATIFIED by order of magnitude, so the
#      control is stratified like everything else. Pooled, it spanned six orders
#      and its own mass-composition gradient read as a signal.
#   2. tie fraction at the cut reported per cell, and cells that cannot produce
#      a number are COUNTED rather than dropped silently. 13.8% vanished
#      unreported in job 8896644.
#   3. census reports POSITION-level retention, which is what the pre-registered
#      freeze condition is written on. It previously reported a cell fraction
#      under the same word.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
echo "=== code (sha256) ==="
sha256sum analysis_plans/model_a2_gate.py
echo ""
$PY analysis_plans/model_a2_gate.py \
    --bands 8 --top-frac 0.10 --floor 100 --verify-null \
    --json-out results_ism_v6/model_a2_gate2_primary.json
echo "=== exit: $? ==="
