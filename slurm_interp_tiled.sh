#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=hi_tiled
#SBATCH --output=results_ism_v6/interp_tiled_%j.log

# ROW_TILED_PERTURBATION_2026-08-02.md. Two predictions registered before this ran:
#   A  peak at a FIXED offset from the AUG          -> head reads initiation context
#   B  peak at min(100, length/2), MOVING with L    -> head reads our fill boundary
# The discriminating axis is whether the peak moves, not where it is.
#
# The self-test runs first and the job stops if it fails. It asserts, on real codes
# and with no model, that the shuffle preserves the fill mask, the junction bits,
# everything outside the tile, and (for the shuffle arm) composition within it.
# A perturbation that moved the fill mask would recreate the leak this tests for.

set -euo pipefail
cd ~/cc/nmd_orf_model_v5_4ct
source ~/.bashrc; conda activate nmd_model

echo "=== code provenance (the sha is what ran, not the branch) ==="
sha256sum analysis_plans/interp_tiled_perturbation.py
echo "=== self-test, must pass before the model is loaded ==="
python analysis_plans/interp_tiled_perturbation.py --self-test \
    --bank results_tensor_v6/nmd_tensor.h5
echo "=== the measurement ==="
python analysis_plans/interp_tiled_perturbation.py \
    --bank results_tensor_v6/nmd_tensor.h5 \
    --ckpt runs/interp_c32_b8_s100/best.pt \
    --sample 12000 --coarse 125 --fine 25 --seed 0
echo "=== exit: $? ==="
