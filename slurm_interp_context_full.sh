#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=interp_ctx_full
#SBATCH --output=analysis_plans/runlog_interp_context_%j.txt

# Initiation context: position versus content. Registered run at --sample 12000 to
# cross the filing seam before the real run; nobody has yet filed a result driven
# by a real producer's trace.
#
# mem=32G because the login-node attempt was killed at exit 137 loading a 477 MB
# tensor plus the model. That is a measurement, not a guess, and 16G is what the
# neighbouring scripts use for a smaller input.
#
# No `set -e`: a non-zero exit is a RESULT to read in the log, not something to hide.
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
export NMD_TOOLS=$HOME/cc/tools
export NMD_CLAIM_VALUES=$PWD/analysis_plans/claim_values.ctx_${SLURM_JOB_ID}.tsv
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python3
echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
$PY $NMD_TOOLS/trace_reads.py --out analysis_plans/ctx_${SLURM_JOB_ID}.trace.tsv \
    --run analysis_plans/interp_tiled_perturbation.py \
    --context --sample 12000 \
    --bank results_tensor_v6/nmd_tensor.h5 \
    --ckpt runs/interp_c32_b8_s100/best.pt \
    --predicted-at fdc847d:analysis_plans/PLAN_INITIATION_POSITION_VS_CONTENT.md
echo "=== producer exit: $? ==="
