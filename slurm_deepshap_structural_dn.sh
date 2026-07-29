#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # UNPINNED 2026-07-29 (reversing the pin earlier the same day).
#   Pinning to v100-sxm2 looked like a 34->14 node trade; it is not. sinfo -N emits one row
#   per node-PARTITION pair, so that 14 was duplicates -- and d1002/d1007/d1009/d1010 are
#   DRAINING, leaving ~2 nodes with capacity. Pinned, job 8826324 projected a start 2h LATER
#   than the oversized job it replaced. Eligibility, not gap size, is the binding constraint.
#   For SHORT work unpinning is free: the wall below covers the ~5x slow draw. Long jobs
#   (joint/atg-stop/all-modes) stay pinned, since unpinned they would need 2-6h walls.
#SBATCH --time=01:30:00   # 14:38 on the fast family, ~73m on the slow one; this
#   wall survives EITHER draw, which is what makes unpinning safe.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_dshap
#SBATCH --output=results_4ct_dn/deepshap_dn_%A_%a.log
#SBATCH --array=1-5

# DEPOSIT-NATIVE structural DeepSHAP, 5 replicates (2026-07-27).
#
# PROTOCOL COPIED EXACTLY from slurm_deepshap_structural.sh, which produced the published
# deepshap_summary_*_run1..5: seed = array task x 100, --n-explain 0 (all test samples),
# --n-background 500, windows 500/500, --branches structural. Deviating from it would make the
# comparison meaningless, so the only change is --results-dir.
#
# NOTE what these replicates vary: the SHAP BACKGROUND SAMPLING, not model training. They are
# not a training-seed sweep, and D11 stands.
#
# --results-dir results_4ct_dn reads the deposit-native checkpoint and writes beside it, so the
# published deepshap_* npz and summary TSVs stay intact for comparison. Without it deepshap.py
# hardcodes results_4ct and would overwrite exactly those.
#
# No `set -e`: a non-zero exit is a RESULT to read in the log.
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
SEED=$((SLURM_ARRAY_TASK_ID * 100))
echo "=== node $(hostname) | run ${SLURM_ARRAY_TASK_ID}, seed=${SEED}, structural, all test, 500 bg ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY deepshap.py \
    --config config_dn.yaml \
    --results-dir results_4ct_dn \
    --n-explain 0 \
    --n-background 500 \
    --atg-window 500 \
    --stop-window 500 \
    --seed ${SEED} \
    --run-id ${SLURM_ARRAY_TASK_ID} \
    --branches structural
echo "=== deepshap run ${SLURM_ARRAY_TASK_ID} exit: $? ==="
