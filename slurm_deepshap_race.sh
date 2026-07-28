#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_race
#SBATCH --output=results_4ct_dn_race/deepshap_race_%A_%a.log
#SBATCH --array=4-5

# RACE COPIES of deepshap replicates 4 and 5, pinned to the SXM2 GPUs (Pete, 2026-07-27).
#
# WHY. The original array (8785576) landed tasks 1-2 on d1017 (V100-SXM2) where they finished
# in ~14:40, and tasks 3-5 on c2204/c2205 (V100-PCIE) where they are >4.7x slower and still
# running. Rather than cancel work that may be nearly done, or wait on the slow hardware, run
# duplicates on the fast GPUs and take whichever finishes first.
#
# --gres=gpu:v100-sxm2:1 is the actual GRES type name on this cluster (sinfo: d1013/d1022 carry
# gpu:v100-sxm2), so this requests the fast variant rather than hoping for it. The generic
# nodes advertise gpu:v100, which is what the first array drew.
#
# THE OUTPUT DIRECTORY IS SEPARATE, and that is the whole point: the originals are still
# running and write deepshap_structural_*_run4.npz and deepshap_summary_*_run4.tsv. Two jobs
# writing the same paths would interleave and corrupt both, and the corruption would be silent
# because each writes at completion. results_4ct_dn_race keeps them apart; the winner is copied
# into results_4ct_dn afterwards, and only if the original has not already produced it.
#
# Everything else is identical to slurm_deepshap_structural_dn.sh -- same seed rule
# (task x 100), same --n-explain 0, --n-background 500, windows 500/500, branches structural.
# A race copy that differs in any parameter is not a substitute for the run it replaces.
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
SEED=$((SLURM_ARRAY_TASK_ID * 100))
echo "=== RACE node $(hostname) | run ${SLURM_ARRAY_TASK_ID}, seed=${SEED} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
$PY deepshap.py \
    --config config_dn.yaml \
    --results-dir results_4ct_dn_race \
    --n-explain 0 \
    --n-background 500 \
    --atg-window 500 \
    --stop-window 500 \
    --seed ${SEED} \
    --run-id ${SLURM_ARRAY_TASK_ID} \
    --branches structural
echo "=== race run ${SLURM_ARRAY_TASK_ID} exit: $? ==="
