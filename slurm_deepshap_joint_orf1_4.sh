#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_joint_orf
#SBATCH --output=results/deepshap_joint_orf%a_%j.log
#SBATCH --array=2-4

# v5 joint DeepSHAP for ORF ranks 2-4 (ranks 0-1 already completed)
# SLURM_ARRAY_TASK_ID maps directly to orf_index (2,3,4)
# All test samples, 500 background (matching ORF 1 run), single run per ORF

cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

ORF_INDEX=${SLURM_ARRAY_TASK_ID}

# Stagger array job starts to avoid simultaneous CUDA init on shared nodes
DELAY=$(( (ORF_INDEX - 2) * 30 ))
echo "Waiting ${DELAY}s before start (staggered launch)"
sleep ${DELAY}

echo "ORF rank ${ORF_INDEX}, JOINT, all test, 500 bg"

python deepshap.py \
    --config config.yaml \
    --n-explain 0 \
    --n-background 500 \
    --atg-window 500 \
    --stop-window 500 \
    --seed 100 \
    --run-id 1 \
    --branches joint \
    --orf-index ${ORF_INDEX}
