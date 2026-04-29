#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_joint_orf
#SBATCH --output=results/deepshap_joint_orf%a_%j.log
#SBATCH --array=2-4

# v5 joint DeepSHAP for ORF ranks 2-4 (ranks 0-1 already completed on Explorer)
# Channing cluster version

cd /proj/regeps/regep00/studies/ExternalCellLines/analyses/repjc/Randell_Lung_Cells_2025/results/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

ORF_INDEX=${SLURM_ARRAY_TASK_ID}

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
