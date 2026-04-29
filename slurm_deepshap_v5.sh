#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_shap
#SBATCH --output=results/deepshap_v5_run%a_%j.log
#SBATCH --array=1-5

# v5 DeepSHAP replicates for ATG=500/STOP=500

cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

SEED=$((SLURM_ARRAY_TASK_ID * 100))

echo "Run ${SLURM_ARRAY_TASK_ID}, seed=${SEED}"

python deepshap.py \
    --config config.yaml \
    --n-explain 2000 \
    --n-background 100 \
    --atg-window 500 \
    --stop-window 500 \
    --seed ${SEED} \
    --run-id ${SLURM_ARRAY_TASK_ID}
