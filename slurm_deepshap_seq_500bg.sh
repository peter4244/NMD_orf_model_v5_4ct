#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_seq500
#SBATCH --output=results/deepshap_seq500bg_run%a_%j.log
#SBATCH --array=1-5

# v5 sequence DeepSHAP: 2000 samples, 500 background, ATG+stop branches
# ~5x longer than 100 background due to larger reference set

cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

SEED=$((SLURM_ARRAY_TASK_ID * 100))

echo "Run ${SLURM_ARRAY_TASK_ID}, seed=${SEED}, ATG+stop, ALL test samples, 500 bg"

python deepshap.py \
    --config config.yaml \
    --n-explain 0 \
    --n-background 500 \
    --atg-window 500 \
    --stop-window 500 \
    --seed ${SEED} \
    --run-id ${SLURM_ARRAY_TASK_ID} \
    --branches atg stop
