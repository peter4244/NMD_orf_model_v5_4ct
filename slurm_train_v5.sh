#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v5_train
#SBATCH --output=results/train_v5_%a_%j.log
#SBATCH --array=1-3

# v5 test runs: 3 window combinations
cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

ATG_SIZES=(100 500 1000)
STOP_SIZES=(1000 1000 2000)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
ATG=${ATG_SIZES[$IDX]}
STOP=${STOP_SIZES[$IDX]}

echo "=== v5 Task ${SLURM_ARRAY_TASK_ID}: ATG=${ATG} STOP=${STOP} ==="

python 03_train.py --config config.yaml --atg-window ${ATG} --stop-window ${STOP}

echo ""
echo "=== Evaluation ATG=${ATG} STOP=${STOP} ==="
python evaluate.py --config config.yaml --atg-window ${ATG} --stop-window ${STOP}

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
