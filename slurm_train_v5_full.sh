#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v5_full
#SBATCH --output=results/train_v5_full_%a_%j.log
#SBATCH --array=1-13

# v5 full grid: 4 ATG × 4 STOP = 16 models, minus the 3 already submitted
# Already running: ATG=100/STOP=1000, ATG=500/STOP=1000, ATG=1000/STOP=2000

cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

ATG_SIZES=(50  50   50   50   100 100  100  500 500  500  500  1000 1000)
STOP_SIZES=(100 500 1000 2000 100 500 2000 100 500 2000 100  500  1000)
#            1   2    3    4   5   6   7    8   9   10   11   12   13

IDX=$((SLURM_ARRAY_TASK_ID - 1))
ATG=${ATG_SIZES[$IDX]}
STOP=${STOP_SIZES[$IDX]}

echo "=== v5 Full Grid Task ${SLURM_ARRAY_TASK_ID}: ATG=${ATG} STOP=${STOP} ==="

python 03_train.py --config config.yaml --atg-window ${ATG} --stop-window ${STOP}

echo ""
echo "=== Evaluation ATG=${ATG} STOP=${STOP} ==="
python evaluate.py --config config.yaml --atg-window ${ATG} --stop-window ${STOP}

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
