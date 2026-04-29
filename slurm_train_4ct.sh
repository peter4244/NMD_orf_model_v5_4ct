#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=4ct_train
#SBATCH --output=results_4ct/train_4ct_%j.log

# 4-cell-type retrain: ATG=500, STOP=500 only
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

echo "=== 4ct retrain: ATG=500 STOP=500 ==="

python 03_train.py --config config.yaml --atg-window 500 --stop-window 500

echo ""
echo "=== Evaluation ==="
python evaluate.py --config config.yaml --atg-window 500 --stop-window 500

echo ""
echo "=== Done ==="
