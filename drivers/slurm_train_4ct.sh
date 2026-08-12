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
# --split IS NOW REQUIRED (2026-07-29). evaluate.py used to hardcode split="test_clean",
# which is why every metrics_*.json in existence is a test-set score -- including the twelve
# that selected the published window config. Routine monitoring scores VAL; the test set is
# a deliberate one-time act needing --final, shown commented below so it cannot happen by
# habit.
python evaluate.py --config config.yaml --atg-window 500 --stop-window 500 --split val
# FINAL evaluation, run deliberately:
# python evaluate.py --config config.yaml --atg-window 500 --stop-window 500 --split test_clean --final

echo ""
echo "=== Done ==="
