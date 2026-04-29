#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_interp
#SBATCH --output=results/interpret_v5_%j.log

cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

TAG="atg500_stop500"

echo "=== Attention interpretation ==="
python 04_interpret_attention.py --results-dir results --tag $TAG

echo ""
echo "=== Structural interpretation ==="
python 05_interpret_structural.py --config config.yaml --tag $TAG --atg-window 500 --stop-window 500

echo ""
echo "=== Interpretation complete ==="
