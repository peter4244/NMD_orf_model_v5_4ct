#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_kshap
#SBATCH --output=results_4ct/kernel_shap_%j.log

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

echo "=== KernelSHAP Branch Decomposition ==="
python 11_kernel_shap_branches.py --tag atg500_stop500 --n-background 500 --seed 42 --explain-split all

echo "=== Done ==="
