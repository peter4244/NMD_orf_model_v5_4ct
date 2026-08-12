#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_exp
#SBATCH --output=results_4ct/export_importance_v5_%j.log

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
eval "$(conda shell.bash hook)"
conda activate nmd_model

python 05_export_sample_importance.py \
    --config config.yaml \
    --tag atg500_stop500 \
    --atg-window 500 \
    --stop-window 500

echo ""
echo "=== Exporting per-sample importance to TSV ==="
python 05b_export_sample_importance_tsv.py --tag atg500_stop500
