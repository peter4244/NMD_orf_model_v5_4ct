#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_motif
#SBATCH --output=results/export_motif_v5_%j.log

cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

TAG="atg500_stop500"

echo "=== DeepSHAP TSV export (run 1) ==="
python 06_export_deepshap_tsv.py --tag ${TAG} --run-id 1

echo ""
echo "=== Motif analysis (run 1) ==="
python 07_motif_analysis.py --tag ${TAG} --run-id 1

echo ""
echo "=== Done ==="
