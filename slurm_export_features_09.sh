#!/bin/bash
#SBATCH --job-name=export_09
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_4ct/export_features_09_%j.log

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

TAG="atg500_stop500"

echo "=== GC content across stop window ==="
python 09_export_gc_content.py --tag ${TAG} --run 1

echo ""
echo "=== Poly(A) annotations ==="
python 09_export_polya.py --tag ${TAG}

echo ""
echo "=== Junction ordinal DeepSHAP ==="
python 09_export_junction_ordinal.py --tag ${TAG} --run 1

echo ""
echo "=== Done ==="
