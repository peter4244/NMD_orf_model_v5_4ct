#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_interp
#SBATCH --output=results_4ct/interpret_v5_%j.log

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

# Read the selected window configuration; do not restate it. A driver that hardcodes
# the tag keeps running the PREVIOUS selection after a re-selection, silently, because
# the old artifacts still exist. Torch-free, so it costs milliseconds.
TAG="$(python3 paths_config.py --selected-tag --config config.yaml)"
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi

echo "=== Attention interpretation ==="
python 04_interpret_attention.py --results-dir results_4ct --tag $TAG

echo ""
echo "=== Structural interpretation ==="
python 05_interpret_structural.py --config config.yaml --tag $TAG --atg-window 500 --stop-window 500

echo ""
echo "=== Interpretation complete ==="
