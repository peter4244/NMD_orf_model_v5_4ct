#!/bin/bash
#SBATCH --job-name=export_09
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_4ct/export_features_09_%j.log

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
eval "$(conda shell.bash hook)"
conda activate nmd_model

# Read the selected window configuration; do not restate it. A driver that hardcodes
# the tag keeps running the PREVIOUS selection after a re-selection, silently, because
# the old artifacts still exist. Torch-free, so it costs milliseconds.
TAG="$(python3 paths_config.py --selected-tag --config config.yaml)"
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi

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
