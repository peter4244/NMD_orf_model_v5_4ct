#!/bin/bash
#SBATCH --job-name=export_09c_gc
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=results_4ct/export_signed_gc_09c_%j.log

# 09c: emit signed SHAP × input for the rolling_gc channel from the 5 joint
# DeepSHAP NPZs. Feeds SF40 in the manuscript supplement. Prereq: same
# NPZs already required by 09b (which you've already run today).

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

# Read the selected window configuration; do not restate it. A driver that hardcodes
# the tag keeps running the PREVIOUS selection after a re-selection, silently, because
# the old artifacts still exist. Torch-free, so it costs milliseconds.
TAG="$(python3 paths_config.py --selected-tag --config config.yaml)"
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi

echo "=== 09c: signed SHAP × input for rolling_gc channel ==="
python 09c_export_signed_gc_channel.py --tag ${TAG} --n-runs 5
