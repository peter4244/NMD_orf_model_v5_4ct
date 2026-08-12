#!/bin/bash
#SBATCH --job-name=export_sf40
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=results_4ct/export_sf40_data_%j.log

# Emit the two TSVs the paper's original SF40 was built on:
#   deepshap_atg_positional_atg500_stop500.tsv (has nmd_freq/ctrl_freq per position/channel)
#   gc_content_across_stop_window_atg500_stop500.tsv (raw rolling-GC content in stop window)
#
# 06 also emits deepshap_stop_regional_atg500_stop500.tsv which has the stop
# equivalent frequencies (in coarser 25bp bins). That plus 09's raw GC content
# gives us everything needed to rebuild SF40 with both metrics (SHAP fraction
# and raw GC content) at both windows.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

# Read the selected window configuration; do not restate it. A driver that hardcodes
# the tag keeps running the PREVIOUS selection after a re-selection, silently, because
# the old artifacts still exist. Torch-free, so it costs milliseconds.
TAG="$(python3 paths_config.py --selected-tag --config config.yaml)"
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi

echo "=== 09: raw rolling-GC content across the AUG window ==="
python 09_export_gc_content.py --tag ${TAG} --run 1 --branch atg

echo ""
echo "=== 09: raw rolling-GC content across the stop window ==="
python 09_export_gc_content.py --tag ${TAG} --run 1 --branch stop

echo ""
echo "Done."
