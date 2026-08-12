#!/bin/bash
#SBATCH --job-name=export_09d_refaug_gc
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=results_4ct/export_gc_refaug_09d_%j.log

# 09d: GC content across AUG + stop windows, NMD population restricted to
# ref-AUG-anchored isoforms only (is_ref_cds=1 on ORF0). Feeds SF40 with a
# scientifically clean cohort — no TD2-anchored NMD isoforms.
#
# Prereq: deepshap_joint_atg500_stop500_run1.npz already exists.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

# Read the selected window configuration; do not restate it. A driver that hardcodes
# the tag keeps running the PREVIOUS selection after a re-selection, silently, because
# the old artifacts still exist. Torch-free, so it costs milliseconds.
TAG="$(python3 paths_config.py --selected-tag --config config.yaml)"
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi

echo "=== 09d: GC content restricted to ref-AUG-anchored NMD isoforms ==="
python 09d_export_gc_content_refaug_only.py --tag ${TAG} --run 1
