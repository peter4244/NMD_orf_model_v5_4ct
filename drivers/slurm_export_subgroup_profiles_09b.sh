#!/bin/bash
#SBATCH --job-name=export_09b
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_4ct/export_subgroup_profiles_09b_%j.log

# 09b: pool 5 joint DeepSHAP NPZs into per-subgroup positional/structural SHAP TSVs.
# Outputs (in results_4ct/):
#   sample_shap_structural_{tag}.tsv               (feeds §7.2)
#   shap_profile_{atg,stop}_joint_{tag}.tsv        (feeds §2.3, §3, §4)
#   shap_profile_{atg,stop}_subgroup_joint_{tag}.tsv (feeds §9.6)
#   motif_logo_{atg,stop}_subgroup_joint_{tag}.tsv (feeds §9.7, §9.8)
# Prereq: 5x deepshap_joint_{tag}_run{1..5}.npz produced by slurm_deepshap_joint.sh

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
eval "$(conda shell.bash hook)"
conda activate nmd_model

# Read the selected window configuration; do not restate it. A driver that hardcodes
# the tag keeps running the PREVIOUS selection after a re-selection, silently, because
# the old artifacts still exist. Torch-free, so it costs milliseconds.
TAG="$(python3 paths_config.py --selected-tag --config config.yaml)"
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi

echo "=== 09b: per-subgroup profiles, 5-run pooled ==="
python 09b_export_subgroup_profiles.py --tag ${TAG} --n-runs 5

echo ""
echo "=== Done ==="
