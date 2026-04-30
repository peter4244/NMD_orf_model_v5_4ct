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

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

TAG="atg500_stop500"

echo "=== 09b: per-subgroup profiles, 5-run pooled ==="
python 09b_export_subgroup_profiles.py --tag ${TAG} --n-runs 5

echo ""
echo "=== Done ==="
