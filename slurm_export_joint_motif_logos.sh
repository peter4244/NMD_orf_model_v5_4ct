#!/bin/bash
#SBATCH --job-name=joint_motif
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=results_4ct/export_joint_motif_logos_%j.log

# Pool 5 joint DeepSHAP NPZs into the master Kozak/stop-codon motif logo TSVs.
# Outputs (in results_4ct/):
#   motif_logo_atg_joint_{tag}.tsv   (feeds §3.1 Kozak motif logo)
#   motif_logo_stop_joint_{tag}.tsv  (feeds §4.1 stop-codon motif logo)
# Prereq: 5x deepshap_joint_{tag}_run{1..5}.npz produced by slurm_deepshap_joint.sh

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

TAG="atg500_stop500"

echo "=== Joint motif logos, 5-run pooled ==="
python scripts/export_joint_motif_logos.py --tag ${TAG} --n-runs 5

echo ""
echo "=== Done ==="
