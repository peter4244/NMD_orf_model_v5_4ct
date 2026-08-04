#!/bin/bash
#SBATCH --job-name=dn_joint_motif
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_4ct_dn/joint_motif_logos_dn_%j.log

# The last two deposit-native model artifacts the deposit ships:
#   motif_logo_atg_joint_atg500_stop500.tsv
#   motif_logo_stop_joint_atg500_stop500.tsv
#
# WHY BATCH, NOT THE LOGIN NODE. export_joint_motif_logos.py loads five joint DeepSHAP npz
# (~160MB compressed each, far larger in memory). That is the same load pattern that got
# 08_export_subgroup_deepshap_tsv.py and 09b_export_subgroup_profiles.py killed rc=137
# interactively -- the login node's per-user cap, not node memory.
#
# WHY NOT slurm_export_joint_motif_logos.sh. That wrapper writes its log into results_4ct and
# calls the script WITHOUT --results-dir, so it defaults to the PUBLISHED tree (W94, the eighth
# instance of that pattern). The python itself takes --results-dir correctly; only the wrapper
# was never updated for the deposit-native run.
set -euo pipefail
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== node $(hostname) ==="
ls results_4ct_dn/deepshap_joint_atg500_stop500_run*.npz | wc -l | xargs echo "joint npz replicates:"

$PY scripts/export_joint_motif_logos.py --results-dir results_4ct_dn --atg 500 --stop 500 --n-runs 5
echo "  exit=$?"

echo
echo "=== produced ==="
ls -la results_4ct_dn/motif_logo_atg_joint_atg500_stop500.tsv \
       results_4ct_dn/motif_logo_stop_joint_atg500_stop500.tsv
