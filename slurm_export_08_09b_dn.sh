#!/bin/bash
#SBATCH --job-name=export_dn_08_09b
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_4ct_dn/export_08_09b_dn_%j.log

# The two deposit-native exports that will not run on a login node.
#
# WHY SLURM. 08_export_subgroup_deepshap_tsv.py and 09b_export_subgroup_profiles.py were both
# killed with rc=137 when run interactively. That is not the node running out of memory -- it
# reported 250G total with 224G free -- it is the login node's per-user cap. Both load the
# full DeepSHAP npz set (5 replicates x ~160MB compressed, far larger in memory), so they need
# a batch allocation. 96G is deliberate headroom over the observed footprint; the other seven
# export scripts run fine interactively and are not repeated here.
#
# WHY --results-dir MATTERS ON EVERY LINE. Both default to results_4ct, the PUBLISHED tree.
# Run without it against a deposit-native question, they exit 0 and report the published
# numbers -- the failure mode that made 10_export re-measure the wrong run before it was
# parameterised. Passed explicitly below.

set -o pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

TAG="atg500_stop500"
RES="results_4ct_dn"
echo "=== node $(hostname) | tag=${TAG} | results=${RES} ==="

overall=0

echo "--- 08_export_subgroup_deepshap_tsv ---"
python3 08_export_subgroup_deepshap_tsv.py --atg 500 --stop 500 --results-dir "${RES}"
rc=$?; echo "--- 08 exit: $rc ---"; [ $rc -ne 0 ] && overall=$rc

echo "--- 09b_export_subgroup_profiles ---"
python3 09b_export_subgroup_profiles.py --tag "${TAG}" --n-runs 5 --results-dir "${RES}"
rc=$?; echo "--- 09b exit: $rc ---"; [ $rc -ne 0 ] && overall=$rc

echo "=== overall exit: $overall ==="

# Propagate the WORST exit code, not the last one: a later success must not mask an earlier
# failure. Ending on an echo previously made SLURM report COMPLETED 0:0 for five jobs that had
# all failed, with sacct saying success while the log said exit 1.
exit $overall
