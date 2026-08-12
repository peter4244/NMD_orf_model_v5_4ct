#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=md_a2_enum
#SBATCH --output=results_ism_v6/model_a2_enumeration_%j.log

# The enumeration SEQ-A2 is blocked on: the dead-fraction distribution that
# freezes three provisional parameters, and the set-definition grid that
# reconciles 1.16 / 1.148 / 1.15 for the unstratified keto ratio.
#
# Time is generous because the composition grid makes six passes over the bank
# and the dead-fraction and banding reports make two more. Each pass slices
# vals_decay per transcript -- (4999, 9833, 4) float32 is 786 MB read whole, and
# h5py is lazy only if you never write [:].
#
# The cluster clone's git HEAD is not the provenance of what runs here; scripts
# are copied on, not pulled. The sha256 printed below is.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python

echo "=== code (sha256) ==="
sha256sum analysis_plans/model_a2_enumeration.py
echo "=== bank ==="
ls -la results_ism_v6/bank_interp_s100.h5
echo ""

$PY analysis_plans/model_a2_enumeration.py \
    --bank results_ism_v6/bank_interp_s100.h5 \
    --floor 100
echo "=== exit: $? ==="
