#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=md_xseed
#SBATCH --output=results_ism_v6/model_crossseed_floor_%j.log

# Does cross-seed agreement survive conditioning on the floor? vals agrees on sign
# 26.8% of the time against a 6.25% chance floor -- above chance, not high. If the
# disagreement is concentrated below each seed's batch-shape offset, above-floor
# positional claims are usable; if it is not, positional claims need the
# discovery/confirmation arm before they mean anything.
#
# Loads one arm across five seeds at a time: 5 x 0.79 GB, freed between arms.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== code actually running (sha256, first 16) ==="
sha256sum analysis_crossseed_floor.py | awk '{print "  " substr($1,1,16), $2}'
echo ""
$PY analysis_crossseed_floor.py results_ism_v6/bank_interp_s*.h5
echo "=== exit: $? ==="
