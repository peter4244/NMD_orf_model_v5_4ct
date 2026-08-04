#!/bin/bash
#SBATCH --job-name=hi_across
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_across_members_%j.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
S=analysis_plans/analysis_ism_regions.py
# PROVENANCE OF WHAT ACTUALLY RUNS, not of the clone. We copy files up rather than
# pull, so `git rev-parse HEAD` here describes the checkout and not the script.
echo "script sha256: $(sha256sum $S)"
echo "git HEAD here: $(git rev-parse --short HEAD 2>/dev/null || echo none)  (NOT the provenance)"
BANKS=$(ls -1 results_ism_v6/bank_interp_s{100,200,300,400,500}.h5 | paste -sd, -)
echo "banks: $BANKS"
for COL in dec cap; do
  echo; echo "################ column=$COL ################"
  $PY -W ignore $S --shards "$BANKS" --column $COL
done
