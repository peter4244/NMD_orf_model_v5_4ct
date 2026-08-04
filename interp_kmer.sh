#!/bin/bash
#SBATCH --job-name=hi_kmer
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_kmer_%j.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
S=analysis_plans/analysis_ism_regions.py
echo "script sha256: $(sha256sum $S)"
BANKS=$(ls -1 results_ism_v6/bank_interp_s{100,200,300,400,500}.h5 | paste -sd, -)
for K in 5 6; do
  echo; echo "################ k=$K, decay ################"
  $PY -W ignore $S --shards "$BANKS" --column dec --kmer $K
done
