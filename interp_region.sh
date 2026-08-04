#!/bin/bash
#SBATCH --job-name=hi_region
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_region_%a_%A.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
S=analysis_plans/analysis_ism_regions.py
echo "script sha256: $(sha256sum $S)"
BANKS=$(ls -1 results_ism_v6/bank_interp_s{100,200,300,400,500}.h5 | paste -sd, -)
case $SLURM_ARRAY_TASK_ID in
  0) echo "### 5 banks, k=5 — region-matched background AND across-member k-mer"
     $PY -W ignore $S --shards "$BANKS" --column dec --kmer 5 ;;
  1) echo "### seed 100, GC-PRESERVING — re-run of the FAILED 8886695_0"
     $PY -W ignore $S --shards results_ism_v6/bank_interp_s100.h5 --column dec --kmer 5 --gc-neutral-only ;;
  2) echo "### seed 100, one GC-CHANGING — re-run of the FAILED 8886746"
     $PY -W ignore $S --shards results_ism_v6/bank_interp_s100.h5 --column dec --kmer 5 --gc-changing-only ;;
esac
