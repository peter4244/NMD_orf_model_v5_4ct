#!/bin/bash
#SBATCH --job-name=hi_gc
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_gc_%a_%A.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
S=analysis_plans/analysis_ism_regions.py
echo "script sha256: $(sha256sum $S)"
case $SLURM_ARRAY_TASK_ID in
  # THE DECISIVE TEST. If clustering survives when only GC-preserving
  # substitutions are scored, the rolling-GC window is not what produces it.
  0) echo "### seed 100, GC-PRESERVING ONLY"
     $PY -W ignore $S --shards results_ism_v6/bank_interp_s100.h5 --column dec --gc-neutral-only ;;
  1) echo "### seed 200, GC-PRESERVING ONLY"
     $PY -W ignore $S --shards results_ism_v6/bank_interp_s200.h5 --column dec --gc-neutral-only ;;
  # RESOLVES THE COUNT DISCREPANCY with the model window's replication. My default
  # is --anchor reference (3,422 of 4,999); theirs uses all. 4999/3422 = 1.46
  # against an observed count ratio of 1.36, so the anchor filter is the leading
  # explanation and this settles it rather than leaving it inferred.
  2) echo "### seed 100, ANCHOR ALL — to reconcile counts with the model window"
     $PY -W ignore $S --shards results_ism_v6/bank_interp_s100.h5 --column dec --anchor all ;;
esac
