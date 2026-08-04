#!/bin/bash
#SBATCH --job-name=hi_seed
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_seed_%a_%A.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
S=analysis_plans/analysis_ism_regions.py
SEEDS=(100 200 300 400 500)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "script sha256: $(sha256sum $S)"
echo "seed $SEED   bank results_ism_v6/bank_interp_s${SEED}.h5"
# DECAY ONLY, deliberately. The run-length positive was measured on vals_decay and
# nothing else; running capture in the same job would invite the two being read as
# one result. Capture goes in its own job if the access window allows, and it was
# pre-registered as not expected to show much.
$PY -W ignore $S --shards results_ism_v6/bank_interp_s${SEED}.h5 --column dec
