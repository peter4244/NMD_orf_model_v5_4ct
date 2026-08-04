#!/bin/bash
#SBATCH --job-name=hi_gcctl
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_gcctl_%j.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
S=analysis_plans/analysis_ism_regions.py
echo "script sha256: $(sha256sum $S)"
/home/p.castaldi/.conda/envs/nmd_model/bin/python -W ignore $S --shards results_ism_v6/bank_interp_s100.h5 --column dec --gc-changing-only
