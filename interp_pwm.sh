#!/bin/bash
#SBATCH --job-name=hi_pwm
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_pwm_%a_%A.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
S=analysis_plans/analysis_pwm_fit.py
echo "script sha256: $(sha256sum $S)"
case $SLURM_ARRAY_TASK_ID in
  0) $PY -W ignore $S --bank results_ism_v6/bank_interp_s100.h5 --width 9 --mode unsigned ;;
  1) $PY -W ignore $S --bank results_ism_v6/bank_interp_s100.h5 --width 9 --mode signed ;;
  2) $PY -W ignore $S --bank results_ism_v6/bank_interp_s200.h5 --width 9 --mode unsigned ;;
  3) $PY -W ignore $S --bank results_ism_v6/bank_interp_s100.h5 --width 15 --mode unsigned ;;
esac
