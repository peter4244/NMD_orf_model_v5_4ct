#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=c6_hist
#SBATCH --output=analysis_plans/runlog_hist_%j.txt
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
export NMD_TOOLS=$HOME/cc/tools
export NMD_CLAIM_VALUES=$PWD/analysis_plans/f_hist_${SLURM_JOB_ID}.tsv
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python3
echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
$PY $NMD_TOOLS/trace_reads.py --out analysis_plans/f_hist_${SLURM_JOB_ID}.trace.tsv     --run analysis_plans/model_weight_histograms.py -- --out analysis_plans/f_hist_${SLURM_JOB_ID}.log --all --seed 100
echo "--- hist exit: $?"
