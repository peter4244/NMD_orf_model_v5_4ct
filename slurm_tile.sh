#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=c6_tile
#SBATCH --output=analysis_plans/runlog_tile_%j.txt
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
export NMD_TOOLS=$HOME/cc/tools
export NMD_CLAIM_VALUES=$PWD/analysis_plans/f_tile_${SLURM_JOB_ID}.tsv
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python3
echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
$PY $NMD_TOOLS/trace_reads.py --out analysis_plans/f_tile_${SLURM_JOB_ID}.trace.tsv     --run analysis_plans/interp_tiled_perturbation.py -- --context --sample 12000 --seed 0     --bank results_tensor_v6/nmd_tensor.h5 --ckpt runs/interp_c32_b8_s100/best.pt     --predicted-at fdc847d:analysis_plans/PLAN_INITIATION_POSITION_VS_CONTENT.md
echo "--- tile exit: $?"
