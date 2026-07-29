#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1   # PINNED 2026-07-29. Generic gpu:1 draws either
#   node family and they differ ~5x on identical work (array 8785576: 14:38 on
#   d1017 vs 1:21:00 on c2204/c2205). With a right-sized wall an unpinned slow
#   draw would be KILLED, and these producers write no partial output.
#SBATCH --time=00:30:00   # measured ~8:08; ~3-4x headroom. Right-sized 2026-07-29:
#   an oversized request is excluded from Slurm backfill, which is what kept job
#   8826175 at PENDING(Priority) for an hour with 14 suitable nodes idle.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=4ct_sweep
#SBATCH --output=results_4ct/train_4ct_sweep_%a_%j.log
#SBATCH --array=1-8

# 4ct window sweep batch 1: first 8 of 12 combinations
# ATG ∈ {100, 500, 1000} × STOP ∈ {100, 500, 1000, 2000}
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
eval "$(conda shell.bash hook)"
conda activate nmd_model

ATG_SIZES=(100  100  100  100  500  500  500  500)
STOP_SIZES=(100  500 1000 2000  100  500 1000 2000)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
ATG=${ATG_SIZES[$IDX]}
STOP=${STOP_SIZES[$IDX]}

echo "=== 4ct Sweep Task ${SLURM_ARRAY_TASK_ID}: ATG=${ATG} STOP=${STOP} ==="

python 03_train.py --config config.yaml --atg-window ${ATG} --stop-window ${STOP}

echo ""
echo "=== Evaluation ATG=${ATG} STOP=${STOP} ==="
# SCORED ON val_clean, NOT test (2026-07-29, D-row 4). The published config was chosen on
# twelve TEST-set scores; re-running the sweep on train+val is what repairs that. val_clean
# RAISES until the HDF5 carries a val_paralog label, which needs 05u re-run with the
# validation chromosomes in scope -- that refusal is deliberate, not a bug to work around.
# Using --split val instead would select on an unscreened validation set: the same leak,
# relocated from test to val.
python evaluate.py --config config.yaml --atg-window ${ATG} --stop-window ${STOP} --split val_clean

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
