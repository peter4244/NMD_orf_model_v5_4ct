#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # UNPINNED 2026-07-29 (reversing the pin earlier the same day).
#   Pinning to v100-sxm2 looked like a 34->14 node trade; it is not. sinfo -N emits one row
#   per node-PARTITION pair, so that 14 was duplicates -- and d1002/d1007/d1009/d1010 are
#   DRAINING, leaving ~2 nodes with capacity. Pinned, job 8826324 projected a start 2h LATER
#   than the oversized job it replaced. Eligibility, not gap size, is the binding constraint.
#   For SHORT work unpinning is free: the wall below covers the ~5x slow draw. Long jobs
#   (joint/atg-stop/all-modes) stay pinned, since unpinned they would need 2-6h walls.
#SBATCH --time=01:00:00   # 8:08 on the fast family, ~41m on the slow one; this
#   wall survives EITHER draw, which is what makes unpinning safe.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=4ct_sw2
#SBATCH --output=results_4ct/train_4ct_sweep2_%a_%j.log
#SBATCH --array=1-4

# 4ct window sweep batch 2: ATG=1000 × STOP ∈ {100, 500, 1000, 2000}
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
eval "$(conda shell.bash hook)"
conda activate nmd_model

ATG_SIZES=(1000 1000 1000 1000)
STOP_SIZES=(100   500 1000 2000)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
ATG=${ATG_SIZES[$IDX]}
STOP=${STOP_SIZES[$IDX]}

echo "=== 4ct Sweep Batch 2 Task ${SLURM_ARRAY_TASK_ID}: ATG=${ATG} STOP=${STOP} ==="

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
