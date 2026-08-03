#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # UNPINNED 2026-07-29 (reversing the pin earlier the same day).
#   Pinning to v100-sxm2 looked like a 34->14 node trade; it is not. sinfo -N emits one row
#   per node-PARTITION pair, so that 14 was duplicates -- and d1002/d1007/d1009/d1010 are
#   DRAINING, leaving ~2 nodes with capacity. Pinned, job 8826324 projected a start 2h LATER
#   than the oversized job it replaced. Eligibility, not gap size, is the binding constraint.
#   For SHORT work unpinning is free: the wall below covers the ~5x slow draw. Long jobs
#   (joint/atg-stop/all-modes) stay pinned, since unpinned they would need 2-6h walls.
#SBATCH --time=01:15:00   # 9:42 on the fast family, ~49m on the slow one; this
#   wall survives EITHER draw, which is what makes unpinning safe.
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_kshap
#SBATCH --output=results_4ct_dn/kshap_dn_%j.log

# DEPOSIT-NATIVE KernelSHAP branch decomposition, full cohort (2026-07-27).
#
# --explain-split all and the 8h wall time are both taken from the tuning already present in
# the working-tree edits to slurm_kernel_shap.sh (2h -> 8h, split test -> all). The full
# cohort is what §5's interpretability claims are computed over.
#
# --results-dir results_4ct_dn reads the deposit-native checkpoint and writes beside it, so the
# published kernel_shap_branch_atg500_stop500_all.tsv stays intact for comparison. Without it
# this script hardcodes results_4ct and would overwrite exactly that file.
#
# No `set -e`: a non-zero exit is a RESULT to read in the log, not something to hide.
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$PY 11_kernel_shap_branches.py --config config_dn.yaml --results-dir results_4ct_dn \
    --tag atg500_stop500 --n-background 500 --seed 42 --member-seed 42 --explain-split all
echo "=== kernel_shap exit: $? ==="
