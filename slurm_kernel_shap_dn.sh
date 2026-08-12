#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # UNPINNED 2026-07-29 (reversing the pin earlier the same day).
#   Pinning to v100-sxm2 looked like a 34->14 node trade; it is not. sinfo -N emits one row
#   per node-PARTITION pair, so that 14 was duplicates -- and d1002/d1007/d1009/d1010 are
#   DRAINING, leaving ~2 nodes with capacity. Pinned, job 8826324 projected a start 2h LATER
#   than the oversized job it replaced. Eligibility, not gap size, is the binding constraint.
#   For SHORT work unpinning is free: the wall below covers the ~5x slow draw. Long jobs
#   (joint/atg-stop/all-modes) stay pinned, since unpinned they would need 2-6h walls.
#SBATCH --time=04:00:00   # 9:42 on the fast family, ~49m on the slow one; this
#   wall survives EITHER draw, which is what makes unpinning safe.
#SBATCH --mem=48G   # RAISED from 16G 2026-08-05. Every DeepSHAP replicate on this
#   same cohort measured 42-50 GB against a 32G request and one was OOM-killed, so a
#   16G request for a full-cohort run at DOUBLE the window width is not a margin.
#   11_kernel_shap_branches.py documents a 2.10 GiB peak at atg1000_stop1000 after its
#   chunking fix, so this should be ample -- the point is that it is no longer a guess.
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_kshap
#SBATCH --output=results_deposit_h5_2026-08-04/kshap_dn_%j.log

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
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# PY resolves in three steps so it works off this machine without changing behaviour on it:
# the authoring env if present, else whatever python3 is on PATH, else a loud failure. It
# previously defaulted to the authoring path unconditionally, which resolves for one account
# and silently points everyone else at a path that does not exist.
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -x "$PY" ] || { echo "FATAL: no python found. Set PY to one with torch and shap installed "\
                       "(see environment-model.yml)." >&2; exit 1; }
# Results tree and tag from the config, never literals. This driver named results_4ct_dn --
# now segregated because its HDF5 was Channing-built -- and the tag atg500_stop500, which
# stopped being the selection when the deposit-native sweep chose atg1000_stop1000.
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1
echo "results-dir=$RESULTS_DIR  tag=$TAG"
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$PY 11_kernel_shap_branches.py --config config_dn.yaml --results-dir "$RESULTS_DIR" \
    --tag "$TAG" --n-background 500 --seed 42 --member-seed 42 --explain-split all --full-cohort
rc=$?
echo "=== kernel_shap exit: $rc ==="
# PROPAGATE IT (fixed 2026-08-02, job 8905262). The last command used to be the echo, so the JOB
# exited with the ECHO's status: python died on an argparse error and SLURM recorded COMPLETED 0:0.
# A job that cannot report failure is worse than no job. Only a file count revealed it.
exit $rc
