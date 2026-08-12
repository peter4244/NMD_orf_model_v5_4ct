#!/bin/bash
#SBATCH --job-name=kshap_cell
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/kshap_cell_%j.log
#
# ONE CELL of Analysis 2 — one configuration, one member, one reference draw, full cohort.
#
# PARTITION AND GPU COME FROM THE COMMAND LINE, NOT FROM HERE, because the first use of this
# script is to MEASURE which of CPU and GPU is faster rather than to assert it:
#
#   mkdir -p slurm_logs
#   sbatch --partition=short             slurm_kshap_cell.sh     # CPU
#   sbatch --partition=gpu --gres=gpu:1  slurm_kshap_cell.sh     # GPU
#
# The two write to DIFFERENT output directories. They produce the same filename by construction
# --- kernel_shap_branch_{tag}_seed{S}_all_run{R}.tsv names the member and the draw, which is
# right, and neither is what differs between these two jobs. Sharing a directory would have one
# silently overwrite the other, which is member_tag's and run_suffix's collision one axis over.
#
# --mem=8G: one chunked full-cohort run peaked at 3.59 GiB on the development machine, against
# 35.3 GiB if the explained split were read eagerly. The number that sizes this request properly
# is Linux MaxRSS, which is what these jobs are partly here to produce; 8G is chosen to clear the
# macOS figure with headroom while staying small enough to backfill.
#
# --time=01:30:00: 8m50s measured on the development machine's CPU at atg2000_stop2000. The
# margin covers an unknown cluster core and, on the gpu partition, the 5x spread between GPU node
# families this repo has already recorded. An oversized request is not free -- one is on record
# here as excluded from backfill and left pending an hour beside idle nodes.
#
# slurm_logs/ MUST EXIST BEFORE SUBMISSION. SLURM opens --output before this script runs, so a
# missing directory kills the job in seconds with no log to say why. That is a recorded failure
# in this repository, not a hypothetical, which is why the mkdir is in the usage above rather
# than only in the body below.
#
# THIS SCRIPT MUST NOT END ON AN ECHO. A trailing successful command replaces python's exit
# status, and an exit code reporting success while the log reported failure is also on record
# here. The status is captured immediately and is the last thing the script does.

set -uo pipefail

TAG=${TAG:-atg2000_stop2000}
MEMBER=${MEMBER:-100}
RUN=${RUN:-1}
CHUNK=${CHUNK:-2048}
PROJECT=/home/p.castaldi/cc/nmd_orf_model_v5_4ct
# If OUTDIR is not given, the probes are separated BY PARTITION -- so an unset SLURM_JOB_PARTITION
# would resolve both jobs to one directory and let one silently overwrite the other, which is the
# collision this separation exists to prevent. Refuse rather than default to a shared name.
if [ -z "${OUTDIR:-}" ]; then
    : "${SLURM_JOB_PARTITION:?unset, so the CPU and GPU probes would share an output directory and
one would overwrite the other; pass OUTDIR= explicitly}"
    OUTDIR="results_interp_all/probe_${SLURM_JOB_PARTITION}_${SLURM_JOB_ID:-manual}"
fi

cd "$PROJECT" || exit 1
mkdir -p "$OUTDIR" slurm_logs

PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
TIMER=""
[ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"

echo "host       $(hostname)"
echo "job        ${SLURM_JOB_ID:-?}   partition ${SLURM_JOB_PARTITION:-?}"
echo "gres       ${SLURM_JOB_GRES:-none}   cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "cell       tag=$TAG member=$MEMBER draw=$RUN chunk=$CHUNK"
echo "outdir     $OUTDIR"
echo "commit     $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "started    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
$PY -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__, '| cuda available', torch.cuda.is_available())"
echo "----------------------------------------------------------------------"

$TIMER $PY 11_kernel_shap_branches.py \
    --config config_dn.yaml \
    --results-dir "$OUTDIR" \
    --checkpoint-dir results_4ct_sweep \
    --tag "$TAG" --member-seed "$MEMBER" --run-id "$RUN" \
    --explain-split all --full-cohort \
    --chunk-size "$CHUNK" --n-background 500 --seed 42
status=$?

echo "----------------------------------------------------------------------"
echo "finished   $(date -u +%Y-%m-%dT%H:%M:%SZ)   python exit $status"
exit $status
