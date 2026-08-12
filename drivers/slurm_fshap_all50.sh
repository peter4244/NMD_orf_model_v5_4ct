#!/bin/bash
#SBATCH --job-name=fshap50
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/fshap50_%A_%a.log
#
# ALL FIFTY CELLS of Analysis 3 — the five-player structural-feature game, over the same
# two configurations x five members x five reference draws as Analysis 2.
#
#   mkdir -p slurm_logs
#   sbatch --partition=short --array=0-49%20 slurm_fshap_all50.sh
#
# THE ARRAY INDEX IS THE ONLY THING THAT VARIES, and it decodes to exactly one (configuration,
# member, draw). Fifty tasks, fifty distinct cells, no cell computed twice and none skipped:
#
#   i // 25   -> configuration   0 = atg2000_stop2000, 1 = atg1000_stop1000
#   i %  25   -> cell within it
#     (i % 25) // 5 -> member seed   100 200 300 400 500
#     (i % 25) %  5 -> reference draw 1..5
#
# ALL FIFTY WRITE INTO ONE DIRECTORY, deliberately. The filename carries the member AND the
# draw -- feature_shap_{tag}_seed{S}_all_run{R}.tsv -- which is precisely what member_tag
# and run_suffix exist to guarantee. Fifty distinct names, and a collision would be a bug in
# those two functions rather than something this script should work around by fanning out into
# fifty directories.
#
# ONE DEVICE FOR ALL FIFTY, and the honest size of the reason. CPU and GPU do not produce
# bit-identical floating point: the same cell run on both differs by up to 5.9e-06 in the
# prediction and 3.3e-06 in the Shapley values, which is ~50x float32 epsilon. But it does NOT
# propagate to what this analysis reports -- the branch shares from the two devices agree to
# 0.0000 percentage points, three or more orders below any spread the design is estimating.
# Measured on Explorer 2026-07-31, jobs 8849989 (short) and 8849990 (gpu).
#
# So a mixed-device array would very probably have been fine. The partition is fixed anyway,
# because it costs nothing and it removes a question from the provenance of the fifty rather
# than leaving one that has to be re-answered later.
#
# %20 caps concurrency: fifty tasks each stream a 2.9 GB HDF5 in chunks, and there is no reason
# to make that the shared filesystem's problem all at once. Measured on the Analysis 2 array,
# raising it to 50 moved concurrency only from 20 to 23 -- the scheduler, not this cap, was the
# near-term limit -- so 20 costs almost nothing.
#
# TIME: 9m57s measured for one full-cohort cell at atg1000_stop1000 on the development machine,
# against 6m16s for the Analysis 2 equivalent. The cluster ran the Analysis 2 cells in 5m31s to
# 10m58s, so 01:30:00 keeps the same margin over an unknown core that it did there. Do NOT round
# it up: an oversized request is on record here as excluded from backfill and left pending an
# hour beside idle nodes.
#
# The three operational rules from slurm_kshap_cell.sh hold here and for the same recorded
# reasons: slurm_logs/ must exist before submission, the script must not end on an echo, and an
# oversized time request is excluded from backfill.

set -uo pipefail

i=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array}

TAGS=(atg2000_stop2000 atg1000_stop1000)
SEEDS=(100 200 300 400 500)
DRAWS=(1 2 3 4 5)

TAG=${TAGS[$((i / 25))]}
MEMBER=${SEEDS[$(((i % 25) / 5))]}
RUN=${DRAWS[$(((i % 25) % 5))]}
CHUNK=${CHUNK:-2048}

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTDIR=${OUTDIR:-results_interp_all}

cd "$PROJECT" || exit 1
mkdir -p "$OUTDIR" slurm_logs

PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
TIMER=""
[ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"

echo "host       $(hostname)"
echo "array      ${SLURM_ARRAY_JOB_ID:-?} task $i of 0-49   partition ${SLURM_JOB_PARTITION:-?}"
echo "cell       tag=$TAG member=$MEMBER draw=$RUN chunk=$CHUNK"
echo "commit     $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "started    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "----------------------------------------------------------------------"

$TIMER $PY 12_feature_shap_structural.py \
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
