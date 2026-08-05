#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00   # RAISED 2026-08-02 FOR FULL COHORT. The 03:30 below it was sized
#   from a measured 1:06:46 on the TEST split (10,520 rows). Interpretation now runs over
#   --split all: 41,765 rows, ~3.97x, so the same work is ~4:25 and 03:30 would have been
#   killed mid-run with nothing written. Old note kept for the measurement it records:
#   measured 1:06:46; ~3-4x headroom. Right-sized 2026-07-29:
#   an oversized request is excluded from Slurm backfill, which is what kept job
#   8826175 at PENDING(Priority) for an hour with 14 suitable nodes idle.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_shap_all
#SBATCH --output=results_deposit_h5_2026-08-04/deepshap_all_dn_%A_%a.log
#SBATCH --array=1-5

# ALL THREE DEPOSIT-NATIVE DeepSHAP DECOMPOSITIONS, one replicate per array task. 2026-07-27.
#
# WHY ONE SCRIPT INSTEAD OF THREE. The QOS caps submitted jobs per user, so three arrays of 5
# cannot be queued together -- the second and third came back QOSMaxSubmitJobPerUserLimit.
# One array of 5, each task running joint -> structural -> atg+stop for its own seed, fits the
# cap and has a real advantage: every mode for replicate N now shares seed N*100, so the modes
# are compared on the same background draw rather than on independently seeded ones.
#
# WHY RE-RUN AT ALL, when the npz are intact and correctly named. Only the SUMMARIES were
# damaged: until the W52 producer fix landed today, deepshap.py wrote
# deepshap_summary_{tag}_run{N}.tsv with no branch mode, so the atg+stop array 8793736
# OVERWROTE the joint summaries. The summary is derivable from the npz, but recomputing it
# outside deepshap.py would mean reimplementing its arithmetic and risking divergence from the
# canonical path. Re-running is cheaper than being unsure. Old files are retained under
# ${RESULTS_DIR}/superseded_mixed_naming_2026-07-27/ per P12.
#
# Summaries now land as deepshap_summary_{tag}_{mode}_run{N}.tsv, mode in
# {joint, structural, atg-stop}, so the filename carries what the content always did.
#
# DETERMINISM: the escape hatch, for the reason the joint script records -- shap's deeplift
# routes MaxPool gradients through max_unpool1d, which has no deterministic CUDA kernel. A
# REPLICATED estimate reports its spread; bitwise determinism is for single-draw headlines.
export NMD_ALLOW_NONDETERMINISM=1

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
# Results tree and windows from the config, never literals: the selection moved to
# atg1000_stop1000 on 2026-08-04 and results_4ct_dn is segregated under deprecated_.
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
SEED=$((SLURM_ARRAY_TASK_ID * 100))
echo "=== node $(hostname) | replicate ${SLURM_ARRAY_TASK_ID}, seed=${SEED} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

overall=0
for MODE in "joint" "structural" "atg stop"; do
    echo "--- branches: ${MODE} ---"
    $PY deepshap.py \
        --config config_dn.yaml \
        --results-dir ${RESULTS_DIR} \
        --n-explain 0 \
        --n-background 500 \
        --atg-window "$ATG" \
        --stop-window "$STOP" \
        --seed ${SEED} \
        --run-id ${SLURM_ARRAY_TASK_ID} \
        --member-seed 42 \
        --split all --full-cohort \
        --branches ${MODE}
    rc=$?
    echo "--- branches ${MODE} exit: $rc ---"
    [ $rc -ne 0 ] && overall=$rc
done

echo "=== replicate ${SLURM_ARRAY_TASK_ID} overall exit: $overall ==="

# PROPAGATE THE REAL EXIT CODE, and propagate the WORST one -- a later success must not mask an
# earlier failure. Ending on an echo previously made SLURM report COMPLETED 0:0 for five jobs
# that had all failed; sacct said success while the log said exit 1.
exit $overall
