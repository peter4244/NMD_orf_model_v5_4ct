#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_shap_all
#SBATCH --output=results_4ct_dn/deepshap_all_dn_%A_%a.log
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
# results_4ct_dn/superseded_mixed_naming_2026-07-27/ per P12.
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
SEED=$((SLURM_ARRAY_TASK_ID * 100))
echo "=== node $(hostname) | replicate ${SLURM_ARRAY_TASK_ID}, seed=${SEED} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

overall=0
for MODE in "joint" "structural" "atg stop"; do
    echo "--- branches: ${MODE} ---"
    $PY deepshap.py \
        --config config_dn.yaml \
        --results-dir results_4ct_dn \
        --n-explain 0 \
        --n-background 500 \
        --atg-window 500 \
        --stop-window 500 \
        --seed ${SEED} \
        --run-id ${SLURM_ARRAY_TASK_ID} \
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
