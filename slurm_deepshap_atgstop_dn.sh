#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=01:45:00   # measured 31:06; ~3-4x headroom. Right-sized 2026-07-29:
#   an oversized request is excluded from Slurm backfill, which is what kept job
#   8826175 at PENDING(Priority) for an hour with 14 suitable nodes idle.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_atgstop
#SBATCH --output=results_deposit_h5_2026-08-04/deepshap_atgstop_dn_%A_%a.log
#SBATCH --array=1-5

# DEPOSIT-NATIVE **ATG + STOP** DeepSHAP, 5 replicates. 2026-07-27.
#
# WHY THIS EXISTS. The deposit-native campaign ran --branches joint and --branches structural
# and stopped there, so ${RESULTS_DIR} holds deepshap_joint_* and deepshap_structural_* npz and
# NOTHING ELSE. The 06-10 export chain consumes deepshap_{atg,stop}_{tag}[_runN].npz -- see
# 06_export_deepshap_tsv.py:5 -- so it cannot run deposit-native at all: it exits with "NPZ
# files not found". Every one of the twelve section-5 export elements downstream of it
# (deepshap_all_orfs_summary, the motif logos, the subgroup profiles, polya, kozak, GC, frame
# periodicity, junction-downstream, stop-codon subgroup, utr5 channel) therefore has no
# deposit-native producer. This run supplies the missing branch decompositions.
#
# NOT the same quantity as the joint run, and must not be pooled with it. --branches joint
# explains the assembled model; --branches atg stop isolates each sequence CNN. The magnitudes
# differ by construction, which is exactly the W52 hazard the guard in 06 now asserts against:
# the summary FILENAME does not encode the branch mode, only the file CONTENT does.
#
# DETERMINISM: the escape hatch, for the reason the joint script records. set_seed calls
# torch.use_deterministic_algorithms(True), and shap's deeplift_grad routes MaxPool gradients
# through max_unpool1d, which has no deterministic CUDA kernel -- mid_pool is MaxPool1d(4) at
# window 500, so the atg and stop branches hit it exactly as the joint run did. Bitwise
# determinism is not the reproducibility mechanism for a REPLICATED estimate whose spread is
# reported; it matters for a single-draw headline like AUC. The flag prints a non-canonical
# warning, which is the disclosure wanted.
export NMD_ALLOW_NONDETERMINISM=1

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
# Results tree and windows from the config, never literals: the selection moved to
# atg1000_stop1000 on 2026-08-04 and results_4ct_dn is segregated under deprecated_.
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
SEED=$((SLURM_ARRAY_TASK_ID * 100))
echo "=== node $(hostname) | ATG+STOP run ${SLURM_ARRAY_TASK_ID}, seed=${SEED}, all test, 500 bg ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

$PY deepshap.py \
    --config config_dn.yaml \
    --results-dir ${RESULTS_DIR} \
    --n-explain 0 \
    --n-background 500 \
    --atg-window "$ATG" \
    --stop-window "$STOP" \
    --seed ${SEED} \
    --run-id ${SLURM_ARRAY_TASK_ID} \
    --branches atg stop
rc=$?
echo "=== atg+stop run ${SLURM_ARRAY_TASK_ID} exit: $rc ==="

# PROPAGATE THE REAL EXIT CODE. Ending on an `echo` made SLURM report COMPLETED 0:0 for five
# jobs that had all failed -- sacct said success while the log said exit 1. A status line that
# cannot be wrong is worth more than one that is usually right.
exit $rc
