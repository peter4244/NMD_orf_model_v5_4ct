#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=01:30:00   # measured 25:28; ~3-4x headroom. Right-sized 2026-07-29:
#   an oversized request is excluded from Slurm backfill, which is what kept job
#   8826175 at PENDING(Priority) for an hour with 14 suitable nodes idle.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_joint
#SBATCH --output=results_deposit_h5_2026-08-04/deepshap_joint_dn_%A_%a.log
#SBATCH --array=1-5

# DEPOSIT-NATIVE **JOINT** DeepSHAP, 5 replicates. Second attempt, 2026-07-27.
#
# WHY THIS EXISTS. Claim 5.6.6's 2.153 and "~15x" come from the JOINT decomposition
# (slurm_deepshap_joint.sh, --branches joint), whose summaries carry joint_atg / joint_stop /
# joint_structural rows. The structural-only run produces an ISOLATED decomposition with a
# `structural` branch and different magnitudes -- n_downstream_ejc is 2.7174 there against
# 2.1232 in the published joint run. Not the same quantity; must not be compared.
#
# WHY ATTEMPT 1 FAILED, and it was self-inflicted. utils.set_seed now calls
# torch.use_deterministic_algorithms(True), and shap's deeplift_grad routes MaxPool gradients
# through max_unpool1d, which has NO deterministic CUDA kernel:
#   "max_unpooling2d_forward_out does not have a deterministic implementation"
# All five tasks died. The structural-only run survived because --branches structural explains
# only the structural MLP; the joint run includes the sequence CNNs, where mid_pool is
# MaxPool1d(4) at window 500. The published joint run predates the hardening, which is why it
# ever worked.
#
# THE ESCAPE HATCH IS THE RIGHT ANSWER HERE, not a workaround. Determinism is not the
# reproducibility mechanism for this computation: the 5 replicates deliberately vary the
# background seed, so the replicate MEAN and SD are the uncertainty statement. Bitwise
# determinism matters for a single-draw headline metric like AUC -- it does not for a
# replicated estimate whose spread is reported. NMD_ALLOW_NONDETERMINISM=1 prints a warning
# that its results are not canonical, which is exactly the disclosure wanted.
export NMD_ALLOW_NONDETERMINISM=1

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
# Results tree and windows from the config, never literals: the selection moved to
# atg1000_stop1000 on 2026-08-04 and results_4ct_dn is segregated under deprecated_.
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
SEED=$((SLURM_ARRAY_TASK_ID * 100))
echo "=== node $(hostname) | JOINT run ${SLURM_ARRAY_TASK_ID}, seed=${SEED}, all test, 500 bg ==="
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
    --branches joint
rc=$?
echo "=== joint run ${SLURM_ARRAY_TASK_ID} exit: $rc ==="

# PROPAGATE THE REAL EXIT CODE. Ending the script on an `echo` made SLURM report
# COMPLETED 0:0 for five jobs that had ALL failed -- sacct said success while the log said
# exit 1, and only reading the log caught it. A status line that cannot be wrong is worth
# more than one that is usually right.
exit $rc
