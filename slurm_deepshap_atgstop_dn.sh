#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_atgstop
#SBATCH --output=results_4ct_dn/deepshap_atgstop_dn_%A_%a.log
#SBATCH --array=1-5

# DEPOSIT-NATIVE **ATG + STOP** DeepSHAP, 5 replicates. 2026-07-27.
#
# WHY THIS EXISTS. The deposit-native campaign ran --branches joint and --branches structural
# and stopped there, so results_4ct_dn holds deepshap_joint_* and deepshap_structural_* npz and
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

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
SEED=$((SLURM_ARRAY_TASK_ID * 100))
echo "=== node $(hostname) | ATG+STOP run ${SLURM_ARRAY_TASK_ID}, seed=${SEED}, all test, 500 bg ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

$PY deepshap.py \
    --config config_dn.yaml \
    --results-dir results_4ct_dn \
    --n-explain 0 \
    --n-background 500 \
    --atg-window 500 \
    --stop-window 500 \
    --seed ${SEED} \
    --run-id ${SLURM_ARRAY_TASK_ID} \
    --branches atg stop
rc=$?
echo "=== atg+stop run ${SLURM_ARRAY_TASK_ID} exit: $rc ==="

# PROPAGATE THE REAL EXIT CODE. Ending on an `echo` made SLURM report COMPLETED 0:0 for five
# jobs that had all failed -- sacct said success while the log said exit 1. A status line that
# cannot be wrong is worth more than one that is usually right.
exit $rc
