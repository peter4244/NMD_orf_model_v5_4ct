#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=00:30:00   # measured 8:08; ~3-4x headroom. Right-sized 2026-07-29:
#   an oversized request is excluded from Slurm backfill, which is what kept job
#   8826175 at PENDING(Priority) for an hour with 14 suitable nodes idle.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_ens
#SBATCH --output=results_4ct_dn/train_ens_dn_%A_%a.log
#SBATCH --array=1-5
#
# Train the 5-member deposit-native ensemble at the SELECTED window configuration.
# NOT SUBMITTED BY DEFAULT -- created 2026-07-29, awaiting Pete's go-ahead.
#
# WHY THIS DRIVER DID NOT EXIST. slurm_train_dn.sh trains ONE model, and the two existing
# arrays (slurm_train_4ct_sweep*.sh) sweep WINDOW SIZES, not seeds. There has never been a
# seed array -- and before 77b4018 there could not usefully have been one: 03_train.py had
# no --seed, took its seed from config, and wrote best_model_{tag}.pt with no seed slot, so
# five array tasks would have torch.save'd to one filename non-atomically. The
# "five-member ensemble" would have been one checkpoint written five times and the
# seed-variability table that JUSTIFIES the ensemble would have read sd = 0.
#
# WHAT THIS MEASURES, and why it is worth running before the window sweep. Run-to-run
# variation has never been measured in this project (D11 fixed one seed). The historical
# sweep's top two configurations differ by 0.00139 AUC on models that early-stopped at
# epoch 3-5. If the seed sd turns out to be of that order, the sweep cannot resolve its own
# top configurations and the tie-break rule decides the answer -- which is precisely the
# input needed to settle D-B3.5 rather than guess at it.
#
# SEEDS 100..500 match the DeepSHAP replicate convention (slurm_deepshap_joint_dn.sh,
# METHODS.md) so "replicate 3" means the same thing across training and attribution.
#
# GPU IS PINNED. Pete measured V100-PCIE at >4.7x slower than V100-SXM2 (2026-07-27,
# slurm_deepshap_race.sh:11-20): array 8785576 landed tasks 1-2 on d1017 (SXM2) at ~14:40
# and tasks 3-5 on c2204/c2205 (PCIE) still running past 1:20. An unpinned seed array would
# produce runtimes varying ~5x, which is noise in the wall-clock budget AND makes members
# arrive at wildly different times. Pinning costs queue latency on a busy partition; that is
# the better trade for 5 short jobs.
#
# Checkpoints land at best_model_{tag}_seed{SEED}.pt -- distinct paths, so this cannot
# overwrite the existing single-model best_model_atg500_stop500.pt (utils.member_tag).

set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python

SEED=$((SLURM_ARRAY_TASK_ID * 100))

# The window configuration comes from the ONE place that names it, not from a literal here.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml)
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi
ATG=$(echo "$TAG" | sed -E 's/^atg([0-9]+)_stop([0-9]+)$/\1/')
STOP=$(echo "$TAG" | sed -E 's/^atg([0-9]+)_stop([0-9]+)$/\2/')

echo "=== node $(hostname) | member ${SLURM_ARRAY_TASK_ID}, seed=${SEED}, ${TAG} (ATG=${ATG} STOP=${STOP}) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

echo "=== TRAIN member ${SLURM_ARRAY_TASK_ID} ==="
$PY 03_train.py --config config_dn.yaml --results-dir results_4ct_dn \
    --atg-window "${ATG}" --stop-window "${STOP}" --seed "${SEED}"
echo "=== train exit: $? ==="

# Scored on VAL. Not test: the ensemble's test performance is a single final number and
# belongs in one deliberate --final run after the members exist, not five times here.
echo "=== EVALUATE member ${SLURM_ARRAY_TASK_ID} on val ==="
$PY evaluate.py --config config_dn.yaml --results-dir results_4ct_dn \
    --atg-window "${ATG}" --stop-window "${STOP}" --member-seed "${SEED}" --split val
echo "=== evaluate exit: $? ==="
