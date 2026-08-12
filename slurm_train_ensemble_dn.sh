#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # UNPINNED 2026-07-29 (reversing the pin earlier the same day).
#   Pinning to v100-sxm2 looked like a 34->14 node trade; it is not. sinfo -N emits one row
#   per node-PARTITION pair, so that 14 was duplicates -- and d1002/d1007/d1009/d1010 are
#   DRAINING, leaving ~2 nodes with capacity. Pinned, job 8826324 projected a start 2h LATER
#   than the oversized job it replaced. Eligibility, not gap size, is the binding constraint.
#   For SHORT work unpinning is free: the wall below covers the ~5x slow draw. Long jobs
#   (joint/atg-stop/all-modes) stay pinned, since unpinned they would need 2-6h walls.
#SBATCH --time=01:00:00   # 8:08 on the fast family, ~41m on the slow one; this
#   wall survives EITHER draw, which is what makes unpinning safe.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_ens
#SBATCH --output=results_deposit_h5_2026-08-04/train_ens_dn_%A_%a.log
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
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# PY resolves in three steps so it works off this machine without changing behaviour on it:
# the authoring env if present, else whatever python3 is on PATH, else a loud failure. It
# previously defaulted to the authoring path unconditionally, which resolves for one account
# and silently points everyone else at a path that does not exist.
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -x "$PY" ] || { echo "FATAL: no python found. Set PY to one with torch and shap installed "\
                       "(see environment-model.yml)." >&2; exit 1; }

SEED=$((SLURM_ARRAY_TASK_ID * 100))

# The window configuration comes from the ONE place that names it, not from a literal here.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml)
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi
ATG=$(echo "$TAG" | sed -E 's/^atg([0-9]+)_stop([0-9]+)$/\1/')
STOP=$(echo "$TAG" | sed -E 's/^atg([0-9]+)_stop([0-9]+)$/\2/')

echo "=== node $(hostname) | member ${SLURM_ARRAY_TASK_ID}, seed=${SEED}, ${TAG} (ATG=${ATG} STOP=${STOP}) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

echo "=== TRAIN member ${SLURM_ARRAY_TASK_ID} ==="
# results_4ct_dn is the DEPRECATED tree -- its HDF5 was built from Channing inputs
# (config_dn.yaml, "REPOINTED 2026-08-04"), and it now sits under deprecated_2026-08-04/.
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
$PY 03_train.py --config config_dn.yaml --results-dir "$RESULTS_DIR" \
    --atg-window "${ATG}" --stop-window "${STOP}" --seed "${SEED}"
echo "=== train exit: $? ==="

# Scored on VAL. Not test: the ensemble's test performance is a single final number and
# belongs in one deliberate --final run after the members exist, not five times here.
echo "=== EVALUATE member ${SLURM_ARRAY_TASK_ID} on val ==="
$PY evaluate.py --config config_dn.yaml --results-dir "$RESULTS_DIR" \
    --atg-window "${ATG}" --stop-window "${STOP}" --member-seed "${SEED}" --split val_clean
echo "=== evaluate exit: $? ==="
