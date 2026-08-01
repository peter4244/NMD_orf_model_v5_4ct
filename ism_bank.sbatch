#!/bin/bash
# One array task builds one (arm, seed) bank. Shards make a task idempotent: a
# transcript already on disk is skipped, so resubmitting the same array after a
# time-limit kill costs the transcript it was in, not the run.
#
# The manifest is one line per bank: <name> <checkpoint> <perm_draws>
#
#   sbatch --array=1-5  ism_bank.sbatch ism/banks_interp.tsv
#
#SBATCH --job-name=nmd_ism
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.out

set -uo pipefail
cd "$HOME/cc/nmd_orf_model_v5_4ct"
P="$HOME/.conda/envs/nmd_model/bin/python"
MANIFEST="${1:?usage: sbatch ism_bank.sbatch <manifest.tsv> [n]}"
N="${2:-1000}"

read -r NAME CKPT DRAWS < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

echo "task $SLURM_ARRAY_TASK_ID: $NAME  checkpoint $CKPT  perm_draws $DRAWS"
echo "host $(hostname)  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# Chained with && so a failed step stops the next one. This project has twice
# had a patch fail and the commit go ahead anyway, and once an scp fail and the
# job submit anyway.
$P -u build_ism_bank.py \
    --tensor results_tensor_v6 \
    --checkpoint "$CKPT" \
    --split results_ism_v6/discovery_confirmation_split.tsv \
    --out "results_ism_v6/bank_${NAME}.h5" \
    --n "$N" \
    --perm-draws "$DRAWS" \
    && echo "finished $(date -u +%FT%TZ)" \
    || { echo "FAILED (rc=$?) — resubmit; finished transcripts are kept as shards"; exit 1; }
