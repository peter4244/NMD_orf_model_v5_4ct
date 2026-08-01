#!/bin/bash
# One array task = one (configuration, seed). Resubmitting the same array after a
# time-limit kill resumes every task from its own checkpoint, so the array is
# idempotent and a preempted run costs the epoch it was in, not the run.
#SBATCH --job-name=nmd_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.out

set -euo pipefail
cd "$HOME/cc/nmd_orf_model_v5_4ct"
P="$HOME/.conda/envs/nmd_model/bin/python"
MANIFEST="${1:?usage: sbatch sweep_v6.sbatch <manifest.tsv>}"

# line 1 is the header, so task N is line N+1
read -r NAME VARIANT C B PERM BSEQ BJUNC SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
OUT="runs/${NAME}_s${SEED}"

echo "task $SLURM_ARRAY_TASK_ID  $NAME seed $SEED  host $(hostname)  $(date -u +%FT%TZ)"
echo "  variant=$VARIANT conv_channels=$C n_bins=$B permute=$PERM blank_seq=$BSEQ blank_junc=$BJUNC"
[ -f "$OUT/checkpoint.pt" ] && echo "  checkpoint present — resuming"

ARGS=(--tensor results_tensor_v6 --out "$OUT" --variant "$VARIANT"
      --conv-channels "$C" --n-bins "$B" --seed "$SEED" --mem-budget-gb 10)
[ "$PERM"  = "1" ] && ARGS+=(--permute-bins)
[ "$BSEQ"  = "1" ] && ARGS+=(--blank-sequence)
[ "$BJUNC" = "1" ] && ARGS+=(--blank-junctions)

$P -u train_v6.py "${ARGS[@]}"
echo "finished $(date -u +%FT%TZ)"
