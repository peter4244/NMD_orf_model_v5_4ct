#!/bin/bash
# The gpu QOS allows 8 submitted jobs and 4 running, so a 40-task array cannot be
# submitted. Each task instead runs a CHUNK of the manifest sequentially: 8 tasks
# x 5 runs, ~1 h each, inside the 8 h limit.
#
# Resubmitting the same array is safe and is the intended recovery. A run that
# already finished is skipped; a run that was cut off mid-training resumes from
# its own checkpoint.
#SBATCH --job-name=nmd_sweep
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
MANIFEST="${1:?usage: sbatch sweep_v6.sbatch <manifest.tsv> [chunk]}"
CHUNK="${2:-5}"

N=$(($(wc -l < "$MANIFEST") - 1))
FIRST=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
LAST=$(( FIRST + CHUNK - 1 )); [ "$LAST" -gt "$N" ] && LAST=$N

echo "task $SLURM_ARRAY_TASK_ID: manifest runs $FIRST..$LAST of $N"
echo "host $(hostname)  $(date -u +%FT%TZ)"
echo

rc=0
for i in $(seq "$FIRST" "$LAST"); do
    read -r NAME VARIANT C B PERM BSEQ BJUNC SEED < <(sed -n "$((i + 1))p" "$MANIFEST")
    OUT="runs/${NAME}_s${SEED}"
    if [ -f "$OUT/history.json" ]; then
        echo "[$i] $NAME seed $SEED — already finished, skipping"
        continue
    fi
    echo "[$i] $NAME seed $SEED  variant=$VARIANT C=$C B=$B perm=$PERM $(date -u +%T)"
    [ -f "$OUT/checkpoint.pt" ] && echo "      resuming from checkpoint"

    ARGS=(--tensor results_tensor_v6 --out "$OUT" --variant "$VARIANT"
          --conv-channels "$C" --n-bins "$B" --seed "$SEED" --mem-budget-gb 10)
    [ "$PERM"  = "1" ] && ARGS+=(--permute-bins)
    [ "$BSEQ"  = "1" ] && ARGS+=(--blank-sequence)
    [ "$BJUNC" = "1" ] && ARGS+=(--blank-junctions)

    if ! $P -u train_v6.py "${ARGS[@]}"; then
        echo "      RUN FAILED (rc=$?) — continuing to the next; resubmit to retry"
        rc=1
    fi
    echo
done
echo "task finished $(date -u +%FT%TZ)  rc=$rc"
exit $rc
